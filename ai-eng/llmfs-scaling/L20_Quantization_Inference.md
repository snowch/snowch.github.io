---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# L19 - Quantization for Inference [DRAFT]

*Shrink models 4-8× with minimal quality loss*

---

A 7B model in FP16 requires 14 GB. With quantization, we can fit it in 2-4 GB with <1% quality degradation. This lesson covers post-training quantization techniques that make LLMs practical on consumer hardware.

By the end of this post, you'll understand:
- INT8 quantization: 2× memory reduction
- INT4 quantization: 4× memory reduction
- GPTQ and AWQ: Advanced weight-only quantization
- Calibration strategies for optimal quality
- Speed vs. quality trade-offs

---

## Part 1: The Quantization Concept

### From 16 bits to 8 bits

**FP16**: 16 bits per number, range $\pm 65,504$

**INT8**: 8 bits per number, range $-128$ to $127$

**Key challenge**: How to map the FP16 range to INT8 without losing information?

### Symmetric Quantization

$$\text{INT8} = \text{round}\left(\frac{\text{FP16}}{\text{scale}}\right)$$

Where $\text{scale} = \frac{\max(|\text{FP16}|)}{127}$

```python
def quantize_symmetric(tensor):
    """Quantize FP16 tensor to INT8."""
    # Find scale factor
    scale = tensor.abs().max() / 127.0

    # Quantize
    tensor_int8 = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)

    return tensor_int8, scale

def dequantize_symmetric(tensor_int8, scale):
    """Dequantize INT8 back to FP16."""
    return tensor_int8.to(torch.float16) * scale

# Example
weight = torch.randn(1024, 1024) * 0.1  # Typical weight range
weight_int8, scale = quantize_symmetric(weight)

print(f"Original: {weight.element_size() * weight.numel() / 1024:.1f} KB")
print(f"Quantized: {weight_int8.element_size() * weight_int8.numel() / 1024:.1f} KB")
# Original: 2048 KB (FP16)
# Quantized: 1024 KB (INT8) → 50% reduction!
```

---

## Part 2: INT8 Weight-Only Quantization

### The Simplest Approach

**Idea**: Quantize only weights (keep activations in FP16).

```python
class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Quantize weights
        weight_int8, scale = quantize_symmetric(linear_layer.weight)

        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('scale', scale)

        # Keep bias in FP16
        if linear_layer.bias is not None:
            self.register_buffer('bias', linear_layer.bias)
        else:
            self.bias = None

    def forward(self, x):
        # Dequantize weights on-the-fly
        weight_fp16 = self.weight_int8.to(torch.float16) * self.scale

        # Standard linear operation
        output = F.linear(x, weight_fp16, self.bias)

        return output

# Convert all Linear layers
def quantize_model(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)

            setattr(parent, child_name, QuantizedLinear(module))

    return model
```

**Memory savings**: 7B model: 14 GB → 7 GB

---

## Part 3: INT8 with Mixed-Precision Decomposition

### The Problem: Outliers

Some weights/activations have outliers (e.g., 1% of values are 100× larger).

**Solution**: Use FP16 for outlier channels, INT8 for the rest.

```python
class MixedPrecisionLinear(nn.Module):
    def __init__(self, linear_layer, outlier_threshold=5.0):
        super().__init__()

        # Identify outlier columns (channels with large values)
        col_max = linear_layer.weight.abs().max(dim=0).values
        outlier_mask = col_max > outlier_threshold

        # Split into INT8 and FP16 partitions
        normal_mask = ~outlier_mask

        self.weight_int8, self.scale = quantize_symmetric(
            linear_layer.weight[:, normal_mask]
        )
        self.weight_fp16 = linear_layer.weight[:, outlier_mask]

        self.register_buffer('normal_mask', normal_mask)
        self.register_buffer('outlier_mask', outlier_mask)

    def forward(self, x):
        # Split input
        x_normal = x[:, :, self.normal_mask]
        x_outlier = x[:, :, self.outlier_mask]

        # INT8 path
        weight_normal = self.weight_int8.to(torch.float16) * self.scale
        output_normal = x_normal @ weight_normal.T

        # FP16 path (outliers)
        output_outlier = x_outlier @ self.weight_fp16.T

        # Combine
        output = output_normal + output_outlier

        return output
```

**Quality improvement**: Keeps outliers in high precision, minimal degradation.

**Used by**: LLM.int8() paper

---

## Part 4: INT4 Quantization (GPTQ)

### Going Even Smaller

**INT4**: Only 16 possible values! (-8 to 7)

**Challenge**: Naive INT4 causes significant quality loss.

**GPTQ Solution**: Layer-wise quantization with error compensation.

### The GPTQ Algorithm

```python
def gptq_quantize_layer(layer, calibration_data):
    """
    Quantize a layer while minimizing error on calibration data.
    """
    W = layer.weight.data  # [out_features, in_features]
    out_features, in_features = W.shape

    # Compute Hessian (importance of each weight)
    H = compute_hessian(layer, calibration_data)

    # Quantize weights one column at a time
    W_quant = torch.zeros_like(W)

    for i in range(in_features):
        # Quantize column i
        w_col = W[:, i]
        w_col_int4, scale = quantize_to_int4(w_col)
        w_col_quant = dequantize_from_int4(w_col_int4, scale)

        # Compute error
        error = w_col - w_col_quant

        # Compensate error in remaining columns (using Hessian)
        W[:, i+1:] -= (error.unsqueeze(1) @ H[i, i+1:].unsqueeze(0))

        W_quant[:, i] = w_col_quant

    return W_quant

def quantize_to_int4(tensor):
    """Quantize to 4-bit integer."""
    scale = tensor.abs().max() / 7.0  # INT4 range: -8 to 7
    tensor_int4 = torch.round(tensor / scale).clamp(-8, 7)
    return tensor_int4, scale
```

**Key insight**: Compensate quantization error of early weights by adjusting later weights.

**Result**: 7B model in 3.5 GB (INT4) with <2% quality loss!

---

## Part 5: AWQ (Activation-aware Weight Quantization)

### The Key Observation

Not all weights are equally important!

**AWQ strategy**:
1. Run calibration data through model
2. Identify "salient" weights (those with large activations)
3. Keep salient weights in higher precision

```python
def awq_quantize_layer(layer, calibration_data):
    """
    Quantize with activation-aware scaling.
    """
    # Get activations for this layer
    activations = get_activations(layer, calibration_data)

    # Compute per-channel salience
    salience = activations.abs().mean(dim=0)  # [in_features]

    # Scale weights by salience before quantizing
    W = layer.weight.data
    W_scaled = W * salience.unsqueeze(0)

    # Quantize scaled weights
    W_int4, scale = quantize_to_int4(W_scaled)

    # Store inverse salience for dequantization
    return W_int4, scale, salience

# During inference
def awq_forward(x, W_int4, scale, salience):
    # Dequantize
    W = dequantize_from_int4(W_int4, scale)

    # Unscale weights
    W = W / salience.unsqueeze(0)

    # Standard matmul
    return x @ W.T
```

**Advantage**: Better quality than GPTQ at same bit-width.

---

## Part 6: Calibration Data

### Why Calibration Matters

Quantization quality depends on the data used to compute scales.

**Good calibration data**:
- Representative of deployment distribution
- Diverse (covers edge cases)
- ~128-512 samples (diminishing returns beyond this)

```python
# Example: Calibration on WikiText
from datasets import load_dataset

def get_calibration_data(tokenizer, num_samples=128):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    calibration_samples = []
    for i in range(num_samples):
        text = dataset[i]['text']
        tokens = tokenizer.encode(text, max_length=2048, truncation=True)
        calibration_samples.append(torch.tensor(tokens))

    return torch.stack(calibration_samples)

# Use for quantization
calibration_data = get_calibration_data(tokenizer)
quantized_model = gptq_quantize(model, calibration_data)
```

---

## Part 7: Quality vs. Speed Trade-offs

### Benchmarks (7B model)

| **Method** | **Memory** | **Speed** | **Quality (MMLU)** |
|---|---|---|---|
| FP16 (baseline) | 14 GB | 1.0× | 68.0% |
| INT8 (weight-only) | 7 GB | 1.3× | 67.5% (-0.5%) |
| INT8 (LLM.int8()) | 7 GB | 1.2× | 67.8% (-0.2%) |
| INT4 (naive) | 3.5 GB | 2.0× | 62.0% (-6.0%) ❌ |
| INT4 (GPTQ) | 3.5 GB | 1.8× | 66.8% (-1.2%) |
| INT4 (AWQ) | 3.5 GB | 1.8× | 67.2% (-0.8%) ✅ |

**Recommendation**: Use AWQ for INT4, or INT8 if quality is critical.

---

## Part 8: Practical Usage

### Using `bitsandbytes` Library

```python
# Install: pip install bitsandbytes
import torch
from transformers import AutoModelForCausalLM

# Load model in INT8
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # INT8 quantization
    device_map="auto"
)

# Generate (same API as FP16!)
output = model.generate(input_ids, max_new_tokens=100)
```

### Using `AutoGPTQ`

```python
# Install: pip install auto-gptq
from auto_gptq import AutoGPTQForCausalLM

# Load INT4 GPTQ model
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0"
)

# Inference
output = model.generate(input_ids, max_new_tokens=100)
```

---

## Part 9: When NOT to Quantize

### Cases Where Full Precision is Needed

1. **Fine-tuning**: QLoRA exists, but full precision is safer
2. **Math-heavy tasks**: Quantization errors accumulate in reasoning chains
3. **Small models**: <1B params already fit in memory
4. **Research**: Harder to debug quantized models

---

## Summary

1. **INT8 weight-only**: 2× memory reduction, minimal quality loss
2. **LLM.int8()**: Mixed precision for outliers, best INT8 quality
3. **GPTQ**: INT4 with error compensation, 4× reduction
4. **AWQ**: Activation-aware, best INT4 quality
5. **Calibration**: Use 128-512 samples from target distribution
6. **Trade-off**: INT8 for quality-critical, INT4 for resource-constrained

**Next Up: L20 – Deployment & Serving.** Putting it all together in production!

---

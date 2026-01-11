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

# L16 - Mixed Precision Training [DRAFT]

*Train 2-3× faster with half the memory using FP16/BF16*

---

In all previous lessons, we used FP32 (32-bit floating point) for everything. But modern GPUs have specialized hardware for FP16 (16-bit) that's **2-8× faster**. Mixed precision training uses FP16 for most operations while keeping FP32 where needed for numerical stability.

By the end of this post, you'll understand:
- The difference between FP32, FP16, and BF16 formats
- Why naive FP16 training fails (gradient underflow)
- Automatic Mixed Precision (AMP) and gradient scaling
- Practical implementation with PyTorch
- Memory savings and speed improvements

---

## Part 1: Floating Point Formats

### FP32 (Single Precision)

**Standard format** used by default in PyTorch:

```
Sign | Exponent (8 bits) | Mantissa (23 bits)
  1  |    8 bits         |     23 bits
```

- **Range**: $\pm 10^{-38}$ to $\pm 10^{38}$
- **Precision**: ~7 decimal digits
- **Memory**: 4 bytes per number

---

### FP16 (Half Precision)

**Smaller format** with less range and precision:

```
Sign | Exponent (5 bits) | Mantissa (10 bits)
  1  |    5 bits        |     10 bits
```

- **Range**: $\pm 10^{-8}$ to $\pm 10^{4}$ (65,504 max!)
- **Precision**: ~3 decimal digits
- **Memory**: 2 bytes per number (50% reduction)

**Problem**: Small gradients (e.g., $10^{-7}$) underflow to zero!

---

### BF16 (BFloat16)

**Google's format** that keeps FP32 range but reduces precision:

```
Sign | Exponent (8 bits) | Mantissa (7 bits)
  1  |    8 bits        |     7 bits
```

- **Range**: Same as FP32 ($\pm 10^{-38}$ to $\pm 10^{38}$)
- **Precision**: ~3 decimal digits (less than FP32, same as FP16)
- **Memory**: 2 bytes per number

**Advantage**: No gradient underflow! (same exponent range as FP32)

---

### Visual Comparison

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# FP32
axes[0].barh(['Sign', 'Exponent', 'Mantissa'], [1, 8, 23], color=['red', 'blue', 'green'])
axes[0].set_xlabel('Bits')
axes[0].set_title('FP32 (32 bits)\nRange: ±10⁻³⁸ to ±10³⁸\nPrecision: 7 digits')
axes[0].set_xlim(0, 25)

# FP16
axes[1].barh(['Sign', 'Exponent', 'Mantissa'], [1, 5, 10], color=['red', 'blue', 'green'])
axes[1].set_xlabel('Bits')
axes[1].set_title('FP16 (16 bits)\nRange: ±10⁻⁸ to ±10⁴ ⚠️\nPrecision: 3 digits')
axes[1].set_xlim(0, 25)

# BF16
axes[2].barh(['Sign', 'Exponent', 'Mantissa'], [1, 8, 7], color=['red', 'blue', 'green'])
axes[2].set_xlabel('Bits')
axes[2].set_title('BF16 (16 bits)\nRange: ±10⁻³⁸ to ±10³⁸ ✅\nPrecision: 3 digits')
axes[2].set_xlim(0, 25)

plt.tight_layout()
plt.show()
```

---

## Part 2: Why Naive FP16 Training Fails

### The Gradient Underflow Problem

Consider a typical gradient magnitude during training:

```python
# In FP32
gradient = 1.2e-7  # Common in deep networks

# Convert to FP16
gradient_fp16 = np.float16(gradient)
print(gradient_fp16)  # Output: 0.0 ❌
```

**What happened?**: FP16's smallest positive number is $\approx 6 \times 10^{-8}$. Anything smaller rounds to zero!

**Visualization: Gradient Distribution**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Histogram showing:
# - X-axis: Gradient magnitude (log scale)
# - Y-axis: Count
# - Vertical lines at FP16 underflow threshold (~6e-8)
# - Shaded region: "These gradients vanish in FP16"
```

### Loss Overflow Problem

Large activations can exceed FP16's max value (65,504):

```python
# Softmax output for 1000 classes
logits = torch.randn(1000) * 10  # Common scale
exp_logits = torch.exp(logits)

# Max value
print(exp_logits.max())  # Could be 1e8
print(torch.finfo(torch.float16).max)  # 65,504

# Result: Overflow to inf in FP16!
```

---

## Part 3: Mixed Precision Training Strategy

### The Solution: Use FP16 + FP32 Selectively

**Core idea**:
1. **Store weights in FP32** (master copy)
2. **Forward pass in FP16** (fast computation)
3. **Loss computation in FP16**
4. **Scale gradients** to prevent underflow
5. **Update weights in FP32** (accumulated precision)

### Gradient Scaling

To prevent underflow, multiply loss by a large constant before backward pass:

$$\text{Loss}_{\text{scaled}} = \text{Loss} \times \text{scale\_factor}$$

```python
# Without scaling
loss = 0.1
gradient = 1e-7  # Would underflow in FP16

# With scaling (scale=1024)
scaled_loss = loss * 1024  # = 102.4
scaled_gradient = 1e-7 * 1024  # = 1.024e-4 (safe!)

# After backward pass, unscale gradients
gradient = scaled_gradient / 1024  # Back to 1e-7 in FP32
```

---

## Part 4: Automatic Mixed Precision (AMP) in PyTorch

### Basic Usage

PyTorch's `torch.amp` handles everything automatically:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = GPT(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create gradient scaler
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        optimizer.zero_grad()

        # Forward pass in FP16
        with autocast():
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Unscale gradients and update weights (in FP32)
        scaler.step(optimizer)
        scaler.update()
```

**That's it!** 3 lines of code for 2× speedup.

---

### What Happens Under the Hood?

1. **`autocast()` context**:
   - Casts eligible ops to FP16 (matmul, conv)
   - Keeps sensitive ops in FP32 (softmax, log, sum)

2. **`scaler.scale(loss)`**:
   - Multiplies loss by scale factor (default: 65536)

3. **`scaler.step(optimizer)`**:
   - Unscales gradients (divide by scale factor)
   - Checks for inf/NaN (from overflow)
   - If valid, updates weights in FP32
   - If invalid, skips update and reduces scale factor

4. **`scaler.update()`**:
   - Adjusts scale factor dynamically
   - Increases if no overflow detected (max out speedup)
   - Decreases if overflow detected (improve stability)

---

## Part 5: BF16 vs. FP16

### When to Use Each

| **Aspect** | **FP16** | **BF16** |
|---|---|---|
| **Hardware support** | V100, A100, 3090, 4090 | A100, 4090, TPUs |
| **Gradient scaling** | Required | Optional |
| **Numerical stability** | Needs careful tuning | More stable |
| **Speed** | 2-3× faster | 2-3× faster |
| **Underflow risk** | High (range: 6e-8 to 65k) | Low (same range as FP32) |

**Recommendation**:
- Use **BF16** if your hardware supports it (A100, H100, 4090)
- Use **FP16** for older GPUs (V100, 3090)

---

### BF16 Training in PyTorch

```python
# Switch to BF16 (simpler, no scaler needed!)
for batch in train_loader:
    optimizer.zero_grad()

    with autocast(dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

    loss.backward()  # No scaling!
    optimizer.step()
```

---

## Part 6: Memory Savings Breakdown

### FP32 Training (Baseline)

For a 7B parameter model:

| **Component** | **Memory** |
|---|---|
| Model weights | 28 GB |
| Gradients | 28 GB |
| Optimizer states (AdamW) | 56 GB |
| Activations | 10 GB |
| **Total** | **122 GB** |

---

### Mixed Precision Training

| **Component** | **Memory** | **Savings** |
|---|---|---|
| Model weights (FP32 master) | 28 GB | - |
| Model weights (FP16 copy) | 14 GB | - |
| Gradients (FP16) | 14 GB | 50% |
| Optimizer states (FP32) | 56 GB | - |
| Activations (FP16) | 5 GB | 50% |
| **Total** | **117 GB** | **~5 GB saved** |

**Wait, that's not much!**

**Reality**: Optimizer states dominate (56 GB). To save more, need optimizer-level changes (see ZeRO optimizer, future lesson).

**But**: 2-3× speed improvement is the real win!

---

## Part 7: Speed Benchmarks

### Realistic Training Speed

**GPT-2 (124M parameters) on A100**:

| **Precision** | **Throughput** | **Speedup** |
|---|---|---|
| FP32 | 15,000 tokens/sec | 1.0× |
| FP16 | 42,000 tokens/sec | 2.8× |
| BF16 | 40,000 tokens/sec | 2.7× |

**Visualization: Training Time Comparison**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Bar chart showing:
# - FP32: 10 hours
# - FP16: 3.6 hours
# - BF16: 3.7 hours
# - Labels: "Same final accuracy!"
```

---

## Part 8: Common Pitfalls and Solutions

### Pitfall 1: Loss Scaling Too Aggressive

**Symptom**: NaN losses after a few steps

```python
# Check scaler state
print(f"Current scale: {scaler.get_scale()}")
# If this keeps decreasing to 1.0, training is unstable
```

**Solution**: Start with lower initial scale

```python
scaler = GradScaler(init_scale=1024)  # Default is 65536
```

---

### Pitfall 2: Gradient Clipping with AMP

**Wrong order causes NaN**:

```python
# ❌ WRONG
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)

# ✅ CORRECT (unscale first!)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

---

### Pitfall 3: Custom Operations

Some ops don't have FP16 kernels:

```python
# Force FP32 for specific ops
with autocast():
    logits = model(input_ids)  # FP16

    # Custom loss that needs FP32
    with autocast(enabled=False):
        loss = custom_loss_fn(logits.float(), labels.float())
```

---

## Part 9: Monitoring Mixed Precision Training

### Key Metrics to Track

```python
# Track gradient scale changes
if step % 100 == 0:
    print(f"Step {step}, Scale: {scaler.get_scale()}")

# Track skipped updates (inf/NaN detections)
if scaler._found_inf_per_device(optimizer):
    print("Warning: Skipped update due to inf/NaN")

# Track loss magnitude
print(f"Loss: {loss.item():.6f}")  # Should stay in reasonable range
```

**Healthy training**:
- Scale stays high (16384-65536)
- Few/no skipped updates
- Loss decreases smoothly

**Unhealthy training**:
- Scale keeps dropping to 1.0
- Frequent skipped updates
- Erratic loss

---

## Part 10: Production Checklist

### Mixed Precision Best Practices

```python
✅ Use autocast() for forward pass
✅ Use GradScaler for backward pass
✅ Unscale before gradient clipping
✅ Keep master weights in FP32
✅ Monitor scale factor and skipped updates
✅ Use BF16 if hardware supports it
✅ Profile to verify speedup (use torch.profiler)
```

### Full Production Example

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

model = GPT(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()
writer = SummaryWriter()

for step, batch in enumerate(train_loader):
    optimizer.zero_grad()

    # Mixed precision forward
    with autocast(dtype=torch.float16):
        logits = model(batch['input_ids'].cuda())
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            batch['labels'].cuda().view(-1)
        )

    # Scaled backward
    scaler.scale(loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    scaler.step(optimizer)
    scale_before = scaler.get_scale()
    scaler.update()

    # Logging
    if step % 100 == 0:
        writer.add_scalar('Loss', loss.item(), step)
        writer.add_scalar('GradScale', scale_before, step)
        print(f"Step {step}, Loss: {loss.item():.4f}, Scale: {scale_before:.0f}")
```

---

## Summary

1. **Mixed precision** uses FP16/BF16 for speed, FP32 for stability
2. **FP16** requires gradient scaling to prevent underflow
3. **BF16** more stable (same range as FP32) but needs newer hardware
4. **PyTorch AMP** makes it 3 lines of code: `autocast()` + `GradScaler`
5. **Speed**: 2-3× faster with minimal code changes
6. **Memory**: ~5 GB saved (bigger wins need optimizer changes)
7. **Monitor** gradient scale and skipped updates

**Next Steps**: For the advanced **Scaling & Optimization** series, we'll cover:
- **L16**: Attention optimizations (Flash Attention, KV cache)
- **L17**: Model parallelism (data/pipeline/tensor parallelism)
- **L18**: Long context handling (RoPE, ALiBi)
- **L19**: Quantization for inference (INT8, INT4, GPTQ)
- **L20**: Deployment & serving (vLLM, continuous batching)

---

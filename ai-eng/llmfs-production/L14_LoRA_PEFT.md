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

# L14 - Parameter-Efficient Fine-Tuning (LoRA) [DRAFT]

*Fine-tune 7B models on a single GPU with Low-Rank Adaptation*

---

In [L10](L10_Fine_tuning_and_Chat.md), we learned about fine-tuning through SFT and RLHF. But full fine-tuning of a 7B model requires 80GB+ of VRAM. **LoRA** (Low-Rank Adaptation) makes fine-tuning possible on consumer hardware by updating only a tiny fraction of parameters.

By the end of this post, you'll understand:
- The mathematical intuition behind low-rank matrices
- How LoRA adapts frozen pretrained weights
- Implementing LoRA from scratch in PyTorch
- QLoRA: Quantized LoRA for even lower memory
- When to use LoRA vs. full fine-tuning

---

## Part 1: The Memory Problem

### Full Fine-Tuning Memory Requirements

For a 7B parameter model:

| **Component** | **Memory (FP32)** | **Memory (FP16)** |
|---|---|---|
| Model weights | 28 GB | 14 GB |
| Gradients | 28 GB | 14 GB |
| Optimizer states (AdamW) | 56 GB | 28 GB |
| **Total** | **112 GB** | **56 GB** |

Even in FP16, this requires multiple A100 GPUs (80GB each).

**The key insight**: Most weight updates are low-rank. We don't need to update the full weight matrix.

---

## Part 2: Low-Rank Matrix Intuition

### Visualizing Matrix Rank

A **full-rank** matrix has independent rows/columns:

$$
W = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
\quad \text{Rank = 3}
$$

A **low-rank** matrix (rank 1) can be written as outer product:

$$
W = \begin{bmatrix}
1 \\ 2 \\ 3
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
=
\begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
3 & 6 & 9
\end{bmatrix}
\quad \text{Rank = 1}
$$

**Key observation**:
- Full matrix: $3 \times 3 = 9$ parameters
- Low-rank factorization: $3 + 3 = 6$ parameters (33% reduction!)

For a $d \times d$ matrix with rank $r$:
- Full: $d^2$ parameters
- Low-rank: $2dr$ parameters
- Savings: $\frac{2dr}{d^2} = \frac{2r}{d}$

**Example**: For $d=4096$ and $r=8$:
- Full: $4096^2 = 16.7M$ parameters
- Low-rank: $2 \times 4096 \times 8 = 65K$ parameters
- **255× reduction!**

---

## Part 3: How LoRA Works

### The LoRA Equation

Instead of updating the full weight matrix $W$:

$$W_{\text{new}} = W + \Delta W$$

LoRA represents $\Delta W$ as a low-rank decomposition:

$$W_{\text{new}} = W + BA$$

Where:
- $W \in \mathbb{R}^{d \times d}$: Original frozen weights
- $B \in \mathbb{R}^{d \times r}$: Trainable "down-projection"
- $A \in \mathbb{R}^{r \times d}$: Trainable "up-projection"
- $r \ll d$: Rank (typically 8-64)

**Visualization**:

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original weight matrix
W = np.random.randn(512, 512) * 0.1
axes[0].imshow(W, cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
axes[0].set_title('Original Weights W\n(512×512 = 262K params)\nFrozen ❄️')
axes[0].set_xlabel('512')
axes[0].set_ylabel('512')

# Matrix A (down-projection)
A = np.random.randn(8, 512) * 0.1
axes[1].imshow(A, cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
axes[1].set_title('Matrix A\n(8×512 = 4K params)\nTrainable 🔥')
axes[1].set_xlabel('512')
axes[1].set_ylabel('8')

# Matrix B (up-projection)
B = np.random.randn(512, 8) * 0.1
axes[2].imshow(B, cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
axes[2].set_title('Matrix B\n(512×8 = 4K params)\nTrainable 🔥')
axes[2].set_xlabel('8')
axes[2].set_ylabel('512')

# Result: W + BA
BA = B @ A
W_new = W + BA
axes[3].imshow(W_new, cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
axes[3].set_title('Updated Weights\nW + BA\n(Total trainable: 8K vs 262K)')
axes[3].set_xlabel('512')
axes[3].set_ylabel('512')

plt.tight_layout()
plt.show()
```

---

## Part 4: Implementing LoRA from Scratch

### LoRA Linear Layer

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Initialize A with random Gaussian, B with zeros
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Scaling factor (usually set to alpha / rank)
        self.scaling = alpha / rank

    def forward(self, x):
        # Standard: W @ x
        # LoRA: W @ x + (B @ A) @ x = W @ x + x @ A @ B
        return (x @ self.lora_A @ self.lora_B) * self.scaling

# Replace a Linear layer with LoRA
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False  # Freeze original weights

        # Add LoRA matrices
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        # Original output + LoRA adaptation
        return self.linear(x) + self.lora(x)

# Usage
original_layer = nn.Linear(4096, 4096)
lora_layer = LoRALinear(original_layer, rank=8)

# Only LoRA parameters require gradients!
trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params:,}")  # 65,536 instead of 16,777,216
```

---

## Part 5: Applying LoRA to a Transformer

### Which Layers to Apply LoRA To?

**Common strategy**: Apply to attention projection matrices (Q, K, V, O).

```python
def apply_lora_to_model(model, rank=8, alpha=16):
    """Apply LoRA to all attention layers."""

    for name, module in model.named_modules():
        # Apply to Q, K, V, O projections in attention
        if isinstance(module, nn.Linear) and any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            # Replace with LoRA version
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    return model

# Usage
model = GPT(config)
model = apply_lora_to_model(model, rank=8)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# Typically 0.1-1% of total parameters!
```

---

## Part 6: Training with LoRA

### Standard Training Loop

```python
# Only optimize LoRA parameters
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # Can use higher LR than full fine-tuning
)

for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass (uses frozen W + trainable BA)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        # Backward pass (only LoRA parameters get gradients)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Memory Comparison

**Full Fine-Tuning (7B model)**:
- Model: 14 GB
- Gradients: 14 GB (for all 7B params)
- Optimizer: 28 GB
- Total: **56 GB**

**LoRA Fine-Tuning (7B model, rank=8)**:
- Model: 14 GB
- Gradients: 0.05 GB (for 35M params only)
- Optimizer: 0.1 GB
- Total: **~14 GB**

**Result**: Fits on a single RTX 3090 (24GB)!

---

## Part 7: QLoRA - Quantized LoRA

### The Next Level: 4-bit Quantization

QLoRA combines:
1. **4-bit quantization** of the frozen base model
2. **LoRA adapters** in FP16

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # NormalFloat4
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on top
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

**Memory savings**:
- Base model (4-bit): 3.5 GB (vs 14 GB in FP16)
- LoRA adapters (FP16): 0.05 GB
- Total: **~4 GB** (fits on a laptop GPU!)

---

## Part 8: Saving and Loading LoRA Adapters

### Save Only the Adapters

```python
# Save only LoRA weights (tiny file!)
torch.save({
    'lora_state_dict': {k: v for k, v in model.state_dict().items() if 'lora' in k}
}, 'lora_adapters.pt')

# File size: ~100 MB instead of 14 GB!
```

### Load Adapters onto Base Model

```python
# Load base model (unchanged)
base_model = GPT.from_pretrained('gpt-7b')

# Apply LoRA architecture
base_model = apply_lora_to_model(base_model, rank=8)

# Load trained adapters
checkpoint = torch.load('lora_adapters.pt')
base_model.load_state_dict(checkpoint['lora_state_dict'], strict=False)
```

**Use case**: Ship multiple "personalities" as separate adapter files:
- `adapters_chatbot.pt` (100 MB)
- `adapters_coder.pt` (100 MB)
- `adapters_creative.pt` (100 MB)

Users download one base model (14 GB) + swap adapters as needed!

---

## Part 9: When to Use LoRA vs. Full Fine-Tuning

| **Criterion** | **Full Fine-Tuning** | **LoRA** |
|---|---|---|
| **Memory** | High (56+ GB) | Low (14 GB) |
| **Training speed** | Slower | Faster (fewer gradients) |
| **Final performance** | Slightly better | Very close (within 1-2%) |
| **Task similarity** | Works for very different tasks | Best for similar tasks |
| **Adapter swapping** | No | Yes (multiple adapters) |

**Rule of thumb**:
- Use **LoRA** for most fine-tuning (chat, instruction-following)
- Use **full fine-tuning** only if:
  - You have unlimited GPU budget
  - Your task is extremely different (e.g., fine-tuning English model for code)

---

## Part 10: Hyperparameter Tuning

### Key Hyperparameters

| **Parameter** | **Typical Range** | **Effect** |
|---|---|---|
| `rank (r)` | 8-64 | Higher = more capacity, more memory |
| `alpha` | 16-32 | Scaling factor (usually 2× rank) |
| `target_modules` | ["q_proj", "v_proj"] | Which layers to adapt |
| `dropout` | 0.0-0.1 | Regularization |
| `learning_rate` | 1e-4 to 3e-4 | Higher than full fine-tuning |

**Grid search example**:
```python
for rank in [8, 16, 32]:
    for alpha in [16, 32]:
        model = apply_lora(base_model, rank=rank, alpha=alpha)
        score = train_and_evaluate(model)
        print(f"Rank={rank}, Alpha={alpha}: {score:.3f}")
```

---

## Part 11: Visualizing LoRA Updates

**Heatmap: Which Layers Change Most?**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Heatmap showing:
# - X-axis: Training steps
# - Y-axis: Layer number
# - Color: Magnitude of ||BA||_F (Frobenius norm)
# - Shows which layers adapt most to new task
```

**Distribution: LoRA Weight Magnitudes**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Histogram comparing:
# - Original weights W (std ≈ 0.02)
# - LoRA updates BA (std ≈ 0.001)
# - Shows updates are small perturbations
```

---

## Summary

1. **LoRA** reduces trainable parameters by 100-1000× using low-rank decomposition
2. Memory drops from **56 GB → 14 GB** for 7B models
3. **QLoRA** adds 4-bit quantization → **~4 GB total**
4. Apply to attention projections (Q, K, V, O) for best results
5. Performance within 1-2% of full fine-tuning
6. Can swap adapters at inference (multiple tasks, one base model)

**Next Up: L15 – Mixed Precision Training.** Make training 2× faster with FP16/BF16!

---

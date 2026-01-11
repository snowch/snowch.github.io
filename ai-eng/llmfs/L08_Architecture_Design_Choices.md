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

# L08 - Architecture Design Choices: Choosing Your Hyperparameters

*From toy models to production: How to design your GPT architecture*

---

We built a complete GPT in [L07 - Assembling the GPT](L07_Assembling_the_GPT.md). But when you look at the code, you see parameters like `d_model`, `n_heads`, and `n_layers`. How do you choose these values?

This isn't arbitrary. There are established patterns, mathematical constraints, and practical trade-offs that guide these decisions. In this lesson, we'll learn how to design architectures for different use cases—from tiny models that run on your laptop to production-scale systems.

By the end of this post, you'll understand:
- The **mathematical constraints** between parameters (why `d_model` must be divisible by `n_heads`)
- **Common architecture patterns** used in real models (GPT-2, GPT-3, Llama)
- **Trade-offs** between model size, speed, and quality
- How to **estimate parameter counts** and memory requirements

```{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
```

---

## Part 1: The Core Hyperparameters

When designing a GPT architecture, you have six primary hyperparameters to choose:

### The Six Knobs

| Parameter | Symbol | Description | Typical Values |
|-----------|--------|-------------|----------------|
| **Model Dimension** | `d_model` | Width of token embeddings | 512, 768, 1024, 2048, 4096 |
| **Number of Layers** | `n_layers` | Depth of the transformer stack | 6, 12, 24, 32, 96 |
| **Attention Heads** | `n_heads` | Parallel attention mechanisms per layer | 8, 12, 16, 32, 64 |
| **FFN Dimension** | `d_ff` | Hidden size in feed-forward network | Usually `4 × d_model` |
| **Vocabulary Size** | `vocab_size` | Number of unique tokens | 32k, 50k, 100k |
| **Context Window** | `max_len` | Maximum sequence length | 512, 2048, 4096, 8192 |

### Mathematical Constraints

Not all combinations are valid. Here are the hard rules:

**1. Head Dimension Constraint**
```python
assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
d_head = d_model // n_heads  # Head dimension
```

**Why?** Multi-head attention splits `d_model` into `n_heads` equal chunks. If `d_model=512` and `n_heads=8`, each head gets `d_head=64` dimensions.

**2. Typical Head Dimension**

In practice, `d_head` is almost always **64**:
- GPT-2 Small: `d_model=768, n_heads=12` → `d_head=64`
- GPT-3: `d_model=12288, n_heads=96` → `d_head=128` (exception)
- Llama 2 7B: `d_model=4096, n_heads=32` → `d_head=128`

This means: **`d_model = n_heads × 64`** (or 128 for very large models)

**3. FFN Expansion Factor**

The feed-forward network typically expands by **4×**:
```python
d_ff = 4 * d_model
```

So if `d_model=768`, then `d_ff=3072`.

---

## Part 2: Real-World Architecture Examples

Let's look at how actual models are configured:

```{code-cell} ipython3
:tags: [remove-input]

import pandas as pd

# Real model configurations
models = {
    "Tiny (Custom)": {"d_model": 512, "n_layers": 6, "n_heads": 8, "params": "~40M"},
    "GPT-2 Small": {"d_model": 768, "n_layers": 12, "n_heads": 12, "params": "124M"},
    "GPT-2 Medium": {"d_model": 1024, "n_layers": 24, "n_heads": 16, "params": "355M"},
    "GPT-2 Large": {"d_model": 1280, "n_layers": 36, "n_heads": 20, "params": "774M"},
    "GPT-2 XL": {"d_model": 1600, "n_layers": 48, "n_heads": 25, "params": "1.5B"},
    "GPT-3 Small": {"d_model": 2048, "n_layers": 24, "n_heads": 16, "params": "350M"},
    "Llama 2 7B": {"d_model": 4096, "n_layers": 32, "n_heads": 32, "params": "7B"},
    "Llama 2 13B": {"d_model": 5120, "n_layers": 40, "n_heads": 40, "params": "13B"},
    "GPT-3 175B": {"d_model": 12288, "n_layers": 96, "n_heads": 96, "params": "175B"},
}

# Create table
df = pd.DataFrame(models).T
df.index.name = "Model"

# Display as markdown table
print("| Model | d_model | n_layers | n_heads | d_head | Parameters |")
print("|-------|---------|----------|---------|--------|------------|")
for model, row in df.iterrows():
    d_head = row['d_model'] // row['n_heads']
    print(f"| {model} | {row['d_model']} | {row['n_layers']} | {row['n_heads']} | {d_head} | {row['params']} |")
```

| Model | d_model | n_layers | n_heads | d_head | Parameters |
|-------|---------|----------|---------|--------|------------|
| Tiny (Custom) | 512 | 6 | 8 | 64 | ~40M |
| GPT-2 Small | 768 | 12 | 12 | 64 | 124M |
| GPT-2 Medium | 1024 | 24 | 16 | 64 | 355M |
| GPT-2 Large | 1280 | 36 | 20 | 64 | 774M |
| GPT-2 XL | 1600 | 48 | 25 | 64 | 1.5B |
| GPT-3 Small | 2048 | 24 | 16 | 128 | 350M |
| Llama 2 7B | 4096 | 32 | 32 | 128 | 7B |
| Llama 2 13B | 5120 | 40 | 40 | 128 | 13B |
| GPT-3 175B | 12288 | 96 | 96 | 128 | 175B |

### Key Observations

1. **Head dimension stays constant**: Almost always 64 or 128
2. **Both depth and width scale**: Larger models increase both `n_layers` and `d_model`
3. **Heads scale with width**: More heads as the model gets wider
4. **Layers increase more slowly**: Depth grows from 6 → 96, while width grows from 512 → 12288

---

## Part 3: Counting Parameters

Understanding parameter count helps you estimate memory and compute requirements.

### Parameter Count Formula

For a transformer with:
- `vocab_size = V`
- `d_model = d`
- `n_layers = L`
- `n_heads = h`
- `max_len = T`

**Total parameters ≈**:

```
Token Embeddings:    V × d
Position Embeddings: T × d (if learned, 0 if sinusoidal)

Per Layer (×L):
  - Multi-Head Attention:
    - Q, K, V projections: 3 × (d × d) = 3d²
    - Output projection: d × d = d²
    - Total: 4d²

  - Feed-Forward Network:
    - First linear: d × 4d = 4d²
    - Second linear: 4d × d = 4d²
    - Total: 8d²

  - Layer Norms: 4d (negligible)

Final Output:
  - LM Head: V × d (often tied to embedding, so 0)

Total ≈ V×d + T×d + L×(4d² + 8d²) = V×d + T×d + 12Ld²
```

**Dominant term**: For large models, `12Ld²` dominates.

### Example: GPT-2 Small

- `V = 50,257`
- `d = 768`
- `L = 12`
- `T = 1024`
- `h = 12`

```
Embeddings:  50,257 × 768 ≈ 38.6M
Positions:   1,024 × 768 ≈ 0.8M
Layers:      12 × 12 × 768² ≈ 85.0M
────────────────────────────────
Total:       ≈ 124.4M parameters
```

This matches the official GPT-2 Small size!

---

## Part 4: Design Decision Tree

How do you choose these parameters for YOUR use case?

### Start with Your Constraint

**If you have limited compute (e.g., laptop, small GPU):**
- Start small: `d_model=512, n_layers=6, n_heads=8`
- Parameters: ~40M
- Can train on consumer hardware

**If you want to fine-tune an existing model:**
- Use a pretrained model (GPT-2, Llama)
- Don't design from scratch

**If you're training from scratch with good compute:**
- GPT-2 Small equivalent: `d_model=768, n_layers=12, n_heads=12` → 124M
- GPT-2 Medium equivalent: `d_model=1024, n_layers=24, n_heads=16` → 355M

### Scaling Strategy

When scaling up, follow these rules:

1. **Scale width and depth together**: Don't just make it wider or deeper
   - Bad: `d_model=4096, n_layers=6` (too shallow)
   - Bad: `d_model=512, n_layers=96` (too narrow)
   - Good: `d_model=1024, n_layers=24`

2. **Keep `d_head` constant at 64 or 128**:
   - When increasing `d_model`, also increase `n_heads` proportionally
   - Example: `d_model=1024, n_heads=16` → `d_head=64` ✓

3. **Use 4× expansion in FFN**:
   - This is a universal constant: `d_ff = 4 * d_model`

4. **Vocabulary size**:
   - 32k-50k for most tasks
   - 100k for multilingual models

5. **Context window**:
   - 512-1024 for older/smaller models
   - 2048-4096 for modern models
   - 8192+ for long-context applications

---

## Part 5: Memory and Compute Estimates

### Memory Requirements

**Training** (most expensive):
- Model parameters: `4 × num_params` bytes (FP32)
- Optimizer states (Adam): `12 × num_params` bytes
- Gradients: `4 × num_params` bytes
- Activations: Depends on batch size and sequence length

**Total**: ~`20 × num_params` bytes for training

**Example: GPT-2 Small (124M params)**
- Training: ~2.5 GB (just model + optimizer)
- Add activations for batch: ~2-4 GB more
- **Total**: ~5-7 GB for training

**Inference** (much cheaper):
- Model: `4 × num_params` bytes (FP32) or `2 × num_params` (FP16)
- Activations: Minimal for single sequence

**Example: GPT-2 Small**
- Inference: ~500 MB (FP32), ~250 MB (FP16)

### Compute Requirements

Compute scales as **`O(12Ld²)`** per token:
- 6 operations in attention: `2×d² × seq_len`
- 6 operations in FFN: `2×4d² + 2×4d²`

**Example**: Llama 2 7B processing 2048 tokens:
- `d=4096, L=32`
- FLOPs ≈ `12 × 32 × 4096² × 2048 ≈ 13 trillion FLOPs`

On an A100 GPU (312 TFLOPS):
- Time ≈ 13T / 312T ≈ **40 milliseconds**

---

## Part 6: Quick Reference Guide

### Choose Your Architecture

```python
# Tiny model (for experimentation)
tiny_config = {
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "vocab_size": 32_000,
    "max_len": 1024
}
# Parameters: ~40M

# Small model (GPT-2 Small equivalent)
small_config = {
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "vocab_size": 50_000,
    "max_len": 2048
}
# Parameters: ~124M

# Medium model (GPT-2 Medium equivalent)
medium_config = {
    "d_model": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "vocab_size": 50_000,
    "max_len": 2048
}
# Parameters: ~355M

# Large model (7B scale)
large_config = {
    "d_model": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "vocab_size": 100_000,
    "max_len": 4096
}
# Parameters: ~7B
```

### Validation Checklist

Before training, verify:
- ✅ `d_model % n_heads == 0`
- ✅ `d_head = d_model // n_heads` is 64 or 128
- ✅ `d_ff = 4 * d_model` (if not specified explicitly)
- ✅ Memory requirements fit your hardware
- ✅ Vocabulary size covers your domain

---

## Summary

1. **Mathematical Constraints**: `d_model` must be divisible by `n_heads`, with `d_head` typically 64 or 128
2. **Scaling**: Increase both width (`d_model`) and depth (`n_layers`) together
3. **Real-World Patterns**: Follow established configurations (GPT-2, Llama) as starting points
4. **Parameter Count**: Dominated by `12Ld²` for the transformer layers
5. **Memory**: Training requires ~20× parameter count in bytes
6. **Design for Your Use Case**: Start small for experimentation, scale up based on performance needs

**Next Up: L09 – Training the LLM.** Now that you've designed your architecture, we'll learn how to train it with modern optimization techniques, learning rate schedules, and gradient accumulation.

---

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

# L05 - Normalization & Residuals: The Plumbing of Deep Networks

*How to stop gradients from vanishing and signals from exploding*

---

We have built the **Multi-Head Attention** engine, but there is a problem. In a deep LLM, we stack these layers dozens of times. As the data passes through these transformations, the numbers can drift: they might become tiny (vanishing) or massive (exploding).

If the numbers get weird, the model stops learning. To fix this, we use two critical "plumbing" techniques:
1.  **Residual (Skip) Connections:** "Don't forget what you just learned."
2.  **Layer Normalization:** "Keep the numbers in a healthy range."

By the end of this post, you'll understand:
- The intuition behind **Add & Norm**.
- Why **Residual Connections** allow for incredibly deep models.
- How to implement **LayerNorm** from scratch.

```{code-cell} ipython3
:tags: [remove-input]

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: Residual Connections (The Skip)

In a standard network, data flows like this: .
In a Transformer, we do this: .

We literally **add the input back to the output**.

### Why?

Imagine you are trying to describe a complex concept. If you only give the "transformed" explanation, you might lose the original context. By adding  back, we provide a "highway" for the original signal to travel through. This makes it much easier for the gradient to flow backwards during training.

---

## Part 2: Layer Normalization (The Leveler)

LayerNorm ensures that for every token, the mean of its features is  and the standard deviation is . This prevents any single feature or layer from dominating the calculation.

Unlike **BatchNorm** (common in CNNs), **LayerNorm** calculates statistics across the features of a single token. This makes it perfect for sequences of varying lengths.

### The Formula

For a vector :
$$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$

1. Calculate mean () and variance () of the features.
2. Subtract mean and divide by standard deviation.
3. **Learnable Parameters:**  (scale) and  (shift) allow the model to "undo" the normalization if it decides that a different range is better for learning.

---

## Part 3: Visualizing the "Add & Norm"

Let's see how LayerNorm tames wild values.

```{code-cell} ipython3
:tags: [remove-input]

# Create a "wild" vector of 100 features
np.random.seed(42)
wild_vector = np.random.normal(loc=5, scale=10, size=100)

# Normalize manually
mean = wild_vector.mean()
std = wild_vector.std()
norm_vector = (wild_vector - mean) / std

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(100), wild_vector, color='red', alpha=0.6)
ax1.set_title(f"Before LayerNorm\n(Mean: {mean:.2f}, Std: {std:.2f})")
ax1.set_ylim(-30, 40)

ax2.bar(range(100), norm_vector, color='green', alpha=0.6)
ax2.set_title(f"After LayerNorm\n(Mean: {norm_vector.mean():.2f}, Std: {norm_vector.std():.2f})")
ax2.set_ylim(-30, 40)

plt.tight_layout()
plt.show()

```

---

## Part 4: Building LayerNorm from Scratch

While PyTorch has `nn.LayerNorm`, building it yourself helps you understand exactly where those learnable parameters () live.

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        # Scale and Shift
        return self.gamma * x_norm + self.beta

```

---

## Summary

1. **Residual Connections** create a "high-speed rail" for the signal, preventing the vanishing gradient problem.
2. **LayerNorm** re-centers the data at every step, keeping the optimization process stable.
3. **The Pattern:** In a Transformer block, we follow the "Norm then Sub-layer" (Pre-Norm) or "Sub-layer then Norm" (Post-Norm) pattern. Most modern LLMs use **Pre-Norm** because it's more stable to train.

**Next Up: L06 – The Causal Mask.** When training a model to predict the next word, how do we stop it from "cheating" by looking at the answer? We'll build the triangular mask that hides the future.

---

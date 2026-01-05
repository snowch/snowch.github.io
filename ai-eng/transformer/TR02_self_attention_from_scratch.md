---
title: "TR02 — Self-Attention from Scratch (Q/K/V, Masking, Shapes)"
description: "Implement causal self-attention from scratch in PyTorch: tensor shapes, Q/K/V projections, masking, softmax, and attention weight inspection."
keywords:
  - self-attention
  - qkv
  - causal mask
  - pytorch
  - transformers from scratch
---

# TR02 — Self-Attention from Scratch (Q/K/V, Masking, Shapes)

Scope A: **PyTorch tensors + autograd**, but we implement attention ourselves (no `nn.MultiheadAttention`).

By the end, you’ll have a clean `CausalSelfAttention` module you can drop into a GPT-style model.

---

## 1) The mental model

For each token position *t*:

- \(q_t\): what I’m looking for
- \(k_i\): what token *i* offers as a match key
- \(v_i\): what content token *i* contributes if selected

Scores: \(s_{t,i} = q_t \cdot k_i\)

Weights: \(w_{t,*} = \text{softmax}(s_{t,*})\)

Output: \(o_t = \sum_i w_{t,i} v_i\)

---

## 2) Shapes cheat sheet (single batch)

Let:

- `B` batch size
- `T` sequence length
- `C = d_model`
- `nh` number of heads
- `hs = C // nh` head size

We store activations as:

- `x`: (B, T, C)
- `q, k, v`: (B, nh, T, hs)
- attention weights: (B, nh, T, T)
- output: (B, T, C)

---

## 3) Causal mask (the “no peeking” rule)

We need to enforce: token *t* can attend only to tokens \(\le t\).

Implementation trick: create a lower-triangular matrix and use it to set illegal logits to \(-\infty\) before softmax.

```python
mask = torch.tril(torch.ones(T, T, device=x.device))
att = att.masked_fill(mask == 0, float('-inf'))
```

---

## 4) Minimal implementation (multi-head causal self-attention)

This is a “from scratch” attention module:

- computes Q/K/V projections
- reshapes into heads
- applies scaled dot-product attention
- applies causal mask
- returns combined output

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # One linear that produces Q, K, V in a single matmul
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (batch, time, channels)

        qkv = self.qkv(x)                 # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)     # each (B, T, C)

        # (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))

        # Softmax -> weights
        w = F.softmax(att, dim=-1)
        w = self.dropout(w)

        # Weighted sum of values: (B, nh, T, hs)
        y = w @ v

        # Recombine heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y
```

---

## 5) Visualize attention weights (tiny example)

A simple debugging/teaching trick: run the module and inspect weights.

Modify the forward pass to optionally return weights:

```python
# inside forward(...)
# return y, w  # if you want weights too
```

Then plot a single head’s attention matrix for a tiny prompt.

```python
import matplotlib.pyplot as plt

# Suppose w is (B, nh, T, T)
head = 0
mat = w[0, head].detach().cpu().numpy()  # (T, T)

plt.figure(figsize=(6, 5))
plt.imshow(mat)
plt.title("Attention weights (head 0)")
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.colorbar()
plt.show()
```

Expected behavior:
- Upper triangle should be ~0 (masked)
- Later tokens can attend to earlier tokens

---

## 6) The bridge to KV cache (why K/V matter)

In this module, `k` and `v` are computed for **every token** at every forward pass.

During inference decode, you generate one token at a time — so recomputing `k` and `v` for old tokens repeatedly is wasteful.

That’s exactly what KV cache avoids.

Next:
- **TR03** will wrap this attention into a full transformer block
- **TR04** will train and generate with a tiny decoder model
- Then your **IN01** KV cache post becomes the optimization sequel

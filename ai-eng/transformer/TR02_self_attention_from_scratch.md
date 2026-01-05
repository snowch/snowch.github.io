---
title: "TR02 — Self-Attention from Scratch"
description: "Implement causal self-attention from scratch in PyTorch: Q/K/V projections, masking, tensor shapes, and attention-weight inspection."
keywords:
  - self-attention
  - qkv
  - causal mask
  - pytorch
  - transformers from scratch
---

# TR02 — Self-Attention from Scratch

Self-attention is the new “operator” that makes transformers different from classic feed-forward networks.

This tutorial builds a **minimal, correct** causal self-attention module that:

- projects embeddings into **Q / K / V**
- computes **scaled dot-product attention**
- applies a **causal mask** (no peeking ahead)
- returns the mixed output in the same shape as the input

This module becomes the attention sub-layer inside the transformer block in TR03.

---

## What you will build

By the end, you will have:

- a working `CausalSelfAttention` module
- a clear shape model for every tensor involved
- a tiny visualization that confirms the causal mask is working

---

## Setup (optional)

```{code-cell} ipython3
:tags: [remove-input]

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
```

---

## Part 1: The attention rule (one head)

For each token position `t`, attention produces an output vector:

- `q_t` (query): what token `t` is looking for
- `k_i` (key): what token `i` offers as a match
- `v_i` (value): what token `i` contributes as content

Mechanically:

1. similarity scores: `score[t, i] = q_t · k_i`
2. normalize: `w[t, :] = softmax(score[t, :])`
3. mix: `out_t = Σ_i w[t, i] v_i`

Matrix form:

\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

---

## Part 2: Shapes you must get right

Most transformer confusion is actually shape confusion.

Assume:

- `B` = batch size
- `T` = sequence length
- `C` = `d_model` (embedding width)
- `nh` = number of heads
- `hs` = head size = `C // nh`

### Core shapes (batch included)

- input `x`: **(B, T, C)**
- projections (before heads): `q, k, v`: **(B, T, C)**
- after splitting into heads: `q, k, v`: **(B, nh, T, hs)**
- attention logits: **(B, nh, T, T)**
- output per head: **(B, nh, T, hs)**
- recombined output: **(B, T, C)**

---

## Part 3: Causal mask (decoder-only rule)

Decoder-only attention must not look forward.

For token positions `0..T-1`, token `t` may attend to `0..t`.

That is exactly a lower-triangular matrix:

```{code-cell} ipython3
:tags: [remove-input]

import torch

T = 4
mask = torch.tril(torch.ones(T, T))
mask
```

When applied to logits, masked positions become `-inf` before softmax, which drives their probability to ~0.

---

## Part 4: Implement multi-head causal self-attention

This implementation does not use `nn.MultiheadAttention`.  
It is the “from scratch” version you can reason about line-by-line.

```{code-cell} ipython3
:tags: [remove-input]

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Produce Q, K, V in one matmul: (B, T, C) -> (B, T, 3C)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Final projection after concatenating heads
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, T, C = x.shape

        qkv = self.qkv(x)                 # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)     # each (B, T, C)

        # (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention logits: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        # Causal mask (broadcast to B, nh)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))

        # Softmax to weights
        w = F.softmax(att, dim=-1)
        w = self.dropout(w)

        # Weighted sum of values: (B, nh, T, hs)
        y = w @ v

        # Recombine heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        if return_weights:
            return y, w
        return y
```

---

## Part 5: Confirm the mask visually (tiny test)

Sanity check: for a short sequence, attention weights should be ~0 above the diagonal.

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)

B, T, C = 1, 8, 32
attn = CausalSelfAttention(d_model=C, n_heads=4, dropout=0.0)

x = torch.randn(B, T, C)
y, w = attn(x, return_weights=True)     # w: (B, nh, T, T)

head = 0
mat = w[0, head].detach().cpu().numpy()

plt.figure(figsize=(6, 5))
plt.imshow(mat)
plt.title("Attention weights (head 0)")
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.colorbar()
plt.show()
```

Expected:

- the upper-right triangle is near zero (masked)
- each row sums to ~1 (softmax)

---

## Summary

Causal self-attention is:

- a learned, input-dependent way to mix token information
- constrained by a causal mask so the model cannot look ahead
- implemented by careful reshaping and matrix multiplies

In TR03, this becomes one sub-layer inside the full transformer block.

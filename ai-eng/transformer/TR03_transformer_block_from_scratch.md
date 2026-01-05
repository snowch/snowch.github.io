---
title: "TR03 — A Transformer Block from Scratch"
description: "Build a GPT-style transformer block in PyTorch: pre-norm LayerNorm, residual connections, MLP, stacking layers, and a minimal decoder-only model."
keywords:
  - transformer block
  - residual connections
  - layernorm
  - decoder-only
  - gpt from scratch
  - pytorch
---

# TR03 — A Transformer Block from Scratch

A transformer block is just two sub-layers wrapped with residuals and normalization:

1. causal self-attention  
2. an MLP (feed-forward network)  

This tutorial assembles the full GPT-style block and stacks it into a minimal decoder-only model.

---

## What you will build

By the end, you will have:

- a `TransformerBlock` that matches the standard decoder-only layout
- a minimal `TinyGPT` model that returns logits over the vocabulary

---

## Part 1: The block layout (pre-norm)

A common modern layout is **pre-norm**:

1. `x = x + Attn(LN(x))`
2. `x = x + MLP(LN(x))`

```{mermaid}
flowchart LR
  X["x"] --> LN1["LayerNorm"]
  LN1 --> A["Causal self-attention"]
  A --> R1["Add residual"]
  R1 --> LN2["LayerNorm"]
  LN2 --> M["MLP"]
  M --> R2["Add residual"]
  R2 --> Y["x'"]
```

---

## Part 2: Attention module (self-contained)

To keep this page executable on its own, the attention module is included here as well.

```{code-cell} ipython3
:tags: [remove-input]

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))

        w = F.softmax(att, dim=-1)
        w = self.dropout(w)

        y = w @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y
```

---

## Part 3: The MLP sub-layer

The transformer MLP expands then contracts:

- `d_model -> 4*d_model -> d_model`
- GELU nonlinearity

```{code-cell} ipython3
:tags: [remove-input]

class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, mult: int = 4):
        super().__init__()
        hidden = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
```

---

## Part 4: The transformer block

```{code-cell} ipython3
:tags: [remove-input]

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

---

## Part 5: A minimal decoder-only model

A decoder-only model needs:

1. token embeddings  
2. positional embeddings  
3. a stack of blocks  
4. an output head to vocabulary logits  

```{code-cell} ipython3
:tags: [remove-input]

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying is a common GPT-style optimization
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_len

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
```

---

## Summary

A transformer block is:

- attention + MLP
- wrapped in residual connections
- stabilized by LayerNorm

A decoder-only model is:

- embeddings + positions
- a stack of blocks
- a vocabulary projection head

TR04 trains this model end-to-end and adds a baseline generation loop.

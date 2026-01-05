---
title: "TR03 — A Transformer Block from Scratch (Attention + MLP + Residuals)"
description: "Build a GPT-style transformer block from scratch in PyTorch: pre-norm, residual connections, MLP, stacking layers, and a minimal decoder-only model."
keywords:
  - transformer block
  - residual connections
  - layernorm
  - decoder-only
  - gpt from scratch
  - pytorch
---

# TR03 — A Transformer Block from Scratch (Attention + MLP + Residuals)

Now we take the attention module from TR02 and build a real transformer block:

- **Pre-norm** LayerNorm
- **Residual** connections
- **MLP** (feed-forward network)
- Stack blocks into a decoder-only model

This aligns with your NN-from-scratch work: attention is the “new operator”, everything else is stable engineering.

---

## 1) The block structure (pre-norm)

A common modern layout is **pre-norm**:

1. `x = x + Attn(LN(x))`
2. `x = x + MLP(LN(x))`

This is stable and widely used.

```{mermaid}
flowchart LR;
  X["x"] --> LN1["LayerNorm"];
  LN1 --> A["Causal self-attention"];
  A --> R1["Add residual"];
  R1 --> LN2["LayerNorm"];
  LN2 --> M["MLP"];
  M --> R2["Add residual"];
  R2 --> Y["x'"];
```

---

## 2) The MLP (feed-forward) sublayer

Classic transformer MLP:

- expand dimension (e.g., 4×)
- nonlinearity (GELU)
- project back

```python
import torch.nn as nn
import torch.nn.functional as F

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

## 3) The transformer block (from scratch)

```python
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

## 4) A minimal decoder-only model (GPT-ish)

We need:
- token embeddings
- positional embeddings
- a stack of blocks
- final layernorm
- linear head to vocab logits

```python
import torch

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (common): share token embedding and output head
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_len

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)                  # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
```

---

## 5) Where “all layers” shows up

This model stacks `n_layers` transformer blocks.
During inference, each block generates K/V for each token → so KV cache is stored **per layer**.

This is the key bridge to your inference engineering posts (KV cache, disaggregation, batching).

Next: TR04 will train this tiny model end-to-end.

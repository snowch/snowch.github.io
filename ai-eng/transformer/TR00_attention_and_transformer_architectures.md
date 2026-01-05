---
title: "TR00 — Attention and Transformer Architectures (The 15-minute Mental Model)"
description: "A quick, visual primer: what attention is, what a transformer block looks like, and the main architecture families (encoder-only, decoder-only, encoder–decoder)."
keywords:
  - attention
  - transformer architectures
  - encoder-only
  - decoder-only
  - encoder-decoder
  - qkv
  - causal mask
---

# TR00 — Attention and Transformer Architectures (The 15-minute Mental Model)

This is the on-ramp for the Transformer series (TR01–TR04).  
It’s deliberately short and visual.

If you’ve already read TR01 and felt like you needed a “map” first, this is that map.

---

## 1) Attention in one sentence

**Attention is a weighted sum of value vectors, where the weights are computed from how well the query matches each key.**

- **Query (Q):** “what I’m looking for”
- **Key (K):** “what I offer as a match”
- **Value (V):** “what I contribute if selected”

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

## 1.5) The attention module (architecture): what happens inside the box?

When you see a diagram that says **“Self-attention (Q,K,V)”**, the module is typically doing these steps:

1. Start from token vectors **x** (shape: `B × T × C`)
2. Project into **Q, K, V** using learned matrices:
   - `Q = x Wq`, `K = x Wk`, `V = x Wv`
3. Split into heads: `C = nh × hs`
4. Compute attention scores (per head): `scores = Q @ K^T / sqrt(hs)`
5. Apply a mask (for decoder-only): prevent looking at future tokens
6. Softmax → weights
7. Weighted sum: `out = weights @ V`
8. Concatenate heads and project back to `C`

Here’s a visual “inside the attention box” view:

```{mermaid}
flowchart LR;
  X["Input x (B,T,C)"] --> P["Linear projections"];
  P --> Q["Q (B,nh,T,hs)"];
  P --> K["K (B,nh,T,hs)"];
  P --> V["V (B,nh,T,hs)"];

  Q --> S["Scores = Q @ K^T / sqrt(hs)"];
  K --> S;

  S --> M["Mask (decoder-only)"];
  M --> W["Softmax -> weights"];
  W --> O["Out = weights @ V"];
  V --> O;

  O --> C["Concat heads -> (B,T,C)"];
  C --> Y["Output (B,T,C)"];
```

If you want to keep TR00 short: the goal is just to know what lives inside the attention box.
TR02 is where we implement it.



Don’t memorize the formula — the mental model is enough.

---

## 2) What a transformer block is

A (decoder) transformer block has two big parts:

1. **Self-attention** (tokens mix information from other tokens)
2. **MLP** (a per-token feed-forward network)

Plus stabilizers:

- **Residual connections** (skip connections)
- **LayerNorm** (stabilizes activations)

```{mermaid}
flowchart LR;
  X["Input (token vectors)"] --> A["Self-attention (Q,K,V)"];
  A --> R1["Residual + LayerNorm"];
  R1 --> M["MLP (feed-forward)"];
  M --> R2["Residual + LayerNorm"];
  R2 --> Y["Output"];
```

A transformer model is this block stacked **L** times.

---

## 3) Three architecture families (the taxonomy)

Transformers come in three common “shapes”:

### A) Encoder-only (BERT-style): *understand* text
- Reads the whole sequence at once
- Uses **bidirectional** attention (can look left and right)
- Great for classification, embeddings, extraction

```{mermaid}
flowchart LR;
  T["Tokens"] --> E["Encoder stack"] --> H["Representations"];
```

### B) Decoder-only (GPT-style): *generate* text
- Generates one token at a time
- Uses **causal** (masked) self-attention (can’t look ahead)
- Great for language modeling, chat, code generation

```{mermaid}
flowchart LR;
  P["Prompt tokens"] --> D["Decoder stack (causal)"] --> N["Next-token logits"];
  N --> G["Generate token and append"];
```

### C) Encoder–decoder (T5-style): *transform* text
- Encoder reads the input (bidirectional)
- Decoder generates output (causal)
- Connected by **cross-attention** (decoder attends to encoder states)
- Great for translation, summarization, structured generation

```{mermaid}
flowchart LR;
  X["Input tokens"] --> E["Encoder stack"] --> H["Encoder states"];
  Y["Output tokens so far"] --> D["Decoder stack (causal)"] --> Z["Next-token logits"];
  H -. "cross-attention" .-> D;
```

In this series, we focus on **decoder-only** because it’s the core of most LLM serving and inference engineering topics (KV cache, batching, etc.).

---

## 4) Self-attention vs cross-attention (quickly)

- **Self-attention:** tokens attend to tokens in the same sequence
- **Cross-attention:** decoder tokens attend to encoder outputs (in encoder–decoder models)

```{mermaid}
flowchart LR;
  subgraph SA["Self-attention"];
    A["Sequence A"] --> AT["Attend within A"];
  end;
  subgraph CA["Cross-attention"];
    B["Decoder tokens"] --> CT["Attend to encoder states"];
    C["Encoder states"] --> CT;
  end;
```

---

## 5) Causal masking (the “no peeking” rule)

Decoder-only models must not use future information when predicting the next token.

For 4 tokens (t0..t3), allowed attention looks like:

| query \ key | t0 | t1 | t2 | t3 |
|---|---:|---:|---:|---:|
| **t0** | ✅ | ❌ | ❌ | ❌ |
| **t1** | ✅ | ✅ | ❌ | ❌ |
| **t2** | ✅ | ✅ | ✅ | ❌ |
| **t3** | ✅ | ✅ | ✅ | ✅ |

Implementation-wise, you usually:
- add \(-\infty\) to masked-out logits
- then softmax yields ~0 probability there

---

## 6) Multi-head attention (why “heads” exist)

One attention “head” might learn one kind of relationship (e.g., syntax).
Multiple heads let the model learn several relationship types in parallel.

Think of it as running several attention mechanisms side-by-side, then concatenating.

```{mermaid}
flowchart LR;
  X["Token vectors"] --> H1["Head 1 attention"];
  X --> H2["Head 2 attention"];
  X --> H3["Head 3 attention"];
  H1 --> C["Concat + linear"];
  H2 --> C;
  H3 --> C;
  C --> Y["Output vectors"];
```

---

## 7) Where TR01–TR04 fit (series roadmap)

- **TR01 — Transformers: Tokens → Next-Token Prediction**  
  Big picture: tokenization, block overview, causal masking, training objective, training vs inference.

- **TR02 — Self-attention from scratch**  
  Implement Q/K/V + causal mask in PyTorch (no `nn.MultiheadAttention`).

- **TR03 — Transformer block from scratch**  
  Add residual + layernorm + MLP; stack blocks into a tiny GPT-like model.

- **TR04 — Train a tiny decoder model**  
  Create a next-token dataset, train end-to-end, and implement generation.

After TR04, your inference engineering posts (like KV cache) become the natural sequel:
- **IN01 — KV Cache Deep Dive** (your post)

---

## 8) What “from scratch” means in this series

We use **PyTorch tensors + autograd**, but we implement:
- attention math ourselves
- block structure ourselves
- training loop and generation ourselves

So you understand the mechanics and the systems implications — without building a whole deep learning framework.

That’s it. You now have the map. 🙂

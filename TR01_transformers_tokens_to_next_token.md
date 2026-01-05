---
title: "TR01 — Transformers: From Tokens to Next-Token Prediction (Training + Inference)"
description: "A practical, engineer-friendly tour of decoder-only transformers: tokenization, attention blocks, causal masking, next-token training, and how inference differs from training."
keywords:
  - transformers
  - decoder-only
  - next-token prediction
  - causal masking
  - llm training
  - llm inference
---

# TR01 — Transformers: From Tokens to Next-Token Prediction (Training + Inference)

You already have **neural networks from scratch** posts (forward/backprop, softmax/cross-entropy, building flexible architectures). This post focuses on what’s *new* in transformers: **self-attention, causal masking, next-token training**, and why inference becomes an engineering problem (KV cache, batching, etc.).

Related reading from your own site:
- NN tutorial: https://snowch.github.io/nn-tutorial-blog/
- NN flexible architecture: https://snowch.github.io/nn-flexible-network-blog/
- KV cache deep dive (systems sequel): IN01 (your post)

---

## 1) What problem are transformers solving?

Classic feed-forward nets and CNNs use **fixed** computation patterns. For sequences, we want:

- **Order matters** (token positions)
- **Long-range dependencies** (a word early can affect meaning later)
- **Parallel training** (use GPU efficiently)

Transformers solve this by letting each token compute a **weighted mix of other tokens** — the weights are *data-dependent* and change per input.

---

## 2) From raw text → token IDs → embeddings

A transformer doesn’t see words. It sees **token IDs**.

- Raw text: `I love Rust!`
- Tokens (illustrative): `["I", " love", " Rust", "!"]`
- Token IDs: `[314, 1234, 9876, 0]` (integers)
- Embeddings: each ID maps to a vector in \(\mathbb{R}^{d_{model}}\)

### Why “BPE” shows up everywhere
Many LLM tokenizers are **BPE-like** (Byte Pair Encoding): they split text into **subword pieces** so the vocab stays manageable while still representing arbitrary text.

For “from scratch” learning, you can start with a *toy* tokenizer (char-level or whitespace). The transformer mechanics are identical.

```{mermaid}
flowchart LR;
  A["Raw text"] --> B["Tokenizer\n(BPE in real LLMs)"];
  B --> C["Token strings"];
  C --> D["Token IDs (ints)"];
  D --> E["Embeddings (vectors)"];
```

---

## 3) The decoder-only transformer block (what you actually run)

A decoder-only transformer (GPT-style) is basically:

1. **Self-attention** (tokens look at earlier tokens)
2. **MLP** (a per-token feed-forward network)
3. **Residual connections + LayerNorm** (stability)

```{mermaid}
flowchart LR;
  X["Embeddings + positions"] --> A["Causal self-attention"];
  A --> R1["Residual + LayerNorm"];
  R1 --> M["MLP (feed-forward)"];
  M --> R2["Residual + LayerNorm"];
  R2 --> OUT["Next layer / logits"];
```

You stack this block **L times**.

> When posts say “KV cache for all layers”, they literally mean: each layer has its own K/V tensors cached for each token.

---

## 4) Self-attention in one sentence (and why it’s different)

Self-attention computes, for each token:

- a **query** vector Q (what I’m looking for)
- **keys** K (what others contain)
- **values** V (the content I’ll blend)

Then it uses similarity between Q and K to compute weights, and returns a weighted sum of values.

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
\]

The important difference vs a dense layer:

- Dense layer weights are **fixed parameters**
- Attention weights are **computed dynamically** from the input (token-to-token relationships)

---

## 5) Causal masking: why the model can’t “peek ahead”

For next-token prediction, token *t* must only attend to tokens \(\le t\).
That’s enforced by a **causal mask**.

For 4 tokens (t0..t3), allowed attention looks like:

| query \ key | t0 | t1 | t2 | t3 |
|---|---:|---:|---:|---:|
| **t0** | ✅ | ❌ | ❌ | ❌ |
| **t1** | ✅ | ✅ | ❌ | ❌ |
| **t2** | ✅ | ✅ | ✅ | ❌ |
| **t3** | ✅ | ✅ | ✅ | ✅ |

This is what makes a transformer a *language model* rather than a general sequence encoder.

---

## 6) How transformers are trained (next-token prediction)

Training objective: given a sequence of tokens \(t_0, t_1, ..., t_n\),
predict \(t_1, t_2, ..., t_{n+1}\).

- Inputs: `t0 t1 t2 t3`
- Targets: `t1 t2 t3 t4`

This is called **teacher forcing**: during training, you feed the *true* previous tokens.

```{mermaid}
flowchart LR;
  P["Input tokens\n(t0 t1 t2 t3)"] --> M["Model (masked)"];
  M --> S["Predict next tokens\n(t1 t2 t3 t4)"];
  S --> L["Cross-entropy loss\n(sum over positions)"];
```

Key training fact: **you compute loss at many positions in parallel** (GPU-friendly).

---

## 7) Training vs inference (the engineering cliff)

### Training (parallel over positions)
- You process the whole sequence at once (with a causal mask).
- You compute logits for every position in one forward pass.

### Inference (autoregressive)
- You do **prefill** once (process prompt, build initial state)
- Then **decode** one token at a time (generate token, append, repeat)

This is where inference performance engineering shows up:

- KV cache
- batching
- speculative decoding
- paged KV management

Your KV cache post (IN01) is the natural next step after this one.

---

## 8) What’s next in this series

- **TR02 — Self-Attention from Scratch:** Q/K/V, causal mask, shapes, and a minimal PyTorch implementation (no `nn.MultiheadAttention`).
- **TR03 — Transformer Block from Scratch:** add residuals, layernorm, MLP, stacking layers into a GPT-like model.
- **TR04 — Train a Tiny Decoder Model:** next-token dataset, training loop, and generation (prefill + decode).

If you can implement and explain these four pieces, you’re already doing “AI engineering”, not just “AI usage”.

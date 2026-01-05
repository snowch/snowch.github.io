---
title: "NN04 — Architecture Patterns: Tensors, Embeddings, Residuals, LayerNorm (Bridge to Transformers)"
description: "A hands-on bridge from feed-forward networks to transformers: how to think in (B,T,C), how embeddings work, and why residual + LayerNorm are everywhere."
keywords:
  - neural networks from scratch
  - embeddings
  - residual connections
  - layernorm
  - tensor shapes
  - bridge to transformers
---

# NN04 — Architecture Patterns: Tensors, Embeddings, Residuals, LayerNorm (Bridge to Transformers)

:::{seealso} Prerequisites
- NN tutorial (dense → softmax → predictions): {doc}`nn_tutorial_blog`
- Flexible architecture (stacking modules cleanly): {doc}`nn_flexible_network_blog`
- Edge detector (learned features, visual intuition): {doc}`nn_edge_detector_blog`
:::


## Learning goals

By the end of this chapter you will be able to:

1. Read and reason about **sequence-shaped tensors**: `(B, T, C)`
2. Explain **embedding lookup** (token IDs → vectors)
3. Recognize and implement **residual connections**: `y = x + f(x)`
4. Explain why transformers use **LayerNorm** (and where it sits in the block)
5. Understand the preview idea behind attention: **dynamic mixing weights**

This chapter is the **bridge** to transformers. By the end of the series (TR00–TR04), you’ll understand how attention works and how transformer blocks are built end-to-end.

---

## 1) Tensor shapes you must be fluent with

### 1.1 Single vector (one example)

A classic NN tutorial starts with one input vector:

- shape: `(C,)`

Example: `C=8` features:

- `x = [x0, x1, ..., x7]`

### 1.2 Batch of vectors

Training is almost always done on batches:

- shape: `(B, C)`

Example: `B=2`, `C=8`:

- `x_batch.shape = (2, 8)`  
  (two examples, each with 8 features)

:::{note} Where you’ve seen this before
In the {doc}`NN tutorial blog <nn_tutorial_blog#the-networks-job>`, you gather a batch of input vectors into a matrix, which is essentially `(B, C)`:
- `B` rows = examples in the batch
- `C` columns = features per example

In that tutorial, you pass a batch matrix `X` into a dense layer (`Y = X @ W + b`), where each row is one example and each column is a feature.

Transformers keep this exact structure but add one more axis: time / token position.
:::

### 1.3 Batch of sequences (transformer default)

Transformers add the sequence dimension because they process **ordered tokens**.  
The model needs a separate axis to keep track of token position so it can mix information across time.

- shape: `(B, T, C)`

Where:
- `B` = batch size
- `T` = number of tokens (sequence length)
- `C` = feature width (`d_model`)

**Concrete example:** `B=2`, `T=4`, `C=8`

- `x.shape = (2, 4, 8)`

Interpretation:
- 2 sequences in the batch
- each sequence has 4 tokens
- each token is represented by an 8-dim vector

```{mermaid}
flowchart TB;
  X["x: (B,T,C) = (2,4,8)"] --> S0["Sequence 0\n4 token-vectors (each length 8)"];
  X --> S1["Sequence 1\n4 token-vectors (each length 8)"];
```

### Check yourself
If `x.shape = (32, 128, 256)`, what are `B`, `T`, and `C`?

---

## 2) Embeddings: token IDs → vectors

Transformers do **not** take words as input.  
They take **token IDs** (integers).

### 2.1 Embedding table

An embedding is a matrix:

- `E.shape = (V, C)`  
  where `V = vocab_size`

Row `i` is the vector for token ID `i`.

### 2.2 Lookup is not a “normal” matrix multiply (but it *can be*)

If your token IDs are:

- `ids = [3, 1, 4, 0]`  (4 tokens)

Then the embedded sequence is:

- `X = [E[3], E[1], E[4], E[0]]`

Shape:
- `X.shape = (T, C)` → `(4, C)`

For a batch of sequences:
- `ids.shape = (B, T)`
- `X.shape = (B, T, C)`

```{mermaid}
flowchart LR;
  A["Token IDs (B,T)"] --> B["Embedding table E (V,C)"];
  B --> C["Vectors X (B,T,C)"];
```

:::{note} Connection to NNFS: embedding lookup = one-hot × matrix
In our NNFS posts, we use **one-hot vectors** (a 1 in one position, zeros elsewhere); see the {doc}`NN tutorial blog <nn_tutorial_blog#part-4-measuring-wrongness---the-loss-function>` for an example.

If `one_hot(id)` is a length-`V` vector, then:

- `one_hot(id) @ E` returns row `E[id]`

So embedding lookup is equivalent to multiplying by a matrix — it’s just implemented as a fast index operation.
:::

### Tiny numeric example (toy)

Assume `V=6`, `C=4`. The embedding table is:

| id | vector |
|---:|---|
| 0 | [ 0.2, -1.1, 0.7, 0.0 ] |
| 1 | [ 0.0,  0.3, 0.1, 0.9 ] |
| 2 | [ -0.4, 0.8, 0.0, 0.2 ] |
| 3 | [ 1.2,  0.0, -0.3, 0.5 ] |
| 4 | [ 0.6, -0.2, 0.9, -0.7 ] |
| 5 | [ -0.1, 0.1, 0.2, 0.3 ] |

Token IDs `[3, 1, 4, 0]` become vectors:

- `E[3], E[1], E[4], E[0]`

### Exercise
Pick `C=3` and invent a tiny embedding table with `V=5`. Encode a 4-token sequence by hand.

---

## 3) Linear layers on sequences (same idea, bigger tensor)

A linear layer maps feature width `C → C2` using:

- `W.shape = (C, C2)`

For **one token vector**:
- `y = x @ W` → shape `(C2,)`

For **a batch of sequences** `(B, T, C)`:
- apply the same linear layer to every token:
- output shape `(B, T, C2)`

:::{note} Where you’ve seen this before
In your **NN tutorial blog**, your dense layer is effectively:

- `Y = X @ W + b`

Transformers do the same operation — but `X` is `(B,T,C)` and you apply the same `W` to every token vector.
:::

### Check yourself
If `x.shape = (16, 64, 128)` and `W` maps `128 → 512`, what is the output shape?

---

## 4) Residual connections (skip connections)

A residual connection means:

$$
y = x + f(x)
$$

### Rule you must follow
For `x + f(x)` to work:

- `x` and `f(x)` must have the **same shape**

In transformers:
- attention output has shape `(B, T, C)`
- MLP output has shape `(B, T, C)`
- so both can be added back to `x`

```{mermaid}
flowchart LR;
  X["x"] --> F["f(x)"];
  X --> ADD["+"];
  F --> ADD;
  ADD --> Y["y"];
```

:::{note} Connection to NNFS: “architecture patterns”
In your **flexible network** post, you already treat layers/blocks as composable modules.

A residual connection is just another architecture pattern:
- compute a block output
- add it back to the input

This one pattern is a big reason deep transformer stacks train reliably.
:::

### Exercise
Give an example of a shape mismatch that would break a residual connection, and how you would fix it.

---

## 5) LayerNorm (why transformers use it)

LayerNorm normalizes each token vector across its feature dimension.

For one token vector `x` of shape `(C,)`:
- compute mean and std across the `C` features
- normalize: `(x - mean) / std`
- apply learned scale and shift

### Why LayerNorm (not BatchNorm)?
BatchNorm depends on batch statistics.
For variable-length sequences and autoregressive inference, LayerNorm behaves consistently.

### Where it appears in transformer blocks (preview)

Common modern pattern is **pre-norm**:

1. `x = x + Attention(LN(x))`
2. `x = x + MLP(LN(x))`

You will implement this in TR03.

---

## 6) Preview: “dynamic weights” (the bridge idea to attention)

Dense layers have fixed weights:

- `y = xW`

Attention creates **input-dependent** mixing weights.

You will see this later as:

- `weights = softmax(QK^T)`
- `out = weights V`

You do not need the details yet.
You only need this bridge idea:

> In attention, the model computes a matrix of weights from the input and then uses it to mix value vectors.

That’s why attention looks different from a normal layer, but it’s still linear algebra.

---

## 7) Roadmap (and how to read the series)

Recommended path:

1) Your NN-from-scratch posts (dense layers, training loop, flexible architectures)
2) **NN04 (this chapter)**
3) {doc}`TR00 <../transformer/TR00_attention_and_transformer_architectures>`: attention + transformer architecture map
4) **TR01–TR04**: implement a tiny decoder-only transformer end-to-end
5) **IN01**: inference engineering (KV cache, batching, memory)

:::{seealso} Next
Read {doc}`../transformer/TR00_attention_and_transformer_architectures` next. It assumes you can read `(B,T,C)` and you know what embeddings / residuals / LayerNorm do.
:::

---
title: "TR01 — Transformers: From Tokens to Next-Token Prediction"
description: "A step-by-step tutorial on decoder-only transformers: tokenization, causal self-attention, next-token training, and what changes during inference."
keywords:
  - transformers
  - decoder-only
  - gpt
  - next-token prediction
  - self-attention
  - causal masking
  - llm training
  - llm inference
---

# TR01 — Transformers: From Tokens to Next-Token Prediction

*A visual, step-by-step guide to how GPT-style transformers work (training + inference).*

* * *

Transformers can seem intimidating because they introduce a new core operation: **self-attention**.  
But at their core, decoder-only transformers do something very specific:

> Read a sequence of tokens and predict the **next** token.

This tutorial builds the “big picture” you need before implementing anything from scratch in TR02–TR04.

## By the end, you’ll understand

- What next-token prediction means (inputs vs targets)
- What tokenization is (and why BPE exists)
- The components of a decoder-only transformer block
- How causal self-attention works (with a concrete 4-token example)
- What “KV cache for all layers” means (in plain, mechanical terms)

---

# Setup (optional)

If you later turn this into an executed notebook, this avoids the common Matplotlib font-cache log line:

```python
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
```

---

## Part 1: The training objective — next-token prediction

Suppose the token sequence is:

`t0  t1  t2  t3  t4`

Training uses a one-token shift:

- **input**:  `t0  t1  t2  t3`
- **target**: `t1  t2  t3  t4`

So the model learns: “given tokens so far, what’s most likely next?”

```{mermaid}
flowchart LR
  A["Input: t0 t1 t2 t3"] --> B["Decoder-only transformer"]
  B --> C["Predict: t1 t2 t3 t4"]
  C --> D["Cross-entropy loss per position"]
```

### Why this trains efficiently on GPUs

During training, you can compute predictions for **all positions at once** (with a causal mask).  
That means fewer Python loops and more large matrix multiplies (good for GPUs).

---

## Part 2: Raw text → tokens → IDs → embeddings

Transformers do not operate on words. They operate on **token IDs** (integers).

```{mermaid}
flowchart LR
  A["Raw text"] --> B["Tokenizer"]
  B --> C["Token strings"]
  C --> D["Token IDs"]
  D --> E["Embeddings"]
```

### A concrete 4-token example

Take:

`I love Rust!`

A GPT-style tokenizer might produce something like:

- token strings (illustrative): `["I", " love", " Rust", "!"]`
- token IDs (illustrative): `[314, 1234, 9876, 0]`

Then an **embedding table** maps each ID to a vector:

- embedding width: `d_model`
- embeddings matrix shape: **(T, d_model)** → here **(4, d_model)**

### What “BPE” means (in plain English)

BPE (Byte Pair Encoding) is a way to build a vocabulary of **subword pieces**:

- common words can become single tokens (efficient)
- rare words can be represented as multiple subword tokens (still possible)
- any text can be represented using the learned merges

For “from scratch” learning, you can ignore BPE and start with:
- a character-level tokenizer, or
- a whitespace tokenizer

The transformer mechanics are identical.

---

## Part 3: What runs inside a decoder-only transformer

A GPT-style model stacks the same block **L times** (L = number of layers):

1. **Causal self-attention** (tokens look backwards)
2. **MLP / feed-forward** (per-token transformation)
3. **Residual connections + LayerNorm** (stability and training dynamics)

```{mermaid}
flowchart LR
  X["x: embeddings + positions"] --> LN1["LayerNorm"]
  LN1 --> A["Causal self-attention"]
  A --> R1["Add residual"]
  R1 --> LN2["LayerNorm"]
  LN2 --> M["MLP"]
  M --> R2["Add residual"]
  R2 --> Y["x: next layer"]
```

At the top, the model produces **logits** (scores over the vocabulary) for each position.  
Softmax converts logits into a probability distribution.

---

## Part 4: Self-attention with a 4-token toy example

Self-attention is easiest to understand by tracking **shapes**.

### Notation guide (same style as NN02)

- `T` = sequence length (number of tokens)  
- `d_model` = embedding width  
- `n_heads` = number of attention heads  
- `d_head` = head width (often `d_model / n_heads`)  

Common shapes (batch omitted for clarity):

- token IDs: **(T,)**
- embeddings: **(T, d_model)**
- per-head Q/K/V: **(T, d_head)**
- attention scores: **(T, T)**

### Step 1: Tokens become vectors

For tokens:

`t0="I"  t1=" love"  t2=" Rust"  t3="!"`

You start with embeddings:

`X` has shape **(4, d_model)**

### Step 2: Each layer produces Q, K, V

Inside a single attention layer, the model projects X into three matrices:

- `Q = X Wq`
- `K = X Wk`
- `V = X Wv`

For one head, each has shape **(T, d_head)**.

Intuition:
- **Q** (query): what this token wants to look for
- **K** (key): what each token offers as a match
- **V** (value): what content gets mixed in

### Step 3: Similarity scores and the causal mask

Attention compares each query to all keys:

`scores = Q Kᵀ` → shape **(T, T)**

Decoder-only models apply a **causal mask** so token *i* cannot attend to tokens *after* i.

Allowed pattern for 4 tokens:

| query \ key | t0 | t1 | t2 | t3 |
|---|---:|---:|---:|---:|
| **t0** | ✅ | ❌ | ❌ | ❌ |
| **t1** | ✅ | ✅ | ❌ | ❌ |
| **t2** | ✅ | ✅ | ✅ | ❌ |
| **t3** | ✅ | ✅ | ✅ | ✅ |

Mechanically, implementations set masked positions to a very negative number (effectively `-inf`) before softmax.

### Step 4: Softmax → weighted sum of V

- `weights = softmax(masked_scores)` (row-wise)
- `output = weights V` → shape **(T, d_head)**

So each token output is a learned mixture of earlier tokens.

```{mermaid}
flowchart LR
  X["X: embeddings"] --> Q["Q = X Wq"]
  X --> K["K = X Wk"]
  X --> V["V = X Wv"]
  Q --> S["scores = Q K^T"]
  K --> S
  S --> CM["mask future positions"]
  CM --> W["softmax -> weights"]
  W --> O["output = weights V"]
  V --> O
```

---

## Part 5: What “all layers” means (and why KV cache exists)

A transformer has **L layers**.  
Each layer computes its own Q/K/V for the current sequence.

So “K/V for all layers” literally means:

- Layer 0 has its own K and V tensors
- Layer 1 has its own K and V tensors
- …
- Layer L-1 has its own K and V tensors

### Why caching K and V helps during inference

During inference, tokens are generated one at a time.  
When a new token arrives at time step `T+1`:

- you compute **new Q/K/V for the new token**
- but you still need **K/V for all previous tokens** to attend over the full context

If you recompute old K/V at every step, you redo the same work repeatedly.

So inference engines store:

- `K_cache[layer, head, time, d_head]`
- `V_cache[layer, head, time, d_head]`

That stored state is the **KV cache**.

---

## Part 6: Training vs inference (why they feel different)

### Training: parallel over positions

- Feed the whole token sequence
- Use a causal mask
- Compute loss for many positions in one forward pass

### Inference: sequential decoding

Inference splits into two phases:

1. **Prefill**: run the prompt once to build initial attention state  
2. **Decode**: generate one token, append it, update KV cache, repeat  

This is where serving becomes a systems problem: memory, batching, cache layout.

---

## Summary

A decoder-only transformer can be summarized as:

1. Tokenize text into IDs  
2. Embed IDs into vectors  
3. Repeat **L blocks** of:
   - causal self-attention (dynamic mixing of earlier tokens)
   - MLP (per-token non-linear transform)
4. Project to vocab logits and train with next-token cross-entropy  

Two key anchors:

- Training is **parallel** over positions (mask enforces causality)
- Inference is **sequential**, so caching K/V becomes essential

---

## Next posts in the series

- **TR02 — Self-Attention from Scratch:** implement Q/K/V + causal masking (with exact shapes).
- **TR03 — Transformer Block from Scratch:** add residuals, LayerNorm, MLP, and stacking.
- **TR04 — Train a Tiny Decoder Model:** dataset, training loop, and generation.

If you want the systems follow-up after TR04, the next topic is KV cache and inference optimizations.

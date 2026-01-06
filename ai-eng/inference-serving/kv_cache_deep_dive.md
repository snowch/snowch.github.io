---
title: "IN01 - KV Cache Deep Dive: Memory Optimization for LLM Inference [DRAFT]"
description: "Understanding the key-value cache mechanism that makes LLM inference practical: what it speeds up, what it doesn't, why memory explodes, and how disaggregation helps at scale."
keywords:
  - llm inference
  - kv cache
  - transformer optimization
  - memory management
  - inference serving
---

# IN01 - KV Cache Deep Dive: Memory Optimization for LLM Inference [DRAFT]

*The critical optimization that makes LLM inference practical — and the memory tradeoff that comes with it.*

In this post, we’ll cover:

- What KV cache is (and why it exists)
- A quick attention primer (Q/K/V in plain terms)
- What KV cache **speeds up** — and what it **doesn’t**
- Why KV cache memory explodes with long context + concurrency
- Why memory fragmentation happens in practice
- How cache disaggregation helps at large scale (and when it’s worth it)

No PhD required — just curiosity about how production LLMs actually work.

## The Problem: Autoregressive Generation Is Expensive

LLMs generate text **one token at a time**. Each new token depends on *all previous tokens* (prompt + already-generated tokens).

If you do the “naive” thing, every time you generate a token you re-run lots of the same work again and again.


### A concrete toy example (visual): prompt tokens → keys/values (and what “all layers” means)

Let’s make this tangible with a tiny, made-up transformer. The *exact* token splits depend on the tokenizer, but the mechanics are the same.

#### Step 0 — From raw text to tokens (what “BPE” means)

Most LLMs don’t operate on words directly; they operate on **tokens** produced by a tokenizer.
A common tokenizer family is **BPE (Byte Pair Encoding)**: it starts from bytes/characters and repeatedly merges the most frequent adjacent pairs to form a vocabulary of “subword” units.

So you might see splits like:

- Raw text: `I love Rust!`
- Tokens (illustrative 4-token example): `["I", " love", " Rust", "!"]`

Sometimes you’ll get surprises (also illustrative), like splitting uncommon words:

- Raw text: `unbelievable`
- Tokens: `["un", "believ", "able"]` (or other subword chunks)

The key point: **the model sees a sequence of token IDs**, not words.

```{mermaid}
flowchart LR;
  A["Raw text prompt"] --> B["Tokenizer (e.g., BPE)"];
  B --> C["Token strings (4 tokens)\n[I] [ love] [ Rust] [!]"];
  C --> D["Token IDs (integers)"];
  D --> E["Embeddings (vectors)"];
```

#### Step 1 — A tiny transformer (2 layers, 2 heads)

We’ll use a toy setup:

- Prompt length **n = 4** tokens: `t0  t1  t2  t3`
- **L = 2** transformer layers
- **2 attention heads**
- `d_model = 8`, so per-head `d_head = 4`

At each layer ℓ, the model holds hidden states for every token:

- **Hℓ** has shape **(n, d_model)** = (4, 8)

Then it computes projections:

- **Qℓ = Hℓ · Wqℓ**
- **Kℓ = Hℓ · Wkℓ**
- **Vℓ = Hℓ · Wvℓ**

With multi-head attention, think of K and V as stored per head:

- **Kℓ** shape ≈ (n, n_heads, d_head) = (4, 2, 4)
- **Vℓ** shape ≈ (n, n_heads, d_head) = (4, 2, 4)

Here’s what “all layers” means visually: **each layer has its own K/V cache**.

```{mermaid}
flowchart TB;
  subgraph L0["Input"];
    T["Tokens: t0 t1 t2 t3"] --> E["Embeddings"];
  end;

  E --> L1["Layer 1 (attention + MLP)"];
  L1 --> L2["Layer 2 (attention + MLP)"];
  L2 --> OUT["Logits -> next token"];

  subgraph C1["KV cache (Layer 1)"];
    K1["K1 cache: 4 rows (t0..t3)"] --> V1["V1 cache: 4 rows (t0..t3)"];
  end;

  subgraph C2["KV cache (Layer 2)"];
    K2["K2 cache: 4 rows (t0..t3)"] --> V2["V2 cache: 4 rows (t0..t3)"];
  end;

  L1 -. "writes K/V for each token" .-> C1;
  L2 -. "writes K/V for each token" .-> C2;
```

#### Step 2 — What happens when you generate the next token?

When generating the next token (call it `t4`):

- You compute **only the new token’s** `k_new` and `v_new` at **each layer**
- You append one row to each layer’s cache:
  - `Kℓ_cache` grows from 4 → 5 rows
  - `Vℓ_cache` grows from 4 → 5 rows

```{mermaid}
flowchart LR;
  subgraph Before["Before generating t4"];
    B1["Layer 1 KV rows: 4"] --> B2["Layer 2 KV rows: 4"];
  end;
  subgraph Step["Generate token t4"];
    S1["Compute (k_new, v_new) at Layer 1"] --> S2["Append to Layer 1 cache"];
    S3["Compute (k_new, v_new) at Layer 2"] --> S4["Append to Layer 2 cache"];
  end;
  subgraph After["After appending t4"];
    A1["Layer 1 KV rows: 5"] --> A2["Layer 2 KV rows: 5"];
  end;

  Before --> Step --> After;
```



#### Step 3 — A tiny “numbers” example (one head, one layer)

To make attention feel less abstract, here’s a **toy single-head** example at **one layer**.

Assume `d_head = 2` and we already cached keys/values for the 4 prompt tokens:

**K cache (4 tokens × 2 dims)**

| token | k = [k1, k2] |
|---|---|
| t0 ("I") | [2, 0] |
| t1 (" love") | [1, 0] |
| t2 (" Rust") | [0, 0] |
| t3 ("!") | [-1, 0] |

**V cache (4 tokens × 2 dims)**

| token | v = [v1, v2] |
|---|---|
| t0 | [10, 0] |
| t1 | [5, 0] |
| t2 | [1, 0] |
| t3 | [0, 0] |

Now we’re generating the next token `t4`. The model computes a **new query** for the current position:

- `q_new = [1, 0]`

**Attention scores** are dot products (scaled by `sqrt(d_head)` in real models; we’ll ignore scaling here for simplicity):

- score(t0) = q·k0 = 1×2 + 0×0 = 2  
- score(t1) = q·k1 = 1×1 + 0×0 = 1  
- score(t2) = q·k2 = 1×0 + 0×0 = 0  
- score(t3) = q·k3 = 1×(-1) + 0×0 = -1  

Softmax turns these into weights (rounded):

- w ≈ [0.644, 0.237, 0.087, 0.032]

Finally, the attention output is a weighted sum of values:

- out = Σ wᵢ · vᵢ  
- out ≈ 0.644·[10,0] + 0.237·[5,0] + 0.087·[1,0] + 0.032·[0,0]  
- out ≈ **[7.713, 0]**

Two important takeaways:

1. **The K/V for t0..t3 were reused** — we didn’t recompute them during decode.
2. The **new query still compares against all cached keys** — long context still costs.

If you want an intuition anchor:

- **K/V are like “index cards” for each token at each layer**
- KV cache keeps those index cards around so you don’t have to recreate them every time you add a new token

### Without KV Cache (naive decode loop)

```{mermaid}
sequenceDiagram
  autonumber
  participant U as User
  participant M as LLM

  U->>M: Prompt tokens (length n)

  loop For each new token
    M->>M: Recompute K,V for all previous tokens (all layers)
    M->>M: Attention over full context
    M-->>U: Next token
  end
```

### With KV Cache (practical decode loop)

```{mermaid}
sequenceDiagram
  autonumber
  participant U as User
  participant M as LLM

  U->>M: Prompt tokens (length n)
  M->>M: Compute K,V for prompt once
  M->>M: Store K,V in KV cache

  loop For each new token
    M->>M: Compute K,V for NEW token only (all layers)
    M->>M: Append to KV cache
    M->>M: Attention using cached K,V
    M-->>U: Next token
  end
```

## Prefill vs Decode: The Two-Phase Mental Model

It helps to split inference into two phases:

- **Prefill**: process the prompt once (compute K/V for the whole prompt, build the initial KV cache)
- **Decode**: generate tokens one-by-one (append new K/V each step)

KV cache helps most during **decode**, because decode is where repeated work would otherwise explode.

## Transformer Attention: Q, K, V in One Page

Attention is easiest to understand if you treat Q/K/V as roles:

- **Query (Q)**: “What am I looking for?”
- **Key (K)**: “What do I contain?”
- **Value (V)**: “If you attend to me, here’s what you get.”

The attention formula is typically written as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

You don’t need to memorize it. The important bit is:

- **Q** from the current token compares against **all cached K**
- those scores weight **all cached V**
- the result produces the output for the next layer / logits

Here’s the flow:

```{mermaid}
flowchart TD;
  X["Input tokens"] --> E["Embeddings"];
  E --> L["Transformer layer i"];

  L --> Q["Compute Q"];
  L --> K["Compute K"];
  L --> V["Compute V"];

  K --> KC["KV cache: Keys"];
  V --> VC["KV cache: Values"];

  Q --> A["Attention = softmax(QK^T / sqrt(d)) * V"];
  KC --> A;
  VC --> A;

  A --> O["Layer output"] --> NEXT["Next layer / logits"];
```

## The KV Cache Optimization

### The Revelation

During decode, for *previous tokens*:

- their **keys** don’t change
- their **values** don’t change

So recomputing K/V for them every decode step is pure waste.

### Pseudocode: without vs with cache

**Without KV cache (wasteful):**
```python
# At each decode step:
Q, K, V = compute_QKV_for_all_tokens(context_tokens)  # recompute everything
y = attention(Q, K, V)
next_token = sample(y)
```

**With KV cache (efficient projections):**
```python
# Prefill once:
K_cache, V_cache = compute_KV_for_prompt(prompt_tokens)

# Decode loop:
for t in range(num_new_tokens):
    q_new, k_new, v_new = compute_QKV_for_new_token(last_token)
    K_cache.append(k_new)
    V_cache.append(v_new)

    y = attention(q_new, K_cache, V_cache)
    next_token = sample(y)
```

**What you saved:** recomputing K/V projections for all previous tokens at every step.

## What KV Cache Does *Not* Fix

KV cache removes repeated **K/V projection work**, but it does **not** remove the need to do attention against the context.

During decode, the model still needs to compute something equivalent to:

- compare the **new query** to **all previous keys**
- produce a weighted sum over **all previous values**

So even with KV cache, decode attention work still grows with context length.

**Practical takeaway:**  
KV cache is necessary — but long context still costs you (compute + memory).

## Speedup Analysis (Accurate Version)

Let:

- \(n\) = current context length (prompt + generated so far)
- \(L\) = number of layers
- \(d\) = hidden dimension

A simplified breakdown per decode step:

| Component (per step) | Without KV Cache | With KV Cache |
|---|---:|---:|
| K/V projections for previous tokens | \(O(n \cdot L \cdot d)\) | **0** (reused) |
| K/V projections for new token | \(O(L \cdot d)\) | \(O(L \cdot d)\) |
| Attention vs context (new token attends to n tokens) | \(O(n \cdot L \cdot d)\) | \(O(n \cdot L \cdot d)\) |
| **Total per step** | big constant * \(O(nLd)\) + redundant work | **smaller constant** * \(O(nLd)\) |

So why is KV cache such a big deal?

Because without KV cache, the *projection work* for old tokens repeats every step and your total work across a response can grow roughly **quadratically** with the number of generated tokens.

With KV cache, projection work is closer to **linear** in generated tokens (plus the attention term, which remains).

## The Memory Tradeoff: Cache Size Explodes at Scale

KV cache stores **K and V tensors** for each token, for each layer.

A very common back-of-the-envelope estimate:

\[
\text{KV Memory} \approx 2 \times L \times n \times d \times p
\]

Where:

- \(2\) is for **K and V**
- \(p\) is bytes per element (e.g., FP16 = 2 bytes)

Here’s a visual factor breakdown:

```{mermaid}
flowchart LR;
  M["KV cache memory per request"] --> F1["2x (K and V)"];
  M --> F2["L layers"];
  M --> F3["n tokens"];
  M --> F4["d hidden dim"];
  M --> F5["p bytes per element"];
```

### Real-World Example: Llama 2 70B (illustrative)

Assume:

- \(L = 80\)
- \(d = 8192\)
- \(n = 4096\)
- \(p = 2\) bytes (FP16)

\[
\text{Memory} = 2 \times 80 \times 4096 \times 8192 \times 2 \approx 10.7 \text{ GB (decimal)} \approx 10.0 \text{ GiB}
\]

That’s **per request** for KV cache.

**Now multiply by concurrency.**  
Batch 32: ~\(10.7 \times 32 \approx 342\) GB of KV cache memory.

And this is separate from model weights (e.g., ~140 GB for a 70B model in FP16).

## Why Memory Fragmentation Shows Up

Even if your GPU has “enough free memory” in total, allocations can fail if free space is chopped into non-contiguous holes.

```{mermaid}
flowchart LR;
  subgraph GM["GPU memory (example)"];
    A["Request A: 10GB"] --> F1["Free: 5GB"];
    F1 --> B["Request B: 8GB"];
    B --> F2["Free: 3GB"];
    F2 --> C["Request C: 12GB"];
  end;
  NOTE["Total free = 8GB, but no contiguous 8GB block"];
  F1 -.-> NOTE;
  F2 -.-> NOTE;
```

This becomes more likely with:

- many concurrent requests
- variable sequence lengths
- frequent allocation/free cycles

## Modern Solution: Cache Disaggregation

If KV cache dominates memory, one idea is to avoid keeping *all* KV cache in GPU memory.

Instead:

- keep a smaller **active working set** on GPU
- keep a larger cache pool in **external memory** (CPU RAM, NVMe, or specialized storage)
- move KV in/out as needed (often with techniques like paging)

```{mermaid}
flowchart LR;
  subgraph T["Traditional (monolithic)"];
    GPU1["GPU memory"] --> W1["Model weights"];
    GPU1 --> C1["KV cache (all requests)"];
  end;

  subgraph D["Disaggregated cache"];
    GPU2["GPU memory"] --> W2["Model weights"];
    GPU2 --> WS["Active KV working set"];
    EXT["External KV store"] --> GPU2;
    GPU2 --> EXT;
  end;
```

**Why this helps:**
- Scale cache capacity independently of GPU memory
- Support longer context and/or higher concurrency
- Enable reuse when many requests share prefixes (depending on your serving stack)

**Tradeoffs:**
- Moving KV has overhead (latency + bandwidth)
- You need careful policies for what stays “hot” on GPU
- More moving parts operationally

> Note: Different vendors and open-source stacks approach disaggregated caching differently. The important idea is the architecture pattern (separating weights + active KV on GPU from a larger KV pool elsewhere), not any single implementation.

## When Disaggregation Is Worth It

| Scenario | Traditional KV Cache | Disaggregated Cache |
|---|---|---|
| Short contexts (<2K) | ✅ simplest + fast | ⚠ overhead not worth it |
| Long contexts (>8K) | ❌ GPU memory bottleneck | ✅ scales beyond GPU memory |
| High concurrency | ❌ hits memory ceiling fast | ✅ higher headroom |
| Shared prefixes (some workloads) | ⚠ limited reuse | ✅ potential reuse wins |
| Low-latency single-user | ✅ best latency | ⚠ extra indirection |

## Practical Engineering Checklist

If you’re building or evaluating an LLM serving stack, ask:

1. What is our **max context** and typical prompt length?
2. What is our target **concurrency** (p95/p99), not just average?
3. Are we memory-bound on **KV cache** or compute-bound on decode?
4. Can we reduce KV footprint (precision/quantization choices) without harming quality?
5. Do we need techniques like **paging / working-set KV** for long context?
6. Are we measuring **prefill vs decode** separately (latency + throughput)?
7. Do we have guardrails for **fragmentation** and variable-length spikes?

## Summary

- KV cache is essential because it avoids recomputing **K/V for old tokens** every decode step.
- KV cache does **not** eliminate attention’s dependence on context length — long context still costs.
- KV memory grows with layers, context length, and concurrency and can quickly dominate GPU memory.
- At large scale, fragmentation and KV growth motivate **disaggregation** patterns (active KV on GPU, larger KV pool elsewhere).

If you want a follow-up post, the natural sequel is:

**IN02 — KV Cache Meets Serving: Prefill/Decode, Continuous Batching, and Where Latency Actually Goes**

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

# L17 - Attention Optimizations [DRAFT]

*Flash Attention, KV Cache, and making attention 10× faster*

---

In [L03](L03_The_Attention_Mechanism.md), we learned that attention is $O(n^2)$ in sequence length. For a 2048-token context, that's 4 million comparisons! This lesson covers the optimizations that make long-context LLMs practical.

By the end of this post, you'll understand:
- Flash Attention: Faster attention with less memory
- KV Cache: Avoiding redundant computation during inference
- Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
- Sliding window attention for ultra-long contexts

---

## Part 1: The Memory Problem

### Standard Attention Memory Usage

Recall the attention computation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Memory bottleneck**: The intermediate attention matrix $QK^T$

For:
- Batch size $B = 32$
- Sequence length $N = 2048$
- Number of heads $H = 12$

Attention matrix size: $B \times H \times N \times N = 32 \times 12 \times 2048 \times 2048$

$$= 1.6 \text{ billion elements} \times 4 \text{ bytes} = \mathbf{6.4 \text{ GB}}$$

**Problem**: This is just for ONE layer! A 24-layer model needs 150+ GB just for attention matrices.

---

## Part 2: Flash Attention

### The Key Insight: Avoid Materializing Attention Matrices

Flash Attention computes attention **without storing** the full $QK^T$ matrix by:

1. **Tiling**: Break sequences into blocks
2. **Recomputation**: Recompute attention scores during backward pass
3. **Fused kernels**: Do softmax + matmul in one GPU kernel

**Result**: Same output, but memory usage drops from $O(N^2)$ to $O(N)$.

### Conceptual Algorithm

```python
# Standard attention (simplified)
def standard_attention(Q, K, V):
    # Shape: [batch, heads, seq, seq]
    scores = Q @ K.T / sqrt(d_k)  # ❌ Materializes N×N matrix

    attn = softmax(scores, dim=-1)
    output = attn @ V
    return output

# Flash Attention (conceptual)
def flash_attention(Q, K, V, block_size=128):
    N = Q.shape[2]  # sequence length
    output = torch.zeros_like(Q)

    # Process in blocks
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            # Load blocks to SRAM (fast memory)
            Q_block = Q[:, :, i:i+block_size, :]
            K_block = K[:, :, j:j+block_size, :]
            V_block = V[:, :, j:j+block_size, :]

            # Compute attention for this block only
            scores_block = Q_block @ K_block.T / sqrt(d_k)
            attn_block = softmax(scores_block, dim=-1)
            output_block = attn_block @ V_block

            # Accumulate results
            output[:, :, i:i+block_size, :] += output_block

    return output
```

**Key difference**: Never allocates the full $N \times N$ matrix!

---

### Using Flash Attention in Practice

```python
# Install: pip install flash-attn
from flash_attn import flash_attn_qkvpacked_func

# Your Q, K, V tensors
# Shape: [batch, seq_len, num_heads, head_dim]
qkv = torch.stack([Q, K, V], dim=2)  # [batch, seq, 3, heads, head_dim]

# Flash Attention forward pass
output = flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    causal=True,  # Apply causal mask
    softmax_scale=1.0 / sqrt(d_k)
)

# Same result as standard attention, but faster and less memory!
```

### Performance Comparison

**GPT-2 (124M) on A100, seq_len=2048**:

| **Method** | **Memory** | **Speed** | **Speedup** |
|---|---|---|---|
| Standard Attention | 24 GB | 100 ms | 1.0× |
| Flash Attention | 8 GB | 15 ms | **6.7×** |

---

## Part 3: KV Cache for Inference

### The Problem: Redundant Computation

During autoregressive generation, we recompute attention for all previous tokens:

```
Step 1: "The" → Compute attention for ["The"]
Step 2: "The cat" → Compute attention for ["The", "cat"]
Step 3: "The cat sat" → Compute attention for ["The", "cat", "sat"]
```

**Waste**: Keys and Values for "The" and "cat" are recomputed every step!

---

### The Solution: Cache K and V

```python
class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # KV Cache (initially empty)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        batch, seq_len, d_model = x.shape

        # Compute Q, K, V for new token(s)
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.cache_k is not None:
                # Concatenate with cached K, V
                k = torch.cat([self.cache_k, k], dim=2)  # [batch, heads, total_seq, head_dim]
                v = torch.cat([self.cache_v, v], dim=2)

            # Update cache
            self.cache_k = k
            self.cache_v = v

        # Standard attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = attn @ v

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.o_proj(output)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
```

### Using KV Cache

```python
model = GPTWithKVCache(config)

# Generation loop
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt)

model.reset_cache()  # Clear cache before generation

for step in range(max_new_tokens):
    # Only pass the LAST token (not entire sequence!)
    if step == 0:
        x = torch.tensor([input_ids])  # First step: full prompt
    else:
        x = torch.tensor([[last_token]])  # Subsequent: only new token

    logits = model(x, use_cache=True)  # Cache grows internally
    last_token = logits[0, -1].argmax().item()
    input_ids.append(last_token)
```

### Memory Trade-off

**Without KV Cache**:
- Memory: $O(1)$ (only current token)
- Compute: $O(N^2)$ per token (recompute all attention)

**With KV Cache**:
- Memory: $O(N)$ (store K, V for all tokens)
- Compute: $O(N)$ per token (only new comparisons)

**Speed improvement**: 10-100× faster for long sequences!

---

## Part 4: Multi-Query Attention (MQA)

### The Problem: KV Cache Memory

For a 7B model with 32 heads and 2048 context:

$$\text{KV cache size} = 2 \times \text{layers} \times \text{heads} \times \text{seq\_len} \times \text{head\_dim}$$
$$= 2 \times 32 \times 32 \times 2048 \times 128 \times 2 \text{ bytes} = \mathbf{1 GB}$$

With batch size 32: **32 GB just for KV cache!**

---

### MQA: Share K and V Across Heads

**Standard Multi-Head Attention**:
- Each head has its own $Q, K, V$

**Multi-Query Attention**:
- Each head has its own $Q$
- **All heads share** the same $K, V$

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Each head gets separate Q projection
        self.q_proj = nn.Linear(d_model, d_model)

        # Only ONE K, V projection (shared across heads)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)

        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Q: separate for each head
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]

        # K, V: shared across all heads
        k = self.k_proj(x).view(batch, seq_len, 1, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, 1, self.head_dim)

        # Broadcast K, V to all heads
        k = k.expand(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.expand(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = attn @ v

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.o_proj(output)
```

**KV Cache Reduction**: $32 \times$ smaller (1 GB → 32 MB)!

---

## Part 5: Grouped Query Attention (GQA)

### MQA vs. GQA: The Trade-off

**MQA**:
- Pros: Smallest KV cache
- Cons: Slight quality degradation (less expressive)

**GQA**: Middle ground
- Group heads together
- Each group shares K, V

**Example**: 32 heads, 4 groups
- 4 separate K, V projections (one per group)
- Each group has 8 heads

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Q: all heads
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        # K, V: num_kv_heads only
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Repeat each KV head for its group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)  # [batch, seq, num_heads, head_dim]
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Standard attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = attn @ v

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.o_proj(output)

# Usage: 32 heads, 8 KV heads (4 queries per KV)
gqa = GroupedQueryAttention(d_model=512, num_heads=32, num_kv_heads=8)
```

**Performance**: Llama 2 uses GQA for optimal quality/speed balance.

---

## Part 6: Sliding Window Attention

### For Ultra-Long Contexts (100k+ tokens)

Full attention at 100k tokens: $100k \times 100k = 10B$ comparisons!

**Solution**: Each token only attends to nearest neighbors.

```python
def sliding_window_attention(q, k, v, window_size=512):
    """
    Each token attends to [i-window_size, i+window_size].
    """
    seq_len = q.size(2)

    # Create sliding window mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False  # Allow attention in window

    # Standard attention with mask
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores.masked_fill_(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    output = attn @ v

    return output
```

**Trade-off**:
- Complexity: $O(N \times W)$ instead of $O(N^2)$
- Quality: Long-range dependencies are weakened

**Used by**: Longformer, BigBird

---

## Summary

1. **Flash Attention**: 5-10× faster by avoiding attention matrix materialization
2. **KV Cache**: 10-100× faster inference by caching Keys and Values
3. **Multi-Query Attention**: 32× smaller KV cache by sharing K, V across heads
4. **Grouped Query Attention**: Balanced trade-off (Llama 2's choice)
5. **Sliding Window**: Linear complexity for ultra-long contexts

**Next Up: L17 – Model Parallelism.** How to train models too large for a single GPU!

---

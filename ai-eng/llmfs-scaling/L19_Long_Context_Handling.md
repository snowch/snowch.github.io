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

# L18 - Long Context Handling [DRAFT]

*Extending context from 2K to 100K+ tokens*

---

In [L02](L02_Embeddings_and_Positional_Encoding.md), we learned about sinusoidal positional encodings. But what if we need 100K token contexts? Absolute positions don't scale well. This lesson covers modern alternatives.

By the end of this post, you'll understand:
- Rotary Positional Embeddings (RoPE) - the current standard
- ALiBi: Attention with Linear Biases
- Context length extrapolation (train on 2K, inference at 8K)
- Sparse attention patterns for efficiency

---

## Part 1: The Problem with Absolute Positions

### Sinusoidal Encodings Don't Extrapolate

Recall from L02:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

**Problem**: Trained on positions 0-2047, what happens at position 5000?

```python
# Model sees position 5000 for the first time
pos = torch.tensor([5000])
pe = positional_encoding(pos, d_model=512)

# Model has NEVER seen these PE values during training!
# Performance degrades significantly
```

**Solution**: Use **relative** positional information instead of absolute.

---

## Part 2: Rotary Positional Embeddings (RoPE)

### The Key Insight: Rotate Query and Key Vectors

Instead of adding position to embeddings, **rotate** Q and K based on their positions.

**Formula**:

$$\text{RoPE}(x, m) = \begin{bmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) \\
\sin(m\theta_1) & \cos(m\theta_1)
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix}$$

Where $m$ is the position and $\theta_i = 10000^{-2i/d}$.

**Key property**: The dot product $Q \cdot K$ encodes relative distance!

$$\text{RoPE}(Q, m) \cdot \text{RoPE}(K, n) = f(Q, K, m-n)$$

Depends on $(m-n)$, not absolute positions.

---

### Implementing RoPE

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    """Precompute rotation frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i*theta)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to queries and keys."""
    # Reshape to (batch, seq, heads, head_dim // 2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Convert to complex numbers
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # Reshape freqs_cis to match
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim//2]

    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Usage in attention
class RoPEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        # Precompute frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len * 2  # Allow extrapolation!
        )

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply RoPE to Q and K (not V!)
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:x.size(1)])

        # Standard attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = attn @ v

        return output
```

---

### Why RoPE Extrapolates

Trained on positions 0-2047, tested at position 5000:

**Absolute PE**: Embedding for 5000 is out-of-distribution ❌

**RoPE**: Relative distance $(5000 - 4995 = 5)$ was seen during training ✅

**Result**: RoPE allows **2-4× context extension** without retraining!

---

## Part 3: ALiBi (Attention with Linear Biases)

### Even Simpler: No PE at All!

**Idea**: Add a negative bias to attention scores based on distance.

$$\text{Attention}_{ij} = q_i \cdot k_j - \lambda \cdot |i - j|$$

Where $\lambda$ is a per-head constant (learned during training).

```python
class ALiBiAttention(nn.Module):
    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads

        # Create ALiBi slopes (one per head)
        slopes = torch.tensor([2 ** (-8 * i / num_heads) for i in range(num_heads)])
        self.register_buffer('slopes', slopes)

        # Precompute distance matrix
        positions = torch.arange(max_seq_len)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        self.register_buffer('distances', distances.abs())

    def forward(self, q, k, v):
        # Standard attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        # Add ALiBi bias: -slope * distance
        seq_len = scores.size(-1)
        alibi_bias = -self.slopes.view(-1, 1, 1) * self.distances[:seq_len, :seq_len]

        scores = scores + alibi_bias

        attn = F.softmax(scores, dim=-1)
        output = attn @ v

        return output
```

**Advantages**:
- No position embeddings needed at all!
- Extrapolates even better than RoPE (up to 10× context)
- Used by BLOOM, MPT models

**Visualization: ALiBi Bias Matrix**

```{code-cell} ipython3
:tags: [remove-input]

# TODO: Heatmap showing:
# - ALiBi bias for one head
# - Diagonal is 0 (no penalty for self-attention)
# - Values decrease linearly away from diagonal
# - Penalizes attending to distant tokens
```

---

## Part 4: Context Length Extrapolation Techniques

### Technique 1: Position Interpolation

**Problem**: Trained on 2048 tokens, want to use 4096 at inference.

**Solution**: Rescale positions to fit the learned range.

```python
# During training: positions 0 to 2047
# At inference (4096 tokens): map [0, 4095] → [0, 2047]

scale = training_length / inference_length  # 2048 / 4096 = 0.5

def rescale_positions(pos, scale):
    return pos * scale

# Apply to RoPE frequencies
freqs_rescaled = precompute_freqs_cis(dim, end, theta=10000.0 / scale)
```

**Result**: Smooth transition to longer contexts.

---

### Technique 2: YaRN (Yet another RoPE extensioN)

**Idea**: Different frequency dimensions need different scaling.

```python
def yarn_scaling(freqs, scale, alpha=1.0):
    """
    Low frequencies (slow oscillations) → no scaling
    High frequencies (fast oscillations) → more scaling
    """
    low_freq_wavelen = training_length / 2  # Wavelength threshold
    high_freq_wavelen = training_length / 8

    # Compute wavelengths for each frequency
    wavelengths = 2 * np.pi / freqs

    # Apply different scaling based on wavelength
    for i, wl in enumerate(wavelengths):
        if wl > low_freq_wavelen:
            # Low frequency: no change
            pass
        elif wl < high_freq_wavelen:
            # High frequency: full scaling
            freqs[i] *= scale
        else:
            # Interpolate between no scaling and full scaling
            ratio = (wl - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
            freqs[i] *= (scale ** (1 - ratio))

    return freqs
```

**Used by**: Llama 2 for extending 4K → 32K context.

---

## Part 5: Sparse Attention Patterns

### For Ultra-Long Contexts (100K+)

Even with RoPE/ALiBi, $O(N^2)$ attention is expensive.

**Solution**: Attend to only a subset of tokens.

### Pattern 1: Block-Sparse Attention

```python
def block_sparse_attention(q, k, v, block_size=64):
    """
    Tokens attend to:
    1. Their own block
    2. First block (global attention)
    """
    seq_len = q.size(2)
    num_blocks = seq_len // block_size

    outputs = []

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size

        # Block-local attention
        q_block = q[:, :, start:end]
        k_local = k[:, :, start:end]
        v_local = v[:, :, start:end]

        # Also attend to first block (global context)
        k_global = k[:, :, :block_size]
        v_global = v[:, :, :block_size]

        # Concatenate
        k_combined = torch.cat([k_local, k_global], dim=2)
        v_combined = torch.cat([v_local, v_global], dim=2)

        # Attention
        scores = (q_block @ k_combined.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(scores, dim=-1)
        output_block = attn @ v_combined

        outputs.append(output_block)

    return torch.cat(outputs, dim=2)
```

**Complexity**: $O(N \times B)$ where $B$ is block size.

---

### Pattern 2: Strided Attention

```python
def strided_attention(q, k, v, stride=64):
    """
    Attend to every stride-th token.
    """
    # Local attention (stride=1)
    local_k = k
    local_v = v

    # Strided attention (every 64th token)
    strided_k = k[:, :, ::stride]
    strided_v = v[:, :, ::stride]

    # Combine
    k_combined = torch.cat([local_k, strided_k], dim=2)
    v_combined = torch.cat([local_v, strided_v], dim=2)

    # Attention
    scores = (q @ k_combined.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn = F.softmax(scores, dim=-1)
    output = attn @ v_combined

    return output
```

**Used by**: Longformer for document understanding.

---

## Part 6: Practical Recommendations

| **Context Length** | **Technique** | **Model Examples** |
|---|---|---|
| 2K - 4K | Sinusoidal or learned PE | GPT-2, BERT |
| 4K - 32K | RoPE + position interpolation | Llama 2, GPT-4 |
| 32K - 100K | ALiBi or YaRN | BLOOM, MPT-30B |
| 100K+ | RoPE + sparse attention | Longformer, BigBird |

---

## Summary

1. **RoPE**: Rotary embeddings encode relative positions, extrapolate 2-4×
2. **ALiBi**: Linear biases, no PE needed, extrapolate up to 10×
3. **Position Interpolation**: Rescale positions to fit training range
4. **YaRN**: Frequency-specific scaling for better extrapolation
5. **Sparse Attention**: Block-sparse or strided for 100K+ contexts

**Next Up: L19 – Quantization for Inference.** Shrink models from FP16 to INT8/INT4!

---

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

# L04 - Multi-Head Attention: Seeing in Parallel

*Why one "eye" isn't enough for complex language*

---

In [L03 - Self-Attention](L03_The_Attention_Mechanism.md), we built a mechanism that allows a word to look at its neighbors. But language is complex. A single word might need to focus on:
1. **Grammar:** "Which verb does this noun belong to?"
2. **Logic:** "What does the word 'it' refer to?"
3. **Relationships:** "Who is performing the action?"

If we only have one attention mechanism, the model has to "squash" all these different questions into one score. **Multi-Head Attention** solves this by giving the model multiple "heads" (independent attention engines) to look at the sentence in parallel.

By the end of this post, you'll understand:
- The intuition of "feature splitting."
- Why we project $Q, K, V$ into different subspaces.
- How to implement the Multi-Head Attention module from scratch.

```{code-cell} ipython3
:tags: [remove-input]

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: The Intuition (The Committee)

Think of Multi-Head Attention as a **committee of experts** rather than a single person.

* **Head 1** might be an expert in syntax (finding the subject and verb).
* **Head 2** might be an expert in semantics (finding synonyms or related concepts).
* **Head 3** might be looking for distant references (connecting a pronoun to a name 20 words back).

By the end, we concatenate all their "opinions" into one final vector.

---

## Part 2: The Logic of Splitting

How do we actually create these heads? We take our large embedding (e.g., ) and split it into smaller chunks.

If we have **8 heads**, each head works on a vector of size .

1. **Linear Projection:** We multiply our input by learned weights to get  for each head.
2. **Parallel Attention:** Each head runs the "Scaled Dot-Product Attention" from L03.
3. **Concatenation:** We stick the 8 resulting vectors back together to get our original 512 size.
4. **Final Linear:** We run it through one last weight matrix to "mix" the information from all heads.

---

## Part 3: Visualizing Multiple Perspectives

Let's imagine how two different heads might look at the same sentence: **"The cat sat on the mat because it was soft."**

```{code-cell} ipython3
:tags: [remove-input]

tokens = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "soft"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Head 1: Focuses on the object property (it -> mat)
head1 = np.zeros((len(tokens), len(tokens)))
head1[tokens.index("it"), tokens.index("mat")] = 0.9
head1[tokens.index("soft"), tokens.index("mat")] = 0.7

# Head 2: Focuses on the subject action (cat -> sat)
head2 = np.zeros((len(tokens), len(tokens)))
head2[tokens.index("cat"), tokens.index("sat")] = 0.8
head2[tokens.index("sat"), tokens.index("cat")] = 0.5

ax1.imshow(head1, cmap='Purples')
ax1.set_title("Head 1: Reference Expert\n(Connects 'it' to 'mat')")
ax1.set_xticks(range(len(tokens))); ax1.set_xticklabels(tokens, rotation=45)
ax1.set_yticks(range(len(tokens))); ax1.set_yticklabels(tokens)

ax2.imshow(head2, cmap='Greens')
ax2.set_title("Head 2: Grammar Expert\n(Connects Subject to Verb)")
ax2.set_xticks(range(len(tokens))); ax2.set_xticklabels(tokens, rotation=45)
ax2.set_yticks(range(len(tokens))); ax2.set_yticklabels(tokens)

plt.tight_layout()
plt.show()

```

---

## Part 4: Implementation from Scratch

Here is how we translate that logic into a PyTorch module. Note how we use `view` and `transpose` to handle all heads at once—this makes the code much faster.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # We define 4 linear transformations
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projections and split into heads
        # Resulting shape: [batch, heads, seq_len, d_k]
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply Scaled Dot-Product Attention (from L03)
        # We reuse the math from the previous blog!
        attn_output, weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concatenate heads back together
        # Shape: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final linear projection
        return self.W_o(attn_output)

```

---

## Summary

1. **Splitting:** We break the large embedding into smaller sub-vectors for each head.
2. **Specialization:** Each head learns to pay attention to different types of information.
3. **Concatenation:** We combine all those specialized "views" into one rich representation.
4. **Efficiency:** By using matrix manipulation, we process all heads simultaneously.

**Next Up: L05 – Normalization & Residuals.** We've built the engine, but it's currently unstable. We'll learn how the "plumbing" of the Transformer (LayerNorm and Skip Connections) allows us to train models that are hundreds of layers deep without them breaking.

---

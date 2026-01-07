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

# L04 - Multi-Head Attention: The Committee of Experts

*Why one brain is good, but eight brains are better.*

---

In [L03 - Self-Attention](L03_The_Attention_Mechanism.md), we built the "Search Engine" of the Transformer. We learned how the word "it" can look up the word "animal" to resolve ambiguity.

But there is a limitation. A single self-attention layer acts like a single pair of eyes. It can focus on **one** aspect of the sentence at a time.

Consider the sentence:
> **"The chicken didn't cross the road because it was too wide."**

To understand this fully, the model needs to do two things simultaneously:
1.  **Syntactic Analysis:** Link "it" to the subject "road" (because roads are wide).
2.  **Semantic Analysis:** Understand that "wide" is a physical property preventing crossing.

If we only have one attention head, the model has to average these different relationships into a single vector. It muddies the waters.

**Multi-Head Attention** solves this by giving the model multiple "heads" (independent attention mechanisms) that run in parallel.

By the end of this post, you'll understand:
- The intuition of the **"Committee of Experts."**
- Why we project vectors into different **Subspaces**.
- How to implement the tensor reshaping magic (`view` and `transpose`) in PyTorch.

:::{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
:::

---

## Part 1: The Intuition (The Committee)

Think of the embedding dimension ($d_{model} = 512$) as a massive report containing everything we know about a word.

If we ask a single person to read that report and summarize "grammar," "tone," "tense," and "meaning" all at once, they might miss details.

Instead, we hire a **Committee of 8 Experts**:
* **Head 1 (The Linguist):** Only looks for Subject-Verb agreement.
* **Head 2 (The Historian):** Looks for past/present tense consistency.
* **Head 3 (The Translator):** Looks for definitions and synonyms.
* ...



In the Transformer, we don't just copy the input 8 times. We **project** the input into 8 different lower-dimensional spaces. This allows each head to specialize.

---

## Part 2: The Math of Projections

In standard Attention, we had sets of weights $W^Q, W^K, W^V$ that transformed our 512-dimensional input into 512-dimensional Queries, Keys, and Values.

In **Multi-Head Attention**, we slice the model dimension ($d_{model}$) into $h$ heads.
$$d_k = d_{model} / h$$

If $d_{model} = 512$ and we have 8 heads, each head works with vectors of size **64**.

### The Process

1.  **Linear Projection:** For each head $i$, we have unique weight matrices ($W_i^Q, W_i^K, W_i^V$). We multiply the input $X$ by these weights to get specific Questions, Keys, and Values for that head.
2.  **Independent Attention:** Each head runs the standard attention formula independently:
    $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
3.  **Concatenation:** We take the output of all 8 heads and glue them back together side-by-side.
    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
4.  **Final Mix:** We multiply by a final output matrix $W^O$ to blend the insights from the committee back into a unified vector.

---

## Part 3: Visualizing Multiple Perspectives

Let's visualize how two different heads might analyze the same sentence.

**Sentence:** "The cat sat on the mat because it was soft."

* **Head 1** focuses on the physical relationship (connecting "it" to "mat").
* **Head 2** focuses on the actor (connecting "sat" to "cat").

Notice how they highlight completely different parts of the matrix.

:::{code-cell} ipython3
:tags: [remove-input]

tokens = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "soft"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Head 1: Reference Resolution (it -> mat)
# Simulating a head that understands physical properties
head1 = np.zeros((len(tokens), len(tokens)))
head1[tokens.index("it"), tokens.index("mat")] = 0.95
head1[tokens.index("soft"), tokens.index("mat")] = 0.8
# Add some background noise
np.random.seed(42)
head1 += np.random.rand(len(tokens), len(tokens)) * 0.05
# Normalize
head1 = head1 / head1.sum(axis=1, keepdims=True)

# Head 2: Syntax / Subject-Verb (sat -> cat)
# Simulating a head that connects verbs to their subjects
head2 = np.zeros((len(tokens), len(tokens)))
head2[tokens.index("cat"), tokens.index("sat")] = 0.6
head2[tokens.index("sat"), tokens.index("cat")] = 0.9
head2 += np.random.rand(len(tokens), len(tokens)) * 0.05
head2 = head2 / head2.sum(axis=1, keepdims=True)

# Plotting Head 1
im1 = ax1.imshow(head1, cmap='Purples', aspect='auto')
ax1.set_title("Head 1: The 'Meaning' Expert\n(Resolving 'it' -> 'mat')", fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(tokens)))
ax1.set_xticklabels(tokens, rotation=45)
ax1.set_yticks(range(len(tokens)))
ax1.set_yticklabels(tokens)
ax1.grid(False)

# Plotting Head 2
im2 = ax2.imshow(head2, cmap='Greens', aspect='auto')
ax2.set_title("Head 2: The 'Grammar' Expert\n(Linking Subject <-> Verb)", fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(tokens, rotation=45)
ax2.set_yticks(range(len(tokens)))
ax2.set_yticklabels(tokens)
ax2.grid(False)

plt.tight_layout()
plt.show()
:::

---

## Part 4: Implementation in PyTorch

Implementing this efficiently requires some tensor gymnastics. We don't actually run a `for` loop over the 8 heads. That would be too slow.

Instead, we use a single large matrix multiply and then **reshape** (view/transpose) the tensor to create a "heads" dimension.

The shape transformation looks like this:
1.  **Input:** `[Batch, Seq_Len, D_Model]`
2.  **Linear & Reshape:** `[Batch, Seq_Len, Heads, D_Head]`
3.  **Transpose:** `[Batch, Heads, Seq_Len, D_Head]`

By swapping axes 1 and 2, we group the "Heads" dimension with the "Batch" dimension. PyTorch then processes all heads in parallel as if they were just extra items in the batch.

:::{code-cell} ipython3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # We define 4 linear layers: Q, K, V projections and the final Output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Project and Split
        # We transform [Batch, Seq, Model] -> [Batch, Seq, Heads, d_k]
        # Then we transpose to [Batch, Heads, Seq, d_k] for matrix multiplication
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention (re-using logic from L03)
        # Scores shape: [Batch, Heads, Seq, Seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply weights to Values
        # Shape: [Batch, Heads, Seq, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # 3. Concatenate
        # Transpose back: [Batch, Seq, Heads, d_k]
        # Flatten: [Batch, Seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final Projection (The "Mix")
        return self.W_o(attn_output)
:::

:::{note}
**Why `.contiguous()`?**
When we `transpose` a tensor in PyTorch, we aren't actually moving data in memory; we are just changing the "stride" (how the computer steps through memory). `view` requires the data to be contiguous in memory. Calling `.contiguous()` creates a fresh copy of the data with the correct memory layout, preventing runtime errors.
:::

---

## Summary

1.  **Multiple Heads:** We split our embedding into $h$ smaller chunks to allow the model to focus on different linguistic features simultaneously.
2.  **Projection:** We use learned linear layers ($W_Q, W_K, W_V$) to project the input into these specialized subspaces.
3.  **Parallelism:** We use tensor reshaping (`view` and `transpose`) to compute attention for all heads at once, rather than looping through them.

**Next Up: L05 – Layer Norm & Residuals.**
We have built the engine (Attention), but if we stack 100 of these layers on top of each other, the gradients will vanish or explode. In L05, we will add the "plumbing" (Normalization and Skip Connections) that allows Deep Learning to actually get *deep*.

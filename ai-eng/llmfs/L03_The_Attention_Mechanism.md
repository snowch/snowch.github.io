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

# L03 - Self-Attention: Understanding Context

*Building the "engine" of the Transformer from scratch*

---

In our previous post, we turned words into vectors and gave them a "stamp" of their position. But the model still doesn't understand context. 

Consider the word **"bank"** in these two sentences:
1. "I went to the **bank** to deposit money."
2. "I sat on the river **bank**."

The embedding for "bank" is the same in both. **Self-Attention** is the mechanism that allows the model to look at the surrounding words ("money" vs "river") and update the vector for "bank" to reflect its current meaning.

By the end of this post, you'll understand:
- The intuition of **Queries, Keys, and Values**.
- How to compute **Attention Scores**.
- How to implement **Scaled Dot-Product Attention** in PyTorch.

```{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import torch
import torch.nn.functional as F
import matplotlib
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: The Intuition (The Filing Cabinet)

Think of Self-Attention like a filing cabinet system:

1. **Query ():** "What am I looking for?" (The word currently being processed).
2. **Key ():** "What do I contain?" (A label on the outside of every other word's folder).
3. **Value ():** "What information do I actually hold?" (The content inside the folder).

To update the meaning of a word, we compare its **Query** against the **Keys** of every other word. If they match well, we take a large portion of that word's **Value**.

---

## Part 2: The Math of Attention

We compute how much two words should "attend" to each other using the **Dot Product**. A high dot product means the vectors are aligned (relevant).

### The Formula

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

1. ****: Compute scores between all pairs of words.
2. ****: Scale the scores so the gradients don't explode (very important for deep networks!).
3. **Softmax**: Turn scores into probabilities that sum to 1.
4. ****: Multiply the probabilities by the Values to get the weighted average.

---

## Part 3: Visualizing the Attention Map

Let's look at what the "Scores" matrix actually looks like for a sample sentence.

```{code-cell} ipython3
:tags: [remove-input]

# Mock attention matrix for "The animal didn't cross the street because it was too tired"
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
data = np.zeros((len(tokens), len(tokens)))

# Highlighting that "it" attends strongly to "animal"
it_idx = tokens.index("it")
animal_idx = tokens.index("animal")
data[it_idx, animal_idx] = 0.8
data[it_idx, it_idx] = 0.2

# Random noise for other relations
np.random.seed(42)
data += np.random.rand(len(tokens), len(tokens)) * 0.1

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='Blues')
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.title("Self-Attention Map: Where is the model 'looking'?")
plt.colorbar(label='Attention Score')
plt.show()

```

---

## Part 4: Building it in PyTorch

We can implement this entire mechanism in just a few lines of code.

```python
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_k))

    def forward(self, Q, K, V, mask=None):
        # 1. Dot product Q and K
        # Q: [batch, heads, seq_len, d_k], K: [batch, heads, seq_len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 2. Apply Mask (optional - we'll cover this in L06!)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax to get weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. Multiply weights by V
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

```

---

## Summary

1. **Self-Attention** allows tokens to "talk" to each other and share context.
2. **** are linear transformations of our input embeddings.
3. **Dot Product** measures the compatibility between tokens.
4. **Contextual Embeddings:** After this layer, the vector for "bank" is no longer generic—it has been mixed with the vectors of nearby words.

**Next Up: L04 – Multi-Head Attention.** Why have one "search" query when you can have eight? We'll learn how to allow the model to focus on different aspects of a sentence simultaneously (e.g., grammar, logic, and entity relationships).

---

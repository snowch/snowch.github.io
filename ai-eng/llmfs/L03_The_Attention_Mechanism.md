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

# L03 - Self-Attention: The Search Engine of Language

*Building the "brain" of the Transformer: How words talk to each other.*

---

In [L02 - Embeddings](L02_Embeddings_and_Positional_Encoding.md), we turned words into vectors and gave them positions. But there is still a fatal flaw: our model treats every word in isolation.

Consider the word **"Bank"**.
1. "The **bank** of the river." (Nature)
2. "The **bank** approved the loan." (Finance)

In a static embedding layer (like `word2vec`), the vector for "bank" is identical in both sentences. But to understand language, the meaning of "bank" must shift based on its neighbors ("river" vs. "loan").

**Self-Attention** is the mechanism that allows words to look at their neighbors and "update" their meaning based on context. It is the difference between a dictionary definition (static) and reading comprehension (dynamic).

By the end of this post, you'll understand:
- The **Query, Key, Value** analogy (it's just a database lookup!).
- Why we need to **scale** our dot products.
- How to implement the famous `softmax(QK^T / sqrt(d))` equation from scratch.

---

## Part 1: The Intuition (The Filing Cabinet)

The math of Attention can look scary, but the concept is simple. It is a **Soft Database Lookup**.

Imagine every word in the sentence is a folder in a filing cabinet. To facilitate a search, every word produces three vectors:

| Vector | Name | Role | Analogy |
| :--- | :--- | :--- | :--- |
| **Q** | **Query** | What I am looking for? | A sticky note I hold up: *"I am looking for adjectives describing me."* |
| **K** | **Key** | What do I contain? | The label on the folder: *"I am an adjective."* |
| **V** | **Value** | The content | The actual document inside the folder: *"Blue."* |

### The Search Process
1. The word "Sky" holds up its **Query** ("Looking for adjectives").
2. It compares this Query against every other word's **Key**.
3. It finds a high match with the word "Blue."
4. It extracts the **Value** from "Blue" and adds it to its own representation.

Now, the vector for "Sky" is no longer just "Sky"; it is "Sky + a little bit of Blue".

---

## Part 2: The Math of Similarity

How do we mathematically calculate "similarity" between a Query and a Key? We use the **Dot Product**.

If two vectors point in the same direction, their dot product is large (positive). If they point in opposite directions, it is negative. If they are perpendicular (unrelated), it is zero.

### The Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break down the "Magic Formula" step-by-step:

1.  **$QK^T$ (The Scores):** We multiply the Query of the current word by the Keys of *all* words. This creates a score matrix (e.g., How much does "Bank" care about "River"?).
2.  **$\sqrt{d_k}$ (The Scaling):** If our vectors are large (e.g., dimension 512), the dot products can become huge numbers. Large numbers kill gradients in neural networks (causing the "vanishing gradient" problem). We shrink the scores back to a stable range by dividing by the square root of the dimension size.
3.  **Softmax (The Probability):** We convert raw scores into probabilities. If "Bank" scores 90 with "River" and 10 with "The", Softmax ensures these sum to 1.0 (e.g., 0.95 vs 0.05).
4.  **$V$ (The Weighted Sum):** Finally, we multiply these probabilities by the **Values**. We keep 95% of the "River" value and only 5% of the "The" value.

---

## Part 3: Visualizing the Attention Map

When we train these models, we can actually *see* these relationships form. In the heatmap below, brighter colors mean a higher attention score.

Notice the structure:
* **Diagonals:** Words usually attend to themselves heavily.
* **Relationships:** The pronoun "it" attends to "animal", resolving the ambiguity.

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Mock attention matrix for "The animal didn't cross the street because it was too tired"
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
data = np.zeros((len(tokens), len(tokens)))

# Highlighting that "it" attends strongly to "animal"
it_idx = tokens.index("it")
animal_idx = tokens.index("animal")
street_idx = tokens.index("street")

# "it" refers to "animal", not "street"
data[it_idx, animal_idx] = 0.85
data[it_idx, street_idx] = 0.05
data[it_idx, it_idx] = 0.1

# Random noise for other relations to simulate a real learned map
np.random.seed(42)
background_noise = np.random.rand(len(tokens), len(tokens)) * 0.05
data += background_noise

# Normalize rows to sum to 1 (like Softmax)
row_sums = data.sum(axis=1)
data = data / row_sums[:, np.newaxis]

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='Blues')
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.title("Self-Attention Map: Visualizing Context\n(Notice 'it' focusing on 'animal')")
plt.xlabel("Key (Source Information)")
plt.ylabel("Query (Current Word Focus)")
plt.colorbar(label='Attention Probability')
plt.tight_layout()
plt.show()

```

---

## Part 4: Implementation in PyTorch

We can implement this entire mechanism—the heart of the Transformer—in fewer than 20 lines of code.

Note the use of `transpose` to flip the K matrix for the dot product, and `masked_fill` which we will use in [L06_The_Causal_Mask.md) to prevent cheating.

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        # 1. Calculate the Dot Product (Scores)
        # q: [batch, heads, seq, d_k]
        # k.transpose: [batch, heads, d_k, seq]
        # scores shape: [batch, heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2. Apply Mask (Optional - vital for GPT!)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax to get probabilities (0.0 to 1.0)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. Multiply by Values to get the weighted context
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

```

---

## Summary

1. **Context Matters:** Standard embeddings are static. Attention makes them dynamic.
2. **Q, K, V:** We project our input into "Queries" (Searches), "Keys" (Labels), and "Values" (Content).
3. **The Mechanism:** We measure similarity between Queries and Keys to create a weighted average of Values.

**Next Up: L04 – Multi-Head Attention.** One attention head is good, but it can only focus on one relationship at a time (e.g., "it"  "animal"). What if we also need to know that "animal" is the *subject* of the sentence? We need more heads!

---

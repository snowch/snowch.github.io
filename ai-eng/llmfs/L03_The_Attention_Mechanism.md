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

In a static embedding layer, the vector for "bank" is identical in both sentences. But to understand language, the meaning of "bank" must shift based on its neighbors ("river" vs. "loan").

**Self-Attention** is the mechanism that allows words to look at their neighbors and "update" their meaning based on context.

By the end of this post, you'll understand:
- The **Query, Key, Value** analogy (it's just a database lookup!).
- Why we need to **scale** our dot products (fixing the "magnitude" bug).
- How to implement the famous attention equation from scratch.



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

In transformers, to determine which words are relevant to each other, we use the **Dot Product**. This operation measures alignment: if two vectors point in the same direction, the result is large and positive. If they point in opposite directions, it is negative.

### Visualizing the "Magnitude Problem"

However, there is a catch. The dot product captures both alignment *and* magnitude. Before we look at the formula, let's visualize why this is dangerous for neural networks.

In the plot below, we compare a **Query (Blue)** against three different **Keys**.
* **K1 (Short):** Perfectly aligned with Q, but small.
* **K2 (Long):** Perfectly aligned with Q, but large.
* **K3 (Misaligned):** Pointing in a different direction.

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def plot_refined_dot_product_v2():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define vectors
    origin = np.array([0, 0])
    q = np.array([3, 1])          # Query Vector

    # Key Vectors
    k1 = np.array([1.5, 0.5])     # Short (0.5x)
    k_exact = np.array([3, 1])    # Exact Match (1.0x)
    k2 = np.array([4.5, 1.5])     # Long (1.5x)
    k3 = np.array([1, 4])         # Misaligned

    # 1. Draw the "Direction of Alignment"
    x_vals = np.linspace(-1, 6, 10)
    y_vals = x_vals * (q[1]/q[0])
    ax.plot(x_vals, y_vals, 'k--', alpha=0.4, label='Alignment Line', linewidth=1.5, zorder=1)

    # 2. Plot Vectors
    # Query (Blue)
    ax.quiver(*origin, *q, color='#1f77b4', scale=1, scale_units='xy', angles='xy', label='Query (Q)', width=0.015, zorder=3)

    # Keys
    ax.quiver(*origin, *k2, color='#98df8a', scale=1, scale_units='xy', angles='xy', label='Key 2 (Long)', width=0.012, zorder=2)
    ax.quiver(*origin, *k1, color='#2ca02c', scale=1, scale_units='xy', angles='xy', label='Key 1 (Short)', width=0.012, zorder=4)
    ax.quiver(*origin, *k3, color='#d62728', scale=1, scale_units='xy', angles='xy', label='Key 3 (Misaligned)', width=0.012, zorder=2)

    # 3. Calculate Dot Products
    dp1 = np.dot(q, k1)
    dp_exact = np.dot(q, k_exact)
    dp2 = np.dot(q, k2)
    dp3 = np.dot(q, k3)

    # 4. Add Annotations

    # Query Label
    ax.text(q[0]-1.6, q[1] + 0.2, ' Query (Q)', color='#1f77b4', fontweight='bold', fontsize=12)

    # EXACT MATCH Label
    ax.annotate(f'Exact Match (K=Q)\nDot: {dp_exact:.1f}', 
                xy=(q[0], q[1]), 
                xytext=(q[0] + 0.4, q[1] - 1.0),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=2),
                color='#ff7f0e', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7f0e", alpha=0.9))

    # K1 Label
    ax.annotate(f'K1 (Short)\nDot: {dp1:.1f}', 
                xy=(k1[0], k1[1]), 
                xytext=(k1[0] + 0.3, k1[1] - 1.0),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
                color='#2ca02c', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.8))
    
    # K2 Label
    ax.text(k2[0], k2[1] - 0.6, f'K2 (Long)\nDot: {dp2:.1f}', color='#5cb85c', fontweight='bold')
    
    # K3 Label
    ax.text(k3[0] + 0.5, k3[1] + 0.3, f'K3 (Misaligned)\nDot: {dp3:.1f}', color='#d62728', fontweight='bold')

    # Styling
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    ax.set_title("Dot Product Scores: The Effect of Length")
    
    plt.tight_layout()
    plt.show()

plot_refined_dot_product_v2()


```

**Why this is a problem:**
Look at the score for **K3 (Misaligned)**. It scored **7.0**.
Now look at **K1 (Short)**. It scored **5.0**.

The "Misaligned" vector beat the "Perfectly Aligned" vector simply because it was longer. If we don't fix this, our model will prioritize "loud" signals (large numbers) over "correct" signals (aligned meaning).

We fix this by **Scaling**: we divide the result by the square root of the dimension ($\sqrt{d_k}$). This normalizes the scores so the model focuses on alignment, not magnitude.

```{important}
**Why $\sqrt{d_k}$ Specifically?**

The scaling factor isn't arbitrary. As dimension $d_k$ increases, dot products between random vectors grow proportionally to $d_k$. By dividing by $\sqrt{d_k}$, we keep the variance of the scores roughly constant regardless of dimensionality.

More importantly, without this scaling, large dot products push the softmax function into regions where gradients are tiny (the "saturation" problem). When scores like 50 or 100 go into softmax, it becomes almost deterministic—one weight approaches 1.0, all others approach 0.0. This prevents the softmax from saturating, keeping gradients flowing during training and allowing the model to attend to multiple positions when needed.
```

### The Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break down the equation step-by-step:

1.  **The Scores ($QK^T$):** We multiply the Query of the current word by the Keys of *all* words.
2.  **The Scaling ($\sqrt{d_k}$):** We shrink the scores to prevent exploding values.
3.  **Softmax (The Probability):** We convert scores into probabilities that sum to 1.0.
4.  **The Weighted Sum ($V$):** We multiply the probabilities by the **Values** to get the final context vector.

### Example Walkthrough: Crunching the Numbers

Let's trace the math using the **Exact Match** and **Key 3** vectors from the plot above.
* **Query (Q):** `[3, 1]`
* **Key (Exact Match):** `[3, 1]`
* **Key 3 (Misaligned):** `[1, 4]`

**Step 1: The Dot Product ($QK^T$)** - Computing Raw Scores
* Score (Exact Match): $(3 \times 3) + (1 \times 1) = 10$
* Score (Misaligned): $(3 \times 1) + (1 \times 4) = 7$

```{note}
**Scaling to Real Models:** We used 2D vectors for easy visualization, but in practice, attention operates in **much higher dimensions** (typically 512D or more). The good news? The math is identical—we're still just computing dot products to measure alignment. The key difference is that in 512 dimensions, we're comparing 512-element vectors. The dot product still measures alignment, but now across hundreds of dimensions simultaneously, allowing the model to capture much richer relationships between words.
```

**Step 2: Scaling ($\sqrt{d_k}$)**
We divide by $\sqrt{2} \approx 1.41$.
* Scaled Score (Exact): $10 / 1.41 \approx 7.09$
* Scaled Score (Misaligned): $7 / 1.41 \approx 4.96$

**Step 3: Softmax**
We exponentiate and normalize to get percentages using the Softmax formula:

$$P(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$

* **Probability (Exact Match):**
    $$P_1 = \frac{e^{7.09}}{e^{7.09} + e^{4.96}} \approx \frac{1199}{1199 + 142} \approx \frac{1199}{1341} \approx \mathbf{89\%}$$

* **Probability (Misaligned):**
    $$P_2 = \frac{e^{4.96}}{e^{7.09} + e^{4.96}} \approx \frac{142}{1341} \approx \mathbf{11\%}$$

Notice how the mechanism successfully identified the aligned vector as the important one, giving it 89% of the attention!

```{important}
**Terminology: Scores vs. Weights**

It's crucial to understand the distinction between these two terms that are often confused:

- **Attention Scores** (also called "logits" or "raw scores"): The values **before** the softmax operation. These are the results of $\frac{QK^T}{\sqrt{d_k}}$ and can be any real number (positive, negative, large, small). In our example: 7.09 and 4.96.

- **Attention Weights** (also called "attention probabilities"): The values **after** the softmax operation. These always sum to 1.0 and represent the percentage of "attention" each position receives. In our example: 89% and 11%.

When debugging attention mechanisms or reading research papers, knowing which one is being discussed is critical. Scores are used for computing gradients, while weights are used for the final weighted sum with the Values.
```

---

## Part 3: Visualizing the Attention Map

When we train these models, we can see these relationships form. In the heatmap below, brighter colors mean a higher attention score.

Notice that the word "it" attends strongly to "animal", resolving the ambiguity.

```{code-cell} ipython3
:tags: [remove-input]

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

We can implement this entire mechanism in fewer than 20 lines of code.

Note the use of `masked_fill`, which we will use in [L06 - The Causal Mask](https://www.google.com/search?q=L06_The_Causal_Mask.md) to prevent the model from "cheating" by looking at future words.

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
        # k.transpose: [batch, heads, d_k, seq] -> We flip the last two dimensions
        # scores shape: [batch, heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2. Apply Mask (Optional - vital for GPT!)
        if mask is not None:
            # We use a very large negative number so Softmax turns it to zero
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
3. **Scaling:** We divide by  to stop the gradients from vanishing when vectors get large.

**Next Up: L04 – Multi-Head Attention.** One attention head is good, but it can only focus on one relationship at a time (e.g., "it"  "animal"). What if we also need to know that "animal" is the *subject* of the sentence? We need more heads!

---

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

In [L02 - Embeddings](L02_Embeddings_and_Positional_Encoding.md), we turned words into vectors and gave them positions. But there is still a fatal flaw: **each word gets the same vector every time**, regardless of context.

Consider the word **"Bank"**.
1. "The **bank** of the river." (Nature)
2. "The **bank** approved the loan." (Finance)

In a static embedding layer (a simple lookup table), the vector for "bank" is **identical** in both sentences. The embedding doesn't know about "river" or "loan"—it just maps token ID → vector. But to understand language, the meaning of "bank" must shift based on its neighbors.

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

# 2D embedding space visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Nature/Geography cluster (spread out more)
nature_words = {
    'river': [-2.5, 1.2],
    'water': [-2.2, 1.8],
    'shore': [-2.8, 0.8],
    'stream': [-2.0, 1.4]
}

# Finance cluster (spread out more)
finance_words = {
    'loan': [2.5, 1.2],
    'money': [2.2, 1.8],
    'account': [2.8, 0.8],
    'deposit': [2.0, 1.4]
}

# The problem: "bank" is stuck in the middle
bank_pos = [0, 1.2]

# Plot nature cluster (green) - labels below points
for word, pos in nature_words.items():
    ax.scatter(pos[0], pos[1], c='green', s=400, alpha=0.5, edgecolors='darkgreen', linewidth=2.5, zorder=2)
    ax.text(pos[0], pos[1]-0.35, word, ha='center', va='top', fontsize=12, fontweight='bold', color='darkgreen')

# Plot finance cluster (blue) - labels below points
for word, pos in finance_words.items():
    ax.scatter(pos[0], pos[1], c='blue', s=400, alpha=0.5, edgecolors='darkblue', linewidth=2.5, zorder=2)
    ax.text(pos[0], pos[1]-0.35, word, ha='center', va='top', fontsize=12, fontweight='bold', color='darkblue')

# Plot "bank" in the middle (red, larger) - label below
ax.scatter(bank_pos[0], bank_pos[1], c='red', s=600, alpha=0.7, edgecolors='darkred', linewidth=3, zorder=3)
ax.text(bank_pos[0], bank_pos[1]-0.45, 'bank', ha='center', va='top', fontsize=14, fontweight='bold', color='darkred')

# Add clearer arrows showing the problem
ax.annotate('', xy=[-1.8, 1.2], xytext=[bank_pos[0]-0.1, bank_pos[1]],
            arrowprops=dict(arrowstyle='->', color='red', lw=3, linestyle='dashed', alpha=0.6))
ax.annotate('', xy=[1.8, 1.2], xytext=[bank_pos[0]+0.1, bank_pos[1]],
            arrowprops=dict(arrowstyle='->', color='red', lw=3, linestyle='dashed', alpha=0.6))

# Add context labels with better positioning
ax.text(-2.5, 2.6, 'Nature/Geography\nContext', ha='center', fontsize=13, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5, edgecolor='darkgreen', linewidth=2))
ax.text(2.5, 2.6, 'Finance/Banking\nContext', ha='center', fontsize=13, color='darkblue', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5, edgecolor='darkblue', linewidth=2))

# Add the problem statement at bottom
ax.text(0, -0.7, '⚠️  Static Embedding Problem', ha='center', fontsize=14, color='red', fontweight='bold')
ax.text(0, -1.1, '"bank" is frozen at ONE location—can\'t shift meaning based on context',
        ha='center', fontsize=11, style='italic', color='darkred')

# Add labels for what the arrows mean
ax.text(-0.9, 0.6, 'Should move\nhere for "river"?', ha='center', fontsize=9, color='red', style='italic')
ax.text(0.9, 0.6, 'Should move\nhere for "loan"?', ha='center', fontsize=9, color='red', style='italic')

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-1.5, 3.2)
ax.set_xlabel('Embedding Dimension 1', fontsize=12, fontweight='bold')
ax.set_ylabel('Embedding Dimension 2', fontsize=12, fontweight='bold')
ax.set_title('Static Embeddings: The "Bank" Problem\n(Same vector whether next to "river" or "loan")',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.2, linestyle='--')
ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)

plt.tight_layout()
plt.show()
```

**The Problem:** In static embeddings, "bank" is frozen at one location in the embedding space. It can't move toward "river" in one sentence and toward "loan" in another. The vector is context-independent.

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

### The Key Advantage: Everything Happens at Once

Remember from [L02](L02_Embeddings_and_Positional_Encoding.md): *"The attention mechanism is **parallel**. It looks at every word in a sentence at the exact same time."*

This is the **breakthrough** that makes Transformers faster than older architectures like RNNs (Recurrent Neural Networks).

**How RNNs Process a Sentence (Sequential):**
```
Input: "The quick brown fox"

Step 1: Process "The"       → Update hidden state
Step 2: Process "quick"     → Update hidden state (using step 1)
Step 3: Process "brown"     → Update hidden state (using step 2)
Step 4: Process "fox"       → Update hidden state (using step 3)

Total: 4 sequential steps (can't parallelize)
```

**How Attention Processes the Same Sentence (Parallel):**
```
Input: "The quick brown fox"

Single Step: ALL words simultaneously:
  - "The"   compares against ["The", "quick", "brown", "fox"]
  - "quick" compares against ["The", "quick", "brown", "fox"]
  - "brown" compares against ["The", "quick", "brown", "fox"]
  - "fox"   compares against ["The", "quick", "brown", "fox"]

Total: 1 parallel step (all comparisons happen at once)
```

**Why This Works:**

The attention mechanism achieves parallelism through **matrix multiplication**. When we compute $QK^T$ (which you'll see shortly), we're not looping through words one-by-one. Instead:

1. **Every word** generates its Query, Key, and Value simultaneously (one matrix operation)
2. **Every Query** compares against **every Key** simultaneously (another matrix operation)
3. **Every word** gets its context-aware representation simultaneously (final matrix operation)

Modern GPUs are **optimized for matrix operations**, so computing attention for 100 words in parallel is barely slower than computing it for 10 words. This is why Transformers can handle such long contexts efficiently.

```{note}
**The Trade-off:** Attention is $O(n^2)$ in sequence length (every word looks at every other word), while RNNs are $O(n)$ (each word processed once). But because attention parallelizes perfectly on modern hardware while RNNs must run sequentially, attention is **much faster** in practice for typical sequence lengths (up to thousands of tokens).
```

Now let's see the math that makes this parallelism possible.

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

In trained models, attention patterns emerge that capture semantic relationships. The heatmap below shows a **simplified example** to illustrate what we might expect: brighter colors represent higher attention weights (post-softmax probabilities).

```{code-cell} ipython3
:tags: [remove-input]

# Simplified example: "The animal didn't cross the street because it was too tired"
# In a real model, "it" would ideally learn to attend to "animal" (not "street")
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
data = np.zeros((len(tokens), len(tokens)))

it_idx = tokens.index("it")
animal_idx = tokens.index("animal")
street_idx = tokens.index("street")

# Simulating ideal attention: "it" refers to "animal"
data[it_idx, animal_idx] = 0.7   # Strong attention to "animal"
data[it_idx, street_idx] = 0.05  # Weak attention to "street"
data[it_idx, it_idx] = 0.15       # Some self-attention
data[it_idx, 6] = 0.1            # Some attention to "because" (context)

# Simple diagonal pattern for other words (attending to themselves)
for i in range(len(tokens)):
    if i != it_idx:
        data[i, i] = 0.8
        # Distribute remaining probability uniformly
        remaining = 0.2 / (len(tokens) - 1)
        for j in range(len(tokens)):
            if i != j:
                data[i, j] = remaining

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='Blues', vmin=0, vmax=1)
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.yticks(range(len(tokens)), tokens)
plt.title("Self-Attention Pattern (Simplified Example)\nRow 'it' shows ~70% attention to 'animal'")
plt.xlabel("Key (attending TO)")
plt.ylabel("Query (attending FROM)")
plt.colorbar(label='Attention Weight')

# Add text annotations for the "it" row to make it clearer
for j, token in enumerate(tokens):
    weight = data[it_idx, j]
    if weight > 0.1:  # Only annotate significant weights
        plt.text(j, it_idx, f'{weight:.0%}',
                ha='center', va='center', color='red', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()


```

```{note}
**This is a pedagogical example, not real trained attention!**

In practice, attention patterns in trained transformers are:
- **Multi-headed**: Each attention head learns different patterns (we'll cover this in L05)
- **Layer-dependent**: Early layers may focus on syntax, later layers on semantics
- **Task-dependent**: What the model attends to depends on the training objective

The pattern above shows the *ideal* behavior we'd hope to see—"it" resolving to "animal" rather than "street". Real attention maps are messier and more nuanced!
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

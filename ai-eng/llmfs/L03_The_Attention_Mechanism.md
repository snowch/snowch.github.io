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

In a static embedding layer (a simple lookup table), the vector for "bank" is **identical** in both sentences. The model sees the same token ID → same vector, regardless of whether "bank" appears near "river" or "loan".

```{code-cell} ipython3
:tags: [remove-input]

import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import numpy as np
import matplotlib.pyplot as plt

# Create side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Define semantic clusters (same for both subplots)
nature_words = {'river': [-2.5, 1.5], 'water': [-2.2, 2.0], 'shore': [-2.8, 1.0]}
finance_words = {'loan': [2.5, 1.5], 'money': [2.2, 2.0], 'account': [2.8, 1.0]}

# Static embedding position for "bank" (same in both contexts!)
bank_static_pos = [0, 1.5]

# Where "bank" SHOULD move with attention
bank_nature_pos = [-1.5, 1.4]   # Shifted toward nature cluster
bank_finance_pos = [1.5, 1.4]   # Shifted toward finance cluster

# --- LEFT PLOT: "The bank of the river" ---
ax1.set_title('Sentence: "The bank of the river"', fontsize=13, fontweight='bold', pad=10)

# Plot nature cluster (highlighted with thicker border)
for word, pos in nature_words.items():
    ax1.scatter(pos[0], pos[1], c='green', s=100, alpha=0.6, edgecolors='darkgreen', linewidth=3, zorder=2)
    ax1.text(pos[0], pos[1]-0.1, word, ha='center', va='top', fontsize=11, fontweight='bold', color='darkgreen')

# Plot finance cluster (faded)
for word, pos in finance_words.items():
    ax1.scatter(pos[0], pos[1], c='lightgray', s=100, alpha=0.3, edgecolors='gray', linewidth=1.5, zorder=1)
    ax1.text(pos[0], pos[1]-0.1, word, ha='center', va='top', fontsize=11, color='gray')

# Static "bank" (red, same position as right plot)
ax1.scatter(bank_static_pos[0], bank_static_pos[1], c='red', s=600, alpha=0.8,
           edgecolors='darkred', linewidth=3, zorder=3, marker='s')
ax1.text(bank_static_pos[0], bank_static_pos[1]-0.5, 'bank\n(static)', ha='center', va='top',
        fontsize=12, fontweight='bold', color='darkred')

# Where it SHOULD be with attention (green ghost)
ax1.scatter(bank_nature_pos[0], bank_nature_pos[1], c='lightgreen', s=600, alpha=0.5,
           edgecolors='green', linewidth=3, linestyle='--', zorder=2, marker='s')
ax1.text(bank_nature_pos[0], bank_nature_pos[1]-0.5, 'bank\n(with attention)', ha='center', va='top',
        fontsize=10, color='green', style='italic')

# Arrow showing desired shift
ax1.annotate('', xy=bank_nature_pos, xytext=bank_static_pos,
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5, linestyle='dashed'))

ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-0.5, 3)
ax1.set_xlabel('Embedding Dimension 1', fontsize=11)
ax1.set_ylabel('Embedding Dimension 2', fontsize=11)
ax1.grid(True, alpha=0.2)
ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

# --- RIGHT PLOT: "The bank approved the loan" ---
ax2.set_title('Sentence: "The bank approved the loan"', fontsize=13, fontweight='bold', pad=10)

# Plot finance cluster (highlighted)
for word, pos in finance_words.items():
    ax2.scatter(pos[0], pos[1], c='blue', s=100, alpha=0.6, edgecolors='darkblue', linewidth=3, zorder=2)
    ax2.text(pos[0], pos[1]-0.1, word, ha='center', va='top', fontsize=11, fontweight='bold', color='darkblue')

# Plot nature cluster (faded)
for word, pos in nature_words.items():
    ax2.scatter(pos[0], pos[1], c='lightgray', s=100, alpha=0.3, edgecolors='gray', linewidth=1.5, zorder=1)
    ax2.text(pos[0], pos[1]-0.1, word, ha='center', va='top', fontsize=11, color='gray')

# Static "bank" (red, SAME position as left plot!)
ax2.scatter(bank_static_pos[0], bank_static_pos[1], c='red', s=600, alpha=0.8,
           edgecolors='darkred', linewidth=3, zorder=3, marker='s')
ax2.text(bank_static_pos[0], bank_static_pos[1]-0.5, 'bank\n(static)', ha='center', va='top',
        fontsize=12, fontweight='bold', color='darkred')

# Where it SHOULD be with attention (blue ghost)
ax2.scatter(bank_finance_pos[0], bank_finance_pos[1], c='lightblue', s=600, alpha=0.5,
           edgecolors='blue', linewidth=3, linestyle='--', zorder=2, marker='s')
ax2.text(bank_finance_pos[0], bank_finance_pos[1]-0.5, 'bank\n(with attention)', ha='center', va='top',
        fontsize=10, color='blue', style='italic')

# Arrow showing desired shift
ax2.annotate('', xy=bank_finance_pos, xytext=bank_static_pos,
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5, linestyle='dashed'))

ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-0.5, 3)
ax2.set_xlabel('Embedding Dimension 1', fontsize=11)
ax2.set_ylabel('Embedding Dimension 2', fontsize=11)
ax2.grid(True, alpha=0.2)
ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

# Overall title
fig.suptitle('⚠️  The Static Embedding Problem: Same Word, Different Context, SAME Vector',
            fontsize=14, fontweight='bold', color='darkred', y=0.98)

plt.tight_layout()
plt.show()
```

**The Problem Visualized:**

Notice the red square labeled "bank (static)" is at the **EXACT SAME POSITION** in both plots. Whether "bank" appears in a sentence about rivers or loans, the static embedding lookup table returns the identical vector.

```{note}
**Why is "bank" positioned in the middle?**

Static embeddings like Word2Vec DO learn from context during training! Because "bank" co-occurs with both nature words ("river", "shore") and finance words ("loan", "money"), the training process positions it between both clusters—it's somewhat similar to BOTH.

This is actually a strength: the embedding captures that "bank" is polysemous (multiple meanings). The problem isn't the training—it's **inference time**. Once trained, the embedding is frozen. The model can't disambiguate which meaning is active in the current sentence because it always returns the same compromise vector.
```

**What We Need (shown by dashed arrows):**
- **Left:** In "The **bank** of the river", we want "bank" to shift toward the nature cluster (green)
- **Right:** In "The **bank** approved the loan", we want "bank" to shift toward the finance cluster (blue)

**Self-Attention** is the mechanism that enables this context-dependent shift—it allows "bank" to dynamically adjust its representation based on surrounding words.

By the end of this post, you'll understand:
- The **Query, Key, Value** analogy (it's just a database lookup!).
- Why we need to **scale** our dot products (fixing the "magnitude" bug).
- How to implement the famous attention equation from scratch.



---

## Part 1: The Intuition (The Filing Cabinet)

**Solving the "Bank" Problem:**

We've seen that static embeddings give "bank" the same vector whether it appears with "river" or "loan". How does attention fix this? It allows each word to **look at its neighbors** and adjust its meaning based on what it finds.

The math of Attention can look scary, but the concept is simple. It is a **Soft Database Lookup**.

Imagine every word in the sentence is a folder in a filing cabinet. To facilitate a search, every word produces three vectors:

| Vector | Name | Role | Analogy |
| :--- | :--- | :--- | :--- |
| **Q** | **Query** | What I am looking for? | A sticky note I hold up: *"I am looking for adjectives describing me."* |
| **K** | **Key** | What do I contain? | The label on the folder: *"I am an adjective."* |
| **V** | **Value** | The content | The actual document inside the folder: *"Blue."* |

### How Attention Works: The Search Process

Let's trace how "bank" would use attention to shift its meaning in "The **bank** of the river":

1. **"bank"** generates its **Query**: "What context am I in?"
2. It compares this Query against every other word's **Key**:
   - **"river"** Key: "I'm a geographic feature" → High match!
   - **"of"** Key: "I'm a preposition" → Low match
   - **"the"** Key: "I'm an article" → Low match
3. "bank" finds the highest match with **"river"**
4. It extracts the **Value** from "river" (its semantic meaning) and adds it to its own representation

Now, the vector for "bank" is no longer just the static embedding; it is "bank + a lot of 'river' + a little bit of 'the' and 'of'". The representation has shifted toward the nature/geography meaning!

```{note}
**Why "Query, Key, Value"?**

This terminology comes from databases:
- **Query**: What you're searching for (like "SELECT WHERE...")
- **Key**: The index you search against (like database keys)
- **Value**: The actual data you retrieve

In attention, every word simultaneously plays all three roles for different parts of the computation.
```

### The Key Advantage: Everything Happens at Once

Remember from [L02](L02_Embeddings_and_Positional_Encoding.md): *"The attention mechanism is **parallel**. It looks at every word in a sentence at the exact same time."*

This is the **breakthrough** that makes Transformers faster AND better at understanding language than older architectures like RNNs (Recurrent Neural Networks).

**How RNNs Process a Sentence (Sequential):**

RNNs maintain a "hidden state"—a vector that **accumulates information** from all previous words. At each step, the hidden state combines the current word with everything seen so far.

Let's trace a concrete example: **pronoun resolution**. When the model processes "it", how does it figure out that "it" refers to "bank"? We'll follow how information flows through the hidden states.

```
Input: "The bank approved the loan because it was well-capitalized"
        ↑    ↑                              ↑
      word 1  word 2                       word 7

Step 1: "The"      → hidden_state_1 = f(embedding("The"))
                      ↓ Contains: [info about "The"]

Step 2: "bank"     → hidden_state_2 = f(embedding("bank"), hidden_state_1)
                      ↓ Contains: [info about "The", "bank"] compressed into one vector

Step 3: "approved" → hidden_state_3 = f(embedding("approved"), hidden_state_2)
                      ↓ Contains: [info about "The", "bank", "approved"] compressed

...

Step 7: "it"       → hidden_state_7 = f(embedding("it"), hidden_state_6)
                      ↓ Contains: [ALL 7 words] compressed into a fixed-size vector

Problem: To understand what "it" refers to, information about "bank" (word 2)
must pass through a chain of compressions before reaching "it" (word 7):

  hidden_state_2 (contains "bank" info)
    ↓ compressed with "approved"
  hidden_state_3
    ↓ compressed with "the"
  hidden_state_4
    ↓ compressed with "loan"
  hidden_state_5
    ↓ compressed with "because"
  hidden_state_6 (used by "it" at step 7)

Information about "bank" has been compressed through 4 intermediate mixing steps.
The more steps between "bank" and "it", the more diluted the information becomes.
This is the "vanishing gradient" problem.

Total: 7 sequential steps (MUST run one-by-one)
```

**The Hidden State Bottleneck:**

The hidden state is **accumulative** (it tries to remember everything), but it achieves this through **compression** into a fixed-size vector (typically 512 or 1024 dimensions).

Think of it like this: After reading "The bank approved the loan because it", the RNN must squeeze all understanding of these 7 words—their meanings, relationships, syntactic roles—into a single vector of fixed size. Then it must use this compressed summary to process "was" and "well-capitalized".

It's like trying to fit an entire Wikipedia article into a tweet, then using only that tweet to write the next paragraph. Some information inevitably gets lost or diluted.

**How Attention Processes the Same Sentence (Parallel):**

Now let's see how attention solves **the same pronoun resolution task**: when "it" needs to figure out what it refers to.

Instead of passing information through a chain of hidden states, attention allows **direct connections** between any two words.

```
Input: "The bank approved the loan because it was well-capitalized"

Single Step: ALL words computed simultaneously via matrix operations:

"it" (word 7) compares its Query directly against ALL Keys:
  Q("it") · K("The")      = 0.05  → Low attention weight
  Q("it") · K("bank")     = 0.82  → HIGH attention weight! ✓
  Q("it") · K("approved") = 0.08  → Low attention weight
  Q("it") · K("the")      = 0.02  → Low attention weight
  Q("it") · K("loan")     = 0.15  → Medium attention weight
  ...

Result: "it" can look DIRECTLY at "bank" (word 2) without any intermediate steps.
No information loss. No compression. Direct access.

Every other word does the same computation simultaneously:
  - "The" attends to all words
  - "bank" attends to all words
  - "approved" attends to all words
  ... (all computed in parallel)

Total: 1 parallel step (all comparisons at once, then weighted sum)
```

**Why Direct Access Matters:**

1. **No information loss**: "it" doesn't need to hope that information about "bank" survived 4 intermediate compressions (hidden states 3→4→5→6)
2. **Long-range dependencies**: Works just as well for word 100 referring to word 1 as for adjacent words
3. **Symmetry**: "bank" can attend to "loan" just as easily as "loan" attends to "bank"
4. **Multiple relationships**: Each word can attend strongly to multiple other words simultaneously (through the weighted sum)

**Concrete Example - Pronoun Resolution:**

Consider: "The **artist** gave the **musician** a **score** because **she** loved **her** composition."

- RNN: By the time we reach "she", information about "artist" and "musician" has been mixed together in the hidden state. Hard to tell who "she" refers to.
- Attention: "she" directly queries both "artist" and "musician", finds stronger semantic match with "musician" (or "artist" depending on learned weights), resolves reference clearly.

**Why This Works:**

The attention mechanism achieves parallelism through **matrix multiplication**. When we compute $QK^T$ (which you'll see shortly), we're not looping through words one-by-one. Instead:

1. **Every word** generates its Query, Key, and Value simultaneously (one matrix operation)
2. **Every Query** compares against **every Key** simultaneously (another matrix operation)
3. **Every word** gets its context-aware representation simultaneously (final matrix operation)

Modern GPUs are **optimized for matrix operations**, so computing attention for 100 words in parallel is barely slower than computing it for 10 words. This is why Transformers can handle such long contexts efficiently.

```{important}
**The Trade-off: Speed vs. Memory**

Attention is $O(n^2)$ in sequence length (every word looks at every other word), while RNNs are $O(n)$ (each word processed once). But because attention parallelizes perfectly on modern hardware while RNNs must run sequentially, attention is **much faster** in practice for typical sequence lengths. Plus, the direct access enables better language understanding, especially for long-range dependencies.

**The Memory Bottleneck:**

However, $O(n^2)$ complexity affects both computation AND **memory**. The attention matrix (scores for all word pairs) grows quadratically:

- 1K tokens → 1M attention scores (1,000 × 1,000)
- 2K tokens → 4M attention scores (2,000 × 2,000)
- 4K tokens → 16M attention scores (4,000 × 4,000)
- 8K tokens → 64M attention scores (8,000 × 8,000)

This is why **context length** is a key specification in LLMs:
- GPT-2 (2019): ~1K tokens
- GPT-3 (2020): ~2K-4K tokens
- GPT-3.5/GPT-4 (2022-23): 4K-32K tokens
- Claude 2 (2023): 100K tokens
- GPT-4 Turbo (2023): 128K tokens
- Gemini 1.5 (2024): 1M+ tokens

**Beyond this lesson:** Advanced techniques like Flash Attention ([L16 - Attention Optimizations](../llmfs-scaling/L16_Attention_Optimizations.md)) reduce memory usage without changing the math, and techniques like RoPE and ALiBi ([L18 - Long Context Handling](../llmfs-scaling/L18_Long_Context_Handling.md)) enable models to handle sequences beyond their original training length. But for now, understand that memory—not speed—is what limits context length in transformers.
```

Now let's see the math that makes this parallelism possible.

---

## Part 2: The Math of Similarity

In Part 1, we said that each word's **Query** compares against other words' **Keys** to find matches. But HOW do we "compare" two vectors? How do we measure if a Query is similar to a Key?

The answer: the **Dot Product**.

The dot product is a mathematical operation that measures alignment between two vectors. If two vectors point in the same direction, the result is large and positive. If they point in opposite directions, it is negative. If they're perpendicular, the result is zero.

**This is how attention computes "relevance"**: Q("it") · K("bank") gives us a score measuring how much "it" should attend to "bank".

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

```{mermaid}
graph TD
    %% --- Styling ---
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:black;
    classDef operation fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef intermediate fill:#e8f5e9,stroke:#388e3c,stroke-width:1px,stroke-dasharray: 5 5,color:black;
    classDef output fill:#d1c4e9,stroke:#512da8,stroke-width:2px,color:black;

    %% --- The Inputs ---
    subgraph INPUTS [The Data]
        %% 1. Define Notes inside so lines don't cross the Title
        noteQ[Note: What I'm looking for] -.-> Q(Q: Queries):::input
        noteK[Note: Folder Labels] -.-> K(K: Keys):::input
        noteV[Note: The actual content] -.-> V(V: Values):::input
    end

    %% --- Step 1: Scores ---
    K -- Transpose --> KT(Kᵀ):::operation
    Q --> MatMul1(MatMul: QKᵀ):::operation
    KT --> MatMul1

    MatMul1 -- "Step 1: Raw Scores" --> RawScores[Raw Scores Matrix]:::intermediate

    %% --- Step 2: Scaling ---
    RawScores --> Scale(Scale: Divide by √d_k):::operation
    Scale -- "Step 2: Attention Scores (logits)" --> ScaledScores[Attention Scores / Logits]:::intermediate

    %% --- Step 3: Softmax ---
    ScaledScores --> Softmax(Softmax Probability):::operation
    Softmax -- "Step 3: Attention Weights (probabilities)" --> AttnWeights[Attention Weights %]:::intermediate

    %% --- Step 4: Weighted Sum ---
    AttnWeights --> MatMul2(MatMul: Weights x V):::operation
    V --> MatMul2

    %% --- Final Output ---
    MatMul2 -- "Step 4: The Weighted Sum" --> FinalOut(Final Context Vectors):::output

    %% --- Analogy Labels (Optional Helpers) ---
    style Q fill:#ffcccb,stroke:#d32f2f
    style K fill:#bbdefb,stroke:#1976d2
    style V fill:#c8e6c9,stroke:#388e3c

    %% --- Link Styling ---
    %% Updated indices to 0,1,2 because the Note links are now defined first (at the top)
    linkStyle 0,1,2 stroke-width:1px,fill:none,stroke:gray,stroke-dasharray: 3 3;
```

```{important}
**Terminology: Scores vs. Weights**

It's crucial to understand the distinction between these two terms that are often confused:

- **Attention Scores** (also called "logits" or "raw scores"): The values **before** the softmax operation (Step 2 output above). These are the results of $\frac{QK^T}{\sqrt{d_k}}$ and can be any real number (positive, negative, large, small).

- **Attention Weights** (also called "attention probabilities"): The values **after** the softmax operation (Step 3 output above). These always sum to 1.0 and represent the percentage of "attention" each position receives.

When debugging attention mechanisms or reading research papers, knowing which one is being discussed is critical. Scores are used for computing gradients, while weights are used for the final weighted sum with the Values.
```

### Example Walkthrough: Crunching the Numbers

Let's trace the math using vectors from the plot above. We'll use the pronoun resolution example: when "it" (query) attends to "animal", "street", and "because" (keys).

* **Query (Q):** `[3, 1]`
* **K_animal (Exact Match):** `[3, 1]`
* **K_street (Misaligned):** `[1, 4]`
* **K_because (Short, aligned):** `[1.5, 0.5]`

**Step 1: The Dot Product ($QK^T$)** - Computing Raw Scores
* Score (animal): $(3 \times 3) + (1 \times 1) = 10$
* Score (street): $(3 \times 1) + (1 \times 4) = 7$
* Score (because): $(3 \times 1.5) + (1 \times 0.5) = 5$

```{note}
**Scaling to Real Models:** We used 2D vectors for easy visualization, but in practice, attention operates in **much higher dimensions** (typically 512D or more). The good news? The math is identical—we're still just computing dot products to measure alignment. The key difference is that in 512 dimensions, we're comparing 512-element vectors. The dot product still measures alignment, but now across hundreds of dimensions simultaneously, allowing the model to capture much richer relationships between words.
```

**Step 2: Scaling ($\sqrt{d_k}$)**
We divide by $\sqrt{2} \approx 1.41$.
* Scaled Score (animal): $10 / 1.41 \approx 7.09$
* Scaled Score (street): $7 / 1.41 \approx 4.96$
* Scaled Score (because): $5 / 1.41 \approx 3.54$

**Step 3: Softmax**
We exponentiate and normalize to get percentages using the Softmax formula:

$$P(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$

* **Probability (animal):**
    $$P_1 = \frac{e^{7.09}}{e^{7.09} + e^{4.96} + e^{3.54}} \approx \frac{1199}{1199 + 142 + 34} \approx \frac{1199}{1375} \approx \mathbf{87\%}$$

* **Probability (street):**
    $$P_2 = \frac{e^{4.96}}{e^{7.09} + e^{4.96} + e^{3.54}} \approx \frac{142}{1375} \approx \mathbf{10\%}$$

* **Probability (because):**
    $$P_3 = \frac{e^{3.54}}{e^{7.09} + e^{4.96} + e^{3.54}} \approx \frac{34}{1375} \approx \mathbf{3\%}$$

Notice how the mechanism successfully identified the aligned vector ("animal") as the important one, giving it 87% of the **attention weights** (recall: weights are the post-softmax probabilities, while scores are the pre-softmax logits 7.09, 4.96, and 3.54). This matches what we'll see in the geometric visualization below.

### Geometric View: The Four Steps of Attention

In the worked example above, we calculated that Q=[3, 1] attending to K_animal=[3, 1], K_street=[1, 4], and K_because=[1.5, 0.5] yields attention weights of 87%, 10%, and 3%. Let's visualize this step-by-step using those exact vectors to see how "it" computes its final context vector.

We'll break attention into its four steps:
1. **Similarity**: Compute dot products Q·K (scores)
2. **Scaling**: Divide by √d_k
3. **Softmax**: Convert to probabilities (weights)
4. **Weighted Sum**: Combine values using those weights

#### Step 1: Similarity in Q-K Space (Computing Scores)

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def arrow(ax, start, end, color=None, lw=3, alpha=1.0, ls='-', z=3, ms=18):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        alpha=alpha,
        zorder=z
    ))

def box(ax, x, y, text, fs=11, ha="left", va="top"):
    ax.text(
        x, y, text, fontsize=fs, ha=ha, va=va,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.65", alpha=0.96)
    )

# Toy example aligned to your walkthrough
Q = np.array([3.0, 1.0])
K = {
    "animal":  np.array([3.0, 1.0]),
    "street":  np.array([1.0, 4.0]),
    "because": np.array([1.5, 0.5]),
}
names = list(K.keys())
scores = np.array([Q @ K[n] for n in names])

# Use matplotlib default color cycle (no hard-coded palette)
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = {n: cycle[i % len(cycle)] for i, n in enumerate(names)}
cQ = cycle[3 % len(cycle)]

fig, ax = plt.subplots(figsize=(7.5, 9))

# Offset Q slightly so it doesn't perfectly overlap K_animal
q_hat = Q / np.linalg.norm(Q)
perp = np.array([-q_hat[1], q_hat[0]])
eps = 0.15
q_start = eps * perp
q_end = q_start + Q
arrow(ax, (q_start[0], q_start[1]), (q_end[0], q_end[1]),
      color=cQ, lw=4, ls="--", z=5, ms=20)

# Keys with SAME thickness, but circle size proportional to score
smax = scores.max()
for i, n in enumerate(names):
    k = K[n]
    # Arrow points to the center of the circle at k[0], k[1]
    arrow(ax, (0, 0), (k[0], k[1]), color=colors[n], lw=3.5, z=4, ms=18)
    # Circle size proportional to score, doubled
    circle_size = 2 * (50 + 350 * (scores[i] / smax))
    ax.scatter([k[0]], [k[1]], s=circle_size, color=colors[n], alpha=0.6, zorder=6, edgecolors='white', linewidths=1.5)

# Color-coded legend with colored markers
from matplotlib.lines import Line2D
legend_elements = []

# Add Q line with note about offset
legend_elements.append(Line2D([0], [0], color=cQ, linestyle='--', lw=3,
                              label=f"Q('it') = {Q.tolist()} (offset for display)"))

# Add colored entries for each key
for i, n in enumerate(names):
    label_text = f"{n:7s}: K={K[n].tolist()}  score={scores[i]:.0f}"
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=colors[n], markersize=10,
                                  label=label_text))

ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
          framealpha=0.95, edgecolor='0.65')

# Add note about circle sizes
ax.text(0.03, 0.02, "Circle size = dot-product score",
        fontsize=10, ha='left', va='bottom', transform=ax.transAxes,
        style='italic', alpha=0.7)

ax.axhline(0, color="k", alpha=0.18)
ax.axvline(0, color="k", alpha=0.18)
ax.grid(True, alpha=0.25)
ax.set_aspect("equal")
ax.set_xlabel("dim 1", fontsize=12)
ax.set_ylabel("dim 2", fontsize=12)
ax.set_xlim(-0.5, 4.8)
ax.set_ylim(-0.5, 5.4)

plt.tight_layout()
plt.show()
```

**What this shows:** The query "it" Q=[3, 1] (blue dashed arrow) compares against each key using the dot product. Notice that K_animal=[3, 1] **perfectly aligns** with Q (score=10), while K_street=[1, 4] points in a different direction (score=7). **Circle size** reflects the raw dot product scores—before any normalization.

#### Steps 2-3: Scaling and Softmax (Scores → Weights)

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()

def box(ax, x, y, text, fs=11, ha="left", va="top"):
    ax.text(
        x, y, text, fontsize=fs, ha=ha, va=va,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.65", alpha=0.96)
    )

Q = np.array([3.0, 1.0])
K = {
    "animal":  np.array([3.0, 1.0]),
    "street":  np.array([1.0, 4.0]),
    "because": np.array([1.5, 0.5]),
}
names = list(K.keys())

d_k = Q.size
scores = np.array([Q @ K[n] for n in names])
logits = scores / np.sqrt(d_k)
w = softmax(logits)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = {n: cycle[i % len(cycle)] for i, n in enumerate(names)}

fig, ax = plt.subplots(figsize=(8.5, 4.8))

y = np.arange(len(names))
bars = ax.barh(y, w, height=0.55)
for i, b in enumerate(bars):
    b.set_color(colors[names[i]])
    b.set_alpha(0.85)

ax.set_yticks(y, labels=names, fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.set_xlabel("attention weight", fontsize=12)
ax.grid(True, axis="x", alpha=0.25)

for i, n in enumerate(names):
    ax.text(w[i] + 0.02, i, f"{w[i]*100:.0f}%  (logit={logits[i]:.2f})",
            va="center", fontsize=12)

box(ax, 0.03, 0.08, "logit = score / √dₖ   (here dₖ=2, √2≈1.41)\nweights = softmax(logits)",
    fs=11, va="bottom")

plt.tight_layout()
plt.show()
```

**What this shows:** We divide each score by √d_k=√2≈1.41 to get "logits", then apply softmax to convert them into probabilities that sum to 1.0. The result: "animal" gets **87%** of the attention (from score=10), "street" gets **10%** (from score=7), and "because" gets **3%** (from score=5). These are the **attention weights**.

#### Step 4: Copy from Values (Weighted Sum in V-Space)

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()

def box(ax, x, y, text, fs=11, ha="left", va="top"):
    ax.text(
        x, y, text, fontsize=fs, ha=ha, va=va,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.65", alpha=0.96)
    )

Q = np.array([3.0, 1.0])
K = {
    "animal":  np.array([3.0, 1.0]),
    "street":  np.array([1.0, 4.0]),
    "because": np.array([1.5, 0.5]),
}
V = {
    "animal":  np.array([2.0, 1.5]),
    "street":  np.array([0.5, 0.3]),
    "because": np.array([-0.5, 1.2]),
}
names = list(K.keys())

d_k = Q.size
scores = np.array([Q @ K[n] for n in names])
logits = scores / np.sqrt(d_k)
w = softmax(logits)
context = sum(w[i] * V[names[i]] for i in range(len(names)))

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = {n: cycle[i % len(cycle)] for i, n in enumerate(names)}
cCTX = cycle[4 % len(cycle)]

fig, ax = plt.subplots(figsize=(7.5, 6.8))

# Plot value points
for i, n in enumerate(names):
    v = V[n]
    ax.scatter([v[0]], [v[1]], s=160, color=colors[n], alpha=0.95, zorder=4)
    ax.text(v[0] + 0.10, v[1] + 0.06, f"V_{n}", fontsize=12)

# Context point
ax.scatter([context[0]], [context[1]], s=260, color=cCTX, marker="*", zorder=6, alpha=0.95)
ax.text(context[0] + 0.10, context[1] + 0.06, "context", fontsize=12)

# Pull lines from context to values (thickness ~ weight)
for i, n in enumerate(names):
    v = V[n]
    lw = 1.5 + 7.0 * (w[i] / w.max())
    ax.plot([context[0], v[0]], [context[1], v[1]],
            linewidth=lw, alpha=0.22, color=colors[n], zorder=2)

# Contribution summary box
contrib_lines = [
    f"{n:7s}: w={w[i]:.2f}  w·V≈[{(w[i]*V[n])[0]:.2f}, {(w[i]*V[n])[1]:.2f}]"
    for i, n in enumerate(names)
]
contrib_lines.append("")
contrib_lines.append(f"context≈[{context[0]:.2f}, {context[1]:.2f}]")
box(ax, 0.03, 0.97, "Values live in a different space than Keys\n\n" + "\n".join(contrib_lines), fs=11)

ax.axhline(0, color="k", alpha=0.18)
ax.axvline(0, color="k", alpha=0.18)
ax.grid(True, alpha=0.25)
ax.set_aspect("equal")
ax.set_xlabel("dim 1", fontsize=12)
ax.set_ylabel("dim 2", fontsize=12)

pts = [context] + [V[n] for n in names]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
ax.set_xlim(min(xs)-1.1, max(xs)+1.1)
ax.set_ylim(min(ys)-1.1, max(ys)+1.1)

plt.tight_layout()
plt.show()
```

**What this shows:** Values live in a **different space** than keys. Using the attention weights from Step 3, we compute a weighted average: context = 0.87×V_animal + 0.10×V_street + 0.03×V_because ≈ [1.79, 1.44]. The thick line to V_animal shows it dominates the contribution. This final context vector becomes the new representation for "it"—enriched with semantic content from "animal".

**Key Insight:** Attention is a **weighted average in value space**, where the weights come from measuring similarity in query-key space. This is why we need separate Q, K, V projections—keys determine *how much* to attend (pronoun resolution via geometric alignment), but values determine *what* information to extract (semantic content).

---

## Part 3: Visualizing the Attention Map

We've explored attention through different lenses—from the "bank" disambiguation problem to pronoun resolution with the geometric view above. Now let's see the **full attention pattern** for our pronoun resolution example as a heatmap.

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

We've seen the intuition (Q/K/V filing cabinet), the math (dot products and scaling), and the visualization (attention heatmaps). Now let's see how remarkably simple the actual code is.

We can implement this entire mechanism in fewer than 20 lines of code.

Note the use of `masked_fill`, which we will use in [L06 - The Causal Mask](L06_The_Causal_Mask.md) to prevent the model from "cheating" by looking at future words.

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
3. **Scaling:** We divide by $\sqrt{d_k}$ to stop the gradients from vanishing when vectors get large.

**Next Up: L04 – Multi-Head Attention.** One attention head is good, but it can only focus on one relationship at a time (e.g., "it" → "animal"). What if we also need to know that "animal" is the *subject* of the sentence? We need more heads!

---

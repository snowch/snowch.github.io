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

# L02 - Embeddings & Positional Encoding: Giving Numbers Meaning [DRAFT]

*How words become vectors in space, and how we tell time without a clock*

---

In [L01 - Tokenization](L01_Tokenization_From_Scratch.md), we turned text into IDs. But to a neural network, ID `464` and ID `465` are just arbitrary numbers. There is no inherent relationship between them.

In this post, we solve two problems:
1. **Meaning:** How do we represent words so that "King" is mathematically closer to "Queen" than it is to "Toaster"?
2. **Order:** Since the attention-based model we’ll build in the next lesson processes all tokens at once, how do we tell it that "The dog bit the man" is different from "The man bit the dog"?

By the end of this post, you'll understand:
- The intuition of **Embedding Spaces**.
- How to implement a lookup table from scratch.
- The beautiful math behind **Sinusoidal Positional Encodings**.

```{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import matplotlib
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: The Embedding Space

Imagine every word is a point in a high-dimensional room. Words with similar meanings are "clumped" together. This is an **Embedding Space**.

An **Embedding Layer** is simply a big lookup table. If our vocabulary size is 10,000 and we want each word to be represented by a vector of 512 numbers, our table is a  matrix.

### The Lookup Operation

When the model sees ID `5`, it doesn't do math on the number 5. It simply grabs the **5th row** of the embedding matrix.

```{code-cell} ipython3
:tags: [remove-input]

# Simplified 2D Embedding Visualization
words = ["King", "Queen", "Man", "Woman", "Apple", "Orange"]
# Arbitrary coordinates to show relationships
coords = np.array([
    [0.1, 0.9], [0.2, 0.85], # Royalty
    [0.15, 0.2], [0.25, 0.15], # People
    [0.8, 0.5], [0.85, 0.45]  # Fruit
])

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], color='blue')

for i, word in enumerate(words):
    plt.annotate(word, (coords[i, 0]+0.02, coords[i, 1]))

plt.title("2D Projection of an Embedding Space")
plt.xlabel("Dimension 1 (e.g., Royalty-ness)")
plt.ylabel("Dimension 2 (e.g., Gender-ness)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

```

---

## Part 2: The Problem of Order

Standard Neural Networks (like the MLPs we built in the [LLM from Scratch series](L01_Tokenization_From_Scratch.md)) process data in a specific order. The attention mechanism we’ll introduce next is **parallel**. It looks at every word in a sentence at the exact same time.

Without help, the attention-based model sees the sentence "The dog bit the man" as a **bag of words**. It has no idea which word came first.

We fix this by adding **Positional Encodings**.

---

## Part 3: Positional Encoding (The Sine/Cosine Trick)

This is often the most confusing part of the Transformer architecture, so let's derive it from scratch.

We need to give every word a unique ID for its position (). But we have constraints:

1. **Values must be bounded:** Neural networks hate large numbers. If we just use integers (), the values for the 500th word will explode the gradients.
2. **Steps must be consistent:** The "distance" between position 1 and 2 should be the same as between position 4 and 5.
3. **Deterministic:** We need to calculate this on the fly for any sentence length.

### The Intuition: The "Binary Clock"

How can we represent large numbers using only s and s? **Binary.**

Look at how bits change as we count up:

| Number | Bit 3 (Slow) | Bit 2 (Medium) | Bit 1 (Fast) |
| --- | --- | --- | --- |
| **0** | 0 | 0 | 0 |
| **1** | 0 | 0 | 1 |
| **2** | 0 | 1 | 0 |
| **3** | 0 | 1 | 1 |
| **4** | 1 | 0 | 0 |

Notice the pattern?

* The **Least Significant Bit (Fast)** alternates every single step:  (High Frequency).
* The **Next Bit (Medium)** alternates every two steps:  (Lower Frequency).
* The **Most Significant Bit (Slow)** alternates every four steps (Lowest Frequency).

Each column oscillates at a different frequency. Together, they create a unique combination for every row.

### From Binary to Continuous (Sine & Cosine)

Transformers use this exact logic, but instead of discrete bits (), we use **continuous waves** ( to ) using Sine and Cosine.

* **Dimension 0** of our position vector is like the "fast bit": it wiggles up and down very quickly as you move along the sentence.
* **Dimension 100** is like the "slow bit": it wiggles up and down very slowly.

By combining these different "wiggles," every single position gets a unique fingerprint vector.

### The Formula

For a position  and dimension :

Don't let the scary  scare you. It’s just a knob that controls the **wavelength**:

* When  is low (low dimension index), the denominator is small  **High Frequency wave.**
* When  is high (high dimension index), the denominator is huge ()  **Low Frequency wave.**

### Visualization

Let's generate the matrix. In the plot below:

* **X-axis:** The Embedding Dimensions (0 to 128).
* **Y-axis:** The Position in the sentence (0 to 50).
* **Color:** The value (Blue is negative, Red is positive).

Notice the "barber pole" pattern? That is the frequencies getting slower as you move to the right (higher dimensions).

```{code-cell} ipython3
:tags: [remove-input]

def get_positional_encoding(max_seq_len, d_model):
    pe = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            div_term = np.exp(i * -(np.log(10000.0) / d_model))
            pe[pos, i] = np.sin(pos * div_term)
            pe[pos, i + 1] = np.cos(pos * div_term)
    return pe

d_model = 128
max_len = 50
pe = get_positional_encoding(max_len, d_model)

plt.figure(figsize=(10, 6))
plt.imshow(pe, cmap='RdBu', aspect='auto')
plt.colorbar(label='Encoding Value')
plt.title("Positional Encoding Matrix (Sine & Cosine)")
plt.xlabel("Embedding Dimension (Frequency decreases →)")
plt.ylabel("Position in Sentence (Time ↓)")
plt.show()


```

### The "Why Addition?" Question

A common question is: *Why do we ADD this to the word embedding? Won't that ruin the word's meaning?*

Imagine our embedding space is a giant 512-dimensional room.

1. **Word Embedding:** "King" moves us to a specific coordinate in that room.
2. **Positional Encoding:** Moves us a *tiny* nudge in a specific direction based on position.

Because the space is so high-dimensional (512, 1024, or even 12288 dimensions in GPT-3), the "nudge" for position doesn't overwrite the meaning. It just "colors" it.

* "King" at pos 1  "King" + slight blue tint.
* "King" at pos 99  "King" + slight red tint.

The model learns to distinguish the "King-ness" (meaning) from the "Red-ness" (position).

Here is the visual proof:

```{code-cell} ipython3
:tags: [remove-input]

# Toy example: 5 tokens, 8 dimensions
toy_len = 5
toy_dim = 8

toy_embeddings = np.random.randn(toy_len, toy_dim)
toy_positions = get_positional_encoding(toy_len, toy_dim)
toy_sum = toy_embeddings + toy_positions

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].imshow(toy_embeddings, aspect="auto", cmap="viridis")
axes[0].set_title("1. Token Embeddings (Meaning)")
axes[1].imshow(toy_positions, aspect="auto", cmap="RdBu")
axes[1].set_title("2. Positional Encodings (Order)")
axes[2].imshow(toy_sum, aspect="auto", cmap="viridis")
axes[2].set_title("3. Sum (Input to Model)")
for ax in axes:
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Token Position")
plt.tight_layout()
plt.show()


```

---

## Part 4: Putting it Together

The final input to our model is:


Now, the vector for "dog" at position 2 is slightly different from the vector for "dog" at position 5. The "meaning" is the same, but the "stamp" of its location is unique.

```python
# Pseudo-code for the input pipeline
word_ids = [464, 2068, 7586] # "The quick brown"
embeddings = embedding_layer(word_ids) # Shape: [3, 512]
positions = positional_encoding_layer(range(len(word_ids))) # Shape: [3, 512]

final_input = embeddings + positions

```

---

## Summary

1. **Embeddings** map discrete IDs to continuous vectors where distance equals similarity.
2. **Positional Encodings** inject a sense of order into a model that otherwise sees everything at once.
3. **Addition:** We simply add these two vectors together. The model learns to separate the "meaning" signal from the "position" signal during training.

## Related reading

Embeddings at Scale Book:

- [Ch01: The Embedding Revolution](https://snowch.github.io/embeddings-at-scale-book/chapters/ch01_embedding_revolution.html)
- [Ch02: Similarity & Distance Metrics](https://snowch.github.io/embeddings-at-scale-book/chapters/ch02_similarity_distance_metrics.html)
- [Ch03: Vector Database Fundamentals](https://snowch.github.io/embeddings-at-scale-book/chapters/ch03_vector_database_fundamentals.html)
- [Ch04: Text Embeddings](https://snowch.github.io/embeddings-at-scale-book/chapters/ch04_text_embeddings.html)

**Next Up: L03 – The Attention Mechanism.** This is the "Aha!" moment of the entire series. We will build the logic that allows the model to decide which words in a sentence are most relevant to each other.

---

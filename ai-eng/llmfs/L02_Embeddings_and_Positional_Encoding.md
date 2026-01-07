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

Instead of just giving the model the word vector, we add a **position signal** to it. Concretely, we build a vector for each position where **each dimension is a sine or cosine value** at a specific frequency. Then we **add that position vector** to the token embedding so the model can tell "same word, different place" apart.

We use sine and cosine waves at different frequencies (typically alternating sin and cos across dimensions). This gives us two useful properties:
1. **Every position is distinct.** Low-frequency waves change slowly, high-frequency waves change quickly, and their combination yields a unique signature per position.
2. **Relative distance is easy to learn.** A shifted sine/cosine pair can be expressed as a linear function of the original pair, which makes it easier for attention layers to infer how far apart two tokens are.

### The Formula

For a position  and dimension :


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
plt.xlabel("Embedding Dimension")
plt.ylabel("Position in Sentence")
plt.show()

```

This heatmap shows the **positional encoding matrix** itself: each row is a position, and each column is an embedding dimension filled with sine/cosine values.

To make the “add the position signal” step concrete, here’s a tiny toy example showing the **embedding**, the **positional encoding**, and the **sum**:

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
axes[0].set_title("Token Embeddings")
axes[1].imshow(toy_positions, aspect="auto", cmap="RdBu")
axes[1].set_title("Positional Encodings")
axes[2].imshow(toy_sum, aspect="auto", cmap="viridis")
axes[2].set_title("Embeddings + Positions")
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

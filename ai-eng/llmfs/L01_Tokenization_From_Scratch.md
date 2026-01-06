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

# L01 - Tokenization From Scratch: Teaching Computers to Read

*A visual guide to Byte Pair Encoding (BPE) and the bridge from text to tensors*

---

In the [Neural Networks from Scratch series](../nnfs/nn_tutorial_blog.md#the-networks-job), we fed our neural networks pixel values ($0$ to $1$). But how do we feed a model a sentence like **"The quick brown fox"**? 

Neural networks don't understand letters, and they don't understand words. They understand **vectors**. To get there, we need a "translator" that turns text into a sequence of integers. This process is called **Tokenization**.

By the end of this post, you'll understand:
- Why we don't just use characters or whole words.
- The intuition behind **Byte Pair Encoding (BPE)**.
- How to build a BPE tokenizer from scratch that handles "Out of Vocabulary" words.

```{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import matplotlib
import numpy as np
import collections

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

```

---

## Part 1: The Goldilocks Problem

To turn text into numbers, we have three choices. Each has a major flaw:

| Method | Example | Vocab Size | Problem |
| --- | --- | --- | --- |
| **Character-level** | `c`, `a`, `t` | Tiny (~100) | Characters have no meaning. Sequences get too long. |
| **Word-level** | `cat` | Massive (50k+) | Can't handle "cats" if it only saw "cat". |
| **Subword-level** | `play`, `##ing` | Medium (32k-50k) | **Just right.** Meaningful and flexible. |

**Subword tokenization** (like BPE) is the industry standard. It ensures that common words are one unit, while rare words are broken into chunks like `un` + `believ` + `able`.

---

## Part 2: Byte Pair Encoding (BPE) Intuition

BPE is an iterative algorithm. It starts with a vocabulary of individual characters and slowly "glues" the most frequent adjacent pairs together to create new tokens.

### The Algorithm:

1. **Initialize:** Break words in your training data into character sequences.
2. **Count:** Find the most frequent pair of adjacent tokens (e.g., 't' and 'h').
3. **Merge:** Create a new token 'th' and replace all occurrences in the data.
4. **Repeat:** Stop when you reach a target vocabulary size.

Let's visualize this "gluing" process:

```{code-cell} ipython3
:tags: [remove-input]

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 10)
ax.set_ylim(-1, 5)
ax.axis('off')

# Box helper
def draw_token(x, y, text, color='lightblue'):
    ax.add_patch(plt.Rectangle((x, y), 1.2, 0.8, color=color, ec='black', lw=2))
    ax.text(x+0.6, y+0.4, text, ha='center', va='center', fontsize=12, family='monospace', fontweight='bold')

# Before
ax.text(2, 4, "Step 1: Raw Characters", ha='center', fontweight='bold')
draw_token(0.5, 2.5, "h")
draw_token(2.0, 2.5, "e")
draw_token(3.5, 2.5, "l")
draw_token(5.0, 2.5, "l")
draw_token(6.5, 2.5, "o")

# Arrow
ax.annotate('Merge (l, l)', xy=(4, 1.2), xytext=(4, 2.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2))

# After
ax.text(2, 0.5, "Step 2: After 1 Merge", ha='center', fontweight='bold')
draw_token(0.5, -0.5, "h")
draw_token(2.0, -0.5, "e")
draw_token(3.5, -0.5, "ll", color='lightgreen') # Merged
draw_token(5.0, -0.5, "o")

plt.show()

```

---

## Part 3: Coding the BPE Learner

To build this from scratch, we need to treat our text as a "dictionary" of frequencies.

```python
# Initial training data: "hug" (10 times), "pug" (5 times), "pun" (12 times)
vocab = {
    'h u g': 10,
    'p u g': 5,
    'p u n': 12,
    'b u n': 4
}

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in v_in:
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = v_in[word]
    return v_out

# Run 1 merge
stats = get_stats(vocab)
best_pair = max(stats, key=stats.get) # In our data, ('u', 'n') is very frequent
vocab = merge_vocab(best_pair, vocab)

print(f"Merged pair: {best_pair}")
# 'p u n' becomes 'p un'

```

---

## Part 4: The Tokenization Pipeline

Once we have learned all our merges, we have a **Vocabulary**. Every token in that vocabulary is assigned a unique **ID** (an integer).

### Why are IDs important?

In our next blog, these IDs will serve as the row indices for our **Embedding Matrix**. If the word "Cat" has ID `542`, the model will look at the 542nd row of its "dictionary" to find the vector representing "Cat".

```{code-cell} ipython3
:tags: [remove-input]

# Visualizing the final pipeline
tokens = ["The", "quick", "brown", "fox"]
ids = [464, 2068, 7586, 21831]

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Draw the flow
start_x = 0.5
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
for i, (txt, tid) in enumerate(zip(tokens, ids)):
    # Text box
    ax.text(
        start_x + i * 2.5,
        2,
        txt,
        ha='center',
        fontsize=14,
        bbox=dict(boxstyle='round', facecolor='white'),
    )
    # Arrow
    ax.annotate(
        '',
        xy=(start_x + i * 2.5, 0.8),
        xytext=(start_x + i * 2.5, 1.7),
        arrowprops=dict(arrowstyle='->'),
    )
    # ID box
    ax.text(
        start_x + i * 2.5,
        0.3,
        f"ID: {tid}",
        ha='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgray'),
    )

ax.set_title("The Final Pipeline: Text → Tokens → Integers (IDs)", fontsize=16, pad=20)
plt.show()

```

---

## Summary

We have successfully bridged the gap between human language and machine numbers.

1. **Tokenization** breaks text into chunks the model can manage.
2. **BPE** allows us to represent any word, even those we've never seen, by breaking them into sub-units.
3. **The Vocabulary** is just a lookup table that maps these sub-units to integers.

## Related reading

Embeddings at Scale Book:

- [Ch01: The Embedding Revolution](https://snowch.github.io/embeddings-at-scale-book/chapters/ch01_embedding_revolution.html)
- [Ch02: Similarity & Distance Metrics](https://snowch.github.io/embeddings-at-scale-book/chapters/ch02_similarity_distance_metrics.html)
- [Ch03: Vector Database Fundamentals](https://snowch.github.io/embeddings-at-scale-book/chapters/ch03_vector_database_fundamentals.html)
- [Ch04: Text Embeddings](https://snowch.github.io/embeddings-at-scale-book/chapters/ch04_text_embeddings.html)

**Next Up: L02 – Embeddings & Positional Encoding.** Now that we have IDs, how do we turn them into high-dimensional vectors that capture "meaning"? And how do we tell the model that "The cat ate the mouse" is different from "The mouse ate the cat"?

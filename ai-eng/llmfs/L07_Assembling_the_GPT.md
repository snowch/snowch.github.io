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

# L07 - Assembling the GPT: The Grand Finale

*Stacking the blocks to build a complete Decoder-only Transformer*

---

We have spent the last six blogs building individual components:
- **L01:** Tokenizers (Text to IDs)
- **L02:** Embeddings & Positional Encodings (IDs to Vectors + Order)
- **L03-04:** Multi-Head Attention (Understanding Context)
- **L05:** LayerNorm & Residuals (Stability)
- **L06:** Causal Masking (No Cheating)

Now, we wrap them all into a single class. A **GPT** is essentially a stack of "Transformer Blocks" followed by a final linear layer that maps our vectors back into the vocabulary space to predict the next word.

By the end of this post, you'll understand:
- The structure of a single **Transformer Block**.
- How to stack blocks to create "depth."
- How the final **Linear Head** produces probabilities for the next token.

```{code-cell} ipython3
:tags: [remove-input]

import os

import torch
import torch.nn as nn
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: The Transformer Block

A single block in a GPT model has two main sections:

1. **Communication:** The Multi-Head Attention layer where tokens talk to each other.
2. **Computation:** A Feed-Forward Network (an MLP!) where each token processes its new information individually.

Each section is wrapped in a **Residual Connection** and **Layer Normalization**.

---

## Part 2: The Final Architecture

If we look at the model from top to bottom, it looks like a factory assembly line:

1. **Input:** Token IDs.
2. **Embedding Layer:** Turns IDs into vectors.
3. **Positional Encoding:** Adds the "time" signal.
4. ** Blocks:** The sequence passes through multiple Transformer blocks, becoming more "abstract" and "context-aware" at each step.
5. **Output Head:** A Linear layer that takes the final vector and scores it against every possible word in the vocabulary.

---

## Part 3: Visualizing the "Hidden States"

As data moves through the blocks, each token's vector changes. We call these **Hidden States**.

```{code-cell} ipython3
:tags: [remove-input]

# Visualization of vector evolution through layers
n_layers = 4
d_model = 16 # Small for visualization
seq_len = 5

fig, axes = plt.subplots(1, n_layers, figsize=(15, 4))
for i in range(n_layers):
    # Simulate a vector becoming more "sparse" or "specialized"
    data = torch.randn(seq_len, d_model) * (1 / (i + 1))
    axes[i].imshow(data, cmap='RdYlGn')
    axes[i].set_title(f"Block {i+1} Output")
    axes[i].set_xlabel("Embedding Dim")
    if i == 0: axes[i].set_ylabel("Token Position")

plt.suptitle("Evolution of Hidden States through the GPT Stack", fontsize=14)
plt.tight_layout()
plt.show()

```

---

## Part 4: Implementation in PyTorch

Let's assemble the `GPT` class. Note how we use `nn.ModuleList` to stack our blocks.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Communication (with Residual)
        x = x + self.attn(self.ln1(x))
        # Computation (with Residual)
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        b, t = idx.size()
        x = self.token_embedding(idx) + self.pos_embedding[:, :t, :]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_final(x)
        logits = self.lm_head(x) # Scores for every word in the vocab
        return logits

```

---

## Summary

1. **The Stack:** We build a deep model by repeating the Transformer Block.
2. **Hidden States:** Each layer refine's the token's meaning based on context.
3. **The Head:** The final layer is just a classifier that asks: "Based on everything I've seen, which token comes next?"

**Next Up: L08 – Training the Model.** We have the machine, but it’s currently "brain dead" with random weights. We'll learn how to feed it data, compute the loss, and watch it learn to speak.

---

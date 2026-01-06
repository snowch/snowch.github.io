---
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

# L08 - Training the LLM: Learning to Speak

*Feeding the model data and watching its "Loss" collapse*

---

We have built the architecture of a GPT in [L07 - Assembling the GPT](L07_Assembling_the_GPT.md). But right now, if you ask it to complete a sentence, it will spit out random gibberish. Its weights are just random numbers.

In this post, we take our "brain-dead" model with random weights and teach it how to speak. We’ll cover the training loop, the **Cross-Entropy Loss** function (which you’ll recognize from your NN series!), and the process of **Gradient Descent** at the scale of an LLM.

To make it smart, we need to **train** it. For an LLM, training is a game of "Guess the Next Token." We give it millions of examples, and every time it guesses wrong, we nudge the weights to make that guess more likely next time.

By the end of this post, you'll understand:
- How to structure **Bigram-style training data**.
- Why **Cross-Entropy Loss** is the perfect "grading" system.
- How the training loop looks for a Transformer.

```{code-cell} ipython3
:tags: [remove-input]

import os
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
matplotlib.set_loglevel("error")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: The Training Task (Next Token Prediction)

The beauty of LLMs is **Self-Supervised Learning**. We don't need humans to label the data. The text itself is the label!

If we have the sentence: **"The cat sat on the mat."**

* Input: `[The]` -> Target: `[cat]`
* Input: `[The, cat]` -> Target: `[sat]`
* Input: `[The, cat, sat]` -> Target: `[on]`

The model learns the statistical structure of language by trying to solve this puzzle billions of times.

---

## Part 2: Measuring Error (Cross-Entropy Loss)

How do we tell the model "how wrong" it was?

The model outputs a probability distribution over the whole vocabulary (e.g., 50,000 words). If the correct next word was "cat", we want the probability for "cat" to be **1.0** and everything else to be **0.0**.

We use **Cross-Entropy Loss**. It calculates the difference between the model's predicted distribution and the "perfect" distribution (a 1 at the correct word).

$$ \text{Loss} = -\log(\text{Probability of correct word}) $$

If the model is confident and correct, the loss is near 0. If it’s confident and wrong, the loss explodes.

---

## Part 3: Visualizing the Training Curve

As the model trains, we track the **Loss**. A "healthy" training run shows a curve that drops sharply and then levels off.

```{code-cell} ipython3
:tags: [remove-input]

epochs = np.arange(0, 100)
# Simulated loss curve: exponential decay + noise
loss = 10 * np.exp(-epochs/20) + 2 + np.random.normal(0, 0.1, 100)

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, color='blue', lw=2, label='Training Loss')
plt.axhline(y=2, color='red', linestyle='--', label='Theoretical Minimum')
plt.title("LLM Training Progress: The 'Learning' Curve")
plt.xlabel("Training Steps (Iterations)")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

```

---

## Part 4: The Training Loop Implementation

Here is how we set up the loop in PyTorch. We use the **AdamW** optimizer, which is the industry standard for Transformers.

```python
# Assuming 'model' is our GPT and 'train_loader' gives us (x, y) pairs
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(epochs):
    for x, y in train_loader:
        # x: input token IDs [batch, seq_len]
        # y: target token IDs (the shifted version of x)
        
        # 1. Forward pass
        logits = model(x) # [batch, seq_len, vocab_size]
        
        # 2. Reshape for loss (PyTorch expects [batch * seq_len, vocab_size])
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = y.view(B*T)
        
        loss = F.cross_entropy(logits, targets)
        
        # 3. Backward pass (Backpropagation!)
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Update weights
        optimizer.step()
        
    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

```

---

## Summary

1. **Self-Supervision:** The model teaches itself by using the next word in a text as its own target.
2. **The Loss Function:** Cross-Entropy penalizes the model for being "surprised" by the correct word.
3. **Optimization:** We use the gradient of the loss to adjust the millions of parameters in our Attention heads and Feed-Forward layers.

**Next Up: L09 – Inference & Sampling.** Now that we have a trained brain, how do we actually get it to "talk" to us? We'll learn about **Temperature**, **Top-K**, and **Top-P** sampling to control the model's creativity.

---

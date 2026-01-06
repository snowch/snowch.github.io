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

# L09 - Inference & Sampling: Controlling the Creativity

*How to talk to a trained brain without it repeating itself*

---

We have a trained GPT model from [L08 - Training the LLM](L08_Training_the_LLM.md). If we give it a prompt, the model will output a list of probabilities for the next word.

This post covers the "magic" of how an LLM actually generates text. Training is about building a probability map; **Inference** is about walking through that map. We will also explore the "knobs" we turn to make the model more creative or more factual.

But how do we choose the "best" word? If we always pick the most likely word (**Greedy Search**), the model often gets stuck in a loop, repeating the same phrase over and over. To make the model sound human, we need to introduce a bit of randomness.

By the end of this post, you'll understand:
- The **Autoregressive Loop** (generating one word at a time).
- How **Temperature** affects the "sharpness" of probabilities.
- Advanced techniques like **Top-K** and **Top-P (Nucleus) sampling**.

```{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import matplotlib

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

```

---

## Part 1: The Autoregressive Loop

Generating text is a loop. The model predicts one token, we append that token to the input, and we feed the new, longer sequence back into the model to get the *next* token.

1. **Prompt:** "The cat"
2. **Model predicts:** "sat" (Prob: 0.8)
3. **New Input:** "The cat sat"
4. **Model predicts:** "on" (Prob: 0.7)
5. **Repeat** until a special "End of Sentence" token is generated.

---

## Part 2: Temperature (The "Creativity" Knob)

Before we pick a word, we can "stretch" or "squash" the probability distribution using **Temperature (T)**.

We divide the raw scores (logits) by **T** before the Softmax:
$$ p_i = \frac{e^{z_i / T}}{\sum e^{z_j / T}} $$

* **Low Temperature (T < 1):** Makes the high probabilities higher and low ones lower. The model becomes very confident and "boring" (Greedy).
* **High Temperature (T > 1):** Flattens the distribution. Rare words get a higher chance of being picked. The model becomes "creative" but potentially nonsensical.

```{code-cell} ipython3
:tags: [remove-input]

def softmax(logits, T):
    e = np.exp(logits / T)
    return e / np.sum(e)

logits = np.array([2.0, 1.0, 0.1, -1.0])
words = ["Apple", "Banana", "Cat", "Dog"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(words, softmax(logits, 0.5), color='blue', alpha=0.7)
ax1.set_title("Low Temperature (T=0.5)\n'Focused / Greedy'")
ax1.set_ylim(0, 1)

ax2.bar(words, softmax(logits, 1.5), color='orange', alpha=0.7)
ax2.set_title("High Temperature (T=1.5)\n'Creative / Random'")
ax2.set_ylim(0, 1)

plt.show()

```

---

## Part 3: Top-K & Top-P Sampling

Even with temperature, sometimes the model picks a word that is just objectively wrong (like a low chance word). To prevent this, we use filters:

### Top-K Sampling

We only look at the top **K** most likely words and ignore everything else. This keeps the model from "veering off the rails."

### Top-P (Nucleus) Sampling

Instead of a fixed number of words, we pick the smallest set of words whose cumulative probability adds up to **P** (e.g., 0.9). This is more dynamic; if the model is very sure, it might only look at 2 words. If it's unsure, it might look at 20.

---

## Part 4: The Inference Code

Here is how we implement the generation loop in PyTorch, including a simple temperature adjustment.

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 1. Crop idx to the last 'block_size' tokens
        idx_cond = idx[:, -block_size:]
        
        # 2. Forward pass to get logits
        logits = model(idx_cond)
        
        # 3. Focus only on the last time step and scale by temperature
        logits = logits[:, -1, :] / temperature
        
        # 4. Optional: Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 5. Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # 6. Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 7. Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

```

---

## Summary

1. **Inference** is a loop where the model's output becomes its next input.
2. **Temperature** controls how much the model deviates from its most likely guess.
3. **Sampling Strategies** (Top-K/Top-P) prune the "long tail" of unlikely words to maintain coherence.

**Next Up: L10 – Fine-tuning (RLHF & Chat).** We have a model that can complete sentences. But how do we turn it into a helpful assistant that answers questions? We'll look at the final step: taking a "Base" model and turning it into a "Chat" model.

---

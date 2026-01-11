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

# L10 - Inference & Sampling: Controlling the Creativity [DRAFT]

*How to talk to a trained brain without it repeating itself*

---

We have a trained GPT model from [L09 - Training the LLM](L09_Training_the_LLM.md). If we give it a prompt, the model will output a list of probabilities for the next word.

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

**Concrete Example:**

Suppose after applying temperature=1.0 to our logits and running softmax, we get these probabilities:

| Token | Probability | Cumulative Probability |
| --- | --- | --- |
| "the" | 0.40 | 0.40 |
| "a" | 0.30 | 0.70 |
| "this" | 0.20 | 0.90 ← **Cutoff at p=0.9** |
| "that" | 0.05 | 0.95 |
| "my" | 0.03 | 0.98 |
| "your" | 0.02 | 1.00 |

**With `top_p = 0.9`:**
1. Sort tokens by probability (already sorted above)
2. Add probabilities until we reach 0.9: "the" (0.40) + "a" (0.30) + "this" (0.20) = 0.90
3. Keep only these 3 tokens: `["the", "a", "this"]`
4. Renormalize: divide each by the sum (0.90) to get a proper probability distribution
5. Sample from this smaller set

**Result:** The model can only choose from "the", "a", or "this"—cutting off the unlikely tokens "that", "my", and "your."

**Why it's better than top-k:**
- **Adaptive:** When the model is confident (one token has 0.95 probability), nucleus might only keep that 1 token. When uncertain (many tokens around 0.1 each), it keeps more options.
- **Quality-based:** Cuts based on probability mass, not arbitrary count

**Visualizing Top-P Filtering:**

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

# Create a mock probability distribution (sorted)
tokens = ['the', 'a', 'this', 'that', 'my', 'your', 'some', 'one', 'any', 'each']
probs = np.array([0.40, 0.30, 0.20, 0.05, 0.03, 0.015, 0.003, 0.001, 0.0005, 0.0005])
cumsum = np.cumsum(probs)

# Top-p threshold
top_p = 0.9

# Find cutoff
cutoff_idx = np.where(cumsum >= top_p)[0][0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Probability distribution with cutoff
colors = ['green' if i <= cutoff_idx else 'lightgray' for i in range(len(tokens))]
bars = ax1.bar(tokens, probs, color=colors, alpha=0.8, edgecolor='black')

# Annotate kept vs. filtered
ax1.axhline(y=probs[cutoff_idx], color='red', linestyle='--', linewidth=2,
            label=f'Cutoff at token {cutoff_idx + 1}')
ax1.text(cutoff_idx + 0.5, probs[cutoff_idx] + 0.02, 'Top-p=0.9\nCutoff',
         color='red', fontweight='bold', fontsize=10)

ax1.set_xlabel('Tokens (sorted by probability)', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Top-P Nucleus Sampling (p=0.9)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticklabels(tokens, rotation=45)

# Add text labels on bars
for i, (bar, prob) in enumerate(zip(bars, probs)):
    height = bar.get_height()
    status = 'KEPT' if i <= cutoff_idx else 'FILTERED'
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f'{prob:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Cumulative probability
ax2.plot(range(len(tokens)), cumsum, 'o-', linewidth=2, markersize=8, color='blue')
ax2.axhline(y=top_p, color='red', linestyle='--', linewidth=2, label=f'p={top_p} threshold')
ax2.axvline(x=cutoff_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Shade the "nucleus"
ax2.fill_between(range(cutoff_idx + 1), 0, [cumsum[i] for i in range(cutoff_idx + 1)],
                 alpha=0.3, color='green', label='Nucleus (kept)')
ax2.fill_between(range(cutoff_idx + 1, len(tokens)), 0,
                 [cumsum[i] for i in range(cutoff_idx + 1, len(tokens))],
                 alpha=0.3, color='gray', label='Tail (filtered)')

ax2.set_xlabel('Token Rank', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('Cumulative Probability Mass', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(tokens, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Key insight from the visualization:**
- **Left plot**: Only the first 3 tokens (green) are kept—they account for 90% of probability mass
- **Right plot**: Cumulative probability curve crosses the 0.9 threshold after 3 tokens
- The remaining 7 tokens (gray) are filtered out despite being possible candidates

This adaptive filtering keeps quality high while allowing flexibility when the model is uncertain.

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

## Part 5: Advanced Sampling Techniques

### max_new_tokens - Controlling Generation Length

The `max_new_tokens` parameter determines how many tokens the model will generate:

```python
generated = generate(model, prompt, max_new_tokens=50)
```

**What it does:**
- Limits the generation loop to exactly 50 iterations
- After 50 tokens, generation stops regardless of content
- Prevents infinite loops and controls costs (API calls are often priced per token)

**How models know when to stop naturally:**
- Models learn to generate special end-of-sequence tokens (like `<|endoftext|>`)
- When the model generates this token, we can stop early (before hitting max_new_tokens)
- During training, these tokens appear at document boundaries

**Typical values:**
- Chatbots: 512-2048 tokens (one response)
- Code completion: 100-500 tokens (one function)
- Creative writing: 1000-4096 tokens (longer passages)

### Repetition Penalty - Preventing Loops

A common problem: models get stuck repeating the same phrase:

```
"The cat sat on the mat. The cat sat on the mat. The cat sat on..."
```

**Solution: Repetition Penalty**
```python
def apply_repetition_penalty(logits, previous_tokens, penalty=1.2):
    for token in set(previous_tokens):
        # Divide logit by penalty (reduces probability)
        logits[token] /= penalty
    return logits
```

**How it works:**
- Track all previously generated tokens
- Before sampling, reduce the logits of tokens that already appeared
- `penalty > 1.0` makes repetition less likely
- `penalty = 1.0` means no penalty (default behavior)

**Typical values:**
- `penalty = 1.0`: No penalty (may repeat)
- `penalty = 1.2`: Mild discouragement of repetition (balanced)
- `penalty = 1.5`: Strong avoidance (may sound unnatural)

### Beam Search - Deterministic Exploration

All the sampling methods above are **stochastic** (random). **Beam Search** is a deterministic alternative:

**How it works:**
1. Instead of sampling 1 token, keep the top `beam_width` candidates
2. For each candidate, generate the next token
3. Evaluate all `beam_width²` possibilities
4. Keep only the top `beam_width` sequences by total probability
5. Repeat until done

**Example with beam_width=2:**
```
Start: "The"
Step 1: Keep ["The cat" (prob=0.8), "The dog" (prob=0.7)]
Step 2: Expand both → ["The cat sat" (0.64), "The cat ran" (0.56),
                       "The dog sat" (0.49), "The dog ran" (0.42)]
        Keep top 2 → ["The cat sat", "The cat ran"]
... continue ...
```

**Beam Search vs. Sampling:**
| Aspect | Beam Search | Sampling (Top-P/Top-K) |
| --- | --- | --- |
| **Determinism** | Always same output | Different every time |
| **Quality** | Finds high-probability sequences | More diverse, creative |
| **Use cases** | Translation, summarization | Creative writing, chat |
| **Speed** | Slower (beam_width parallel paths) | Faster (single path) |

**When to use:**
- **Beam search:** When you want the "safest" or most likely output (translation, factual Q&A)
- **Sampling:** When you want variety and creativity (storytelling, brainstorming)

---

## Summary

1. **Inference** is a loop where the model's output becomes its next input.
2. **Temperature** controls how much the model deviates from its most likely guess (sharpness of distribution).
3. **Sampling Strategies** (Top-K/Top-P) prune the "long tail" of unlikely words to maintain coherence.
4. **max_new_tokens** controls generation length and prevents runaway generation.
5. **Repetition Penalty** prevents the model from getting stuck in loops by penalizing already-used tokens.
6. **Beam Search** offers a deterministic alternative to sampling, finding high-probability sequences for tasks requiring consistency.

**Next Up: L11 – Fine-tuning (RLHF & Chat).** We have a model that can complete sentences. But how do we turn it into a helpful assistant that answers questions? We'll look at the final step: taking a "Base" model and turning it into a "Chat" model.

---

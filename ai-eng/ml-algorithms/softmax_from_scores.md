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

# Softmax: From Scores to Probabilities

*How neural networks turn raw scores into probabilities for classification*

---

```{code-cell} ipython3
:tags: [remove-input]

# Setup
import logging
import numpy as np
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
```

---

## Part 3: From Scores to Probabilities - Softmax

We saw that neurons produce **scores** — higher means better pattern match. But for classification, we need **probabilities**: "What's the chance this is an edge?"

Our network has two output neurons (one for "Edge", one for "No Edge"), each producing a score. So we have a **vector of scores** $\mathbf{z} = [z_{edge}, z_{no\_edge}]$. These raw scores can be any number — positive, negative, or zero.

<div style="background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 15px 0;">

**📘 Terminology: Logits**

The raw scores before softmax are called **logits**. You'll see this term everywhere in machine learning. 

- Logits can be any real number (−∞ to +∞)
- Softmax converts logits → probabilities (0 to 1, summing to 1)

</div>

**Softmax** converts this score vector into a **probability vector** where:
1. All values are positive
2. They sum to 1.0

### The Intuition: It's Just a Ratio

At its core, softmax answers: **"What fraction of the total is each score?"**

If we only had positive scores, we could just divide by the sum:

$$p_j = \frac{z_j}{\sum_k z_k}$$

Where:
- $j$ is the class index (e.g., $j=0$ for "Edge", $j=1$ for "No Edge")
- $p_j$ is the probability for class $j$
- The sum is over all classes $k$

For example, scores $[3, 1]$ would give $P_0 = 3/(3+1) = 0.75$ and $P_1 = 1/(3+1) = 0.25$.

**The problem:** Scores can be negative or zero, which breaks this.

**The solution:** First apply $e^z$ to make everything positive, then take the ratio.

### The Formula

$$p_j = \frac{e^{z_j}}{\sum_{k} e^{z_k}}$$

The exponential also **amplifies differences** — a score of 5 vs 3 becomes $e^5$ vs $e^3$ (148 vs 20), making the network more confident.

The visualization below shows the 3-step process:

```{code-cell} ipython3
:tags: [remove-input]

# Show softmax with edge detection example - 3 step process
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Example scores for an edge image
scores = np.array([2.5, 0.5])  # [Edge score, No Edge score]
labels = ['Edge', 'No Edge']
colors = ['#2ecc71', '#a8e6cf']  # nicer greens

# Step 1: Raw scores (logits)
ax1 = axes[0]
bars1 = ax1.bar(labels, scores, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Value', fontsize=10)
ax1.set_ylim(0, 4)
for bar, val in zip(bars1, scores):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.1f}', 
             ha='center', fontsize=11, fontweight='bold')
ax1.set_title('Step 1: Raw Scores\n(Logits)', fontsize=11, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', ec='#f0c36d'), y=1.02)
ax1.text(0.5, -0.18, 'Can be any number', ha='center', transform=ax1.transAxes, 
         fontsize=9, style='italic', color='#666')
ax1.tick_params(axis='both', labelsize=9)

# Step 2: Exponentials
exp_scores = np.exp(scores)
ax2 = axes[1]
bars2 = ax2.bar(labels, exp_scores, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Value', fontsize=10)
ax2.set_ylim(0, 15)
for bar, val in zip(bars2, exp_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.4, f'{val:.1f}', 
             ha='center', fontsize=11, fontweight='bold')
ax2.set_title('Step 2: Exponentials\n($e^z$)', fontsize=11, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', ec='#f0c36d'), y=1.02)
ax2.text(0.5, -0.18, 'All positive now!', ha='center', transform=ax2.transAxes, 
         fontsize=9, style='italic', color='#666')
ax2.tick_params(axis='both', labelsize=9)

# Step 3: Normalize (the ratio)
probs = exp_scores / np.sum(exp_scores)
ax3 = axes[2]
bars3 = ax3.bar(labels, probs, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Probability', fontsize=10)
ax3.set_ylim(0, 1.15)
for bar, val in zip(bars3, probs):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.0%}', 
             ha='center', fontsize=11, fontweight='bold')
ax3.set_title('Step 3: Normalize\n(Divide by sum)', fontsize=11, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', ec='#f0c36d'), y=1.02)
ax3.text(0.5, -0.18, 'Sum = 1 (always!)', ha='center', transform=ax3.transAxes, 
         fontsize=9, style='italic', color='#666')
ax3.tick_params(axis='both', labelsize=9)

plt.suptitle('Softmax: From Scores to Probabilities', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)

# Add arrows between plots using figure coordinates (after tight_layout)
fig.text(0.328, 0.5, '→', fontsize=28, ha='center', va='center', fontweight='bold', color='#555')
fig.text(0.673, 0.5, '→', fontsize=28, ha='center', va='center', fontweight='bold', color='#555')

plt.show()
```

**Implementation:**

```python
# z is a vector of scores (logits) for all classes
z = np.array([2.5, 0.5])            # e.g., [Edge score, No-Edge score]

exp_z = np.exp(z)                   # exponentiate each element
p = exp_z / np.sum(exp_z)           # normalize → probability vector

print(f"Logits z: {z}")             # [2.5, 0.5]
print(f"exp(z):   {exp_z}")         # [12.18, 1.65]
print(f"Probs p:  {p}")             # [0.88, 0.12]
```

<br>

This computes all $p_j$ values at once: `p[j]` = $\frac{e^{z_j}}{\sum_k e^{z_k}}$

> **Note:** In practice, we subtract the max for numerical stability. This prevents overflow with large scores but gives the same result:
> ```python
> exp_z = np.exp(z - np.max(z))        # stable exponentiation
> p = exp_z / np.sum(exp_z)            # normalize
> ```

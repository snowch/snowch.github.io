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

This post is a standalone explanation of softmax that you can reference from any classifier walkthrough, whether it is an image model, a text model, or a simple toy example.

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

## Where Does Softmax Fit?

Before diving into how softmax works, let's see where it sits in a classification pipeline. Here's a simple edge detection network:

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'Where Does Softmax Fit? Example: Edge Detection Network',
        fontsize=14, fontweight='bold', ha='center')

# Define vertical center for alignment
vcenter = 3.6

# Input layer - 5x5 grid centered vertically
input_x = 1.5
ax.text(input_x, 6.5, 'Input', fontsize=11, fontweight='bold', ha='center')
ax.text(input_x, 6.0, '(25 pixels)', fontsize=9, ha='center', style='italic')

# Draw a 5x5 grid centered at vcenter
grid_size = 0.25
grid_height = 5 * grid_size
grid_start_y = vcenter + grid_height/2 - grid_size/2

for i in range(5):
    for j in range(5):
        rect = plt.Rectangle((input_x - 0.625 + j*grid_size, grid_start_y - i*grid_size),
                            grid_size*0.9, grid_size*0.9,
                            facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

# Hidden layers (abstract representation) - centered at vcenter
hidden_x = 5
ax.text(hidden_x, 6.5, 'Hidden Layers', fontsize=11, fontweight='bold', ha='center')
ax.text(hidden_x, 6.0, '(learns features)', fontsize=9, ha='center', style='italic')

# Box centered at vcenter
box_height = 2.5
rect_hidden = plt.Rectangle((hidden_x - 1, vcenter - box_height/2), 2, box_height,
                           facecolor='lightyellow', edgecolor='orange',
                           linewidth=2, alpha=0.3)
ax.add_patch(rect_hidden)
ax.text(hidden_x, vcenter, '•  •  •', fontsize=20, ha='center', va='center')

# Connection lines from input to hidden
ax.plot([input_x + 0.65, hidden_x - 1], [vcenter + 0.5, vcenter + 0.3],
        'gray', alpha=0.4, linewidth=2)
ax.plot([input_x + 0.65, hidden_x - 1], [vcenter - 0.5, vcenter - 0.3],
        'gray', alpha=0.4, linewidth=2)

# Output scores (THE KEY PART) - centered around vcenter
output_x = 9
ax.text(output_x, 6.5, 'Output Layer', fontsize=11, fontweight='bold', ha='center')
ax.text(output_x, 6.0, '(raw scores)', fontsize=9, ha='center', style='italic', color='#e67e22')

# Two output neurons with scores, centered at vcenter
neuron_spacing = 0.8
output_ys = [vcenter + neuron_spacing/2, vcenter - neuron_spacing/2]
output_labels = ['Edge', 'No Edge']
scores = [2.5, 0.5]
colors = ['#3498db', '#95a5a6']

for y, label, score, color in zip(output_ys, output_labels, scores, colors):
    circle = plt.Circle((output_x, y), 0.35, color=color, ec='black', linewidth=2, alpha=0.7)
    ax.add_patch(circle)
    ax.text(output_x - 0.9, y, label, fontsize=10, ha='right', va='center')
    ax.text(output_x + 0.6, y, f'{score}', fontsize=11, ha='left', va='center',
            fontweight='bold', family='monospace')

# Connection from hidden to output
ax.plot([hidden_x + 1, output_x - 0.35], [vcenter, output_ys[0]],
        'gray', alpha=0.4, linewidth=2)
ax.plot([hidden_x + 1, output_x - 0.35], [vcenter, output_ys[1]],
        'gray', alpha=0.4, linewidth=2)

# THE SOFTMAX OPERATION (highlighted)
softmax_x = 12.0
ax.text(softmax_x, 6.5, 'Softmax', fontsize=11, fontweight='bold', ha='center', color='green')
ax.text(softmax_x, 6.0, r'(scores $\rightarrow$ probs)', fontsize=9, ha='center',
        style='italic', color='green')

# Big arrow pointing to softmax operation - centered in whitespace
arrow_start = output_x + 0.9
arrow_end = softmax_x - 0.8
arrow_center = (arrow_start + arrow_end) / 2

ax.annotate('', xy=(arrow_end, vcenter), xytext=(arrow_start, vcenter),
           arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax.text(arrow_center, vcenter + 0.5, 'softmax', fontsize=10,
        ha='center', fontweight='bold', color='green',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', ec='green', linewidth=2))

# Result: Probabilities - same positions as output layer
probs = [0.88, 0.12]
for y, label, prob, color in zip(output_ys, output_labels, probs, colors):
    circle = plt.Circle((softmax_x, y), 0.35, color=color, ec='green', linewidth=2.5, alpha=0.9)
    ax.add_patch(circle)
    ax.text(softmax_x + 0.7, y, f'{prob:.0%}', fontsize=11, ha='left', va='center',
            fontweight='bold', family='monospace', color='green')

# Annotations - positioned below the neurons
ax.text(output_x, 1.8, 'Can be any number\n(positive or negative)',
        ha='center', fontsize=9, style='italic', color='#666',
        bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.7))

ax.text(softmax_x, 1.8, 'Always positive\nSum = 1.0',
        ha='center', fontsize=9, style='italic', color='#666',
        bbox=dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.7))

# Bottom note
ax.text(7, 0.5, 'This blog focuses on the softmax step: converting raw scores into probabilities',
        ha='center', fontsize=10, style='italic', color='#555',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', ec='gray', linewidth=1.5))

plt.tight_layout()
plt.show()
```

The key point: **Softmax operates on the final layer's raw scores**, converting them into probabilities that sum to 1. The rest of this post explains how softmax works, regardless of the network architecture.

---

## From Scores to Probabilities - Softmax

In a classification model, the final layer produces **scores** (also called logits) — higher means the model favors that class more. But for classification, we need **probabilities**: "What's the chance this is an edge?"

Suppose our model has two output neurons (one for "Edge", one for "No Edge"), each producing a score. So we have a **vector of scores** $\mathbf{z} = [z_{edge}, z_{no\_edge}]$. These raw scores can be any number — positive, negative, or zero.

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
- $z_j$ is the raw score for class $j$
- $p_j$ is the resulting probability for class $j$
- The sum is over all classes $k$

For example, scores $[3, 1]$ would give $p_0 = 3/(3+1) = 0.75$ and $p_1 = 1/(3+1) = 0.25$.

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

> **Note on Numerical Stability:**
> In practice, we subtract the max score before exponentiating to prevent overflow. This is safe because the constants cancel in the ratio:
>
> $$\frac{e^{z_j - c}}{\sum_k e^{z_k - c}} = \frac{e^{z_j}/e^{c}}{\sum_k (e^{z_k}/e^{c})} = \frac{e^{z_j}}{\sum_k e^{z_k}}$$
>
> So we can choose $c = \max(z)$ without changing the result:
> ```python
> exp_z = np.exp(z - np.max(z))        # stable: prevents overflow
> p = exp_z / np.sum(exp_z)            # same result as before
> ```

### Multi-Class Example

Softmax works for any number of classes, not just two. Here's a 3-class example:

```python
# Multi-class classification: Cat, Dog, Bird
z = np.array([3.2, 1.3, 0.2])           # scores for each class

exp_z = np.exp(z - np.max(z))           # stable computation
p = exp_z / np.sum(exp_z)               # normalize

print(f"Logits:  {z}")                  # [3.2, 1.3, 0.2]
print(f"Probs:   {p}")                  # [0.70, 0.24, 0.06]
print(f"Sum:     {p.sum():.1f}")        # 1.0

# Interpretation: 70% Cat, 24% Dog, 6% Bird
```

The highest score (3.2 for Cat) gets the highest probability (70%), but the other classes still have non-zero probabilities. This is useful for understanding model confidence and handling uncertain predictions.

---

## Common Pitfalls

<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0;">

**⚠️ Things to Watch Out For**

**1. Don't apply softmax to already-normalized outputs**
- If your network already outputs probabilities (sum to 1), softmax is redundant
- Example: Don't apply softmax after another softmax layer

**2. Softmax is for classification, not regression**
- Use softmax when predicting categories (cat, dog, bird)
- Don't use it for continuous values (e.g., predicting temperature or price)
- For regression, use raw outputs or other activation functions

**3. Softmax is differentiable**
- This is crucial for training neural networks with backpropagation
- The gradient flows through softmax during training
- You typically use cross-entropy loss with softmax for classification

**4. Temperature scaling**
- Dividing logits by temperature $T$ before softmax affects confidence:
  - $T > 1$: Less confident, more uniform probabilities
  - $T < 1$: More confident, sharper probabilities
  - Default $T = 1$ (standard softmax)

</div>

---

## Summary

**Softmax** converts raw scores (logits) into probabilities:
- **Input:** Raw scores that can be any number
- **Output:** Probabilities between 0 and 1 that sum to 1
- **Method:** Exponentiate to make positive, then normalize

**Key formula:** $p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$

This makes softmax the standard final layer activation for multi-class classification in neural networks.

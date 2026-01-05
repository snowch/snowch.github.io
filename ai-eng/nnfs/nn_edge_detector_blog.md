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

# NN01 - Edge Detection Intuition: A Single Neuron as Pattern Matching

*A hands-on guide to understanding how a single neuron detects patterns in images*

---

Have you ever wondered how neural networks can recognize faces, read handwriting, or detect objects in photos? It all starts with something surprisingly simple: **pattern matching**.

In this post, we'll build an edge detector from scratch to understand the fundamental operation at the heart of all neural networks. By the end, you'll have an intuitive grasp of:

- How a single neuron works (it's just multiplication and addition!)
- Why **weights** determine what patterns a neuron responds to
- How **ReLU activation** acts as a gate
- Why **bias** matters for controlling sensitivity

No prior deep learning knowledge required — just basic math intuition.

+++

## The Big Picture: What Does a Neuron Actually Do?

A neuron in a neural network does three things:

1. **Multiply** each input by a learned weight
2. **Add** all the products together (plus a bias term)
3. **Apply an activation function** (we'll use ReLU)

Mathematically:

$$z = \sum_{i} x_i \cdot w_i + b = (x_1 \cdot w_1) + (x_2 \cdot w_2) + \ldots + (x_n \cdot w_n) + b$$

$$\text{output} = \text{ReLU}(z) = \max(0, z)$$

That's it! The magic is in *what values the weights take* — they determine what pattern the neuron responds to.

**Intuition:** Think of each input as an expert's opinion, and the weights as how much you trust each expert. A large positive weight means "this input is very important." A negative weight means "if this input is high, I'm *less* interested."

```{code-cell} ipython3
:tags: [remove-input]

# Setup
import numpy as np

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
```

```{code-cell} ipython3
:tags: [remove-input]

def weighted_sum(inputs, weights, bias=0):
    """Compute the weighted sum: z = sum(x * w) + b"""
    return np.sum(inputs * weights) + bias

def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, z)

def neuron_output(inputs, weights, bias=0):
    """Complete neuron: weighted sum followed by ReLU"""
    z = weighted_sum(inputs, weights, bias)
    return relu(z), z
```

## What is ReLU and Why Do We Need It?

**ReLU** (Rectified Linear Unit) is the simplest activation function:

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

It acts as a **gate**: positive signals pass through unchanged, negative signals get blocked.

**Why is this useful?** Without an activation function, stacking layers would be pointless — the whole network would collapse into a single linear transformation. ReLU introduces *non-linearity*, which allows the network to learn complex patterns.

```{code-cell} ipython3
:tags: [remove-input]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x = np.linspace(-5, 5, 100)
y_relu = np.maximum(0, x)

axes[0].plot(x, x, 'b--', alpha=0.5, label='Identity (no activation)')
axes[0].plot(x, y_relu, 'r-', linewidth=2.5, label='ReLU')
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].fill_between(x[x<0], 0, x[x<0], alpha=0.2, color='red', label='Blocked region')
axes[0].set_xlabel('Input (z)', fontsize=12)
axes[0].set_ylabel('Output', fontsize=12)
axes[0].set_title('ReLU: The Simplest Activation Function', fontsize=13)
axes[0].legend()
axes[0].set_xlim(-5, 5)
axes[0].set_ylim(-2, 5)

sample_z = [-2.5, -1, 0, 1, 2.5]
sample_relu = [max(0, z) for z in sample_z]
colors = ['#d62728' if z <= 0 else '#2ca02c' for z in sample_z]

axes[1].bar(range(len(sample_z)), sample_relu, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_xticks(range(len(sample_z)))
axes[1].set_xticklabels([f'z={z}' for z in sample_z])
axes[1].set_ylabel('ReLU(z)', fontsize=12)
axes[1].set_title('ReLU Blocks Negative Values', fontsize=13)

for i, (z, out) in enumerate(zip(sample_z, sample_relu)):
    label = f'{out}' if out > 0 else 'blocked!'
    axes[1].annotate(label, (i, out + 0.15), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
```

## Building an Edge Detector

Now let's see how a neuron can detect a specific pattern: a **vertical edge**.

Imagine a tiny 5×5 grayscale image (25 pixels). A neuron connected to this image has:
- **25 inputs** (one per pixel, values 0–1 where 0=black, 1=white)
- **25 weights** (one per connection)
- **1 output** (after weighted sum + ReLU)

### Designing the Weights

To detect a vertical edge (dark on left, bright on right), we set:
- **Negative weights (-1)** on the left columns → "I want darkness here"
- **Positive weights (+1)** on the right columns → "I want brightness here"
- **Zero weights (0)** in the middle → "I don't care about this region"

When an image with this exact pattern arrives, the positive and negative contributions reinforce each other, giving a high score. When the pattern doesn't match, contributions cancel out or go negative.

```{code-cell} ipython3
:tags: [remove-input]

# The key insight: these weights define WHAT the neuron looks for
edge_weights = np.array([
    [-1, -1, 0, +1, +1],
    [-1, -1, 0, +1, +1],
    [-1, -1, 0, +1, +1],
    [-1, -1, 0, +1, +1],
    [-1, -1, 0, +1, +1]
], dtype=float)
```

```{code-cell} ipython3
:tags: [remove-input]

fig, ax = plt.subplots(figsize=(7, 6))
cmap = LinearSegmentedColormap.from_list('edge', ['#d62728', 'white', '#1f77b4'])
im = ax.imshow(edge_weights, cmap=cmap, vmin=-1, vmax=1)

for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{edge_weights[i, j]:+.0f}', ha='center', va='center', fontsize=16, fontweight='bold')

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_title('Vertical Edge Detector Weights\n(Red = -1 "want dark", Blue = +1 "want bright")', fontsize=13)
plt.colorbar(im, ax=ax, label='Weight Value', shrink=0.8)
plt.tight_layout()
plt.show()
```

## Testing Our Detector on Different Images

Let's create several test images to see how our detector responds:

| Image | Description | Expected Response |
|-------|-------------|-------------------|
| Perfect Edge | Dark left, bright right | **Strong** (pattern matches!) |
| Shifted Edge | Edge in wrong position | **Reduced** (partial match) |
| Uniform Gray | No edge at all | **Zero** (no contrast) |
| Inverted Edge | Bright left, dark right | **Blocked** (opposite of what we want) |
| Horizontal Edge | Edge rotates 90° | **Weak** (wrong orientation) |

```{code-cell} ipython3
:tags: [remove-input]

# Test images
perfect_edge = np.array([[0.1, 0.1, 0.5, 0.9, 0.9]] * 5)
edge_shifted_left = np.array([[0.1, 0.5, 0.9, 0.9, 0.9]] * 5)
edge_shifted_right = np.array([[0.1, 0.1, 0.1, 0.5, 0.9]] * 5)
uniform_gray = np.full((5, 5), 0.5)
inverted_edge = np.array([[0.9, 0.9, 0.5, 0.1, 0.1]] * 5)
horizontal_edge = np.array([[0.1]*5, [0.1]*5, [0.5]*5, [0.9]*5, [0.9]*5])
diagonal_edge = np.array([[0.1, 0.1, 0.1, 0.5, 0.9],
                          [0.1, 0.1, 0.5, 0.9, 0.9],
                          [0.1, 0.5, 0.9, 0.9, 0.9],
                          [0.5, 0.9, 0.9, 0.9, 0.9],
                          [0.9, 0.9, 0.9, 0.9, 0.9]])

test_images = [
    ("Perfect Edge", perfect_edge),
    ("Edge Shifted Left", edge_shifted_left),
    ("Edge Shifted Right", edge_shifted_right),
    ("Uniform Gray", uniform_gray),
    ("Inverted Edge", inverted_edge),
    ("Horizontal Edge", horizontal_edge),
    ("Diagonal Edge", diagonal_edge)
]
```

```{code-cell} ipython3
:tags: [remove-input]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

cmap_weights = LinearSegmentedColormap.from_list('edge', ['#d62728', 'white', '#1f77b4'])
axes[0].imshow(edge_weights, cmap=cmap_weights, vmin=-1, vmax=1)
axes[0].set_title('DETECTOR\nWEIGHTS', fontsize=12, fontweight='bold', color='#1f77b4')
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f'{edge_weights[i, j]:+.0f}', ha='center', va='center', fontsize=10)

for idx, (name, img) in enumerate(test_images):
    axes[idx + 1].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[idx + 1].set_title(name, fontsize=11)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Test Images for Our Edge Detector', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Computing the Neuron's Response

For each image, we compute:

$$z = \sum_{i,j} \text{image}[i,j] \times \text{weight}[i,j] + b$$

$$\text{output} = \max(0, z)$$

> **Note:** We're setting **bias $b = 0$** for now to keep things simple. This lets us focus purely on how the weight pattern matches the input. We'll explore bias later!

```{code-cell} ipython3
:tags: [remove-input]

def analyze_response(image, weights, name):
    products = image * weights
    z = np.sum(products)
    output = max(0, z)
    return {'name': name, 'image': image, 'products': products, 'weighted_sum': z, 'relu_output': output}

results = [analyze_response(img, edge_weights, name) for name, img in test_images]
```

```{code-cell} ipython3
:tags: [remove-input]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = [r['name'] for r in results]
weighted_sums = [r['weighted_sum'] for r in results]
relu_outputs = [r['relu_output'] for r in results]

colors_ws = ['#2ca02c' if ws > 0 else '#d62728' for ws in weighted_sums]
axes[0].barh(names, weighted_sums, color=colors_ws, alpha=0.8, edgecolor='black')
axes[0].axvline(x=0, color='black', linewidth=2)
axes[0].set_xlabel('Weighted Sum (z)', fontsize=12)
axes[0].set_title('Step 1: Weighted Sum\n(before ReLU)', fontsize=13, fontweight='bold')
for i, v in enumerate(weighted_sums):
    axes[0].text(v + (0.3 if v >= 0 else -0.8), i, f'{v:.1f}', va='center', fontsize=10, fontweight='bold')

colors_relu = ['#2ca02c' if o > 0 else '#999999' for o in relu_outputs]
axes[1].barh(names, relu_outputs, color=colors_relu, alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Output', fontsize=12)
axes[1].set_title('Step 2: After ReLU\n(negative values blocked)', fontsize=13, fontweight='bold')
for i, (v, ws) in enumerate(zip(relu_outputs, weighted_sums)):
    label = f'{v:.1f}' if v > 0 else f'blocked (was {ws:.1f})'
    axes[1].text(v + 0.3, i, label, va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
```

## Visualizing the Math: Step by Step

Let's look inside the calculation to see exactly how the score emerges from element-wise multiplication.

```{code-cell} ipython3
:tags: [remove-input]

def plot_detailed_analysis(result, weights):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(result['image'], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Input: {result['name']}", fontsize=12, fontweight='bold')
    for i in range(5):
        for j in range(5):
            color = 'white' if result['image'][i, j] < 0.5 else 'black'
            axes[0].text(j, i, f'{result["image"][i, j]:.1f}', ha='center', va='center', fontsize=9, color=color)
    
    cmap_w = LinearSegmentedColormap.from_list('edge', ['#d62728', 'white', '#1f77b4'])
    axes[1].imshow(weights, cmap=cmap_w, vmin=-1, vmax=1)
    axes[1].set_title('× Weights', fontsize=12)
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, f'{weights[i, j]:+.0f}', ha='center', va='center', fontsize=10)
    
    products = result['products']
    max_abs = max(abs(products.min()), abs(products.max()), 0.1)
    axes[2].imshow(products, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title('= Products (pixel × weight)', fontsize=12)
    for i in range(5):
        for j in range(5):
            axes[2].text(j, i, f'{products[i, j]:+.2f}', ha='center', va='center', fontsize=8)
    
    axes[3].axis('off')
    if result['relu_output'] > 6:
        verdict, verdict_color = "✓ STRONG match!", '#2ca02c'
    elif result['relu_output'] > 2:
        verdict, verdict_color = "~ Partial match", '#ff7f0e'
    elif result['relu_output'] > 0:
        verdict, verdict_color = "~ Weak match", '#ff7f0e'
    else:
        verdict, verdict_color = "✗ Blocked by ReLU", '#d62728'
    
    summary = f"Sum of products:\nz = {result['weighted_sum']:.2f}\n\nAfter ReLU:\nmax(0, {result['weighted_sum']:.2f}) = {result['relu_output']:.2f}\n\n{verdict}"
    axes[3].text(0.1, 0.5, summary, fontsize=13, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=verdict_color, linewidth=2))
    
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

for result in [results[0], results[4], results[3]]:
    plot_detailed_analysis(result, edge_weights)
```

## The Flattened View: It's Just a Dot Product!

The 5×5 matrix representation is convenient for visualization, but the actual computation is simply a **dot product** of two 25-element vectors:

$$z = \vec{x} \cdot \vec{w} = x_0 w_0 + x_1 w_1 + \ldots + x_{24} w_{24}$$

Let's see what this looks like when we "unroll" the matrices into strips:

```{code-cell} ipython3
:tags: [remove-input]

def plot_flattened_view(image, weights, name):
    img_flat = image.flatten()
    weights_flat = weights.flatten()
    products_flat = img_flat * weights_flat
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 5))
    cmap_gray = plt.cm.gray
    cmap_weights = LinearSegmentedColormap.from_list('edge', ['#d62728', 'white', '#1f77b4'])
    
    img_colors = cmap_gray(img_flat)
    axes[0].bar(range(25), np.ones(25), color=img_colors, edgecolor='black', linewidth=0.5, width=1.0)
    axes[0].set_xlim(-0.5, 24.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Image (x)', fontsize=10)
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    for i, v in enumerate(img_flat):
        axes[0].text(i, 0.5, f'{v:.1f}', ha='center', va='center', fontsize=7, color='white' if v < 0.5 else 'black')
    axes[0].set_title(f'{name}: Flattened into 25-element vectors', fontsize=12, fontweight='bold')
    
    weight_colors = [cmap_weights((w + 1) / 2) for w in weights_flat]
    axes[1].bar(range(25), np.ones(25), color=weight_colors, edgecolor='black', linewidth=0.5, width=1.0)
    axes[1].set_xlim(-0.5, 24.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Weights (w)', fontsize=10)
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    for i, v in enumerate(weights_flat):
        axes[1].text(i, 0.5, f'{v:+.0f}', ha='center', va='center', fontsize=8, fontweight='bold')
    
    max_abs = max(abs(products_flat.min()), abs(products_flat.max()), 0.1)
    product_colors = [plt.cm.RdBu((p / max_abs + 1) / 2) for p in products_flat]
    axes[2].bar(range(25), np.ones(25), color=product_colors, edgecolor='black', linewidth=0.5, width=1.0)
    axes[2].set_xlim(-0.5, 24.5)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('x × w', fontsize=10)
    axes[2].set_yticks([])
    axes[2].set_xticks([])
    for i, v in enumerate(products_flat):
        axes[2].text(i, 0.5, f'{v:+.1f}', ha='center', va='center', fontsize=6)
    
    for i in range(1, 5):
        for ax in axes[:3]:
            ax.axvline(x=i*5-0.5, color='black', linewidth=2)
    
    axes[3].axis('off')
    z = np.sum(products_flat)
    axes[3].text(0.5, 0.5, f'z = Σ(x·w) = {z:.2f}    →    ReLU(z) = {max(0, z):.2f}',
                ha='center', va='center', fontsize=14, fontweight='bold', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    plt.tight_layout()
    plt.show()

plot_flattened_view(perfect_edge, edge_weights, "Perfect Edge")
plot_flattened_view(inverted_edge, edge_weights, "Inverted Edge")
```

## Key Insight: Alignment Determines Response

The neuron's output depends entirely on **how well the input aligns with the weight pattern**:

| Scenario | What Happens | Result |
|----------|--------------|--------|
| **Perfect match** | Dark pixels × negative weights → positive contributions<br>Bright pixels × positive weights → positive contributions<br>All reinforce! | **High output** |
| **Inverted pattern** | Dark pixels × positive weights → small contributions<br>Bright pixels × negative weights → negative contributions | **Negative → blocked** |
| **No pattern** | Everything averages out | **~Zero** |

+++

## The Role of Bias: Adjusting Sensitivity

So far we've set bias $b = 0$. But what does bias actually do?

- **Weights** control *what pattern* the neuron responds to
- **Bias** controls *how easily* the neuron activates — it shifts the threshold

$$z = \sum(x \cdot w) + b$$

- **Positive bias** → Easier to activate (even weak matches pass through)
- **Negative bias** → Harder to activate (only strong matches pass through)

Think of bias as the neuron's "baseline mood" — a positive bias means it's eager to fire, while a negative bias means it needs more convincing.

```{code-cell} ipython3
:tags: [remove-input]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

biases = np.arange(-4, 5)
z_base_uniform = np.sum(uniform_gray * edge_weights)
outputs_uniform = [max(0, z_base_uniform + b) for b in biases]

colors = ['#2ca02c' if o > 0 else '#d62728' for o in outputs_uniform]
axes[0].bar(biases, outputs_uniform, color=colors, edgecolor='black', alpha=0.8)
axes[0].axhline(y=0, color='black', linewidth=1)
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=2, label='No bias')
axes[0].set_xlabel('Bias Value', fontsize=12)
axes[0].set_ylabel('ReLU Output', fontsize=12)
axes[0].set_title('Uniform Gray Image\n(weighted sum = 0 without bias)', fontsize=13, fontweight='bold')
axes[0].legend()

subtle_edge = np.array([[0.4, 0.4, 0.5, 0.6, 0.6]] * 5)
z_base_subtle = np.sum(subtle_edge * edge_weights)
outputs_subtle = [max(0, z_base_subtle + b) for b in biases]

colors = ['#2ca02c' if o > 0 else '#d62728' for o in outputs_subtle]
axes[1].bar(biases, outputs_subtle, color=colors, edgecolor='black', alpha=0.8)
axes[1].axhline(y=0, color='black', linewidth=1)
axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=2, label='No bias')
axes[1].set_xlabel('Bias Value', fontsize=12)
axes[1].set_ylabel('ReLU Output', fontsize=12)
axes[1].set_title(f'Subtle Edge (low contrast)\n(weighted sum = {z_base_subtle:.1f} without bias)', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()
```

Notice how:
- With **negative bias**, even the subtle edge gets blocked
- With **positive bias**, weak signals (or even no signal!) can activate the neuron

In a trained network, each neuron learns its own optimal bias along with its weights.

+++

## Different Detectors for Different Patterns

Our vertical edge detector is just one example. By changing the weights, we can detect completely different patterns:

```{code-cell} ipython3
:tags: [remove-input]

detectors = {
    'Vertical Edge': np.array([[-1, -1, 0, +1, +1]] * 5, dtype=float),
    'Horizontal Edge': np.array([[-1]*5, [-1]*5, [0]*5, [+1]*5, [+1]*5], dtype=float),
    'Diagonal Edge': np.array([[-1, -1, -1, 0, +1], [-1, -1, 0, +1, +1], [-1, 0, +1, +1, +1], [0, +1, +1, +1, +1], [+1, +1, +1, +1, +1]], dtype=float),
    'Center Spot': np.array([[-1, -1, -1, -1, -1], [-1, +1, +1, +1, -1], [-1, +1, +2, +1, -1], [-1, +1, +1, +1, -1], [-1, -1, -1, -1, -1]], dtype=float) / 2,
}

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
cmap_w = LinearSegmentedColormap.from_list('edge', ['#d62728', 'white', '#1f77b4'])

for idx, (name, weights) in enumerate(detectors.items()):
    axes[0, idx].imshow(weights, cmap=cmap_w, vmin=-1, vmax=1)
    axes[0, idx].set_title(name, fontsize=11, fontweight='bold')
    axes[0, idx].set_xticks([])
    axes[0, idx].set_yticks([])

axes[0, 0].set_ylabel('Detector\nWeights', fontsize=12, fontweight='bold')

for idx, (name, weights) in enumerate(detectors.items()):
    z = np.sum(perfect_edge * weights)
    output = max(0, z)
    color = '#2ca02c' if output > 3 else '#ff7f0e' if output > 0 else '#d62728'
    axes[1, idx].bar(['Response'], [output], color=color, edgecolor='black', linewidth=2)
    axes[1, idx].set_ylim(0, 10)
    axes[1, idx].set_title(f'z={z:.1f} → {output:.1f}', fontsize=11)

axes[1, 0].set_ylabel('Response to\nVertical Edge', fontsize=12, fontweight='bold')
plt.suptitle('Different Detectors Respond to Different Patterns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## From Single Neurons to Deep Networks

A real neural network has thousands or millions of neurons arranged in layers:

- **Layer 1** neurons learn simple patterns (edges, colors, textures)
- **Layer 2** neurons combine Layer 1 outputs to detect shapes (curves, corners)
- **Layer 3** neurons combine shapes into parts (eyes, wheels, windows)
- **Output layer** combines parts into final classifications (cat, car, digit)

This hierarchy — from simple to complex — is why deep learning works so well. And it all starts with the simple pattern matching we've explored here!

**The key insight:** We don't design these weights by hand. During training, the network automatically discovers what patterns are useful for the task. Backpropagation adjusts each weight to reduce errors, and useful detectors emerge naturally.

+++

## Try It Yourself!

Modify the image below and re-run to see how the detector responds:

```{code-cell} ipython3
:tags: [remove-input]

# Try modifying this image!
my_image = np.array([
    [0.0, 0.0, 0.5, 1.0, 1.0],
    [0.0, 0.0, 0.5, 1.0, 1.0],
    [0.0, 0.0, 0.5, 1.0, 1.0],
    [0.0, 0.0, 0.5, 1.0, 1.0],
    [0.0, 0.0, 0.5, 1.0, 1.0]
])

my_result = analyze_response(my_image, edge_weights, "My Custom Image")
plot_detailed_analysis(my_result, edge_weights)
```

---

## Summary

We've built an edge detector from scratch and discovered the core principles that power all neural networks:

| Concept | What It Does | Analogy |
|---------|--------------|----------|
| **Weights** | Define what pattern the neuron detects | A template to match against |
| **Weighted Sum** | Measure how well input matches the pattern | A similarity score |
| **Bias** | Shift the activation threshold | The neuron's "eagerness" to fire |
| **ReLU** | Block negative responses | A gate that only passes matches |

### The Big Ideas

1. **A neuron is a pattern matcher.** High output = good match, zero output = poor match.

2. **Weights encode knowledge.** The specific values determine what the neuron "looks for."

3. **Learning = finding good weights.** Training adjusts weights until useful patterns emerge.

4. **Depth creates hierarchy.** Simple detectors combine into complex recognizers.

This simple mechanism — multiply, sum, activate — repeated billions of times with learned weights, is how neural networks learn to see, read, translate, and even generate art.

---

*Now you understand the atom of deep learning. Everything else is scale and clever architecture!*

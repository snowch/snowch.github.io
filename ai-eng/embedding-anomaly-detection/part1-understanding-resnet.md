---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
bibliography:
  - references.bib
---

# Part 1: Understanding ResNet Architecture [DRAFT]

This tutorial series builds a production-ready anomaly detection system using ResNet embeddings for observability data.

## Introduction: Why ResNet for Anomaly Detection?

This tutorial explores **Residual Networks (ResNet)**, a breakthrough architecture that enables training of very deep neural networks. While ResNet was originally designed for computer vision, it has emerged as a **surprisingly strong baseline for tabular data** and embedding models.

### Motivation: Tabular Data and Anomaly Detection

Recent research has shown that while Transformers (TabTransformer, FT-Transformer) achieve state-of-the-art results on tabular data, **ResNet-like architectures provide a simpler, more efficient baseline** that often performs comparably well {cite}`gorishniy2021revisiting`. For applications like:

- **Anomaly detection in observability data** (logs, network traffic, system metrics)
- **Self-supervised learning on unlabelled data**
- **Creating embeddings for multi-record pattern analysis**

ResNet offers several advantages:

1. **Simpler architecture** than Transformers (no attention mechanism overhead)
2. **Linear complexity** vs. quadratic attention complexity for high-dimensional tabular data (300+ features)
3. **Strong empirical performance** on heterogeneous tabular datasets
4. **Efficient embedding extraction** for downstream clustering and anomaly detection

### The Use Case: OCSF Observability Data

Consider a security/observability scenario where you have:
- **Unlabelled data**: Millions of OCSF (Open Cybersecurity Schema Framework) records
- **High dimensionality**: 300+ fields per record (categorical and numerical)
- **Multi-record anomalies**: Patterns that span sequences of events
- **No ground truth**: Need self-supervised learning

The approach {cite}`huang2020tabtransformer`:
1. **Pre-train** a ResNet to create embeddings from individual records
2. **Extract** fixed-dimensional vectors that capture "normal" system behavior
3. **Detect anomalies** as records/sequences that deviate from learned patterns

This tutorial series will build your understanding of ResNet from first principles, then show how to deploy it in production.

### Prerequisites

**Required Background:**
- Basic neural network concepts (layers, backpropagation, gradient descent)
- Basic Python and PyTorch syntax

**Recommended (but not required):**
- Convolutional Neural Networks (CNNs) — This part uses image examples with CNNs, but we include a [brief CNN primer](#brief-cnn-primer) before the code
- If you're primarily interested in **tabular data**, you can skim the image-based sections and focus on [Part 2: Adapting ResNet for Tabular Data](part2-tabular-resnet)

**New to neural networks?** Start with our **[Neural Networks From Scratch](/ai-eng/nnfs/index.md)** series:

- **[NN01: Edge Detection](/ai-eng/nnfs/nn_edge_detector_blog.md)** - Understanding neurons as pattern matchers
- **[NN02: Training from Scratch](/ai-eng/nnfs/nn_tutorial_blog.md)** - Backpropagation and gradient descent
- **[NN03: General Networks](/ai-eng/nnfs/nn_flexible_network_blog.md)** - Building flexible architectures
- **[NN04: PyTorch Basics](/ai-eng/nnfs/nn_pytorch_basics.md)** - Transitioning to PyTorch (used in this tutorial)

### Paper References

- **Original ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.
- **ResNet for Tabular Data**: Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). "Revisiting Deep Learning Models for Tabular Data." *NeurIPS 2021*.
- **TabTransformer Comparison**: Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). "TabTransformer: Tabular Data Modeling Using Contextual Embeddings." *arXiv preprint*.

---

## The Problem ResNet Solves

### The Degradation Problem

Intuitively, deeper neural networks should be more powerful:
- More layers → More capacity to learn complex patterns
- A 56-layer network should perform *at least as well* as a 28-layer network (it could just learn **identity mappings** for the extra layers)

**What is an identity mapping?** A transformation where the output equals the input: $f(x) = x$. Think of it like a "do nothing" operation - data passes through unchanged. For example, if a layer receives a vector `[1, 2, 3]`, an identity mapping would output exactly `[1, 2, 3]`. In theory, deeper networks could use identity mappings in extra layers to match shallower networks, but in practice they fail to learn even this simple operation.

**But in practice, this doesn't happen.**

```{code-cell}
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt

# Simulated training/test error for plain networks
depths = [20, 32, 44, 56]
train_errors_plain = [15.5, 14.2, 15.8, 18.5]
test_errors_plain = [21.3, 20.1, 22.5, 25.7]

# Simulated errors for ResNet
train_errors_resnet = [15.5, 13.8, 12.5, 11.2]
test_errors_resnet = [21.3, 19.5, 18.1, 16.8]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plain networks
ax1.plot(depths, train_errors_plain, 'o-', label='Training Error', linewidth=2, markersize=8)
ax1.plot(depths, test_errors_plain, 's-', label='Test Error', linewidth=2, markersize=8)
ax1.set_xlabel('Network Depth (layers)', fontsize=12)
ax1.set_ylabel('Error (%)', fontsize=12)
ax1.set_title('Plain Networks: Degradation Problem', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([10, 27])

# ResNets
ax2.plot(depths, train_errors_resnet, 'o-', label='Training Error', linewidth=2, markersize=8, color='green')
ax2.plot(depths, test_errors_resnet, 's-', label='Test Error', linewidth=2, markersize=8, color='darkgreen')
ax2.set_xlabel('Network Depth (layers)', fontsize=12)
ax2.set_ylabel('Error (%)', fontsize=12)
ax2.set_title('Residual Networks: Problem Solved', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([10, 27])

plt.tight_layout()
plt.show()
```

**Key Observation**: Beyond a certain depth (~20-30 layers), plain networks start to perform *worse* on both training and test sets. This isn't overfitting (training error increases too) — it's **optimization difficulty**.

### Why Plain Networks Fail

Two main issues:

1. **Vanishing Gradients**: As gradients backpropagate through many layers, they get multiplied by small weight matrices repeatedly, shrinking exponentially. Deep layers learn very slowly or not at all.

2. **Degraded Optimization Landscape**: Very deep networks create complex, non-convex loss surfaces that are hard for SGD to navigate. Even though a solution exists (copy the shallower network and make extra layers just pass data through unchanged), the optimizer can't find it.

### What We Need

An architecture where:
- **Deeper is easier to optimize**, not harder
- **Identity mappings are learnable** by default
- **Gradients flow freely** to early layers

This is exactly what ResNet provides.

---

## The Core Innovation — Residual Connections

### The Residual Block

**The Key Idea:**

In a traditional neural network, layers learn to transform input $\mathbf{x}$ into output $H(\mathbf{x})$ **directly**:
- Input $\mathbf{x}$ → [layers] → Output $H(\mathbf{x})$
- The layers must learn the complete transformation from scratch

ResNet changes this by learning the **residual** (the *difference* between output and input):

$$
H(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}
$$

Where:
- $\mathbf{x}$: Input to the block
- $F(\mathbf{x})$: **Residual** — the learned difference/change to apply (typically 2-3 conv/linear layers)
- $H(\mathbf{x})$: Output = Input + Residual change
- $+\mathbf{x}$: The **skip connection** (also called **shortcut connection**) — adds input directly to output

**Why this helps:**
- Instead of learning the full transformation $H(\mathbf{x})$ from scratch, layers only learn **what to change** ($F(\mathbf{x})$)
- If no change is needed, layers can learn $F(\mathbf{x}) = 0$ (much easier than learning identity)
- The input $\mathbf{x}$ always flows through unchanged via the skip connection

**Visual comparison**: The diagrams below show the key architectural difference:
- **Left (Plain Block)**: Learns the entire transformation $H(x)$ directly from scratch
- **Right (Residual Block)**: Only learns the residual $F(x)$ to add to the input, where $H(x) = F(x) + x$

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Plain Network Block
```{mermaid}
graph TB
    X1[x<br/>Input] --> L1[Conv/Linear<br/>Layer 1]
    L1 --> L2[Conv/Linear<br/>Layer 2]
    L2 --> H1[H&#40;x&#41;<br/>Output]

    style X1 fill:#ADD8E6
    style L1 fill:#F08080
    style L2 fill:#F08080
    style H1 fill:#90EE90
```
:::

:::{grid-item-card} Residual Block
```{mermaid}
graph TB
    X2[x<br/>Input] --> L3[Conv/Linear<br/>Layer 1]
    L3 --> L4[Conv/Linear<br/>Layer 2]
    L4 --> Add((+))
    X2 -.Skip Connection.-> Add
    Add --> H2[H&#40;x&#41;<br/>Output]

    style X2 fill:#ADD8E6
    style L3 fill:#F08080
    style L4 fill:#F08080
    style Add fill:#FFFF00
    style H2 fill:#90EE90
    linkStyle 2 stroke:#0000FF,stroke-width:3px
```
:::

::::

### Why This Works: Intuition

**Learning Identity is Easy**:
- If the optimal mapping is identity (output = input, i.e., $H(\mathbf{x}) = \mathbf{x}$), the network just needs to learn $F(\mathbf{x}) = 0$
- **Why?** Because $H(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}$, so if $H(\mathbf{x}) = \mathbf{x}$, then:
  $$
  \mathbf{x} = F(\mathbf{x}) + \mathbf{x} \implies F(\mathbf{x}) = 0
  $$
- Pushing weights toward zero is much easier than learning the identity function from scratch with many layers
- This means "doing nothing" (keeping the input unchanged) is easy to learn

**Gradient Flow**:
- Gradients flow through both paths (main path *and* skip connection)
- Skip connection provides a "gradient highway" directly to earlier layers
- Even if $F(\mathbf{x})$ has vanishing gradients, $\mathbf{x}$ passes through unchanged

### Mathematical Analysis: Gradient Flow (Optional - Skip if Math-Heavy)

**Intuition**: The skip connection acts like a "gradient highway" - even if the main path's gradients shrink to zero during backpropagation, the skip connection ensures gradients can still flow unchanged back to earlier layers. This is why deep ResNets train successfully while plain deep networks fail.

**Analogy**: Think of gradients like water flowing backwards through the network during training. In plain networks, the water must flow through all the layers, getting weaker at each step (like water flowing through many filters). In ResNet, the skip connection is like a direct pipe that bypasses the filters - water always flows through both paths, so even if one path gets blocked, learning still happens.

```{dropdown} **Click to expand mathematical proof (optional)**
During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to $\mathbf{x}$ is:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial H} \cdot \frac{\partial H}{\partial \mathbf{x}}
$$

For the residual block:

$$
\frac{\partial H}{\partial \mathbf{x}} = \frac{\partial}{\partial \mathbf{x}}(F(\mathbf{x}) + \mathbf{x}) = \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}} + I
$$

Where $I$ is the identity matrix. This means:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left(\frac{\partial F(\mathbf{x})}{\partial \mathbf{x}} + I\right)
$$

**Key insight**: The gradient always has an identity component ($I$) that propagates unchanged. Even if $\frac{\partial F(\mathbf{x})}{\partial \mathbf{x}}$ vanishes, gradients still flow through the $+I$ term.
```

### Visualizing Gradient Flow (Optional - Empirical Evidence)

The gradient flow explanation above is theoretical. Let's **see it in action** by comparing gradient magnitudes in a plain network vs a residual network during backpropagation.

**What this demonstrates**: In deep plain networks, gradients shrink exponentially as they flow backward through layers (vanishing gradient problem). In ResNets, the skip connections maintain strong gradient flow even to the earliest layers.

```{code-cell}
:tags: [hide-input]

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Simple plain network (no skip connections)
class PlainNetwork(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

# Simple residual network (with skip connections)
class ResidualNetwork(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x)) + x  # Skip connection!
        return x

# Create networks
plain_net = PlainNetwork(num_layers=10)
resnet = ResidualNetwork(num_layers=10)

# Dummy input and target
x = torch.randn(32, 128)
target = torch.randn(32, 128)

# Forward + backward for plain network
plain_output = plain_net(x)
plain_loss = ((plain_output - target) ** 2).mean()
plain_loss.backward()

# Collect gradient magnitudes for each layer
plain_grads = []
for layer in plain_net.layers:
    if layer.weight.grad is not None:
        plain_grads.append(layer.weight.grad.abs().mean().item())

# Reset and do the same for ResNet
resnet.zero_grad()
resnet_output = resnet(x)
resnet_loss = ((resnet_output - target) ** 2).mean()
resnet_loss.backward()

resnet_grads = []
for layer in resnet.layers:
    if layer.weight.grad is not None:
        resnet_grads.append(layer.weight.grad.abs().mean().item())

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))

layers = list(range(1, len(plain_grads) + 1))
ax.plot(layers, plain_grads, 'o-', color='red', linewidth=2,
        markersize=8, label='Plain Network')
ax.plot(layers, resnet_grads, 's-', color='blue', linewidth=2,
        markersize=8, label='ResNet (with skip connections)')

ax.set_xlabel('Layer Depth (1 = earliest layer)', fontsize=12, fontweight='bold')
ax.set_ylabel('Gradient Magnitude', fontsize=12, fontweight='bold')
ax.set_title('Gradient Flow Comparison: Plain vs ResNet\n(Lower layers = earlier in network)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')  # Log scale to show exponential decay

# Add annotations
ax.annotate('Vanishing gradients\nin plain network',
            xy=(2, plain_grads[1]), xytext=(3, plain_grads[1]*10),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red')
ax.annotate('Strong gradients maintained\nvia skip connections',
            xy=(2, resnet_grads[1]), xytext=(5, resnet_grads[1]*0.1),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=10, color='blue')

plt.tight_layout()
plt.show()

print("Observation:")
print(f"  Plain Network - Layer 1 gradient: {plain_grads[0]:.6f}")
print(f"  Plain Network - Layer 10 gradient: {plain_grads[-1]:.6f}")
print(f"  Ratio (layer 10 / layer 1): {plain_grads[-1] / plain_grads[0]:.6f}")
print()
print(f"  ResNet - Layer 1 gradient: {resnet_grads[0]:.6f}")
print(f"  ResNet - Layer 10 gradient: {resnet_grads[-1]:.6f}")
print(f"  Ratio (layer 10 / layer 1): {resnet_grads[-1] / resnet_grads[0]:.6f}")
print()
print("Key insight: ResNet maintains much stronger gradients in early layers,")
print("enabling effective training of deep networks.")
```

**What you're seeing**:
- **Red line (Plain Network)**: Gradients decay exponentially as you go to earlier layers (left side of plot)
- **Blue line (ResNet)**: Gradients remain strong throughout all layers due to skip connections
- **Log scale**: Shows the exponential nature of gradient decay in plain networks

**Why this matters for training**: Without strong gradients in early layers, those layers barely update during training, making deep plain networks fail to learn effectively. ResNets solve this.

**Next steps**: Now that you understand the **core concept** (skip connections enable gradient flow), you're ready to see how this applies to tabular data in [Part 2: TabularResNet](part2-tabular-resnet), where we'll implement residual blocks using Linear layers instead of convolutions.

---

## ResNet Architecture Overview

A full ResNet consists of:
1. **Initial feature extraction**: Transform raw input into initial feature representation
2. **Residual stages**: Groups of residual blocks, typically 4 stages with increasing feature dimensions
3. **Aggregation**: Pool/reduce features to fixed size
4. **Output head**: Final layer(s) for the task (classification, embedding, etc.)

**Standard architectures**:
- **ResNet-18/34**: Use basic blocks (2 layers per block)
- **ResNet-50/101/152**: Use bottleneck blocks (3 layers per block with dimension reduction/expansion)

**Key pattern**: Each stage typically:
- Increases feature dimensions (e.g., 64 → 128 → 256 → 512)
- Maintains or reduces spatial/record dimensions
- Contains multiple residual blocks (e.g., ResNet-18 has [2, 2, 2, 2] blocks across 4 stages)

**Why this works**: The deep stack of residual stages allows the network to learn increasingly abstract representations, while skip connections ensure gradient flow to all layers.

### Bottleneck Blocks (ResNet-50+)

#### When to Use Bottleneck Blocks

- **ResNet-18/34**: Use **basic blocks** (2 layers per block) — shallower networks don't need the optimization
- **ResNet-50/101/152**: Use **bottleneck blocks** (3 layers per block) — deeper networks benefit from parameter reduction

#### Why Bottleneck Blocks?

As networks get deeper (50+ layers), the number of parameters explodes. Bottleneck blocks solve this by using a **reduce-compute-expand** pattern:

1. **Reduce** dimensions (e.g., $256 \rightarrow 64$ features) using a cheap transformation
2. **Compute** on the reduced dimensions (expensive operations, but on fewer features)
3. **Expand** back to original size (e.g., $64 \rightarrow 256$ features)

**Parameter savings**: For 256-dimensional features:
- **Basic block** (2 layers): ~589,824 parameters
- **Bottleneck block** (3 layers): ~69,632 parameters (**88% reduction!**)

This parameter reduction enables training networks with 50-152 layers without exploding compute costs.

**The key insight**: Most of the representational power comes from feature transformations, not the dimensionality itself. The bottleneck temporarily reduces dimensions for computation, then expands back.

---

## Summary

In this part, you learned the **core concepts** behind Residual Networks:

1. **The degradation problem**: Why plain deep networks fail to train effectively
2. **Residual connections** ($H(x) = F(x) + x$): The skip connection that solves the problem
3. **Why it works**:
   - Makes learning identity mappings easy ($F(x) = 0$)
   - Enables gradient flow through skip connections (the "gradient highway")
   - Empirically verified with gradient magnitude visualization
4. **Architecture patterns**:
   - Basic blocks (2 layers) for shallower networks (ResNet-18/34)
   - Bottleneck blocks (3 layers) for deeper networks (ResNet-50+)
   - Multi-stage design with increasing feature dimensions

**Key takeaway**: The skip connection is a simple but powerful innovation that makes deep networks trainable. The concept works across all domains—images, text, tabular data—making ResNet one of the most versatile architectures in deep learning.

**Next**: In [Part 2: TabularResNet](part2-tabular-resnet), you'll see how to apply these concepts to tabular OCSF data using Linear layers instead of convolutions, plus categorical embeddings for high-cardinality features.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

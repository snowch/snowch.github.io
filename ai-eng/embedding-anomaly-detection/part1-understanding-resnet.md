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

---

### Brief CNN Primer

Before implementing the residual block, let's briefly explain **Convolutional Neural Networks (CNNs)** used in the image-based examples. If you're only interested in tabular data, you can skim this section and jump to [Part 2: Adapting ResNet for Tabular Data](part2-tabular-resnet).

**What are Convolutions?**

A **convolution layer** (`nn.Conv2d`) slides a small filter (kernel) across an image to detect patterns like edges, textures, or shapes:

- **Input**: Image with shape `(batch, channels, height, width)` — e.g., `(1, 3, 32, 32)` for RGB image
- **Kernel size**: Size of the sliding window — e.g., `3×3` means look at 3×3 pixel patches
- **Stride**: How many pixels to move the window — `stride=1` moves 1 pixel at a time, `stride=2` skips every other pixel (downsampling)
- **Padding**: Add zeros around the border — `padding=1` maintains spatial dimensions
- **Channels**: Like "feature maps" — input has 3 channels (R, G, B), output might have 64 channels (64 learned patterns)

**Example:**
```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# Takes 3-channel RGB image → outputs 64 feature maps
# With padding=1, spatial dimensions stay the same (32×32 → 32×32)
```

**Key differences from fully connected layers:**
- **Fully connected (Linear)**: Every input connects to every output — used for tabular data
- **Convolution (Conv2d)**: Local connectivity with sliding window — used for images to detect spatial patterns

**Why this matters for ResNet:**
- **Part 1** (this tutorial) uses Conv2d for image examples (easier to visualize residual connections)
- **Part 2** adapts to Linear layers for tabular data (your main use case!)
- The **residual connection concept** (F(x) + x) works identically for both

### Code: Basic Residual Block

Now let's implement what we've learned. The code below shows how to build a BasicResidualBlock in PyTorch. This implementation demonstrates:
1. How to create the skip connection (`F(x) + x`)
2. Where to place batch normalization and activations for optimal gradient flow
3. How to handle dimension mismatches with projection shortcuts

**Why this matters**: Understanding this implementation is crucial for Part 2, where we'll adapt the same residual connection concept to tabular data using linear layers instead of convolutions.

```{code-cell}
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicResidualBlock(nn.Module):
    """
    Basic ResNet block for computer vision.
    Two 3x3 convolutions with a skip connection.

    Architecture: x -> [conv3x3 -> BN -> ReLU -> conv3x3 -> BN] -> + x -> ReLU -> out
                      \____________________ F(x) ____________________/
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first conv (1 for same size, 2 for downsampling)
        """
        super().__init__()

        # Main path: two conv layers (the F(x) part)
        # First conv: may downsample if stride=2
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            bias=False  # No bias because we use BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # Normalize after conv1

        # Second conv: always stride=1 (no downsampling)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            bias=False  # No bias because we use BatchNorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels)  # Normalize after conv2

        # Skip connection (the identity shortcut)
        self.skip = nn.Sequential()  # Default: identity (do nothing)

        # If dimensions change, we need a projection shortcut
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 conv to match dimensions (spatial and channel)
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass: H(x) = F(x) + x

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
        Returns:
            Output tensor of shape (batch, out_channels, height/stride, width/stride)
        """
        # Main path: F(x) = conv2(ReLU(BN(conv1(x))))
        out = self.conv1(x)           # First convolution
        out = self.bn1(out)            # Batch normalization
        out = F.relu(out)              # Activation

        out = self.conv2(out)          # Second convolution
        out = self.bn2(out)            # Batch normalization
        # Note: No ReLU here! We apply it after adding the skip connection

        # Add skip connection: F(x) + x
        # This is the KEY innovation - gradients can flow through both paths
        out += self.skip(x)

        # Final activation (after the addition)
        out = F.relu(out)

        return out

# Test the block
block = BasicResidualBlock(64, 64)
x = torch.randn(1, 64, 32, 32)  # Batch of 1, 64 channels, 32x32 image
output = block(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Residual block works! ✓")
```

**Key Design Choices**:

1. **Batch Normalization (BatchNorm)**: Normalizes activations to have mean=0 and std=1 for each mini-batch
   - Stabilizes training by reducing internal covariate shift
   - Allows higher learning rates (faster convergence)
   - Acts as a regularizer (reduces need for dropout in image models)
   - Note: We use `bias=False` in conv layers because BatchNorm has its own learnable bias term

2. **ReLU after addition**: Apply activation *after* adding the skip connection
   - Maintains non-linearity while preserving gradient flow through the skip path
   - The skip connection passes gradients unchanged (no activation to saturate)

3. **Projection shortcuts**: When dimensions change (stride>1 or different channel counts)
   - Use 1×1 conv to match spatial resolution and channel dimensions
   - Ensures we can add `F(x) + x` when shapes differ
   - Also uses BatchNorm for consistency

---

## Building a Complete ResNet

### ResNet Architecture Overview

A full ResNet consists of:
1. **Initial convolution**: Extract low-level features (e.g., 7×7 conv for images)
2. **Residual stages**: Groups of residual blocks with increasing depth
3. **Global pooling**: Aggregate spatial information
4. **Classification head**: Final fully connected layer

Standard architectures (for ImageNet):
- **ResNet-18/34**: Use basic blocks (2 conv layers per block)
- **ResNet-50/101/152**: Use bottleneck blocks (3 conv layers with 1×1, 3×3, 1×1 pattern)

#### ResNet-18 Architecture

The diagram below shows the complete ResNet-18 structure. Notice how each stage (green boxes) progressively reduces spatial dimensions (from 224×224 to 1×1) while increasing the number of channels (from 3 to 512). This pattern extracts increasingly abstract features: early stages detect edges and textures, while later stages recognize complex patterns and objects.

**Reading the diagram**: Each box shows `operation (dimensions)` → `output shape`. The format `height×width×channels` tells you the data dimensions at each step. The 4 green stages contain the residual blocks (each stage has 2 blocks for ResNet-18), giving us **18 total layers** (1 initial conv + 8 stage convs + 8 skip paths + 1 FC).

```{mermaid}
graph TB
    Input["Input Image<br/>224×224×3"] --> Conv1["Conv1 (7×7, 64)<br/>112×112×64"]
    Conv1 --> MaxPool["MaxPool<br/>56×56×64"]
    MaxPool --> Stage1["Stage 1<br/>2× BasicBlock<br/>56×56×64"]
    Stage1 --> Stage2["Stage 2<br/>2× BasicBlock<br/>28×28×128"]
    Stage2 --> Stage3["Stage 3<br/>2× BasicBlock<br/>14×14×256"]
    Stage3 --> Stage4["Stage 4<br/>2× BasicBlock<br/>7×7×512"]
    Stage4 --> AvgPool["AvgPool<br/>1×1×512"]
    AvgPool --> FC["FC (1000 classes)<br/>1000"]

    style Input fill:#ADD8E6
    style Conv1 fill:#F08080
    style MaxPool fill:#FFFFE0
    style Stage1 fill:#90EE90
    style Stage2 fill:#90EE90
    style Stage3 fill:#90EE90
    style Stage4 fill:#90EE90
    style AvgPool fill:#FFFFE0
    style FC fill:#F08080
```

**Why this structure?** The pattern of "downsample spatially, increase channels" is fundamental to deep learning on images. Lower layers with larger spatial dimensions capture fine details (like edges), while higher layers with more channels but smaller spatial dimensions capture semantic meaning (like "this is a cat"). The residual connections in each stage (green boxes) enable training this 18-layer network effectively.

### Bottleneck Block (ResNet-50+)

#### When to Use Bottleneck Blocks

- **ResNet-18/34**: Use **basic blocks** (2 conv layers) — shallower networks don't need the optimization
- **ResNet-50/101/152**: Use **bottleneck blocks** (3 conv layers) — deeper networks benefit from parameter reduction

#### Why Bottleneck Blocks?

As networks get deeper (50+ layers), the number of parameters explodes. Bottleneck blocks solve this by using a **reduce-compute-expand** pattern:

$$
\text{1×1 conv (reduce)} \rightarrow \text{3×3 conv (compute)} \rightarrow \text{1×1 conv (expand)}
$$

**The Strategy:**
1. **Reduce** dimensions with 1×1 conv (cheap: $256 \rightarrow 64$ channels)
2. **Compute** spatial features with 3×3 conv (expensive operation, but on fewer channels)
3. **Expand** back to original size with 1×1 conv (cheap: $64 \rightarrow 256$ channels)

**The name "bottleneck"** comes from the narrow middle section that reduces computational cost while preserving representational power.

#### Parameter Savings Example

Compare two ways to process 256-channel features:

**Basic Block** (two 3×3 convs):
- $256 \times 3 \times 3 \times 256 = 589{,}824$ parameters

**Bottleneck Block** (1×1, 3×3, 1×1):
- $256 \times 1 \times 1 \times 64 = 16{,}384$ (reduce)
- $64 \times 3 \times 3 \times 64 = 36{,}864$ (compute)
- $64 \times 1 \times 1 \times 256 = 16{,}384$ (expand)
- **Total**: $69{,}632$ (**12% of original!**)

This 88% parameter reduction enables training networks with 50-152 layers without exploding compute costs.

```{code-cell}
class BottleneckBlock(nn.Module):
    """
    Bottleneck block for deeper ResNets (ResNet-50/101/152).
    Uses the "reduce-compute-expand" pattern: 1x1 -> 3x3 -> 1x1

    Architecture: x -> [1x1 -> BN -> ReLU -> 3x3 -> BN -> ReLU -> 1x1 -> BN] -> + x -> ReLU
                      \_______________________ F(x) _______________________/

    This saves 88% parameters compared to basic blocks for 256-channel features!
    """
    expansion = 4  # The final 1x1 conv expands by this factor

    def __init__(self, in_channels, bottleneck_channels, stride=1):
        """
        Args:
            in_channels: Input channels (e.g., 256)
            bottleneck_channels: Narrow "bottleneck" channels (e.g., 64)
            stride: Stride for the 3x3 conv (1 for same size, 2 for downsampling)

        Output will have in_channels * expansion channels (e.g., 64 * 4 = 256)
        """
        super().__init__()

        out_channels = bottleneck_channels * self.expansion

        # Step 1: REDUCE dimensions with 1x1 conv (256 -> 64 channels)
        # This makes the next 3x3 conv much cheaper
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # Step 2: COMPUTE spatial features with 3x3 conv (64 -> 64 channels)
        # This is the expensive operation, but on fewer channels
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # Step 3: EXPAND back to original dimensions with 1x1 conv (64 -> 256 channels)
        # This restores the channel count for the skip connection
        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 conv to match dimensions
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass through bottleneck: H(x) = F(x) + x

        The bottleneck pattern reduces parameters while maintaining representational power.
        """
        # REDUCE: 1x1 conv shrinks channels
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # COMPUTE: 3x3 conv learns spatial patterns (on fewer channels)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # EXPAND: 1x1 conv restores channels
        out = self.conv3(out)
        out = self.bn3(out)

        # Add skip connection
        out += self.skip(x)
        out = F.relu(out)

        return out

# Test bottleneck block
bottleneck = BottleneckBlock(in_channels=256, bottleneck_channels=64, stride=1)
x = torch.randn(1, 256, 28, 28)
output = bottleneck(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # Should be (1, 256, 28, 28)
print(f"Bottleneck saves 88% parameters compared to basic block!")
print(f"Bottleneck block works! ✓")
```

### Full ResNet-18 Implementation

```{code-cell}
class ResNet(nn.Module):
    """
    Complete ResNet architecture for image classification.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        """
        Args:
            block: BasicResidualBlock or BottleneckBlock
            num_blocks: List of block counts per stage, e.g., [2, 2, 2, 2] for ResNet-18
            num_classes: Number of output classes
        """
        super().__init__()
        self.in_channels = 64

        # Initial convolution (for CIFAR-10: 3x3 instead of 7x7)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual stages
        self.stage1 = self._make_stage(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(block, 512, num_blocks[3], stride=2)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(self, block, out_channels, num_blocks, stride):
        """Create a stage with multiple residual blocks."""
        layers = []

        # First block (may downsample with stride)
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # Remaining blocks (stride=1)
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual stages
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # Global pooling + classifier
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)

        return out

def ResNet18(num_classes=10):
    """ResNet-18: [2, 2, 2, 2] blocks per stage"""
    return ResNet(BasicResidualBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    """ResNet-34: [3, 4, 6, 3] blocks per stage"""
    return ResNet(BasicResidualBlock, [3, 4, 6, 3], num_classes)

# Test ResNet-18
model = ResNet18(num_classes=10)
x = torch.randn(2, 3, 32, 32)  # CIFAR-10 sized images
output = model(x)
print(f"\nResNet-18 Test:")
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Training Example on CIFAR-10

**What is CIFAR-10?**

CIFAR-10 (Canadian Institute For Advanced Research) is a classic computer vision benchmark dataset containing:
- **60,000 color images** (50,000 training, 10,000 test)
- **32×32 pixels** per image (small, but challenging)
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Balanced**: 6,000 images per class

It's widely used for:
- Testing neural network architectures (ResNet achieves ~95% accuracy)
- Educational examples (small enough to train quickly)
- Benchmarking improvements in deep learning

For our ResNet tutorial, CIFAR-10 provides a perfect testbed to see how residual connections enable deeper networks to perform better than shallow ones.

```{code-cell}
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data augmentation for training (helps prevent overfitting)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Crop to 32x32 after padding by 4
    transforms.RandomHorizontalFlip(),         # Flip 50% of images horizontally
    transforms.ToTensor(),                     # Convert PIL image to tensor [0, 1]
    transforms.Normalize(                      # Normalize to mean=0, std=1
        (0.4914, 0.4822, 0.4465),             # CIFAR-10 mean per channel (R,G,B)
        (0.2023, 0.1994, 0.2010)              # CIFAR-10 std per channel (R,G,B)
    ),
])

# Test transforms: no augmentation, just normalization
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Note: In actual training, you would download and load the full dataset:
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
# This is a minimal example showing the training structure

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch (one complete pass through training data).

    Args:
        model: ResNet model to train
        dataloader: DataLoader with training batches
        optimizer: Optimizer (e.g., SGD)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: 'cuda' or 'cpu'

    Returns:
        (avg_loss, accuracy): Average loss and accuracy for this epoch
    """
    model.train()  # Set model to training mode (enables dropout, batchnorm updates)
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move data to GPU/CPU
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()              # Clear gradients from previous batch
        outputs = model(inputs)             # Get predictions
        loss = criterion(outputs, targets)  # Compute loss

        # Backward pass
        loss.backward()   # Compute gradients via backpropagation
        optimizer.step()  # Update weights using gradients

        # Track statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)       # Get class with highest probability
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss={loss.item():.3f}, Acc={100.*correct/total:.2f}%")

    return total_loss / len(dataloader), 100. * correct / total

# Training setup (example - not executed)
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(num_classes=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training loop
for epoch in range(200):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
"""

print("Training structure defined (see code for full training loop)")
print("Expected performance on CIFAR-10:")
print("  ResNet-18: ~95% test accuracy")
print("  ResNet-34: ~95.5% test accuracy")
```

---

## Summary

In this part, you learned:

1. **The degradation problem** that prevents plain networks from going deep
2. **Residual connections** ($H(x) = F(x) + x$) that enable gradient flow
3. **Basic and bottleneck blocks** for different network depths
4. **Complete ResNet architecture** for image classification
5. **Training methodology** on CIFAR-10

**Next**: In [Part 2](part2-tabular-resnet), we'll adapt this architecture for tabular data using fully connected layers instead of convolutions, and add categorical embeddings for OCSF observability data.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

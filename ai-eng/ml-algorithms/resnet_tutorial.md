---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# ResNet: Residual Networks for Deep Learning [DRAFT]

## Introduction: Why ResNet for Anomaly Detection?

This tutorial explores **Residual Networks (ResNet)**, a breakthrough architecture that enables training of very deep neural networks. While ResNet was originally designed for computer vision, it has emerged as a **surprisingly strong baseline for tabular data** and embedding models.

### Motivation: Tabular Data and Anomaly Detection

Recent research has shown that while Transformers (TabTransformer, FT-Transformer) achieve state-of-the-art results on tabular data, **ResNet-like architectures provide a simpler, more efficient baseline** that often performs comparably well{cite}`gorishniy2021revisiting`. For applications like:

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

The approach{cite}`huang2020tabtransformer`:
1. **Pre-train** a ResNet to create embeddings from individual records
2. **Extract** fixed-dimensional vectors that capture "normal" system behavior
3. **Detect anomalies** as records/sequences that deviate from learned patterns

This tutorial will build your understanding of ResNet from first principles, then show how to adapt it for tabular embeddings.

### Paper References

- **Original ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.
- **ResNet for Tabular Data**: Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). "Revisiting Deep Learning Models for Tabular Data." *NeurIPS 2021*.
- **TabTransformer Comparison**: Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). "TabTransformer: Tabular Data Modeling Using Contextual Embeddings." *arXiv preprint*.

---

## Part 1: The Problem ResNet Solves

### The Degradation Problem

Intuitively, deeper neural networks should be more powerful:
- More layers → More capacity to learn complex patterns
- A 56-layer network should perform *at least as well* as a 28-layer network (it could just learn identity mappings for the extra layers)

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

2. **Degraded Optimization Landscape**: Very deep networks create complex, non-convex loss surfaces that are hard for SGD to navigate. Even though a solution exists (copy shallower network + identity layers), the optimizer can't find it.

### What We Need

An architecture where:
- **Deeper is easier to optimize**, not harder
- **Identity mappings are learnable** by default
- **Gradients flow freely** to early layers

This is exactly what ResNet provides.

---

## Part 2: The Core Innovation — Residual Connections

### The Residual Block

Instead of learning a direct mapping $H(\mathbf{x})$, ResNet learns the **residual** (the difference):

$$
H(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}
$$

Where:
- $\mathbf{x}$: Input to the block
- $F(\mathbf{x})$: Learned transformation (typically 2-3 conv/linear layers)
- $H(\mathbf{x})$: Output (input + learned residual)

The **skip connection** (also called **shortcut connection**) adds the input directly to the output.

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plain Block
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Plain Network Block', fontsize=14, fontweight='bold', pad=20)

# Input
ax1.add_patch(FancyBboxPatch((1, 0.5), 2, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightblue', linewidth=2))
ax1.text(2, 1, 'x', ha='center', va='center', fontsize=14, fontweight='bold')

# Layer 1
ax1.add_patch(FancyBboxPatch((1, 3), 2, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='coral', linewidth=2))
ax1.text(2, 3.6, 'Conv/Linear', ha='center', va='center', fontsize=11)

# Layer 2
ax1.add_patch(FancyBboxPatch((1, 5.5), 2, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='coral', linewidth=2))
ax1.text(2, 6.1, 'Conv/Linear', ha='center', va='center', fontsize=11)

# Output
ax1.add_patch(FancyBboxPatch((1, 8), 2, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightgreen', linewidth=2))
ax1.text(2, 8.5, 'H(x)', ha='center', va='center', fontsize=14, fontweight='bold')

# Arrows
ax1.arrow(2, 1.5, 0, 1.3, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
ax1.arrow(2, 4.2, 0, 1.1, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
ax1.arrow(2, 6.7, 0, 1.1, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)

ax1.text(5, 4.5, 'Learns H(x) directly', ha='left', va='center', fontsize=12,
         style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Residual Block
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Residual Block', fontsize=14, fontweight='bold', pad=20)

# Input
ax2.add_patch(FancyBboxPatch((1, 0.5), 2, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightblue', linewidth=2))
ax2.text(2, 1, 'x', ha='center', va='center', fontsize=14, fontweight='bold')

# Layer 1
ax2.add_patch(FancyBboxPatch((1, 3), 2, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='coral', linewidth=2))
ax2.text(2, 3.6, 'Conv/Linear', ha='center', va='center', fontsize=11)

# Layer 2
ax2.add_patch(FancyBboxPatch((1, 5.5), 2, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='coral', linewidth=2))
ax2.text(2, 6.1, 'Conv/Linear', ha='center', va='center', fontsize=11)

# Addition node
ax2.add_patch(plt.Circle((2, 7.5), 0.4, edgecolor='black', facecolor='yellow', linewidth=2))
ax2.text(2, 7.5, '+', ha='center', va='center', fontsize=18, fontweight='bold')

# Output
ax2.add_patch(FancyBboxPatch((1, 8.5), 2, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightgreen', linewidth=2))
ax2.text(2, 9, 'H(x)', ha='center', va='center', fontsize=14, fontweight='bold')

# Main path arrows
ax2.arrow(2, 1.5, 0, 1.3, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
ax2.arrow(2, 4.2, 0, 1.1, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)
ax2.arrow(2, 6.7, 0, 0.6, head_width=0.3, head_length=0.15, fc='black', ec='black', linewidth=2)
ax2.arrow(2, 7.9, 0, 0.45, head_width=0.3, head_length=0.15, fc='black', ec='black', linewidth=2)

# Skip connection (curved)
skip = mpatches.FancyArrowPatch((2.5, 1), (2.5, 7.1),
                                connectionstyle="arc3,rad=1.5",
                                arrowstyle='->', mutation_scale=25,
                                linewidth=3, color='blue')
ax2.add_patch(skip)
ax2.text(5.5, 4, 'Skip Connection', ha='left', va='center', fontsize=11,
         color='blue', fontweight='bold')

# Labels
ax2.text(0.5, 4.5, 'F(x)', ha='center', va='center', fontsize=13,
         style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax2.text(5.5, 7.5, 'H(x) = F(x) + x', ha='left', va='center', fontsize=12,
         style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
```

### Why This Works: Intuition

**Learning Identity is Easy**:
- If the optimal mapping is identity ($H(\mathbf{x}) = \mathbf{x}$), the network just needs to learn $F(\mathbf{x}) = 0$
- Pushing weights toward zero is much easier than learning exact identity mappings from scratch

**Gradient Flow**:
- Gradients flow through both paths (main path *and* skip connection)
- Skip connection provides a "gradient highway" directly to earlier layers
- Even if $F(\mathbf{x})$ has vanishing gradients, $\mathbf{x}$ passes through unchanged

### Mathematical Analysis: Gradient Flow

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

### Code: Basic Residual Block

```{code-cell}
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicResidualBlock(nn.Module):
    """
    Basic ResNet block for computer vision.
    Two 3x3 convolutions with a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path: two conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Need to match dimensions
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path: F(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add skip connection: F(x) + x
        out += self.skip(x)

        # Final activation
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
1. **Batch Normalization**: Normalizes activations, helps with training stability
2. **ReLU after addition**: Maintains non-linearity while preserving gradient flow
3. **Projection shortcuts**: When dimensions change (stride or channels), use 1×1 conv to match sizes

---

## Part 3: Building a Complete ResNet

### ResNet Architecture Overview

A full ResNet consists of:
1. **Initial convolution**: Extract low-level features (e.g., 7×7 conv for images)
2. **Residual stages**: Groups of residual blocks with increasing depth
3. **Global pooling**: Aggregate spatial information
4. **Classification head**: Final fully connected layer

Standard architectures (for ImageNet):
- **ResNet-18/34**: Use basic blocks (2 conv layers per block)
- **ResNet-50/101/152**: Use bottleneck blocks (3 conv layers with 1×1, 3×3, 1×1 pattern)

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('ResNet-18 Architecture', fontsize=16, fontweight='bold', pad=20)

# Layer definitions
layers = [
    ("Input Image", 0.5, 'lightblue', '224×224×3'),
    ("Conv1 (7×7, 64)", 1.5, 'lightcoral', '112×112×64'),
    ("MaxPool", 2.3, 'lightyellow', '56×56×64'),
    ("Stage 1\n2× BasicBlock", 3.5, 'lightgreen', '56×56×64'),
    ("Stage 2\n2× BasicBlock", 5, 'lightgreen', '28×28×128'),
    ("Stage 3\n2× BasicBlock", 6.5, 'lightgreen', '14×14×256'),
    ("Stage 4\n2× BasicBlock", 8, 'lightgreen', '7×7×512'),
    ("AvgPool", 9.3, 'lightyellow', '1×1×512'),
    ("FC (1000 classes)", 10.5, 'lightcoral', '1000'),
]

y_pos = 11
for i, (name, offset, color, shape) in enumerate(layers):
    y = y_pos - offset

    # Draw box
    ax.add_patch(mpatches.FancyBboxPatch((1, y-0.35), 4, 0.7,
                                          boxstyle="round,pad=0.05",
                                          edgecolor='black', facecolor=color,
                                          linewidth=2))

    # Add text
    ax.text(3, y, name, ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, y, shape, ha='center', va='center', fontsize=10,
            family='monospace', style='italic')

    # Draw arrow (except for last layer)
    if i < len(layers) - 1:
        next_y = y_pos - layers[i+1][1]
        ax.arrow(3, y - 0.4, 0, (next_y - y) + 0.7,
                head_width=0.3, head_length=0.1, fc='black', ec='black', linewidth=1.5)

# Add annotations
ax.text(8.5, 7, 'Total: 18 layers\n(conv + FC)', ha='left', va='center',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.text(0.2, 6, 'Residual\nBlocks', ha='left', va='center', fontsize=10,
        color='darkgreen', fontweight='bold', rotation=90)

plt.tight_layout()
plt.show()
```

### Bottleneck Block (ResNet-50+)

For deeper networks, use **bottleneck blocks** to reduce computational cost:

$$
\text{1×1 conv (reduce)} \rightarrow \text{3×3 conv} \rightarrow \text{1×1 conv (expand)}
$$

Example: $256 \rightarrow 64 \rightarrow 64 \rightarrow 256$ channels

This reduces the number of parameters from $256 \times 3 \times 3 \times 256 = 589{,}824$ to:
- $256 \times 1 \times 1 \times 64 = 16{,}384$
- $64 \times 3 \times 3 \times 64 = 36{,}864$
- $64 \times 1 \times 1 \times 256 = 16{,}384$
- **Total**: $69{,}632$ (12% of original)

```{code-cell}
class BottleneckBlock(nn.Module):
    """
    Bottleneck block for deeper ResNets (ResNet-50/101/152).
    Uses 1x1 -> 3x3 -> 1x1 convolutions.
    """
    expansion = 4  # Output channels = in_channels * 4

    def __init__(self, in_channels, bottleneck_channels, stride=1):
        super().__init__()

        out_channels = bottleneck_channels * self.expansion

        # 1x1 conv: reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # 3x3 conv: main computation
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # 1x1 conv: expand dimensions
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.skip(x)  # Residual connection
        out = F.relu(out)

        return out

# Test bottleneck block
bottleneck = BottleneckBlock(in_channels=256, bottleneck_channels=64, stride=1)
x = torch.randn(1, 256, 28, 28)
output = bottleneck(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # Should be (1, 256, 28, 28)
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

```{code-cell}
:tags: [skip-execution]

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Note: In actual training, you would download and load the full dataset
# This is a minimal example showing the training structure

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
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

## Part 4: Adapting ResNet for Tabular Data

### Why ResNet for Tabular Data?

The Gorishniy et al. (2021) paper "Revisiting Deep Learning Models for Tabular Data" found that:

1. **ResNet is competitive with Transformers** on many tabular benchmarks
2. **Much simpler architecture**: No attention mechanism, easier to train
3. **Better computational efficiency**: $O(n \cdot d)$ vs. $O(d^2)$ for Transformers with $d$ features
4. **Strong baseline**: Should be tried before more complex models

### Key Differences from Image ResNet

For tabular data, we need to modify:

1. **Replace 2D convolutions** with fully connected layers
2. **Handle heterogeneous features**: Mix of categorical and numerical columns
3. **Add embeddings** for categorical features
4. **Adjust normalization**: BatchNorm or LayerNorm for tabular features
5. **Extract embeddings** for downstream tasks (anomaly detection, clustering)

### Architecture for Tabular Data

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_title('Tabular ResNet Architecture for OCSF Data', fontsize=16, fontweight='bold', pad=20)

# Components
components = [
    ("Input Features", 0.5, 'lightblue', '300+ fields'),
    ("Categorical Embeddings\n+ Numerical Features", 1.8, 'lightyellow', 'd_model dim'),
    ("Linear Projection", 3, 'lightcoral', 'd_model'),
    ("Residual Block 1\n(Linear → BN → ReLU → Linear → BN)", 4.5, 'lightgreen', 'd_model'),
    ("Residual Block 2", 6, 'lightgreen', 'd_model'),
    ("Residual Block 3", 7.5, 'lightgreen', 'd_model'),
    ("...", 8.5, 'white', '...'),
    ("Residual Block N", 9.5, 'lightgreen', 'd_model'),
    ("Layer Norm", 11, 'lightyellow', 'd_model'),
    ("Embedding Vector", 12.3, 'gold', 'd_model'),
    ("(Optional) Classification Head", 13.3, 'lightcoral', 'num_classes'),
]

y_base = 13.5
for i, (name, offset, color, dim) in enumerate(components):
    y = y_base - offset

    if name == "...":
        ax.text(3, y, '...', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.arrow(3, y + 0.3, 0, -0.5, head_width=0.3, head_length=0.1,
                fc='black', ec='black', linewidth=1.5)
        continue

    # Draw box
    ax.add_patch(mpatches.FancyBboxPatch((0.8, y-0.35), 4.4, 0.7,
                                          boxstyle="round,pad=0.05",
                                          edgecolor='black', facecolor=color,
                                          linewidth=2))

    # Labels
    ax.text(3, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.5, y, dim, ha='center', va='center', fontsize=9,
            family='monospace', style='italic')

    # Arrows
    if i < len(components) - 1 and components[i+1][0] != "...":
        next_y = y_base - components[i+1][1]
        if abs(next_y - y) > 0.1:
            ax.arrow(3, y - 0.4, 0, (next_y - y) + 0.75,
                    head_width=0.3, head_length=0.1, fc='black', ec='black', linewidth=1.5)

# Annotations
ax.add_patch(mpatches.FancyBboxPatch((6, 9.5), 3.5, 0.6,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='darkgreen', facecolor='lightgreen',
                                      linewidth=2, linestyle='--'))
ax.text(7.75, 9.8, 'Extract here for\nanomalies', ha='center', va='center',
        fontsize=9, fontweight='bold', color='darkgreen')

# Skip connection illustration
skip_y_start = y_base - 4.5
skip_y_end = y_base - 6
ax.annotate('', xy=(6.2, skip_y_end), xytext=(6.2, skip_y_start),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue',
                          connectionstyle="arc3,rad=.5"))
ax.text(7.5, (skip_y_start + skip_y_end)/2, 'Skip\nConnection',
        ha='left', va='center', fontsize=9, color='blue', fontweight='bold')

plt.tight_layout()
plt.show()
```

### Tabular Residual Block

```{code-cell}
class TabularResidualBlock(nn.Module):
    """
    Residual block for tabular data using fully connected layers.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # Two fully connected layers (analogous to two convs in image ResNet)
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(d_model, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Main path: F(x)
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # Skip connection: F(x) + x
        out = out + residual
        out = F.relu(out)
        out = self.dropout2(out)

        return out

# Test tabular residual block
block = TabularResidualBlock(d_model=128)
x = torch.randn(32, 128)  # Batch of 32 samples, 128 features
output = block(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print("Tabular residual block works! ✓")
```

### Complete Tabular ResNet with Embeddings

```{code-cell}
class TabularResNet(nn.Module):
    """
    ResNet architecture for tabular data with categorical embeddings.

    Suitable for:
    - High-dimensional tabular data (e.g., OCSF with 300+ fields)
    - Mixed categorical and numerical features
    - Self-supervised learning on unlabelled data
    - Embedding extraction for anomaly detection
    """
    def __init__(
        self,
        num_numerical_features,
        categorical_cardinalities,  # List of unique values per categorical feature
        d_model=256,
        num_blocks=6,
        dropout=0.1,
        num_classes=None,  # None for embedding-only mode
    ):
        """
        Args:
            num_numerical_features: Number of continuous features
            categorical_cardinalities: List of cardinalities for each categorical feature
            d_model: Hidden dimension size
            num_blocks: Number of residual blocks
            dropout: Dropout probability
            num_classes: Number of output classes (None for unsupervised embeddings)
        """
        super().__init__()

        self.num_numerical = num_numerical_features
        self.num_categorical = len(categorical_cardinalities)

        # Categorical feature embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model // 4)
            for cardinality in categorical_cardinalities
        ])

        # Numerical feature projection
        self.numerical_projection = nn.Linear(num_numerical_features, d_model // 2)

        # Total input dimension after embeddings
        embedding_dim = (d_model // 4) * len(categorical_cardinalities) + (d_model // 2)

        # Initial projection to d_model
        self.initial_projection = nn.Linear(embedding_dim, d_model)
        self.initial_bn = nn.BatchNorm1d(d_model)

        # Residual blocks
        self.blocks = nn.ModuleList([
            TabularResidualBlock(d_model, dropout)
            for _ in range(num_blocks)
        ])

        # Final layer norm (for embedding extraction)
        self.final_norm = nn.LayerNorm(d_model)

        # Optional classification head
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )

    def forward(self, numerical_features, categorical_features, return_embedding=False):
        """
        Args:
            numerical_features: (batch_size, num_numerical) tensor
            categorical_features: (batch_size, num_categorical) tensor of indices
            return_embedding: If True, return embedding vector instead of classification

        Returns:
            If return_embedding=True: (batch_size, d_model) embedding tensor
            Otherwise: (batch_size, num_classes) logits (if classifier exists)
        """
        batch_size = numerical_features.size(0)

        # Process categorical features through embeddings
        cat_embeddings = []
        for i, embedding_layer in enumerate(self.embeddings):
            cat_embeddings.append(embedding_layer(categorical_features[:, i]))

        cat_embed = torch.cat(cat_embeddings, dim=1) if cat_embeddings else torch.empty(batch_size, 0)

        # Process numerical features
        num_embed = self.numerical_projection(numerical_features)

        # Concatenate all features
        x = torch.cat([cat_embed, num_embed], dim=1)

        # Initial projection
        x = self.initial_projection(x)
        x = self.initial_bn(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        embedding = self.final_norm(x)

        # Return embedding or classification
        if return_embedding or self.classifier is None:
            return embedding
        else:
            return self.classifier(embedding)

# Example usage
num_numerical = 50  # e.g., network_bytes_in, duration, etc.
categorical_cardinalities = [100, 50, 200, 1000]  # e.g., user_id, status_id, entity_id, etc.

model = TabularResNet(
    num_numerical_features=num_numerical,
    categorical_cardinalities=categorical_cardinalities,
    d_model=256,
    num_blocks=6,
    num_classes=None  # Embedding mode for anomaly detection
)

# Create dummy data
batch_size = 32
numerical_data = torch.randn(batch_size, num_numerical)
categorical_data = torch.randint(0, 50, (batch_size, len(categorical_cardinalities)))

# Get embeddings
embeddings = model(numerical_data, categorical_data, return_embedding=True)
print(f"\nTabular ResNet Test:")
print(f"Numerical input shape: {numerical_data.shape}")
print(f"Categorical input shape: {categorical_data.shape}")
print(f"Embedding output shape: {embeddings.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Design Considerations for OCSF Data

When applying this to your 300+ field OCSF schema:

1. **Feature Selection**: Not all 300+ fields may be informative
   - Use domain knowledge to select relevant fields
   - Or use feature importance from tree-based models
   - Typical embedding models use 50-200 features

2. **Handling High Cardinality**: For fields like `entity_id`, `user_name`
   - Use hashing trick: `hash(entity_id) % embedding_size`
   - Or learn a shared "unknown" embedding for rare values
   - Consider using larger embedding dimensions for high-cardinality features

3. **Missing Values**: OCSF records may have sparse fields
   - Add a special "missing" category for categorical features
   - Use zero or mean imputation for numerical features
   - Or add a binary "is_missing" indicator feature

4. **Temporal Features**: For `timestamp` fields
   - Extract: hour_of_day, day_of_week, time_since_last_event
   - Treat as numerical or cyclical (sin/cos encoding) features

---

## Part 5: Self-Supervised Training for Anomaly Detection

### The Training Strategy

Since your observability data is **unlabelled**, you need self-supervised learning. Two effective approaches from the TabTransformer paper:

#### 1. Masked Feature Prediction (MFP)

Randomly mask some features and train the model to reconstruct them:

```python
def masked_feature_prediction_loss(model, numerical, categorical):
    """
    Mask random features and predict them.
    """
    # Randomly select features to mask (e.g., 15% of features)
    mask_prob = 0.15

    # Mask categorical features
    masked_categorical = categorical.clone()
    cat_mask = torch.rand_like(categorical.float()) < mask_prob
    masked_categorical[cat_mask] = 0  # 0 = [MASK] token

    # Get embeddings from masked input
    embedding = model(numerical, masked_categorical, return_embedding=True)

    # Predict original categorical values
    # (requires adding prediction heads to the model)
    predictions = model.categorical_predictors(embedding)

    # Compute cross-entropy loss only on masked positions
    loss = F.cross_entropy(predictions[cat_mask], categorical[cat_mask])

    return loss
```

#### 2. Contrastive Learning

Learn embeddings where similar records cluster together:

```python
def contrastive_loss(embeddings, temperature=0.07):
    """
    SimCLR-style contrastive loss for tabular data.
    Augmentations: add noise, drop features, etc.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)

    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    # Create positive pairs (augmented versions of same record)
    # ... (implementation details omitted for brevity)

    return loss
```

### Anomaly Detection After Training

Once trained, use embeddings for anomaly detection:

```{code-cell}
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Simulate trained embeddings for normal and anomalous data
np.random.seed(42)

# Normal data: clustered around origin
normal_embeddings = np.random.randn(500, 2) * 0.5 + np.array([0, 0])

# Anomalies: scattered outliers
anomaly_embeddings = np.random.randn(50, 2) * 1.5 + np.array([3, 3])

all_embeddings = np.vstack([normal_embeddings, anomaly_embeddings])
labels = np.array([0]*500 + [1]*50)  # 0=normal, 1=anomaly

# Method 1: Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_predictions = lof.fit_predict(all_embeddings)

# Method 2: Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_predictions = iso_forest.fit_predict(all_embeddings)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Ground truth
axes[0].scatter(all_embeddings[labels==0, 0], all_embeddings[labels==0, 1],
                c='blue', alpha=0.6, label='Normal', s=30)
axes[0].scatter(all_embeddings[labels==1, 0], all_embeddings[labels==1, 1],
                c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
axes[0].set_title('Ground Truth', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LOF results
axes[1].scatter(all_embeddings[lof_predictions==1, 0], all_embeddings[lof_predictions==1, 1],
                c='blue', alpha=0.6, label='Predicted Normal', s=30)
axes[1].scatter(all_embeddings[lof_predictions==-1, 0], all_embeddings[lof_predictions==-1, 1],
                c='red', alpha=0.8, label='Predicted Anomaly', s=50, marker='x')
axes[1].set_title('Local Outlier Factor', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Isolation Forest results
axes[2].scatter(all_embeddings[iso_predictions==1, 0], all_embeddings[iso_predictions==1, 1],
                c='blue', alpha=0.6, label='Predicted Normal', s=30)
axes[2].scatter(all_embeddings[iso_predictions==-1, 0], all_embeddings[iso_predictions==-1, 1],
                c='red', alpha=0.8, label='Predicted Anomaly', s=50, marker='x')
axes[2].set_title('Isolation Forest', fontsize=13, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score

print("\nAnomaly Detection Performance:")
print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 56)

for name, preds in [("Local Outlier Factor", lof_predictions),
                     ("Isolation Forest", iso_predictions)]:
    # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
    binary_preds = (preds == -1).astype(int)

    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)

    print(f"{name:<20} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")
```

### Multi-Record Anomaly Detection

For detecting anomalies across sequences of events:

```{code-cell}
:tags: [remove-output]

class SequenceAnomalyDetector(nn.Module):
    """
    Detect anomalies in sequences of events using embeddings.
    """
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()

        # LSTM or Transformer to model sequences of embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           batch_first=True, dropout=0.1)

        # Predict "normality score"
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability of being normal
        )

    def forward(self, sequence_embeddings):
        """
        Args:
            sequence_embeddings: (batch_size, sequence_length, embedding_dim)

        Returns:
            normality_scores: (batch_size,) probability of sequence being normal
        """
        # Process sequence
        lstm_out, (hidden, cell) = self.lstm(sequence_embeddings)

        # Use final hidden state for scoring
        score = self.scorer(hidden[-1])

        return score.squeeze()

# Example usage
sequence_detector = SequenceAnomalyDetector(embedding_dim=256, hidden_dim=128)

# Simulate a sequence of 10 events
sequence = torch.randn(1, 10, 256)  # 1 sequence, 10 events, 256-dim embeddings
normality_score = sequence_detector(sequence)

print(f"Sequence shape: {sequence.shape}")
print(f"Normality score: {normality_score.item():.3f}")
print(f"Interpretation: {normality_score.item():.3f} probability of being normal behavior")
```

### Practical Workflow for OCSF Data

1. **Data Preparation**:
   - Parse OCSF JSON records
   - Extract numerical and categorical features
   - Handle missing values and normalize

2. **Model Training** (Self-Supervised):
   - Train TabularResNet with masked feature prediction
   - Validate on held-out unlabelled data (reconstruction error)
   - Train for 50-100 epochs with early stopping

3. **Embedding Extraction**:
   - Pass all records through trained model
   - Extract embeddings from `final_norm` layer
   - Store in vector database (e.g., Faiss, Pinecone)

4. **Anomaly Detection**:
   - **Point anomalies**: Use LOF or Isolation Forest on embeddings
   - **Sequence anomalies**: Use SequenceAnomalyDetector on event sequences
   - **Contextual anomalies**: Compare embedding distance to entity-specific clusters

5. **Visualization & Interpretation**:
   - Use t-SNE or UMAP to visualize embedding space
   - Identify anomaly clusters
   - Trace back to original OCSF fields for root cause analysis

---

## Part 6: Comparison with Transformers

### When to Use ResNet vs. Transformers

Based on the research{cite}`gorishniy2021revisiting`:

**Use ResNet when:**
- ✅ You have many features (100-300+) and attention complexity is prohibitive
- ✅ You want a simple, interpretable baseline
- ✅ Training resources are limited (ResNet trains faster)
- ✅ You need fast inference (no attention overhead)

**Use Transformers (TabTransformer/FT-Transformer) when:**
- ✅ You need state-of-the-art performance at any cost
- ✅ Features have complex interactions (attention can model this better)
- ✅ You have sufficient compute for quadratic attention
- ✅ You want to leverage pre-trained models

### Performance Comparison

From Gorishniy et al. (2021) on tabular benchmarks:

| Model | Avg. Rank | Training Time | Inference Time |
|-------|-----------|---------------|----------------|
| FT-Transformer | **1.2** | 3× baseline | 2× baseline |
| ResNet | **1.5** | 1× baseline | 1× baseline |
| TabTransformer | 2.1 | 3× baseline | 2× baseline |
| MLP | 3.8 | 0.8× baseline | 0.8× baseline |

**Interpretation**: FT-Transformer wins on average, but ResNet is competitive (rank 1.5 vs. 1.2) with much better efficiency.

### Hybrid Approach

For your use case, consider:

1. **Start with ResNet**: Establish a strong baseline quickly
2. **Benchmark Transformer**: Compare FT-Transformer if performance is insufficient
3. **Ensemble**: Combine ResNet and Transformer embeddings for best results

---

## Conclusion

### What You've Learned

1. **The degradation problem** and why skip connections solve it
2. **ResNet architecture** from basic blocks to full networks
3. **Adapting ResNet for tabular data** with embeddings
4. **Self-supervised training** for unlabelled observability data
5. **Embedding extraction** for anomaly detection
6. **Multi-record anomaly detection** using sequence models

### Next Steps

1. **Implement a simple ResNet** on a toy dataset (CIFAR-10)
2. **Adapt for your OCSF data**:
   - Select top 50-100 most informative fields
   - Create categorical embeddings for high-cardinality features
   - Train with masked feature prediction
3. **Extract embeddings** and visualize with t-SNE
4. **Compare with TabTransformer** (see FT-Transformer tutorial — coming soon!)
5. **Deploy** anomaly detection pipeline in production

### Further Reading

- **Original ResNet Paper**: He et al., "Deep Residual Learning for Image Recognition" (2016)
- **ResNet for Tabular**: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (2021)
- **TabTransformer**: Huang et al., "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" (2020)
- **FT-Transformer**: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (2021) — same paper, different model

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

(Note: For proper citation rendering, you would add a `references.bib` file with the BibTeX entries for the papers mentioned.)

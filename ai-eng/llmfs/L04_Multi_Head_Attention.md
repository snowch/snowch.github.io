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

# L04 - Multi-Head Attention: The Committee of Experts

*Why one brain is good, but eight brains are better.*

---

In [L03 - Self-Attention](L03_The_Attention_Mechanism.md), we built the "Search Engine" of the Transformer. We learned how the word "it" can look up the word "animal" to resolve ambiguity.

But there is a limitation. A single self-attention layer acts like a single pair of eyes. It can focus on **one** aspect of the sentence at a time.

Consider the sentence:
> **"The chicken didn't cross the road because it was too wide."**

To understand this fully, the model needs to do two things simultaneously:
1.  **Syntactic Analysis:** Link "it" to the subject "road" (because roads are wide).
2.  **Semantic Analysis:** Understand that "wide" is a physical property preventing crossing.

If we only have one attention head, the model has to average these different relationships into a single vector. It muddies the waters.

**Multi-Head Attention** solves this by giving the model multiple "heads" (independent attention mechanisms) that run in parallel.

By the end of this post, you'll understand:
- The intuition of the **"Committee of Experts."**
- Why we project vectors into different **Subspaces**.
- How to implement the tensor reshaping magic (`view` and `transpose`) in PyTorch.

:::{code-cell} ipython3
:tags: [remove-input]

import os
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
:::

---

## Part 1: The Intuition (The Committee)

Think of the embedding dimension ($d_{model} = 512$) as a massive report containing everything we know about a word.

If we ask a single person to read that report and summarize "grammar," "tone," "tense," and "meaning" all at once, they might miss details.

Instead, we hire a **Committee of 8 Experts**:
* **Head 1 (The Linguist):** Only looks for Subject-Verb agreement.
* **Head 2 (The Historian):** Looks for past/present tense consistency.
* **Head 3 (The Translator):** Looks for definitions and synonyms.
* ...

In the Transformer, we don't just copy the input 8 times. We **project** the input into 8 different lower-dimensional spaces. This allows each head to specialize.

:::{note} Why Lower Dimensions? Why Not Give Each Head the Full 512 Dimensions?

**The Short Answer:** Computational efficiency and forced specialization.

If each of the 8 heads used the full $d_{model} = 512$ dimensions:
- We'd need **8× the parameters** ($W^Q, W^K, W^V$ for each head would each be $512 \times 512$ instead of $512 \times 64$)
- We'd need **8× the computation** (each attention operation scales with $d_k$)
- Heads might learn **redundant patterns** rather than specializing

By splitting the dimensions ($d_k = d_{model}/h = 64$):
- **Total parameters stay constant:** 8 heads × 64 dims ≈ 1 head × 512 dims
- **Computational cost is comparable** to single-head attention
- **The constraint forces specialization:** Each head must compress its focus into fewer dimensions, encouraging it to capture distinct linguistic patterns

Think of it like hiring specialists with limited notepads. If each expert had unlimited space, they might all write the same general report. But with only 64 dimensions, each head is forced to be selective and focus on what matters most to its specialized role.
:::

:::{important} The Roles Are Learned, Not Assigned

When we say "Head 1 (The Linguist)" we're using a metaphor for intuition. In reality:

- **You only specify `num_heads=8`** in your code
- **The model learns** what each head should focus on during training through backpropagation
- **The "roles" are emergent** - they're patterns discovered by gradient descent, not programmed by you
- **Descriptive, not prescriptive** - Labels like "Linguist" are what researchers assign *after* analyzing what a trained model learned

You can't tell the model "Head 1, you focus on grammar!" - it discovers its own patterns that minimize loss. Different training runs or datasets might result in different specializations.
:::

Let's visualize this "filtering" process. In the plot below:
* **The Input (Mixed Info):** The large multi-colored bar represents the full word embedding ($d=512$).
* **The Split (Equal Parts):** We project this into 8 **equal-sized** subspaces ($d_k = 64$).
* **The Result:** Each head gets a vector that is 1/8th the size of the original, containing only the specific info it needs.

:::{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_multihead_projection_concept():
    fig, ax = plt.subplots(figsize=(14, 8)) # Increased size slightly
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # --- Configuration ---
    input_x = 1
    output_x = 10
    
    # Coordinates for the "Heads" (Outputs)
    # We show them essentially "stacking up" to equal the total concept, 
    # but separated to show independence.
    y_positions = [7, 4.5, 2] 
    colors = ['#FF9999', '#99FF99', '#9999FF'] 
    labels = ["Head 1\nGrammar", "Head 2\nTense", "Head 3\nMeaning"]

    # FONT SIZES
    font_title = 20
    font_label = 16
    font_math = 14
    font_annot = 14

    # --- 1. Draw Input Vector (The "General Report") ---
    height = 6
    width = 1.5
    base_y = 1.5
    
    # Container
    input_rect = patches.Rectangle((input_x, base_y), width, height, linewidth=3, edgecolor='black', facecolor='none', zorder=5)
    ax.add_patch(input_rect)
    
    # "Mixed Info" bands
    num_segments = 20
    seg_height = height / num_segments
    np.random.seed(42) 
    segment_colors = plt.get_cmap('tab20').colors
    for i in range(num_segments):
        color = segment_colors[i % len(segment_colors)]
        rect = patches.Rectangle((input_x, base_y + i*seg_height), width, seg_height, facecolor=color, alpha=0.7, edgecolor='none')
        ax.add_patch(rect)

    # Label Input
    ax.text(input_x + width/2, base_y - 0.6, "Input Embedding\n($d_{model}=512$)", ha='center', va='top', fontweight='bold', fontsize=font_label)

    # --- 2. Draw The Projections (Arrows & Matrices) ---
    
    proj_start_x = input_x + width + 0.2
    proj_end_x = output_x - 0.2
    
    for i, (y_c, color, label) in enumerate(zip(y_positions, colors, labels)):
        # A. Arrow from Input Center to Head Center
        arrow = patches.FancyArrowPatch(
            (proj_start_x, base_y + height/2), (proj_end_x, y_c),
            arrowstyle='-|>,head_width=0.6,head_length=1.0',
            connectionstyle=f"arc3,rad={(i-1)*-0.15}", 
            color=color, lw=4, zorder=2, alpha=0.8
        )
        ax.add_patch(arrow)
        
        # B. Matrix Box (The "Lens")
        mid_x = (proj_start_x + proj_end_x) / 2
        mid_y = (base_y + height/2 + y_c) / 2 + (i-1)*0.5 # Slight offset for visual separation
        
        # Matrix Label
        ax.text(mid_x, mid_y + 0.6, f"$W_{i}$", ha='center', va='center', color=color, fontweight='bold', fontsize=font_label, zorder=6)

        # C. Output Subspace Blocks
        # Make them look identical in size
        out_h = 1.8
        out_w = 1.5
        out_rect = patches.Rectangle((output_x, y_c - out_h/2), out_w, out_h, facecolor=color, edgecolor='black', lw=2, alpha=0.9)
        ax.add_patch(out_rect)
        
        # Label each head
        ax.text(output_x + out_w + 0.3, y_c, label, ha='left', va='center', fontsize=font_label, color='black')
        # Math label showing the split
        ax.text(output_x + out_w + 0.3, y_c - 0.6, "($d_k=64$)", ha='left', va='center', fontsize=font_math, color='#555')

    # --- 3. Final Annotations describing the Split ---
    
    # Title
    ax.text(7, 8.5, "Multi-Head Projection: Dividing the Work", ha='center', va='center', fontsize=font_title, fontweight='bold')
    
    # Explanation of the split
    ax.text(7, 0.5, "Total Dimensions (512) $\\div$ Heads (8) = 64 dims per Head", 
            ha='center', va='center', fontsize=font_label, fontweight='bold', 
            bbox=dict(facecolor='#f0f0f0', edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()

plot_multihead_projection_concept()
:::

:::{important} Technical Note: What actually gets split?

You might look at the diagram and wonder: *"Does Head 1 just look at the first 64 numbers of the input?"*

**No.** That would be disastrous, because the first 64 numbers of the embedding might not contain the specific grammar information Head 1 needs.

The process happens in two specific steps:

1.  **The Mix (Linear Layer):** First, the input vector (512) is multiplied by the weight matrix ($W$). This operation has access to the **entire** input vector. It blends all the information together.
2.  **The Split (Reshape):** The *result* of that multiplication is a new 512-dimensional vector. **This new vector** is what gets chopped into 8 chunks of 64.

So, Head 1 *can* see the whole input, but the Linear Layer ensures that the information Head 1 needs ends up in the "first chunk" (indices 0-63) of the output.
:::

Let's visualize this crucial distinction:

:::{code-cell} ipython3
:tags: [remove-input]

def plot_mix_then_split():
    """
    Visualizes the two-step process: Mix (Linear Transform) then Split (Reshape)
    This clarifies that we DON'T just slice the input into chunks.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))

    # ========== SUBPLOT 1: WRONG APPROACH (Direct Split) ==========
    ax1 = axes[0]
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 6.5)
    ax1.axis('off')

    # Title
    ax1.text(7, 6, "WRONG: Direct Split (Naive Approach)",
             ha='center', va='center', fontsize=18, fontweight='bold', color='#d32f2f',
             bbox=dict(facecolor='#ffcdd2', edgecolor='#d32f2f', boxstyle='round,pad=0.7', linewidth=2))

    # Input vector (colorful segments representing different kinds of information)
    input_x = 1
    input_y = 2
    input_width = 4
    input_height = 1.5

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9', '#FD79A8', '#FDCB6E']
    segment_width = input_width / 8

    for i in range(8):
        rect = patches.Rectangle(
            (input_x + i * segment_width, input_y),
            segment_width, input_height,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=2
        )
        ax1.add_patch(rect)

    ax1.text(input_x + input_width/2, input_y - 0.4,
             "Input Vector (512 dims)\n[0-63][64-127][128-191]...[448-511]",
             ha='center', va='top', fontsize=12, fontweight='bold')

    # Arrow pointing to sliced outputs
    arrow_start_x = input_x + input_width + 0.3
    arrow_end_x = 9

    # Three example heads
    head_positions = [4.5, 2.5, 0.5]
    head_labels = ["Head 1\n(dims 0-63)", "Head 2\n(dims 64-127)", "Head 8\n(dims 448-511)"]
    head_colors_wrong = [colors[0], colors[1], colors[7]]

    for i, (y_pos, label, color) in enumerate(zip(head_positions, head_labels, head_colors_wrong)):
        # Arrow
        arrow = patches.FancyArrowPatch(
            (arrow_start_x, input_y + input_height/2),
            (arrow_end_x, y_pos),
            arrowstyle='-|>,head_width=0.5,head_length=0.8',
            connectionstyle=f"arc3,rad={(i-1)*0.2}",
            color=color,
            lw=3,
            alpha=0.7
        )
        ax1.add_patch(arrow)

        # Head box
        head_rect = patches.Rectangle((arrow_end_x + 0.2, y_pos - 0.4), 1.5, 0.8,
                                       facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(head_rect)
        ax1.text(arrow_end_x + 0.95, y_pos, label, ha='center', va='center',
                 fontsize=11, fontweight='bold')

    # Problem annotation
    ax1.text(7, 0.3,
             "Problem: Each head only sees a fixed slice of the input.\nHead 1 can't access information in dimensions 64-511!",
             ha='center', va='center', fontsize=12,
             bbox=dict(facecolor='#ffcdd2', edgecolor='#d32f2f', boxstyle='round,pad=0.6', linewidth=2),
             style='italic')

    # ========== SUBPLOT 2: CORRECT APPROACH (Mix then Split) ==========
    ax2 = axes[1]
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 6.5)
    ax2.axis('off')

    # Title
    ax2.text(7, 6, "CORRECT: Mix (Linear Transform) then Split (Reshape)",
             ha='center', va='center', fontsize=18, fontweight='bold', color='#2e7d32',
             bbox=dict(facecolor='#c8e6c9', edgecolor='#2e7d32', boxstyle='round,pad=0.7', linewidth=2))

    # Input vector
    for i in range(8):
        rect = patches.Rectangle(
            (input_x + i * segment_width, input_y),
            segment_width, input_height,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=2
        )
        ax2.add_patch(rect)

    ax2.text(input_x + input_width/2, input_y - 0.4,
             "Input Vector (512 dims)",
             ha='center', va='top', fontsize=12, fontweight='bold')

    # Linear transformation (Weight matrix)
    matrix_x = 5.5
    matrix_y = 2
    matrix_width = 1.2
    matrix_height = 1.5

    # Draw weight matrix as a grid
    matrix_rect = patches.Rectangle((matrix_x, matrix_y), matrix_width, matrix_height,
                                     facecolor='#E8EAF6', edgecolor='#3F51B5', linewidth=3)
    ax2.add_patch(matrix_rect)

    # Grid pattern inside matrix
    for i in range(5):
        ax2.plot([matrix_x, matrix_x + matrix_width],
                 [matrix_y + i*matrix_height/5, matrix_y + i*matrix_height/5],
                 'k-', alpha=0.3, linewidth=0.5)
        ax2.plot([matrix_x + i*matrix_width/5, matrix_x + i*matrix_width/5],
                 [matrix_y, matrix_y + matrix_height],
                 'k-', alpha=0.3, linewidth=0.5)

    ax2.text(matrix_x + matrix_width/2, matrix_y + matrix_height + 0.3,
             "$W^Q$ (512×512)", ha='center', va='bottom', fontsize=12, fontweight='bold', color='#3F51B5')
    ax2.text(matrix_x + matrix_width/2, matrix_y - 0.3,
             "Mixes ALL\n512 dims", ha='center', va='top', fontsize=10, style='italic', color='#3F51B5')

    # Arrow from input to matrix
    arrow1 = patches.FancyArrowPatch(
        (input_x + input_width, input_y + input_height/2),
        (matrix_x - 0.1, matrix_y + matrix_height/2),
        arrowstyle='->,head_width=0.5,head_length=0.8',
        color='black',
        lw=3
    )
    ax2.add_patch(arrow1)
    ax2.text((input_x + input_width + matrix_x)/2, input_y + input_height/2 + 0.4,
             "Step 1: Mix", ha='center', fontsize=11, fontweight='bold', color='#1976D2')

    # Transformed vector (still 512, but mixed)
    transformed_x = 7.5
    for i in range(8):
        # Make colors slightly different/blended to show mixing
        rect = patches.Rectangle(
            (transformed_x + i * segment_width, input_y),
            segment_width, input_height,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=2,
            alpha=0.6,
            hatch='//'
        )
        ax2.add_patch(rect)

    # Arrow from matrix to transformed
    arrow2 = patches.FancyArrowPatch(
        (matrix_x + matrix_width + 0.1, matrix_y + matrix_height/2),
        (transformed_x - 0.1, input_y + input_height/2),
        arrowstyle='->,head_width=0.5,head_length=0.8',
        color='black',
        lw=3
    )
    ax2.add_patch(arrow2)

    ax2.text(transformed_x + input_width/2, input_y - 0.4,
             "Transformed Vector (512 dims)\nEach dim is a blend of ALL input dims",
             ha='center', va='top', fontsize=11, fontweight='bold', style='italic')

    # Split annotation
    split_x = transformed_x + input_width + 0.3
    ax2.text(split_x, input_y + input_height/2,
             "Step 2: Reshape\n.view(8, 64)",
             ha='left', va='center', fontsize=11, fontweight='bold',
             color='#2e7d32',
             bbox=dict(facecolor='#C8E6C9', edgecolor='#2e7d32', boxstyle='round,pad=0.5', linewidth=2))

    # Benefit annotation
    ax2.text(7, 0.3,
             "Each head receives 64 dims that contain information blended from ALL 512 input dims!\nThe weight matrix $W^Q$ learns to put relevant info in the right positions.",
             ha='center', va='center', fontsize=12,
             bbox=dict(facecolor='#c8e6c9', edgecolor='#2e7d32', boxstyle='round,pad=0.6', linewidth=2),
             fontweight='bold')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

plot_mix_then_split()
:::

---

## Part 2: The Multi-Head Pipeline

Now that we understand the "why" (Specialization), let's look at the "how" (The Pipeline).

The Multi-Head Attention mechanism isn't a single black box; it is a specific sequence of operations. It allows the model to process information in parallel and then synthesize the results.

**The 4-Step Process**

1.  **Linear Projections (The Split):** We don't just use the raw input. We multiply the input $Q, K, V$ by specific weight matrices ($W^Q_i, W^K_i, W^V_i$) for each head. This creates the specialized "subspaces" we saw in Part 1.
2.  **Independent Attention:** Each head runs the standard Scaled Dot-Product Attention independently.
    $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
3.  **Concatenation:** We take the output vectors from all 8 heads and glue them back together side-by-side.
4.  **Final Linear (The Mix):** We pass this long concatenated vector through one last linear layer ($W^O$) to blend the insights from all the experts into a single unified vector.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

Let's visualize this flow:

:::{mermaid}
graph TD
    subgraph Inputs
        Q(Query)
        K(Key)
        V(Value)
    end

    Q --> LQ[1. Linear Projections]
    K --> LK[1. Linear Projections]
    V --> LV[1. Linear Projections]

    LQ --> H1[2. Head 1 Attention]
    LK --> H1
    LV --> H1

    LQ --> H2[2. Head ... Attention]
    LK --> H2
    LV --> H2

    LQ --> H8[2. Head 8 Attention]
    LK --> H8
    LV --> H8

    H1 --> C[3. Concatenate]
    H2 --> C
    H8 --> C

    C --> O[4. Final Linear Transform]
    O --> Out[Multi-Head Output]

    style C fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style O fill:#ffecb3,stroke:#ff6f00,stroke-width:2px
    style Out fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
:::

---

## Part 3: Visualizing Multiple Perspectives

Let's visualize how two different heads might analyze the same sentence.

**Sentence:** "The cat sat on the mat because it was soft."

* **Head 1** focuses on the physical relationship (connecting "it" to "mat").
* **Head 2** focuses on the actor (connecting "sat" to "cat").

Notice how they highlight completely different parts of the matrix.

:::{code-cell} ipython3
:tags: [remove-input]

tokens = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "soft"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Head 1: Reference Resolution (it -> mat)
# Simulating a head that understands physical properties
head1 = np.zeros((len(tokens), len(tokens)))
head1[tokens.index("it"), tokens.index("mat")] = 0.95
head1[tokens.index("soft"), tokens.index("mat")] = 0.8
# Add some background noise
np.random.seed(42)
head1 += np.random.rand(len(tokens), len(tokens)) * 0.05
# Normalize
head1 = head1 / head1.sum(axis=1, keepdims=True)

# Head 2: Syntax / Subject-Verb (sat -> cat)
# Simulating a head that connects verbs to their subjects
head2 = np.zeros((len(tokens), len(tokens)))
head2[tokens.index("cat"), tokens.index("sat")] = 0.6
head2[tokens.index("sat"), tokens.index("cat")] = 0.9
head2 += np.random.rand(len(tokens), len(tokens)) * 0.05
head2 = head2 / head2.sum(axis=1, keepdims=True)

# Plotting Head 1
im1 = ax1.imshow(head1, cmap='Purples', aspect='auto')
ax1.set_title("Head 1: The 'Meaning' Expert\n(Resolving 'it' -> 'mat')", fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(tokens)))
ax1.set_xticklabels(tokens, rotation=45)
ax1.set_yticks(range(len(tokens)))
ax1.set_yticklabels(tokens)
ax1.grid(False)

# Plotting Head 2
im2 = ax2.imshow(head2, cmap='Greens', aspect='auto')
ax2.set_title("Head 2: The 'Grammar' Expert\n(Linking Subject <-> Verb)", fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(tokens, rotation=45)
ax2.set_yticks(range(len(tokens)))
ax2.set_yticklabels(tokens)
ax2.grid(False)

plt.tight_layout()
plt.show()
:::

---

## Part 4: Implementation in PyTorch

Implementing this efficiently requires some tensor gymnastics. We don't actually run a `for` loop over the 8 heads. That would be too slow.

Instead, we use a single large matrix multiply and then **reshape** (view/transpose) the tensor to create a "heads" dimension.

The shape transformation looks like this:
1.  **Input:** `[Batch, Seq_Len, D_Model]`
2.  **Linear & Reshape:** `[Batch, Seq_Len, Heads, D_Head]`
3.  **Transpose:** `[Batch, Heads, Seq_Len, D_Head]`

By swapping axes 1 and 2, we group the "Heads" dimension with the "Batch" dimension. PyTorch then processes all heads in parallel as if they were just extra items in the batch.

Let's visualize these tensor transformations:

:::{code-cell} ipython3
:tags: [remove-input]

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_tensor_3d(ax, origin, dims, color, alpha=0.7, label='', edge_color='black'):
    """
    Draw a 3D box representing a tensor.
    origin: (x, y, z) starting point
    dims: (width, height, depth) corresponding to tensor dimensions
    """
    x, y, z = origin
    w, h, d = dims

    # Define the 8 vertices of the box
    vertices = [
        [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],  # bottom face
        [x, y, z+d], [x+w, y, z+d], [x+w, y+h, z+d], [x, y+h, z+d]  # top face
    ]

    # Define the 6 faces using vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # top
    ]

    # Create the 3D polygon collection
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=edge_color, linewidth=1.5)
    ax.add_collection3d(poly)

    return vertices

def plot_tensor_transformations():
    """Visualize the view() and transpose() operations in multi-head attention."""
    fig = plt.figure(figsize=(16, 10))

    # We'll create 3 subplots showing the key transformations
    steps = [
        {
            'title': 'Step 1: After Linear Projection',
            'shape': '[2, 10, 512]',
            'dims': (4, 2, 1),  # visual dimensions (not actual values)
            'labels': ['Seq=10', 'Batch=2', 'D_model=512'],
            'color': '#90CAF9',
            'description': 'Flat 512-dimensional vectors\nfor each token in sequence'
        },
        {
            'title': 'Step 2: After .view(2, 10, 8, 64)',
            'shape': '[2, 10, 8, 64]',
            'dims': (4, 2, 1.5),  # slightly taller to show split
            'labels': ['Seq=10', 'Batch=2', 'Heads=8\n×\nD_k=64'],
            'color': '#FFB74D',
            'description': 'Split 512 dims into 8 heads × 64 dims\n(no data movement, just reshape)'
        },
        {
            'title': 'Step 3: After .transpose(1, 2)',
            'shape': '[2, 8, 10, 64]',
            'dims': (2, 4, 1.5),  # swap width and height
            'labels': ['Heads=8', 'Batch=2', 'Seq=10\n×\nD_k=64'],
            'color': '#81C784',
            'description': 'Swap axes: Heads and Seq dimensions\nNow heads are independent!'
        }
    ]

    for idx, step in enumerate(steps):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        # Draw the tensor
        origin = (0, 0, 0)
        draw_tensor_3d(ax, origin, step['dims'], step['color'], alpha=0.7)

        # Set labels for axes
        w, h, d = step['dims']
        ax.text(w/2, -0.5, 0, step['labels'][0], ha='center', fontsize=11, fontweight='bold')
        ax.text(-0.5, h/2, 0, step['labels'][1], ha='center', fontsize=11, fontweight='bold', rotation=90)
        ax.text(w + 0.3, h, d/2, step['labels'][2], ha='left', fontsize=11, fontweight='bold')

        # Title and shape
        ax.text2D(0.5, 0.95, step['title'], transform=ax.transAxes,
                  ha='center', fontsize=13, fontweight='bold')
        ax.text2D(0.5, 0.88, f"Shape: {step['shape']}", transform=ax.transAxes,
                  ha='center', fontsize=11, color='#333', style='italic')

        # Description
        ax.text2D(0.5, 0.08, step['description'], transform=ax.transAxes,
                  ha='center', fontsize=10, style='italic',
                  bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.8))

        # Set axis limits
        ax.set_xlim([-1, max(w, h, d) + 1])
        ax.set_ylim([-1, max(w, h, d) + 1])
        ax.set_zlim([-0.5, max(w, h, d) + 0.5])

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 0.8])

    # Add arrows between subplots to show progression
    fig.text(0.31, 0.5, '→\n.view()', ha='center', va='center', fontsize=16, fontweight='bold', color='#FF6F00')
    fig.text(0.64, 0.5, '→\n.transpose(1,2)', ha='center', va='center', fontsize=16, fontweight='bold', color='#FF6F00')

    # Add overall title
    fig.suptitle('Tensor Shape Transformations in Multi-Head Attention',
                 fontsize=16, fontweight='bold', y=0.98)

    # Key insight box
    fig.text(0.5, 0.02,
             'Key Insight: After transpose, each head is independent and can be processed in parallel!\n'
             'PyTorch treats [Batch × Heads] as a combined batch dimension.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(facecolor='#E3F2FD', edgecolor='#1976D2', boxstyle='round,pad=0.8', linewidth=2))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

plot_tensor_transformations()
:::

### Shape Transformation Table

Let's trace the exact tensor shapes through a concrete example with **batch=2, seq=10, d_model=512, heads=8**:

| Operation | Shape | Description |
| --- | --- | --- |
| **Input** `x` | `[2, 10, 512]` | Raw input: 2 sequences, each with 10 tokens, 512-dim embeddings |
| **After** `W_q(x)` | `[2, 10, 512]` | Linear projection (still flat) |
| **After** `.view(2, 10, 8, 64)` | `[2, 10, 8, 64]` | Reshape: Split 512 dims into 8 heads × 64 dims each |
| **After** `.transpose(1, 2)` | `[2, 8, 10, 64]` | Swap seq and heads: Now we have 8 "parallel attention mechanisms" |
| **Attention computation** | `[2, 8, 10, 64]` | Each head computes attention independently |
| **After** `.transpose(1, 2)` | `[2, 10, 8, 64]` | Swap back: Prepare for concatenation |
| **After** `.contiguous().view(2, 10, 512)` | `[2, 10, 512]` | Flatten: Merge 8 heads back into single 512-dim vector |
| **After** `W_o(x)` | `[2, 10, 512]` | Final projection |

The key insight: dimensions 1 and 2 get swapped twice—once to parallelize the heads, and once to merge them back together.

:::{code-cell} ipython3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # We define 4 linear layers: Q, K, V projections and the final Output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Project and Split
        # We transform [Batch, Seq, Model] -> [Batch, Seq, Heads, d_k]
        # Then we transpose to [Batch, Heads, Seq, d_k] for matrix multiplication
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention (re-using logic from L03)
        # Scores shape: [Batch, Heads, Seq, Seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply weights to Values
        # Shape: [Batch, Heads, Seq, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # 3. Concatenate
        # Transpose back: [Batch, Seq, Heads, d_k]
        # Flatten: [Batch, Seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final Projection (The "Mix")
        return self.W_o(attn_output)
:::

:::{note}
**Why `.contiguous()`? Understanding Memory Layout**

When we `transpose` a tensor in PyTorch, we aren't actually moving data in memory; we are just changing the "stride" (how the computer steps through memory).

**Memory Layout Explanation:**
- Tensors are stored as 1D arrays in memory. Multi-dimensional tensors use "strides" to map from indices to memory locations.
- When you call `.transpose()`, PyTorch creates a new **view** with different strides, but the underlying data stays in the same physical order in memory.
- The `view()` operation requires the data to be laid out in a specific order in memory (row-major, also called C-contiguous).
- If the tensor isn't contiguous after transpose, calling `.view()` would give incorrect results or raise an error.

**What `.contiguous()` does:**
- Creates a new tensor with data physically rearranged in memory to match the current logical shape.
- This is a **copy operation**, so it has a performance cost, but it's necessary to ensure correctness.

**Example:**
```python
x = torch.randn(2, 3, 4)
x_t = x.transpose(1, 2)  # Creates a view, not contiguous
print(x_t.is_contiguous())  # False
x_c = x_t.contiguous()      # Creates a contiguous copy
print(x_c.is_contiguous())  # True
```

This is a PyTorch implementation detail; the math doesn't care about memory layout, but the computer does!
:::

---

## Summary

1.  **Multiple Heads:** We split our embedding into $h$ smaller chunks to allow the model to focus on different linguistic features simultaneously.
2.  **Projection:** We use learned linear layers ($W_Q, W_K, W_V$) to project the input into these specialized subspaces.
3.  **Parallelism:** We use tensor reshaping (`view` and `transpose`) to compute attention for all heads at once, rather than looping through them.

**Next Up: L05 – Layer Norm & Residuals.**
We have built the engine (Attention), but if we stack 100 of these layers on top of each other, the gradients will vanish or explode. In L05, we will add the "plumbing" (Normalization and Skip Connections) that allows Deep Learning to actually get *deep*.

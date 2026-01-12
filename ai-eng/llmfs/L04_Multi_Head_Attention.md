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

::::{important} Technical Note: What actually gets split? (The Input Projections)

You might look at the diagram and wonder: *"Does Head 1 just look at the first 64 numbers of the input?"*

**No.** That would be disastrous, because the first 64 numbers of the embedding might not contain the specific grammar information Head 1 needs.

The process happens in two specific steps:

1.  **The Mix (Linear Layer):** First, the input vector (512) is multiplied by a weight matrix ($W^Q$, $W^K$, or $W^V$), which is **learned during training**. This operation has access to the **entire** input vector. It blends all the information together.
2.  **The Split (Reshape):** The *result* of that multiplication is a new 512-dimensional vector. **This new vector** is what gets chopped into 8 chunks of 64.

So, Head 1 *can* see the whole input, but the Linear Layer's **learned weights** ensure that the information Head 1 needs ends up in the "first chunk" (indices 0-63) of the output.

**Note:** This happens for each of the three input projections (Query, Key, Value). Later, after all heads complete their attention computations and get concatenated, there's one more linear transformation ($W^O$) that mixes the results from all heads (see Part 2, Step 4).


Let's visualize this crucial distinction:

:::{code-cell} ipython3
:tags: [remove-input]

def plot_mix_then_split():
    """
    Visualizes the two-step process: Mix (Linear Transform) then Split (Reshape).
    Shows horizontal flow: Input -> W^Q -> Mixed -> Reshape -> Heads
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Geometry
    input_x, y = 1.0, 3.4
    input_width, input_height = 5.2, 1.15
    n_heads = 8
    seg_w = input_width / n_heads

    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DFE6E9', '#FD79A8', '#FDCB6E'
    ]

    # -------------------------
    # Input vector (512 dims)
    # -------------------------
    for i in range(n_heads):
        ax.add_patch(
            patches.Rectangle(
                (input_x + i * seg_w, y),
                seg_w, input_height,
                facecolor=colors[i],
                edgecolor='black',
                linewidth=2.2
            )
        )

    ax.text(
        input_x + input_width / 2, y + input_height + 0.35,
        "Input vector (512 dims)",
        ha='center', va='bottom',
        fontsize=14, fontweight='bold'
    )

    # -------------------------
    # W^Q matrix
    # -------------------------
    matrix_x = input_x + input_width + 0.9
    matrix_y = y + input_height / 2 - 0.95
    mw, mh = 1.55, 1.9

    ax.add_patch(
        patches.Rectangle(
            (matrix_x, matrix_y), mw, mh,
            facecolor='#E8EAF6',
            edgecolor='#3F51B5',
            linewidth=3
        )
    )

    # Grid pattern inside matrix
    grid = 8
    for i in range(grid + 1):
        ax.plot(
            [matrix_x, matrix_x + mw],
            [matrix_y + i * mh / grid, matrix_y + i * mh / grid],
            'k-', alpha=0.25, linewidth=0.8
        )
        ax.plot(
            [matrix_x + i * mw / grid, matrix_x + i * mw / grid],
            [matrix_y, matrix_y + mh],
            'k-', alpha=0.25, linewidth=0.8
        )

    # Step 1 label ABOVE W^Q
    step1_x = matrix_x + mw / 2
    step1_y = matrix_y + mh + 0.85
    ax.text(
        step1_x, step1_y,
        "Step 1: Mix",
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color='#1976D2',
        bbox=dict(
            facecolor='white',
            edgecolor='#1976D2',
            boxstyle='round,pad=0.35',
            linewidth=2
        )
    )

    # Matrix labels
    ax.text(
        matrix_x + mw / 2, matrix_y + mh + 0.25,
        r"$W^Q$",
        ha='center', va='bottom',
        fontsize=16, fontweight='bold',
        color='#3F51B5'
    )
    ax.text(
        matrix_x + mw / 2, matrix_y - 0.18,
        "512×512",
        ha='center', va='top',
        fontsize=11, fontweight='bold',
        color='#3F51B5'
    )

    # Arrow: input -> matrix
    ax.add_patch(
        patches.FancyArrowPatch(
            (input_x + input_width + 0.1, y + input_height / 2),
            (matrix_x - 0.15, matrix_y + mh / 2),
            arrowstyle='->,head_width=0.5,head_length=0.7',
            lw=4,
            color='#1976D2'
        )
    )

    # -------------------------
    # Mixed vector (512 dims)
    # -------------------------
    mixed_x = matrix_x + mw + 0.9
    for i in range(n_heads):
        ax.add_patch(
            patches.Rectangle(
                (mixed_x + i * seg_w, y),
                seg_w, input_height,
                facecolor='#9C27B0',
                edgecolor='black',
                linewidth=2.2,
                alpha=0.72,
                hatch='///'
            )
        )

    ax.text(
        mixed_x + input_width / 2, y + input_height + 0.35,
        "Mixed vector (512 dims)",
        ha='center', va='bottom',
        fontsize=14, fontweight='bold'
    )

    ax.text(
        mixed_x + 0.25, y - 0.30,
        "Each mixed dim uses ALL 512 inputs",
        ha='left', va='top',
        fontsize=10.5,
        style='italic',
        color='#6A1B9A'
    )

    # Arrow: matrix -> mixed
    ax.add_patch(
        patches.FancyArrowPatch(
            (matrix_x + mw + 0.15, matrix_y + mh / 2),
            (mixed_x - 0.15, y + input_height / 2),
            arrowstyle='->,head_width=0.5,head_length=0.7',
            lw=4,
            color='#1976D2'
        )
    )

    # -------------------------
    # Step 2: Reshape + heads
    # -------------------------
    down_x = mixed_x + input_width / 2
    head_y = 1.55
    head_h = 0.70

    reshape_top = y - 0.65
    arrow_end = head_y + head_h + 0.10

    ax.annotate(
        '',
        xy=(down_x, arrow_end),
        xytext=(down_x, reshape_top),
        arrowprops=dict(
            arrowstyle='->,head_width=0.5,head_length=0.7',
            lw=4,
            color='#2e7d32'
        )
    )

    ax.text(
        down_x + 1.8, (reshape_top + arrow_end) / 2 + 0.15,
        "Step 2: Reshape\nview(8, 64)",
        ha='left', va='center',
        fontsize=11.5, fontweight='bold',
        color='#2e7d32',
        bbox=dict(
            facecolor='#E8F5E9',
            edgecolor='#2e7d32',
            boxstyle='round,pad=0.38',
            linewidth=2
        )
    )

    # Heads row
    for i in range(n_heads):
        ax.add_patch(
            patches.Rectangle(
                (mixed_x + i * seg_w, head_y),
                seg_w, head_h,
                facecolor='#9C27B0',
                edgecolor='black',
                linewidth=1.8,
                alpha=0.78
            )
        )
        ax.text(
            mixed_x + i * seg_w + seg_w / 2,
            head_y + head_h / 2,
            f"H{i+1}",
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color='white'
        )

    ax.text(
        mixed_x + input_width / 2, head_y - 0.34,
        "8 heads × 64 dims each",
        ha='center', va='top',
        fontsize=12, fontweight='bold'
    )

    # Key insight
    ax.text(
        8.5, 0.55,
        "Key idea: The learned weight matrix $W^Q$ mixes ALL input dims, THEN we split.\nEach head receives features computed from the entire input.",
        ha='center', va='center',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.show()

plot_mix_then_split()
:::

::::

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

Now let's see how to implement multi-head attention efficiently in code. Remember the key insight from our Technical Note: we multiply by $W^Q$ (which mixes ALL 512 input dimensions), then split the result into 8 heads.

For a single input vector, this is straightforward. But in practice, we process **batches** of sequences (e.g., batch=2, seq=10). We could loop through each head one at a time, but that would be too slow.

Instead, PyTorch uses clever **tensor reshaping** to process all heads in parallel:

1. **Single matrix multiply:** Apply $W^Q$ to the entire batch at once → `[Batch, Seq_Len, D_Model]`
2. **Reshape (view):** Split the 512 dimensions into 8 heads × 64 dims → `[Batch, Seq_Len, Heads, D_Head]`
3. **Transpose:** Swap axes so heads can be processed in parallel → `[Batch, Heads, Seq_Len, D_Head]`

By swapping axes 1 and 2, we group the "Heads" dimension with the "Batch" dimension. PyTorch then processes all heads in parallel as if they were just extra items in the batch.

Let's visualize these tensor transformations:

:::{code-cell} ipython3
:tags: [remove-input]

import matplotlib.patheffects as pe

def plot_tensor_transformations(light_scale=2.0):
    """
    Visualize the view() and transpose() operations as a clean 2D horizontal flow.

    Fixes:
      - Orange box main dimension label contrast (adds light bbox + darker text)
      - Green head labels larger + outlined for readability
      - light_scale=2.0 doubles grey/italic captions + top annotation boxes
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---- layout ----
    y_center = 4.75
    box_width = 3.5
    box_height = 1.4

    # Fonts
    main_dim_fs = 16
    header_fs = 15
    op_fs = 12
    light_fs = int(round(10 * light_scale))  # grey/italic captions
    anno_fs = int(round(9 * light_scale))    # top annotation boxes
    key_fs = 13
    head_fs = 11  # increased for H1..H8

    def op_bbox(face, edge):
        return dict(facecolor=face, edgecolor=edge, boxstyle="round,pad=0.35", linewidth=2)

    # -------------------------
    # Step 1: [2, 10, 512]
    # -------------------------
    x1 = 1.0
    ax.add_patch(
        patches.Rectangle(
            (x1, y_center - box_height / 2),
            box_width,
            box_height,
            facecolor="#90CAF9",
            edgecolor="#1976D2",
            linewidth=3,
        )
    )

    ax.text(
        x1 + box_width / 2,
        y_center,
        "[2, 10, 512]",
        ha="center",
        va="center",
        fontsize=main_dim_fs,
        fontweight="bold",
        color="#0D47A1",
    )

    ax.text(
        x1 + box_width / 2,
        y_center + box_height / 2 + 0.70,
        r"After $W^Q(x)$",
        ha="center",
        va="bottom",
        fontsize=header_fs,
        fontweight="bold",
    )

    ax.text(
        x1 + box_width / 2,
        y_center - box_height / 2 - 0.45,
        "Flat 512 dims",
        ha="center",
        va="top",
        fontsize=light_fs,
        style="italic",
        color="#555",
    )

    # -------------------------
    # Arrow + .view()
    # -------------------------
    arrow1_start = x1 + box_width + 0.15
    arrow1_end = x1 + box_width + 0.95
    ax.add_patch(
        patches.FancyArrowPatch(
            (arrow1_start, y_center),
            (arrow1_end, y_center),
            arrowstyle="->,head_width=0.55,head_length=0.65",
            lw=4,
            color="#FF6F00",
        )
    )

    ax.text(
        (arrow1_start + arrow1_end) / 2,
        y_center + 0.85,
        ".view(2, 10, 8, 64)",
        ha="center",
        va="bottom",
        fontsize=op_fs,
        fontweight="bold",
        color="#FF6F00",
        bbox=op_bbox("#FFF3E0", "#FF6F00"),
    )

    # -------------------------
    # Step 2: [2, 10, 8, 64]
    # -------------------------
    x2 = arrow1_end + 0.15

    n_layers = 8
    layer_height = box_height / n_layers
    for i in range(n_layers):
        ax.add_patch(
            patches.Rectangle(
                (x2, y_center - box_height / 2 + i * layer_height),
                box_width,
                layer_height,
                facecolor="#FFB74D",
                edgecolor="#F57C00",
                linewidth=1.4,
                alpha=0.88,
            )
        )

    # Contrast fix: darker text + light background behind it
    ax.text(
        x2 + box_width / 2,
        y_center,
        "[2, 10, 8, 64]",
        ha="center",
        va="center",
        fontsize=main_dim_fs,
        fontweight="bold",
        color="#2D1B00",
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.22"),
    )

    ax.text(
        x2 + box_width / 2,
        y_center + box_height / 2 + 0.70,
        "After .view()",
        ha="center",
        va="bottom",
        fontsize=header_fs,
        fontweight="bold",
    )

    ax.text(
        x2 + box_width / 2,
        y_center - box_height / 2 - 0.45,
        "8 heads × 64 dims",
        ha="center",
        va="top",
        fontsize=light_fs,
        style="italic",
        color="#555",
    )

    # -------------------------
    # Arrow + .transpose()
    # -------------------------
    arrow2_start = x2 + box_width + 0.15
    arrow2_end = x2 + box_width + 1.05
    ax.add_patch(
        patches.FancyArrowPatch(
            (arrow2_start, y_center),
            (arrow2_end, y_center),
            arrowstyle="->,head_width=0.55,head_length=0.65",
            lw=4,
            color="#2E7D32",
        )
    )

    ax.text(
        (arrow2_start + arrow2_end) / 2,
        y_center + 0.85,
        ".transpose(1, 2)",
        ha="center",
        va="bottom",
        fontsize=op_fs,
        fontweight="bold",
        color="#2E7D32",
        bbox=op_bbox("#E8F5E9", "#2E7D32"),
    )

    # -------------------------
    # Step 3: [2, 8, 10, 64]
    # -------------------------
    x3 = arrow2_end + 0.15

    n_segments = 8
    segment_width = box_width / n_segments
    for i in range(n_segments):
        ax.add_patch(
            patches.Rectangle(
                (x3 + i * segment_width, y_center - box_height / 2),
                segment_width,
                box_height,
                facecolor="#81C784",
                edgecolor="#388E3C",
                linewidth=1.8,
                alpha=0.92,
            )
        )
        t = ax.text(
            x3 + i * segment_width + segment_width / 2,
            y_center,
            f"H{i+1}",
            ha="center",
            va="center",
            fontsize=head_fs,
            fontweight="bold",
            color="white",
        )
        # Outline makes white text readable against mid-tone greens
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    ax.text(
        x3 + box_width / 2,
        y_center + box_height / 2 + 0.70,
        "After .transpose(1, 2)",
        ha="center",
        va="bottom",
        fontsize=header_fs,
        fontweight="bold",
    )

    ax.text(
        x3 + box_width / 2,
        y_center + box_height / 2 + 1.10,
        "[2, 8, 10, 64]",
        ha="center",
        va="bottom",
        fontsize=main_dim_fs,
        fontweight="bold",
        color="#1B5E20",
    )

    ax.text(
        x3 + box_width / 2,
        y_center - box_height / 2 - 0.45,
        "Heads are independent!",
        ha="center",
        va="top",
        fontsize=light_fs,
        style="italic",
        color="#555",
    )

    # -------------------------
    # Top annotations
    # -------------------------
    top_y = y_center + box_height / 2 + 2.10
    ax.text(
        x1 + box_width / 2,
        top_y,
        "Batch=2, Seq=10\nD_model=512",
        ha="center",
        va="bottom",
        fontsize=anno_fs,
        color="#666",
        bbox=dict(facecolor="white", edgecolor="#90CAF9", boxstyle="round,pad=0.30", linewidth=1.2),
    )

    ax.text(
        x2 + box_width / 2,
        top_y,
        "Batch=2, Seq=10\nHeads=8, D_k=64",
        ha="center",
        va="bottom",
        fontsize=anno_fs,
        color="#666",
        bbox=dict(facecolor="white", edgecolor="#FFB74D", boxstyle="round,pad=0.30", linewidth=1.2),
    )

    ax.text(
        x3 + box_width / 2,
        top_y,
        "Batch=2, Heads=8\nSeq=10, D_k=64",
        ha="center",
        va="bottom",
        fontsize=anno_fs,
        color="#666",
        bbox=dict(facecolor="white", edgecolor="#81C784", boxstyle="round,pad=0.30", linewidth=1.2),
    )

    # -------------------------
    # Bottom key insight
    # -------------------------
    ax.text(
        8.5,
        1.05,
        "Key Insight: After transpose, PyTorch processes all 8 heads in parallel\n"
        "by treating [Batch × Heads] as a combined batch dimension.",
        ha="center",
        va="center",
        fontsize=key_fs,
        fontweight="bold",
        bbox=dict(facecolor="#E3F2FD", edgecolor="#1976D2", boxstyle="round,pad=0.70", linewidth=2),
    )

    plt.tight_layout()
    plt.show()

plot_tensor_transformations(light_scale=2.0)
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
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
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

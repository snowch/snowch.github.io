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

# Part 2: Adapting ResNet for Tabular Data

ResNet fundamentals from Part 1, we now adapt the architecture for tabular observability data.

## Why ResNet for Tabular Data?

The Gorishniy et al. (2021) paper "Revisiting Deep Learning Models for Tabular Data" found that:

1. **ResNet is competitive with Transformers** on many tabular benchmarks
2. **Much simpler architecture**: No attention mechanism, easier to train
3. **Better computational efficiency**: $O(n \cdot d)$ vs. $O(d^2)$ for Transformers with $d$ features
4. **Strong baseline**: Should be tried before more complex models

## Key Differences from Image ResNet

For tabular data, we need to modify:

1. **Replace 2D convolutions** with fully connected layers
2. **Handle heterogeneous features**: Mix of categorical and numerical columns
3. **Add embeddings** for categorical features
4. **Adjust normalization**: BatchNorm or LayerNorm for tabular features
5. **Extract embeddings** for downstream tasks (anomaly detection, clustering)

## Architecture for Tabular Data

The tabular ResNet follows this flow:

1. **Input Layer**: Raw OCSF features (300+ fields)
   - Categorical features: `user_id`, `status_id`, `entity_id`, etc.
   - Numerical features: `network_bytes_in`, `duration`, `timestamp`, etc.

2. **Feature Embedding Layer**:
   - Categorical → Embedding vectors (e.g., 64-dim per category)
   - Numerical → Linear projection
   - Concatenate all → Single feature vector

3. **Tabular Residual Blocks** (stacked 4-8 times):
   - **Linear → BatchNorm1d → ReLU → Dropout → Linear → BatchNorm1d**
   - Add skip connection: `output = F(x) + x`
   - Final ReLU activation
   - Note: These use **Linear layers**, not Conv2d (no spatial structure in tabular data)

4. **Output Layer**:
   - **For embeddings**: Extract from last residual block (use for anomaly detection)
   - **For classification**: Add linear head (e.g., predict anomaly class)

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 13)
ax.axis('off')
ax.set_title('Tabular ResNet Architecture for OCSF Data', fontsize=16, fontweight='bold', pad=20)

# Define layers with positions
layers = [
    # (name, y_position, color, width, height, annotation)
    ("Input: OCSF Features\n(300+ fields: categorical + numerical)", 12, 'lightblue', 5, 0.8, None),
    ("Categorical Embeddings", 10.5, 'lightyellow', 2.2, 0.6, "e.g., user_id → 64-dim"),
    ("Numerical Projection", 10.5, 'lightyellow', 2.2, 0.6, "e.g., bytes_in → normalized"),
    ("Concatenate Features", 9, 'wheat', 5, 0.6, "Combined feature vector"),
    ("Initial Linear Projection", 7.8, 'lightcoral', 5, 0.6, "Project to d_model (e.g., 256)"),
    ("Tabular Residual Block 1", 6.5, 'lightgreen', 5, 0.7, "Linear-BN1d-ReLU-Dropout-Linear-BN1d + skip"),
    ("Tabular Residual Block 2", 5.3, 'lightgreen', 5, 0.7, "Linear layers, not Conv2d"),
    ("...", 4.3, None, 5, 0.5, None),
    ("Tabular Residual Block N", 3.3, 'lightgreen', 5, 0.7, "Same tabular architecture"),
    ("Layer Normalization", 2, 'lightyellow', 5, 0.6, "Final normalization"),
    ("Embedding Vector (d_model)", 0.8, 'gold', 5, 0.7, "Extract here for anomaly detection ✓"),
]

# Draw main flow
x_center = 7
for i, layer in enumerate(layers):
    if layer[0] == "...":
        ax.text(x_center, layer[1], '...', ha='center', va='center',
                fontsize=24, fontweight='bold')
        continue

    name, y_pos, color, width, height, annotation = layer

    # Special positioning for categorical/numerical (side by side)
    if "Categorical" in name:
        x_start = x_center - width - 0.3
        ax.add_patch(mpatches.FancyBboxPatch((x_start, y_pos - height/2), width, height,
                                              boxstyle="round,pad=0.08",
                                              edgecolor='black', facecolor=color, linewidth=2))
        ax.text(x_start + width/2, y_pos, name, ha='center', va='center',
                fontsize=9, fontweight='bold')
        if annotation:
            ax.text(x_start + width/2, y_pos - 0.6, annotation, ha='center', va='top',
                    fontsize=7, style='italic', color='gray')
        continue

    if "Numerical" in name:
        x_start = x_center + 0.3
        ax.add_patch(mpatches.FancyBboxPatch((x_start, y_pos - height/2), width, height,
                                              boxstyle="round,pad=0.08",
                                              edgecolor='black', facecolor=color, linewidth=2))
        ax.text(x_start + width/2, y_pos, name, ha='center', va='center',
                fontsize=9, fontweight='bold')
        if annotation:
            ax.text(x_start + width/2, y_pos - 0.6, annotation, ha='center', va='top',
                    fontsize=7, style='italic', color='gray')
        # Draw arrows from cat+num to concatenate
        ax.annotate('', xy=(x_center, 9.3), xytext=(x_start + width/2, y_pos - height/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax.annotate('', xy=(x_center, 9.3), xytext=(x_center - width - 0.3 + width/2, y_pos - height/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        continue

    # Regular centered boxes
    x_start = x_center - width/2
    ax.add_patch(mpatches.FancyBboxPatch((x_start, y_pos - height/2), width, height,
                                          boxstyle="round,pad=0.08",
                                          edgecolor='black', facecolor=color, linewidth=2))
    ax.text(x_center, y_pos, name, ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Add annotations below boxes
    if annotation:
        if "anomaly detection" in annotation:
            # Special highlight for extraction point
            ax.text(x_center, y_pos - height/2 - 0.25, annotation, ha='center', va='top',
                    fontsize=9, fontweight='bold', color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                             edgecolor='darkgreen', linewidth=1.5))
        else:
            ax.text(x_center, y_pos - height/2 - 0.2, annotation, ha='center', va='top',
                    fontsize=7, style='italic', color='gray')

# Draw main flow arrows (skip categorical/numerical which have custom arrows)
arrow_pairs = [
    (12, 11.2),  # Input to embeddings
    (9.3, 8.1),  # Concat to projection
    (7.5, 6.85), # Projection to block 1
    (6.15, 5.65), # Block 1 to 2
    (4.95, 4.6), # Block 2 to ...
    (4.0, 3.65), # ... to block N
    (2.95, 2.3), # Block N to LayerNorm
    (1.7, 1.15), # LayerNorm to embedding
]

for y_start, y_end in arrow_pairs:
    ax.annotate('', xy=(x_center, y_end), xytext=(x_center, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Add skip connection visualization
skip_x = 11
skip_y_start = 6.5
skip_y_end = 5.3
ax.annotate('', xy=(skip_x, skip_y_end), xytext=(skip_x, skip_y_start),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='blue',
                          connectionstyle="arc3,rad=0.8"))
ax.text(skip_x + 1.2, (skip_y_start + skip_y_end)/2, 'Skip\nConnection',
        ha='left', va='center', fontsize=9, color='blue', fontweight='bold')

# Add side annotations
ax.text(0.5, 6.5, 'Feature\nExtraction', ha='left', va='center',
        fontsize=10, style='italic', color='darkblue', rotation=90)
ax.text(0.5, 4, 'Tabular\nResNet', ha='left', va='center',
        fontsize=10, style='italic', color='darkgreen', rotation=90)
ax.text(0.5, 1.5, 'Output', ha='left', va='center',
        fontsize=10, style='italic', color='darkred', rotation=90)

# Add legend for block types
legend_x = 11.5
legend_y = 11.5
ax.text(legend_x, legend_y, 'Component Types:', fontsize=9, fontweight='bold')
legend_items = [
    ('Input/Output', 'lightblue', legend_y - 0.5),
    ('Feature Processing', 'lightyellow', legend_y - 1.0),
    ('Tabular Residual Blocks', 'lightgreen', legend_y - 1.5),
    ('Embedding Extraction', 'gold', legend_y - 2.0),
]
for label, color, y in legend_items:
    ax.add_patch(mpatches.Rectangle((legend_x, y - 0.15), 0.3, 0.3,
                                     facecolor=color, edgecolor='black', linewidth=1))
    ax.text(legend_x + 0.4, y, label, ha='left', va='center', fontsize=7)

plt.tight_layout()
plt.show()
```

## Tabular Residual Block

```{code-cell}
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularResidualBlock(nn.Module):
    """
    Residual block for tabular data using fully connected layers.

    Key differences from image ResNet:
    - Uses Linear layers instead of Conv2d (no spatial structure in tabular data)
    - Uses BatchNorm1d instead of BatchNorm2d (1D features, not 2D images)
    - Adds dropout for regularization (common in tabular deep learning)

    Architecture: x -> [Linear -> BN1d -> ReLU -> Dropout -> Linear -> BN1d] -> + x -> ReLU -> Dropout
    """
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: Feature dimension (must be same for input/output)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        # Two fully connected layers (analogous to two convs in image ResNet)
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)  # BatchNorm1d for tabular (not 2d for images)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(d_model, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, d_model) tensor of features
        Returns:
            (batch_size, d_model) tensor after residual transformation
        """
        # Save input for skip connection
        residual = x

        # Main path: F(x) = fc2(dropout(relu(bn(fc1(x)))))
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)  # Regularization

        out = self.fc2(out)
        out = self.bn2(out)

        # Skip connection: F(x) + x (the key innovation)
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

## Complete Tabular ResNet with Embeddings

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

## Design Considerations for OCSF Data

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

## Comparison with Image ResNet

| Aspect | Image ResNet | Tabular ResNet |
|--------|-------------|----------------|
| **Layers** | Conv2d | Linear |
| **Normalization** | BatchNorm2d | BatchNorm1d / LayerNorm |
| **Regularization** | Minimal (BatchNorm sufficient) | Dropout (prevents overfitting) |
| **Input** | Fixed-size images (H×W×C) | Variable features (categorical + numerical) |
| **Embeddings** | None (raw pixels) | Categorical embeddings required |
| **Skip Connection** | $F(x) + x$ | $F(x) + x$ (identical!) |

The **core innovation** (residual connections) remains the same, but the implementation details adapt to the structure of tabular data.

---

## Summary

In this part, you learned:

1. **Why ResNet works** for tabular data (competitive with Transformers, simpler)
2. **Key architectural differences** (Linear vs Conv2d, BatchNorm1d, dropout)
3. **Categorical embeddings** for high-cardinality features
4. **Complete TabularResNet implementation** with embedding extraction
5. **Design considerations** for real-world OCSF data

**Next**: In [Part 3](part3-self-supervised-training), we'll train this model using self-supervised learning on unlabelled observability data, and learn how to evaluate embedding quality.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

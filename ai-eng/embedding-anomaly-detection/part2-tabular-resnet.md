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

# Part 2: Adapting ResNet for Tabular Data [DRAFT]

Building on the ResNet fundamentals from Part 1, we now adapt the architecture for tabular observability data.

**What changes**: Instead of processing images with 2D convolutions, we'll use fully connected (Linear) layers to process rows of tabular data. The core residual connection concept ($F(x) + x$) remains identical, but the implementation details adapt to handle mixed categorical and numerical features.

## Why ResNet for Tabular Data?

The Gorishniy et al. (2021) paper "Revisiting Deep Learning Models for Tabular Data" found that:

1. **ResNet is competitive with Transformers** on many tabular benchmarks
2. **Much simpler architecture**: No attention mechanism, easier to train
3. **Better computational efficiency**: $O(n \cdot d)$ vs. $O(d^2)$ for Transformers with $d$ features
4. **Strong baseline**: Should be tried before more complex models

## Key Terminology

Before diving into the architecture, let's define some terms specific to tabular deep learning:

- **Fully connected (Linear) layers**: Every input feature connects to every output feature. Unlike convolutions that look at local patches, linear layers consider all features together. Think of it like a weighted combination of all input features.

- **High-cardinality features**: Categorical features with many unique values (e.g., `user_id` with 10,000+ users, `entity_id` with millions of values). These require special handling since one-hot encoding would create enormous sparse vectors.

- **Categorical embeddings**: Convert categorical values into dense numerical vectors (like word embeddings in NLP). For example, `user_id=12345` might map to a learned 64-dimensional vector `[0.2, -0.5, 0.8, ...]` that captures user characteristics.

- **BatchNorm1d / LayerNorm**: Normalization techniques that stabilize training by ensuring activations don't explode or vanish. BatchNorm normalizes across the batch dimension, while LayerNorm normalizes across features.

- **Dropout**: Randomly sets a percentage of features to zero during training to prevent overfitting. It's like forcing the network to learn multiple ways to make predictions rather than relying on specific feature combinations.

## Key Differences from Image ResNet

**Note**: If you're familiar with Image ResNets (originally designed for computer vision), you'll notice several key adaptations for tabular data. If you're new to ResNet, don't worry - the core concept (skip connections enabling deep networks) is the same, as explained in Part 1.

For tabular data, we need to modify:

1. **Replace 2D convolutions** with fully connected layers (Linear layers that connect all features)
2. **Handle heterogeneous features**: Mix of categorical and numerical columns
3. **Add embeddings** for categorical features (especially high-cardinality ones)
4. **Adjust normalization**: BatchNorm1d or LayerNorm for 1D tabular features (not 2D images)
5. **Add dropout** for regularization (more important for tabular data than images)
6. **Extract embeddings** for downstream tasks (anomaly detection, clustering)

## Architecture for Tabular Data

The diagram below visualizes the complete data flow through TabularResNet. Notice how categorical and numerical features are processed separately at first (left and right paths), then merged and passed through residual blocks. The green boxes are where the residual connections enable deep learning on tabular data.

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

```{mermaid}
graph TB
    Input["Input: OCSF Features<br/>(300+ fields: categorical + numerical)"]

    CatEmbed["Categorical Embeddings<br/><i>e.g., user_id → 64-dim</i>"]
    NumProj["Numerical Projection<br/><i>e.g., bytes_in → normalized</i>"]

    Concat["Concatenate Features<br/><i>Combined feature vector</i>"]

    InitProj["Initial Linear Projection<br/><i>Project to d_model (e.g., 256)</i>"]

    Block1["Tabular Residual Block 1<br/><i>Linear-BN1d-ReLU-Dropout-Linear-BN1d + skip</i>"]
    Block2["Tabular Residual Block 2<br/><i>Linear layers, not Conv2d</i>"]
    Dots["⋮"]
    BlockN["Tabular Residual Block N<br/><i>Same tabular architecture</i>"]

    LayerNorm["Layer Normalization<br/><i>Final normalization</i>"]

    Embedding["<b>Embedding Vector (d_model)</b><br/><i>Extract here for anomaly detection ✓</i>"]

    Input --> CatEmbed
    Input --> NumProj
    CatEmbed --> Concat
    NumProj --> Concat
    Concat --> InitProj
    InitProj --> Block1
    Block1 --> Block2
    Block2 --> Dots
    Dots --> BlockN
    BlockN --> LayerNorm
    LayerNorm --> Embedding

    style Input fill:#ADD8E6,stroke:#333,stroke-width:2px
    style CatEmbed fill:#FFFFE0,stroke:#333,stroke-width:2px
    style NumProj fill:#FFFFE0,stroke:#333,stroke-width:2px
    style Concat fill:#F5DEB3,stroke:#333,stroke-width:2px
    style InitProj fill:#F08080,stroke:#333,stroke-width:2px
    style Block1 fill:#90EE90,stroke:#333,stroke-width:2px
    style Block2 fill:#90EE90,stroke:#333,stroke-width:2px
    style BlockN fill:#90EE90,stroke:#333,stroke-width:2px
    style LayerNorm fill:#FFFFE0,stroke:#333,stroke-width:2px
    style Embedding fill:#FFD700,stroke:#006400,stroke-width:3px
    style Dots fill:none,stroke:none
```

**Note**: Each residual block contains an internal skip connection (not shown in this high-level diagram for clarity).


## Tabular Residual Block

Now let's implement the core building block. This code demonstrates how we adapt the residual connection concept from Part 1 to work with tabular data. If you're familiar with image ResNets, the key changes are:
1. **Linear layers** instead of Conv2d (tabular data has no spatial structure)
2. **BatchNorm1d** instead of BatchNorm2d (1D feature vectors, not 2D images)
3. **Dropout** added for regularization (critical for preventing overfitting on tabular data)

**Why this code matters**: This block is the foundation of TabularResNet. Understanding how residual connections work with fully connected layers will help you adapt ResNet to other non-image domains.

```{code-cell}
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularResidualBlock(nn.Module):
    """
    Residual block for tabular data using fully connected layers.

    Implements the core ResNet skip connection (H(x) = F(x) + x) adapted for tabular data:
    - Uses Linear layers (tabular data has no spatial structure like images)
    - Uses BatchNorm1d for 1D feature vectors
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

Now we'll build the complete model by combining all the pieces: categorical embeddings, numerical feature processing, and stacked residual blocks. This code shows you:

1. **How to handle mixed data types** (categorical + numerical) in a single model
2. **Embedding dimensions** for categorical features with different cardinalities
3. **Feature concatenation** strategy to combine different input types
4. **Embedding extraction** for downstream anomaly detection tasks

**Real-world application**: This is the exact architecture you'll use in Part 3 for self-supervised training on OCSF data. The `return_embedding=True` mode extracts the dense vector representations we'll use for anomaly detection via vector database similarity search.

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

        if cat_embeddings:
            cat_embed = torch.cat(cat_embeddings, dim=1)
        else:
            cat_embed = torch.empty(batch_size, 0)

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
# Numerical: network_bytes_in, duration, etc.
num_numerical = 50
# Categorical: user_id, status_id, entity_id, etc.
categorical_cardinalities = [100, 50, 200, 1000]

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
   - Use **hashing trick**: Apply a hash function to map unbounded values to fixed indices:
     `hash(entity_id) % embedding_size`. For example, `hash("user_abc123") % 1000 = 456`
     maps to embedding index 456. This allows handling millions of unique values with a
     fixed embedding table size, though different values may collide (share the same embedding).
   - Or learn a shared "unknown" embedding for rare values (values seen < N times in training)
   - Consider using larger embedding dimensions for high-cardinality features
     (e.g., 128-dim for `user_id` vs 32-dim for `status_code`)

3. **Missing Values**: OCSF records may have sparse fields
   - Add a special "missing" category for categorical features
   - Use zero or mean imputation for numerical features
   - Or add a binary "is_missing" indicator feature

4. **Temporal Features**: For `timestamp` fields
   - Extract: hour_of_day, day_of_week, time_since_last_event
   - Treat as numerical or cyclical (sin/cos encoding) features

### Adapting to Other Observability Data

While this series uses OCSF security logs, the same architecture works for any structured
observability data. Here's how to adapt the feature selection for different data types:

**Telemetry/Metrics Data:**
```python
# Example: Prometheus-style metrics with labels
categorical_features = ['host', 'service', 'metric_name',
                       'environment', 'region']
numerical_features = ['value', 'hour_of_day', 'day_of_week',
                     'moving_avg_1h', 'std_dev_1h']
# High-cardinality: 'host' (thousands), 'service' (hundreds)
```

**Distributed Traces:**
```python
# Example: OpenTelemetry span data
categorical_features = ['service_name', 'operation', 'status_code',
                       'error_type', 'parent_span_id']
numerical_features = ['duration_ms', 'span_count', 'error_count',
                     'queue_time_ms']
# High-cardinality: 'parent_span_id', 'trace_id' (millions)
```

**Configuration Data:**
```python
# Example: Kubernetes configs, deployment manifests
categorical_features = ['resource_type', 'namespace',
                       'deployment_strategy', 'image_tag']
numerical_features = ['replica_count', 'cpu_limit', 'memory_limit',
                     'version_number']
# High-cardinality: 'image_tag', 'config_hash'
```

**Application Logs (JSON/Structured):**
```python
# Example: Application event logs
categorical_features = ['log_level', 'component', 'user_id',
                       'transaction_type', 'error_code']
numerical_features = ['response_time_ms', 'bytes_processed',
                     'retry_count', 'cache_hit_rate']
# High-cardinality: 'user_id', 'session_id', 'transaction_id'
```

**Key principle**: Any data that can be represented as `(categorical_features, numerical_features)` pairs works with TabularResNet. The embedding model learns representations specific to your domain.

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

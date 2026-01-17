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

# Part 3: Self-Supervised Training [DRAFT]

Learn how to train TabularResNet on unlabelled OCSF data using self-supervised learning techniques.

## What is Self-Supervised Learning?

**The challenge**: You have millions of OCSF security logs but no labels telling you which are "normal" vs "anomalous". Traditional supervised learning requires labeled data (e.g., "this event is malicious"), which is expensive and often unavailable for new anomaly types.

**The solution**: **Self-supervised learning** creates training tasks automatically from the data itself, without human labels. The model learns useful representations by solving these artificial tasks.

**Analogy**: Think of it like learning a language by filling in blanks. If you read "The cat sat on the ___", you can learn about cats and furniture even without explicit teaching. The sentence structure itself provides the supervision.

For tabular data, we use two main self-supervised approaches:

## The Training Strategy

Since your observability data is **unlabelled**, you need self-supervised learning. Two effective approaches from the TabTransformer paper:

### 1. Masked Feature Prediction (MFP)

**The idea**: Hide some features in your data and train the model to predict what's missing.

**How it works**:
- Take an OCSF record: `{user_id: 12345, status: success, bytes: 1024, duration: 5.2}`
- Randomly mask 15% of features: `{user_id: [MASK], status: success, bytes: 1024, duration: [MASK]}`
- Train the model to predict the masked values: `user_id = 12345`, `duration = 5.2`

**Why this works**: To predict missing features, the model must learn relationships between features (e.g., "successful logins from user 12345 typically transfer ~1000 bytes in ~5 seconds"). These learned relationships create useful embeddings that capture "normal" behavior patterns.

**Practical code example**:

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

### 2. Contrastive Learning

**The idea**: Train the model so that similar records have similar embeddings, while different records have different embeddings.

**How it works**:
1. Take an OCSF record (e.g., a login event)
2. Create two slightly different versions by adding noise (e.g., add ±5% to `bytes`, randomly change some categorical values)
3. Train the model so these two versions have **similar embeddings** (they're "positive pairs")
4. Meanwhile, ensure embeddings from different records stay **far apart** (they're "negative pairs")

**Analogy**: It's like teaching someone to recognize faces. Show them two photos of the same person from different angles and say "these are the same." Show them photos of different people and say "these are different." Over time, they learn what makes faces similar or different.

**Why this works**: The model learns which variations in features are superficial (noise) vs meaningful (different events). Records with similar embeddings represent similar system behaviors, making it easy to detect anomalies as records with unusual embeddings.

**Key terms**:
- **Augmentation**: Creating slightly modified versions of data (e.g., adding noise to numerical features)
- **Positive pairs**: Two augmented versions of the same record (should have similar embeddings)
- **Negative pairs**: Augmented versions from different records (should have different embeddings)
- **Temperature**: A parameter controlling how strictly the model enforces similarity (lower = stricter)
- **SimCLR**: A popular contrastive learning framework we adapt for tabular data

**Implementation**:

```{code-cell}
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularAugmentation:
    """
    Data augmentation for tabular data used in contrastive learning.
    """
    def __init__(self, noise_level=0.1, dropout_prob=0.2):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob

    def augment_numerical(self, numerical_features):
        """Add Gaussian noise to numerical features."""
        noise = torch.randn_like(numerical_features) * self.noise_level
        return numerical_features + noise

    def augment_categorical(self, categorical_features, cardinalities):
        """Randomly replace some categorical features with random values."""
        augmented = categorical_features.clone()
        mask = torch.rand_like(categorical_features.float()) < self.dropout_prob

        for i, cardinality in enumerate(cardinalities):
            # Replace masked values with random categories
            random_cats = torch.randint(0, cardinality,
                                       (categorical_features.size(0),),
                                       device=categorical_features.device)
            augmented[:, i] = torch.where(mask[:, i], random_cats,
                                         categorical_features[:, i])

        return augmented

def contrastive_loss(model, numerical, categorical, cardinalities,
                    temperature=0.07, augmenter=None):
    """
    SimCLR-style contrastive loss for tabular data.

    Args:
        model: TabularResNet model
        numerical: (batch_size, num_numerical) numerical features
        categorical: (batch_size, num_categorical) categorical features
        cardinalities: List of cardinalities for categorical features
        temperature: Temperature parameter for softmax
        augmenter: TabularAugmentation instance

    Returns:
        Contrastive loss value
    """
    if augmenter is None:
        augmenter = TabularAugmentation()

    batch_size = numerical.size(0)

    # Create two augmented views of each sample
    # View 1: First augmentation
    num_aug1 = augmenter.augment_numerical(numerical)
    cat_aug1 = augmenter.augment_categorical(categorical, cardinalities)
    embeddings1 = model(num_aug1, cat_aug1, return_embedding=True)

    # View 2: Second augmentation (independent)
    num_aug2 = augmenter.augment_numerical(numerical)
    cat_aug2 = augmenter.augment_categorical(categorical, cardinalities)
    embeddings2 = model(num_aug2, cat_aug2, return_embedding=True)

    # Concatenate both views: [emb1_batch, emb2_batch]
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)

    # Normalize embeddings (important for cosine similarity)
    embeddings = F.normalize(embeddings, dim=1)

    # Compute similarity matrix: (2*batch_size, 2*batch_size)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
    # For sample i: positive is at index i+batch_size
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),  # For first half
        torch.arange(0, batch_size)                 # For second half
    ], dim=0).to(numerical.device)

    # Mask to remove self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=numerical.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # Compute cross-entropy loss
    # Each row: softmax over all other samples, target is the positive pair
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss

# Example usage (needs TabularResNet from Part 2)
print("Contrastive learning setup complete")
print("Use contrastive_loss() with your TabularResNet model for self-supervised training")
```

## Complete Training Loop

Now let's put it all together. The code below shows a complete training pipeline including:
1. **Dataset creation** from OCSF numerical and categorical features
2. **Training function** that runs contrastive learning for multiple epochs
3. **Example usage** with simulated OCSF data

**Why this code matters**: This is the actual training loop you'll use in production. Understanding the data flow (Dataset → DataLoader → Model → Loss → Optimizer) is crucial for customizing the training process to your specific OCSF schema.

```{code-cell}
from torch.utils.data import Dataset, DataLoader

class OCSFDataset(Dataset):
    """
    Dataset for OCSF observability data.
    """
    def __init__(self, numerical_data, categorical_data):
        """
        Args:
            numerical_data: (num_samples, num_numerical_features) numpy array
            categorical_data: (num_samples, num_categorical_features) numpy array
        """
        self.numerical = torch.FloatTensor(numerical_data)
        self.categorical = torch.LongTensor(categorical_data)

    def __len__(self):
        return len(self.numerical)

    def __getitem__(self, idx):
        return self.numerical[idx], self.categorical[idx]

def train_self_supervised(model, dataloader, optimizer, cardinalities,
                         num_epochs=50, device='cpu'):
    """
    Train TabularResNet using contrastive learning.

    Args:
        model: TabularResNet model
        dataloader: DataLoader with (numerical, categorical) batches
        optimizer: PyTorch optimizer
        cardinalities: List of categorical feature cardinalities
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'

    Returns:
        List of loss values per epoch
    """
    model = model.to(device)
    model.train()

    augmenter = TabularAugmentation(noise_level=0.1, dropout_prob=0.2)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for numerical, categorical in dataloader:
            numerical = numerical.to(device)
            categorical = categorical.to(device)

            # Compute contrastive loss
            loss = contrastive_loss(model, numerical, categorical,
                                   cardinalities, augmenter=augmenter)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")

    return losses

# Example usage (with dummy data)
import numpy as np

# Simulate OCSF data
num_samples = 10000
num_numerical = 50
num_categorical = 4
categorical_cardinalities = [100, 50, 200, 1000]

# Create dummy dataset
np.random.seed(42)
numerical_data = np.random.randn(num_samples, num_numerical)
categorical_data = np.random.randint(0, 50, (num_samples, num_categorical))

# Create dataset and dataloader
dataset = OCSFDataset(numerical_data, categorical_data)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

print(f"Dataset created: {len(dataset)} samples")
print(f"Numerical features: {num_numerical}")
print(f"Categorical features: {num_categorical}")
print(f"Ready for self-supervised training")
```

## Training Best Practices

**Note on Data Types**: While the examples use OCSF security logs, self-supervised learning works identically for **any structured observability data** (telemetry, traces, configs, application logs). The training pipeline (`Dataset → DataLoader → Model → Loss`) remains the same—just change the feature columns to match your data schema.

### 1. Data Preprocessing

Before training, you need to prepare your data (OCSF logs, metrics, traces, etc.). This involves:
- **StandardScaler**: Normalizes numerical features to have mean=0 and std=1 (prevents features with large values from dominating)
- **LabelEncoder**: Converts categorical strings to integers (e.g., `"login" → 0`, `"logout" → 1`)

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_ocsf_data(df, numerical_cols, categorical_cols):
    """
    Preprocess OCSF DataFrame for training.

    Args:
        df: Pandas DataFrame with OCSF data
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names

    Returns:
        (numerical_array, categorical_array, encoders, scaler)
    """
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_data = scaler.fit_transform(df[numerical_cols].fillna(0))

    # Encode categorical features
    encoders = {}
    categorical_data = []

    for col in categorical_cols:
        encoder = LabelEncoder()
        # Handle unseen categories by adding an "unknown" category
        encoded = encoder.fit_transform(df[col].fillna('MISSING').astype(str))
        categorical_data.append(encoded)
        encoders[col] = encoder

    categorical_data = np.column_stack(categorical_data)

    return numerical_data, categorical_data, encoders, scaler
```

### 2. Hyperparameter Tuning

Hyperparameters control how your model learns and generalize. Getting them right can mean the difference between embeddings that clearly separate anomalies and embeddings that don't work at all.

**Key hyperparameters to tune**:

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `d_model` | 256 | 128-512 | Embedding capacity |
| `num_blocks` | 6 | 4-12 | Model depth |
| `dropout` | 0.1 | 0.0-0.3 | Regularization |
| `batch_size` | 256 | 64-512 | Training stability |
| `learning_rate` | 1e-3 | 1e-4 to 1e-2 | Convergence speed |
| `temperature` | 0.07 | 0.05-0.2 | Contrastive learning difficulty |

**Recommendation**: Start with defaults, then tune learning rate and batch size first.

### 3. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# Example training with LR scheduling
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

for epoch in range(num_epochs):
    # Training loop
    train_loss = train_epoch(...)

    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch}: Loss={train_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
```

### 4. Early Stopping

```python
class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Practical Workflow for OCSF Data

### Step-by-Step Training Pipeline

```python
def train_ocsf_anomaly_detector(
    ocsf_df,
    numerical_cols,
    categorical_cols,
    d_model=256,
    num_blocks=6,
    num_epochs=100,
    batch_size=256,
    device='cuda'
):
    """
    Complete pipeline to train custom TabularResNet embedding model on OCSF data.

    Args:
        ocsf_df: DataFrame with OCSF records
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        d_model: Model hidden dimension
        num_blocks: Number of residual blocks
        num_epochs: Training epochs
        batch_size: Batch size
        device: 'cuda' or 'cpu'

    Returns:
        Trained model, scaler, and encoders
    """
    # 1. Preprocess data
    print("Preprocessing data...")
    numerical_data, categorical_data, encoders, scaler = preprocess_ocsf_data(
        ocsf_df, numerical_cols, categorical_cols
    )

    # 2. Create dataset and dataloader
    print("Creating dataset...")
    dataset = OCSFDataset(numerical_data, categorical_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 3. Initialize model
    print("Initializing model...")
    categorical_cardinalities = [
        len(encoders[col].classes_) for col in categorical_cols
    ]

    # Import TabularResNet from Part 2
    from part2_tabular_resnet import TabularResNet

    model = TabularResNet(
        num_numerical_features=len(numerical_cols),
        categorical_cardinalities=categorical_cardinalities,
        d_model=d_model,
        num_blocks=num_blocks,
        dropout=0.1,
        num_classes=None  # Embedding mode
    )

    # 4. Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 5. Train
    print(f"Training for {num_epochs} epochs...")
    losses = train_self_supervised(
        model, dataloader, optimizer, categorical_cardinalities,
        num_epochs=num_epochs, device=device
    )

    # 6. Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'encoders': encoders,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'hyperparameters': {
            'd_model': d_model,
            'num_blocks': num_blocks,
            'categorical_cardinalities': categorical_cardinalities
        }
    }

    torch.save(checkpoint, 'ocsf_anomaly_detector.pt')
    print("Model saved to ocsf_anomaly_detector.pt")

    return model, scaler, encoders

# Example usage (uncomment to run)
# model, scaler, encoders = train_ocsf_anomaly_detector(
#     ocsf_df=my_dataframe,
#     numerical_cols=['network_bytes_in', 'duration', ...],
#     categorical_cols=['user_id', 'status_id', 'entity_id', ...],
#     num_epochs=50
# )
```

---

## Summary

In this part, you learned:

1. **Two self-supervised approaches**: Masked Feature Prediction and Contrastive Learning
2. **Contrastive learning implementation** with TabularAugmentation
3. **Complete training loop** with best practices
4. **Hyperparameter tuning** strategies
5. **End-to-end pipeline** for OCSF data

**Next**: In [Part 4](part4-embedding-quality), we'll learn how to evaluate the quality of learned embeddings using visualization and quantitative metrics.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

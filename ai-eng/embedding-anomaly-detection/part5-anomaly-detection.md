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

# Part 5: Anomaly Detection Methods

::::{grid} 1
:gutter: 2

:::{grid-item-card} Series: Embedding-Based Anomaly Detection for Observability
:link: index
:link-type: doc

[← Part 4: Evaluating Embedding Quality](part4-embedding-quality) | **Part 5 of 7** | [Next: Part 6 - Production Deployment →](part6-production-deployment)

Apply various anomaly detection algorithms to your validated embeddings for OCSF observability data.
:::
::::

## Overview of Anomaly Detection Methods

Once you have high-quality embeddings, you can detect anomalies using:

1. **Density-based**: Local Outlier Factor (LOF)
2. **Tree-based**: Isolation Forest
3. **Distance-based**: k-NN distance, Mahalanobis distance
4. **Clustering-based**: Distance from cluster centroids
5. **Sequence-based**: Multi-record anomalies (LSTM, Transformer)

Each method has different strengths. We'll implement all of them and compare.

---

## 1. Local Outlier Factor (LOF)

LOF identifies outliers based on local density deviation.

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def detect_anomalies_lof(embeddings, contamination=0.1, n_neighbors=20):
    """
    Detect anomalies using Local Outlier Factor.

    Args:
        embeddings: (num_samples, embedding_dim) array
        contamination: Expected proportion of anomalies
        n_neighbors: Number of neighbors for density estimation

    Returns:
        predictions: -1 for anomalies, 1 for normal
        scores: Negative outlier factor (more negative = more anomalous)
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(embeddings)

    # Negative outlier scores (use negative_outlier_factor_)
    scores = lof.negative_outlier_factor_

    return predictions, scores

# Simulate embeddings with anomalies
np.random.seed(42)

# Normal data: 3 clusters
normal_cluster1 = np.random.randn(200, 256) * 0.5
normal_cluster2 = np.random.randn(200, 256) * 0.5 + 3.0
normal_cluster3 = np.random.randn(200, 256) * 0.5 - 3.0
normal_embeddings = np.vstack([normal_cluster1, normal_cluster2, normal_cluster3])

# Anomalies: scattered outliers
anomaly_embeddings = np.random.uniform(-8, 8, (60, 256))

all_embeddings = np.vstack([normal_embeddings, anomaly_embeddings])
true_labels = np.array([0]*600 + [1]*60)  # 0=normal, 1=anomaly

# Detect anomalies
predictions, scores = detect_anomalies_lof(all_embeddings, contamination=0.1, n_neighbors=20)

# Convert predictions: -1 (anomaly) → 1, 1 (normal) → 0
predicted_labels = (predictions == -1).astype(int)

# Evaluate
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Local Outlier Factor (LOF) Results:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-Score:  {f1:.3f}")
print(f"\nInterpretation:")
print(f"  Precision = {precision:.1%} of flagged items are true anomalies")
print(f"  Recall = {recall:.1%} of true anomalies were detected")
```

### Visualizing LOF Results

```{code-cell}
from sklearn.manifold import TSNE

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Ground truth
ax1.scatter(embeddings_2d[true_labels==0, 0], embeddings_2d[true_labels==0, 1],
            c='blue', alpha=0.6, label='Normal', s=30)
ax1.scatter(embeddings_2d[true_labels==1, 0], embeddings_2d[true_labels==1, 1],
            c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
ax1.set_title('Ground Truth', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# LOF predictions
ax2.scatter(embeddings_2d[predicted_labels==0, 0], embeddings_2d[predicted_labels==0, 1],
            c='blue', alpha=0.6, label='Predicted Normal', s=30)
ax2.scatter(embeddings_2d[predicted_labels==1, 0], embeddings_2d[predicted_labels==1, 1],
            c='red', alpha=0.8, label='Predicted Anomaly', s=50, marker='x')
ax2.set_title(f'LOF Predictions (F1={f1:.3f})', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. Isolation Forest

Isolation Forest isolates anomalies by randomly partitioning the data.

```{code-cell}
from sklearn.ensemble import IsolationForest

def detect_anomalies_iforest(embeddings, contamination=0.1, n_estimators=100):
    """
    Detect anomalies using Isolation Forest.

    Args:
        embeddings: Embedding vectors
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees

    Returns:
        predictions, scores
    """
    iforest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    predictions = iforest.fit_predict(embeddings)
    scores = iforest.score_samples(embeddings)  # Lower = more anomalous

    return predictions, scores

# Detect anomalies
predictions_if, scores_if = detect_anomalies_iforest(all_embeddings, contamination=0.1)

# Convert predictions
predicted_labels_if = (predictions_if == -1).astype(int)

# Evaluate
precision_if = precision_score(true_labels, predicted_labels_if)
recall_if = recall_score(true_labels, predicted_labels_if)
f1_if = f1_score(true_labels, predicted_labels_if)

print(f"Isolation Forest Results:")
print(f"  Precision: {precision_if:.3f}")
print(f"  Recall:    {recall_if:.3f}")
print(f"  F1-Score:  {f1_if:.3f}")
```

---

## 3. Distance-Based Methods

### k-NN Distance

Points with large distance to k-th nearest neighbor are anomalies.

```{code-cell}
from sklearn.neighbors import NearestNeighbors

def detect_anomalies_knn(embeddings, k=5, threshold_percentile=90):
    """
    Detect anomalies using k-NN distance.

    Args:
        embeddings: Embedding vectors
        k: Number of nearest neighbors
        threshold_percentile: Percentile for anomaly threshold

    Returns:
        predictions, scores
    """
    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1)  # +1 because point itself is included
    nbrs.fit(embeddings)

    # Compute distances to k-th nearest neighbor
    distances, indices = nbrs.kneighbors(embeddings)
    knn_distances = distances[:, -1]  # Distance to k-th neighbor

    # Threshold: anomalies are in top (100-threshold_percentile)%
    threshold = np.percentile(knn_distances, threshold_percentile)
    predictions = (knn_distances > threshold).astype(int)

    return predictions, knn_distances

# Detect anomalies
predicted_labels_knn, scores_knn = detect_anomalies_knn(all_embeddings, k=5, threshold_percentile=90)

# Evaluate
precision_knn = precision_score(true_labels, predicted_labels_knn)
recall_knn = recall_score(true_labels, predicted_labels_knn)
f1_knn = f1_score(true_labels, predicted_labels_knn)

print(f"k-NN Distance Results:")
print(f"  Precision: {precision_knn:.3f}")
print(f"  Recall:    {recall_knn:.3f}")
print(f"  F1-Score:  {f1_knn:.3f}")
```

---

## 4. Mahalanobis Distance

Accounts for correlations in the data.

```{code-cell}
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def detect_anomalies_mahalanobis(embeddings, threshold_pvalue=0.01):
    """
    Detect anomalies using Mahalanobis distance.

    Args:
        embeddings: Embedding vectors
        threshold_pvalue: P-value threshold for chi-squared test

    Returns:
        predictions, scores
    """
    # Compute mean and covariance
    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings.T)

    # Add small regularization to avoid singular matrix
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        print("Warning: Singular covariance matrix, using pseudo-inverse")
        cov_inv = np.linalg.pinv(cov_reg)

    # Compute Mahalanobis distance for each point
    mahal_distances = np.array([
        mahalanobis(x, mean, cov_inv) for x in embeddings
    ])

    # Chi-squared threshold
    df = embeddings.shape[1]  # Degrees of freedom = embedding dimension
    threshold = chi2.ppf(1 - threshold_pvalue, df)

    # Anomalies: Mahalanobis distance exceeds threshold
    predictions = (mahal_distances > np.sqrt(threshold)).astype(int)

    return predictions, mahal_distances

# Detect anomalies
predicted_labels_mahal, scores_mahal = detect_anomalies_mahalanobis(all_embeddings, threshold_pvalue=0.01)

# Evaluate
precision_mahal = precision_score(true_labels, predicted_labels_mahal)
recall_mahal = recall_score(true_labels, predicted_labels_mahal)
f1_mahal = f1_score(true_labels, predicted_labels_mahal)

print(f"Mahalanobis Distance Results:")
print(f"  Precision: {precision_mahal:.3f}")
print(f"  Recall:    {recall_mahal:.3f}")
print(f"  F1-Score:  {f1_mahal:.3f}")
```

---

## 5. Clustering-Based Anomaly Detection

```{code-cell}
from sklearn.cluster import KMeans

def detect_anomalies_clustering(embeddings, n_clusters=3, threshold_percentile=95):
    """
    Detect anomalies as points far from cluster centroids.

    Args:
        embeddings: Embedding vectors
        n_clusters: Number of clusters
        threshold_percentile: Distance percentile for anomaly threshold

    Returns:
        predictions, scores
    """
    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    # Compute distance to nearest cluster centroid
    distances_to_centroids = np.min(kmeans.transform(embeddings), axis=1)

    # Threshold
    threshold = np.percentile(distances_to_centroids, threshold_percentile)
    predictions = (distances_to_centroids > threshold).astype(int)

    return predictions, distances_to_centroids

# Detect anomalies
predicted_labels_cluster, scores_cluster = detect_anomalies_clustering(
    all_embeddings, n_clusters=3, threshold_percentile=95
)

# Evaluate
precision_cluster = precision_score(true_labels, predicted_labels_cluster)
recall_cluster = recall_score(true_labels, predicted_labels_cluster)
f1_cluster = f1_score(true_labels, predicted_labels_cluster)

print(f"Clustering-Based Results:")
print(f"  Precision: {precision_cluster:.3f}")
print(f"  Recall:    {recall_cluster:.3f}")
print(f"  F1-Score:  {f1_cluster:.3f}")
```

---

## 6. Multi-Record Sequence Anomaly Detection

For detecting anomalies across sequences of events (e.g., multi-step attacks).

```{code-cell}
import torch
import torch.nn as nn

class SequenceAnomalyDetector(nn.Module):
    """
    Detect anomalies in sequences of events using embeddings.
    """
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()

        # LSTM to model sequences of embeddings
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

print(f"\nSequence Anomaly Detection:")
print(f"  Sequence shape: {sequence.shape}")
print(f"  Normality score: {normality_score.item():.3f}")
print(f"  Interpretation: Lower score = more likely anomaly sequence")
print(f"\nUse case: Detect multi-step attacks or unusual event patterns")
```

---

## 7. Method Comparison

```{code-cell}
def compare_anomaly_methods(embeddings, true_labels):
    """
    Compare all anomaly detection methods.

    Args:
        embeddings: Embedding vectors
        true_labels: Ground truth (0=normal, 1=anomaly)

    Returns:
        Comparison DataFrame
    """
    results = []

    # LOF
    pred_lof, _ = detect_anomalies_lof(embeddings, contamination=0.1)
    pred_lof = (pred_lof == -1).astype(int)
    results.append({
        'Method': 'Local Outlier Factor',
        'Precision': precision_score(true_labels, pred_lof),
        'Recall': recall_score(true_labels, pred_lof),
        'F1-Score': f1_score(true_labels, pred_lof)
    })

    # Isolation Forest
    pred_if, _ = detect_anomalies_iforest(embeddings, contamination=0.1)
    pred_if = (pred_if == -1).astype(int)
    results.append({
        'Method': 'Isolation Forest',
        'Precision': precision_score(true_labels, pred_if),
        'Recall': recall_score(true_labels, pred_if),
        'F1-Score': f1_score(true_labels, pred_if)
    })

    # k-NN Distance
    pred_knn, _ = detect_anomalies_knn(embeddings, k=5)
    results.append({
        'Method': 'k-NN Distance',
        'Precision': precision_score(true_labels, pred_knn),
        'Recall': recall_score(true_labels, pred_knn),
        'F1-Score': f1_score(true_labels, pred_knn)
    })

    # Mahalanobis
    pred_mahal, _ = detect_anomalies_mahalanobis(embeddings)
    results.append({
        'Method': 'Mahalanobis Distance',
        'Precision': precision_score(true_labels, pred_mahal),
        'Recall': recall_score(true_labels, pred_mahal),
        'F1-Score': f1_score(true_labels, pred_mahal)
    })

    # Clustering
    pred_cluster, _ = detect_anomalies_clustering(embeddings, n_clusters=3)
    results.append({
        'Method': 'Clustering-Based',
        'Precision': precision_score(true_labels, pred_cluster),
        'Recall': recall_score(true_labels, pred_cluster),
        'F1-Score': f1_score(true_labels, pred_cluster)
    })

    # Sort by F1-Score
    results = sorted(results, key=lambda x: x['F1-Score'], reverse=True)

    # Print comparison table
    print("\n" + "="*70)
    print("ANOMALY DETECTION METHOD COMPARISON")
    print("="*70)
    print(f"{'Rank':<6} {'Method':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r['Method']:<25} {r['Precision']:<12.3f} {r['Recall']:<12.3f} {r['F1-Score']:<12.3f}")
    print("="*70)

    return results

# Run comparison
comparison_results = compare_anomaly_methods(all_embeddings, true_labels)
```

---

## 8. Threshold Tuning

### Precision-Recall Curve

```{code-cell}
from sklearn.metrics import precision_recall_curve, auc

def plot_precision_recall_curve(true_labels, scores, method_name):
    """
    Plot precision-recall curve for threshold tuning.

    Args:
        true_labels: Ground truth
        scores: Anomaly scores (higher = more anomalous)
        method_name: Name of the method
    """
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'{method_name} (AUC={pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve: {method_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nPrecision-Recall AUC: {pr_auc:.3f}")
    print(f"Use this curve to select the best threshold for your use case")

# Example: Plot PR curve for k-NN distance
plot_precision_recall_curve(true_labels, scores_knn, "k-NN Distance")
```

---

## 9. Production Pipeline

### Complete Anomaly Detection Pipeline

```{code-cell}
class AnomalyDetectionPipeline:
    """
    Production-ready anomaly detection pipeline.
    """
    def __init__(self, model, scaler, encoders, method='lof', contamination=0.1):
        """
        Args:
            model: Trained TabularResNet
            scaler: Fitted StandardScaler for numerical features
            encoders: Dict of LabelEncoders for categorical features
            method: 'lof', 'iforest', 'knn', or 'ensemble'
            contamination: Expected anomaly rate
        """
        self.model = model
        self.scaler = scaler
        self.encoders = encoders
        self.method = method
        self.contamination = contamination
        self.detector = None

    def fit(self, embeddings):
        """Fit the anomaly detector on normal data."""
        if self.method == 'lof':
            self.detector = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True  # For online prediction
            )
        elif self.method == 'iforest':
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == 'knn':
            self.detector = NearestNeighbors(n_neighbors=5)

        self.detector.fit(embeddings)
        print(f"Anomaly detector ({self.method}) fitted on {len(embeddings)} samples")

    def predict(self, ocsf_records):
        """
        Predict anomalies for new OCSF records.

        Args:
            ocsf_records: List of OCSF dictionaries or DataFrame

        Returns:
            predictions: Binary array (1=anomaly, 0=normal)
            scores: Anomaly scores
        """
        # TODO: Implement preprocessing and embedding extraction
        # This is a simplified example
        pass

print("AnomalyDetectionPipeline class defined")
print("Usage:")
print("  pipeline = AnomalyDetectionPipeline(model, scaler, encoders, method='lof')")
print("  pipeline.fit(training_embeddings)")
print("  predictions, scores = pipeline.predict(new_ocsf_records)")
```

---

## Summary

In this part, you learned:

1. **Five anomaly detection methods**: LOF, Isolation Forest, k-NN, Mahalanobis, Clustering
2. **Sequence anomaly detection** using LSTMs for multi-record patterns
3. **Method comparison** framework for selecting the best approach
4. **Threshold tuning** using precision-recall curves
5. **Production pipeline** for real-time anomaly detection

**Key Takeaways:**
- **Isolation Forest** works well for high-dimensional embeddings
- **LOF** is good for detecting local density deviations
- **Ensemble methods** combining multiple detectors often perform best
- **Tune thresholds** based on your precision/recall requirements

**Next**: In [Part 6](part6-production-deployment), we'll deploy this system to production with REST APIs, model serving, and integration with observability platforms.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

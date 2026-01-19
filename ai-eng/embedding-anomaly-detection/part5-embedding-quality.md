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

# Part 5: Evaluating Embedding Quality

Learn how to evaluate and validate the quality of learned embeddings before deploying to production.

## Why Evaluate Embeddings?

After training your TabularResNet using self-supervised learning ([Part 4](part4-self-supervised-training)), you need to verify that the embeddings are:

1. **Meaningful**: Similar records cluster together
2. **Discriminative**: Different types of records are separated
3. **Robust**: Small input perturbations don't drastically change embeddings
4. **Useful**: Enable effective anomaly detection downstream

Poor embeddings lead to poor anomaly detection. This part teaches you how to measure and improve embedding quality.

---

## Evaluation Techniques Overview

We'll use four complementary approaches to evaluate embedding quality:

1. **Visualization** (t-SNE, UMAP): See how embeddings cluster in 2D space
2. **Cluster metrics** (Silhouette, Davies-Bouldin): Quantify cluster separation
3. **Robustness testing**: Verify embeddings are stable under perturbations
4. **Downstream task performance**: Test on actual anomaly detection

**Key terminology**:
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding): Reduces high-dimensional embeddings to 2D while preserving local structure. Good for visualization but can distort global relationships.
- **UMAP** (Uniform Manifold Approximation and Projection): Similar to t-SNE but better preserves global structure. Generally faster and more scalable.
- **Perplexity**: A t-SNE parameter that balances attention between local and global aspects (think of it as "expected number of neighbors"). Typical values: 5-50.
- **Silhouette Score**: Measures how similar a point is to its own cluster vs other clusters. Range: -1 to +1 (higher is better).
- **Davies-Bouldin Index**: Measures average similarity between each cluster and its most similar one. Lower values indicate better separation.

---

## 1. Embedding Space Visualization

### t-SNE Visualization

Now let's visualize embeddings in 2D. This code demonstrates how to use t-SNE to project high-dimensional embeddings (e.g., 256-dim) into 2D for visualization. Look for clear cluster separation - anomalies should appear as outliers or in sparse regions.

```{code-cell}
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_embeddings_tsne(embeddings, labels=None, title="Embedding Space (t-SNE)", perplexity=30):
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: (num_samples, embedding_dim) numpy array
        labels: Optional labels for coloring points
        title: Plot title
        perplexity: t-SNE perplexity parameter (5-50 typical)

    Returns:
        matplotlib figure
    """
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=f"Class {label}", alpha=0.6, s=30)

        ax.legend(loc='best')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=30)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Example: Simulate embeddings for normal and anomalous data
np.random.seed(42)

# Normal data: 3 clusters
normal_cluster1 = np.random.randn(200, 256) * 0.5 + np.array([0, 0] + [0]*254)
normal_cluster2 = np.random.randn(200, 256) * 0.5 + np.array([3, 3] + [0]*254)
normal_cluster3 = np.random.randn(200, 256) * 0.5 + np.array([-3, 3] + [0]*254)

# Anomalies: scattered outliers
anomalies = np.random.randn(60, 256) * 2.0 + np.array([5, -5] + [0]*254)

all_embeddings = np.vstack([normal_cluster1, normal_cluster2, normal_cluster3, anomalies])
labels = np.array([0]*200 + [1]*200 + [2]*200 + [3]*60)

fig = visualize_embeddings_tsne(all_embeddings, labels, title="OCSF Embeddings (t-SNE)")
plt.show()

print("✓ t-SNE visualization complete")
print("  - Look for clear cluster separation")
print("  - Anomalies should be outliers or in sparse regions")
```

### UMAP Visualization

UMAP (Uniform Manifold Approximation and Projection) often preserves global structure better than t-SNE.

```{code-cell}
# Note: UMAP requires installation: pip install umap-learn
# This is a code example (not executed in the tutorial)

def visualize_embeddings_umap(embeddings, labels=None, title="Embedding Space (UMAP)", n_neighbors=15):
    """
    Visualize embeddings using UMAP.

    Args:
        embeddings: (num_samples, embedding_dim) numpy array
        labels: Optional labels for coloring
        title: Plot title
        n_neighbors: UMAP n_neighbors parameter (5-50 typical)

    Returns:
        matplotlib figure
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None

    # Run UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot (same as t-SNE code)
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=f"Class {label}", alpha=0.6, s=30)

        ax.legend(loc='best')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=30)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

print("UMAP visualization function defined")
print("Usage: visualize_embeddings_umap(embeddings, labels)")
```

**When to use which?**

| Method | Best For | Preserves |
|--------|----------|-----------|
| **t-SNE** | Local structure, cluster identification | Neighborhoods |
| **UMAP** | Global structure, distance relationships | Both local & global |

---

## 2. Cluster Quality Metrics

### Silhouette Score

Measures how similar an object is to its own cluster compared to other clusters.

```{code-cell}
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans

def evaluate_cluster_quality(embeddings, n_clusters=3):
    """
    Evaluate clustering quality using silhouette score.

    Args:
        embeddings: (num_samples, embedding_dim) array
        n_clusters: Number of clusters to find

    Returns:
        Dictionary with metrics
    """
    # Run clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Overall silhouette score
    silhouette_avg = silhouette_score(embeddings, cluster_labels)

    # Per-sample silhouette scores
    sample_silhouette_values = silhouette_samples(embeddings, cluster_labels)

    metrics = {
        'silhouette_score': silhouette_avg,
        'cluster_labels': cluster_labels,
        'per_sample_scores': sample_silhouette_values,
        'cluster_sizes': np.bincount(cluster_labels)
    }

    return metrics

# Example
metrics = evaluate_cluster_quality(all_embeddings[:600], n_clusters=3)  # Only normal data

print(f"\nCluster Quality Metrics:")
print(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
print(f"  Interpretation:")
print(f"    1.0: Perfect separation")
print(f"    0.5-0.7: Reasonable structure")
print(f"    < 0.25: Poor clustering")
print(f"\n  Cluster sizes: {metrics['cluster_sizes']}")

# Visualize silhouette scores per cluster
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10

for i in range(3):
    # Get silhouette scores for cluster i
    ith_cluster_silhouette_values = metrics['per_sample_scores'][metrics['cluster_labels'] == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.tab10(i / 10.0)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
    y_lower = y_upper + 10

# Add average silhouette score line
ax.axvline(x=metrics['silhouette_score'], color="red", linestyle="--", label=f"Avg: {metrics['silhouette_score']:.3f}")

ax.set_title("Silhouette Plot for Clusters", fontsize=14, fontweight='bold')
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Cluster", fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()
```

### Davies-Bouldin Index

Lower values indicate better clustering (well-separated, compact clusters).

```{code-cell}
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def comprehensive_cluster_metrics(embeddings, n_clusters_range=range(2, 10)):
    """
    Compute multiple clustering metrics for different numbers of clusters.

    Args:
        embeddings: Embedding array
        n_clusters_range: Range of cluster counts to try

    Returns:
        DataFrame with metrics
    """
    results = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute metrics
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)

        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'inertia': kmeans.inertia_
        })

    return results

# Example
results = comprehensive_cluster_metrics(all_embeddings[:600])

print("\nClustering Metrics Across Different K:")
print(f"{'K':<5} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-Harabasz':<18}")
print("-" * 55)
for r in results:
    print(f"{r['n_clusters']:<5} {r['silhouette']:<12.3f} {r['davies_bouldin']:<16.3f} {r['calinski_harabasz']:<18.1f}")

print("\nInterpretation:")
print("  - Silhouette: Higher is better (max 1.0)")
print("  - Davies-Bouldin: Lower is better (min 0.0)")
print("  - Calinski-Harabasz: Higher is better (no upper bound)")
```

---

## 3. Embedding Robustness

### Perturbation Stability

Good embeddings should be stable under small input perturbations.

```{code-cell}
def evaluate_embedding_stability(model, numerical, categorical, num_perturbations=10, noise_level=0.1):
    """
    Evaluate embedding stability under input perturbations.

    Args:
        model: Trained TabularResNet
        numerical: Original numerical features
        categorical: Original categorical features
        num_perturbations: Number of perturbed versions
        noise_level: Std of Gaussian noise

    Returns:
        Average cosine similarity between original and perturbed embeddings
    """
    model.eval()

    with torch.no_grad():
        # Original embedding
        original_embedding = model(numerical, categorical, return_embedding=True)

        similarities = []

        for _ in range(num_perturbations):
            # Add noise to numerical features
            perturbed_numerical = numerical + torch.randn_like(numerical) * noise_level

            # Get perturbed embedding
            perturbed_embedding = model(perturbed_numerical, categorical, return_embedding=True)

            # Compute cosine similarity
            similarity = F.cosine_similarity(original_embedding, perturbed_embedding, dim=1)
            similarities.append(similarity.mean().item())

    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    print(f"Embedding Stability Test:")
    print(f"  Avg Cosine Similarity: {avg_similarity:.3f} ± {std_similarity:.3f}")
    print(f"  Interpretation:")
    print(f"    > 0.95: Very stable (robust to noise)")
    print(f"    0.85-0.95: Moderately stable")
    print(f"    < 0.85: Unstable (may need more training)")

    return avg_similarity, std_similarity

print("Embedding stability evaluation function defined")
print("Usage: evaluate_embedding_stability(model, numerical, categorical)")
```

---

## 4. Downstream Task Performance

### Proxy Task: K-NN Classification

If you have some labeled data, use k-NN accuracy as a proxy metric.

```{code-cell}
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def evaluate_knn_classification(embeddings, labels, k=5):
    """
    Evaluate embedding quality using k-NN classification.

    Args:
        embeddings: Embedding vectors
        labels: Ground truth labels
        k: Number of neighbors

    Returns:
        Cross-validated accuracy
    """
    knn = KNeighborsClassifier(n_neighbors=k)

    # 5-fold cross-validation
    scores = cross_val_score(knn, embeddings, labels, cv=5, scoring='accuracy')

    print(f"k-NN Classification (k={k}):")
    print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Interpretation: Higher accuracy = better embeddings")

    return scores.mean(), scores.std()

# Example with simulated labels
labels_subset = labels[:600]  # Only normal data (3 classes)
knn_acc, knn_std = evaluate_knn_classification(all_embeddings[:600], labels_subset, k=5)
```

---

## 5. Benchmark Comparison

### Compare Different Models

```{code-cell}
def compare_embedding_models(embeddings_dict, labels, metric='silhouette'):
    """
    Compare multiple embedding models.

    Args:
        embeddings_dict: Dict of {model_name: embeddings}
        labels: Ground truth labels
        metric: 'silhouette' or 'knn'

    Returns:
        Comparison results
    """
    results = []

    for model_name, embeddings in embeddings_dict.items():
        if metric == 'silhouette':
            # Cluster and compute silhouette
            kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            metric_name = "Silhouette"

        elif metric == 'knn':
            # k-NN accuracy
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = cross_val_score(knn, embeddings, labels, cv=5)
            score = scores.mean()
            metric_name = "k-NN Accuracy"

        results.append({
            'model': model_name,
            'score': score
        })

    # Sort by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"\nModel Comparison ({metric_name}):")
    print(f"{'Rank':<6} {'Model':<20} {'Score':<10}")
    print("-" * 40)
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r['model']:<20} {r['score']:.4f}")

    return results

# Example: Compare ResNet with different hyperparameters
embeddings_dict = {
    'ResNet-256-6blocks': all_embeddings[:600],  # Simulated
    'ResNet-128-4blocks': all_embeddings[:600] + np.random.randn(600, 256) * 0.05,  # Simulated
    'ResNet-512-8blocks': all_embeddings[:600] + np.random.randn(600, 256) * 0.03,  # Simulated
}

comparison = compare_embedding_models(embeddings_dict, labels_subset, metric='silhouette')
```

---

## 6. Production Checklist

Before deploying embeddings to production, verify:

### Quality Checklist

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| **Silhouette Score** | > 0.5 | ✓ |
| **Davies-Bouldin Index** | < 1.0 | ✓ |
| **k-NN Accuracy** (if labels available) | > 0.85 | ✓ |
| **Embedding Stability** | > 0.90 | ✓ |
| **Visual Inspection** | Clear clusters in t-SNE/UMAP | ✓ |
| **Anomaly Detection F1** | > 0.80 (Part 6) | Pending |

### Code: Automated Quality Report

```{code-cell}
def generate_embedding_quality_report(embeddings, labels=None, model=None, save_path='embedding_report.html'):
    """
    Generate comprehensive embedding quality report.

    Args:
        embeddings: Embedding vectors
        labels: Optional ground truth labels
        model: Optional trained model for stability testing
        save_path: Path to save HTML report

    Returns:
        Dictionary with all metrics
    """
    report = {
        'timestamp': np.datetime64('now'),
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1]
    }

    # 1. Cluster quality
    n_clusters = len(np.unique(labels)) if labels is not None else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    report['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
    report['davies_bouldin_index'] = davies_bouldin_score(embeddings, cluster_labels)
    report['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)

    # 2. k-NN if labels available
    if labels is not None:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_scores = cross_val_score(knn, embeddings, labels, cv=5)
        report['knn_accuracy_mean'] = knn_scores.mean()
        report['knn_accuracy_std'] = knn_scores.std()

    # 3. Quality verdict
    passed = report['silhouette_score'] > 0.5 and report['davies_bouldin_index'] < 1.0

    report['quality_verdict'] = 'PASS' if passed else 'FAIL'

    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING QUALITY REPORT")
    print("="*60)
    print(f"Samples: {report['num_samples']:,} | Embedding Dim: {report['embedding_dim']}")
    print(f"\nCluster Quality:")
    print(f"  Silhouette Score:      {report['silhouette_score']:.3f} {'✓' if report['silhouette_score'] > 0.5 else '✗'}")
    print(f"  Davies-Bouldin Index:  {report['davies_bouldin_index']:.3f} {'✓' if report['davies_bouldin_index'] < 1.0 else '✗'}")
    print(f"  Calinski-Harabasz:     {report['calinski_harabasz_score']:.1f}")

    if 'knn_accuracy_mean' in report:
        print(f"\nClassification (k-NN):")
        print(f"  Accuracy: {report['knn_accuracy_mean']:.3f} ± {report['knn_accuracy_std']:.3f}")

    print(f"\nVERDICT: {report['quality_verdict']}")
    print("="*60)

    return report

# Example
report = generate_embedding_quality_report(all_embeddings[:600], labels_subset)
```

---

## Summary

In this part, you learned:

1. **Visualization techniques** (t-SNE, UMAP) for exploring embedding space
2. **Cluster quality metrics** (Silhouette, Davies-Bouldin, Calinski-Harabasz)
3. **Robustness evaluation** through perturbation stability
4. **Downstream task performance** using k-NN classification
5. **Automated quality reporting** for production readiness

**Next**: In [Part 6](part6-anomaly-detection), we'll use these validated embeddings to detect anomalies using various algorithms (LOF, Isolation Forest, distance-based methods).

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

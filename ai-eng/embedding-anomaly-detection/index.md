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

# Embedding-Based Anomaly Detection for Observability

A comprehensive 7-part tutorial series on building production-ready anomaly detection systems using ResNet embeddings for OCSF observability data.

---

## Series Overview

This tutorial series takes you from ResNet fundamentals to deploying and monitoring a complete anomaly detection system in production. You'll learn how to:

- Build embeddings from high-dimensional tabular data
- Train models using self-supervised learning
- Detect anomalies using multiple algorithms
- Deploy to production with proper monitoring

**Target Audience**: ML engineers, security engineers, and data scientists working with observability data

**Prerequisites**:
- Basic Python and PyTorch
- Understanding of neural networks (or complete our [Neural Networks From Scratch](/ai-eng/nnfs/index.md) series first)

---

## Tutorial Series

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Part 1: Understanding ResNet Architecture
:link: part1-understanding-resnet
:link-type: doc

Learn the fundamentals of Residual Networks, including:
- The degradation problem in deep networks
- Residual connections and gradient flow
- Building complete ResNet architectures
- Training on CIFAR-10

**Foundation** · 45 min read
:::

:::{grid-item-card} Part 2: Adapting ResNet for Tabular Data
:link: part2-tabular-resnet
:link-type: doc

Adapt ResNet for observability data:
- Replace convolutions with linear layers
- Categorical embeddings for high-cardinality features
- Complete TabularResNet implementation
- Design considerations for OCSF data

**Architecture** · 30 min read
:::

:::{grid-item-card} Part 3: Self-Supervised Training
:link: part3-self-supervised-training
:link-type: doc

Train on unlabelled data:
- Masked Feature Prediction (MFP)
- Contrastive learning with augmentation
- Complete training pipeline
- Hyperparameter tuning strategies

**Training** · 35 min read
:::

:::{grid-item-card} Part 4: Evaluating Embedding Quality
:link: part4-embedding-quality
:link-type: doc

Validate embedding quality before deployment:
- t-SNE and UMAP visualization
- Cluster quality metrics (Silhouette, Davies-Bouldin)
- Embedding robustness testing
- Production readiness checklist

**Verification** · 30 min read
:::

:::{grid-item-card} Part 5: Anomaly Detection Methods
:link: part5-anomaly-detection
:link-type: doc

Apply detection algorithms:
- Local Outlier Factor (LOF)
- Isolation Forest
- Distance-based methods
- Sequence anomaly detection (LSTMs)
- Method comparison framework

**Detection** · 40 min read
:::

:::{grid-item-card} Part 6: Production Deployment
:link: part6-production-deployment
:link-type: doc

Deploy to production:
- REST API with FastAPI
- Docker containerization
- Model versioning with MLflow
- A/B testing framework
- Real-time vs batch inference

**Deployment** · 45 min read
:::

:::{grid-item-card} Part 7: Production Monitoring
:link: part7-production-monitoring
:link-type: doc

Monitor and maintain the system:
- Embedding drift detection
- Alert quality metrics
- Automated retraining triggers
- Incident response tools
- Cost optimization

**Monitoring** · 35 min read
:::

:::{grid-item-card} Complete Series
:class-card: sd-border-primary

**Total**: ~4 hours of comprehensive, hands-on content

All code examples are executable and production-ready.
:::

::::

---

## What You'll Build

By the end of this series, you'll have:

1. **TabularResNet Model**: Trained on OCSF observability data using self-supervised learning
2. **Anomaly Detector**: Multi-method ensemble (LOF + Isolation Forest)
3. **Production API**: FastAPI service with health checks and metrics
4. **Monitoring Dashboard**: Track drift, alert quality, and performance
5. **Retraining Pipeline**: Automated triggers based on performance degradation

### System Architecture

```{mermaid}
graph TB
    subgraph "Data Ingestion"
        OCSF[OCSF Data Stream]
    end

    subgraph "Feature Engineering"
        Preprocessor[Feature Extraction<br/>Categorical + Numerical]
    end

    subgraph "Model Serving"
        Embedding[TabularResNet<br/>Embedding Generation]
        Detector[Anomaly Detector<br/>LOF/IForest]
    end

    subgraph "Monitoring & Alerting"
        Drift[Drift Detection]
        Metrics[Quality Metrics]
        Alerts[Alert Manager]
    end

    subgraph "MLOps"
        Registry[Model Registry<br/>MLflow]
        Retraining[Auto Retraining]
    end

    OCSF --> Preprocessor
    Preprocessor --> Embedding
    Embedding --> Detector
    Detector --> Alerts

    Embedding -.Monitor.-> Drift
    Detector -.Monitor.-> Metrics

    Drift -.Trigger.-> Retraining
    Metrics -.Trigger.-> Retraining

    Retraining -.Update.-> Registry
    Registry -.Deploy.-> Embedding
    Registry -.Deploy.-> Detector

    style OCSF fill:#ADD8E6
    style Embedding fill:#90EE90
    style Detector fill:#FFA500
    style Drift fill:#FF6347
    style Metrics fill:#DDA0DD
```

---

## Key Concepts

### Why ResNet for Tabular Data?

Research by Gorishniy et al. (2021) found that ResNet:
- **Competes with Transformers** on tabular benchmarks
- **Simpler architecture**: No attention mechanism
- **Better efficiency**: O(n·d) vs O(d²) complexity
- **Strong baseline**: Try before complex models

### Why Embeddings for Anomaly Detection?

Embeddings compress high-dimensional OCSF data (300+ fields) into dense vectors that:
- Capture semantic relationships
- Enable efficient distance calculations
- Support multiple detection algorithms
- Generalize to new anomaly types

---

## Learning Path

### For ML Engineers
**Focus**: End-to-end production system
1. Part 1 (skim) → Part 2 → Part 3
2. Parts 4-5 (deep dive on evaluation)
3. Parts 6-7 (deployment and monitoring)

### For Security Engineers
**Focus**: Applying to OCSF data
1. Part 1 (overview only) → Part 2 (deep dive)
2. Part 5 (detection methods)
3. Part 7 (monitoring alerts)

### For Researchers
**Focus**: Model architecture and training
1. Parts 1-2 (architecture details)
2. Parts 3-4 (training and evaluation)
3. Compare with your own methods

---

## Code Repository

All code from this series is available in executable notebooks. Each part includes:
- **Runnable code cells**: Test concepts immediately
- **Visualizations**: Understand embeddings and anomalies
- **Production examples**: Real-world deployment patterns

---

## Citation

If you use this tutorial series in your work, please cite:

```
Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
Revisiting Deep Learning Models for Tabular Data.
Neural Information Processing Systems (NeurIPS).

He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep Residual Learning for Image Recognition.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
```

---

## Related Content

### Prerequisites
- [Neural Networks From Scratch](/ai-eng/nnfs/index.md) - Learn NN fundamentals

### Related Tutorials
- [Alternating Least Squares (ALS)](/ai-eng/ml-algorithms/als_tutorial.md) - Matrix factorization
- [Latent Factors](/ai-eng/ml-algorithms/latent_factors.md) - Understanding embeddings
- [Softmax](/ai-eng/ml-algorithms/softmax_from_scores.md) - From scores to probabilities

---

## Get Started

Ready to build your anomaly detection system? Start with [Part 1: Understanding ResNet Architecture](part1-understanding-resnet)!

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

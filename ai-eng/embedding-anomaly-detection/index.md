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

# Embedding-Based Anomaly Detection for Observability [DRAFT]

A comprehensive 8-part tutorial series on building production-ready anomaly detection systems using ResNet embeddings for OCSF (Open Cybersecurity Schema Framework) observability data.

**What you'll learn**: How to build, train, and deploy a **custom embedding model** (TabularResNet) specifically designed for OCSF observability data. This model transforms security logs and system metrics into vector representations. Anomaly detection happens entirely through vector database similarity search—no separate detection model needed. The system processes streaming OCSF events in near real-time to automatically identify unusual behavior.

---

## Series Overview

This tutorial series takes you from ResNet fundamentals to deploying and monitoring a complete anomaly detection system in production. You'll learn how to:

- Build and train a custom TabularResNet embedding model using self-supervised learning on unlabeled OCSF logs
- Deploy the custom embedding model as a FastAPI service for near real-time inference
- Store embeddings in a vector database for fast k-NN similarity search
- Detect anomalies purely through vector DB operations (k-NN distance scoring—no classical DL detection model)
- Monitor embedding quality and trigger automated retraining of the embedding model when drift is detected

**Target Audience**: ML engineers, security engineers, and data scientists working with observability data

**Applicability**: While this series uses **OCSF security logs** as the running example, the TabularResNet embedding approach applies to **any structured observability data**:
- **Telemetry/Metrics**: Time-series data (CPU%, memory, latency) with metadata (host, service, region) → convert to tabular rows
- **Configuration data**: Key-value pairs, settings, deployment configs → naturally tabular
- **Distributed traces**: Span attributes (service, duration, status_code, error) → tabular features per span
- **Application logs**: JSON logs, syslog, custom formats → any structured schema works

**The key requirement**: Your data can be represented as **rows with categorical and numerical features**. If you can create a pandas DataFrame from your data, you can use this approach.

**Prerequisites**:
- Basic Python and PyTorch
- Understanding of neural networks (or complete our [Neural Networks From Scratch](/ai-eng/nnfs/index.md) series first)

**Key Terms** (explained in detail throughout the series):
- **Embeddings**: Dense numerical vectors that capture the essence of complex data (like converting a security log into a list of numbers)
- **Self-supervised learning**: Training a model without labeled data by creating learning tasks from the data itself
- **Vector database**: A specialized database for storing and quickly searching through embeddings based on similarity
- **ResNet**: A deep learning architecture that uses "residual connections" to train very deep networks effectively

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

:::{grid-item-card} Part 8: Multi-Source Correlation
:link: part8-multi-source-correlation
:link-type: doc

Extend to multiple data sources for root cause analysis:
- Training separate models for logs, metrics, traces, config
- Unified vector database with metadata tags
- Temporal correlation across sources
- Causal graph construction
- Automated root cause ranking

**Advanced** · 50 min read
:::

:::{grid-item-card} Complete Series
:class-card: sd-border-primary

**Total**: ~5 hours of comprehensive, hands-on content

All code examples are executable and production-ready.
:::

::::

---

## What You'll Build

By the end of this series, you'll have:

1. **Custom TabularResNet Embedding Model**: Trained from scratch on your OCSF data using self-supervised learning
2. **Embedding Service**: FastAPI REST API that serves the custom TabularResNet model, generating embeddings for OCSF events via HTTP requests
3. **Vector Database**: Stores embeddings and performs k-NN similarity search at scale
4. **Vector-Based Anomaly Detection**: Detection through pure vector DB operations (k-NN distance, density)—no classical DL detection model
5. **Monitoring & Alerting**: Track embedding drift, detection quality, and system health
6. **Automated Retraining**: Triggers retraining of the custom embedding model based on drift and performance degradation

**Optional Extension (Part 8)**: For advanced production deployments, extend the system to correlate anomalies across multiple observability data sources (logs, metrics, traces, configuration changes) for automated root cause analysis.

### System Architecture

This diagram shows the complete end-to-end system you'll build. OCSF events stream in near real-time through the following pipeline:

1. **Preprocessing**: Extract and normalize features from each OCSF event
2. **Embedding generation**: TabularResNet (the only ML model) generates a vector for each event
3. **Vector DB storage**: Embeddings are indexed for fast k-NN similarity search
4. **Anomaly scoring**: Simple code logic computes scores using vector DB distances—NOT a separate ML model, just threshold-based calculations
5. **Alerting**: Trigger alerts for high-scoring anomalies

The monitoring components (shown in red/purple) continuously track embedding drift and system health, triggering automatic retraining of the embedding model when needed.

**Key architectural point**:
- **What we deploy**: A custom TabularResNet embedding model trained on your OCSF data
- **What we DON'T deploy**: A classical DL model for anomaly detection (no separate classifier, predictor, or scoring model)
- **How detection works**: Pure vector database operations (k-NN distance calculations, density estimation)

**Diagram legend**:
- **Solid arrows** (→): Near real-time data flow for each OCSF event
- **Dotted arrows** (⇢): Monitoring and feedback loops (periodic checks)
- **Colors**: Blue=Data input, Green=Embedding model (only ML model), Yellow=Vector storage, Orange=Scoring logic (not a model), Red/Purple=Monitoring

```{mermaid}
graph TB
    subgraph "Data Ingestion"
        OCSF[OCSF Data Stream]
    end

    subgraph "Feature Engineering"
        Preprocessor[Feature Extraction<br/>Categorical + Numerical]
    end

    subgraph "Vector Search"
        VectorDB[Vector DB<br/>Index + Similarity Search]
    end

    subgraph "Inference Pipeline"
        Embedding[TabularResNet<br/>Embedding Model]
        Detector[Anomaly Scoring Logic<br/>k-NN Distance/Thresholds]
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
    Embedding --> VectorDB
    VectorDB --> Detector
    Detector --> Alerts

    Embedding -.Monitor.-> Drift
    Detector -.Monitor.-> Metrics

    Drift -.Trigger.-> Retraining
    Metrics -.Trigger.-> Retraining

    Retraining -.Update.-> Registry
    Registry -.Deploy.-> Embedding

    style OCSF fill:#ADD8E6
    style Embedding fill:#90EE90
    style VectorDB fill:#FFD700
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

### Why a Vector Database?

A vector database makes similarity search the **central** mechanism for anomaly detection by:
- Storing and indexing embeddings for fast nearest-neighbor queries
- Enabling k-NN distance scoring, density estimation, and thresholding at scale
- Supporting incremental updates as new normal behavior arrives
- Providing consistent retrieval for both batch and near real-time pipelines

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

## Further Reading

For deeper understanding of embedding concepts and vector databases used in this series:

- **[Embeddings at Scale](https://snowch.github.io/embeddings-at-scale-book/)** - Comprehensive guide to building production embedding systems, covering vector databases, similarity search, and scaling considerations

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

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

# Observability & Anomaly Detection

*Build production-ready anomaly detection systems using deep learning for observability data*

---

This section covers advanced techniques for applying machine learning to observability data, with a focus on building production-grade anomaly detection systems using ResNet embeddings and self-supervised learning.

## What's Here

These tutorials guide you through building a complete anomaly detection system from scratch:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Topic
  - What You'll Learn
* - ResNet Architecture
  - *Adapting residual networks for tabular observability data* — Understanding how to build custom TabularResNet models for time-series and structured data
* - Self-Supervised Learning
  - *Training without labels using contrastive learning* — Learn techniques to train embeddings on unlabeled observability data
* - Production Deployment
  - *FastAPI services and vector databases* — Deploy real-time inference systems with monitoring and automated retraining
```

## Learning Approach

**Production-First:** Every example is designed to be deployable in production environments, not just toy demonstrations.

**Hands-On Code:** All tutorials are executable Jupyter notebooks with sample data—run them, modify them, build your own systems!

**Complete MLOps:** Learn the full lifecycle from data preparation through deployment, monitoring, and retraining.

## Full Tutorial Series

For the complete, comprehensive guide to building anomaly detection systems with ResNet embeddings:

**[Observability Anomaly Detection →](https://snowch.github.io/observability-anomaly-detection)**

The full tutorial series includes:
- **ResNet Architecture**: Understanding residual networks and adapting them for tabular data
- **Feature Engineering**: Transforming OCSF observability data into model-ready features
- **Self-Supervised Learning**: Training on unlabeled data using contrastive learning
- **Embedding Quality**: Evaluating embeddings with quantitative and qualitative methods
- **Anomaly Detection**: Applying distance-based, density-based, and ensemble methods
- **Production Deployment**: FastAPI services, vector databases, and real-time inference
- **MLOps**: Monitoring, drift detection, and automated retraining
- **Multi-Source Correlation**: Root cause analysis across logs, metrics, and traces

## What You'll Build

A complete production system including:
- Custom TabularResNet model trained with self-supervised learning
- Embedding service (FastAPI) for real-time inference
- Vector database for fast k-NN similarity search
- Anomaly detection through vector operations
- Monitoring and automated retraining pipeline

## Why Observability + ML Matters

Understanding how to apply ML to observability data is critical for:
- **Security Operations:** Detecting anomalous behavior and potential threats
- **System Reliability:** Identifying performance degradation before it impacts users
- **Root Cause Analysis:** Correlating events across distributed systems
- **Proactive Monitoring:** Moving from reactive alerts to predictive insights
- **Scale:** Handling massive observability data volumes efficiently

## Prerequisites

These tutorials assume:
- Basic Python and PyTorch knowledge
- Understanding of neural networks (or see our [Neural Networks From Scratch](https://snowch.github.io/ai-eng/nnfs/) series)
- Familiarity with REST APIs and deployment concepts helpful but not required

## Target Audience

- ML engineers building anomaly detection systems
- Security engineers working with observability data
- Data scientists interested in self-supervised learning
- Anyone applying deep learning to tabular/observability data

## Where to Start

**New to neural networks?** Start with our [Neural Networks From Scratch](https://snowch.github.io/ai-eng/nnfs/) series to build foundations.

**Ready to build?** Dive into the [full tutorial series](https://snowch.github.io/observability-anomaly-detection) and start building your anomaly detection system.

---

*These tutorials use production-grade tools and techniques used by leading tech companies to monitor and secure their infrastructure at scale.*

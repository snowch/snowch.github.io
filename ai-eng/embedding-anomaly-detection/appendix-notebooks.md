---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Appendix: Notebooks and Sample Data

**Problem**: You want to follow along with hands-on code without reading through all the tutorial text.

**Solution**: This appendix provides Jupyter notebooks and sample OCSF data that you can run immediately.

---

## Quick Start

1. Download the notebooks and sample data below
2. Install dependencies: `pip install pandas numpy torch scikit-learn matplotlib pyarrow`
3. Open the notebooks in Jupyter and run cells

---

## Download Sample Data

Pre-generated OCSF data (~15 minutes of synthetic observability events):

{download}`Download sample data (ocsf_data.zip) <./data.zip>`

**Contents:**
- `ocsf_logs.parquet` - ~5,700 application log events (59 columns)
- `ocsf_traces.parquet` - ~2,800 distributed trace spans (17 columns)
- `ocsf_metrics.parquet` - ~7,000 metric data points (33 columns)
- `ocsf_eval_subset.parquet` - 1,000 labeled events for evaluation (~2% anomaly rate)

**Anomaly types in the data:**
- Cache miss storms
- Database timeouts
- Memory leaks
- Slow queries

---

## Notebooks

View the executed notebooks with output, or download to run yourself:

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| [Feature Engineering](notebooks/03-feature-engineering) | Load OCSF data, extract temporal features, encode for ML | Sample data |
| [Self-Supervised Training](notebooks/04-self-supervised-training) | Train TabularResNet with contrastive learning | Part 3 output |
| [Anomaly Detection](notebooks/06-anomaly-detection) | Compare k-NN, LOF, Isolation Forest detection | Part 4 output |

{download}`Download all notebooks (notebooks.zip) <./notebooks.zip>`

---

## Notebook Workflow

```{mermaid}
graph LR
    subgraph data["Sample Data"]
        parquet["ocsf_logs.parquet"]
    end

    subgraph nb1["03-feature-engineering"]
        load["Load OCSF"]
        temporal["Extract temporal features"]
        encode["Encode for TabularResNet"]
    end

    subgraph nb2["04-self-supervised-training"]
        model["Build TabularResNet"]
        train["Contrastive learning"]
        embed["Extract embeddings"]
    end

    subgraph nb3["06-anomaly-detection"]
        knn["k-NN distance"]
        lof["LOF"]
        iso["Isolation Forest"]
        ensemble["Ensemble voting"]
    end

    parquet --> load
    load --> temporal --> encode
    encode --> model --> train --> embed
    embed --> knn & lof & iso --> ensemble

    style data fill:#e1ffe1
    style nb1 fill:#e1f5ff
    style nb2 fill:#fff4e1
    style nb3 fill:#ffe1e1
```

---

## What Each Notebook Covers

### 03-feature-engineering.ipynb

**Goal**: Transform raw OCSF data into feature arrays for TabularResNet.

**Key steps:**
1. Load OCSF parquet data
2. Explore schema (60 columns with nested objects flattened)
3. Extract temporal features (hour, day, cyclical sin/cos encoding)
4. Select categorical and numerical feature subsets
5. Handle missing values
6. Encode with LabelEncoder + StandardScaler

**Output:** `numerical_features.npy`, `categorical_features.npy`, `feature_artifacts.pkl`

---

### 04-self-supervised-training.ipynb

**Goal**: Train embeddings on unlabeled OCSF data using contrastive learning.

**Key steps:**
1. Load processed features from Part 3
2. Build TabularResNet model (categorical embeddings + residual blocks)
3. Implement SimCLR-style contrastive loss with data augmentation
4. Train for 20 epochs
5. Extract embeddings for all records
6. Visualize with t-SNE

**Output:** `embeddings.npy`, `tabular_resnet.pt`

---

### 06-anomaly-detection.ipynb

**Goal**: Detect anomalies using multiple algorithms and compare performance.

**Key steps:**
1. Load embeddings from Part 4
2. k-NN distance-based detection (average distance to neighbors)
3. Local Outlier Factor (density-based)
4. Isolation Forest (tree-based)
5. Ensemble voting (2/3 agreement)
6. Evaluate on labeled subset (if available)
7. Inspect top anomalies

**Output:** `anomaly_predictions.parquet`

---

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
torch>=2.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pyarrow>=10.0.0
```

Install with:
```bash
pip install pandas numpy torch scikit-learn matplotlib pyarrow
```

### Alternative: Run with Docker

Don't want to install Python locally? Use the official PyTorch Jupyter image:

```bash
# From the directory containing notebooks/ and data/
docker run -it -p 8889:8888 -v "${PWD}":/home/jovyan/work quay.io/jupyter/pytorch-notebook
```

Then open `http://localhost:8889` in your browser and navigate to `work/notebooks/`.

---

## Generating Your Own Data

Want more data or different anomaly scenarios? See [Appendix: Generating Training Data](appendix-generating-training-data) for a Docker Compose stack that generates realistic observability data.

---

## Summary

This appendix provides everything needed to run the tutorial hands-on:

1. **Sample data**: Pre-generated OCSF parquet files with ~15K events
2. **Notebooks**: Three Jupyter notebooks covering the core workflow
3. **No setup required**: Just download, install dependencies, and run

**Workflow**: Feature Engineering → Self-Supervised Training → Anomaly Detection

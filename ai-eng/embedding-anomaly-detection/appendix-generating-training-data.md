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

# Appendix: Generating Training Data for Observability

**Problem**: You want to follow this tutorial series but don't have OCSF observability data to train on.

**Solution**: This appendix shows how to generate realistic observability data (logs, metrics, traces, config changes) using Docker Compose to spin up actual infrastructure, then instrument it to produce training data you can release as open source.

---

## Overview: Synthetic Observability Stack

We'll create a realistic microservices environment that generates:
- **Logs**: Application errors, access logs, system events
- **Metrics**: CPU, memory, latency, throughput
- **Traces**: Distributed request spans
- **Config Changes**: Deployments, feature flags, scaling events

**Architecture**:

```{mermaid}
graph TB
    subgraph apps["Application Services"]
        webapi["web-api<br/>(Flask)"]
        auth["auth-service<br/>(Node.js)"]
        worker["payment-worker<br/>(Celery)"]
        postgres[("postgres-db")]
        redis[("redis-cache")]
    end

    subgraph obs["Observability Stack"]
        prom["Prometheus<br/>(metrics)"]
        otel["OpenTelemetry<br/>(traces)"]
        fluentd["Fluentd<br/>(logs)"]
    end

    subgraph gen["Load Generation"]
        loadgen["load-generator<br/>(traffic + anomalies)"]
    end

    webapi -->|queries| postgres
    webapi -->|cache| redis
    webapi -->|auth| auth
    worker -->|jobs| redis
    worker -->|writes| postgres

    webapi -.->|metrics| prom
    auth -.->|metrics| prom
    worker -.->|metrics| prom
    postgres -.->|metrics| prom
    redis -.->|metrics| prom

    webapi -.->|traces| otel
    auth -.->|traces| otel

    webapi -.->|logs| fluentd
    auth -.->|logs| fluentd
    worker -.->|logs| fluentd

    loadgen -->|HTTP requests| webapi

    style apps fill:#e1f5ff
    style obs fill:#fff4e1
    style gen fill:#ffe1f5
```

**Diagram explanation**:
- **Solid lines**: Data/service dependencies
- **Dotted lines**: Observability instrumentation
- **Application Services** (blue): Multi-service architecture generating realistic traffic
- **Observability Stack** (yellow): Collects logs, metrics, and traces
- **Load Generator** (pink): Creates normal traffic + anomaly scenarios

---

## Download Code Files

All code files from this appendix are available for download:

{download}`Download appendix-code.zip <./appendix-code.zip>`

The zip contains a complete, runnable stack:
- `docker-compose.yml` - Infrastructure configuration
- `services/web-api/` - Flask service with observability (app.py, Dockerfile, requirements.txt)
- `services/load-generator/` - Traffic generator (generate_load.py, Dockerfile, requirements.txt)
- `services/auth-service/` - Node.js auth service (app.js, Dockerfile, package.json)
- `services/payment-worker/` - Celery worker (worker.py, Dockerfile, requirements.txt)
- `config/` - Prometheus, OpenTelemetry, and Fluentd configurations
- `scripts/` - OCSF converter and optional labeling script

---

## Part 1: Docker Compose Infrastructure

### docker-compose.yml

Create a realistic multi-service application with built-in observability:

```{literalinclude} appendix-code/docker-compose.yml
:language: yaml
```

### Configuration Files

The observability stack requires these configuration files:

**config/prometheus.yml** - Metrics collection:

```{literalinclude} appendix-code/config/prometheus.yml
:language: yaml
```

**config/otel-collector-config.yml** - Distributed tracing:

```{literalinclude} appendix-code/config/otel-collector-config.yml
:language: yaml
```

**config/fluentd.conf** - Log aggregation:

```{literalinclude} appendix-code/config/fluentd.conf
:language: text
```

---

## Part 2: Instrumented Services

### Web API Service

**services/web-api/app.py** - Flask service with comprehensive observability:

```{literalinclude} appendix-code/services/web-api/app.py
:language: python
```

**services/web-api/Dockerfile**:

```{literalinclude} appendix-code/services/web-api/Dockerfile
:language: dockerfile
```

**services/web-api/requirements.txt**:

```{literalinclude} appendix-code/services/web-api/requirements.txt
:language: text
```

### Auth Service

**services/auth-service/app.js** - Node.js authentication service:

```{literalinclude} appendix-code/services/auth-service/app.js
:language: javascript
```

**services/auth-service/Dockerfile**:

```{literalinclude} appendix-code/services/auth-service/Dockerfile
:language: dockerfile
```

**services/auth-service/package.json**:

```{literalinclude} appendix-code/services/auth-service/package.json
:language: json
```

### Payment Worker

**services/payment-worker/worker.py** - Celery background worker:

```{literalinclude} appendix-code/services/payment-worker/worker.py
:language: python
```

**services/payment-worker/Dockerfile**:

```{literalinclude} appendix-code/services/payment-worker/Dockerfile
:language: dockerfile
```

**services/payment-worker/requirements.txt**:

```{literalinclude} appendix-code/services/payment-worker/requirements.txt
:language: text
```

---

## Part 3: Load Generator

**services/load-generator/generate_load.py** - Generates realistic traffic patterns with controlled anomalies:

```{literalinclude} appendix-code/services/load-generator/generate_load.py
:language: python
```

**services/load-generator/Dockerfile**:

```{literalinclude} appendix-code/services/load-generator/Dockerfile
:language: dockerfile
```

**services/load-generator/requirements.txt**:

```{literalinclude} appendix-code/services/load-generator/requirements.txt
:language: text
```

---

## Part 4: Collecting and Converting to OCSF

### scripts/convert_to_ocsf.py

Convert collected observability data to OCSF format:

```{literalinclude} appendix-code/scripts/convert_to_ocsf.py
:language: python
```

---

## Part 5: Running the Data Generation Pipeline

### Quick Start

```bash
# 1. Download and extract the code (or use the zip from this appendix)
cd appendix-code

# 2. Start all services (builds containers on first run)
docker compose up -d

# 3. Verify services are running
docker compose ps

# 4. Let the load generator run for a while (e.g., 5-10 minutes for demo, 2 hours for full dataset)
# The load-generator service automatically sends traffic to web-api

# 5. Export Docker logs and convert to OCSF format
docker compose logs --no-color > ./logs/docker.log
python scripts/convert_to_ocsf.py --log-file ./logs/docker.log

# Or pipe directly:
docker compose logs --no-color | python scripts/convert_to_ocsf.py --stdin

# 6. Output file (ready for Parts 2-3 of tutorial):
# - data/ocsf_logs.parquet (application logs in OCSF format)
```

### Prerequisites

Before running, ensure you have:
- Docker and Docker Compose installed
- Python 3.8+ with pandas and pyarrow: `pip install pandas pyarrow`

### Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clears all data)
docker compose down -v
```

### Generated Dataset Statistics

After 2 hours of generation, you'll have:
- **~72,000 normal events** (10 RPS * 7200 seconds)
- **~6-8 anomaly scenarios** (5% anomaly probability)
- **Event types**:
  - 60% user browsing (GET /api/users)
  - 25% search operations
  - 15% checkout operations
- **Anomaly types** (unlabeled - for natural occurrence):
  - Memory leak (gradual degradation over 30-60 min)
  - Connection pool exhaustion (sudden spike)
  - Cache invalidation storm (burst of DB hits)
  - Slow query cascade (thread pool exhaustion)

**Important**: The generated data is **unlabeled** - anomalies occur naturally in the traffic patterns without explicit labels. This matches the series approach:
- **Part 4**: Self-supervised learning (no labels needed)
- **Part 6**: Unsupervised anomaly detection (LOF, Isolation Forest, k-NN)

---

## Part 6: Using Generated Data in Tutorial (No Labels Needed)

Now you can use the generated data in Part 2 (TabularResNet) and Part 3 (Feature Engineering). **No labeling required** - the self-supervised and unsupervised methods work on raw observability data.

```python
# In Part 3: Feature Engineering
import pandas as pd

# Load your generated OCSF data (unlabeled)
ocsf_df = pd.read_parquet('data/ocsf_logs.parquet')

# Now follow Part 3 to extract features
categorical_features = ['service', 'http_method', 'status_id']
numerical_features = ['duration', 'time', 'severity_id']

# Continue with feature engineering from Part 3...
```

### Optional: Generating Labels for Method Evaluation

If you want to **evaluate** different anomaly detection methods (Part 6, Section 7: Method Comparison), you can optionally label a small subset for validation:

**scripts/label_subset_for_evaluation.py**

```{literalinclude} appendix-code/scripts/label_subset_for_evaluation.py
:language: python
```

**When to use labels**:
- ✅ **For evaluation only** (Part 6, Section 7: comparing LOF vs Isolation Forest vs k-NN)
- ❌ **Not for training** (Part 4 uses self-supervised learning on unlabeled data)

---

## Part 7: Releasing as Open Source

### README.md for the repository

```markdown
# OCSF Training Data Generator

Generate realistic observability data in OCSF format for **self-supervised** anomaly detection training.

## What This Generates

- 2 hours of realistic observability data
- ~72,000 normal events + 6-8 anomaly scenarios (unlabeled)
- Logs, metrics, and traces in OCSF format
- No labels required - use with self-supervised learning

## Quick Start

```bash
docker-compose up -d
docker-compose up load-generator
python scripts/convert_to_ocsf.py

# Optional: Generate small labeled subset for evaluation only
python scripts/label_subset_for_evaluation.py
```

## Output Datasets

- `data/ocsf_logs.parquet` - Application logs (unlabeled)
- `data/ocsf_metrics.parquet` - System metrics (unlabeled)
- `data/ocsf_traces.parquet` - Distributed traces (unlabeled)
- `data/ocsf_eval_subset.parquet` - Small labeled subset for evaluation (optional)

## Use Cases

- Self-supervised training with TabularResNet (Part 4)
- Unsupervised anomaly detection (Part 6: LOF, Isolation Forest, k-NN)
- Testing observability systems
- Learning OCSF schema

## Important: No Labels Needed

This generator creates **unlabeled** data that works with:
- **Self-supervised learning** (Part 4: contrastive learning, masked prediction)
- **Unsupervised anomaly detection** (Part 6: LOF, Isolation Forest, k-NN)

Labels are only needed for evaluation/comparison (Part 6, Section 7), not training.

## License

MIT - Free to use for commercial and non-commercial purposes

---

## Summary

This appendix provides a complete, open-source solution for generating realistic **unlabeled** observability data for self-supervised learning:

1. **Docker Compose stack**: Multi-service application with observability
2. **Instrumented services**: Generate logs, metrics, traces
3. **Load generator**: Creates normal traffic + anomaly scenarios (naturally occurring, unlabeled)
4. **OCSF converter**: Standardizes data to OCSF format
5. **Optional evaluation labels**: Small labeled subset for comparing detection methods (Part 6, Section 7 only)
6. **Ready for tutorial**: Outputs Parquet files for Parts 2-9

**Key difference from supervised learning**:
- ✅ **Generates unlabeled data** for self-supervised training (Part 4)
- ✅ **Works with unsupervised detection** (Part 6: LOF, Isolation Forest, k-NN)
- ⚠️ **Labels are optional** - only for evaluation, not training

**To use in the tutorial series**: Links from Part 3 (Feature Engineering) and Part 4 (Self-Supervised Training) point to this appendix for readers who need training data.

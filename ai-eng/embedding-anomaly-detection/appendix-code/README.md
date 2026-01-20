# OCSF Training Data Generator

Generate realistic observability data in OCSF format for **self-supervised** anomaly detection training.

## What This Generates

- Realistic observability data from a multi-service application
- Normal traffic + anomaly scenarios (unlabeled)
- **Logs**, **metrics**, and **traces** in OCSF format
- No labels required - use with self-supervised learning

## Quick Start

```bash
# 1. Create required directories (with write permissions for containers)
# Note: If you previously ran docker compose, the directory may be root-owned
sudo rm -rf ./logs/otel 2>/dev/null || true
mkdir -p ./logs/otel ./data
chmod 777 ./logs/otel

# 2. Start all services
docker compose up -d

# 3. Let the load generator run (5-10 min for demo, 2 hours for full dataset)
# Traffic is automatically generated

# 4. Export and convert all data types to OCSF format:

# Logs (from Docker):
docker compose logs --no-color > ./logs/docker.log
python scripts/convert_to_ocsf.py --log-file ./logs/docker.log

# Traces (from OpenTelemetry):
python scripts/convert_traces_to_ocsf.py --trace-file ./logs/otel/traces.jsonl

# Metrics (from Prometheus):
python scripts/export_prometheus_metrics.py --duration 10

# 5. (Optional) Generate small labeled subset for evaluation
python scripts/label_subset_for_evaluation.py
```

## Prerequisites

- Docker 20.10+ with Compose plugin (uses `docker compose`, not `docker-compose`)
- Python 3.8+ with: `pip install pandas pyarrow`

## Output Datasets

After running the export scripts:

- `data/ocsf_logs.parquet` - Application logs in OCSF format (unlabeled)
- `data/ocsf_traces.parquet` - Distributed traces in OCSF format (unlabeled)
- `data/ocsf_metrics.parquet` - System metrics in OCSF format (unlabeled)
- `data/ocsf_eval_subset.parquet` - Small labeled subset for evaluation (optional)

## Architecture

The stack includes:
- **web-api**: Flask service with observability instrumentation
- **auth-service**: Node.js authentication service
- **payment-worker**: Background job processor
- **load-generator**: Generates realistic traffic patterns with anomalies
- **postgres**: Database
- **redis**: Cache
- **prometheus**: Metrics collection
- **otel-collector**: Distributed tracing (exports to ./logs/otel/)
- **fluentd**: Log aggregation

## Export Scripts

| Script | Data Source | Output |
|--------|-------------|--------|
| `convert_to_ocsf.py` | Docker logs | `ocsf_logs.parquet` |
| `convert_traces_to_ocsf.py` | OTel collector | `ocsf_traces.parquet` |
| `export_prometheus_metrics.py` | Prometheus API | `ocsf_metrics.parquet` |

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

## Stopping Services

```bash
docker compose down      # Stop services
docker compose down -v   # Stop and remove volumes
```

## License

MIT - Free to use for commercial and non-commercial purposes

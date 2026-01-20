# OCSF Training Data Generator

Generate realistic observability data in OCSF format for **self-supervised** anomaly detection training.

## What This Generates

- Realistic observability data from a multi-service application
- Normal traffic + anomaly scenarios (unlabeled)
- Application logs in OCSF format
- No labels required - use with self-supervised learning

## Quick Start

```bash
# 1. Start all services
docker compose up -d

# 2. Let the load generator run (5-10 min for demo, 2 hours for full dataset)
# Traffic is automatically generated

# 3. Export logs and convert to OCSF format
docker compose logs --no-color > ./logs/docker.log
python scripts/convert_to_ocsf.py --log-file ./logs/docker.log

# 4. (Optional) Generate small labeled subset for evaluation
python scripts/label_subset_for_evaluation.py
```

## Prerequisites

- Docker and Docker Compose
- Python 3.8+ with: `pip install pandas pyarrow`

## Output Dataset

- `data/ocsf_logs.parquet` - Application logs in OCSF format (unlabeled)
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
- **otel-collector**: Distributed tracing
- **fluentd**: Log aggregation

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

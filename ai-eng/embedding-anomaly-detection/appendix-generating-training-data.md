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

## Part 1: Docker Compose Infrastructure

### docker-compose.yml

Create a realistic multi-service application with built-in observability:

```yaml
version: '3.8'

services:
  # Application Services
  web-api:
    build: ./services/web-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/appdb
      - REDIS_URL=redis://redis:6379
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
    depends_on:
      - postgres
      - redis
      - auth-service
    labels:
      - "monitoring=enabled"
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: web-api

  auth-service:
    build: ./services/auth-service
    ports:
      - "8001:8001"
    environment:
      - LDAP_URL=ldap://openldap:389
    labels:
      - "monitoring=enabled"
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: auth-service

  payment-worker:
    build: ./services/payment-worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/appdb
    depends_on:
      - redis
      - postgres
    labels:
      - "monitoring=enabled"
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: payment-worker

  # Data Stores
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=appdb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    labels:
      - "monitoring=enabled"

  redis:
    image: redis:7-alpine
    labels:
      - "monitoring=enabled"

  # Observability Stack
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  otel-collector:
    image: otel/opentelemetry-collector:latest
    volumes:
      - ./config/otel-collector-config.yml:/etc/otel/config.yml
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    command: ["--config=/etc/otel/config.yml"]

  fluentd:
    image: fluent/fluentd:latest
    volumes:
      - ./config/fluentd.conf:/fluentd/etc/fluent.conf
      - ./logs:/var/log/fluentd
    ports:
      - "24224:24224"
      - "24224:24224/udp"

  # Load Generator (creates traffic patterns)
  load-generator:
    build: ./services/load-generator
    depends_on:
      - web-api
    environment:
      - TARGET_URL=http://web-api:8000
      - NORMAL_RPS=10
      - ANOMALY_PROBABILITY=0.05

volumes:
  postgres_data:
  prometheus_data:
```

---

## Part 2: Instrumented Services

### services/web-api/app.py

Flask service with comprehensive observability:

```python
from flask import Flask, request, jsonify
import time
import random
import logging
from prometheus_client import Counter, Histogram, generate_latest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import psycopg2
import redis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","service":"web-api","level":"%(levelname)s","message":"%(message)s","trace_id":"%(trace_id)s"}'
)
logger = logging.getLogger(__name__)

# Set up tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

# Set up metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration', ['query_type'])

app = Flask(__name__)

# Database connection
db_conn = psycopg2.connect("postgresql://postgres:password@postgres:5432/appdb")
cache = redis.Redis(host='redis', port=6379, decode_responses=True)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Fetch user data - normal operation."""
    with tracer.start_as_current_span("get_user") as span:
        span.set_attribute("user.id", user_id)
        start_time = time.time()

        try:
            # Check cache first
            cached = cache.get(f"user:{user_id}")
            if cached:
                request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
                request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
                logger.info(f"Cache hit for user {user_id}", extra={'trace_id': span.get_span_context().trace_id})
                return jsonify({"user_id": user_id, "source": "cache"})

            # Query database
            db_start = time.time()
            cursor = db_conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            db_query_duration.labels(query_type='SELECT').observe(time.time() - db_start)

            if result:
                cache.setex(f"user:{user_id}", 300, str(result))
                request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
                request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
                logger.info(f"User {user_id} fetched from database", extra={'trace_id': span.get_span_context().trace_id})
                return jsonify({"user_id": user_id, "source": "database"})
            else:
                request_count.labels(method='GET', endpoint='/api/users', status=404).inc()
                logger.warning(f"User {user_id} not found", extra={'trace_id': span.get_span_context().trace_id})
                return jsonify({"error": "User not found"}), 404

        except Exception as e:
            request_count.labels(method='GET', endpoint='/api/users', status=500).inc()
            logger.error(f"Error fetching user {user_id}: {str(e)}", extra={'trace_id': span.get_span_context().trace_id})
            return jsonify({"error": "Internal server error"}), 500

@app.route('/api/checkout', methods=['POST'])
def checkout():
    """Checkout operation - can trigger anomalies."""
    with tracer.start_as_current_span("checkout") as span:
        start_time = time.time()

        # Simulate anomalies based on environment variable
        anomaly_prob = float(os.getenv('ANOMALY_PROBABILITY', 0.05))

        if random.random() < anomaly_prob:
            # ANOMALY: Simulate various failure modes
            anomaly_type = random.choice(['db_timeout', 'memory_leak', 'slow_query', 'cache_miss_storm'])

            if anomaly_type == 'db_timeout':
                time.sleep(5)  # Simulate slow query
                logger.error("Database timeout during checkout", extra={'trace_id': span.get_span_context().trace_id})
                request_count.labels(method='POST', endpoint='/api/checkout', status=504).inc()
                return jsonify({"error": "Database timeout"}), 504

            elif anomaly_type == 'memory_leak':
                # Simulate memory leak by holding large objects
                leak = ["x" * 1000000 for _ in range(100)]  # 100MB allocation
                logger.warning("High memory allocation during checkout", extra={'trace_id': span.get_span_context().trace_id})

            elif anomaly_type == 'slow_query':
                time.sleep(random.uniform(2, 5))
                logger.warning(f"Slow checkout processing: {time.time() - start_time:.2f}s", extra={'trace_id': span.get_span_context().trace_id})

            elif anomaly_type == 'cache_miss_storm':
                # Simulate cache invalidation causing DB overload
                for i in range(50):
                    cache.delete(f"user:{i}")
                logger.error("Cache miss storm detected", extra={'trace_id': span.get_span_context().trace_id})

        # Normal checkout flow
        request_count.labels(method='POST', endpoint='/api/checkout', status=200).inc()
        request_duration.labels(method='POST', endpoint='/api/checkout').observe(time.time() - start_time)
        logger.info("Checkout completed successfully", extra={'trace_id': span.get_span_context().trace_id})
        return jsonify({"status": "success"})

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

## Part 3: Load Generator

### services/load-generator/generate_load.py

Generates realistic traffic patterns with controlled anomalies:

```python
import requests
import time
import random
from datetime import datetime, timedelta
import json

class LoadGenerator:
    """
    Generate realistic observability data with normal and anomalous patterns.
    """
    def __init__(self, target_url, normal_rps=10, anomaly_probability=0.05):
        self.target_url = target_url
        self.normal_rps = normal_rps
        self.anomaly_probability = anomaly_probability
        self.session = requests.Session()

    def generate_normal_traffic(self):
        """Generate normal user traffic patterns."""
        patterns = [
            # Pattern 1: User browsing
            lambda: self.session.get(f"{self.target_url}/api/users/{random.randint(1, 1000)}"),

            # Pattern 2: Search
            lambda: self.session.get(f"{self.target_url}/api/search?q=product"),

            # Pattern 3: Checkout (normal)
            lambda: self.session.post(f"{self.target_url}/api/checkout", json={"cart_id": random.randint(1, 100)}),
        ]

        pattern = random.choice(patterns)
        try:
            response = pattern()
            print(f"[NORMAL] {response.status_code} - {response.url}")
        except Exception as e:
            print(f"[ERROR] {str(e)}")

    def generate_anomaly_scenario(self):
        """Generate specific anomaly scenarios."""
        scenarios = [
            self.scenario_deployment_memory_leak,
            self.scenario_database_connection_pool_exhaustion,
            self.scenario_cache_invalidation_storm,
            self.scenario_slow_query_cascade,
        ]

        scenario = random.choice(scenarios)
        print(f"\n[ANOMALY] Starting scenario: {scenario.__name__}")
        scenario()

    def scenario_deployment_memory_leak(self):
        """
        Simulate memory leak after deployment (gradual degradation).

        Sequence:
        1. Deployment completes (config change)
        2. Memory usage gradually increases
        3. GC pressure increases
        4. Query latency spikes
        5. Connection pool exhaustion
        """
        print("  → Simulating deployment with memory leak")

        # Generate increasing load over 5 minutes
        for i in range(60):
            try:
                # Each request allocates more memory in the service
                response = self.session.post(
                    f"{self.target_url}/api/checkout",
                    json={"trigger_memory_leak": True}
                )
                print(f"  → Minute {i//12}: Memory pressure increasing")
                time.sleep(5)
            except Exception as e:
                print(f"  → Service degraded: {str(e)}")
                break

    def scenario_database_connection_pool_exhaustion(self):
        """
        Simulate DB connection pool exhaustion.

        Sequence:
        1. Spike in concurrent requests
        2. Slow queries hold connections
        3. Pool exhausted
        4. New requests timeout
        """
        print("  → Simulating DB connection pool exhaustion")

        import concurrent.futures

        def slow_request():
            try:
                response = self.session.get(
                    f"{self.target_url}/api/users/{random.randint(1, 1000)}",
                    timeout=10
                )
                return response.status_code
            except:
                return 500

        # Flood with 100 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(slow_request) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        print(f"  → Results: {results.count(200)} success, {results.count(500)} failed")

    def scenario_cache_invalidation_storm(self):
        """
        Simulate cache invalidation causing DB overload.

        Sequence:
        1. Cache gets invalidated (deployment or manual flush)
        2. All requests hit database
        3. DB overloaded
        4. Latency spikes
        """
        print("  → Simulating cache invalidation storm")

        # Trigger cache invalidation
        self.session.post(f"{self.target_url}/admin/cache/flush")

        # Generate high read traffic (all cache misses)
        for i in range(100):
            try:
                response = self.session.get(
                    f"{self.target_url}/api/users/{random.randint(1, 10000)}"
                )
                if i % 10 == 0:
                    print(f"  → Cache miss {i}: {response.elapsed.total_seconds():.2f}s")
            except:
                pass
            time.sleep(0.1)

    def scenario_slow_query_cascade(self):
        """
        Simulate slow query causing cascading failure.

        Sequence:
        1. One slow query blocks resources
        2. Other queries queue up
        3. Thread pool exhaustion
        4. Service unresponsive
        """
        print("  → Simulating slow query cascade")

        # Trigger slow query
        self.session.post(
            f"{self.target_url}/api/analytics",
            json={"trigger_slow_query": True}
        )

        time.sleep(30)  # Let cascade develop

    def run(self, duration_minutes=60):
        """
        Run load generator for specified duration.

        Args:
            duration_minutes: How long to generate traffic
        """
        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        print(f"Starting load generator (RPS: {self.normal_rps}, anomaly prob: {self.anomaly_probability})")
        print(f"Will run until: {end_time}")

        request_count = 0
        anomaly_count = 0

        while datetime.now() < end_time:
            # Decide if this cycle is normal or anomaly
            if random.random() < self.anomaly_probability:
                self.generate_anomaly_scenario()
                anomaly_count += 1
            else:
                # Generate normal traffic at target RPS
                for _ in range(self.normal_rps):
                    self.generate_normal_traffic()
                    time.sleep(1.0 / self.normal_rps)
                    request_count += 1

            # Print progress every minute
            if request_count % (self.normal_rps * 60) == 0:
                print(f"\n[PROGRESS] Generated {request_count} requests, {anomaly_count} anomaly scenarios")

if __name__ == '__main__':
    import os

    target_url = os.getenv('TARGET_URL', 'http://web-api:8000')
    normal_rps = int(os.getenv('NORMAL_RPS', 10))
    anomaly_probability = float(os.getenv('ANOMALY_PROBABILITY', 0.05))

    generator = LoadGenerator(target_url, normal_rps, anomaly_probability)
    generator.run(duration_minutes=120)  # Run for 2 hours
```

---

## Part 4: Collecting and Converting to OCSF

### scripts/convert_to_ocsf.py

Convert collected observability data to OCSF format:

```python
import json
from datetime import datetime
import pandas as pd

class OCSFConverter:
    """
    Convert raw observability data to OCSF (Open Cybersecurity Schema Framework) format.

    OCSF provides standardized schemas for observability events.
    """

    def convert_logs_to_ocsf(self, fluentd_log_file):
        """
        Convert Fluentd logs to OCSF format.

        Args:
            fluentd_log_file: Path to Fluentd JSON log file

        Returns:
            List of OCSF-formatted events
        """
        ocsf_events = []

        with open(fluentd_log_file, 'r') as f:
            for line in f:
                log_entry = json.loads(line)

                # Map to OCSF Application Activity (class_uid: 6001)
                ocsf_event = {
                    "class_uid": 6001,  # Application Activity
                    "category_uid": 6,   # Application Activity category
                    "severity_id": self._map_severity(log_entry.get('level', 'INFO')),
                    "time": int(datetime.fromisoformat(log_entry['timestamp']).timestamp() * 1000),
                    "metadata": {
                        "version": "1.0.0",
                        "product": {
                            "name": log_entry.get('service', 'unknown'),
                            "vendor_name": "MyCompany"
                        }
                    },
                    "activity_id": 1,  # Log
                    "status_id": 1 if log_entry.get('status', 200) < 400 else 2,
                    "message": log_entry.get('message', ''),
                    "observables": [
                        {
                            "name": "trace_id",
                            "type": "Process ID",
                            "value": log_entry.get('trace_id', '')
                        }
                    ],
                    "http_request": {
                        "http_method": log_entry.get('method', ''),
                        "url": {
                            "path": log_entry.get('endpoint', '')
                        }
                    } if 'method' in log_entry else None,
                    "duration": log_entry.get('duration_ms', 0)
                }

                ocsf_events.append(ocsf_event)

        return ocsf_events

    def convert_metrics_to_ocsf(self, prometheus_metrics_file):
        """
        Convert Prometheus metrics to OCSF format.

        Args:
            prometheus_metrics_file: Path to Prometheus metrics export

        Returns:
            List of OCSF-formatted metric events
        """
        ocsf_events = []

        # Read Prometheus metrics (simplified - actual implementation would parse Prometheus format)
        df = pd.read_json(prometheus_metrics_file)

        for _, row in df.iterrows():
            # Map to OCSF System Activity (class_uid: 1001)
            ocsf_event = {
                "class_uid": 1001,  # System Activity
                "category_uid": 1,   # System Activity category
                "time": int(row['timestamp'] * 1000),
                "metadata": {
                    "version": "1.0.0"
                },
                "device": {
                    "hostname": row.get('instance', 'unknown'),
                    "type_id": 1  # Server
                },
                "metric": {
                    "name": row['metric_name'],
                    "value": row['value'],
                    "unit": row.get('unit', '')
                }
            }

            ocsf_events.append(ocsf_event)

        return ocsf_events

    def _map_severity(self, log_level):
        """Map log level to OCSF severity."""
        severity_map = {
            'DEBUG': 1,
            'INFO': 2,
            'WARNING': 3,
            'ERROR': 4,
            'CRITICAL': 5
        }
        return severity_map.get(log_level.upper(), 2)

    def save_to_parquet(self, ocsf_events, output_path):
        """
        Save OCSF events to Parquet for training.

        Args:
            ocsf_events: List of OCSF-formatted events
            output_path: Path to save Parquet file
        """
        df = pd.DataFrame(ocsf_events)
        df.to_parquet(output_path, compression='snappy')
        print(f"Saved {len(df)} OCSF events to {output_path}")

if __name__ == '__main__':
    converter = OCSFConverter()

    # Convert logs
    log_events = converter.convert_logs_to_ocsf('/var/log/fluentd/app.log')
    converter.save_to_parquet(log_events, '/data/ocsf_logs.parquet')

    # Convert metrics
    metric_events = converter.convert_metrics_to_ocsf('/data/prometheus_metrics.json')
    converter.save_to_parquet(metric_events, '/data/ocsf_metrics.parquet')

    print(f"Generated {len(log_events) + len(metric_events)} total OCSF events")
```

---

## Part 5: Running the Data Generation Pipeline

### Quick Start

```bash
# 1. Clone the repository (once you've created it)
git clone https://github.com/yourusername/ocsf-training-data-generator
cd ocsf-training-data-generator

# 2. Start the infrastructure
docker-compose up -d

# 3. Wait for services to be healthy
docker-compose ps

# 4. Start load generation (2 hours of data)
docker-compose up load-generator

# 5. Collect generated data
python scripts/convert_to_ocsf.py

# 6. Output files (ready for Parts 2-3 of tutorial):
# - data/ocsf_logs.parquet (application logs)
# - data/ocsf_metrics.parquet (system metrics)
# - data/ocsf_traces.parquet (distributed traces)
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

```python
import pandas as pd

def label_evaluation_subset(ocsf_events_path, sample_size=1000):
    """
    Label a small subset for evaluating anomaly detection methods.

    This is OPTIONAL - only needed if you want to compare method performance.
    The main training (Part 4) doesn't need labels.

    Args:
        ocsf_events_path: Path to OCSF Parquet file
        sample_size: How many events to label for evaluation

    Returns:
        Small labeled DataFrame for validation
    """
    df = pd.read_parquet(ocsf_events_path)

    # Sample a subset
    eval_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Simple heuristic labeling for evaluation
    eval_df['is_anomaly'] = 0

    # Label obvious anomalies (high latency, errors)
    eval_df.loc[eval_df['duration'] > 2000, 'is_anomaly'] = 1  # >2s latency
    eval_df.loc[eval_df['status_id'] == 2, 'is_anomaly'] = 1   # 5xx errors

    anomaly_count = eval_df['is_anomaly'].sum()
    print(f"Labeled {anomaly_count} / {len(eval_df)} events as anomalies for evaluation")
    print(f"Anomaly rate: {anomaly_count/len(eval_df)*100:.2f}%")

    # Save small evaluation set
    output_path = 'data/ocsf_eval_subset.parquet'
    eval_df.to_parquet(output_path)
    print(f"Saved evaluation subset to {output_path}")

    return eval_df

if __name__ == '__main__':
    # Generate small labeled subset for Part 6 evaluation (optional)
    label_evaluation_subset('data/ocsf_logs.parquet', sample_size=1000)
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
```

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

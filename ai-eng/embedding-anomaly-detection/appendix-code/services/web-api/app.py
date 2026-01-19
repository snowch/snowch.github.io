from flask import Flask, request, jsonify
import time
import random
import logging
import os
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

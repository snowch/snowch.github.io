from flask import Flask, request, jsonify, Response
import time
import random
import logging
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Set up logging - simple format that works for all loggers
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","service":"web-api","level":"%(levelname)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)

# Disable verbose werkzeug logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Set up tracing (optional - graceful if otel-collector unavailable)
tracer = None
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://otel-collector:4317')
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
    logger.info(f"OpenTelemetry tracing enabled, exporting to {otlp_endpoint}")
except Exception as e:
    logger.warning(f"OpenTelemetry tracing disabled: {e}")

# Set up metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration', ['query_type'])

# Database and cache connections (lazy initialization)
db_conn = None
cache = None

def get_db():
    global db_conn
    if db_conn is None:
        try:
            import psycopg2
            db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@postgres:5432/appdb')
            db_conn = psycopg2.connect(db_url)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable: {e}")
    return db_conn

def get_cache():
    global cache
    if cache is None:
        try:
            import redis as redis_lib
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
            cache = redis_lib.from_url(redis_url, decode_responses=True)
            cache.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
    return cache


@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Fetch user data - normal operation."""
    start_time = time.time()
    trace_id = None

    # Start span if tracing available
    span = None
    if tracer:
        span = tracer.start_span("get_user")
        span.set_attribute("user.id", user_id)
        trace_id = span.get_span_context().trace_id

    try:
        # Check cache first
        redis_cache = get_cache()
        if redis_cache:
            try:
                cached = redis_cache.get(f"user:{user_id}")
                if cached:
                    request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
                    request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
                    logger.info(f"Cache hit for user {user_id}, trace_id={trace_id}")
                    return jsonify({"user_id": user_id, "source": "cache"})
            except Exception as e:
                logger.warning(f"Cache error: {e}")

        # Query database
        conn = get_db()
        if conn:
            try:
                db_start = time.time()
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
                result = cursor.fetchone()
                db_query_duration.labels(query_type='SELECT').observe(time.time() - db_start)

                if result:
                    if redis_cache:
                        redis_cache.setex(f"user:{user_id}", 300, str(result))
                    request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
                    request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
                    logger.info(f"User {user_id} fetched from database, trace_id={trace_id}")
                    return jsonify({"user_id": user_id, "source": "database"})
            except Exception as e:
                logger.warning(f"Database error: {e}")

        # User not found or DB unavailable - return mock data for demo
        request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
        request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
        logger.info(f"Returning mock user {user_id}, trace_id={trace_id}")
        return jsonify({"user_id": user_id, "source": "mock", "name": f"User {user_id}"})

    except Exception as e:
        request_count.labels(method='GET', endpoint='/api/users', status=500).inc()
        logger.error(f"Error fetching user {user_id}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

    finally:
        if span:
            span.end()


@app.route('/api/checkout', methods=['POST'])
def checkout():
    """Checkout operation - can trigger anomalies."""
    start_time = time.time()
    trace_id = None

    span = None
    if tracer:
        span = tracer.start_span("checkout")
        trace_id = span.get_span_context().trace_id

    try:
        # Simulate anomalies based on environment variable
        anomaly_prob = float(os.getenv('ANOMALY_PROBABILITY', 0.05))

        if random.random() < anomaly_prob:
            # ANOMALY: Simulate various failure modes
            anomaly_type = random.choice(['db_timeout', 'memory_leak', 'slow_query', 'cache_miss_storm'])

            if anomaly_type == 'db_timeout':
                time.sleep(5)  # Simulate slow query
                logger.error(f"Database timeout during checkout, trace_id={trace_id}")
                request_count.labels(method='POST', endpoint='/api/checkout', status=504).inc()
                return jsonify({"error": "Database timeout"}), 504

            elif anomaly_type == 'memory_leak':
                # Simulate memory leak by holding large objects
                leak = ["x" * 1000000 for _ in range(100)]  # 100MB allocation
                logger.warning(f"High memory allocation during checkout, trace_id={trace_id}")

            elif anomaly_type == 'slow_query':
                time.sleep(random.uniform(2, 5))
                logger.warning(f"Slow checkout processing: {time.time() - start_time:.2f}s, trace_id={trace_id}")

            elif anomaly_type == 'cache_miss_storm':
                # Simulate cache invalidation causing DB overload
                redis_cache = get_cache()
                if redis_cache:
                    for i in range(50):
                        redis_cache.delete(f"user:{i}")
                logger.error(f"Cache miss storm detected, trace_id={trace_id}")

        # Normal checkout flow
        request_count.labels(method='POST', endpoint='/api/checkout', status=200).inc()
        request_duration.labels(method='POST', endpoint='/api/checkout').observe(time.time() - start_time)
        logger.info(f"Checkout completed successfully, trace_id={trace_id}")
        return jsonify({"status": "success"})

    finally:
        if span:
            span.end()


@app.route('/api/search', methods=['GET'])
def search():
    """Search endpoint for load generator."""
    start_time = time.time()
    query = request.args.get('q', '')

    # Simulate search delay
    time.sleep(random.uniform(0.01, 0.1))

    request_count.labels(method='GET', endpoint='/api/search', status=200).inc()
    request_duration.labels(method='GET', endpoint='/api/search').observe(time.time() - start_time)
    logger.info(f"Search completed for query: {query}")

    return jsonify({"results": [], "query": query})


@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    logger.info("Starting web-api service on port 8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)

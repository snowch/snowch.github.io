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

# Part 3: Feature Engineering for OCSF Data [DRAFT]

Learn how to transform raw OCSF security logs into the feature arrays that TabularResNet expects.

**The challenge**: You have raw OCSF JSON events with nested fields, optional values, and 300+ possible attributes. TabularResNet needs clean numerical and categorical arrays. This part bridges that gap.

**What you'll learn**:
1. Loading and parsing OCSF JSON structure
2. Extracting and flattening nested fields
3. Engineering temporal, aggregation, and derived features
4. Handling missing values and high cardinality
5. Building an end-to-end feature pipeline

---

## Understanding OCSF Structure

The [Open Cybersecurity Schema Framework (OCSF)](https://schema.ocsf.io/) provides a standardized schema for security events. Each event has:

- **class_name**: Event type (Authentication, Network Activity, File Activity, etc.)
- **Core fields**: severity_id, time, activity_id, status_id
- **Nested objects**: actor, src_endpoint, dst_endpoint, http_request, etc.
- **Optional fields**: Many fields are present only for specific event types

**Example OCSF Authentication Event**:

```json
{
  "class_name": "Authentication",
  "class_uid": 3002,
  "severity_id": 1,
  "activity_id": 1,
  "time": 1672531200000,
  "status_id": 1,
  "actor": {
    "user": {
      "name": "john.doe",
      "uid": "12345",
      "email": "john.doe@company.com"
    },
    "session": {
      "uid": "sess-98765",
      "created_time": 1672531195000
    }
  },
  "src_endpoint": {
    "ip": "192.168.1.100",
    "port": 54321,
    "location": {
      "city": "San Francisco",
      "country": "US"
    }
  },
  "http_request": {
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "url": {
      "hostname": "api.company.com",
      "path": "/auth/login"
    }
  },
  "auth_protocol": "OAuth 2.0",
  "logon_type": "Interactive"
}
```

**Key observations**:
- Deeply nested structure (actor.user.name is 3 levels deep)
- Mix of strings, numbers, timestamps
- Some fields always present, others optional
- High-cardinality fields (user.uid, session.uid)

---

## Loading OCSF Data

### From JSON Files

```{code-cell}
:tags: [skip-execution]

import json
import pandas as pd

def load_ocsf_from_file(filepath):
    """Load OCSF events from newline-delimited JSON file."""
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events

# Example usage
events = load_ocsf_from_file('ocsf_events.jsonl')
print(f"Loaded {len(events)} OCSF events")
```

### From Streaming Sources

```{code-cell}
:tags: [skip-execution]

# Kafka example (using confluent-kafka)
from confluent_kafka import Consumer

def consume_ocsf_from_kafka(topic, bootstrap_servers):
    """
    Consume OCSF events from Kafka topic.

    Args:
        topic: Kafka topic name
        bootstrap_servers: Kafka broker addresses

    Yields:
        Parsed OCSF event dictionaries
    """
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': 'ocsf-feature-engineering',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # Parse OCSF JSON
            event = json.loads(msg.value().decode('utf-8'))
            yield event

    finally:
        consumer.close()

# Usage
# for event in consume_ocsf_from_kafka('ocsf-events', 'localhost:9092'):
#     features = extract_features(event)
```

---

## Extracting Raw Features

### Flattening Nested Fields

```{code-cell}
import pandas as pd

def flatten_ocsf_event(event, prefix='', sep='_'):
    """
    Recursively flatten nested OCSF event structure.

    Args:
        event: OCSF event dictionary
        prefix: Current key prefix for recursion
        sep: Separator for nested keys

    Returns:
        Flat dictionary with dot-notation keys

    Example:
        {'actor': {'user': {'name': 'john'}}}
        → {'actor_user_name': 'john'}
    """
    flat = {}

    for key, value in event.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flat.update(flatten_ocsf_event(value, new_key, sep))
        elif isinstance(value, list):
            # For lists, take first element or length
            if len(value) > 0:
                if isinstance(value[0], dict):
                    flat.update(flatten_ocsf_event(value[0], new_key, sep))
                else:
                    flat[new_key] = value[0]
            flat[f"{new_key}_count"] = len(value)
        else:
            flat[new_key] = value

    return flat

# Example
event = {
    "class_name": "Authentication",
    "severity_id": 1,
    "actor": {
        "user": {"name": "john.doe", "uid": "12345"}
    },
    "src_endpoint": {
        "ip": "192.168.1.100"
    }
}

flat_event = flatten_ocsf_event(event)
print("Flattened event:")
for key, value in flat_event.items():
    print(f"  {key}: {value}")
```

### Selecting Core Features

Not all 300+ possible OCSF fields are useful. Start with a focused set:

```{code-cell}
def extract_core_features(event):
    """
    Extract core features from OCSF event.

    Returns dictionary with categorical and numerical features.
    """
    flat = flatten_ocsf_event(event)

    # Categorical features (strings/IDs that represent categories)
    categorical = {
        'class_name': flat.get('class_name'),
        'severity_id': flat.get('severity_id'),
        'activity_id': flat.get('activity_id'),
        'status_id': flat.get('status_id'),
        'user_name': flat.get('actor_user_name'),
        'src_ip': flat.get('src_endpoint_ip'),
        'dst_ip': flat.get('dst_endpoint_ip'),
        'auth_protocol': flat.get('auth_protocol'),
        'http_method': flat.get('http_request_http_method'),
    }

    # Numerical features (continuous values)
    numerical = {
        'time': flat.get('time', 0),
        'src_port': flat.get('src_endpoint_port', 0),
        'dst_port': flat.get('dst_endpoint_port', 0),
        'bytes_in': flat.get('traffic_bytes_in', 0),
        'bytes_out': flat.get('traffic_bytes_out', 0),
        'duration': flat.get('duration', 0),
    }

    return categorical, numerical

# Example
cat_features, num_features = extract_core_features(event)
print("\nCategorical features:", cat_features)
print("Numerical features:", num_features)
```

---

## Engineering Temporal Features

Timestamps are critical for security anomaly detection. Extract multiple temporal features:

```{code-cell}
from datetime import datetime
import numpy as np

def extract_temporal_features(timestamp_ms):
    """
    Extract temporal features from Unix timestamp (milliseconds).

    Args:
        timestamp_ms: Unix timestamp in milliseconds

    Returns:
        Dictionary of temporal features (both categorical and numerical)
    """
    if timestamp_ms is None or timestamp_ms == 0:
        return {
            'hour_of_day': 0,
            'day_of_week': 0,
            'is_weekend': 0,
            'is_business_hours': 0,
            'hour_sin': 0.0,
            'hour_cos': 1.0,
            'day_sin': 0.0,
            'day_cos': 1.0,
        }

    dt = datetime.fromtimestamp(timestamp_ms / 1000.0)

    # Basic temporal features
    hour = dt.hour
    day_of_week = dt.weekday()  # 0=Monday, 6=Sunday

    # Derived boolean features
    is_weekend = int(day_of_week >= 5)  # Saturday or Sunday
    is_business_hours = int(9 <= hour < 17)  # 9 AM to 5 PM

    # Cyclical encoding (hour and day wrap around)
    # Sin/cos encoding preserves circular nature (23:00 is close to 00:00)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    return {
        'hour_of_day': hour,  # Categorical: 0-23
        'day_of_week': day_of_week,  # Categorical: 0-6
        'is_weekend': is_weekend,  # Binary
        'is_business_hours': is_business_hours,  # Binary
        'hour_sin': hour_sin,  # Numerical cyclical
        'hour_cos': hour_cos,  # Numerical cyclical
        'day_sin': day_sin,  # Numerical cyclical
        'day_cos': day_cos,  # Numerical cyclical
    }

# Example
timestamp = 1672531200000  # 2023-01-01 00:00:00 UTC
temporal = extract_temporal_features(timestamp)
print("\nTemporal features:")
for key, value in temporal.items():
    print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
```

**Why cyclical encoding?**
- Problem: Hour 23 and hour 0 are only 1 hour apart, but numerically 23 units apart
- Solution: Encode as (sin, cos) pair on unit circle
- Result: Similar hours have similar encodings (23:00 and 00:00 are close in (sin, cos) space)

---

## Engineering Derived Features

Create higher-level features from raw values:

```{code-cell}
def extract_derived_features(event):
    """
    Create derived features from raw OCSF fields.

    Examples: rates, ratios, domain extraction, etc.
    """
    flat = flatten_ocsf_event(event)

    features = {}

    # Network transfer rates
    duration = flat.get('duration', 0)
    bytes_in = flat.get('traffic_bytes_in', 0)
    bytes_out = flat.get('traffic_bytes_out', 0)

    if duration > 0:
        features['bytes_per_second'] = (bytes_in + bytes_out) / (duration / 1000.0)
    else:
        features['bytes_per_second'] = 0.0

    # Total bytes
    features['total_bytes'] = bytes_in + bytes_out

    # Byte ratio (upload vs download)
    if bytes_in + bytes_out > 0:
        features['upload_ratio'] = bytes_out / (bytes_in + bytes_out)
    else:
        features['upload_ratio'] = 0.5

    # Domain extraction from URL
    hostname = flat.get('http_request_url_hostname', '')
    if hostname:
        # Extract top-level domain
        parts = hostname.split('.')
        features['tld'] = parts[-1] if len(parts) > 0 else 'unknown'
        features['domain_length'] = len(hostname)
    else:
        features['tld'] = 'unknown'
        features['domain_length'] = 0

    # User agent analysis
    user_agent = flat.get('http_request_user_agent', '')
    features['has_user_agent'] = int(len(user_agent) > 0)
    features['user_agent_length'] = len(user_agent)

    # Check for suspicious patterns
    features['is_default_port'] = int(flat.get('dst_endpoint_port') in [80, 443, 22, 21])

    return features

# Example
event_with_traffic = {
    "duration": 5000,  # 5 seconds
    "traffic": {"bytes_in": 1024000, "bytes_out": 512000},
    "http_request": {
        "url": {"hostname": "api.company.com"},
        "user_agent": "Mozilla/5.0..."
    },
    "dst_endpoint": {"port": 443}
}

derived = extract_derived_features(event_with_traffic)
print("\nDerived features:")
for key, value in derived.items():
    print(f"  {key}: {value}")
```

---

## Engineering Aggregation Features

For anomaly detection, behavior over time is crucial:

```{code-cell}
:tags: [skip-execution]

from collections import defaultdict, deque
from datetime import datetime, timedelta

class FeatureAggregator:
    """
    Maintain rolling window aggregations for OCSF events.

    Tracks per-user statistics over time windows.
    """

    def __init__(self, window_minutes=60):
        self.window_minutes = window_minutes
        self.window_seconds = window_minutes * 60

        # Store recent events per user
        self.user_events = defaultdict(lambda: deque())

    def add_event(self, event):
        """Add event and compute aggregations."""
        flat = flatten_ocsf_event(event)
        user = flat.get('actor_user_name', 'unknown')
        timestamp = flat.get('time', 0) / 1000.0  # Convert to seconds

        # Add to user's event history
        self.user_events[user].append({
            'timestamp': timestamp,
            'event': flat
        })

        # Remove events outside window
        cutoff = timestamp - self.window_seconds
        while (self.user_events[user] and
               self.user_events[user][0]['timestamp'] < cutoff):
            self.user_events[user].popleft()

        # Compute aggregations
        return self._compute_aggregations(user, flat)

    def _compute_aggregations(self, user, current_event):
        """Compute aggregations over user's recent events."""
        recent = self.user_events[user]

        if len(recent) == 0:
            return {
                'event_count_1h': 0,
                'failed_count_1h': 0,
                'unique_ips_1h': 0,
                'avg_bytes_1h': 0.0,
                'time_since_last_event': 0.0,
            }

        events = [e['event'] for e in recent]

        # Count events
        event_count = len(events)

        # Count failures
        failed_count = sum(1 for e in events if e.get('status_id') != 1)

        # Unique source IPs
        unique_ips = len(set(e.get('src_endpoint_ip') for e in events
                            if e.get('src_endpoint_ip')))

        # Average bytes
        total_bytes = sum(e.get('traffic_bytes_in', 0) + e.get('traffic_bytes_out', 0)
                         for e in events)
        avg_bytes = total_bytes / event_count if event_count > 0 else 0.0

        # Time since last event
        if len(recent) >= 2:
            time_since_last = (recent[-1]['timestamp'] -
                             recent[-2]['timestamp'])
        else:
            time_since_last = 0.0

        return {
            'event_count_1h': event_count,
            'failed_count_1h': failed_count,
            'unique_ips_1h': unique_ips,
            'avg_bytes_1h': avg_bytes,
            'time_since_last_event': time_since_last,
        }

# Usage example
aggregator = FeatureAggregator(window_minutes=60)

# Process events
for event in events:
    agg_features = aggregator.add_event(event)
    # Combine with other features
    all_features = {**extract_core_features(event), **agg_features}
```

**Aggregation features are powerful for anomaly detection**:
- Sudden spike in failed logins → Brute force attack
- Many unique IPs from one user → Compromised account
- Unusual bytes transferred → Data exfiltration

---

## Handling Missing Values

OCSF events have sparse, optional fields. Strategy:

```{code-cell}
def handle_missing_values(features_dict, categorical_cols, numerical_cols):
    """
    Handle missing values in feature dictionary.

    Args:
        features_dict: Dictionary of extracted features
        categorical_cols: List of categorical feature names
        numerical_cols: List of numerical feature names

    Returns:
        Dictionary with missing values handled
    """
    processed = {}

    # Categorical: use special "MISSING" category
    for col in categorical_cols:
        value = features_dict.get(col)
        if value is None or value == '' or pd.isna(value):
            processed[col] = 'MISSING'
        else:
            processed[col] = str(value)

    # Numerical: use 0 or add binary indicator
    for col in numerical_cols:
        value = features_dict.get(col)
        if value is None or pd.isna(value):
            processed[col] = 0.0
            processed[f'{col}_is_missing'] = 1  # Binary indicator
        else:
            processed[col] = float(value)
            processed[f'{col}_is_missing'] = 0

    return processed

# Example
incomplete_features = {
    'user_name': 'john.doe',
    'src_ip': None,  # Missing
    'bytes_in': 1024,
    'duration': None,  # Missing
}

cat_cols = ['user_name', 'src_ip']
num_cols = ['bytes_in', 'duration']

processed = handle_missing_values(incomplete_features, cat_cols, num_cols)
print("\nProcessed features with missing value handling:")
for key, value in processed.items():
    print(f"  {key}: {value}")
```

---

## Handling High Cardinality

IP addresses, user IDs, and session IDs can have millions of unique values.

### Hashing Trick

```{code-cell}
def hash_categorical_feature(value, num_buckets=1000):
    """
    Hash high-cardinality categorical to fixed number of buckets.

    Args:
        value: Original categorical value (string)
        num_buckets: Number of hash buckets

    Returns:
        Integer in range [0, num_buckets-1]
    """
    if value is None or value == '':
        return 0

    # Use Python's built-in hash
    return hash(str(value)) % num_buckets

# Example: Hash IP addresses to 1000 buckets
ip = "192.168.1.100"
hashed_ip = hash_categorical_feature(ip, num_buckets=1000)
print(f"\nOriginal IP: {ip}")
print(f"Hashed to bucket: {hashed_ip}")

# Different IPs map to different buckets (usually)
ips = ["192.168.1.100", "10.0.0.5", "172.16.0.1"]
for ip in ips:
    print(f"{ip} → bucket {hash_categorical_feature(ip, 1000)}")
```

### IP Subnet Encoding

```{code-cell}
def encode_ip_subnet(ip_address):
    """
    Encode IP as subnet (keep first 3 octets, hash last).

    Preserves network structure while reducing cardinality.
    """
    if not ip_address:
        return 'MISSING'

    parts = ip_address.split('.')
    if len(parts) != 4:
        return 'INVALID'

    # Keep first 3 octets (subnet), anonymize last octet
    subnet = '.'.join(parts[:3])
    return f"{subnet}.x"

# Example
ips = ["192.168.1.100", "192.168.1.101", "10.0.0.5"]
for ip in ips:
    print(f"{ip} → {encode_ip_subnet(ip)}")
```

---

## End-to-End Feature Pipeline

Putting it all together:

```{code-cell}
:tags: [skip-execution]

class OCSFFeatureExtractor:
    """
    End-to-end feature extraction pipeline for OCSF events.
    """

    def __init__(self, window_minutes=60):
        self.aggregator = FeatureAggregator(window_minutes)

        # Define feature schema
        self.categorical_features = [
            'class_name', 'severity_id', 'activity_id', 'status_id',
            'user_name', 'src_ip_subnet', 'auth_protocol',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours'
        ]

        self.numerical_features = [
            'src_port', 'dst_port', 'bytes_per_second', 'total_bytes',
            'upload_ratio', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'event_count_1h', 'failed_count_1h', 'unique_ips_1h',
            'avg_bytes_1h', 'time_since_last_event'
        ]

    def extract(self, event):
        """
        Extract all features from OCSF event.

        Returns:
            Dictionary with categorical and numerical features
        """
        # Core features
        cat_core, num_core = extract_core_features(event)

        # Temporal features
        timestamp = event.get('time', 0)
        temporal = extract_temporal_features(timestamp)

        # Derived features
        derived = extract_derived_features(event)

        # Aggregation features
        agg = self.aggregator.add_event(event)

        # Combine all features
        all_features = {**cat_core, **num_core, **temporal, **derived, **agg}

        # IP subnet encoding
        all_features['src_ip_subnet'] = encode_ip_subnet(
            all_features.get('src_ip'))

        # Handle missing values
        processed = handle_missing_values(
            all_features,
            self.categorical_features,
            self.numerical_features
        )

        return processed

    def batch_extract(self, events):
        """
        Extract features from multiple events.

        Returns:
            pandas DataFrame with one row per event
        """
        feature_dicts = [self.extract(event) for event in events]
        return pd.DataFrame(feature_dicts)

# Usage
extractor = OCSFFeatureExtractor(window_minutes=60)

# Process single event
features = extractor.extract(event)

# Process batch
# df = extractor.batch_extract(events)
# print(df.head())
```

---

## Preparing for TabularResNet

Final step: Convert to format expected by Part 2's TabularResNet:

```{code-cell}
:tags: [skip-execution]

from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def prepare_for_tabular_resnet(df, categorical_cols, numerical_cols):
    """
    Prepare OCSF features for TabularResNet.

    Args:
        df: DataFrame with extracted features
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names

    Returns:
        numerical_array: (N, num_numerical) normalized array
        categorical_array: (N, num_categorical) integer-encoded array
        encoders: Dictionary of LabelEncoders for each categorical
        scaler: StandardScaler for numerical features
        categorical_cardinalities: List of unique values per categorical
    """
    # Encode categorical features
    encoders = {}
    categorical_data = []

    for col in categorical_cols:
        encoder = LabelEncoder()
        # Add 'UNKNOWN' to handle new values at inference time
        unique_vals = list(df[col].unique()) + ['UNKNOWN']
        encoder.fit(unique_vals)
        encoded = encoder.transform(df[col])
        categorical_data.append(encoded)
        encoders[col] = encoder

    categorical_array = np.column_stack(categorical_data)
    categorical_cardinalities = [len(enc.classes_) for enc in encoders.values()]

    # Scale numerical features
    scaler = StandardScaler()
    numerical_array = scaler.fit_transform(df[numerical_cols])

    return (numerical_array, categorical_array,
            encoders, scaler, categorical_cardinalities)

# Usage
# numerical, categorical, encoders, scaler, cardinalities = \
#     prepare_for_tabular_resnet(df, categorical_cols, numerical_cols)

# Now ready for TabularResNet from Part 2!
# model = TabularResNet(
#     num_numerical_features=numerical.shape[1],
#     categorical_cardinalities=cardinalities,
#     d_model=256,
#     num_blocks=6
# )
```

---

## Complete Example: OCSF to Model Input

```{code-cell}
:tags: [skip-execution]

# Step 1: Load OCSF events
events = load_ocsf_from_file('ocsf_events.jsonl')
print(f"Loaded {len(events)} events")

# Step 2: Extract features
extractor = OCSFFeatureExtractor(window_minutes=60)
df = extractor.batch_extract(events)
print(f"Extracted features shape: {df.shape}")

# Step 3: Prepare for TabularResNet
numerical, categorical, encoders, scaler, cardinalities = prepare_for_tabular_resnet(
    df,
    extractor.categorical_features,
    extractor.numerical_features
)

print(f"\nReady for TabularResNet:")
print(f"  Numerical features: {numerical.shape}")
print(f"  Categorical features: {categorical.shape}")
print(f"  Categorical cardinalities: {cardinalities}")

# Step 4: Create TabularResNet model (from Part 2)
# model = TabularResNet(
#     num_numerical_features=numerical.shape[1],
#     categorical_cardinalities=cardinalities,
#     d_model=256,
#     num_blocks=6,
#     num_classes=None  # Embedding mode
# )

# Step 5: Train (see Part 4 - Self-Supervised Training)
```

---

## Summary

In this part, you learned:

1. **OCSF structure**: Nested JSON with 300+ optional fields
2. **Feature extraction**: Flattening, core features, temporal patterns
3. **Feature engineering**: Derived features, aggregations, cyclical encoding
4. **Data challenges**: Missing values, high cardinality, sparse fields
5. **End-to-end pipeline**: OCSF JSON → DataFrame → TabularResNet arrays

**Key takeaways**:
- Not all OCSF fields are useful - start with 20-50 core features
- Temporal features (hour, day) and aggregations (event counts) are critical for anomaly detection
- Cyclical encoding (sin/cos) preserves circular nature of time
- Hashing trick handles high-cardinality fields (IPs, user IDs)
- Missing values need explicit handling (special category or indicator)

**Next**: In [Part 4](part4-self-supervised-training), you'll use these engineered features to train TabularResNet with self-supervised learning on unlabelled OCSF data.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

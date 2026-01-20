#!/usr/bin/env python3
"""
Convert OpenTelemetry traces to OCSF (Open Cybersecurity Schema Framework) format.

Usage:
    # After running docker compose for a while:
    python scripts/convert_traces_to_ocsf.py --trace-file ./logs/otel/traces.jsonl

    # Or pipe directly:
    cat ./logs/otel/traces.jsonl | python scripts/convert_traces_to_ocsf.py --stdin
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Try to import pandas, provide helpful error if missing
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


class OCSFTraceConverter:
    """
    Convert OpenTelemetry traces to OCSF format.
    Maps to OCSF Network Activity class (class_uid: 4001) for HTTP spans.
    """

    def convert_traces_to_ocsf(self, trace_lines):
        """
        Convert OTLP JSON traces to OCSF format.

        Args:
            trace_lines: Iterable of JSON lines from otel-collector file exporter

        Returns:
            List of OCSF-formatted events
        """
        ocsf_events = []

        for line in trace_lines:
            line = line.strip()
            if not line:
                continue

            try:
                trace_data = json.loads(line)
                events = self._parse_trace_data(trace_data)
                ocsf_events.extend(events)
            except json.JSONDecodeError:
                continue

        return ocsf_events

    def _parse_trace_data(self, trace_data):
        """Parse OTLP trace data and convert spans to OCSF events."""
        events = []

        # OTLP format has resourceSpans -> scopeSpans -> spans
        resource_spans = trace_data.get('resourceSpans', [])

        for rs in resource_spans:
            # Extract resource attributes (service info)
            resource = rs.get('resource', {})
            resource_attrs = self._extract_attributes(resource.get('attributes', []))
            service_name = resource_attrs.get('service.name', 'unknown')

            scope_spans = rs.get('scopeSpans', [])
            for ss in scope_spans:
                spans = ss.get('spans', [])
                for span in spans:
                    ocsf_event = self._span_to_ocsf(span, service_name, resource_attrs)
                    if ocsf_event:
                        events.append(ocsf_event)

        return events

    def _span_to_ocsf(self, span, service_name, resource_attrs):
        """Convert a single span to OCSF format."""
        try:
            # Extract span attributes
            span_attrs = self._extract_attributes(span.get('attributes', []))

            # Parse timestamps (nanoseconds to milliseconds)
            start_time_ns = int(span.get('startTimeUnixNano', 0))
            end_time_ns = int(span.get('endTimeUnixNano', 0))
            start_time_ms = start_time_ns // 1_000_000
            duration_ms = (end_time_ns - start_time_ns) / 1_000_000

            # Determine span kind
            span_kind = span.get('kind', 0)
            kind_map = {0: 'UNSPECIFIED', 1: 'INTERNAL', 2: 'SERVER', 3: 'CLIENT', 4: 'PRODUCER', 5: 'CONSUMER'}

            # Extract HTTP attributes if present
            http_method = span_attrs.get('http.method', span_attrs.get('http.request.method', ''))
            http_status = span_attrs.get('http.status_code', span_attrs.get('http.response.status_code', 0))
            http_url = span_attrs.get('http.url', span_attrs.get('url.full', ''))
            http_route = span_attrs.get('http.route', span_attrs.get('http.target', ''))

            # Map status code to OCSF status_id
            status_code = span.get('status', {}).get('code', 0)
            if status_code == 2:  # ERROR
                status_id = 2  # Failure
            elif status_code == 1:  # OK
                status_id = 1  # Success
            else:
                status_id = 0  # Unknown

            # Map to OCSF API Activity (class_uid: 6003) for HTTP spans
            ocsf_event = {
                "class_uid": 6003,  # API Activity
                "category_uid": 6,  # Application Activity
                "activity_id": 1,   # Traffic
                "severity_id": 2 if status_id == 2 else 1,
                "time": start_time_ms,
                "duration": duration_ms,
                "status_id": status_id,
                "message": span.get('name', ''),
                "metadata": {
                    "version": "1.0.0",
                    "product": {
                        "name": service_name,
                        "vendor_name": "Demo"
                    }
                },
                # Trace context
                "trace_id": span.get('traceId', ''),
                "span_id": span.get('spanId', ''),
                "parent_span_id": span.get('parentSpanId', ''),
                "span_kind": kind_map.get(span_kind, 'UNKNOWN'),
                # Service info
                "service": service_name,
                # HTTP info (if present)
                "http_method": http_method,
                "http_status": http_status,
                "http_url": http_url,
                "http_route": http_route,
            }

            # Add user attribute if present
            if 'user.id' in span_attrs:
                ocsf_event['user_id'] = span_attrs['user.id']

            return ocsf_event

        except Exception as e:
            return None

    def _extract_attributes(self, attributes):
        """Extract key-value pairs from OTLP attributes format."""
        result = {}
        for attr in attributes:
            key = attr.get('key', '')
            value = attr.get('value', {})
            # OTLP values can be stringValue, intValue, boolValue, etc.
            if 'stringValue' in value:
                result[key] = value['stringValue']
            elif 'intValue' in value:
                result[key] = int(value['intValue'])
            elif 'boolValue' in value:
                result[key] = value['boolValue']
            elif 'doubleValue' in value:
                result[key] = value['doubleValue']
        return result

    def save_to_parquet(self, ocsf_events, output_path):
        """Save OCSF events to Parquet for training."""
        if not ocsf_events:
            print("Warning: No events to save")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(ocsf_events)
        df.to_parquet(output_path, compression='snappy')
        print(f"Saved {len(df)} OCSF trace events to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenTelemetry traces to OCSF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert traces from file:
  python scripts/convert_traces_to_ocsf.py --trace-file ./logs/otel/traces.jsonl

  # Or pipe directly:
  cat ./logs/otel/traces.jsonl | python scripts/convert_traces_to_ocsf.py --stdin
        """
    )
    parser.add_argument('--trace-file',
                        help='Path to OTLP traces file (JSONL format)')
    parser.add_argument('--stdin', action='store_true',
                        help='Read from stdin')
    parser.add_argument('--output-dir', default='./data',
                        help='Output directory for parquet files (default: ./data)')

    args = parser.parse_args()

    if not args.trace_file and not args.stdin:
        print("Error: Must specify --trace-file or --stdin")
        print()
        print("After running docker compose, traces are exported to:")
        print("  ./logs/otel/traces.jsonl")
        print()
        print("Convert with:")
        print("  python scripts/convert_traces_to_ocsf.py --trace-file ./logs/otel/traces.jsonl")
        sys.exit(1)

    # Check if trace file exists
    if args.trace_file and not os.path.exists(args.trace_file):
        print(f"Warning: Trace file not found: {args.trace_file}")
        print()
        print("This can happen if:")
        print("  1. The logs/otel directory doesn't exist or lacks write permissions")
        print("  2. The otel-collector hasn't received any traces yet")
        print("  3. The web-api OpenTelemetry instrumentation failed to initialize")
        print()
        print("To fix:")
        print("  mkdir -p ./logs/otel && chmod 777 ./logs/otel")
        print("  docker compose restart otel-collector")
        print("  # Wait a few minutes for traces to be generated")
        print("  docker compose logs otel-collector | tail -20")
        sys.exit(1)

    converter = OCSFTraceConverter()

    # Read trace lines
    if args.stdin:
        print("Reading from stdin...")
        trace_lines = sys.stdin.readlines()
    else:
        print(f"Reading from {args.trace_file}...")
        with open(args.trace_file, 'r') as f:
            trace_lines = f.readlines()

    print(f"Processing {len(trace_lines)} lines...")

    # Convert traces
    trace_events = converter.convert_traces_to_ocsf(trace_lines)

    if not trace_events:
        print("No valid trace events found.")
        print("Make sure docker compose services are running and generating traces.")
        print("Check if ./logs/otel/traces.jsonl exists and has content.")
        sys.exit(1)

    # Save to parquet
    output_path = os.path.join(args.output_dir, 'ocsf_traces.parquet')
    converter.save_to_parquet(trace_events, output_path)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total spans: {len(trace_events)}")
    df = pd.DataFrame(trace_events)
    print(f"  Services: {df['service'].unique().tolist()}")
    if 'span_kind' in df.columns:
        print(f"  Span kinds: {df['span_kind'].value_counts().to_dict()}")
    if 'http_method' in df.columns:
        methods = df[df['http_method'] != '']['http_method'].value_counts()
        if not methods.empty:
            print(f"  HTTP methods: {methods.to_dict()}")


if __name__ == '__main__':
    main()

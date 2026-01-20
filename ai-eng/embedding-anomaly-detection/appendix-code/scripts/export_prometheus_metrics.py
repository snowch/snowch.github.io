#!/usr/bin/env python3
"""
Export Prometheus metrics to OCSF (Open Cybersecurity Schema Framework) format.

Usage:
    # With docker compose running:
    python scripts/export_prometheus_metrics.py

    # Custom Prometheus URL:
    python scripts/export_prometheus_metrics.py --prometheus-url http://localhost:9090

    # Export specific time range (last N minutes):
    python scripts/export_prometheus_metrics.py --duration 60
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Try to import pandas, provide helpful error if missing
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


class PrometheusMetricsExporter:
    """
    Export Prometheus metrics to OCSF format.
    Maps to OCSF Metric class (class_uid: 4001).
    """

    def __init__(self, prometheus_url='http://localhost:9090'):
        self.prometheus_url = prometheus_url.rstrip('/')

    def _query_prometheus(self, query, start_time=None, end_time=None, step='15s'):
        """Query Prometheus API for metrics."""
        if start_time and end_time:
            # Range query
            url = f"{self.prometheus_url}/api/v1/query_range"
            params = f"query={query}&start={start_time}&end={end_time}&step={step}"
        else:
            # Instant query
            url = f"{self.prometheus_url}/api/v1/query"
            params = f"query={query}"

        full_url = f"{url}?{params}"

        try:
            req = Request(full_url)
            with urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                if data.get('status') == 'success':
                    return data.get('data', {}).get('result', [])
                else:
                    print(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
                    return []
        except (URLError, HTTPError) as e:
            print(f"Error querying Prometheus: {e}")
            return []

    def _get_metric_names(self):
        """Get list of available metric names from Prometheus."""
        url = f"{self.prometheus_url}/api/v1/label/__name__/values"
        try:
            req = Request(url)
            with urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                if data.get('status') == 'success':
                    return data.get('data', [])
        except (URLError, HTTPError) as e:
            print(f"Error getting metric names: {e}")
        return []

    def export_metrics(self, duration_minutes=10, step='15s'):
        """
        Export all metrics from the last N minutes.

        Args:
            duration_minutes: How many minutes of data to export
            step: Query step interval

        Returns:
            List of OCSF-formatted metric events
        """
        ocsf_events = []

        # Time range
        end_time = time.time()
        start_time = end_time - (duration_minutes * 60)

        # Get available metrics
        metric_names = self._get_metric_names()
        if not metric_names:
            print("No metrics found in Prometheus")
            return []

        # Filter to relevant metrics (skip internal Prometheus metrics)
        relevant_metrics = [
            m for m in metric_names
            if not m.startswith('prometheus_')
            and not m.startswith('promhttp_')
            and not m.startswith('go_')
            and not m.startswith('scrape_')
        ]

        print(f"Found {len(relevant_metrics)} relevant metrics")

        for metric_name in relevant_metrics:
            results = self._query_prometheus(
                metric_name,
                start_time=start_time,
                end_time=end_time,
                step=step
            )

            for result in results:
                events = self._result_to_ocsf(metric_name, result)
                ocsf_events.extend(events)

        return ocsf_events

    def _result_to_ocsf(self, metric_name, result):
        """Convert a Prometheus result to OCSF metric events."""
        events = []

        # Extract labels
        labels = result.get('metric', {})
        service = labels.get('job', labels.get('service', 'unknown'))

        # Remove __name__ from labels for cleaner output
        labels_clean = {k: v for k, v in labels.items() if k != '__name__'}

        # Get values (either 'value' for instant or 'values' for range)
        if 'values' in result:
            # Range query result
            for timestamp, value in result['values']:
                event = self._create_ocsf_metric(
                    metric_name, service, labels_clean,
                    timestamp, value
                )
                if event:
                    events.append(event)
        elif 'value' in result:
            # Instant query result
            timestamp, value = result['value']
            event = self._create_ocsf_metric(
                metric_name, service, labels_clean,
                timestamp, value
            )
            if event:
                events.append(event)

        return events

    def _create_ocsf_metric(self, metric_name, service, labels, timestamp, value):
        """Create a single OCSF metric event."""
        try:
            # Convert timestamp to milliseconds
            time_ms = int(float(timestamp) * 1000)

            # Parse value
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                return None

            # Determine metric type from name
            metric_type = self._infer_metric_type(metric_name)

            # Map to OCSF Metric class (class_uid: 4001)
            ocsf_event = {
                "class_uid": 4001,      # System Activity - Metric
                "category_uid": 4,       # System Activity
                "activity_id": 1,        # Log
                "severity_id": 1,        # Informational
                "time": time_ms,
                "message": f"{metric_name}: {numeric_value}",
                "metadata": {
                    "version": "1.0.0",
                    "product": {
                        "name": service,
                        "vendor_name": "Demo"
                    }
                },
                # Metric-specific fields
                "metric_name": metric_name,
                "metric_value": numeric_value,
                "metric_type": metric_type,
                "service": service,
                "labels": json.dumps(labels) if labels else "{}",
            }

            # Extract common label dimensions
            if 'endpoint' in labels:
                ocsf_event['endpoint'] = labels['endpoint']
            if 'method' in labels:
                ocsf_event['http_method'] = labels['method']
            if 'status' in labels:
                ocsf_event['http_status'] = labels['status']
            if 'instance' in labels:
                ocsf_event['instance'] = labels['instance']

            return ocsf_event

        except Exception as e:
            return None

    def _infer_metric_type(self, metric_name):
        """Infer Prometheus metric type from naming conventions."""
        if metric_name.endswith('_total') or metric_name.endswith('_count'):
            return 'counter'
        elif metric_name.endswith('_bucket'):
            return 'histogram'
        elif metric_name.endswith('_sum'):
            return 'sum'
        elif '_gauge' in metric_name or metric_name.endswith('_current'):
            return 'gauge'
        else:
            return 'gauge'  # Default assumption

    def save_to_parquet(self, ocsf_events, output_path):
        """Save OCSF events to Parquet for training."""
        if not ocsf_events:
            print("Warning: No events to save")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(ocsf_events)
        df.to_parquet(output_path, compression='snappy')
        print(f"Saved {len(df)} OCSF metric events to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export Prometheus metrics to OCSF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export metrics from running Prometheus (default: last 10 minutes):
  python scripts/export_prometheus_metrics.py

  # Export last hour of metrics:
  python scripts/export_prometheus_metrics.py --duration 60

  # Custom Prometheus URL:
  python scripts/export_prometheus_metrics.py --prometheus-url http://prometheus:9090
        """
    )
    parser.add_argument('--prometheus-url', default='http://localhost:9090',
                        help='Prometheus server URL (default: http://localhost:9090)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration in minutes to export (default: 10)')
    parser.add_argument('--step', default='15s',
                        help='Query step interval (default: 15s)')
    parser.add_argument('--output-dir', default='./data',
                        help='Output directory for parquet files (default: ./data)')

    args = parser.parse_args()

    print(f"Connecting to Prometheus at {args.prometheus_url}...")

    exporter = PrometheusMetricsExporter(args.prometheus_url)

    # Export metrics
    print(f"Exporting last {args.duration} minutes of metrics...")
    metric_events = exporter.export_metrics(
        duration_minutes=args.duration,
        step=args.step
    )

    if not metric_events:
        print("No metrics found.")
        print("Make sure docker compose services are running and Prometheus is scraping.")
        print("Check Prometheus targets at: http://localhost:9090/targets")
        sys.exit(1)

    # Save to parquet
    output_path = os.path.join(args.output_dir, 'ocsf_metrics.parquet')
    exporter.save_to_parquet(metric_events, output_path)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total metric events: {len(metric_events)}")
    df = pd.DataFrame(metric_events)
    print(f"  Services: {df['service'].unique().tolist()}")
    print(f"  Unique metrics: {df['metric_name'].nunique()}")
    print(f"  Metric types: {df['metric_type'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()

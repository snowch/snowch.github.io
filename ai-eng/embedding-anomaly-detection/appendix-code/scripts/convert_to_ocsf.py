#!/usr/bin/env python3
"""
Convert raw observability data to OCSF (Open Cybersecurity Schema Framework) format.

Usage:
    # After running docker compose for a while:
    python scripts/convert_to_ocsf.py

    # Or generate sample data for testing:
    python scripts/convert_to_ocsf.py --generate-sample
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import random

# Try to import pandas, provide helpful error if missing
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


class OCSFConverter:
    """
    Convert raw observability data to OCSF (Open Cybersecurity Schema Framework) format.

    OCSF provides standardized schemas for observability events.
    """

    def convert_logs_to_ocsf(self, log_source):
        """
        Convert logs to OCSF format.

        Args:
            log_source: Path to log file, directory of log files, or list of log entries

        Returns:
            List of OCSF-formatted events
        """
        ocsf_events = []
        log_entries = []

        # Handle different input types
        if isinstance(log_source, list):
            log_entries = log_source
        elif os.path.isfile(log_source):
            log_entries = self._read_log_file(log_source)
        elif os.path.isdir(log_source):
            for f in Path(log_source).glob('*.log'):
                log_entries.extend(self._read_log_file(f))
            for f in Path(log_source).glob('*.json'):
                log_entries.extend(self._read_log_file(f))

        for log_entry in log_entries:
            if isinstance(log_entry, str):
                try:
                    log_entry = json.loads(log_entry)
                except json.JSONDecodeError:
                    continue

            ocsf_event = self._log_to_ocsf(log_entry)
            if ocsf_event:
                ocsf_events.append(ocsf_event)

        return ocsf_events

    def _read_log_file(self, filepath):
        """Read log entries from a file."""
        entries = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(line)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
        return entries

    def _log_to_ocsf(self, log_entry):
        """Convert a single log entry to OCSF format."""
        try:
            # Parse timestamp
            timestamp_str = log_entry.get('timestamp', '')
            try:
                # Handle format: "2026-01-20 04:53:37,757"
                if ',' in timestamp_str:
                    timestamp_str = timestamp_str.replace(',', '.')
                dt = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                time_ms = int(dt.timestamp() * 1000)
            except:
                time_ms = int(datetime.now().timestamp() * 1000)

            # Map to OCSF Application Activity (class_uid: 6001)
            ocsf_event = {
                "class_uid": 6001,  # Application Activity
                "category_uid": 6,   # Application Activity category
                "severity_id": self._map_severity(log_entry.get('level', 'INFO')),
                "time": time_ms,
                "metadata": {
                    "version": "1.0.0",
                    "product": {
                        "name": log_entry.get('service', 'unknown'),
                        "vendor_name": "Demo"
                    }
                },
                "activity_id": 1,  # Log
                "message": log_entry.get('message', ''),
                "service": log_entry.get('service', 'unknown'),
            }

            # Extract trace_id if present in message
            message = log_entry.get('message', '')
            if 'trace_id=' in message:
                try:
                    trace_id = message.split('trace_id=')[1].split()[0].strip(',')
                    ocsf_event['trace_id'] = trace_id
                except:
                    pass

            # Determine status from message content
            if 'error' in message.lower() or 'failed' in message.lower():
                ocsf_event['status_id'] = 2  # Failure
            else:
                ocsf_event['status_id'] = 1  # Success

            # Extract duration if present
            if 'processed successfully in' in message:
                try:
                    duration_str = message.split('in ')[1].split('ms')[0]
                    ocsf_event['duration'] = float(duration_str)
                except:
                    pass

            return ocsf_event

        except Exception as e:
            print(f"Warning: Could not convert log entry: {e}")
            return None

    def _map_severity(self, log_level):
        """Map log level to OCSF severity."""
        severity_map = {
            'DEBUG': 1,
            'INFO': 2,
            'WARNING': 3,
            'WARN': 3,
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
        if not ocsf_events:
            print("Warning: No events to save")
            return

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(ocsf_events)
        df.to_parquet(output_path, compression='snappy')
        print(f"Saved {len(df)} OCSF events to {output_path}")


def generate_sample_logs(count=1000):
    """Generate sample log data for demonstration."""
    print(f"Generating {count} sample log entries...")

    services = ['web-api', 'auth-service', 'payment-worker']
    levels = ['INFO', 'INFO', 'INFO', 'INFO', 'WARNING', 'ERROR']  # Weighted toward INFO
    messages = [
        ("Checkout completed successfully, trace_id={trace_id}", "INFO"),
        ("User {user_id} fetched from database, trace_id={trace_id}", "INFO"),
        ("Cache hit for user {user_id}, trace_id={trace_id}", "INFO"),
        ("Search completed for query: product", "INFO"),
        ("Payment {payment_id} processed successfully in {duration:.2f}ms", "INFO"),
        ("Slow checkout processing: {duration:.2f}s, trace_id={trace_id}", "WARNING"),
        ("High memory allocation during checkout, trace_id={trace_id}", "WARNING"),
        ("Database timeout during checkout, trace_id={trace_id}", "ERROR"),
        ("Cache miss storm detected, trace_id={trace_id}", "ERROR"),
    ]

    logs = []
    base_time = datetime.now() - timedelta(hours=2)

    for i in range(count):
        msg_template, level = random.choice(messages)
        service = random.choice(services)

        # Generate message with placeholders filled
        message = msg_template.format(
            trace_id=random.randint(10**30, 10**38),
            user_id=random.randint(1, 1000),
            payment_id=random.randint(10000, 99999),
            duration=random.uniform(0.1, 5.0) if 'duration' in msg_template else 0
        )

        timestamp = base_time + timedelta(seconds=i * 7.2)  # ~500 events/hour

        logs.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            "service": service,
            "level": level,
            "message": message
        })

    return logs


def main():
    parser = argparse.ArgumentParser(description='Convert observability data to OCSF format')
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate sample data instead of reading from logs')
    parser.add_argument('--log-dir', default='./logs',
                        help='Directory containing log files (default: ./logs)')
    parser.add_argument('--output-dir', default='./data',
                        help='Output directory for parquet files (default: ./data)')
    parser.add_argument('--count', type=int, default=1000,
                        help='Number of sample events to generate (default: 1000)')

    args = parser.parse_args()

    converter = OCSFConverter()

    # Determine log source
    if args.generate_sample:
        log_entries = generate_sample_logs(args.count)
    elif os.path.isdir(args.log_dir):
        log_entries = args.log_dir
        print(f"Reading logs from {args.log_dir}")
    else:
        print(f"Log directory {args.log_dir} not found.")
        print("Options:")
        print("  1. Run 'docker compose up' first to generate logs")
        print("  2. Use --generate-sample to create sample data")
        print("  3. Specify a different directory with --log-dir")
        sys.exit(1)

    # Convert logs
    log_events = converter.convert_logs_to_ocsf(log_entries)

    if not log_events:
        print("No log events found. Try --generate-sample for demo data.")
        sys.exit(1)

    # Save to parquet
    output_path = os.path.join(args.output_dir, 'ocsf_logs.parquet')
    converter.save_to_parquet(log_events, output_path)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total events: {len(log_events)}")
    if log_events:
        df = pd.DataFrame(log_events)
        print(f"  Services: {df['service'].unique().tolist()}")
        print(f"  Severity distribution:")
        for sev, count in df['severity_id'].value_counts().items():
            sev_name = {1: 'DEBUG', 2: 'INFO', 3: 'WARNING', 4: 'ERROR', 5: 'CRITICAL'}.get(sev, 'UNKNOWN')
            print(f"    {sev_name}: {count}")


if __name__ == '__main__':
    main()

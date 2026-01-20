#!/usr/bin/env python3
"""
Convert raw observability data to OCSF (Open Cybersecurity Schema Framework) format.

Usage:
    # 1. Run docker compose for a while to generate logs:
    docker compose up -d
    sleep 60  # Let it run for a minute

    # 2. Export logs and convert to OCSF:
    docker compose logs --no-color web-api payment-worker auth-service > ./logs/docker.log
    python scripts/convert_to_ocsf.py --log-file ./logs/docker.log

    # Or pipe directly:
    docker compose logs --no-color | python scripts/convert_to_ocsf.py --stdin
"""

import json
import os
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path

# Try to import pandas, provide helpful error if missing
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


class OCSFConverter:
    """
    Convert raw observability data to OCSF (Open Cybersecurity Schema Framework) format.
    """

    def convert_logs_to_ocsf(self, log_lines):
        """
        Convert logs to OCSF format.

        Args:
            log_lines: Iterable of log lines (strings)

        Returns:
            List of OCSF-formatted events
        """
        ocsf_events = []

        for line in log_lines:
            line = line.strip()
            if not line:
                continue

            ocsf_event = self._parse_log_line(line)
            if ocsf_event:
                ocsf_events.append(ocsf_event)

        return ocsf_events

    def _parse_log_line(self, line):
        """Parse a single log line and convert to OCSF format."""

        # Try to extract JSON from the line
        # Docker compose logs format: "container-name-1  | {json...}"
        json_match = re.search(r'\{.*\}', line)
        if json_match:
            try:
                log_entry = json.loads(json_match.group())
                return self._log_to_ocsf(log_entry, line)
            except json.JSONDecodeError:
                pass

        # Try parsing as plain JSON line
        try:
            log_entry = json.loads(line)
            return self._log_to_ocsf(log_entry, line)
        except json.JSONDecodeError:
            pass

        # Skip non-JSON lines (like startup messages)
        return None

    def _log_to_ocsf(self, log_entry, raw_line=""):
        """Convert a parsed log entry to OCSF format."""
        try:
            # Parse timestamp
            timestamp_str = log_entry.get('timestamp', '')
            try:
                if ',' in timestamp_str:
                    timestamp_str = timestamp_str.replace(',', '.')
                dt = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                time_ms = int(dt.timestamp() * 1000)
            except:
                time_ms = int(datetime.now().timestamp() * 1000)

            # Determine service from log entry or raw line
            service = log_entry.get('service', 'unknown')
            if service == 'unknown' and raw_line:
                # Try to extract from docker compose log prefix
                if 'web-api' in raw_line:
                    service = 'web-api'
                elif 'payment-worker' in raw_line:
                    service = 'payment-worker'
                elif 'auth-service' in raw_line:
                    service = 'auth-service'

            message = log_entry.get('message', '')
            level = log_entry.get('level', 'INFO')

            # Map to OCSF Application Activity (class_uid: 6001)
            ocsf_event = {
                "class_uid": 6001,
                "category_uid": 6,
                "severity_id": self._map_severity(level),
                "time": time_ms,
                "metadata": {
                    "version": "1.0.0",
                    "product": {
                        "name": service,
                        "vendor_name": "Demo"
                    }
                },
                "activity_id": 1,
                "message": message,
                "service": service,
                "level": level,
            }

            # Extract trace_id if present
            if 'trace_id=' in message:
                try:
                    trace_id = message.split('trace_id=')[1].split()[0].strip(',')
                    ocsf_event['trace_id'] = trace_id
                except:
                    pass

            # Determine status from message content
            if 'error' in message.lower() or 'failed' in message.lower() or 'timeout' in message.lower():
                ocsf_event['status_id'] = 2  # Failure
            else:
                ocsf_event['status_id'] = 1  # Success

            # Extract duration if present
            duration_match = re.search(r'(\d+\.?\d*)\s*ms', message)
            if duration_match:
                ocsf_event['duration'] = float(duration_match.group(1))

            return ocsf_event

        except Exception as e:
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
        return severity_map.get(str(log_level).upper(), 2)

    def save_to_parquet(self, ocsf_events, output_path):
        """Save OCSF events to Parquet for training."""
        if not ocsf_events:
            print("Warning: No events to save")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(ocsf_events)
        df.to_parquet(output_path, compression='snappy')
        print(f"Saved {len(df)} OCSF events to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Docker service logs to OCSF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export docker logs and convert:
  docker compose logs --no-color > ./logs/docker.log
  python scripts/convert_to_ocsf.py --log-file ./logs/docker.log

  # Or pipe directly:
  docker compose logs --no-color | python scripts/convert_to_ocsf.py --stdin
        """
    )
    parser.add_argument('--log-file',
                        help='Path to log file (e.g., from docker compose logs)')
    parser.add_argument('--stdin', action='store_true',
                        help='Read from stdin (for piping docker compose logs)')
    parser.add_argument('--output-dir', default='./data',
                        help='Output directory for parquet files (default: ./data)')

    args = parser.parse_args()

    if not args.log_file and not args.stdin:
        print("Error: Must specify --log-file or --stdin")
        print()
        print("To capture logs from running Docker services:")
        print("  docker compose logs --no-color > ./logs/docker.log")
        print("  python scripts/convert_to_ocsf.py --log-file ./logs/docker.log")
        print()
        print("Or pipe directly:")
        print("  docker compose logs --no-color | python scripts/convert_to_ocsf.py --stdin")
        sys.exit(1)

    converter = OCSFConverter()

    # Read log lines
    if args.stdin:
        print("Reading from stdin...")
        log_lines = sys.stdin.readlines()
    else:
        print(f"Reading from {args.log_file}...")
        with open(args.log_file, 'r') as f:
            log_lines = f.readlines()

    print(f"Processing {len(log_lines)} lines...")

    # Convert logs
    log_events = converter.convert_logs_to_ocsf(log_lines)

    if not log_events:
        print("No valid log events found.")
        print("Make sure docker compose services are running and generating JSON logs.")
        sys.exit(1)

    # Save to parquet
    output_path = os.path.join(args.output_dir, 'ocsf_logs.parquet')
    converter.save_to_parquet(log_events, output_path)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total events: {len(log_events)}")
    df = pd.DataFrame(log_events)
    print(f"  Services: {df['service'].unique().tolist()}")
    print(f"  Severity distribution:")
    for sev, count in df['severity_id'].value_counts().sort_index().items():
        sev_name = {1: 'DEBUG', 2: 'INFO', 3: 'WARNING', 4: 'ERROR', 5: 'CRITICAL'}.get(sev, 'UNKNOWN')
        print(f"    {sev_name}: {count}")

    if 'status_id' in df.columns:
        failures = (df['status_id'] == 2).sum()
        print(f"  Failures: {failures} ({100*failures/len(df):.1f}%)")


if __name__ == '__main__':
    main()

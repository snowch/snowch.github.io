#!/usr/bin/env python3
"""
Convert raw observability data to OCSF (Open Cybersecurity Schema Framework) format.

This converter produces rich OCSF-compliant data with:
- All required fields (class_uid, category_uid, severity_id, time, type_uid, metadata)
- Nested objects (actor, src_endpoint, dst_endpoint, http_request, http_response)
- Flattened versions for direct ML use (actor.user.name -> actor_user_name)

Usage:
    docker compose logs --no-color > ./logs/docker.log
    python scripts/convert_to_ocsf.py --log-file ./logs/docker.log
"""

import json
import os
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


class OCSFConverter:
    """Convert raw observability data to OCSF format with nested objects."""

    # OCSF class definitions
    OCSF_CLASSES = {
        6001: "Web Resources Activity",
        6002: "Application Lifecycle",
        6003: "API Activity",
        6004: "Web Resource Access Activity",
    }

    # Activity types for API Activity (6003)
    ACTIVITY_NAMES = {
        0: "Unknown",
        1: "Create",
        2: "Read",
        3: "Update",
        4: "Delete",
    }

    # Severity mapping
    SEVERITY_MAP = {
        'DEBUG': 1,
        'INFO': 2,
        'WARNING': 3,
        'WARN': 3,
        'ERROR': 4,
        'CRITICAL': 5,
        'FATAL': 6,
    }

    def convert_logs_to_ocsf(self, log_lines):
        """Convert logs to OCSF format with full schema compliance."""
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
        # Try to extract JSON from docker compose log format: "container-name-1  | {json...}"
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

        return None

    def _log_to_ocsf(self, log_entry, raw_line=""):
        """Convert a parsed log entry to full OCSF format."""
        try:
            # Parse timestamp
            time_ms = log_entry.get('time')
            if not time_ms:
                timestamp_str = log_entry.get('timestamp', '')
                try:
                    if ',' in timestamp_str:
                        timestamp_str = timestamp_str.replace(',', '.')
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace(' ', 'T'))
                    time_ms = int(dt.timestamp() * 1000)
                except:
                    time_ms = int(datetime.now().timestamp() * 1000)

            # Get service name
            service = log_entry.get('service', 'unknown')
            if service == 'unknown' and raw_line:
                for svc in ['web-api', 'payment-worker', 'auth-service']:
                    if svc in raw_line:
                        service = svc
                        break

            # Get OCSF class info (default to API Activity)
            class_uid = log_entry.get('class_uid', 6003)
            class_name = log_entry.get('class_name', self.OCSF_CLASSES.get(class_uid, "API Activity"))
            category_uid = log_entry.get('category_uid', 6)
            category_name = log_entry.get('category_name', "Application Activity")
            activity_id = log_entry.get('activity_id', 1)
            activity_name = log_entry.get('activity_name', self.ACTIVITY_NAMES.get(activity_id, "Unknown"))

            # Calculate type_uid
            type_uid = log_entry.get('type_uid', class_uid * 100 + activity_id)

            # Get severity
            level = log_entry.get('level', 'INFO')
            severity_id = log_entry.get('severity_id', self.SEVERITY_MAP.get(str(level).upper(), 2))

            # Get status info
            status_id = log_entry.get('status_id', 1)
            status = log_entry.get('status', 'Success' if status_id == 1 else 'Failure')
            status_code = log_entry.get('status_code', '200')

            # Build the OCSF event with all fields
            ocsf_event = {
                # Required fields
                "class_uid": class_uid,
                "class_name": class_name,
                "category_uid": category_uid,
                "category_name": category_name,
                "activity_id": activity_id,
                "activity_name": activity_name,
                "type_uid": type_uid,
                "severity_id": severity_id,
                "time": time_ms,

                # Metadata (required)
                "metadata": json.dumps(log_entry.get('metadata', {
                    "version": "1.0.0",
                    "product": {"name": service, "vendor_name": "Demo"}
                })),

                # Status fields (recommended)
                "status_id": status_id,
                "status": status,
                "status_code": status_code,

                # Message
                "message": log_entry.get('message', ''),
                "service": service,
                "level": level,

                # Duration (if present)
                "duration": log_entry.get('duration'),

                # Trace context
                "trace_id": log_entry.get('trace_id'),

                # Raw data (for debugging/training)
                "raw_data": raw_line[:2000] if raw_line else None,
            }

            # Handle nested objects - store as JSON strings for parquet compatibility
            # but also flatten key fields for ML

            # Actor object
            actor = log_entry.get('actor', {})
            if actor:
                ocsf_event["actor"] = json.dumps(actor)
                # Flatten common actor fields
                user = actor.get('user', {})
                ocsf_event["actor_user_uid"] = user.get('uid')
                ocsf_event["actor_user_name"] = user.get('name')
                ocsf_event["actor_user_email"] = user.get('email')
                session = actor.get('session', {})
                ocsf_event["actor_session_uid"] = session.get('uid')

            # Source endpoint
            src_endpoint = log_entry.get('src_endpoint', {})
            if src_endpoint:
                ocsf_event["src_endpoint"] = json.dumps(src_endpoint)
                ocsf_event["src_endpoint_ip"] = src_endpoint.get('ip')
                ocsf_event["src_endpoint_port"] = src_endpoint.get('port')
                ocsf_event["src_endpoint_domain"] = src_endpoint.get('domain')

            # Destination endpoint
            dst_endpoint = log_entry.get('dst_endpoint', {})
            if dst_endpoint:
                ocsf_event["dst_endpoint"] = json.dumps(dst_endpoint)
                ocsf_event["dst_endpoint_ip"] = dst_endpoint.get('ip')
                ocsf_event["dst_endpoint_port"] = dst_endpoint.get('port')
                ocsf_event["dst_endpoint_svc_name"] = dst_endpoint.get('svc_name')

            # HTTP request
            http_request = log_entry.get('http_request', {})
            if http_request:
                ocsf_event["http_request"] = json.dumps(http_request)
                ocsf_event["http_request_method"] = http_request.get('method')
                ocsf_event["http_request_user_agent"] = http_request.get('user_agent')
                url = http_request.get('url', {})
                ocsf_event["http_request_url_path"] = url.get('path')
                ocsf_event["http_request_url_hostname"] = url.get('hostname')
                ocsf_event["http_request_url_scheme"] = url.get('scheme')

            # HTTP response
            http_response = log_entry.get('http_response', {})
            if http_response:
                ocsf_event["http_response"] = json.dumps(http_response)
                ocsf_event["http_response_code"] = http_response.get('code')
                ocsf_event["http_response_latency"] = http_response.get('latency')

            # Device info
            device = log_entry.get('device', {})
            if device:
                ocsf_event["device"] = json.dumps(device)
                ocsf_event["device_hostname"] = device.get('hostname')
                ocsf_event["device_type"] = device.get('type')

            # Resources (affected resources)
            resources = log_entry.get('resources', [])
            if resources:
                ocsf_event["resources"] = json.dumps(resources)
                if resources and len(resources) > 0:
                    ocsf_event["resource_type"] = resources[0].get('type')
                    ocsf_event["resource_uid"] = resources[0].get('uid')

            # Anomaly info (custom extension)
            anomaly = log_entry.get('anomaly', {})
            if anomaly:
                ocsf_event["anomaly"] = json.dumps(anomaly)
                ocsf_event["anomaly_type"] = anomaly.get('type')
                ocsf_event["anomaly_severity"] = anomaly.get('severity')

            # Error info
            error = log_entry.get('error', {})
            if error:
                ocsf_event["error"] = json.dumps(error)
                ocsf_event["error_message"] = error.get('message')
                ocsf_event["error_type"] = error.get('type')

            # Store any unmapped fields
            known_fields = {
                'timestamp', 'time', 'service', 'level', 'message', 'metadata',
                'class_uid', 'class_name', 'category_uid', 'category_name',
                'activity_id', 'activity_name', 'type_uid', 'severity_id',
                'status_id', 'status', 'status_code', 'duration', 'trace_id',
                'actor', 'src_endpoint', 'dst_endpoint', 'http_request',
                'http_response', 'device', 'resources', 'anomaly', 'error'
            }
            unmapped = {k: v for k, v in log_entry.items() if k not in known_fields}
            if unmapped:
                ocsf_event["unmapped"] = json.dumps(unmapped)

            # Remove None values
            ocsf_event = {k: v for k, v in ocsf_event.items() if v is not None}

            return ocsf_event

        except Exception as e:
            return None

    def save_to_parquet(self, ocsf_events, output_path):
        """Save OCSF events to Parquet for training."""
        if not ocsf_events:
            print("Warning: No events to save")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(ocsf_events)
        df.to_parquet(output_path, compression='snappy')
        print(f"Saved {len(df)} OCSF events to {output_path}")
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Convert Docker service logs to OCSF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  docker compose logs --no-color > ./logs/docker.log
  python scripts/convert_to_ocsf.py --log-file ./logs/docker.log

  # Or pipe directly:
  docker compose logs --no-color | python scripts/convert_to_ocsf.py --stdin
        """
    )
    parser.add_argument('--log-file', help='Path to log file')
    parser.add_argument('--stdin', action='store_true', help='Read from stdin')
    parser.add_argument('--output-dir', default='./data', help='Output directory (default: ./data)')

    args = parser.parse_args()

    if not args.log_file and not args.stdin:
        print("Error: Must specify --log-file or --stdin")
        print()
        print("To capture logs from running Docker services:")
        print("  docker compose logs --no-color > ./logs/docker.log")
        print("  python scripts/convert_to_ocsf.py --log-file ./logs/docker.log")
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
    df = converter.save_to_parquet(log_events, output_path)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total events: {len(log_events)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Services: {df['service'].unique().tolist()}")

    print(f"\n  Severity distribution:")
    for sev, count in df['severity_id'].value_counts().sort_index().items():
        sev_name = {1: 'DEBUG', 2: 'INFO', 3: 'WARNING', 4: 'ERROR', 5: 'CRITICAL'}.get(sev, 'UNKNOWN')
        print(f"    {sev_name}: {count}")

    if 'status_id' in df.columns:
        failures = (df['status_id'] == 2).sum()
        print(f"  Failures: {failures} ({100*failures/len(df):.1f}%)")

    # Show available columns
    print(f"\n  Available columns ({len(df.columns)}):")
    # Group columns by category
    core_cols = [c for c in df.columns if c in ['class_uid', 'class_name', 'category_uid', 'category_name', 'activity_id', 'activity_name', 'type_uid', 'severity_id', 'time', 'status_id', 'status', 'status_code', 'message', 'service', 'level', 'duration', 'trace_id']]
    nested_cols = [c for c in df.columns if c in ['metadata', 'actor', 'src_endpoint', 'dst_endpoint', 'http_request', 'http_response', 'device', 'resources', 'anomaly', 'error', 'unmapped', 'raw_data']]
    flat_cols = [c for c in df.columns if c not in core_cols + nested_cols]

    print(f"    Core OCSF fields: {', '.join(core_cols)}")
    print(f"    Nested objects (JSON): {', '.join(nested_cols)}")
    print(f"    Flattened fields: {', '.join(flat_cols)}")


if __name__ == '__main__':
    main()

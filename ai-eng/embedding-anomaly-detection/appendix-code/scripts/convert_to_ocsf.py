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

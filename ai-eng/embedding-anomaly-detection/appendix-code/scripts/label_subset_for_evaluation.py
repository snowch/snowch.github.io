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

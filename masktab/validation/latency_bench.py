import pandas as pd
import time
import logging
from typing import Dict, Any
from masktab.inference.engine import predict

logger = logging.getLogger(__name__)

def benchmark_inference_latency(
    task_name: str, 
    db_path: str, 
    sample_csv_path: str, 
    batch_size: int = 100, 
    n_trials: int = 10
) -> Dict[str, Any]:
    """
    Evaluates inference latency over multiple trials.
    Returns stats and checks against a soft latency target.
    """
    df = pd.read_csv(sample_csv_path)
    replace = False
    
    if len(df) < batch_size:
        logger.warning(f"sample_csv_path has fewer rows ({len(df)}) than batch_size ({batch_size}). Sampling with replacement. Results may be indicative only.")
        replace = True
        
    latencies = []
    
    for _ in range(n_trials):
        batch_df = df.sample(n=batch_size, replace=replace)
        
        start_time = time.perf_counter()
        predict(
            input_data=batch_df,
            task_name=task_name,
            db_path=db_path,
            enable_drift_check=False
        )
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000) # ms
        
    latencies_series = pd.Series(latencies)
    mean_lat = latencies_series.mean()
    median_lat = latencies_series.median()
    p95_lat = latencies_series.quantile(0.95)
    max_lat = latencies_series.max()
    
    # Target: 100ms for batch_size=100. Scale proportionally.
    target_ms = 100.0 * (batch_size / 100.0)
    meets_target = mean_lat <= target_ms
    
    return {
        "mean_ms": mean_lat,
        "median_ms": median_lat,
        "p95_ms": p95_lat,
        "max_ms": max_lat,
        "target_ms": target_ms,
        "meets_target": bool(meets_target),
        "n_trials": n_trials,
        "batch_size": batch_size
    }

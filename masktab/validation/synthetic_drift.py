import pandas as pd
from typing import List, Dict, Any
from masktab.inference.engine import predict
from masktab.drift.psi import InsufficientBatchSizeError

def inject_synthetic_drift(df: pd.DataFrame, features_to_shift: List[str], scale_factor: float = 3.0) -> pd.DataFrame:
    """
    Injects synthetic drift into a dataframe by multiplying specific numerical features by scale_factor.
    This is meant for testing and validation only, not production inference.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")
        
    df_shifted = df.copy()
    
    for feature in features_to_shift:
        if feature not in df_shifted.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe columns")
            
        if not pd.api.types.is_numeric_dtype(df_shifted[feature]):
            raise ValueError(f"Cannot shift non-numeric feature '{feature}'")
            
        df_shifted[feature] = df_shifted[feature] * scale_factor
        
    return df_shifted

def run_drift_validation(
    task_name: str, 
    db_path: str, 
    holdout_csv_path: str, 
    features_to_shift: List[str], 
    scale_factor: float = 3.0
) -> Dict[str, Any]:
    """
    Runs prediction on a baseline holdout set and a synthetically drifted version of it.
    Validates that drift detection flags the shifted features correctly.
    """
    df = pd.read_csv(holdout_csv_path)
    if len(df) < 50:
        raise InsufficientBatchSizeError("Holdout batch size must be >= 50 for drift validation.")
        
    df_shifted = inject_synthetic_drift(df, features_to_shift, scale_factor)
    
    # Run baseline prediction
    baseline_results = predict(
        input_data=df,
        task_name=task_name,
        db_path=db_path,
        enable_drift_check=True
    )
    
    baseline_res = baseline_results[0] if isinstance(baseline_results, list) and baseline_results else {}
    
    # Run drifted prediction
    drifted_results = predict(
        input_data=df_shifted,
        task_name=task_name,
        db_path=db_path,
        enable_drift_check=True
    )
    
    drifted_res = drifted_results[0] if isinstance(drifted_results, list) and drifted_results else {}
    
    baseline_drifted_features = baseline_res.get("drifted_features", [])
    shifted_drifted_features = drifted_res.get("drifted_features", [])
    
    passed = True
    for feat in features_to_shift:
        if feat in baseline_drifted_features:
            passed = False
        if feat not in shifted_drifted_features:
            passed = False
            
    return {
        "baseline_drift_detected": baseline_res.get("drift_detected"),
        "baseline_drifted_features": baseline_drifted_features,
        "shifted_drift_detected": drifted_res.get("drift_detected"),
        "shifted_drifted_features": shifted_drifted_features,
        "passed_validation": passed
    }

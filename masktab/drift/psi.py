import numpy as np
import logging

logger = logging.getLogger(__name__)

class InsufficientBatchSizeError(Exception):
    """Raised when the inference batch size is too small to compute a stable PSI."""
    pass

def compute_psi(training_values: np.ndarray, inference_values: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) for a single numerical feature.
    """
    if len(training_values) == 0 or len(inference_values) == 0:
        raise ValueError("training_values and inference_values cannot be empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
        
    # Bin edges must be computed from training_values only.
    # Using inference-derived bins would make PSI meaningless as the bins would
    # adapt to the new distribution, masking any potential drift.
    bin_edges = np.quantile(training_values, q=np.linspace(0, 1, n_bins + 1))
    
    # Handle low-cardinality features by deduplicating edges
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < len(bin_edges):
        logger.warning(
            f"Feature has fewer distinct values than n_bins ({n_bins}). "
            f"Reducing effective bin count to {max(1, len(unique_edges) - 1)}."
        )
        bin_edges = unique_edges
        
    # It's possible that unique_edges only has 1 element if the feature is perfectly constant.
    # We still need at least 2 edges to define 1 bin.
    if len(bin_edges) < 2:
        # Fallback to a single bin covering all possible values
        bin_edges = np.array([-np.inf, np.inf])

    # Count occurrences in each bin
    # np.histogram uses [a, b) for all but the last bin, which is [a, b].
    expected_counts, _ = np.histogram(training_values, bins=bin_edges)
    actual_counts, _ = np.histogram(inference_values, bins=bin_edges)
    
    total_expected = len(training_values)
    total_actual = len(inference_values)
    
    actual_bins_count = len(expected_counts)
    
    # Add epsilon to avoid zero-probability bins that would produce log(0) or division by zero in the PSI formula.
    epsilon = 1e-4
    
    expected_pct = (expected_counts + epsilon) / (total_expected + actual_bins_count * epsilon)
    actual_pct = (actual_counts + epsilon) / (total_actual + actual_bins_count * epsilon)
    
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)
    
    if np.isnan(psi) or np.isinf(psi):
        raise RuntimeError(f"Computed PSI is {psi}, which is invalid (NaN or inf).")
        
    return float(psi)

def compute_psi_multi_feature(
    training_df_preprocessed: np.ndarray, 
    inference_df_preprocessed: np.ndarray, 
    feature_names: list[str], 
    n_bins: int = 10
) -> dict[str, float]:
    """
    Computes PSI for multiple features independently.
    Note: one-hot expanded categorical columns are treated as independent binary numerical features.
    """
    train_cols = training_df_preprocessed.shape[1] if len(training_df_preprocessed.shape) > 1 else 1
    inf_cols = inference_df_preprocessed.shape[1] if len(inference_df_preprocessed.shape) > 1 else 1
    
    if train_cols != inf_cols or train_cols != len(feature_names):
        raise ValueError(
            f"Shape mismatch: training_df_preprocessed has {train_cols} columns, "
            f"inference_df_preprocessed has {inf_cols} columns, "
            f"but {len(feature_names)} feature_names were provided."
        )
        
    inf_rows = inference_df_preprocessed.shape[0] if len(inference_df_preprocessed.shape) > 1 else len(inference_df_preprocessed)
    if inf_rows < 50:
        raise InsufficientBatchSizeError(
            f"Inference batch size ({inf_rows}) is less than the minimum required (50) for stable PSI computation."
        )
        
    psi_scores = {}
    for i, feature_name in enumerate(feature_names):
        if train_cols == 1:
            train_vals = training_df_preprocessed
            inf_vals = inference_df_preprocessed
        else:
            train_vals = training_df_preprocessed[:, i]
            inf_vals = inference_df_preprocessed[:, i]
            
        psi_scores[feature_name] = compute_psi(train_vals, inf_vals, n_bins)
        
    return psi_scores

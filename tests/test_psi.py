import pytest
import numpy as np
from masktab.drift.psi import compute_psi, compute_psi_multi_feature, InsufficientBatchSizeError

def test_compute_psi_identical_distributions():
    np.random.seed(42)
    train_vals = np.random.normal(0, 1, 1000)
    inf_vals = np.random.normal(0, 1, 1000)
    psi = compute_psi(train_vals, inf_vals)
    assert psi < 0.1  # Identical should be near 0

def test_compute_psi_shifted_distribution():
    np.random.seed(42)
    train_vals = np.random.normal(0, 1, 1000)
    inf_vals = np.random.normal(0, 3, 1000)  # 3x rescaling
    psi = compute_psi(train_vals, inf_vals)
    assert psi > 0.2  # Shifted should be > 0.2

def test_compute_psi_errors():
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_psi(np.array([]), np.array([1, 2]))
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_psi(np.array([1, 2]), np.array([]))
    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        compute_psi(np.array([1, 2]), np.array([1, 2]), n_bins=1)

def test_compute_psi_low_cardinality(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    train_vals = np.array([1, 1, 1, 1, 1])
    inf_vals = np.array([1, 1, 1, 1, 1])
    psi = compute_psi(train_vals, inf_vals, n_bins=10)
    assert psi < 0.1
    assert "fewer distinct values than n_bins" in caplog.text

def test_compute_psi_multi_feature():
    np.random.seed(42)
    train_df = np.random.normal(0, 1, (1000, 2))
    inf_df = train_df.copy() # Use exact same array to ensure PSI near 0
    
    scores = compute_psi_multi_feature(train_df, inf_df, ["feat1", "feat2"])
    assert "feat1" in scores
    assert "feat2" in scores
    assert scores["feat1"] < 0.1
    assert scores["feat2"] < 0.1

def test_compute_psi_multi_feature_errors():
    train_df = np.random.normal(0, 1, (100, 2))
    inf_df = np.random.normal(0, 1, (100, 2))
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_psi_multi_feature(train_df, inf_df, ["feat1"])
        
    inf_df_small = np.random.normal(0, 1, (40, 2))
    with pytest.raises(InsufficientBatchSizeError, match="less than the minimum required"):
        compute_psi_multi_feature(train_df, inf_df_small, ["feat1", "feat2"])

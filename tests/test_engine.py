import pytest
import sqlite3
import os
import torch
import torch.nn as nn
import pandas as pd
import json
from unittest.mock import patch, MagicMock

from masktab.inference.engine import predict, NoActiveHeadError, ArtifactMissingError
from masktab.inference.caching import InferenceCache
from masktab.db.connection import get_connection

# Dummy models
class DummyEncoder(nn.Module):
    def forward(self, x):
        return x

class DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.task_type = "classification"
    def forward(self, x):
        # 2 classes
        out = torch.zeros((x.shape[0], 2))
        out[:, 1] = 10.0 # Will always predict class 1 with high confidence
        return out

@pytest.fixture
def setup_env(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    
    # Save artifacts
    enc_path = str(tmp_path / "enc.pt")
    head_path = str(tmp_path / "head.pt")
    
    # We save actual objects since the engine uses torch.load
    torch.save(DummyEncoder(), enc_path)
    torch.save(DummyHead(), head_path)
    
    conn.execute("INSERT INTO encoder_versions (artifact_path, pretraining_dataset_hash, n_samples, n_epochs, val_reconstruction_loss) VALUES (?, 'hash', 10, 1, 0.1)", (enc_path,))
    conn.execute("INSERT INTO task_head_versions (encoder_version_id, task_name, artifact_path, labeled_dataset_hash, n_samples, val_metric_name, val_metric_value, is_active) VALUES (1, 'test_task', ?, 'hash', 10, 'F1', 0.9, 1)", (head_path,))
    
    # Preprocessing config
    scaler_params = json.dumps({"mean": 0.0, "scale": 1.0})
    conn.execute(
        "INSERT INTO preprocessing_configs (dataset_hash, feature_name, scaler_type, scaler_params_json, imputation_value) VALUES ('hash', 'f1', 'StandardScaler', ?, '0')",
        (scaler_params,)
    )
    conn.commit()
    conn.close()
    return db_path

@patch("masktab.inference.engine.torch.load")
def test_predict_single_row(mock_load, setup_env):
    mock_load.side_effect = lambda path, **kwargs: DummyEncoder() if "enc.pt" in path else DummyHead()
    input_data = {"f1": 1.0}
    res = predict(input_data, "test_task", setup_env, confidence_threshold=0.99)
    assert isinstance(res, dict)
    assert res["prediction"] == 1
    assert res["drift_detected"] is None
    assert "Drift check skipped: single-row input" in res["notes"]

@patch("masktab.inference.engine.torch.load")
def test_predict_batch_no_drift(mock_load, setup_env):
    mock_load.side_effect = lambda path, **kwargs: DummyEncoder() if "enc.pt" in path else DummyHead()
    input_data = pd.DataFrame({"f1": [1.0] * 10}) # < 50, so skipped
    res = predict(input_data, "test_task", setup_env)
    assert isinstance(res, list)
    assert len(res) == 10
    assert res[0]["drift_detected"] is None
    assert any("skipped" in note for note in res[0]["notes"])

@patch("masktab.inference.engine.torch.load")
def test_predict_batch_drift(mock_load, setup_env):
    mock_load.side_effect = lambda path, **kwargs: DummyEncoder() if "enc.pt" in path else DummyHead()
    input_data = pd.DataFrame({"f1": [10.0] * 60}) # >= 50
    res = predict(input_data, "test_task", setup_env, enable_drift_check=True)
    assert len(res) == 60
    assert res[0]["drift_detected"] is True # 10 is far from mean 0, std 1
    
    # Verify drift_log
    conn = get_connection(setup_env)
    count = conn.execute("SELECT COUNT(*) as c FROM drift_log").fetchone()["c"]
    assert count > 0
    conn.close()

def test_predict_errors(setup_env):
    input_data = {"f1": 1.0}
    with pytest.raises(NoActiveHeadError):
        predict(input_data, "nonexistent", setup_env)

    # Missing artifact
    conn = get_connection(setup_env)
    conn.execute("UPDATE task_head_versions SET artifact_path = 'missing.pt'")
    conn.commit()
    conn.close()
    
    with pytest.raises(ArtifactMissingError):
        predict(input_data, "test_task", setup_env)

@patch("masktab.inference.engine.torch.load")
@patch("masktab.inference.engine.Preprocessor.transform")
def test_predict_cache(mock_transform, mock_load, setup_env):
    mock_load.side_effect = lambda path, **kwargs: DummyEncoder() if "enc.pt" in path else DummyHead()
    # Setup mock
    import numpy as np
    mock_transform.return_value = (np.array([[1.0]]), ["f1"])
    
    cache = InferenceCache()
    input_data = pd.DataFrame({"f1": [1.0]})
    
    # First call, cache miss
    res1 = predict(input_data, "test_task", setup_env, cache=cache)
    assert "Cache miss" in res1[0]["notes"]
    assert mock_transform.call_count == 1
    
    # Second call, cache hit
    res2 = predict(input_data, "test_task", setup_env, cache=cache)
    assert "Cache hit" in res2[0]["notes"]
    assert mock_transform.call_count == 1 # unchanged!

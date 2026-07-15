import pytest
import os
import sqlite3
import pandas as pd
from unittest.mock import patch
from masktab.training.pretrain import pretrain_encoder
from masktab.training.finetune import finetune_head, EncoderVersionNotFoundError, FeatureDimensionMismatchError
from masktab.db.connection import get_connection

@pytest.fixture
def db_path(tmp_path):
    db_file = str(tmp_path / "test.db")
    get_connection(db_file).close()
    return db_file

@pytest.fixture
def synthetic_pretrain_csv(tmp_path):
    df = pd.DataFrame({
        "num1": range(100),
        "cat1": ["A", "B", "C", "D"] * 25
    })
    csv_path = str(tmp_path / "pretrain.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def synthetic_finetune_csv(tmp_path):
    df = pd.DataFrame({
        "num1": range(100),
        "cat1": ["A", "B", "C", "D"] * 25,
        "target_clf": ["Class0", "Class1"] * 50,
        "target_reg": [float(x) for x in range(100)]
    })
    csv_path = str(tmp_path / "finetune.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
    
@pytest.fixture
def encoder_version_id(synthetic_pretrain_csv, db_path, tmp_path):
    res = pretrain_encoder(
        csv_path=synthetic_pretrain_csv,
        db_path=db_path,
        epochs=2,
        batch_size=16,
        artifact_dir=str(tmp_path / "encoders"),
        random_seed=42
    )
    return res["encoder_version_id"]

def test_finetune_head_classification(encoder_version_id, synthetic_finetune_csv, db_path, tmp_path):
    res = finetune_head(
        encoder_version_id=encoder_version_id,
        csv_path=synthetic_finetune_csv,
        db_path=db_path,
        target_column="target_clf",
        task_type="classification",
        feature_columns=["num1", "cat1"],
        epochs=3,
        batch_size=16,
        artifact_dir=str(tmp_path / "heads"),
        task_name="test_clf"
    )
    assert res["task_name"] == "test_clf"
    assert res["val_metric_name"] == "F1"
    assert os.path.exists(res["artifact_path"])
    
def test_finetune_head_regression(encoder_version_id, synthetic_finetune_csv, db_path, tmp_path):
    res = finetune_head(
        encoder_version_id=encoder_version_id,
        csv_path=synthetic_finetune_csv,
        db_path=db_path,
        target_column="target_reg",
        task_type="regression",
        feature_columns=["num1", "cat1"],
        epochs=3,
        batch_size=16,
        artifact_dir=str(tmp_path / "heads"),
        task_name="test_reg"
    )
    assert res["task_name"] == "test_reg"
    assert res["val_metric_name"] == "RMSE"
    
def test_finetune_missing_encoder(synthetic_finetune_csv, db_path, tmp_path):
    with pytest.raises(EncoderVersionNotFoundError):
        finetune_head(
            encoder_version_id=999,
            csv_path=synthetic_finetune_csv,
            db_path=db_path,
            target_column="target_clf",
            task_type="classification",
            feature_columns=["num1", "cat1"],
            task_name="test_missing"
        )
        
def test_finetune_leakage(encoder_version_id, synthetic_finetune_csv, db_path, tmp_path):
    with pytest.raises(ValueError, match="cannot be in feature_columns"):
        finetune_head(
            encoder_version_id=encoder_version_id,
            csv_path=synthetic_finetune_csv,
            db_path=db_path,
            target_column="target_clf",
            task_type="classification",
            feature_columns=["num1", "target_clf"],
            task_name="test_leakage"
        )
        
def test_finetune_dimension_mismatch(encoder_version_id, tmp_path, db_path):
    df = pd.DataFrame({
        "num1": range(100),
        "num2": range(100), # extra feature not in pretrain
        "target_clf": ["Class0", "Class1"] * 50
    })
    csv_path = str(tmp_path / "mismatch.csv")
    df.to_csv(csv_path, index=False)
    
    with pytest.raises(FeatureDimensionMismatchError):
        finetune_head(
            encoder_version_id=encoder_version_id,
            csv_path=csv_path,
            db_path=db_path,
            target_column="target_clf",
            task_type="classification",
            feature_columns=["num1", "num2"],
            task_name="test_mismatch"
        )
        
def test_finetune_atomic_active_transition(encoder_version_id, synthetic_finetune_csv, db_path, tmp_path):
    # First head
    finetune_head(
        encoder_version_id=encoder_version_id,
        csv_path=synthetic_finetune_csv,
        db_path=db_path,
        target_column="target_clf",
        task_type="classification",
        feature_columns=["num1", "cat1"],
        epochs=1,
        batch_size=16,
        artifact_dir=str(tmp_path / "heads"),
        task_name="task_atomic"
    )
    
    # Second head, same task_name
    finetune_head(
        encoder_version_id=encoder_version_id,
        csv_path=synthetic_finetune_csv,
        db_path=db_path,
        target_column="target_clf",
        task_type="classification",
        feature_columns=["num1", "cat1"],
        epochs=1,
        batch_size=16,
        artifact_dir=str(tmp_path / "heads"),
        task_name="task_atomic"
    )
    
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT id, is_active FROM task_head_versions WHERE task_name='task_atomic' ORDER BY id")
    rows = cursor.fetchall()
    assert len(rows) == 2
    assert rows[0][1] == 0 # First should be deactivated
    assert rows[1][1] == 1 # Second should be active
    conn.close()

def test_finetune_missing_artifact_on_disk(encoder_version_id, synthetic_finetune_csv, db_path, tmp_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT artifact_path FROM encoder_versions WHERE id=?", (encoder_version_id,))
    path = cursor.fetchone()[0]
    conn.close()
    
    os.remove(path)
    
    with pytest.raises(FileNotFoundError, match="Encoder artifact missing on disk"):
         finetune_head(
            encoder_version_id=encoder_version_id,
            csv_path=synthetic_finetune_csv,
            db_path=db_path,
            target_column="target_clf",
            task_type="classification",
            feature_columns=["num1", "cat1"],
            task_name="test_disk_missing"
        )

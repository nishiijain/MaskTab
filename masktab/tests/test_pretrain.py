import pytest
import os
import sqlite3
import pandas as pd
from unittest.mock import patch
from masktab.training.pretrain import pretrain_encoder
from masktab.db.connection import get_connection

@pytest.fixture
def synthetic_csv(tmp_path):
    df = pd.DataFrame({
        "num1": range(200),
        "num2": [x * 0.5 for x in range(200)],
        "cat1": ["A", "B", "C", "D"] * 50,
        "cat2": ["X", "Y"] * 100,
        "cat3": [f"Cat{i}" for i in range(200)], # high cardinality -> embedding
        "num3": [10.0] * 200
    })
    csv_path = str(tmp_path / "synthetic.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def db_path(tmp_path):
    db_file = str(tmp_path / "test.db")
    # Init schema
    get_connection(db_file).close()
    return db_file

@pytest.fixture
def artifact_dir(tmp_path):
    return str(tmp_path / "artifacts")

def test_pretrain_encoder_success_and_reuse(synthetic_csv, db_path, artifact_dir):
    # First run
    res1 = pretrain_encoder(
        csv_path=synthetic_csv,
        db_path=db_path,
        epochs=3,
        batch_size=16,
        artifact_dir=artifact_dir,
        random_seed=42
    )
    
    assert res1["encoder_version_id"] == 1
    assert os.path.exists(res1["artifact_path"])
    
    # Check DB
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT * FROM encoder_versions WHERE id=1")
    row = cursor.fetchone()
    assert row is not None
    conn.close()
    
    # Second run (should reuse)
    res2 = pretrain_encoder(
        csv_path=synthetic_csv,
        db_path=db_path,
        epochs=3,
        batch_size=16,
        artifact_dir=artifact_dir,
        random_seed=42
    )
    
    assert res2["encoder_version_id"] == 2
    assert os.path.exists(res2["artifact_path"])

def test_pretrain_encoder_exception_rollback(synthetic_csv, db_path, artifact_dir):
    # Mock loss function to raise on 2nd batch or so
    with patch("masktab.training.pretrain.masked_reconstruction_loss", side_effect=RuntimeError("Injected failure")):
        with pytest.raises(RuntimeError, match="Injected failure"):
            pretrain_encoder(
                csv_path=synthetic_csv,
                db_path=db_path,
                epochs=3,
                batch_size=16,
                artifact_dir=artifact_dir
            )
            
    # Check no artifact created
    if os.path.exists(artifact_dir):
        assert len(os.listdir(artifact_dir)) == 0
        
    # Check no DB row created
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM encoder_versions")
    count = cursor.fetchone()[0]
    assert count == 0
    conn.close()

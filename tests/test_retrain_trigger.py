import pytest
import sqlite3
import os
from unittest.mock import patch
from masktab.training.retrain_trigger import evaluate_and_trigger_retrain
from masktab.db.connection import get_connection

@pytest.fixture
def setup_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    
    # Insert a dummy task head
    conn.execute("INSERT INTO encoder_versions (artifact_path, pretraining_dataset_hash, n_samples, n_epochs, val_reconstruction_loss) VALUES ('', 'hash', 10, 1, 0.1)")
    conn.execute("INSERT INTO task_head_versions (encoder_version_id, task_name, artifact_path, labeled_dataset_hash, n_samples, val_metric_name, val_metric_value, is_active) VALUES (1, 'test_task', '', 'hash', 10, 'F1', 0.9, 1)")
    conn.commit()
    conn.close()
    return db_path

def test_evaluate_and_trigger_retrain_none(setup_db):
    res = evaluate_and_trigger_retrain(
        "test_task", setup_db, "dummy.csv", "target", "classification",
        {"retrain_recommendation": "none"}
    )
    assert res["action_taken"] == "none"
    
    # verify log
    conn = get_connection(setup_db)
    row = conn.execute("SELECT * FROM fine_tuning_log WHERE task_name = 'test_task'").fetchone()
    assert row is not None
    assert row["trigger_reason"] == "none"
    conn.close()

@patch("masktab.training.retrain_trigger.finetune_head")
def test_evaluate_and_trigger_retrain_head_only(mock_finetune, setup_db):
    mock_finetune.return_value = {"head_version_id": 2}
    
    res = evaluate_and_trigger_retrain(
        "test_task", setup_db, "dummy.csv", "target", "classification",
        {"retrain_recommendation": "head_only"}
    )
    assert res["action_taken"] == "head_only_retrain"
    assert res["new_head_version_id"] == 2
    
    mock_finetune.assert_called_once()
    
    conn = get_connection(setup_db)
    row = conn.execute("SELECT * FROM fine_tuning_log WHERE task_name = 'test_task'").fetchone()
    assert row["trigger_reason"] == "head_only"
    assert row["new_head_version_id"] == 2
    conn.close()

@patch("masktab.training.retrain_trigger.pretrain_encoder")
@patch("masktab.training.retrain_trigger.finetune_head")
def test_evaluate_and_trigger_retrain_full(mock_finetune, mock_pretrain, setup_db):
    mock_pretrain.return_value = {"encoder_version_id": 2}
    mock_finetune.return_value = {"head_version_id": 3}
    
    res = evaluate_and_trigger_retrain(
        "test_task", setup_db, "dummy.csv", "target", "classification",
        {"retrain_recommendation": "full"}
    )
    assert res["action_taken"] == "full_retrain"
    assert res["new_encoder_version_id"] == 2
    assert res["new_head_version_id"] == 3
    
    mock_pretrain.assert_called_once()
    mock_finetune.assert_called_once()
    
    conn = get_connection(setup_db)
    row = conn.execute("SELECT * FROM fine_tuning_log WHERE task_name = 'test_task'").fetchone()
    assert row["trigger_reason"] == "full"
    conn.close()

def test_evaluate_and_trigger_retrain_invalid(setup_db):
    with pytest.raises(ValueError, match="Invalid retrain_recommendation"):
        evaluate_and_trigger_retrain(
            "test_task", setup_db, "dummy.csv", "target", "classification",
            {"retrain_recommendation": "invalid"}
        )

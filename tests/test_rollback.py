import pytest
import sqlite3
import os
import torch
import torch.nn as nn
from masktab.registry.rollback import rollback_head, rollback_encoder, list_versions, EncoderDimensionMismatchError

@pytest.fixture
def setup_db(tmp_path):
    db_path = str(tmp_path / "test_rollback.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE encoder_versions (id INTEGER PRIMARY KEY, artifact_path TEXT, pretraining_dataset_hash TEXT, n_samples INTEGER, n_epochs INTEGER, val_reconstruction_loss REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        cursor.execute("CREATE TABLE task_head_versions (id INTEGER PRIMARY KEY, encoder_version_id INTEGER, task_name TEXT, artifact_path TEXT, labeled_dataset_hash TEXT, n_samples INTEGER, val_metric_name TEXT, val_metric_value REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, is_active INTEGER DEFAULT 1)")
        
        # Insert encoders
        cursor.execute("INSERT INTO encoder_versions (id, artifact_path, pretraining_dataset_hash, n_samples, n_epochs, val_reconstruction_loss) VALUES (1, ?, 'hash1', 100, 10, 0.1)", (str(tmp_path / "enc1.pt"),))
        cursor.execute("INSERT INTO encoder_versions (id, artifact_path, pretraining_dataset_hash, n_samples, n_epochs, val_reconstruction_loss) VALUES (2, ?, 'hash1', 100, 10, 0.05)", (str(tmp_path / "enc2.pt"),))
        
        # Insert task heads
        cursor.execute("INSERT INTO task_head_versions (id, encoder_version_id, task_name, artifact_path, labeled_dataset_hash, n_samples, val_metric_name, val_metric_value, is_active) VALUES (1, 1, 'taskA', ?, 'hash2', 50, 'loss', 0.2, 0)", (str(tmp_path / "head1.pt"),))
        cursor.execute("INSERT INTO task_head_versions (id, encoder_version_id, task_name, artifact_path, labeled_dataset_hash, n_samples, val_metric_name, val_metric_value, is_active) VALUES (2, 1, 'taskA', ?, 'hash2', 50, 'loss', 0.15, 1)", (str(tmp_path / "head2.pt"),))
        cursor.execute("INSERT INTO task_head_versions (id, encoder_version_id, task_name, artifact_path, labeled_dataset_hash, n_samples, val_metric_name, val_metric_value, is_active) VALUES (3, 1, 'taskB', ?, 'hash2', 50, 'loss', 0.1, 1)", (str(tmp_path / "head3.pt"),))
        
        conn.commit()
    return db_path, tmp_path

def create_mock_encoder(path, hidden_dims):
    # Dummy mock
    state = {"config": {"hidden_dims": hidden_dims}}
    torch.save(state, path)
    
def test_rollback_head_idempotent(setup_db):
    db_path, tmp_path = setup_db
    
    # Active is 2
    res1 = rollback_head('taskA', 2, db_path)
    assert res1['previous_active_id'] == 2
    assert res1['new_active_id'] == 2
    
def test_rollback_head_success(setup_db):
    db_path, tmp_path = setup_db
    
    # Change active from 2 to 1
    res1 = rollback_head('taskA', 1, db_path)
    assert res1['previous_active_id'] == 2
    assert res1['new_active_id'] == 1
    
    # Check DB
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM task_head_versions WHERE task_name = 'taskA' AND is_active = 1")
        assert cursor.fetchone()["id"] == 1

def test_rollback_head_different_task(setup_db):
    db_path, tmp_path = setup_db
    
    # Try to set head 3 (taskB) as active for taskA
    with pytest.raises(ValueError, match="exists but belongs to task 'taskB'"):
        rollback_head('taskA', 3, db_path)

def test_rollback_encoder_dimension_mismatch(setup_db):
    db_path, tmp_path = setup_db
    
    create_mock_encoder(str(tmp_path / "enc1.pt"), (256, 128))
    create_mock_encoder(str(tmp_path / "enc2.pt"), (256, 64))
    
    with pytest.raises(EncoderDimensionMismatchError):
        rollback_encoder('taskA', 2, db_path)

def test_rollback_encoder_success(setup_db):
    db_path, tmp_path = setup_db
    
    create_mock_encoder(str(tmp_path / "enc1.pt"), (256, 128))
    create_mock_encoder(str(tmp_path / "enc2.pt"), (512, 128))
    
    res = rollback_encoder('taskA', 2, db_path)
    assert res['previous_head_id'] == 2
    assert res['new_head_id'] == 4 # Next auto-increment
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, encoder_version_id, is_active FROM task_head_versions WHERE id = 4")
        row = cursor.fetchone()
        assert row["is_active"] == 1
        assert row["encoder_version_id"] == 2
        
        cursor.execute("SELECT is_active FROM task_head_versions WHERE id = 2")
        assert cursor.fetchone()["is_active"] == 0

def test_list_versions(setup_db):
    db_path, tmp_path = setup_db
    
    res = list_versions('taskA', db_path)
    assert res['task_name'] == 'taskA'
    assert len(res['history']) == 2
    
    # Ordered by id DESC
    assert res['history'][0]['head_id'] == 2
    assert res['history'][0]['is_active'] is True
    
    assert res['history'][1]['head_id'] == 1
    assert res['history'][1]['is_active'] is False

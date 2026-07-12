import pytest
import os
from masktab.data.hashing import compute_dataset_hash

def test_compute_dataset_hash_success(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("dummy content")
    features = ["age", "income"]
    
    hash_val = compute_dataset_hash(str(file_path), features)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64

def test_compute_dataset_hash_missing_file():
    with pytest.raises(FileNotFoundError):
        compute_dataset_hash("nonexistent.csv", ["age"])

def test_compute_dataset_hash_empty_features(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("dummy")
    with pytest.raises(ValueError, match="feature_names cannot be empty"):
        compute_dataset_hash(str(file_path), [])

def test_compute_dataset_hash_stable_order(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("dummy")
    
    hash1 = compute_dataset_hash(str(file_path), ["age", "income"])
    hash2 = compute_dataset_hash(str(file_path), ["income", "age"])
    assert hash1 == hash2

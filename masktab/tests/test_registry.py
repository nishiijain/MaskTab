import pytest
import sqlite3
import os
from masktab.data.registry import save_preprocessing_config, load_preprocessing_config

@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    schema_path = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
    with open(schema_path, "r") as f:
        connection.executescript(f.read())
    yield connection
    connection.close()

def test_load_missing_config(conn):
    assert load_preprocessing_config(conn, "hash123") is None

def test_save_and_load_config(conn):
    config = {
        "num_col": {
            "type": "numerical",
            "imputation_value": 1.5,
            "scaler_mean": 2.0,
            "scaler_scale": 0.5
        },
        "cat_col": {
            "type": "categorical",
            "encoding_strategy": "onehot",
            "imputation_value": "a",
            "categories": ["a", "b"]
        },
        "cat_embed": {
            "type": "categorical",
            "encoding_strategy": "embedding",
            "imputation_value": "x",
            "encoding_map": {"x": 1, "y": 2}
        }
    }
    
    for fname, fconfig in config.items():
        save_preprocessing_config(conn, "hash123", fname, fconfig)
        
    loaded = load_preprocessing_config(conn, "hash123")
    assert loaded == config

def test_save_preprocessing_config_overwrite(conn):
    config1 = {
        "type": "numerical",
        "imputation_value": 1.0,
        "scaler_mean": 2.0,
        "scaler_scale": 1.0
    }
    config2 = {
        "type": "numerical",
        "imputation_value": 5.0,
        "scaler_mean": 10.0,
        "scaler_scale": 2.0
    }
    
    save_preprocessing_config(conn, "hash_update", "feat1", config1)
    save_preprocessing_config(conn, "hash_update", "feat1", config2)
    
    loaded = load_preprocessing_config(conn, "hash_update")
    assert loaded["feat1"]["imputation_value"] == 5.0

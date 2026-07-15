import pytest
import pandas as pd
from unittest.mock import patch
from masktab.training.pretrain import pretrain_encoder
from masktab.training.benchmark import run_benchmark
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
def synthetic_benchmark_csv(tmp_path):
    df = pd.DataFrame({
        "num1": range(100),
        "cat1": ["A", "B", "C", "D"] * 25,
        "target_clf": ["Class0", "Class1"] * 50
    })
    csv_path = str(tmp_path / "benchmark.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
    
@pytest.fixture
def encoder_version_id(synthetic_pretrain_csv, db_path, tmp_path):
    res = pretrain_encoder(
        csv_path=synthetic_pretrain_csv,
        db_path=db_path,
        epochs=1,
        batch_size=16,
        artifact_dir=str(tmp_path / "encoders"),
        random_seed=42
    )
    return res["encoder_version_id"]

def test_run_benchmark(encoder_version_id, synthetic_benchmark_csv, db_path):
    res = run_benchmark(
        encoder_version_id=encoder_version_id,
        csv_path=synthetic_benchmark_csv,
        db_path=db_path,
        target_column="target_clf",
        task_type="classification",
        epochs=1,
        batch_size=16
    )
    
    assert "pretrained_full" in res
    assert "random_encoder_full" in res
    assert "supervised_full" in res
    assert "supervised_partial" in res
    assert "pretrained_within_5pct_of_supervised" in res
    assert "pretrained_beats_partial_supervised" in res
    
    # Check boolean computation explicitly via mocking
    with patch('masktab.training.benchmark.finetune_head') as mock_finetune, \
         patch('masktab.training.benchmark.train_head_internal') as mock_train:
         
         # 0.90 is pretrained_full
         mock_finetune.return_value = {"val_metric_name": "F1", "val_metric_value": 0.90}
         
         # (head, val_metric_name, val_metric_value, epochs_run, early_stopped)
         mock_train.side_effect = [
             (None, "F1", 0.60, 1, False), # random_encoder_full
             (None, "F1", 0.92, 1, False), # supervised_full
             (None, "F1", 0.85, 1, False), # supervised_partial
         ]
         
         res_mock = run_benchmark(
             encoder_version_id=encoder_version_id,
             csv_path=synthetic_benchmark_csv,
             db_path=db_path,
             target_column="target_clf",
             task_type="classification",
             feature_columns=["num1", "cat1"],
             epochs=1,
             batch_size=16
         )
         
         # 0.90 >= 0.95 * 0.92 (which is 0.874) -> True
         assert res_mock["pretrained_within_5pct_of_supervised"] is True
         # 0.90 > 0.85 -> True
         assert res_mock["pretrained_beats_partial_supervised"] is True
         
def test_run_benchmark_regression(encoder_version_id, db_path, tmp_path):
    df = pd.DataFrame({
        "num1": range(100),
        "cat1": ["A", "B", "C", "D"] * 25,
        "target_reg": [float(x) for x in range(100)]
    })
    csv_path = str(tmp_path / "benchmark_reg.csv")
    df.to_csv(csv_path, index=False)
    
    # Run once unmocked to populate DB
    res = run_benchmark(
        encoder_version_id=encoder_version_id,
        csv_path=csv_path,
        db_path=db_path,
        target_column="target_reg",
        task_type="regression",
        feature_columns=["num1", "cat1"],
        epochs=1,
        batch_size=16
    )
    
    with patch('masktab.training.benchmark.finetune_head') as mock_finetune, \
         patch('masktab.training.benchmark.train_head_internal') as mock_train:
         
         # RMSE, lower is better. Let's say pretrained is 1.2
         mock_finetune.return_value = {"val_metric_name": "RMSE", "val_metric_value": 1.2, "val_metric_name": "RMSE"}
         
         mock_train.side_effect = [
             (None, "RMSE", 2.0, 1, False), # random
             (None, "RMSE", 1.1, 1, False), # supervised_full
             (None, "RMSE", 1.5, 1, False), # supervised_partial
         ]
         
         res_mock = run_benchmark(
             encoder_version_id=encoder_version_id,
             csv_path=csv_path,
             db_path=db_path,
             target_column="target_reg",
             task_type="regression",
             feature_columns=["num1", "cat1"],
             epochs=1,
             batch_size=16
         )
         
         # 1.2 <= 1.05 * 1.1 (1.155) -> False (1.2 is worse than 1.155)
         assert res_mock["pretrained_within_5pct_of_supervised"] is False
         # 1.2 < 1.5 -> True
         assert res_mock["pretrained_beats_partial_supervised"] is True

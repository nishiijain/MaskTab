import pytest
import os
import pandas as pd
from masktab.training.pretrain import pretrain_encoder
from masktab.training.finetune import finetune_head
from masktab.training.benchmark import run_benchmark
from masktab.inference.engine import predict
from masktab.validation.synthetic_drift import run_drift_validation
from masktab.validation.latency_bench import benchmark_inference_latency
from scripts.download_datasets import fetch_uci_covertype, DatasetUnavailableError
from masktab.db.connection import get_connection

# Constant for subsampling due to large size (581,012 rows)
COVERTYPE_TEST_SAMPLE_SIZE = 5000

@pytest.fixture(scope="module")
def covertype_csv(tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("data"))
    try:
        raw_csv_path = fetch_uci_covertype(out_dir)
        
        # Subsample for test speed
        df = pd.read_csv(raw_csv_path)
        if len(df) > COVERTYPE_TEST_SAMPLE_SIZE:
            df = df.sample(COVERTYPE_TEST_SAMPLE_SIZE, random_state=42)
            
        sampled_csv_path = os.path.join(out_dir, "covertype_sampled.csv")
        df.to_csv(sampled_csv_path, index=False)
        return sampled_csv_path
        
    except DatasetUnavailableError as e:
        pytest.skip(f"Covertype dataset unavailable: {e}")

def test_e2e_covertype(covertype_csv, tmp_path):
    db_path = str(tmp_path / "masktab_cov.db")
    get_connection(db_path).close()
    
    artifact_dir = str(tmp_path / "artifacts")
    
    df_temp = pd.read_csv(covertype_csv, nrows=1)
    feature_cols = [c for c in df_temp.columns if c != "Cover_Type"]

    # 1. Pretrain
    pretrain_res = pretrain_encoder(
        csv_path=covertype_csv,
        db_path=db_path,
        feature_columns=feature_cols,
        epochs=5, # small for test
        batch_size=256,
        learning_rate=1e-3,
        mask_prob=0.3,
        alpha=0.5,
        val_split=0.15,
        artifact_dir=os.path.join(artifact_dir, "encoders"),
        random_seed=42,
        early_stopping_patience=2
    )
    enc_id = pretrain_res["encoder_version_id"]
    
    # 2. Finetune
    ft_res = finetune_head(
        encoder_version_id=enc_id,
        csv_path=covertype_csv,
        db_path=db_path,
        target_column="Cover_Type",
        task_type="classification",
        n_classes=7,
        epochs=5, # small for test
        batch_size=128,
        artifact_dir=os.path.join(artifact_dir, "heads"),
        task_name="covertype_class"
    )
    assert ft_res["task_name"] == "covertype_class"
    
    # 3. Benchmark
    bench_res = run_benchmark(
        encoder_version_id=enc_id,
        csv_path=covertype_csv,
        db_path=db_path,
        target_column="Cover_Type",
        task_type="classification",
        n_classes=7,
        epochs=5,
        batch_size=128
    )
    print(f"Pretrained within 5pct: {bench_res['pretrained_within_5pct_of_supervised']}")
    
    # 4. Predict
    df = pd.read_csv(covertype_csv)
    holdout = df.sample(min(100, len(df)), random_state=42)
    holdout_path = str(tmp_path / "holdout_cov.csv")
    holdout.to_csv(holdout_path, index=False)
    
    pred_res = predict(
        input_data=holdout,
        task_name="covertype_class",
        db_path=db_path,
        enable_drift_check=True
    )
    assert isinstance(pred_res, list)
    assert len(pred_res) == len(holdout)
    
    # 5. Drift Validation
    num_cols = holdout.select_dtypes(include='number').columns.tolist()
    # Ensure we don't pick the target column if it happens to be numeric
    num_cols = [c for c in num_cols if c != "Cover_Type"]
    assert len(num_cols) >= 2, "Need at least 2 numeric columns for drift injection"
    features_to_shift = num_cols[:2]
    
    drift_res = run_drift_validation(
        task_name="covertype_class",
        db_path=db_path,
        holdout_csv_path=holdout_path,
        features_to_shift=features_to_shift,
        scale_factor=3.0
    )
    assert isinstance(drift_res, dict)
    assert "passed_validation" in drift_res
    print(f"Drift validation passed: {drift_res['passed_validation']}")
    
    # 6. Latency Bench
    lat_res = benchmark_inference_latency(
        task_name="covertype_class",
        db_path=db_path,
        sample_csv_path=holdout_path,
        batch_size=100,
        n_trials=3
    )
    print(f"Meets target latency: {lat_res['meets_target']}")

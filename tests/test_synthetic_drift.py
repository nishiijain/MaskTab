import pytest
import pandas as pd
from masktab.validation.synthetic_drift import inject_synthetic_drift, run_drift_validation
from masktab.drift.psi import InsufficientBatchSizeError

def test_inject_synthetic_drift_immutability():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "cat1": ["A", "B", "C"]
    })
    
    df_copy = df.copy()
    
    df_shifted = inject_synthetic_drift(df, ["num1"], 2.0)
    
    # Original is unchanged
    pd.testing.assert_frame_equal(df, df_copy)
    
    # Shifted is multiplied
    assert df_shifted["num1"].tolist() == [2.0, 4.0, 6.0]
    assert df_shifted["cat1"].tolist() == ["A", "B", "C"]

def test_inject_synthetic_drift_non_numeric():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "cat1": ["A", "B", "C"]
    })
    with pytest.raises(ValueError, match="Cannot shift non-numeric feature"):
        inject_synthetic_drift(df, ["cat1"], 2.0)

def test_inject_synthetic_drift_bad_scale():
    df = pd.DataFrame({"num1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="scale_factor must be > 0"):
        inject_synthetic_drift(df, ["num1"], -1.0)

def test_run_drift_validation_insufficient_batch(tmp_path):
    df = pd.DataFrame({"num1": range(10)})
    csv_path = str(tmp_path / "test.csv")
    df.to_csv(csv_path, index=False)
    
    with pytest.raises(InsufficientBatchSizeError):
        run_drift_validation("taskA", "fake.db", csv_path, ["num1"])

# We mock predict to avoid running the full pipeline in unit tests for run_drift_validation
# E2E tests will run the real one.
def test_run_drift_validation_logic(tmp_path, monkeypatch):
    df = pd.DataFrame({"num1": range(100), "num2": range(100)})
    csv_path = str(tmp_path / "test.csv")
    df.to_csv(csv_path, index=False)
    
    # Mock predict
    def mock_predict(input_data, task_name, db_path, enable_drift_check):
        # We can detect if it's the shifted one by checking mean
        mean = input_data["num1"].mean()
        if mean > 100:
            return [{"drift_detected": True, "drifted_features": ["num1"]}]
        else:
            return [{"drift_detected": False, "drifted_features": []}]
            
    monkeypatch.setattr("masktab.validation.synthetic_drift.predict", mock_predict)
    
    res = run_drift_validation("taskA", "fake.db", csv_path, ["num1"], scale_factor=3.0)
    
    assert res["baseline_drift_detected"] is False
    assert res["shifted_drift_detected"] is True
    assert res["passed_validation"] is True

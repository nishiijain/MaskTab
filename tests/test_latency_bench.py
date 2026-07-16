import pytest
import pandas as pd
import logging
from masktab.validation.latency_bench import benchmark_inference_latency

def test_latency_bench_stats(tmp_path, monkeypatch):
    df = pd.DataFrame({"num1": range(100)})
    csv_path = str(tmp_path / "test.csv")
    df.to_csv(csv_path, index=False)
    
    def mock_predict(*args, **kwargs):
        pass # mock zero latency
        
    monkeypatch.setattr("masktab.validation.latency_bench.predict", mock_predict)
    
    res = benchmark_inference_latency("taskA", "fake.db", csv_path, batch_size=10, n_trials=5)
    
    assert "mean_ms" in res
    assert "median_ms" in res
    assert "p95_ms" in res
    assert "max_ms" in res
    
    assert res["mean_ms"] <= res["max_ms"]
    assert res["median_ms"] <= res["max_ms"]
    assert res["p95_ms"] <= res["max_ms"]
    
def test_latency_bench_sampling_warning(tmp_path, monkeypatch, caplog):
    df = pd.DataFrame({"num1": range(10)})
    csv_path = str(tmp_path / "test.csv")
    df.to_csv(csv_path, index=False)
    
    def mock_predict(*args, **kwargs):
        pass
        
    monkeypatch.setattr("masktab.validation.latency_bench.predict", mock_predict)
    
    with caplog.at_level(logging.WARNING):
        benchmark_inference_latency("taskA", "fake.db", csv_path, batch_size=20, n_trials=2)
        
    assert "sampling with replacement" in caplog.text.lower()

import pytest
from masktab.drift.detector import DriftDetector

def test_detector_initialization():
    with pytest.raises(ValueError, match="psi_threshold must be strictly greater than 0"):
        DriftDetector(psi_threshold=0)
    with pytest.raises(ValueError, match="full_retrain_psi_threshold must be strictly greater than 0"):
        DriftDetector(full_retrain_psi_threshold=0)
    with pytest.raises(ValueError, match="cannot be lower than psi_threshold"):
        DriftDetector(psi_threshold=0.3, full_retrain_psi_threshold=0.2)
    with pytest.raises(ValueError, match="full_retrain_feature_fraction must be between 0.0 and 1.0"):
        DriftDetector(full_retrain_feature_fraction=1.5)

def test_detector_evaluate_empty():
    detector = DriftDetector()
    res = detector.evaluate({})
    assert res["retrain_recommendation"] == "none"
    assert res["drift_detected"] is False
    assert res["drifted_features"] == []
    assert res["severe_drift_features"] == []

def test_detector_evaluate_none():
    detector = DriftDetector(psi_threshold=0.2, full_retrain_psi_threshold=0.25)
    res = detector.evaluate({"f1": 0.1, "f2": 0.15})
    assert res["retrain_recommendation"] == "none"
    assert res["drift_detected"] is False

def test_detector_evaluate_head_only():
    detector = DriftDetector(psi_threshold=0.2, full_retrain_psi_threshold=0.25, full_retrain_feature_fraction=0.5)
    res = detector.evaluate({"f1": 0.22, "f2": 0.15, "f3": 0.1})
    # f1 is > 0.2 but < 0.25 -> drifted, not severe.
    assert res["retrain_recommendation"] == "head_only"
    assert res["drift_detected"] is True
    assert "f1" in res["drifted_features"]
    assert len(res["severe_drift_features"]) == 0

def test_detector_evaluate_full():
    detector = DriftDetector(psi_threshold=0.2, full_retrain_psi_threshold=0.25, full_retrain_feature_fraction=0.3)
    # 2 out of 4 features are > 0.25 (0.5 fraction) -> > 0.3 full retrain threshold
    res = detector.evaluate({"f1": 0.3, "f2": 0.26, "f3": 0.1, "f4": 0.1})
    assert res["retrain_recommendation"] == "full"
    assert res["drift_detected"] is True
    assert "f1" in res["severe_drift_features"]
    assert "f2" in res["severe_drift_features"]

import pytest
import torch
from masktab.models.task_head import TaskHead

def test_task_head_init_classification():
    head = TaskHead(input_dim=128, task_type="classification", n_classes=3)
    assert head.get_task_type() == "classification"
    assert head.get_n_classes() == 3
    assert head.head.out_features == 3
    
def test_task_head_init_classification_invalid():
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        TaskHead(input_dim=128, task_type="classification", n_classes=1)
        
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        TaskHead(input_dim=128, task_type="classification", n_classes=None)

def test_task_head_init_regression():
    head = TaskHead(input_dim=128, task_type="regression")
    assert head.get_task_type() == "regression"
    assert head.get_n_classes() is None
    assert head.head.out_features == 1
    
def test_task_head_init_regression_invalid():
    with pytest.raises(ValueError, match="n_classes must be None"):
        TaskHead(input_dim=128, task_type="regression", n_classes=3)
        
def test_task_head_init_invalid_type():
    with pytest.raises(ValueError, match="task_type must be"):
        TaskHead(input_dim=128, task_type="unknown")

def test_task_head_forward():
    head = TaskHead(input_dim=64, task_type="classification", n_classes=5)
    x = torch.randn(10, 64)
    out = head(x)
    assert out.shape == (10, 5)
    
def test_task_head_forward_dim_mismatch():
    head = TaskHead(input_dim=64, task_type="classification", n_classes=5)
    x = torch.randn(10, 32)
    with pytest.raises(RuntimeError, match="Expected input with last dimension 64"):
        head(x)

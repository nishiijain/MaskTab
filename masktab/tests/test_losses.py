import pytest
import torch
from masktab.models.losses import masked_reconstruction_loss

def test_masked_reconstruction_loss_valid():
    reconstructed = torch.tensor([[0.5, 2.0], [0.1, -1.0]])
    original = torch.tensor([[1.0, 2.0], [0.0, -1.5]])
    mask = torch.tensor([[True, False], [False, True]])
    feature_types = ["numerical", "categorical"]
    
    loss, components = masked_reconstruction_loss(reconstructed, original, mask, feature_types, alpha=0.5)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    # Num loss: (0.5 - 1.0)^2 = 0.25 (only 1 masked pos)
    # Cat loss: (-1.0 - (-1.5))^2 = 0.25 (only 1 masked pos)
    # Total loss: 0.5 * 0.25 + 0.5 * 0.25 = 0.25
    assert torch.isclose(loss, torch.tensor(0.25))
    assert components["numerical_loss"] == 0.25
    assert components["categorical_loss"] == 0.25

def test_masked_reconstruction_loss_invalid_alpha():
    reconstructed = torch.zeros(2, 2)
    original = torch.zeros(2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    feature_types = ["numerical", "categorical"]
    
    with pytest.raises(ValueError, match="alpha must be in"):
        masked_reconstruction_loss(reconstructed, original, mask, feature_types, alpha=1.5)

def test_masked_reconstruction_loss_dimension_mismatch():
    reconstructed = torch.zeros(2, 2)
    original = torch.zeros(2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    feature_types = ["numerical"]
    
    with pytest.raises(ValueError, match="Dimension mismatch: len\\(feature_types\\)=1, reconstructed.shape\\[-1\\]=2, original.shape\\[-1\\]=2, mask.shape\\[-1\\]=2"):
        masked_reconstruction_loss(reconstructed, original, mask, feature_types)

def test_masked_reconstruction_loss_zero_masked():
    reconstructed = torch.tensor([[1.0, 2.0]])
    original = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[False, False]])
    feature_types = ["numerical", "categorical"]
    
    loss, components = masked_reconstruction_loss(reconstructed, original, mask, feature_types, alpha=0.5)
    
    assert torch.isclose(loss, torch.tensor(0.0))
    assert components["numerical_loss"] == 0.0
    assert components["categorical_loss"] == 0.0

import pytest
import torch
from masktab.models.masking import generate_feature_mask, apply_mask

def test_generate_feature_mask_valid():
    mask = generate_feature_mask(10, 5, 0.3)
    assert mask.shape == (10, 5)
    assert mask.dtype == torch.bool

def test_generate_feature_mask_invalid_prob():
    with pytest.raises(ValueError, match="mask_prob must be in"):
        generate_feature_mask(10, 5, 1.0)
    with pytest.raises(ValueError, match="mask_prob must be in"):
        generate_feature_mask(10, 5, -0.1)

def test_generate_feature_mask_invalid_dims():
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        generate_feature_mask(0, 5, 0.3)
    with pytest.raises(ValueError, match="num_features must be >= 1"):
        generate_feature_mask(10, 0, 0.3)

def test_generate_feature_mask_reproducibility():
    gen1 = torch.Generator().manual_seed(42)
    mask1 = generate_feature_mask(10, 5, 0.5, generator=gen1)
    
    gen2 = torch.Generator().manual_seed(42)
    mask2 = generate_feature_mask(10, 5, 0.5, generator=gen2)
    
    assert torch.equal(mask1, mask2)

def test_apply_mask():
    x = torch.ones(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    
    masked_x = apply_mask(x, mask, mask_value=0.0)
    
    assert torch.equal(masked_x, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    # Ensure original is not mutated
    assert torch.equal(x, torch.ones(2, 3))

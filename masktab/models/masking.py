import torch
import logging

logger = logging.getLogger(__name__)

def generate_feature_mask(batch_size: int, num_features: int, mask_prob: float = 0.3, generator: torch.Generator | None = None) -> torch.BoolTensor:
    if mask_prob < 0.0 or mask_prob >= 1.0:
        # We reject exactly 1.0 because masking 100% of features leaves the encoder with no signal.
        # This is very likely a caller bug, not a valid use case.
        raise ValueError(f"mask_prob must be in [0.0, 1.0), got {mask_prob}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_features < 1:
        raise ValueError(f"num_features must be >= 1, got {num_features}")
        
    # Generate mask: True means masked
    mask = torch.rand((batch_size, num_features), generator=generator) < mask_prob
    
    if mask_prob < 0.05 or not mask.any():
        logger.warning(f"Generated mask has {mask.sum()} masked positions. Batch effectiveness may be low.")
        
    return mask

def apply_mask(x: torch.Tensor, mask: torch.BoolTensor, mask_value: float = 0.0) -> torch.Tensor:
    x_masked = x.clone()
    x_masked[mask] = mask_value
    return x_masked

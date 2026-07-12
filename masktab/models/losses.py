import torch

def masked_reconstruction_loss(reconstructed: torch.Tensor, original: torch.Tensor, mask: torch.BoolTensor, feature_types: list[str], alpha: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        
    if len(feature_types) != reconstructed.shape[-1] or len(feature_types) != original.shape[-1] or len(feature_types) != mask.shape[-1]:
        raise ValueError(
            f"Dimension mismatch: len(feature_types)={len(feature_types)}, "
            f"reconstructed.shape[-1]={reconstructed.shape[-1]}, "
            f"original.shape[-1]={original.shape[-1]}, "
            f"mask.shape[-1]={mask.shape[-1]}"
        )
        
    numerical_indices = []
    categorical_indices = []
    for i, ftype in enumerate(feature_types):
        if ftype == "numerical":
            numerical_indices.append(i)
        elif ftype == "categorical":
            categorical_indices.append(i)
        else:
            raise ValueError(f"Unknown feature type: {ftype}")
            
    # MSE Loss computed element-wise
    squared_error = (reconstructed - original) ** 2
    masked_sq_err = squared_error * mask.float()
    
    # Numerical loss
    if len(numerical_indices) > 0:
        num_mask = mask[:, numerical_indices]
        num_sq_err = masked_sq_err[:, numerical_indices]
        num_masked_count = num_mask.sum()
        if num_masked_count > 0:
            numerical_loss = num_sq_err.sum() / num_masked_count
        else:
            numerical_loss = torch.tensor(0.0, device=reconstructed.device)
    else:
        numerical_loss = torch.tensor(0.0, device=reconstructed.device)
        
    # Categorical loss
    # NOTE: For this MVP, we treat categorical columns as already ordinally encoded scalar values
    # and compute masked MSE identically to numerical. True categorical reconstruction would
    # require per-column class logits and is a noted extension.
    if len(categorical_indices) > 0:
        cat_mask = mask[:, categorical_indices]
        cat_sq_err = masked_sq_err[:, categorical_indices]
        cat_masked_count = cat_mask.sum()
        if cat_masked_count > 0:
            categorical_loss = cat_sq_err.sum() / cat_masked_count
        else:
            categorical_loss = torch.tensor(0.0, device=reconstructed.device)
    else:
        categorical_loss = torch.tensor(0.0, device=reconstructed.device)
        
    loss_tensor = alpha * numerical_loss + (1 - alpha) * categorical_loss
    
    components_dict = {
        "numerical_loss": float(numerical_loss.item()),
        "categorical_loss": float(categorical_loss.item())
    }
    
    return loss_tensor, components_dict

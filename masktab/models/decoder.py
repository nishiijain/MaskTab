import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: tuple = (256,)):
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if not hidden_dims or len(hidden_dims) < 1:
            raise ValueError(f"hidden_dims must have at least 1 element, got {hidden_dims}")
            
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        layers = []
        current_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.latent_dim:
            raise RuntimeError(f"Expected input with last dimension {self.latent_dim}, but got {x.shape[-1]}")
        return self.network(x)

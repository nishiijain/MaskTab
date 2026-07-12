import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple = (256, 128)):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not hidden_dims or len(hidden_dims) < 1:
            raise ValueError(f"hidden_dims must have at least 1 element, got {hidden_dims}")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(f"Expected input with last dimension {self.input_dim}, but got {x.shape[-1]}")
        return self.network(x)
        
    def get_output_dim(self) -> int:
        return self.hidden_dims[-1]

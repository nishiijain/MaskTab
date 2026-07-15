import torch
import torch.nn as nn

class TaskHead(nn.Module):
    def __init__(self, input_dim: int, task_type: str, n_classes: int | None = None):
        super().__init__()
        
        if task_type not in {"classification", "regression"}:
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")
            
        if task_type == "classification":
            if n_classes is None or n_classes < 2:
                raise ValueError(f"For classification, n_classes must be >= 2, got {n_classes}")
            out_dim = n_classes
        else: # regression
            if n_classes is not None:
                raise ValueError(f"For regression, n_classes must be None, got {n_classes}")
            out_dim = 1
            
        self.input_dim = input_dim
        self.task_type = task_type
        self.n_classes = n_classes
        
        self.head = nn.Linear(input_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(f"Expected input with last dimension {self.input_dim}, but got {x.shape[-1]}")
        return self.head(x)
        
    def get_task_type(self) -> str:
        return self.task_type
        
    def get_n_classes(self) -> int | None:
        return self.n_classes

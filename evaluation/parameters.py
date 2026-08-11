import torch
import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    """
    Counts total trainable and non-trainable structural parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())

def count_nonzero_parameters(model: nn.Module) -> int:
    """
    Counts non-zero elements across all tensors in the model.
    """
    return sum(torch.count_nonzero(p).item() for p in model.parameters())

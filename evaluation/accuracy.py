import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> float:
    """
    Evaluates top-1 accuracy of a model on the given dataset loader.
    Returns float accuracy in range [0.0, 1.0].
    """
    # Automatically fall back to CPU if model is quantized (quantized::linear_dynamic is CPU-only in PyTorch)
    if getattr(model, "is_quantized", False) or any("quantized" in type(m).__module__.lower() or "quantized" in type(m).__name__.lower() for m in model.modules()):
        device = "cpu"

    model.to(device)
    model.eval()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total if total > 0 else 0.0

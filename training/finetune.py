import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.0003,
    device: str = "cpu"
) -> float:
    """
    Fast fine-tuning loop to adapt surviving weights after pruning/layer removal.
    Runs for a small number of epochs (1..3).
    Returns average training loss.
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    total_loss = 0.0
    total_samples = 0

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / max(1, total_samples)

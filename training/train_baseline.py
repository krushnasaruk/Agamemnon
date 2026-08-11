import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from models.baseline_cnn import BaselineCNN, build_baseline_cnn
from data.cifar_loader import get_cifar_dataloaders
from evaluation.accuracy import evaluate_accuracy
from evaluation.parameters import count_parameters
from evaluation.flops import count_flops
from evaluation.latency import measure_latency, get_model_size_mb

def train_baseline_model(
    dataset_name: str = "CIFAR-10",
    data_dir: str = "./data",
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = "cpu",
    save_path: str = "./checkpoints/baseline_cnn.pt"
) -> Tuple[BaselineCNN, dict]:
    """
    Trains baseline oversized CNN to high accuracy and records initial baseline metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"--- Training Baseline CNN on {dataset_name} ({epochs} epochs) ---")

    train_loader, val_loader = get_cifar_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size
    )

    model = build_baseline_cnn(channels=[64, 128, 256, 256, 512], num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        scheduler.step()
        epoch_loss = running_loss / total
        acc = evaluate_accuracy(model, val_loader, device=device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "channels": model.get_channel_config(),
                "accuracy": acc
            }, save_path)

    # Load best model checkpoint
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    acc = evaluate_accuracy(model, val_loader, device=device)
    params = count_parameters(model)
    flops = count_flops(model)
    latency = measure_latency(model, device=device)
    size_mb = get_model_size_mb(model)

    metrics = {
        "accuracy": acc,
        "parameters": params,
        "flops": flops,
        "latency_ms": latency,
        "size_mb": size_mb,
        "channels": model.get_channel_config()
    }

    print("\n=== Baseline CNN Training Complete ===")
    print(f"Accuracy: {acc*100:.2f}% | Parameters: {params:,} | FLOPs: {flops:,} | Latency: {latency:.2f}ms | Size: {size_mb:.2f}MB")
    return model, metrics

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_baseline_model(epochs=5, device=device)

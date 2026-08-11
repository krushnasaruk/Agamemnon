import copy
import torch
from typing import List, Tuple
from models.baseline_cnn import BaselineCNN
from pruning.channel_reduction import prune_layer_channels
from evaluation.accuracy import evaluate_accuracy

def greedy_prune_step(
    model: BaselineCNN,
    val_loader,
    device: str = "cpu",
    reduction_ratio: float = 0.25
) -> Tuple[BaselineCNN, List[int], int]:
    """
    Evaluates pruning each layer individually and selects the layer pruning choice
    that yields highest top-1 validation accuracy.
    """
    best_acc = -1.0
    best_model = model
    best_channels = model.get_channel_config()
    best_layer_idx = 0

    channels = model.get_channel_config()
    for layer_idx in range(len(channels)):
        cand_model, cand_channels = prune_layer_channels(
            model=model,
            layer_idx=layer_idx,
            reduction_ratio=reduction_ratio,
            min_channels=8
        )
        cand_acc = evaluate_accuracy(cand_model, val_loader, device=device)
        if cand_acc > best_acc:
            best_acc = cand_acc
            best_model = cand_model
            best_channels = cand_channels
            best_layer_idx = layer_idx

    return best_model, best_channels, best_layer_idx

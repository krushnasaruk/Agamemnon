import torch
import torch.nn as nn
from typing import List, Tuple
from models.baseline_cnn import BaselineCNN
from pruning.channel_reduction import prune_layer_channels

def magnitude_prune_model(
    model: BaselineCNN,
    target_ratio: float = 0.5
) -> Tuple[BaselineCNN, List[int]]:
    """
    Applies global L1-magnitude structured channel pruning uniformly across all conv layers.
    """
    current_model = model
    channels = model.get_channel_config()
    for layer_idx in range(len(channels)):
        current_model, channels = prune_layer_channels(
            model=current_model,
            layer_idx=layer_idx,
            reduction_ratio=target_ratio,
            min_channels=8
        )
    return current_model, channels

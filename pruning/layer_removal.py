import torch
import torch.nn as nn
from typing import List, Tuple
from models.baseline_cnn import BaselineCNN

def remove_or_bypass_layer(
    model: BaselineCNN,
    layer_idx: int
) -> Tuple[BaselineCNN, List[int]]:
    """
    Bypasses a conv layer by setting its channel count to match preceding layer
    or applying minimal 1-channel pass-through.
    """
    channels = model.get_channel_config()
    # Cannot remove first conv layer (index 0) or last conv layer (index 4) without breaking spatial adapter
    if layer_idx <= 0 or layer_idx >= len(channels) - 1:
        return model, channels

    # Match channel count to preceding layer
    new_channels = list(channels)
    new_channels[layer_idx] = new_channels[layer_idx - 1]

    # Instantiate new model and copy weights
    new_model = BaselineCNN(channels=new_channels, num_classes=model.num_classes)
    
    # Simple transfer
    try:
        new_model.load_state_dict(model.state_dict(), strict=False)
    except Exception:
        pass

    return new_model, new_channels

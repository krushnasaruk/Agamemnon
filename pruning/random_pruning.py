import random
import torch
import torch.nn as nn
from typing import List, Tuple
from models.baseline_cnn import BaselineCNN

def random_prune_model(
    model: BaselineCNN,
    target_ratio: float = 0.5
) -> Tuple[BaselineCNN, List[int]]:
    """
    Randomly selects channels across all layers to achieve target overall parameter reduction ratio.
    """
    channels = model.get_channel_config()
    new_channels = [max(8, int(c * (1.0 - target_ratio))) for c in channels]

    new_model = BaselineCNN(channels=new_channels, num_classes=model.num_classes)
    
    # Randomly select filter indices and transfer weight slices
    conv_layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
    bn_layers = [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]

    new_conv_layers = [new_model.conv1, new_model.conv2, new_model.conv3, new_model.conv4, new_model.conv5]
    new_bn_layers = [new_model.bn1, new_model.bn2, new_model.bn3, new_model.bn4, new_model.bn5]

    prev_indices = None
    for i in range(5):
        orig_out_c = channels[i]
        new_out_c = new_channels[i]
        
        # Select random subset of out-channels
        indices = torch.tensor(sorted(random.sample(range(orig_out_c), new_out_c)))
        
        # Copy conv weights
        if i == 0:
            new_conv_layers[i].weight.data.copy_(conv_layers[i].weight.data[indices])
        else:
            weight_slice = conv_layers[i].weight.data[indices][:, prev_indices]
            new_conv_layers[i].weight.data.copy_(weight_slice)
            
        # Copy BN weights
        new_bn_layers[i].weight.data.copy_(bn_layers[i].weight.data[indices])
        new_bn_layers[i].bias.data.copy_(bn_layers[i].bias.data[indices])
        new_bn_layers[i].running_mean.copy_(bn_layers[i].running_mean[indices])
        new_bn_layers[i].running_var.copy_(bn_layers[i].running_var[indices])
        
        prev_indices = indices

    # Copy FC layer
    new_model.fc.weight.data.copy_(model.fc.weight.data[:, prev_indices])
    new_model.fc.bias.data.copy_(model.fc.bias.data)

    return new_model, new_channels

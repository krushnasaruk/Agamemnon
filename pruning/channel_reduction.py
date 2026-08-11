import torch
import torch.nn as nn
from typing import List, Tuple
from models.baseline_cnn import BaselineCNN

def prune_layer_channels(
    model: BaselineCNN,
    layer_idx: int,
    reduction_ratio: float = 0.25,
    min_channels: int = 8
) -> Tuple[BaselineCNN, List[int]]:
    """
    Prunes channels of a specific layer in BaselineCNN using L1-norm filter importance.
    Preserves surviving weights for fast fine-tuning.
    
    layer_idx: 0 to 4 (corresponding to conv1..conv5)
    reduction_ratio: fraction of channels to remove (e.g. 0.25 = remove 25%)
    min_channels: safety floor to prevent layer collapse
    """
    current_channels = model.get_channel_config()
    orig_c = current_channels[layer_idx]
    
    # Calculate target channel count
    target_c = max(min_channels, int(orig_c * (1.0 - reduction_ratio)))
    if target_c >= orig_c:
        return model, current_channels

    new_channels = list(current_channels)
    new_channels[layer_idx] = target_c

    # Instantiate new model with updated channel dimensions
    new_model = BaselineCNN(channels=new_channels, num_classes=model.num_classes)

    conv_layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
    bn_layers = [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]

    new_conv_layers = [new_model.conv1, new_model.conv2, new_model.conv3, new_model.conv4, new_model.conv5]
    new_bn_layers = [new_model.bn1, new_model.bn2, new_model.bn3, new_model.bn4, new_model.bn5]

    # Calculate L1 norm for filters of target layer
    target_conv = conv_layers[layer_idx]
    weight_data = target_conv.weight.data # Shape: [out_c, in_c, k, k]
    l1_norms = weight_data.abs().sum(dim=[1, 2, 3])
    
    # Get indices of top channels to keep
    top_indices = torch.topk(l1_norms, k=target_c, largest=True).indices.sort().values

    # Copy weights for all layers before layer_idx directly
    for i in range(layer_idx):
        new_conv_layers[i].weight.data.copy_(conv_layers[i].weight.data)
        new_bn_layers[i].weight.data.copy_(bn_layers[i].weight.data)
        new_bn_layers[i].bias.data.copy_(bn_layers[i].bias.data)
        new_bn_layers[i].running_mean.copy_(bn_layers[i].running_mean)
        new_bn_layers[i].running_var.copy_(bn_layers[i].running_var)

    # Copy target pruned layer out-channels
    new_conv_layers[layer_idx].weight.data.copy_(weight_data[top_indices])
    target_bn = bn_layers[layer_idx]
    new_target_bn = new_bn_layers[layer_idx]
    new_target_bn.weight.data.copy_(target_bn.weight.data[top_indices])
    new_target_bn.bias.data.copy_(target_bn.bias.data[top_indices])
    new_target_bn.running_mean.copy_(target_bn.running_mean[top_indices])
    new_target_bn.running_var.copy_(target_bn.running_var[top_indices])

    # If target layer is not the last conv layer, slice next layer's in-channels
    if layer_idx < 4:
        next_conv = conv_layers[layer_idx + 1]
        new_next_conv = new_conv_layers[layer_idx + 1]
        # Slice in-channels according to top_indices
        new_next_conv.weight.data.copy_(next_conv.weight.data[:, top_indices, :, :])

        # Copy remaining downstream layers
        for i in range(layer_idx + 2, 5):
            new_conv_layers[i].weight.data.copy_(conv_layers[i].weight.data)
            new_bn_layers[i].weight.data.copy_(bn_layers[i].weight.data)
            new_bn_layers[i].bias.data.copy_(bn_layers[i].bias.data)
            new_bn_layers[i].running_mean.copy_(bn_layers[i].running_mean)
            new_bn_layers[i].running_var.copy_(bn_layers[i].running_var)
    else:
        # If last conv layer was pruned, update FC layer input weights
        new_model.fc.weight.data.copy_(model.fc.weight.data[:, top_indices])
        new_model.fc.bias.data.copy_(model.fc.bias.data)

    if layer_idx < 4:
        # Always copy FC layer if not modified above
        new_model.fc.weight.data.copy_(model.fc.weight.data)
        new_model.fc.bias.data.copy_(model.fc.bias.data)

    return new_model, new_channels

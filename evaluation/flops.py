import torch
import torch.nn as nn
from typing import Tuple

def count_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> int:
    """
    Analytically calculates total FLOPs (Floating Point Operations) for standard CNNs
    including Conv2d and Linear layers.
    """
    total_flops = 0
    is_quantized = getattr(model, "is_quantized", False) or any("quantized" in type(m).__module__.lower() or "quantized" in type(m).__name__.lower() for m in model.modules())
    if is_quantized:
        device = torch.device("cpu")
    else:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    hooks = []
    
    def conv2d_hook(module: nn.Conv2d, input: Tuple[torch.Tensor], output: torch.Tensor):
        nonlocal total_flops
        batch_size = input[0].shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        bias_ops = 1 if module.bias is not None else 0
        # Multiply-Accumulate = 2 FLOPs per op
        flops_per_instance = 2 * kernel_ops + bias_ops
        total_flops += batch_size * module.out_channels * out_h * out_w * flops_per_instance

    def linear_hook(module: nn.Linear, input: Tuple[torch.Tensor], output: torch.Tensor):
        nonlocal total_flops
        batch_size = input[0].shape[0]
        weight_ops = module.in_features
        bias_ops = 1 if module.bias is not None else 0
        flops_per_instance = 2 * weight_ops + bias_ops
        total_flops += batch_size * module.out_features * flops_per_instance

    # Register forward hooks on conv and linear modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    dummy_input = torch.randn(*input_size, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    return total_flops

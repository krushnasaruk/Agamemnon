import torch
import torch.nn as nn

def quantize_model(model: nn.Module, dtype=torch.qint8) -> nn.Module:
    """
    Applies dynamic post-training INT8 quantization to linear layers and model parameters.
    """
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model.to("cpu"),
        {nn.Linear, nn.Conv2d},
        dtype=dtype
    )
    return quantized_model

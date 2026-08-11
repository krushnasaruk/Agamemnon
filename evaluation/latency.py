import time
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 32, 32),
    runs: int = 50,
    warmup: int = 10,
    device: str = "cpu"
) -> float:
    """
    Measures physical inference latency (in milliseconds) of a forward pass.
    """
    # Automatically fall back to CPU if model is quantized (quantized::linear_dynamic is CPU-only in PyTorch)
    if getattr(model, "is_quantized", False) or any("quantized" in type(m).__module__.lower() or "quantized" in type(m).__name__.lower() for m in model.modules()):
        device = "cpu"

    model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_size, device=device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # High-precision measurement
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / runs) * 1000.0
    return avg_latency_ms

def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculates estimated in-memory model footprint size in Megabytes (MB).
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return float(size_all_mb)

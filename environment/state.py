import numpy as np
from typing import List

class StateRepresentation:
    """
    Constructs normalized vector representations of current architecture state.
    Vector structure (length = 10):
      [0] normalized_accuracy (0.0 to 1.0)
      [1] normalized_params (params / base_params)
      [2] normalized_flops (flops / base_flops)
      [3] normalized_latency (latency / base_latency)
      [4..8] layer_channel_ratios (c_i / c_i_baseline)
      [9] normalized_step (step / max_steps)
    """
    def __init__(
        self,
        base_channels: List[int] = [64, 128, 256, 256, 512],
        base_params: int = 5_000_000,
        base_flops: int = 300_000_000,
        base_latency: float = 20.0,
        max_steps: int = 15
    ):
        self.base_channels = list(base_channels)
        self.base_params = max(1, base_params)
        self.base_flops = max(1, base_flops)
        self.base_latency = max(0.001, base_latency)
        self.max_steps = max(1, max_steps)

    def encode(
        self,
        accuracy: float,
        params: int,
        flops: int,
        latency: float,
        current_channels: List[int],
        step: int
    ) -> np.ndarray:
        norm_acc = float(np.clip(accuracy, 0.0, 1.0))
        norm_params = float(params / self.base_params)
        norm_flops = float(flops / self.base_flops)
        norm_latency = float(latency / self.base_latency)

        channel_ratios = [
            float(curr / base) for curr, base in zip(current_channels, self.base_channels)
        ]
        norm_step = float(step / self.max_steps)

        state_vector = np.array(
            [norm_acc, norm_params, norm_flops, norm_latency] + channel_ratios + [norm_step],
            dtype=np.float32
        )
        return state_vector

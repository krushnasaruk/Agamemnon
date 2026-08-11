from typing import Dict, Any

class MultiObjectiveReward:
    """
    Computes constraint-based multi-objective reward combining accuracy, parameter reduction,
    FLOPs reduction, and hardware latency drop.
    """
    def __init__(
        self,
        target_accuracy: float = 0.90,
        base_params: int = 5_000_000,
        base_flops: int = 300_000_000,
        base_latency: float = 20.0,
        w_acc: float = 2.0,
        w_params: float = 1.0,
        w_flops: float = 1.0,
        w_latency: float = 1.0,
        penalty_multiplier: float = 10.0
    ):
        self.target_accuracy = target_accuracy
        self.base_params = max(1, base_params)
        self.base_flops = max(1, base_flops)
        self.base_latency = max(0.001, base_latency)
        self.w_acc = w_acc
        self.w_params = w_params
        self.w_flops = w_flops
        self.w_latency = w_latency
        self.penalty_multiplier = penalty_multiplier

    def compute_reward(
        self,
        accuracy: float,
        params: int,
        flops: int,
        latency: float
    ) -> float:
        if accuracy < self.target_accuracy:
            # Heavy penalty proportional to accuracy deficit below target
            accuracy_deficit = self.target_accuracy - accuracy
            return float(-self.penalty_multiplier * accuracy_deficit * 100.0)

        # Accuracy constraint satisfied -> Reward efficiency gains
        acc_bonus = (accuracy - self.target_accuracy) * self.w_acc * 10.0
        param_gain = (1.0 - (params / self.base_params)) * self.w_params * 5.0
        flops_gain = (1.0 - (flops / self.base_flops)) * self.w_flops * 5.0
        latency_gain = (1.0 - (latency / self.base_latency)) * self.w_latency * 5.0

        total_reward = acc_bonus + param_gain + flops_gain + latency_gain
        return float(total_reward)

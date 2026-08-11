import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from typing import Tuple, Dict, Any, Optional

from models.baseline_cnn import BaselineCNN, build_baseline_cnn
from pruning.channel_reduction import prune_layer_channels
from pruning.layer_removal import remove_or_bypass_layer
from pruning.quantization import quantize_model
from evaluation.accuracy import evaluate_accuracy
from evaluation.parameters import count_parameters
from evaluation.flops import count_flops
from evaluation.latency import measure_latency
from environment.state import StateRepresentation
from environment.actions import ACTION_SPACE_SIZE, ACTION_DESCRIPTIONS
from environment.reward import MultiObjectiveReward
from training.finetune import finetune_model

class ArchitectureSearchEnv(gym.Env):
    """
    Gymnasium environment for RL-based neural network architecture search and compression.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        baseline_model: BaselineCNN,
        train_loader,
        val_loader,
        device: str = "cpu",
        target_accuracy: float = 0.90,
        max_steps: int = 15,
        finetune_epochs: int = 1,
        finetune_lr: float = 0.0003
    ):
        super(ArchitectureSearchEnv, self).__init__()

        self.initial_baseline_model = baseline_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.target_accuracy = target_accuracy
        self.max_steps = max_steps
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr

        # Evaluate baseline characteristics
        self.current_model = copy.deepcopy(self.initial_baseline_model).to(self.device)
        self.base_accuracy = evaluate_accuracy(self.current_model, self.val_loader, device=self.device)
        self.base_params = count_parameters(self.current_model)
        self.base_flops = count_flops(self.current_model)
        self.base_latency = measure_latency(self.current_model, device=self.device)
        self.base_channels = self.current_model.get_channel_config()

        # State & Reward modules
        self.state_encoder = StateRepresentation(
            base_channels=self.base_channels,
            base_params=self.base_params,
            base_flops=self.base_flops,
            base_latency=self.base_latency,
            max_steps=self.max_steps
        )
        self.reward_calculator = MultiObjectiveReward(
            target_accuracy=self.target_accuracy,
            base_params=self.base_params,
            base_flops=self.base_flops,
            base_latency=self.base_latency
        )

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(10,), dtype=np.float32
        )

        self.current_step_count = 0
        self.is_quantized = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_model = copy.deepcopy(self.initial_baseline_model).to(self.device)
        self.current_step_count = 0
        self.is_quantized = False

        state = self._get_current_state(self.base_accuracy)
        info = {
            "accuracy": self.base_accuracy,
            "params": self.base_params,
            "flops": self.base_flops,
            "latency": self.base_latency,
            "channels": self.base_channels
        }
        return state, info

    def _get_current_state(self, accuracy: float) -> np.ndarray:
        curr_channels = self.current_model.get_channel_config() if hasattr(self.current_model, 'get_channel_config') else []
        curr_params = count_parameters(self.current_model)
        curr_flops = count_flops(self.current_model)
        eval_device = "cpu" if self.is_quantized else self.device
        curr_latency = measure_latency(self.current_model, device=eval_device)

        return self.state_encoder.encode(
            accuracy=accuracy,
            params=curr_params,
            flops=curr_flops,
            latency=curr_latency,
            current_channels=curr_channels,
            step=self.current_step_count
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step_count += 1
        terminated = False
        truncated = self.current_step_count >= self.max_steps

        action_desc = ACTION_DESCRIPTIONS.get(action, "Unknown")

        # Action 0: Terminate
        if action == 0:
            terminated = True
        elif 1 <= action <= 5:
            # Channel reduction on Conv layer (action - 1)
            layer_idx = action - 1
            self.current_model, _ = prune_layer_channels(
                model=self.current_model,
                layer_idx=layer_idx,
                reduction_ratio=0.25,
                min_channels=8
            )
            # Short fine-tuning to adapt surviving weights
            if self.finetune_epochs > 0:
                finetune_model(
                    model=self.current_model,
                    train_loader=self.train_loader,
                    epochs=self.finetune_epochs,
                    lr=self.finetune_lr,
                    device=self.device
                )
        elif 6 <= action <= 8:
            # Bypass/Remove conv layer (action - 5)
            layer_idx = action - 5
            self.current_model, _ = remove_or_bypass_layer(
                model=self.current_model,
                layer_idx=layer_idx
            )
            if self.finetune_epochs > 0:
                finetune_model(
                    model=self.current_model,
                    train_loader=self.train_loader,
                    epochs=self.finetune_epochs,
                    lr=self.finetune_lr,
                    device=self.device
                )
        elif action == 9:
            # Apply INT8 dynamic quantization
            self.current_model = quantize_model(self.current_model)
            self.is_quantized = True
            terminated = True  # Quantization is terminal action

        # Evaluate performance metrics
        eval_device = "cpu" if self.is_quantized else self.device
        acc = evaluate_accuracy(self.current_model, self.val_loader, device=eval_device)
        params = count_parameters(self.current_model)
        flops = count_flops(self.current_model)
        latency = measure_latency(self.current_model, device=eval_device)

        # Compute multi-objective reward
        reward = self.reward_calculator.compute_reward(
            accuracy=acc, params=params, flops=flops, latency=latency
        )

        state = self._get_current_state(acc)
        info = {
            "action": action,
            "action_desc": action_desc,
            "accuracy": acc,
            "params": params,
            "flops": flops,
            "latency": latency,
            "channels": self.current_model.get_channel_config() if hasattr(self.current_model, 'get_channel_config') else [],
            "reward": reward
        }

        return state, reward, terminated, truncated, info

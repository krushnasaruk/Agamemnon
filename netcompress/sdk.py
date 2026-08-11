import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from environment.rl_environment import ArchitectureSearchEnv
from agent.ppo_agent import PPOAgent
from agent.dqn_agent import DQNAgent
from evaluation.accuracy import evaluate_accuracy
from evaluation.parameters import count_parameters
from evaluation.flops import count_flops
from evaluation.latency import measure_latency, get_model_size_mb

class RLCompressor:
    """
    High-level Python SDK wrapper for RL-NetCompress.
    Enables external PyTorch developers to compress custom models automatically with 3 lines of code.
    """
    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        train_loader: Optional[DataLoader] = None,
        target_accuracy: float = 0.85,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.baseline_model = model.to(self.device)
        self.val_loader = val_loader
        self.train_loader = train_loader or val_loader
        self.target_accuracy = target_accuracy
        
        self.initial_acc = evaluate_accuracy(self.baseline_model, self.val_loader, device=self.device)
        self.initial_params = count_parameters(self.baseline_model)
        self.initial_flops = count_flops(self.baseline_model)
        self.initial_latency = measure_latency(self.baseline_model, device=self.device)
        
        self.compressed_model = None
        self.history = []

    def compress(
        self,
        episodes: int = 30,
        agent_type: str = "PPO",
        lr: float = 0.0003,
        gamma: float = 0.99
    ) -> nn.Module:
        """
        Executes autonomous RL architecture search on the target neural network.
        """
        env = ArchitectureSearchEnv(
            baseline_model=self.baseline_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            target_accuracy=self.target_accuracy
        )

        if agent_type.upper() == "PPO":
            agent = PPOAgent(state_dim=10, action_dim=10, lr=lr, gamma=gamma, device=self.device)
        else:
            agent = DQNAgent(state_dim=10, action_dim=10, lr=lr, gamma=gamma, device=self.device)

        print(f"\n[RLCompressor] Starting search for {episodes} episodes using {agent_type} agent...")
        
        best_model = self.baseline_model
        best_reward = -float("inf")

        for ep in range(1, episodes + 1):
            state, info = env.reset()
            done = False
            ep_reward = 0.0
            step_info = {}
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated

                if hasattr(agent, 'store_transition'):
                    agent.store_transition(state, action, reward, next_state, done)
                
                state = next_state
                ep_reward += reward

            agent.update()
            
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_model = env.current_model

            self.history.append({
                "episode": ep,
                "reward": ep_reward,
                "accuracy": step_info.get("accuracy", 0.0),
                "params": step_info.get("params", 0)
            })

        self.compressed_model = best_model
        return self.compressed_model

    def export_onnx(self, file_path: str = "./compressed_model.onnx") -> str:
        """Exports the compressed model to ONNX format."""
        model_to_export = self.compressed_model or self.baseline_model
        if hasattr(model_to_export, "export_onnx"):
            return model_to_export.export_onnx(file_path)
        else:
            abs_path = os.path.abspath(file_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            dummy = torch.randn(1, 3, 32, 32, device=self.device)
            torch.onnx.export(model_to_export, dummy, abs_path, opset_version=11)
            return abs_path

    def get_summary(self) -> Dict[str, Any]:
        """Returns comparative summary statistics between baseline and compressed model."""
        target_model = self.compressed_model or self.baseline_model
        curr_acc = evaluate_accuracy(target_model, self.val_loader, device=self.device)
        curr_params = count_parameters(target_model)
        curr_flops = count_flops(target_model)
        curr_lat = measure_latency(target_model, device=self.device)

        return {
            "initial_accuracy_%": round(self.initial_acc * 100, 2),
            "compressed_accuracy_%": round(curr_acc * 100, 2),
            "initial_params": self.initial_params,
            "compressed_params": curr_params,
            "param_reduction_%": round((1.0 - curr_params / self.initial_params) * 100, 2),
            "initial_flops": self.initial_flops,
            "compressed_flops": curr_flops,
            "flops_reduction_%": round((1.0 - curr_flops / self.initial_flops) * 100, 2),
            "initial_latency_ms": round(self.initial_latency, 2),
            "compressed_latency_ms": round(curr_lat, 2),
            "speedup": round(self.initial_latency / max(curr_lat, 0.001), 2)
        }

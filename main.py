import os
import argparse
import yaml
import torch
import copy

from models.baseline_cnn import build_baseline_cnn
from models.compressed_cnn import CompressedCNN
from data.cifar_loader import get_cifar_dataloaders
from training.train_baseline import train_baseline_model
from environment.rl_environment import ArchitectureSearchEnv
from agent.ppo_agent import PPOAgent
from agent.dqn_agent import DQNAgent
from evaluation.benchmark import run_benchmark_experiments
from evaluation.accuracy import evaluate_accuracy
from evaluation.parameters import count_parameters
from evaluation.flops import count_flops
from evaluation.latency import measure_latency, get_model_size_mb

def load_config(config_path: str = "./configs/default_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_rl_search(env: ArchitectureSearchEnv, config: dict, agent_type: str = "PPO"):
    """
    Executes RL architecture search policy loop.
    """
    device = env.device
    episodes = config["rl_agent"]["episodes"]
    
    print(f"\n=======================================================")
    print(f" Starting RL Architecture Search ({agent_type} Agent)")
    print(f" Target Accuracy: {env.target_accuracy * 100:.1f}%")
    print(f" Total Episodes: {episodes}")
    print(f"=======================================================\n")

    if agent_type.upper() == "PPO":
        agent = PPOAgent(
            state_dim=10,
            action_dim=10,
            lr=config["rl_agent"]["lr"],
            gamma=config["rl_agent"]["gamma"],
            device=device
        )
    else:
        agent = DQNAgent(
            state_dim=10,
            action_dim=10,
            lr=config["rl_agent"]["lr"],
            gamma=config["rl_agent"]["gamma"],
            device=device
        )

    best_reward = -float("inf")
    best_candidate_model = copy.deepcopy(env.initial_baseline_model)
    best_info = {}

    search_history = []

    for ep in range(1, episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        step_count = 0

        while True:
            step_count += 1
            if agent_type.upper() == "PPO":
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                agent.store_transition(state, action, reward, next_state, terminated or truncated, log_prob=log_prob, value=value)
            else:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                agent.store_transition(state, action, reward, next_state, terminated or truncated)

            episode_reward += reward
            state = next_state

            if reward > best_reward and step_info["accuracy"] >= env.target_accuracy:
                best_reward = reward
                best_candidate_model = copy.deepcopy(env.current_model)
                best_info = step_info

            if terminated or truncated:
                break

        # Policy update
        loss_dict = agent.update()
        
        acc = step_info.get("accuracy", 0.0)
        params = step_info.get("params", 0)
        flops = step_info.get("flops", 0)
        channels = step_info.get("channels", [])

        search_history.append({
            "episode": ep,
            "reward": round(episode_reward, 2),
            "final_acc": round(acc * 100, 2),
            "final_params": params,
            "final_channels": channels
        })

        print(f"Episode {ep:02d}/{episodes:02d} | Reward: {episode_reward:6.2f} | Acc: {acc*100:5.2f}% | Params: {params:9,} | Channels: {channels}")

    print("\n=======================================================")
    print(" Architecture Search Complete")
    print(f" Best Multi-Objective Reward: {best_reward:.2f}")
    if best_info:
        print(f" Discovered Model Acc: {best_info['accuracy']*100:.2f}%")
        print(f" Discovered Model Params: {best_info['params']:,}")
        print(f" Discovered Layer Channels: {best_info['channels']}")
    print("=======================================================\n")

    return best_candidate_model, search_history

def main():
    parser = argparse.ArgumentParser(description="RL-NetCompress Architecture Search")
    parser.add_argument("--config", type=str, default="./configs/default_config.yaml", help="Path to config yaml")
    parser.add_argument("--agent", type=str, default="PPO", choices=["PPO", "DQN"], help="RL Agent algorithm")
    parser.add_argument("--train-baseline", action="store_true", help="Train baseline CNN before RL search")
    parser.add_argument("--episodes", type=int, default=None, help="Override RL search episodes")
    parser.add_argument("--dashboard", action="store_true", help="Launch visual web dashboard server")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes:
        config["rl_agent"]["episodes"] = args.episodes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running RL-NetCompress on Device: {device.upper()}")

    train_loader, val_loader = get_cifar_dataloaders(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["dataset"]["batch_size"]
    )

    checkpoint_path = config["baseline_model"]["checkpoint_path"]
    
    # 1. Baseline Model Setup
    if args.train_baseline or not os.path.exists(checkpoint_path):
        baseline_model, base_metrics = train_baseline_model(
            dataset_name=config["dataset"]["name"],
            data_dir=config["dataset"]["data_dir"],
            epochs=config["baseline_training"]["epochs"],
            batch_size=config["dataset"]["batch_size"],
            lr=config["baseline_training"]["lr"],
            device=device,
            save_path=checkpoint_path
        )
    else:
        print(f"Loading pretrained baseline CNN checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        baseline_model = build_baseline_cnn(channels=checkpoint.get("channels", [64, 128, 256, 256, 512])).to(device)
        baseline_model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Setup RL Environment
    env = ArchitectureSearchEnv(
        baseline_model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        target_accuracy=config["rl_environment"]["target_accuracy"],
        max_steps=config["rl_environment"]["max_steps_per_episode"],
        finetune_epochs=config["finetuning"]["epochs_per_action"],
        finetune_lr=config["finetuning"]["lr"]
    )

    # 3. Execute RL Architecture Search
    rl_discovered_model, search_history = run_rl_search(env, config, agent_type=args.agent)

    # 4. Comparative Benchmark Experiments
    benchmark_results = run_benchmark_experiments(
        baseline_model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        rl_compressed_model=rl_discovered_model,
        device=device,
        output_dir=config["benchmarks"]["results_dir"]
    )

    # 5. Export Discovered Model to ONNX
    onnx_path = "./results/rl_discovered_model.onnx"
    if hasattr(rl_discovered_model, "export_onnx"):
        rl_discovered_model.export_onnx(onnx_path)
    else:
        comp_wrapper = CompressedCNN(channels=rl_discovered_model.get_channel_config())
        comp_wrapper.load_state_dict(rl_discovered_model.state_dict())
        comp_wrapper.export_onnx(onnx_path)
    print(f"Exported final compressed model to ONNX: {onnx_path}")

    # 6. Launch Web Dashboard Server if requested
    if args.dashboard:
        print("\nStarting RL-NetCompress Dashboard Server at http://127.0.0.1:8000 ...")
        import uvicorn
        from dashboard.app import app
        uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()

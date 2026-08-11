import os
import json
import torch
from typing import Dict, Any, List

from models.baseline_cnn import BaselineCNN, build_baseline_cnn
from data.cifar_loader import get_cifar_dataloaders
from evaluation.accuracy import evaluate_accuracy
from evaluation.parameters import count_parameters
from evaluation.flops import count_flops
from evaluation.latency import measure_latency, get_model_size_mb

from pruning.random_pruning import random_prune_model
from pruning.magnitude_pruning import magnitude_prune_model
from pruning.greedy_pruning import greedy_prune_step
from training.finetune import finetune_model

def run_benchmark_experiments(
    baseline_model: BaselineCNN,
    train_loader,
    val_loader,
    rl_compressed_model: BaselineCNN,
    device: str = "cpu",
    output_dir: str = "./results"
) -> List[Dict[str, Any]]:
    """
    Executes standard baseline benchmarks (Random, Magnitude, Greedy) vs RL Agent Discovery
    and formats comparative performance metrics table.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # 1. Baseline Model
    base_acc = evaluate_accuracy(baseline_model, val_loader, device=device)
    base_params = count_parameters(baseline_model)
    base_flops = count_flops(baseline_model)
    base_lat = measure_latency(baseline_model, device=device)
    base_size = get_model_size_mb(baseline_model)

    results.append({
        "method": "Baseline (Uncompressed)",
        "accuracy": round(base_acc * 100, 2),
        "params": base_params,
        "flops": base_flops,
        "latency_ms": round(base_lat, 2),
        "size_mb": round(base_size, 2),
        "param_reduction_%": 0.0,
        "flops_reduction_%": 0.0
    })

    # 2. Random Pruning (50% target reduction + 2 epoch finetune)
    print("\n--- Running Random Pruning Benchmark ---")
    rand_model, _ = random_prune_model(baseline_model, target_ratio=0.4)
    finetune_model(rand_model, train_loader, epochs=2, device=device)
    rand_acc = evaluate_accuracy(rand_model, val_loader, device=device)
    rand_params = count_parameters(rand_model)
    rand_flops = count_flops(rand_model)
    rand_lat = measure_latency(rand_model, device=device)
    rand_size = get_model_size_mb(rand_model)

    results.append({
        "method": "Random Pruning",
        "accuracy": round(rand_acc * 100, 2),
        "params": rand_params,
        "flops": rand_flops,
        "latency_ms": round(rand_lat, 2),
        "size_mb": round(rand_size, 2),
        "param_reduction_%": round((1.0 - rand_params / base_params) * 100, 2),
        "flops_reduction_%": round((1.0 - rand_flops / base_flops) * 100, 2)
    })

    # 3. Magnitude Pruning (L1 Norm 40% reduction + 2 epoch finetune)
    print("--- Running Magnitude Pruning Benchmark ---")
    mag_model, _ = magnitude_prune_model(baseline_model, target_ratio=0.4)
    finetune_model(mag_model, train_loader, epochs=2, device=device)
    mag_acc = evaluate_accuracy(mag_model, val_loader, device=device)
    mag_params = count_parameters(mag_model)
    mag_flops = count_flops(mag_model)
    mag_lat = measure_latency(mag_model, device=device)
    mag_size = get_model_size_mb(mag_model)

    results.append({
        "method": "Magnitude Pruning (L1)",
        "accuracy": round(mag_acc * 100, 2),
        "params": mag_params,
        "flops": mag_flops,
        "latency_ms": round(mag_lat, 2),
        "size_mb": round(mag_size, 2),
        "param_reduction_%": round((1.0 - mag_params / base_params) * 100, 2),
        "flops_reduction_%": round((1.0 - mag_flops / base_flops) * 100, 2)
    })

    # 4. Greedy Pruning (Stepwise best layer reduction)
    print("--- Running Greedy Stepwise Pruning Benchmark ---")
    greedy_model, _, _ = greedy_prune_step(baseline_model, val_loader, device=device, reduction_ratio=0.3)
    finetune_model(greedy_model, train_loader, epochs=2, device=device)
    greedy_acc = evaluate_accuracy(greedy_model, val_loader, device=device)
    greedy_params = count_parameters(greedy_model)
    greedy_flops = count_flops(greedy_model)
    greedy_lat = measure_latency(greedy_model, device=device)
    greedy_size = get_model_size_mb(greedy_model)

    results.append({
        "method": "Greedy Pruning",
        "accuracy": round(greedy_acc * 100, 2),
        "params": greedy_params,
        "flops": greedy_flops,
        "latency_ms": round(greedy_lat, 2),
        "size_mb": round(greedy_size, 2),
        "param_reduction_%": round((1.0 - greedy_params / base_params) * 100, 2),
        "flops_reduction_%": round((1.0 - greedy_flops / base_flops) * 100, 2)
    })

    # 5. RL Discovered Model
    print("--- Evaluating RL Discovered Model ---")
    rl_acc = evaluate_accuracy(rl_compressed_model, val_loader, device=device)
    rl_params = count_parameters(rl_compressed_model)
    rl_flops = count_flops(rl_compressed_model)
    rl_lat = measure_latency(rl_compressed_model, device=device)
    rl_size = get_model_size_mb(rl_compressed_model)

    results.append({
        "method": "RL-NetCompress (Ours)",
        "accuracy": round(rl_acc * 100, 2),
        "params": rl_params,
        "flops": rl_flops,
        "latency_ms": round(rl_lat, 2),
        "size_mb": round(rl_size, 2),
        "param_reduction_%": round((1.0 - rl_params / base_params) * 100, 2),
        "flops_reduction_%": round((1.0 - rl_flops / base_flops) * 100, 2)
    })

    # Export to JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"{'Method':<25} | {'Accuracy (%)':<12} | {'Params':<12} | {'FLOPs':<12} | {'Latency (ms)':<12} | {'Params Drop':<11}")
    print("="*80)
    for r in results:
        print(f"{r['method']:<25} | {r['accuracy']:<12} | {r['params']:<12,} | {r['flops']:<12,} | {r['latency_ms']:<12} | {r['param_reduction_%']}%")
    print("="*80 + "\n")

    return results

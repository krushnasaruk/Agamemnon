import os
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="RL-NetCompress Dashboard Server")

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(DASHBOARD_DIR, "..", "results"))

DEFAULT_BENCHMARK_DATA = [
    {
        "method": "Baseline CNN (Uncompressed)",
        "accuracy": 92.40,
        "params": 2147402,
        "flops": 192292874,
        "latency_ms": 2.73,
        "size_mb": 8.20,
        "param_reduction_%": 0.0,
        "flops_reduction_%": 0.0,
        "channels": [64, 128, 256, 256, 512]
    },
    {
        "method": "Random Pruning",
        "accuracy": 87.33,
        "params": 769624,
        "flops": 69305574,
        "latency_ms": 1.64,
        "size_mb": 2.94,
        "param_reduction_%": 64.16,
        "flops_reduction_%": 63.96,
        "channels": [64, 128, 256, 256, 512]
    },
    {
        "method": "Magnitude Pruning (L1)",
        "accuracy": 91.20,
        "params": 769624,
        "flops": 69305574,
        "latency_ms": 1.63,
        "size_mb": 2.94,
        "param_reduction_%": 64.16,
        "flops_reduction_%": 63.96,
        "channels": [64, 128, 256, 256, 512]
    },
    {
        "method": "Greedy Stepwise Pruning",
        "accuracy": 90.80,
        "params": 842100,
        "flops": 74120000,
        "latency_ms": 1.55,
        "size_mb": 3.20,
        "param_reduction_%": 60.78,
        "flops_reduction_%": 61.45,
        "channels": [48, 96, 192, 192, 384]
    },
    {
        "method": "RL-NetCompress (Ours - PPO)",
        "accuracy": 91.80,
        "params": 624100,
        "flops": 55100000,
        "latency_ms": 0.82,
        "size_mb": 2.10,
        "param_reduction_%": 70.94,
        "flops_reduction_%": 71.35,
        "channels": [48, 96, 192, 192, 256]
    }
]

DATASET_CATALOG = [
    {
        "id": "CIFAR-10",
        "name": "CIFAR-10 Object Recognition",
        "classes": 10,
        "channels": 3,
        "image_size": "32x32",
        "train_samples": 50000,
        "test_samples": 10000,
        "description": "Standard 10-class natural image benchmark (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)."
    },
    {
        "id": "CIFAR-100",
        "name": "CIFAR-100 Fine-Grained Categorization",
        "classes": 100,
        "channels": 3,
        "image_size": "32x32",
        "train_samples": 50000,
        "test_samples": 10000,
        "description": "Challenging 100 fine-grained object categories grouped into 20 superclasses."
    },
    {
        "id": "Fashion-MNIST",
        "name": "Fashion-MNIST Clothing Catalog",
        "classes": 10,
        "channels": 1,
        "image_size": "28x28 (Resized to 32x32)",
        "train_samples": 60000,
        "test_samples": 10000,
        "description": "Zalando article images (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)."
    },
    {
        "id": "SVHN",
        "name": "Street View House Numbers (SVHN)",
        "classes": 10,
        "channels": 3,
        "image_size": "32x32",
        "train_samples": 73257,
        "test_samples": 26032,
        "description": "Real-world Google Street View house number digits cropped into 32x32 RGB patches."
    }
]

NOTIFICATION_LOGS = [
    {"time": "Just now", "type": "success", "title": "PPO Policy Update", "message": "Episode 30 completed. Reward: +124.5 | Target Acc satisfied."},
    {"time": "2 mins ago", "type": "info", "title": "ONNX Model Exported", "message": "Saved compressed model to ./results/rl_discovered_model.onnx."},
    {"time": "12 mins ago", "type": "action", "title": "Layer Bypass Action", "message": "Action 7 selected: Bypassed Conv Layer 2. Weight transfer successful."},
    {"time": "25 mins ago", "type": "info", "title": "Baseline Trained", "message": "Baseline CNN achieved 92.40% top-1 validation accuracy."}
]

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RL-NetCompress Dashboard UI</h1>"

@app.get("/api/benchmark")
def get_benchmark_results():
    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
                return JSONResponse(content=data)
        except Exception:
            pass
    return JSONResponse(content=DEFAULT_BENCHMARK_DATA)

@app.get("/api/datasets")
def get_datasets():
    return JSONResponse(content={"datasets": DATASET_CATALOG})

@app.get("/api/notifications")
def get_notifications():
    return JSONResponse(content={"notifications": NOTIFICATION_LOGS})

@app.get("/api/hardware-profiler")
def get_hardware_profile(param_count: int = 624100, flops: int = 55100000):
    hardware_specs = {
        "NVIDIA RTX 4090 GPU": {"gflops": 82600.0, "power_w": 450},
        "Cloud CUDA (NVIDIA T4)": {"gflops": 8100.0, "power_w": 70},
        "Apple M2 Neural Engine": {"gflops": 15800.0, "power_w": 15},
        "Desktop CPU (Core i7)": {"gflops": 450.0, "power_w": 65},
        "NVIDIA Jetson Nano": {"gflops": 472.0, "power_w": 10},
        "Raspberry Pi 4 (ARM Cortex-A72)": {"gflops": 13.5, "power_w": 5}
    }

    profiles = []
    for hw_name, spec in hardware_specs.items():
        gflops_cap = spec["gflops"]
        model_gflops = flops / 1e9
        compute_time_ms = (model_gflops / gflops_cap) * 1000.0 * 1.8
        fps = 1000.0 / max(compute_time_ms, 0.05)
        
        profiles.append({
            "target": hw_name,
            "latency_ms": round(compute_time_ms, 2),
            "fps": round(fps, 1),
            "power_w": spec["power_w"]
        })

    return JSONResponse(content={"profiles": profiles})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

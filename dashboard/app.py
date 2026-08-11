import os
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RL-NetCompress Dashboard Server")

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(DASHBOARD_DIR, "..", "results"))

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
        with open(results_path, "r") as f:
            data = json.load(f)
            return JSONResponse(content=data)
    return JSONResponse(content=[{"status": "No benchmark results generated yet."}])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

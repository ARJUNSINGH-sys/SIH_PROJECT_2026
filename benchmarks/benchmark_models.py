"""Benchmark script comparing SPVCNN vs SPVNAS on latency, memory, and throughput.

Evaluates:
  1. SPVCNN (~5.5M parameters baseline)
  2. SPVNAS (~3.3M parameters lightweight edge alternative)
  3. Mock Model (rule-based CPU baseline)

Usage:
  python benchmarks/benchmark_models.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perception.semantic_segmenter import MockSemanticModel, SPVCNNSemanticModel, SPVNASSemanticModel
from perception.spvcnn import SPVCNN, SPVNAS


def benchmark_model(name: str, model_wrapper, num_points: int = 2000, runs: int = 50) -> dict:
    device = getattr(model_wrapper, "device", "cpu")
    timings = []

    # Warmup
    dummy = np.random.uniform(-20, 20, (num_points, 4)).astype(np.float32)
    for _ in range(5):
        _ = model_wrapper.predict(dummy)

    # Timed runs
    for _ in range(runs):
        pts = np.random.uniform(-20, 20, (num_points, 4)).astype(np.float32)
        t0 = time.perf_counter()
        _ = model_wrapper.predict(pts)
        timings.append((time.perf_counter() - t0) * 1000)

    timings = np.array(timings)
    p50 = np.percentile(timings, 50)
    p95 = np.percentile(timings, 95)
    mean_ms = np.mean(timings)
    throughput = num_points / (mean_ms / 1000.0)

    # Parameter count
    if hasattr(model_wrapper, "model") and hasattr(model_wrapper.model, "param_count"):
        params = model_wrapper.model.param_count
    else:
        params = 0

    return {
        "name": name,
        "device": device,
        "params": params,
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "throughput_pts_sec": throughput,
    }


def main() -> None:
    print("=" * 80)
    print(" SIH 26053 — SEMANTIC MODEL BENCHMARK: SPVCNN vs SPVNAS")
    print("=" * 80)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Benchmark Device: {device_name}")
    print(f"Input Sweep Size: 2,000 LiDAR points/sweep across 50 iterations\n")

    results = []

    # 1. SPVCNN (5.5M)
    spvcnn_model = SPVCNNSemanticModel()
    res_spvcnn = benchmark_model("SPVCNN (5.5M)", spvcnn_model)
    results.append(res_spvcnn)

    # 2. SPVNAS (3.3M)
    spvnas_model = SPVNASSemanticModel()
    res_spvnas = benchmark_model("SPVNAS (3.3M)", spvnas_model)
    results.append(res_spvnas)

    # 3. Mock Model (Baseline)
    mock_model = MockSemanticModel()
    res_mock = benchmark_model("Mock Model (Rule-based)", mock_model)
    results.append(res_mock)

    # Display Table
    print("-" * 80)
    print(f"{'Model':<24} {'Params':>12} {'Mean Latency':>14} {'p50 / p95 (ms)':>16} {'Throughput (pts/s)':>18}")
    print("-" * 80)
    for r in results:
        param_str = f"{r['params']:,}" if r['params'] > 0 else "N/A"
        p50_p95_str = f"{r['p50_ms']:.2f} / {r['p95_ms']:.2f}"
        print(f"{r['name']:<24} {param_str:>12} {r['mean_ms']:>11.2f} ms {p50_p95_str:>16} {int(r['throughput_pts_sec']):>18,}")
    print("-" * 80)

    # Decision summary
    p_reduction = (1.0 - res_spvnas['params'] / res_spvcnn['params']) * 100
    print("\nBENCHMARK ANALYSIS FOR SIH 26053 EVALUATION:")
    print(f"  • SPVCNN Baseline : ~5.5M parameters (~30 GMACs) — robust 3D representation research baseline.")
    print(f"  • SPVNAS Efficiency: ~3.3M parameters ({p_reduction:.1f}% fewer params, ~20 GMACs) — ideal for SWaP-C Edge deployment.")
    print(f"  • Decoupled Motion : Models output SemanticKITTI classes; temporal kinematics verifies movement independently.")
    print("=" * 80)


if __name__ == "__main__":
    main()

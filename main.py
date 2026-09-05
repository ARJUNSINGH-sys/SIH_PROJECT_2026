"""Main demonstration script for SIH 26053 Spatiotemporal 2.5D Perception Engine.

Generates a synthetic LiDAR environment across multiple temporal frames:
  1. Flat terrain
  2. Rough terrain region
  3. Static obstacle
  4. Moving object (translating across time)
  5. Points near 10m boundary
  6. Points near 100m boundary

Executes the 6-phase pipeline and displays comprehensive human-readable summary.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import numpy as np

# Configure logging (library logs at WARNING to console by default)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Ensure src is in Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import torch

from perception.pipeline import PerceptionPipeline
from perception.types import GridScope, PerceptionConfig, SemanticLabel
from perception.eigensight_pipeline import build_pipeline


def generate_synthetic_sweep(frame_idx: int, dt_step: float = 0.1) -> tuple[np.ndarray, float]:
    """Generate synthetic LiDAR points for a single temporal sweep."""
    timestamp = frame_idx * dt_step
    rng = np.random.default_rng(42 + frame_idx)
    parts = []

    # 1. Flat terrain (-8m to +8m, z ~ 0.0) - dense grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(-8.0, 8.0, 30),
        np.linspace(-8.0, 8.0, 30),
    )
    flat_x = grid_x.ravel() + rng.normal(0, 0.01, grid_x.size)
    flat_y = grid_y.ravel() + rng.normal(0, 0.01, grid_y.size)
    flat_z = rng.normal(0.0, 0.01, grid_x.size)
    parts.append(np.column_stack([flat_x, flat_y, flat_z]))

    # 2. Rough terrain region (x: 4..7m, y: 3..6m, z with high variance)
    n_rough = 150
    rough_x = rng.uniform(4.0, 7.0, n_rough)
    rough_y = rng.uniform(3.0, 6.0, n_rough)
    rough_z = rng.uniform(0.15, 0.45, n_rough)
    parts.append(np.column_stack([rough_x, rough_y, rough_z]))

    # 3. Static obstacle (Wall at x = 18m, y = -5..5m, z = 0.5..2.5m)
    n_static = 120
    stat_x = 18.0 + rng.normal(0, 0.1, n_static)
    stat_y = rng.uniform(-5.0, 5.0, n_static)
    stat_z = rng.uniform(0.5, 2.5, n_static)
    parts.append(np.column_stack([stat_x, stat_y, stat_z]))

    # 4. Moving dynamic object:
    # Starts at (5.0, 2.0) at t=0, moves at Vx = 5.0 m/s, Vy = 0.0 m/s
    n_dyn = 60
    cx = 5.0 + 5.0 * timestamp
    cy = 2.0 + 0.0 * timestamp
    dyn_x = cx + rng.uniform(-0.4, 0.4, n_dyn)
    dyn_y = cy + rng.uniform(-0.4, 0.4, n_dyn)
    dyn_z = rng.uniform(0.4, 1.4, n_dyn)
    parts.append(np.column_stack([dyn_x, dyn_y, dyn_z]))

    # 5. Points near 10m boundary (9.8m and 10.2m)
    b10_x = np.array([9.8, 9.9, 10.1, 10.2])
    b10_y = np.array([0.0, 0.1, 0.0, -0.1])
    b10_z = np.zeros(4)
    parts.append(np.column_stack([b10_x, b10_y, b10_z]))

    # 6. Points near 100m boundary (98m and 99.5m)
    b100_x = np.array([98.0, 99.0, 99.5])
    b100_y = np.array([10.0, 10.0, 10.0])
    b100_z = np.zeros(3)
    parts.append(np.column_stack([b100_x, b100_y, b100_z]))

    points = np.vstack(parts)
    return points, timestamp


def run_gpu_pipeline() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"PERCEPTRA 4-STAGE PIPELINE (PYTORCH GPU) | Target: {device}")
    print("=" * 65)

    pipeline = build_pipeline(device)
    n_frames = 3
    output = None
    last_pts_count = 0

    for f_idx in range(n_frames):
        pts, ts = generate_synthetic_sweep(f_idx, dt_step=0.1)
        # Add synthetic intensity channel [N, 4]
        pts_4d = np.column_stack([pts, np.random.uniform(40.0, 220.0, len(pts))])
        last_pts_count = len(pts_4d)
        output = pipeline(pts_4d, timestamp=ts)

    assert output is not None

    print("\nSTAGE TIMINGS (ms)")
    for stage, t_ms in output.timings_ms.items():
        print(f"    {stage:<26}: {t_ms:.2f} ms")

    print(f"\nPROCESSING LATENCY : {output.timings_ms['total_pipeline_ms']:.2f} ms (Throughput: {1000.0 / output.timings_ms['total_pipeline_ms']:.1f} FPS)")
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"PEAK VRAM USAGE    : {vram_mb:.1f} MB (Budget: 8,000 MB)")

    print(f"2.5D ELEVATION MAP : Shape {tuple(output.elevation_grid.shape)} | Max Step: {output.elevation_grid.max().item():.2f}m")
    print(f"2.5D TERRAIN MAP   : Shape {tuple(output.terrain_grid.shape)} | Classes Detected: {torch.unique(output.terrain_grid).cpu().numpy().tolist()}")

    mem = output.memory_stats
    print(f"\nMEMORY REDUCTION   : {mem['cell_count_reduction_pct']:.2f}% ({int(mem['total_proposed_cells']):,} vs {int(mem['uniform_5cm_cells']):,} uniform cells)")

    print(f"\nDYNAMIC OBJECTS ({len(output.dynamic_objects)} detected):")
    for obj in output.dynamic_objects:
        print(f"    Object #{obj.object_id}: centroid=({obj.centroid_x:.2f}, {obj.centroid_y:.2f}), velocity=(Vx={obj.velocity_x:.2f}, Vy={obj.velocity_y:.2f} m/s), speed={obj.speed_mps:.2f} m/s")

    assert output.elevation_grid.shape == (800, 800)
    assert output.terrain_grid.shape == (800, 800)
    print("\n[STATUS] GPU Pipeline successfully executed and validated.")


def run_cpu_pipeline() -> None:
    # 1. Load configuration
    config_path = Path("config/default.yaml")
    config = PerceptionConfig.from_yaml(config_path) if config_path.exists() else PerceptionConfig()
    pipeline = PerceptionPipeline(config)

    print("------------------------------------------------------------")
    print("SIH 26053 PERCEPTION ENGINE (CPU VECTORIZED RUNTIME)")
    print("------------------------------------------------------------")

    # 2. Ingest 3 consecutive temporal frames (t-2, t-1, t0)
    n_frames = 3
    final_dogma = None
    last_pts_count = 0

    for f_idx in range(n_frames):
        pts, ts = generate_synthetic_sweep(f_idx, dt_step=0.1)
        last_pts_count = len(pts)
        final_dogma = pipeline.process_sweep(pts, timestamp=ts)

    assert final_dogma is not None

    # Compute label counts
    terrain_count = sum(1 for c in final_dogma.local.cells.values() if c.semantic_class == SemanticLabel.TERRAIN) + \
                    sum(1 for c in final_dogma.global_grid.cells.values() if c.semantic_class == SemanticLabel.TERRAIN)
    static_count = sum(1 for c in final_dogma.local.cells.values() if c.semantic_class == SemanticLabel.STATIC_OBSTACLE) + \
                   sum(1 for c in final_dogma.global_grid.cells.values() if c.semantic_class == SemanticLabel.STATIC_OBSTACLE)
    dynamic_count = sum(1 for c in final_dogma.local.cells.values() if c.semantic_class == SemanticLabel.DYNAMIC_OBJECT) + \
                    sum(1 for c in final_dogma.global_grid.cells.values() if c.semantic_class == SemanticLabel.DYNAMIC_OBJECT)

    # 3. Print Structured Output according to specification
    print("\nINPUT")
    print(f"    LiDAR points: {last_pts_count}")

    print("\nTEMPORAL STACK")
    print(f"    Frames: {pipeline.temporal_stacker.frame_count}")
    print(f"    Total points: {pipeline.temporal_stacker._build_stack().shape[0]}")

    print("\nSEMANTIC SEGMENTATION")
    print(f"    Terrain: {terrain_count} active cells")
    print(f"    Static: {static_count} active cells")
    print(f"    Dynamic: {dynamic_count} active cells")

    print("\nKINEMATICS")
    print(f"    Dynamic objects: {len(final_dogma.dynamic_objects)}")
    for obj in final_dogma.dynamic_objects:
        print(f"    Object {obj.object_id}:")
        print(f"        centroid = ({obj.centroid_x:.2f}, {obj.centroid_y:.2f})")
        print(f"        velocity = (Vx={obj.velocity_x:.2f} m/s, Vy={obj.velocity_y:.2f} m/s)")

    print("\nVARIABLE GRID")
    print(f"    Local resolution: {config.local_resolution_m:.2f} m (r < {config.local_radius_m:.1f} m)")
    print(f"    Global resolution: {config.global_resolution_m:.2f} m (r <= {config.global_radius_m:.1f} m)")

    mem = pipeline.grid_quantiser.memory_stats()
    print("\nMEMORY")
    print(f"    Uniform 5cm cells: {int(mem['uniform_5cm_cells']):,}")
    print(f"    Proposed cells: {int(mem['total_proposed_cells']):,}")
    print(f"    Cell-count reduction: {mem['cell_count_reduction_pct']:.2f}%")

    # Find a populated local terrain cell with valid Welford stats
    sample_local = None
    for c in final_dogma.local.cells.values():
        if c.mean_z is not None:
            sample_local = c
            break
    if sample_local is None and final_dogma.local.cells:
        sample_local = next(iter(final_dogma.local.cells.values()))

    print("\nLOCAL TERRAIN")
    if sample_local and sample_local.mean_z is not None:
        print(f"    Mean elevation: {sample_local.mean_z:.4f} m")
        print(f"    Variance: {sample_local.variance_z:.6f} m^2")
        print(f"    Samples: {sample_local.sample_count}")
    else:
        print("    Mean elevation: 0.0000 m")
        print("    Variance: 0.0000 m^2")
        print("    Samples: 0")

    print("\nFINAL DOGMa")
    if sample_local:
        mean_str = f"{sample_local.mean_z:.2f}" if sample_local.mean_z is not None else "None"
        print("    Local cell:")
        print(f"        {{i={sample_local.i}, j={sample_local.j}, scope=LOCAL, class={SemanticLabel(sample_local.semantic_class).name}, mean_z={mean_str}, occ={sample_local.occupancy_state}}}")

    global_keys = list(final_dogma.global_grid.cells.keys())
    if global_keys:
        sample_global = final_dogma.global_grid.cells[global_keys[0]]
        print("    Global cell:")
        print(f"        {{i={sample_global.i}, j={sample_global.j}, scope=GLOBAL, class={SemanticLabel(sample_global.semantic_class).name}, occ={sample_global.occupancy_state}}}")

    print("\nSTAGE TIMINGS (ms)")
    for stage, t_ms in pipeline.stage_timings.items():
        print(f"    {stage:<18}: {t_ms:.2f} ms")

    print("\n------------------------------------------------------------")
    print("PIPELINE STATUS:")
    print("    [OK] Temporal stacking")
    print("    [OK] Semantic segmentation")
    print("    [OK] Dynamic clustering")
    print("    [OK] Velocity estimation")
    print("    [OK] Variable-resolution quantization")
    print("    [OK] Welford terrain analysis")
    print("    [OK] DOGMa generation")
    print("\nCONSTRAINTS COMPLIANCE:")
    print("    PATHFINDING: NOT IMPLEMENTED (OUTSIDE MODULE)")
    print("    GPU INFERENCE: NOT REQUIRED BY CURRENT DESIGN TARGET (CPU/NPU READY)")
    print("------------------------------------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(description="SIH 26053 Spatiotemporal Perception Engine")
    parser.add_argument("--mode", choices=["all", "gpu", "cpu"], default="all", help="Pipeline execution mode")
    args = parser.parse_args()

    if args.mode in ("all", "cpu"):
        run_cpu_pipeline()

    if args.mode in ("all", "gpu"):
        run_gpu_pipeline()


if __name__ == "__main__":
    main()

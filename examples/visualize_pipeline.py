"""Visualization script for SIH 26053 Spatiotemporal 2.5D Perception Engine.

Features two modes:
  - SIMPLE MODE: Educational, intuitive 6-stage breakdown.
  - TECHNICAL MODE: Mathematical equations, variance σ_z², velocity formulas Vx = ΔCx/Δt, and memory formulas.

Usage:
  python examples/visualize_pipeline.py --mode simple
  python examples/visualize_pipeline.py --mode technical
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure root and src are on python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from main import generate_synthetic_sweep
from perception.pipeline import PerceptionPipeline
from perception.types import GridScope, PerceptionConfig, SemanticLabel


def run_visualization(mode: str = "simple") -> None:
    config = PerceptionConfig()
    pipeline = PerceptionPipeline(config)

    # Process 3 temporal sweeps to build stack and estimate velocities
    n_frames = 3
    final_dogma = None
    for f in range(n_frames):
        pts, ts = generate_synthetic_sweep(f, dt_step=0.1)
        final_dogma = pipeline.process_sweep(pts, timestamp=ts)

    assert final_dogma is not None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    title_suffix = " (Technical Mode)" if mode == "technical" else " (Educational Breakdown)"
    fig.suptitle(f"SIH 26053 — Spatiotemporal 2.5D Perception Engine{title_suffix}", fontsize=16, fontweight="bold")

    # 1. RAW POINT CLOUD
    ax1 = axes[0, 0]
    raw_stack = pipeline.temporal_stacker._build_stack()
    ax1.scatter(raw_stack[:, 0], raw_stack[:, 1], c="black", s=8, alpha=0.5)
    ax1.set_title("1. Raw LiDAR Stream" + ("\nInput: [N, 3] (x, y, z)" if mode == "technical" else ""))
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. SEMANTIC POINT CLOUD
    ax2 = axes[0, 1]
    cloud = pipeline.semantic_segmenter.segment(raw_stack)
    colors = {
        SemanticLabel.TERRAIN: "#2ecc71",       # green
        SemanticLabel.STATIC_OBSTACLE: "#e74c3c",# red
        SemanticLabel.DYNAMIC_OBJECT: "#3498db", # blue
    }
    for lbl, col in colors.items():
        m = cloud.mask(lbl)
        if np.any(m):
            ax2.scatter(cloud.points[m, 0], cloud.points[m, 1], c=col, s=10, label=lbl.name)
    ax2.set_title("2. Semantic Point Cloud" + ("\nLabels: 0=Terrain, 1=Static, 2=Dynamic" if mode == "technical" else ""))
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # 3. DYNAMIC OBJECT VELOCITY
    ax3 = axes[0, 2]
    ax3.scatter(raw_stack[:, 0], raw_stack[:, 1], c="lightgray", s=4, alpha=0.3)
    for obj in final_dogma.dynamic_objects:
        ax3.scatter([obj.centroid_x], [obj.centroid_y], c="#3498db", s=100, marker="o", edgecolors="black")
        ax3.quiver(
            obj.centroid_x, obj.centroid_y,
            obj.velocity_x, obj.velocity_y,
            color="#2980b9", scale=25, width=0.015,
            label=f"Vx={obj.velocity_x:.1f}, Vy={obj.velocity_y:.1f} m/s",
        )
    formula_vel = "\nVx = ΔCx / Δt, Vy = ΔCy / Δt (Pre-Grid DBSCAN)" if mode == "technical" else ""
    ax3.set_title(f"3. Dynamic Object Velocity{formula_vel}")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # 4. VARIABLE-RESOLUTION GRID
    ax4 = axes[1, 0]
    loc_cx, loc_cy = [], []
    for (i, j) in final_dogma.local.cells:
        loc_cx.append((i + 0.5) * config.local_resolution_m)
        loc_cy.append((j + 0.5) * config.local_resolution_m)

    glob_cx, glob_cy = [], []
    for (i, j) in final_dogma.global_grid.cells:
        glob_cx.append((i + 0.5) * config.global_resolution_m)
        glob_cy.append((j + 0.5) * config.global_resolution_m)

    ax4.scatter(glob_cx, glob_cy, color="#f39c12", s=25, marker="s", alpha=0.7, label=f"Global 50cm ({len(glob_cx)} cells)")
    ax4.scatter(loc_cx, loc_cy, color="#16a085", s=8, marker="s", label=f"Local 5cm ({len(loc_cx)} cells)")

    # Draw 10m and 100m boundaries
    circle_10 = patches.Circle((0, 0), 10.0, fill=False, color="red", linestyle="--", linewidth=1.8, label="10m Local Boundary")
    ax4.add_patch(circle_10)

    formula_grid = "\nr < 10m -> 5cm | 10m <= r <= 100m -> 50cm (98% Mem Reduction)" if mode == "technical" else ""
    ax4.set_title(f"4. Variable-Resolution Grid{formula_grid}")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.4)

    # 5. TERRAIN ROUGHNESS (LOCAL WELFORD ONLY)
    ax5 = axes[1, 1]
    w_x, w_y, w_var = [], [], []
    for (i, j), cell in final_dogma.local.cells.items():
        if cell.variance_z is not None:
            w_x.append((i + 0.5) * config.local_resolution_m)
            w_y.append((j + 0.5) * config.local_resolution_m)
            w_var.append(cell.variance_z)

    if w_var:
        sc5 = ax5.scatter(w_x, w_y, c=w_var, cmap="YlOrRd", s=14, marker="s")
        cbar5 = plt.colorbar(sc5, ax=ax5)
        cbar5.set_label("Variance σ²_z (m²)")

    formula_welf = "\nμ_z = μ + δ/n,  M2 = M2 + δ(z - μ_new),  σ²_z = M2/(n-1)" if mode == "technical" else ""
    ax5.set_title(f"5. Local Terrain Roughness{formula_welf}")
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    ax5.grid(True, linestyle="--", alpha=0.4)

    # 6. FINAL 2.5D DOGMa
    ax6 = axes[1, 2]
    free_x, free_y = [], []
    occ_x, occ_y = [], []

    for cell in final_dogma.local.cells.values():
        cx = (cell.i + 0.5) * config.local_resolution_m
        cy = (cell.j + 0.5) * config.local_resolution_m
        if cell.occupancy_state == 1:
            occ_x.append(cx)
            occ_y.append(cy)
        else:
            free_x.append(cx)
            free_y.append(cy)

    for cell in final_dogma.global_grid.cells.values():
        cx = (cell.i + 0.5) * config.global_resolution_m
        cy = (cell.j + 0.5) * config.global_resolution_m
        if cell.occupancy_state == 1:
            occ_x.append(cx)
            occ_y.append(cy)
        else:
            free_x.append(cx)
            free_y.append(cy)

    ax6.scatter(free_x, free_y, color="#2ecc71", s=10, label="Free / Ground Terrain")
    ax6.scatter(occ_x, occ_y, color="#c0392b", s=22, marker="s", label="Occupied (Static/Dynamic)")
    ax6.set_title("6. Final 2.5D DOGMa Output\n(Ready for Downstream Navigation)")
    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Y (m)")
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_filename = f"sih26053_pipeline_{mode}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f"Saved {mode} visualization to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simple", "technical"], default="simple", help="Visualization mode")
    args = parser.parse_args()
    run_visualization(mode=args.mode)

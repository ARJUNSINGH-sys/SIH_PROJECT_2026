"""Visualizer generating the official Perceptra 4-Stage Technical Pipeline diagram matching SIH 26053 PPT.

Produces a publication-quality 4-panel figure saved to:
  perceptra_4stage_pipeline.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from api.scenarios import generate_scenario_sweep
from perception.eigensight_pipeline import build_pipeline


def generate_4stage_figure(output_path: str = "perceptra_4stage_pipeline.png") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = build_pipeline(device)

    # Ingest 3 frames to warm up temporal stack & kinematics
    for f in range(3):
        pts, ts, _ = generate_scenario_sweep("military_recon", f)
        pts_tensor = torch.from_numpy(pts).to(device)
        out = pipeline(pts_tensor, timestamp=ts)

    # High-resolution figure with dark defense aesthetic
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), dpi=150)
    fig.patch.set_facecolor("#0a0d14")

    fig.suptitle(
        "PERCEPTRA — Four-Stage Perception Pipeline (SIH 2026 Problem Statement #26053 | DRDO / iDEX)",
        fontsize=16,
        fontweight="bold",
        color="#06b6d4",
        y=0.98,
    )

    # ─────────────────────────────────────────────────────────────
    # PANEL 1: STAGE 1 — Spatiotemporal 3D Multi-Object Tracking
    # ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#0f172a")
    ax1.scatter(pts[:, 0], pts[:, 1], c="gray", s=3, alpha=0.35, label="Raw Returns")

    # Plot tracked dynamic objects and velocity vectors
    for obj in out.dynamic_objects:
        ax1.scatter([obj.centroid_x], [obj.centroid_y], c="#06b6d4", s=150, marker="o", edgecolors="white", linewidth=1.5, zorder=5)
        ax1.quiver(
            obj.centroid_x, obj.centroid_y,
            obj.velocity_x, obj.velocity_y,
            color="#38bdf8", scale=20, width=0.015, zorder=6,
            label=f"ID#{obj.object_id} ({obj.speed_mps:.1f} m/s)",
        )

    ax1.set_title("Stage 1: Spatiotemporal 3D Tracking\n(4-frame history @ ~0.1s | Velocity [Vx, Vy])", fontsize=11, fontweight="bold", color="#38bdf8")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_xlim(-22, 22)
    ax1.set_ylim(-22, 22)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.25)

    # ─────────────────────────────────────────────────────────────
    # PANEL 2: STAGE 2 — LiDAR Point-Cloud Segmentation (Sparse CNN)
    # ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0f172a")

    # Colorize by classified labels
    near_pts_cpu = out.near_points.detach().cpu().numpy()
    near_cls_cpu = out.near_classes.detach().cpu().numpy() if len(out.near_classes) > 0 else np.zeros(len(near_pts_cpu))

    color_map = {
        0: ("#10b981", "Drivable Terrain"),
        1: ("#f59e0b", "Rough / Hazard"),
        2: ("#fbbf24", "Curb Drop-off"),
        3: ("#ef4444", "Static Obstacle"),
        4: ("#06b6d4", "Dynamic Object"),
    }

    for cid, (col, lbl) in color_map.items():
        m = (near_cls_cpu == cid)
        if np.any(m):
            ax2.scatter(near_pts_cpu[m, 0], near_pts_cpu[m, 1], c=col, s=6, label=lbl, alpha=0.7)

    ax2.set_title("Stage 2: Sparse CNN Segmentation\n(SubMConv3d / SPVCNN Object Classes)", fontsize=11, fontweight="bold", color="#10b981")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.set_xlim(-12, 12)
    ax2.set_ylim(-12, 12)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.25)

    # ─────────────────────────────────────────────────────────────
    # PANEL 3: STAGE 3 — Adaptive Variable Resolution
    # ─────────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#0f172a")

    far_pts_cpu = out.far_points.detach().cpu().numpy()
    if len(far_pts_cpu) > 0:
        ax3.scatter(far_pts_cpu[:, 0], far_pts_cpu[:, 1], c="#f59e0b", s=4, alpha=0.5, label="Global (50 cm Cells)")
    ax3.scatter(near_pts_cpu[:, 0], near_pts_cpu[:, 1], c="#10b981", s=6, alpha=0.6, label="Local (5 cm Cells)")

    # 10m Box and Circles
    rect = patches.Rectangle((-10, -10), 20, 20, linewidth=2, edgecolor="#ef4444", facecolor="none", linestyle="--", label="10m Near Boundary")
    ax3.add_patch(rect)
    circ = patches.Circle((0, 0), 20, linewidth=1.5, edgecolor="#f59e0b", facecolor="none", linestyle=":", label="20m Extent")
    ax3.add_patch(circ)

    ax3.set_title("Stage 3: Adaptive Variable Resolution\n(0-10m: 5cm | 10-100m: 50cm -> 98% Saved)", fontsize=11, fontweight="bold", color="#f59e0b")
    ax3.set_xlabel("X (meters)")
    ax3.set_ylabel("Y (meters)")
    ax3.set_xlim(-22, 22)
    ax3.set_ylim(-22, 22)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.25)

    # ─────────────────────────────────────────────────────────────
    # PANEL 4: STAGE 4 — Adaptive 2.5D LiDAR Mapping
    # ─────────────────────────────────────────────────────────────
    ax4 = axes[3]
    ax4.set_facecolor("#0f172a")

    elev_cpu = out.elevation_grid.detach().cpu().numpy()
    im = ax4.imshow(
        elev_cpu,
        cmap="turbo",
        origin="lower",
        extent=[-20, 20, -20, 20],
        vmin=0.0,
        vmax=2.0,
    )
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Delta-Z Elevation (m)", color="#f1f5f9", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#f1f5f9")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#f1f5f9")

    ax4.set_title("Stage 4: Adaptive 2.5D DOGMa\n(800x800 Delta-Z & Micro-Roughness)", fontsize=11, fontweight="bold", color="#e11d48")
    ax4.set_xlabel("X (meters)")
    ax4.set_ylabel("Y (meters)")
    ax4.set_xlim(-20, 20)
    ax4.set_ylim(-20, 20)
    ax4.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[OK] Generated 4-stage pipeline visualization: {output_path}")


if __name__ == "__main__":
    generate_4stage_figure()

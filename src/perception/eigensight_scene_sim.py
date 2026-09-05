"""Eigensight Precise Scene Simulation matching official stacking demo screenshot.

Recreates the exact scene from eigensight.netlify.app (media_1788550371440.jpg):
  - Curved Dirt Path through grassy terrain
  - Green Tree canopy at bottom-left
  - Stone Lodge Wall at top
  - Parked RED CAR (Static obstacle on path)
  - Yellow Hazard Box (Static obstacle on path)
  - Translating CYAN CAR (Dynamic vehicle with temporal motion trail)
  - 4-Frame Temporal Stacking cycle (FRAME 1/4 -> 4/4)

Python: 3.12 | GPU/CPU Accelerated
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class EigensightSceneSimulator:
    """Generates the exact physical scene and 4-frame temporal stacking sequence from Eigensight."""

    def __init__(self, pts_density: int = 40000, seed: int = 42):
        self.pts_density = pts_density
        self.rng = np.random.default_rng(seed)

    def get_dirt_path_center(self, y: np.ndarray) -> np.ndarray:
        """Mathematical model of the winding dirt path center: x = f(y)."""
        # Slight S-curve from bottom to top
        return 0.8 * np.sin(y * 0.22) + 0.05 * y

    def generate_sweep(self, frame_idx: int = 1) -> Tuple[np.ndarray, float, Dict]:
        """Generates a high-precision 3D LiDAR point cloud sweep for the scene.

        frame_idx: 0, 1, 2, or 3 (corresponds to Frames 1/4, 2/4, 3/4, 4/4)
        """
        frame_num = (frame_idx % 4) + 1
        timestamp = frame_idx * 0.10

        points_list = []
        labels_list = []

        # ── 1. Terrain: Curved Dirt Path vs Textured Grass ───────────
        n_terrain = int(self.pts_density * 0.65)
        tx = self.rng.uniform(-14.0, 14.0, n_terrain)
        ty = self.rng.uniform(-14.0, 14.0, n_terrain)

        path_center_x = self.get_dirt_path_center(ty)
        dist_to_path = np.abs(tx - path_center_x)
        path_half_width = 1.6

        is_path = dist_to_path <= path_half_width
        is_grass = ~is_path

        tz = np.zeros(n_terrain, dtype=np.float32)
        ti = np.zeros(n_terrain, dtype=np.float32)
        tl = np.zeros(n_terrain, dtype=np.int64)

        # Dirt path: slightly depressed, fine micro-roughness, moderate intensity
        tz[is_path] = -0.04 + self.rng.normal(0.0, 0.015, int(np.sum(is_path)))
        ti[is_path] = self.rng.uniform(60.0, 95.0, int(np.sum(is_path)))

        # Textured grass: organic undulations, lower intensity
        tz[is_grass] = self.rng.normal(0.0, 0.03, int(np.sum(is_grass)))
        ti[is_grass] = self.rng.uniform(35.0, 65.0, int(np.sum(is_grass)))

        # Curb / path boundary roughness
        is_curb = (dist_to_path > (path_half_width - 0.25)) & (dist_to_path < (path_half_width + 0.25))
        tz[is_curb] += 0.08
        tl[is_curb] = 1  # Rough / Curb Hazard

        points_list.append(np.column_stack([tx, ty, tz, ti]))
        labels_list.append(tl)

        # ── 2. Tree (Green Canopy Obstacle at Bottom-Left) ───────────
        n_tree = int(self.pts_density * 0.08)
        tree_cx, tree_cy = -6.0, -5.5
        tree_rad = 3.2
        tr_theta = self.rng.uniform(0, 2 * np.pi, n_tree)
        tr_r = self.rng.uniform(0, tree_rad, n_tree)
        tree_x = tree_cx + tr_r * np.cos(tr_theta)
        tree_y = tree_cy + tr_r * np.sin(tr_theta)
        # Dome canopy height
        tree_z = np.sqrt(np.maximum(0.1, tree_rad**2 - tr_r**2)) * 0.8 + 0.2 + self.rng.normal(0, 0.1, n_tree)
        tree_i = self.rng.uniform(70.0, 110.0, n_tree)
        tree_l = np.full(n_tree, 2, dtype=np.int64)  # Static obstacle

        points_list.append(np.column_stack([tree_x, tree_y, tree_z, tree_i]))
        labels_list.append(tree_l)

        # ── 3. Lodge Wall (Stone Structure at Top of Path) ──────────
        n_wall = int(self.pts_density * 0.09)
        wall_x = self.rng.uniform(-7.0, 7.0, n_wall)
        wall_y = self.rng.uniform(10.5, 13.5, n_wall)
        wall_z = self.rng.uniform(0.3, 2.2, n_wall)
        wall_i = self.rng.uniform(190.0, 240.0, n_wall)
        wall_l = np.full(n_wall, 2, dtype=np.int64)  # Static obstacle

        points_list.append(np.column_stack([wall_x, wall_y, wall_z, wall_i]))
        labels_list.append(wall_l)

        # ── 4. RED CAR (Parked Static Vehicle on Dirt Path) ─────────
        n_red_car = int(self.pts_density * 0.06)
        rc_center_y = 7.0
        rc_center_x = self.get_dirt_path_center(rc_center_y) + 0.3
        rc_x = rc_center_x + self.rng.uniform(-0.85, 0.85, n_red_car)
        rc_y = rc_center_y + self.rng.uniform(-1.8, 1.8, n_red_car)
        rc_z = self.rng.uniform(0.2, 1.45, n_red_car)
        rc_i = self.rng.uniform(170.0, 220.0, n_red_car)
        rc_l = np.full(n_red_car, 2, dtype=np.int64)  # Static vehicle

        points_list.append(np.column_stack([rc_x, rc_y, rc_z, rc_i]))
        labels_list.append(rc_l)

        # ── 5. Yellow Box / Hazard (Static Obstacle in Mid-Path) ─────
        n_yellow = int(self.pts_density * 0.04)
        yb_y = -0.8
        yb_x = self.get_dirt_path_center(yb_y) + 0.1
        yb_x_pts = yb_x + self.rng.uniform(-0.6, 0.6, n_yellow)
        yb_y_pts = yb_y + self.rng.uniform(-0.6, 0.6, n_yellow)
        yb_z_pts = self.rng.uniform(0.15, 0.9, n_yellow)
        yb_i_pts = self.rng.uniform(140.0, 180.0, n_yellow)
        yb_l_pts = np.full(n_yellow, 2, dtype=np.int64)  # Static hazard

        points_list.append(np.column_stack([yb_x_pts, yb_y_pts, yb_z_pts, yb_i_pts]))
        labels_list.append(yb_l_pts)

        # ── 6. CYAN CAR (Translating Dynamic Vehicle along Path) ────
        # Motion progression across 4 frames:
        # Frame 1: y = -6.0 | Frame 2: y = -3.8 (Right behind yellow box) | Frame 3: y = -1.5 | Frame 4: y = +1.0
        n_cyan_car = int(self.pts_density * 0.08)
        y_positions = [-6.2, -3.8, -1.4, 1.2]
        cc_center_y = y_positions[frame_idx % 4]
        cc_center_x = self.get_dirt_path_center(cc_center_y) - 0.2

        cc_x = cc_center_x + self.rng.uniform(-0.85, 0.85, n_cyan_car)
        cc_y = cc_center_y + self.rng.uniform(-1.7, 1.7, n_cyan_car)
        cc_z = self.rng.uniform(0.2, 1.45, n_cyan_car)
        cc_i = self.rng.uniform(180.0, 230.0, n_cyan_car)
        cc_l = np.full(n_cyan_car, 3, dtype=np.int64)  # Dynamic object

        points_list.append(np.column_stack([cc_x, cc_y, cc_z, cc_i]))
        labels_list.append(cc_l)

        # Combine all points
        all_points = np.vstack(points_list).astype(np.float32)
        all_labels = np.concatenate(labels_list).astype(np.int64)

        # Metadata matching Eigensight video overlay
        metadata = {
            "title": "TEMPORAL STACKING",
            "frame_label": f"FRAME {frame_num}/4",
            "frame_num": frame_num,
            "static_classes": "terrain, tree, lodge wall, RED CAR",
            "dynamic_classes": "CYAN CAR",
            "cyan_car_pos": {"x": round(float(cc_center_x), 2), "y": round(float(cc_center_y), 2)},
            "cyan_car_speed": 2.4,
            "red_car_pos": {"x": round(float(rc_center_x), 2), "y": round(float(rc_center_y), 2)},
            "tree_pos": {"x": tree_cx, "y": tree_cy},
            "lodge_wall_y": 11.5,
        }

        return all_points, timestamp, metadata

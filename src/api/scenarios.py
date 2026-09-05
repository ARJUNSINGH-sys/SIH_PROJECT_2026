"""Synthetic LiDAR Scenario Generator for SIH 26053 Perception Engine.

Generates realistic defense UGV operating scenarios:
  1. Off-Road Military Recon (rough terrain, boulders, ditch hazards, moving enemy vehicle)
  2. Urban Crosswalk & Intersection (flat asphalt, curbs, sidewalks, pedestrians, cars)
  3. Hazardous Drop-offs & Wall Obstacles (concrete barriers, trenches, potholes)
  4. Extreme Range Dual-Tier Horizon (100m envelope verifying 5cm vs 50cm boundary)
"""

from __future__ import annotations

import numpy as np
from perception.eigensight_scene_sim import EigensightSceneSimulator


def generate_scenario_sweep(
    scenario_name: str,
    frame_idx: int,
    dt: float = 0.1,
    num_points: int = 15000,
) -> tuple[np.ndarray, float, dict]:
    """Generates a point cloud sweep [N, 4] -> (x, y, z, intensity) for the given scenario and frame."""
    timestamp = frame_idx * dt
    rng = np.random.default_rng(100 + frame_idx)
    parts = []
    metadata = {"scenario": scenario_name, "frame": frame_idx, "timestamp": timestamp}

    if scenario_name in ("eigensight_stacking", "default"):
        sim = EigensightSceneSimulator(pts_density=num_points, seed=42 + frame_idx)
        return sim.generate_sweep(frame_idx=frame_idx)

    elif scenario_name == "military_recon":
        # 1. Undulating Rough Ground (-20m to +20m)
        n_ground = int(num_points * 0.65)
        gx = rng.uniform(-20.0, 20.0, n_ground)
        gy = rng.uniform(-20.0, 20.0, n_ground)
        # Ground height with undulating sine wave terrain + noise
        gz = 0.08 * np.sin(gx * 0.5) * np.cos(gy * 0.5) + rng.normal(0.0, 0.03, n_ground)
        gi = rng.uniform(40.0, 110.0, n_ground)
        parts.append(np.column_stack([gx, gy, gz, gi]))

        # 2. Local Rough Boulders & Ditches (x: 3..6m, y: -4..-1m)
        n_rough = int(num_points * 0.15)
        rx = rng.uniform(3.0, 6.0, n_rough)
        ry = rng.uniform(-4.0, -1.0, n_rough)
        rz = rng.uniform(0.12, 0.35, n_rough)  # Curb/drop-off roughness
        ri = rng.uniform(20.0, 60.0, n_rough)
        parts.append(np.column_stack([rx, ry, rz, ri]))

        # 3. Static Blast Wall Obstacle at x = 7.0m, y = 2..8m
        n_wall = int(num_points * 0.10)
        wx = 7.0 + rng.normal(0.0, 0.1, n_wall)
        wy = rng.uniform(2.0, 8.0, n_wall)
        wz = rng.uniform(0.3, 2.2, n_wall)
        wi = rng.uniform(180.0, 240.0, n_wall)
        parts.append(np.column_stack([wx, wy, wz, wi]))

        # 4. Moving Target (Patrol UGV moving at Vx = 3.5 m/s, Vy = 1.0 m/s)
        n_dyn = int(num_points * 0.10)
        cx = -8.0 + 3.5 * timestamp
        cy = -6.0 + 1.0 * timestamp
        dx = cx + rng.uniform(-0.8, 0.8, n_dyn)
        dy = cy + rng.uniform(-0.5, 0.5, n_dyn)
        dz = rng.uniform(0.3, 1.6, n_dyn)
        di = rng.uniform(150.0, 230.0, n_dyn)
        parts.append(np.column_stack([dx, dy, dz, di]))

    elif scenario_name == "urban_crosswalk":
        # 1. Flat Road Surface (-15m to +15m)
        n_road = int(num_points * 0.60)
        rx = rng.uniform(-15.0, 15.0, n_road)
        ry = rng.uniform(-15.0, 15.0, n_road)
        rz = rng.normal(0.0, 0.015, n_road)
        ri = rng.uniform(50.0, 90.0, n_road)
        parts.append(np.column_stack([rx, ry, rz, ri]))

        # 2. Elevated Sidewalk & Curb (y > 4m, step of +0.18m)
        n_curb = int(num_points * 0.15)
        cx = rng.uniform(-15.0, 15.0, n_curb)
        cy = rng.uniform(4.0, 8.0, n_curb)
        cz = 0.18 + rng.normal(0.0, 0.02, n_curb)
        ci = rng.uniform(70.0, 120.0, n_curb)
        parts.append(np.column_stack([cx, cy, cz, ci]))

        # 3. Static Lamp Posts / Buildings
        n_pole = int(num_points * 0.08)
        px = 5.0 + rng.normal(0.0, 0.15, n_pole)
        py = 6.0 + rng.normal(0.0, 0.15, n_pole)
        pz = rng.uniform(0.2, 4.0, n_pole)
        pi = rng.uniform(180.0, 255.0, n_pole)
        parts.append(np.column_stack([px, py, pz, pi]))

        # 4. Moving Pedestrian (crossing road at Vy = 1.3 m/s)
        n_ped = int(num_points * 0.07)
        ped_x = 2.0 + rng.uniform(-0.3, 0.3, n_ped)
        ped_y = -5.0 + 1.3 * timestamp + rng.uniform(-0.3, 0.3, n_ped)
        ped_z = rng.uniform(0.1, 1.8, n_ped)
        ped_i = rng.uniform(110.0, 170.0, n_ped)
        parts.append(np.column_stack([ped_x, ped_y, ped_z, ped_i]))

        # 5. Moving Vehicle (driving down road at Vx = -6.0 m/s)
        n_car = int(num_points * 0.10)
        car_x = 12.0 - 6.0 * timestamp + rng.uniform(-1.8, 1.8, n_car)
        car_y = -2.0 + rng.uniform(-0.9, 0.9, n_car)
        car_z = rng.uniform(0.2, 1.5, n_car)
        car_i = rng.uniform(190.0, 245.0, n_car)
        parts.append(np.column_stack([car_x, car_y, car_z, car_i]))

    elif scenario_name == "extreme_range":
        # Full 100m Dual-tier Verification
        n_near = int(num_points * 0.50)
        nx = rng.uniform(-9.5, 9.5, n_near)
        ny = rng.uniform(-9.5, 9.5, n_near)
        nz = rng.normal(0.0, 0.02, n_near)
        ni = rng.uniform(30.0, 150.0, n_near)
        parts.append(np.column_stack([nx, ny, nz, ni]))

        # Boundary ring points (r = 9.8m and 10.5m)
        theta = rng.uniform(0, 2 * np.pi, 500)
        r_ring = rng.choice([9.8, 10.3], 500) + rng.normal(0, 0.05, 500)
        parts.append(np.column_stack([r_ring * np.cos(theta), r_ring * np.sin(theta), np.zeros(500), np.full(500, 200.0)]))

        # Far-field targets (out to 80-95m)
        n_far = int(num_points * 0.40)
        far_r = rng.uniform(15.0, 95.0, n_far)
        far_theta = rng.uniform(0, 2 * np.pi, n_far)
        fx = far_r * np.cos(far_theta)
        fy = far_r * np.sin(far_theta)
        fz = rng.uniform(0.0, 2.0, n_far)
        fi = rng.uniform(50.0, 180.0, n_far)
        parts.append(np.column_stack([fx, fy, fz, fi]))

    else:
        # Default: Standard synthetic benchmark sweep
        gx = rng.uniform(-18.0, 18.0, num_points)
        gy = rng.uniform(-18.0, 18.0, num_points)
        gz = rng.normal(0.0, 0.05, num_points)
        gi = rng.uniform(30.0, 200.0, num_points)
        parts.append(np.column_stack([gx, gy, gz, gi]))

    points = np.vstack(parts).astype(np.float32)
    return points, timestamp, metadata

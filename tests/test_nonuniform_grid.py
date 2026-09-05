"""Unit tests for Buerkle et al. (IEEE IV 2020) Non-Uniform Occupancy Grid."""

import numpy as np
import pytest

from perception.nonuniform_grid import (
    GridPartition,
    NonUniformGrid,
    build_default_9partition,
)


def test_build_default_9partition():
    """Validates 9-partition layout according to Table II & Fig. 2 of paper."""
    partitions = build_default_9partition(radius=100.0)
    assert len(partitions) == 9

    # Check center partition (k=4) is finest tier @ 0.1m
    center = partitions[4]
    assert center.k == 4
    assert center.x_min == -20.0 and center.x_max == 20.0
    assert center.y_min == -20.0 and center.y_max == 20.0
    assert center.delta_x == 0.1 and center.delta_y == 0.1
    assert center.nx == 400 and center.ny == 400
    assert center.n_cells == 160_000

    # Check total cells across 9 partitions
    total_cells = sum(p.n_cells for p in partitions)
    assert total_cells == 640_000

    # Verify contiguous offsets
    expected_offset = 0
    for p in partitions:
        assert p.offset == expected_offset
        expected_offset += p.n_cells


def test_f_and_g_mapping_roundtrip():
    """Verifies Cartesian-to-index f(x,y) and index-to-Cartesian g(i) mappings."""
    partitions = build_default_9partition(radius=100.0)
    grid = NonUniformGrid(partitions, radius=100.0)

    # Sample points across all 9 partitions
    xs = np.array([-80.0, 0.0, 80.0, -80.0, 0.0, 80.0, -80.0, 0.0, 80.0], dtype=np.float32)
    ys = np.array([-80.0, -80.0, -80.0, 0.0, 0.0, 0.0, 80.0, 80.0, 80.0], dtype=np.float32)

    indices = grid.f(xs, ys)
    assert np.all(indices >= 0)
    assert len(np.unique(indices)) == len(indices)

    # Invert mapping: g(f(x, y)) should yield cell centers containing (x, y)
    cx, cy = grid.g(indices)
    assert not np.isnan(cx).any()
    assert not np.isnan(cy).any()

    # Center must be within half-cell of original point
    dx = np.abs(cx - xs)
    dy = np.abs(cy - ys)
    assert np.all(dx <= 0.4)
    assert np.all(dy <= 0.4)


def test_out_of_bounds_mapping():
    """Points outside [-radius, +radius] must map to index -1."""
    partitions = build_default_9partition(radius=100.0)
    grid = NonUniformGrid(partitions, radius=100.0)

    oob_x = np.array([-150.0, 150.0, 0.0], dtype=np.float32)
    oob_y = np.array([0.0, 0.0, 200.0], dtype=np.float32)

    indices = grid.f(oob_x, oob_y)
    assert np.all(indices == -1)

    # g(-1) should return NaN
    cx, cy = grid.g(indices)
    assert np.all(np.isnan(cx))
    assert np.all(np.isnan(cy))


def test_dempster_shafer_update():
    """Verifies Dempster-Shafer belief tracking and normalization."""
    partitions = build_default_9partition(radius=100.0)
    grid = NonUniformGrid(partitions, radius=100.0)

    # Initial state: all unknown (channel 3 = 1.0)
    assert np.allclose(grid.belief[:, 3], 1.0)
    assert np.allclose(grid.belief[:, :3], 0.0)

    # Points with dynamic obstacle (label 3) and static obstacle (label 2)
    pts = np.array([
        [5.0, 5.0, 0.5, 100.0],   # Dynamic target
        [-10.0, -10.0, 1.2, 80.0], # Static obstacle
    ], dtype=np.float32)
    labels = np.array([3, 2], dtype=np.int64)

    grid.update_scan(pts, labels, rover_x=0.0, rover_y=0.0)

    # Beliefs must sum to 1.0 across all cells
    sums = grid.belief.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)

    # Check cell classification
    cell_labels = grid.label_cells()
    # Find index of dynamic point
    dyn_idx = grid.f(np.array([5.0], dtype=np.float32), np.array([5.0], dtype=np.float32))[0]
    # Find index of static point
    stat_idx = grid.f(np.array([-10.0], dtype=np.float32), np.array([-10.0], dtype=np.float32))[0]

    assert cell_labels[dyn_idx] == 3  # Dynamic (D)
    assert cell_labels[stat_idx] == 2  # Static (S)


def test_free_space_transfer_motion_compensation():
    """Verifies ego-motion compensation (Eq. 6-7 and Fig. 6)."""
    partitions = build_default_9partition(radius=100.0)
    grid = NonUniformGrid(partitions, radius=100.0)

    # Put a static obstacle at (10, 0)
    pts = np.array([[10.0, 0.0, 1.0, 100.0]], dtype=np.float32)
    labels = np.array([2], dtype=np.int64)
    grid.update_scan(pts, labels, rover_x=0.0, rover_y=0.0)

    # Forward ego-motion of 2 meters: dx = +2.0
    grid.free_space_transfer(ego_dx=2.0, ego_dy=0.0)

    # In rover's new frame, obstacle should now be at (8, 0)
    new_obs_idx = grid.f(np.array([8.0], dtype=np.float32), np.array([0.0], dtype=np.float32))[0]
    assert grid.belief[new_obs_idx, 1] > 0.1  # Static mass preserved


def test_to_dense_grids_output_shapes():
    """Verifies dense grid exports match the (400, 400) API contract."""
    partitions = build_default_9partition(radius=100.0)
    grid = NonUniformGrid(partitions, radius=100.0)

    pts = np.random.uniform(-15.0, 15.0, (1000, 4)).astype(np.float32)
    labels = np.random.choice([0, 1, 2, 3], size=1000)
    grid.update_scan(pts, labels, rover_x=0.0, rover_y=0.0)

    local_std, local_var, local_elev, local_cls, global_binary, near_pts, far_pts = grid.to_dense_grids(0.0, 0.0)

    assert local_std.shape == (400, 400)
    assert local_var.shape == (400, 400)
    assert local_elev.shape == (400, 400)
    assert local_cls.shape == (400, 400)
    assert global_binary.shape == (400, 400)
    assert set(np.unique(global_binary)).issubset({0, 1})


def test_memory_stats_reduction():
    """Verifies 96% cell count reduction vs uniform baseline."""
    partitions = build_default_9partition(radius=100.0)
    grid = NonUniformGrid(partitions, radius=100.0)
    stats = grid.memory_stats()

    assert stats["uniform_5cm_cells"] == 16_000_000
    assert stats["nonuniform_total_cells"] == 640_000
    assert np.isclose(stats["cell_count_reduction_pct"], 96.0)


def test_get_belief_grids():
    """Verifies downsampled 100x100 belief grids for web rendering."""
    partitions = build_default_9partition(radius=100.0)
    grid = NonUniformGrid(partitions, radius=100.0)
    beliefs = grid.get_belief_grids()

    assert "dynamic" in beliefs
    assert "static" in beliefs
    assert "free" in beliefs
    assert beliefs["dynamic"].shape == (100, 100)
    assert beliefs["static"].shape == (100, 100)
    assert beliefs["free"].shape == (100, 100)

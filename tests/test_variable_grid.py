"""Tests for Module 4 — Variable-Resolution Grid Quantiser."""

import numpy as np

from perception.types import GridScope, PerceptionConfig, SemanticLabel, SemanticPointCloud
from perception.variable_grid import VariableGridQuantiser


def test_radial_dual_tier_routing():
    config = PerceptionConfig(
        local_radius_m=10.0,
        global_radius_m=100.0,
        local_resolution_m=0.05,
        global_resolution_m=0.50,
    )
    quantiser = VariableGridQuantiser(config)

    points = np.array([
        [3.0, 4.0, 0.0, 0.0],      # r = 5.0m  -> LOCAL
        [9.9, 0.0, 0.0, 0.0],      # r = 9.9m  -> LOCAL
        [10.1, 0.0, 0.0, 0.0],     # r = 10.1m -> GLOBAL
        [50.0, 50.0, 0.0, 0.0],    # r = 70.7m -> GLOBAL
        [99.0, 0.0, 0.0, 0.0],     # r = 99.0m -> GLOBAL
        [150.0, 0.0, 0.0, 0.0],    # r = 150m  -> DISCARDED
    ])
    labels = np.zeros(len(points), dtype=np.int64)
    cloud = SemanticPointCloud(points, labels)

    grid = quantiser.quantise(cloud)
    assert grid.n_local_points == 2
    assert grid.n_global_points == 3
    assert grid.n_discarded == 1

    # Check local cells have scope LOCAL
    for cell in grid.local_cells.values():
        assert cell.scope == GridScope.LOCAL

    # Check global cells have scope GLOBAL
    for cell in grid.global_cells.values():
        assert cell.scope == GridScope.GLOBAL


def test_memory_cell_count_reduction():
    config = PerceptionConfig()
    quantiser = VariableGridQuantiser(config)
    stats = quantiser.memory_stats()

    assert stats["uniform_5cm_cells"] == 16_000_000
    assert stats["proposed_local_cells"] == 160_000
    assert stats["proposed_global_cells"] == 160_000
    assert stats["total_proposed_cells"] == 320_000
    assert np.isclose(stats["cell_count_reduction_pct"], 98.0)

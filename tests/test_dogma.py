"""Tests for Module 6 — DOGMa Builder and PerceptionPipeline."""

import numpy as np

from perception.pipeline import PerceptionPipeline
from perception.types import (
    GridScope,
    PerceptionConfig,
    SemanticLabel,
)


def test_full_pipeline_dogma_construction():
    config = PerceptionConfig(model_architecture="mock")
    pipeline = PerceptionPipeline(config)

    # Synthetic LiDAR sweep:
    # 1. Terrain points in local zone
    terrain_local = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.02, 0.01],
        [1.02, 1.0, 0.0],
    ])
    # 2. Static obstacle in global zone
    static_global = np.array([
        [20.0, 20.0, 1.5],
        [20.0, 20.2, 1.8],
        [20.2, 20.0, 2.0],
    ])
    sweep = np.vstack([terrain_local, static_global])

    dogma = pipeline.process_sweep(sweep, timestamp=0.0)

    assert dogma.local_active_cells > 0
    assert dogma.global_active_cells > 0
    assert dogma.total_active_cells == dogma.local_active_cells + dogma.global_active_cells

    # Check local cell has terrain statistics populated
    local_keys = list(dogma.local.cells.keys())
    first_local = dogma.local.cells[local_keys[0]]
    assert first_local.scope == GridScope.LOCAL
    assert first_local.mean_z is not None
    assert first_local.variance_z is not None

    # Check global cell does NOT store detailed Z statistics
    global_keys = list(dogma.global_grid.cells.keys())
    first_global = dogma.global_grid.cells[global_keys[0]]
    assert first_global.scope == GridScope.GLOBAL
    assert first_global.mean_z is None
    assert first_global.variance_z is None
    assert first_global.occupancy_state == 1  # High static obstacle -> occupied

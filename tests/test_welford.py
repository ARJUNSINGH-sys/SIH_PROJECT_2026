"""Tests for Module 5 — Welford Algorithm and Terrain Analyser."""

import numpy as np

from perception.terrain_analyzer import TerrainAnalyser
from perception.types import GridScope, PerceptionConfig, WelfordAccumulator
from perception.variable_grid import GridCell, QuantisedGrid


def test_welford_accumulator_math():
    acc = WelfordAccumulator()
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    for s in samples:
        acc.update(s)

    mean, var = acc.finalize()
    assert np.isclose(mean, np.mean(samples))
    assert np.isclose(var, np.var(samples, ddof=1))
    assert acc.min_z == 1.0
    assert acc.max_z == 5.0


def test_welford_empty_and_single_point():
    acc_empty = WelfordAccumulator()
    mean_e, var_e = acc_empty.finalize()
    assert mean_e == 0.0
    assert var_e == 0.0

    acc_single = WelfordAccumulator()
    acc_single.update(3.5)
    mean_s, var_s = acc_single.finalize()
    assert mean_s == 3.5
    assert var_s == 0.0


def test_terrain_analyser_roughness_detection():
    config = PerceptionConfig(
        terrain_min_samples=3,
        terrain_variance_threshold=0.01,
    )
    analyser = TerrainAnalyser(config)

    # Prepare a quantised grid with 2 local cells:
    # Cell 1: Smooth flat terrain (low variance)
    # Cell 2: Rough rocky terrain (high variance)
    grid = QuantisedGrid()
    c_smooth = GridCell(i=0, j=0, scope=GridScope.LOCAL)
    c_smooth.z_values = [0.01, 0.02, 0.01, 0.00, 0.02]

    c_rough = GridCell(i=1, j=1, scope=GridScope.LOCAL)
    c_rough.z_values = [0.0, 0.5, 0.1, 0.8, 0.3]

    grid.local_cells[(0, 0)] = c_smooth
    grid.local_cells[(1, 1)] = c_rough

    results = analyser.analyse(grid)
    assert len(results) == 2

    assert not results[(0, 0)].is_rough
    assert results[(1, 1)].is_rough

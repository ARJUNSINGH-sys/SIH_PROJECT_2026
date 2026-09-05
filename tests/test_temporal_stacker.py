"""Tests for Module 1 — Temporal Stacker and validation."""

import numpy as np
import pytest

from perception.temporal_stacker import TemporalStacker
from perception.types import PerceptionConfig
from perception.validation import validate_point_cloud, validate_timestamp


def test_temporal_stacking_rolling_buffer():
    config = PerceptionConfig(max_temporal_frames=3)
    stacker = TemporalStacker(config)

    # 1st sweep at t=10.0
    p1 = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.5]])
    stack1 = stacker.add_sweep(p1, timestamp=10.0)
    assert len(stack1) == 2
    assert np.allclose(stack1[:, 3], 0.0)

    # 2nd sweep at t=10.1
    p2 = np.array([[1.1, 2.1, 0.0]])
    stack2 = stacker.add_sweep(p2, timestamp=10.1)
    assert len(stack2) == 3
    # First 2 points should have dt = 10.0 - 10.1 = -0.1
    assert np.allclose(stack2[:2, 3], -0.1)
    # Newest point has dt = 0.0
    assert np.isclose(stack2[2, 3], 0.0)

    # 3rd sweep at t=10.2
    p3 = np.array([[1.2, 2.2, 0.0]])
    stack3 = stacker.add_sweep(p3, timestamp=10.2)
    assert len(stack3) == 4
    assert np.allclose(stack3[:2, 3], -0.2)
    assert np.isclose(stack3[2, 3], -0.1)
    assert np.isclose(stack3[3, 3], 0.0)

    # 4th sweep at t=10.3 (should drop 1st sweep because max_frames=3)
    p4 = np.array([[1.3, 2.3, 0.0]])
    stack4 = stacker.add_sweep(p4, timestamp=10.3)
    assert len(stack4) == 3  # 1 pt from sweep 2, 1 pt from sweep 3, 1 pt from sweep 4
    assert np.isclose(stack4[0, 3], -0.2)
    assert np.isclose(stack4[1, 3], -0.1)
    assert np.isclose(stack4[2, 3], 0.0)


def test_invalid_lidar_input():
    # NaN and Inf filtering
    bad_pts = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, 2.0, 3.0],
        [1.0, np.inf, 3.0],
        [4.0, 5.0, 6.0],
    ])
    clean = validate_point_cloud(bad_pts, expected_cols=3)
    assert len(clean) == 2
    assert np.allclose(clean[0], [1.0, 2.0, 3.0])
    assert np.allclose(clean[1], [4.0, 5.0, 6.0])

    # Wrong shape
    with pytest.raises(ValueError):
        validate_point_cloud(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        validate_point_cloud(np.zeros((10, 4)), expected_cols=3)

    # Invalid timestamp
    with pytest.raises(ValueError):
        validate_timestamp(float("nan"))

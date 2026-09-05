"""Input validation utilities for LiDAR point cloud data."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def validate_point_cloud(points: np.ndarray, expected_cols: int = 3) -> np.ndarray:
    """Validate and sanitise a raw LiDAR point cloud array.

    Checks for: wrong dimensions, NaN, Inf, empty arrays.
    Returns a clean copy with invalid rows removed.

    Raises ValueError on structurally invalid input.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(points).__name__}")

    if points.ndim != 2:
        raise ValueError(f"Expected 2D array [N, {expected_cols}], got shape {points.shape}")

    if points.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns (got {points.shape[1]}). "
            f"Shape: {points.shape}"
        )

    if len(points) == 0:
        logger.warning("Received empty point cloud")
        return points.copy()

    # Remove rows containing NaN or Inf
    finite_mask = np.all(np.isfinite(points), axis=1)
    n_invalid = int(np.sum(~finite_mask))
    if n_invalid > 0:
        logger.warning("Removed %d points containing NaN/Inf", n_invalid)

    clean = points[finite_mask].copy()
    return clean


def validate_timestamp(timestamp: float) -> None:
    """Validate a single timestamp value."""
    if not np.isfinite(timestamp):
        raise ValueError(f"Timestamp must be finite, got {timestamp}")


def validate_labels(labels: np.ndarray, valid_set: set[int] = {0, 1, 2}) -> None:
    """Validate that all semantic labels belong to the expected label set."""
    unique = set(np.unique(labels).tolist())
    invalid = unique - valid_set
    if invalid:
        raise ValueError(f"Invalid semantic labels found: {invalid}. Expected: {valid_set}")

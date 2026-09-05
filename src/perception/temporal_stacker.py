"""Module 1 — Temporal Stacker.

Maintains a rolling history of LiDAR sweeps and produces a
temporally-stacked point cloud with relative timestamps.

Input:  points [N, 3] (x, y, z)  +  timestamp float
Output: stacked [N_total, 4] (x, y, z, delta_t)
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from .types import PerceptionConfig
from .validation import validate_point_cloud, validate_timestamp

logger = logging.getLogger(__name__)


class TemporalStacker:
    """Rolling temporal buffer that stacks N sweeps with relative timestamps."""

    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config
        # Each entry: (points_xyz [N,3], absolute_timestamp)
        self._buffer: deque[tuple[np.ndarray, float]] = deque(
            maxlen=config.max_temporal_frames
        )

    def add_sweep(self, points: np.ndarray, timestamp: float) -> np.ndarray:
        """Ingest one LiDAR sweep and return the full temporal stack.

        Parameters
        ----------
        points : ndarray [N, 3]
            Raw LiDAR points (x, y, z).
        timestamp : float
            Absolute timestamp of this sweep.

        Returns
        -------
        ndarray [N_total, 4]
            Temporally stacked points (x, y, z, delta_t).
            The newest frame has delta_t = 0. Previous frames have
            negative relative timestamps.
        """
        validate_timestamp(timestamp)
        clean = validate_point_cloud(points, expected_cols=3)

        self._buffer.append((clean, timestamp))
        logger.debug(
            "Added sweep: %d points at t=%.4f (buffer: %d/%d)",
            len(clean), timestamp, len(self._buffer), self.config.max_temporal_frames,
        )

        return self._build_stack()

    def _build_stack(self) -> np.ndarray:
        """Concatenate buffered sweeps with relative timestamps."""
        if not self._buffer:
            return np.empty((0, 4), dtype=np.float64)

        # Reference time is the newest sweep
        t_current = self._buffer[-1][1]

        parts: list[np.ndarray] = []
        for pts, t_abs in self._buffer:
            n = len(pts)
            if n == 0:
                continue
            dt = t_abs - t_current  # negative for older frames, 0 for current
            dt_col = np.full((n, 1), dt, dtype=np.float64)
            parts.append(np.hstack([pts.astype(np.float64), dt_col]))

        if not parts:
            return np.empty((0, 4), dtype=np.float64)

        stacked = np.vstack(parts)
        logger.debug("Temporal stack: %d total points from %d frames", len(stacked), len(self._buffer))
        return stacked

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

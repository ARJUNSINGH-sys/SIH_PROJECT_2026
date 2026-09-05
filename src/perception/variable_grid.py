"""Module 4 — Variable-Resolution Grid Quantiser.

Routes points to LOCAL (5 cm) or GLOBAL (50 cm) cells based on
radial distance: r = sqrt(x² + y²).

    LOCAL:   0 <= r <  10 m   →  0.05 m cells
    GLOBAL: 10 <= r <= 100 m  →  0.50 m cells

All heavy indexing uses NumPy vectorisation — no Python loops over
millions of points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .types import GridScope, PerceptionConfig, SemanticPointCloud

logger = logging.getLogger(__name__)


@dataclass
class GridCell:
    """Accumulated points for a single grid cell."""
    i: int
    j: int
    scope: GridScope
    z_values: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


@dataclass
class QuantisedGrid:
    """Result of variable-resolution spatial quantisation."""
    local_cells: dict[tuple[int, int], GridCell] = field(default_factory=dict)
    global_cells: dict[tuple[int, int], GridCell] = field(default_factory=dict)
    n_local_points: int = 0
    n_global_points: int = 0
    n_discarded: int = 0


class VariableGridQuantiser:
    """Dual-tier radial variable-resolution spatial quantiser."""

    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config

    def quantise(self, semantic_cloud: SemanticPointCloud) -> QuantisedGrid:
        """Assign each point to the appropriate grid cell.

        Points beyond global_radius_m are discarded.
        """
        result = QuantisedGrid()
        pts = semantic_cloud.points  # [N, 4]: x, y, z, dt
        labels = semantic_cloud.labels

        if len(pts) == 0:
            return result

        x, y, z, dt = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        r = np.sqrt(x ** 2 + y ** 2)  # radial distance from ego-centre

        c = self.config

        # ── Masks ──────────────────────────────────────────────────
        local_mask = r < c.local_radius_m
        global_mask = (r >= c.local_radius_m) & (r <= c.global_radius_m)
        # Points beyond global_radius_m are discarded

        result.n_discarded = int(np.sum(~local_mask & ~global_mask))

        # ── Local grid (5 cm) ──────────────────────────────────────
        self._fill_cells(
            result.local_cells, GridScope.LOCAL,
            x[local_mask], y[local_mask], z[local_mask], dt[local_mask],
            labels[local_mask], c.local_resolution_m,
        )
        result.n_local_points = int(np.sum(local_mask))

        # ── Global grid (50 cm) ────────────────────────────────────
        self._fill_cells(
            result.global_cells, GridScope.GLOBAL,
            x[global_mask], y[global_mask], z[global_mask], dt[global_mask],
            labels[global_mask], c.global_resolution_m,
        )
        result.n_global_points = int(np.sum(global_mask))

        logger.info(
            "Grid quantisation: local=%d pts, global=%d pts, discarded=%d",
            result.n_local_points, result.n_global_points, result.n_discarded,
        )
        return result

    @staticmethod
    def _fill_cells(
        cells: dict[tuple[int, int], GridCell],
        scope: GridScope,
        x: np.ndarray, y: np.ndarray, z: np.ndarray, dt: np.ndarray,
        labels: np.ndarray, resolution: float,
    ) -> None:
        """Vectorised cell assignment using integer floor indexing."""
        if len(x) == 0:
            return

        # i = floor(x / resolution), j = floor(y / resolution)
        i_arr = np.floor(x / resolution).astype(np.int64)
        j_arr = np.floor(y / resolution).astype(np.int64)

        # Find unique (i, j) cell keys
        keys = np.column_stack([i_arr, j_arr])
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        for idx in range(len(unique_keys)):
            ci, cj = int(unique_keys[idx, 0]), int(unique_keys[idx, 1])
            member_mask = inverse == idx
            cell = cells.get((ci, cj))
            if cell is None:
                cell = GridCell(i=ci, j=cj, scope=scope)
                cells[(ci, cj)] = cell
            cell.z_values.extend(z[member_mask].tolist())
            cell.labels.extend(labels[member_mask].tolist())
            cell.timestamps.extend(dt[member_mask].tolist())

    def memory_stats(self) -> dict[str, float]:
        """Compute theoretical cell counts and cell-count reduction."""
        c = self.config
        uniform_cells = (2 * c.global_radius_m / c.local_resolution_m) ** 2
        local_cells = (2 * c.local_radius_m / c.local_resolution_m) ** 2
        global_cells = (2 * c.global_radius_m / c.global_resolution_m) ** 2
        proposed = local_cells + global_cells
        reduction = (1.0 - proposed / uniform_cells) * 100.0
        return {
            "uniform_5cm_cells": uniform_cells,
            "proposed_local_cells": local_cells,
            "proposed_global_cells": global_cells,
            "total_proposed_cells": proposed,
            "cell_count_reduction_pct": reduction,
        }

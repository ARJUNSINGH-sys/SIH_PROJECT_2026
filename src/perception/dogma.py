"""Module 6 — DOGMa Builder.

Assembles the final 2.5D Dynamic Occupancy Grid Map from:
    - quantised grid cells (local + global)
    - terrain analysis results (local only)
    - dynamic object kinematics (separate object list)

Output: DogMaFrame containing LocalDogMa, GlobalDogMa,
        and DynamicObject list.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from .terrain_analyzer import TerrainAnalysisResult
from .types import (
    DogmaCell,
    DogMaFrame,
    DynamicObject,
    GlobalDogMa,
    GridScope,
    LocalDogMa,
    PerceptionConfig,
    SemanticLabel,
)
from .variable_grid import QuantisedGrid

logger = logging.getLogger(__name__)


class DogMaBuilder:
    """Constructs the final DOGMa frame."""

    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config

    def build(
        self,
        grid: QuantisedGrid,
        terrain_results: dict[tuple[int, int], TerrainAnalysisResult],
        dynamic_objects: list[DynamicObject],
        timestamp: float,
    ) -> DogMaFrame:
        """Construct the complete DOGMa frame.

        Parameters
        ----------
        grid : QuantisedGrid
            Spatially quantised point cloud.
        terrain_results : dict
            Welford terrain analysis keyed by (i, j) for local cells.
        dynamic_objects : list[DynamicObject]
            Pre-grid kinematics output.
        timestamp : float
            Current perception cycle time.
        """
        local = LocalDogMa(resolution_m=self.config.local_resolution_m)
        global_grid = GlobalDogMa(resolution_m=self.config.global_resolution_m)

        # ── Local cells ────────────────────────────────────────────
        for (ci, cj), cell in grid.local_cells.items():
            dominant_label = self._dominant_label(cell.labels)

            terrain = terrain_results.get((ci, cj))
            mean_z = terrain.mean_z if terrain else None
            var_z = terrain.variance_z if terrain else None
            sample_count = terrain.sample_count if terrain else len(cell.z_values)

            # Occupancy: occupied if static or dynamic points present
            occupied = dominant_label in (SemanticLabel.STATIC_OBSTACLE, SemanticLabel.DYNAMIC_OBJECT)

            dc = DogmaCell(
                i=ci, j=cj,
                scope=GridScope.LOCAL,
                semantic_class=dominant_label,
                occupancy_state=1 if occupied else 0,
                mean_z=mean_z,
                variance_z=var_z,
                sample_count=sample_count,
                confidence=min(1.0, sample_count / 10.0),
                timestamp=timestamp,
                valid=True,
            )
            local.cells[(ci, cj)] = dc

        # ── Global cells ───────────────────────────────────────────
        for (ci, cj), cell in grid.global_cells.items():
            dominant_label = self._dominant_label(cell.labels)
            occupied = dominant_label in (SemanticLabel.STATIC_OBSTACLE, SemanticLabel.DYNAMIC_OBJECT)

            dc = DogmaCell(
                i=ci, j=cj,
                scope=GridScope.GLOBAL,
                semantic_class=dominant_label,
                occupancy_state=1 if occupied else 0,
                mean_z=None,       # detailed Z info not preserved in global grid
                variance_z=None,
                sample_count=len(cell.z_values),
                confidence=min(1.0, len(cell.z_values) / 5.0),
                timestamp=timestamp,
                valid=True,
            )
            global_grid.cells[(ci, cj)] = dc

        frame = DogMaFrame(
            local=local,
            global_grid=global_grid,
            dynamic_objects=dynamic_objects,
            timestamp=timestamp,
            local_active_cells=len(local.cells),
            global_active_cells=len(global_grid.cells),
        )

        logger.info(
            "DOGMa built: local=%d cells, global=%d cells, dynamic_objects=%d",
            frame.local_active_cells, frame.global_active_cells,
            len(dynamic_objects),
        )
        return frame

    @staticmethod
    def _dominant_label(labels: list[int]) -> int:
        """Return the most common semantic label in a cell."""
        if not labels:
            return SemanticLabel.TERRAIN
        counter = Counter(labels)
        return counter.most_common(1)[0][0]

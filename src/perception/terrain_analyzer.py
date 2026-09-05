"""Module 5 — Terrain Analyser.

Applies streaming Welford statistics to LOCAL 5 cm grid cells ONLY.
Does NOT run over the entire 100 m global region.

Exposes per-cell: mean_z, variance_z, sample_count, roughness_score.

Variance can represent rocks, vegetation, slopes, scan noise, mixed
surfaces, or registration error — NOT exclusively potholes.
"""

from __future__ import annotations

import logging

from .types import PerceptionConfig, WelfordAccumulator
from .variable_grid import GridCell, QuantisedGrid

logger = logging.getLogger(__name__)


class TerrainAnalysisResult:
    """Welford statistics for one local cell."""
    __slots__ = ("i", "j", "mean_z", "variance_z", "sample_count",
                 "min_z", "max_z", "roughness_score", "is_rough")

    def __init__(
        self,
        i: int, j: int,
        mean_z: float, variance_z: float, sample_count: int,
        min_z: float, max_z: float,
        roughness_score: float, is_rough: bool,
    ) -> None:
        self.i = i
        self.j = j
        self.mean_z = mean_z
        self.variance_z = variance_z
        self.sample_count = sample_count
        self.min_z = min_z
        self.max_z = max_z
        self.roughness_score = roughness_score
        self.is_rough = is_rough


class TerrainAnalyser:
    """Computes streaming Welford elevation statistics on local grid cells."""

    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config

    def analyse(self, grid: QuantisedGrid) -> dict[tuple[int, int], TerrainAnalysisResult]:
        """Run Welford analysis on all local cells with sufficient samples.

        Returns dict keyed by (i, j) of terrain statistics.
        """
        results: dict[tuple[int, int], TerrainAnalysisResult] = {}

        for (ci, cj), cell in grid.local_cells.items():
            if len(cell.z_values) < self.config.terrain_min_samples:
                continue

            acc = WelfordAccumulator()
            for z in cell.z_values:
                acc.update(z)

            mean_z, var_z = acc.finalize()

            # Roughness score is currently based on variance.
            # Future: can fuse with semantic, slope, point density.
            roughness = var_z
            is_rough = roughness > self.config.terrain_variance_threshold

            results[(ci, cj)] = TerrainAnalysisResult(
                i=ci, j=cj,
                mean_z=mean_z,
                variance_z=var_z,
                sample_count=acc.n,
                min_z=acc.min_z,
                max_z=acc.max_z,
                roughness_score=roughness,
                is_rough=is_rough,
            )

        logger.info(
            "Terrain analysis: %d local cells analysed (%d rough)",
            len(results),
            sum(1 for r in results.values() if r.is_rough),
        )
        return results

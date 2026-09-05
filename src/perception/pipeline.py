"""Pipeline Orchestrator — PerceptionPipeline.

Executes all six perception phases in strict sequential order:

    1. Ingest LiDAR
    2. Temporal stacking
    3. Semantic segmentation (SPVCNN / SPVNAS / Mock)
    4. Dynamic-object extraction + pre-grid kinematics
    5. Variable-resolution spatial quantisation
    6. Local Welford terrain analysis
    7. DOGMa construction

NO path planner. NO controller. NO steering. NO A*.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .dogma import DogMaBuilder
from .kinematics_engine import KinematicsEngine
from .semantic_segmenter import SemanticModel, SemanticSegmenter, create_semantic_model
from .temporal_stacker import TemporalStacker
from .terrain_analyzer import TerrainAnalyser
from .types import DogMaFrame, PerceptionConfig
from .variable_grid import VariableGridQuantiser

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """End-to-end spatiotemporal 2.5D perception engine."""

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        semantic_model: SemanticModel | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()
        self.temporal_stacker = TemporalStacker(self.config)

        # Initialize model: explicit model > configured model architecture
        model = semantic_model or create_semantic_model(self.config)
        self.semantic_segmenter = SemanticSegmenter(model)

        self.kinematics_engine = KinematicsEngine(self.config)
        self.grid_quantiser = VariableGridQuantiser(self.config)
        self.terrain_analyser = TerrainAnalyser(self.config)
        self.dogma_builder = DogMaBuilder(self.config)
        self.stage_timings: dict[str, float] = {}

    def process_sweep(
        self,
        points: np.ndarray,
        timestamp: float,
    ) -> DogMaFrame:
        """Run a single LiDAR sweep through all perception phases."""
        timings: dict[str, float] = {}

        # ── Phase 1 & 2: Temporal stacking ─────────────────────────
        t0 = time.perf_counter()
        temporal_stack = self.temporal_stacker.add_sweep(points, timestamp)
        timings["temporal_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 3: Semantic segmentation (SPVCNN / SPVNAS) ──────
        t0 = time.perf_counter()
        semantic_cloud = self.semantic_segmenter.segment(temporal_stack)
        timings["segmentation_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 4: Pre-grid kinematics ───────────────────────────
        t0 = time.perf_counter()
        dynamic_objects = self.kinematics_engine.estimate(semantic_cloud, timestamp)
        timings["kinematics_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 5: Variable-resolution quantisation ──────────────
        t0 = time.perf_counter()
        grid = self.grid_quantiser.quantise(semantic_cloud)
        timings["quantisation_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 6: Local Welford terrain analysis ────────────────
        t0 = time.perf_counter()
        terrain_results = self.terrain_analyser.analyse(grid)
        timings["terrain_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 7: DOGMa construction ────────────────────────────
        t0 = time.perf_counter()
        dogma = self.dogma_builder.build(grid, terrain_results, dynamic_objects, timestamp)
        timings["dogma_ms"] = (time.perf_counter() - t0) * 1000

        self.stage_timings = timings
        total = sum(timings.values())
        logger.info("Pipeline complete in %.1f ms: %s", total, timings)

        return dogma

    def reset(self) -> None:
        """Reset all stateful modules."""
        self.temporal_stacker.clear()
        self.kinematics_engine.clear()
        self.stage_timings = {}

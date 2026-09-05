"""Core data types for SIH 26053 perception engine.

Velocity belongs to an object/track or a velocity field,
not intrinsically to every LiDAR point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ── SemanticKITTI 19-Class Ontology ──────────────────────────────
class SemanticKITTIClass(IntEnum):
    """Standard SemanticKITTI evaluation classes.

    SPVCNN / SPVNAS outputs these fine-grained classes.
    """
    UNLABELED = 0
    OUTLIER = 1
    CAR = 10
    BICYCLE = 11
    BUS = 13
    MOTORCYCLE = 15
    ON_RAILS = 16
    TRUCK = 18
    OTHER_VEHICLE = 20
    PERSON = 30
    BICYCLIST = 31
    MOTORCYCLIST = 32
    ROAD = 40
    PARKING = 44
    SIDEWALK = 48
    OTHER_GROUND = 49
    BUILDING = 50
    FENCE = 51
    OTHER_STRUCTURE = 52
    LANE_MARKING = 60
    VEGETATION = 70
    TRUNK = 71
    TERRAIN = 72
    POLE = 80
    TRAFFIC_SIGN = 81
    OTHER_OBJECT = 99


# ── Mission Semantic Labels ───────────────────────────────────────
class SemanticLabel(IntEnum):
    """SIH 26053 Mission Classes.

    Decoupled from motion:
      TERRAIN: Drivable ground surface
      STATIC_OBSTACLE: Fixed impassable structure
      DYNAMIC_OBJECT: Moving candidate (verified by temporal kinematics)
    """
    TERRAIN = 0
    STATIC_OBSTACLE = 1
    DYNAMIC_OBJECT = 2


# ── Model Architecture Type ───────────────────────────────────────
class ModelArchitecture(str, Enum):
    SPVCNN = "spvcnn"   # ~5.5M parameters (research baseline)
    SPVNAS = "spvnas"   # ~3.3M parameters (lightweight edge efficiency)
    MOCK = "mock"       # Rule-based synthetic baseline


# ── Grid Scope ─────────────────────────────────────────────────────
class GridScope(IntEnum):
    LOCAL = 0   # 0 <= r < 10 m,  5 cm resolution
    GLOBAL = 1  # 10 <= r <= 100 m,  50 cm resolution


# ── Configuration ──────────────────────────────────────────────────
@dataclass(frozen=True)
class PerceptionConfig:
    """Single source of configurable parameters loaded from YAML."""
    local_radius_m: float = 10.0
    global_radius_m: float = 100.0
    local_resolution_m: float = 0.05
    global_resolution_m: float = 0.50

    max_temporal_frames: int = 3

    model_architecture: str = "spvcnn"
    terrain_label: int = 0
    static_label: int = 1
    dynamic_label: int = 2
    confidence_threshold: float = 0.40

    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 3
    max_association_distance: float = 3.0
    velocity_noise_threshold_mps: float = 0.30  # Speed threshold to confirm moving object

    terrain_min_samples: int = 3
    terrain_variance_threshold: float = 0.0064

    @staticmethod
    def from_yaml(path: str | Path) -> PerceptionConfig:
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found, using defaults", path)
            return PerceptionConfig()
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        loc = d.get("local", {})
        glb = d.get("global", {})
        tmp = d.get("temporal", {})
        sem = d.get("semantic", {})
        kin = d.get("kinematics", {})
        ter = d.get("terrain", {})
        return PerceptionConfig(
            local_radius_m=loc.get("radius_m", 10.0),
            global_radius_m=glb.get("radius_m", 100.0),
            local_resolution_m=loc.get("resolution_m", 0.05),
            global_resolution_m=glb.get("resolution_m", 0.50),
            max_temporal_frames=tmp.get("max_frames", 3),
            model_architecture=sem.get("model_architecture", "spvcnn"),
            terrain_label=sem.get("terrain_label", 0),
            static_label=sem.get("static_label", 1),
            dynamic_label=sem.get("dynamic_label", 2),
            confidence_threshold=sem.get("confidence_threshold", 0.40),
            dbscan_eps=kin.get("dbscan_eps", 1.0),
            dbscan_min_samples=kin.get("dbscan_min_samples", 3),
            max_association_distance=kin.get("max_association_distance", 3.0),
            velocity_noise_threshold_mps=kin.get("velocity_noise_threshold_mps", 0.30),
            terrain_min_samples=ter.get("min_samples", 3),
            terrain_variance_threshold=ter.get("variance_threshold", 0.0064),
        )

    @property
    def local_cells_per_side(self) -> int:
        return round(2 * self.local_radius_m / self.local_resolution_m)

    @property
    def global_cells_per_side(self) -> int:
        return round(2 * self.global_radius_m / self.global_resolution_m)


# ── Semantic Point Cloud ───────────────────────────────────────────
@dataclass
class SemanticPointCloud:
    """Temporal point cloud with per-point semantic labels.

    points:  ndarray [N, 4] -> (x, y, z, delta_t)
    labels:  ndarray [N]    -> SemanticLabel integers
    raw_kitti_classes: ndarray [N] optional fine-grained SemanticKITTI IDs
    """
    points: np.ndarray
    labels: np.ndarray
    raw_kitti_classes: np.ndarray | None = None

    def __post_init__(self) -> None:
        assert self.points.ndim == 2 and self.points.shape[1] == 4
        assert self.labels.ndim == 1
        assert len(self.points) == len(self.labels)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def mask(self, label: int) -> np.ndarray:
        """Return boolean mask selecting points with the given label."""
        return self.labels == label


# ── Dynamic Object (velocity belongs to object, not to points) ────
@dataclass
class DynamicObject:
    """Tracked dynamic object with centroid and planar velocity."""
    object_id: int
    centroid_x: float
    centroid_y: float
    velocity_x: float
    velocity_y: float
    point_count: int
    confidence: float
    timestamp: float
    is_moving: bool = True
    object_type: str = "dynamic_candidate"


# ── Welford Accumulator ───────────────────────────────────────────
@dataclass
class WelfordAccumulator:
    """Streaming Welford online statistics for a single cell."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_z: float = float("inf")
    max_z: float = float("-inf")

    def update(self, z: float) -> None:
        """Incorporate one new z observation."""
        self.n += 1
        delta = z - self.mean
        self.mean += delta / self.n
        delta2 = z - self.mean
        self.m2 += delta * delta2
        self.min_z = min(self.min_z, z)
        self.max_z = max(self.max_z, z)

    def finalize(self) -> tuple[float, float]:
        """Return (mean, sample_variance). Variance is 0 if n < 2."""
        if self.n < 2:
            return self.mean, 0.0
        return self.mean, self.m2 / (self.n - 1)


# ── DOGMa Cells ───────────────────────────────────────────────────
@dataclass
class DogmaCell:
    """One cell of the 2.5D Dynamic Occupancy Grid Map."""
    i: int
    j: int
    scope: GridScope
    semantic_class: int = SemanticLabel.TERRAIN
    occupancy_state: int = 0   # 0 = free, 1 = occupied
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0   # temporal height-change feature
    mean_z: float | None = None
    variance_z: float | None = None
    sample_count: int = 0
    confidence: float = 0.0
    timestamp: float = 0.0
    valid: bool = False


@dataclass
class LocalDogMa:
    """Local high-resolution 5 cm DOGMa grid (r < 10 m)."""
    cells: dict[tuple[int, int], DogmaCell] = field(default_factory=dict)
    resolution_m: float = 0.05


@dataclass
class GlobalDogMa:
    """Global coarse 50 cm DOGMa grid (10 m <= r <= 100 m)."""
    cells: dict[tuple[int, int], DogmaCell] = field(default_factory=dict)
    resolution_m: float = 0.50


@dataclass
class DogMaFrame:
    """Complete DOGMa output for one perception cycle."""
    local: LocalDogMa = field(default_factory=LocalDogMa)
    global_grid: GlobalDogMa = field(default_factory=GlobalDogMa)
    dynamic_objects: list[DynamicObject] = field(default_factory=list)
    timestamp: float = 0.0
    local_active_cells: int = 0
    global_active_cells: int = 0

    @property
    def total_active_cells(self) -> int:
        return self.local_active_cells + self.global_active_cells

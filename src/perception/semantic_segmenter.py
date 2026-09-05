"""Module 2 — Semantic Segmenter.

Integrates SPVCNN (~5.5M) and SPVNAS (~3.3M) Sparse Point-Voxel Convolution Models
with the SemanticKITTI Mission Label Mapper.

Core Architecture Principle:
  1. SPVCNN / SPVNAS answers: "What is it?" (car, person, road, building, vegetation...)
  2. Mission Label Mapper maps to: TERRAIN, STATIC, DYNAMIC (Candidate)
  3. Temporal Kinematics answers: "Did it move?" -> Vx, Vy
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .spvcnn import SPVCNN, SPVNAS
from .types import (
    ModelArchitecture,
    PerceptionConfig,
    SemanticKITTIClass,
    SemanticLabel,
    SemanticPointCloud,
)

logger = logging.getLogger(__name__)


# ── SemanticKITTI 19-Class to Mission Label Mapping ───────────────
# Mapping from 19 SemanticKITTI index (0..18) to mission SemanticLabel
SEMANTICKITTI_19_CLASSES = [
    (0, "car", SemanticLabel.DYNAMIC_OBJECT),
    (1, "bicycle", SemanticLabel.DYNAMIC_OBJECT),
    (2, "motorcycle", SemanticLabel.DYNAMIC_OBJECT),
    (3, "truck", SemanticLabel.DYNAMIC_OBJECT),
    (4, "other-vehicle", SemanticLabel.DYNAMIC_OBJECT),
    (5, "person", SemanticLabel.DYNAMIC_OBJECT),
    (6, "bicyclist", SemanticLabel.DYNAMIC_OBJECT),
    (7, "motorcyclist", SemanticLabel.DYNAMIC_OBJECT),
    (8, "road", SemanticLabel.TERRAIN),
    (9, "parking", SemanticLabel.TERRAIN),
    (10, "sidewalk", SemanticLabel.TERRAIN),
    (11, "other-ground", SemanticLabel.TERRAIN),
    (12, "building", SemanticLabel.STATIC_OBSTACLE),
    (13, "fence", SemanticLabel.STATIC_OBSTACLE),
    (14, "vegetation", SemanticLabel.STATIC_OBSTACLE),
    (15, "trunk", SemanticLabel.STATIC_OBSTACLE),
    (16, "terrain", SemanticLabel.TERRAIN),
    (17, "pole", SemanticLabel.STATIC_OBSTACLE),
    (18, "traffic-sign", SemanticLabel.STATIC_OBSTACLE),
]

# Quick lookup array [19] -> SemanticLabel
SEMANTICKITTI_TO_MISSION = np.array([m[2] for m in SEMANTICKITTI_19_CLASSES], dtype=np.int64)


def map_semantickitti_to_mission(kitti_class_indices: np.ndarray) -> np.ndarray:
    """Map fine-grained 19 SemanticKITTI class predictions to mission labels (Terrain/Static/Dynamic)."""
    valid_indices = np.clip(kitti_class_indices, 0, len(SEMANTICKITTI_TO_MISSION) - 1)
    return SEMANTICKITTI_TO_MISSION[valid_indices]


class SemanticModel(ABC):
    """Abstract interface for point-wise semantic classification."""

    @abstractmethod
    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Classify each point.

        Parameters
        ----------
        points : ndarray [N, 4]
            (x, y, z, delta_t)

        Returns
        -------
        (mission_labels [N], fine_grained_classes [N])
        """
        ...


class SPVCNNSemanticModel(SemanticModel):
    """SPVCNN Sparse Point-Voxel Convolution Model (~5.5M params).

    Research baseline for high-accuracy 3D point cloud semantic segmentation.
    Targeted for CPU / NPU onboard inference or GPU acceleration.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SPVCNN(in_channels=4, num_classes=19).to(self.device)
        self.model.eval()

        if weights_path and Path(weights_path).exists():
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                logger.info("Loaded SPVCNN weights from %s on %s", weights_path, self.device)
            except Exception as e:
                logger.warning("Could not load SPVCNN weights (%s). Running with initialized weights.", e)

    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        x_tensor = torch.from_numpy(points.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            kitti_preds = torch.argmax(logits, dim=-1).cpu().numpy()

        mission_labels = map_semantickitti_to_mission(kitti_preds)
        return mission_labels, kitti_preds


class SPVNASSemanticModel(SemanticModel):
    """SPVNAS Lightweight Point-Voxel NAS Model (~3.3M params, ~20 GMACs).

    Lightweight efficiency benchmark designed for SWaP-C constrained edge compute.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SPVNAS(in_channels=4, num_classes=19).to(self.device)
        self.model.eval()

        if weights_path and Path(weights_path).exists():
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                logger.info("Loaded SPVNAS weights from %s on %s", weights_path, self.device)
            except Exception as e:
                logger.warning("Could not load SPVNAS weights (%s). Running with initialized weights.", e)

    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        x_tensor = torch.from_numpy(points.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            kitti_preds = torch.argmax(logits, dim=-1).cpu().numpy()

        mission_labels = map_semantickitti_to_mission(kitti_preds)
        return mission_labels, kitti_preds


class MockSemanticModel(SemanticModel):
    """Deterministic rule-based mock for fast prototyping and testing without heavy dependencies."""

    def __init__(
        self,
        ground_threshold: float = 0.15,
        obstacle_threshold: float = 0.30,
    ) -> None:
        self.ground_threshold = ground_threshold
        self.obstacle_threshold = obstacle_threshold

    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(points)
        labels = np.full(n, SemanticLabel.TERRAIN, dtype=np.int64)
        kitti_classes = np.full(n, 8, dtype=np.int64)  # default: road (8)

        if n == 0:
            return labels, kitti_classes

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dt = points[:, 3]

        z_min = np.min(z)
        ground_ceil = z_min + self.ground_threshold

        elevated_mask = z > ground_ceil
        labels[elevated_mask] = SemanticLabel.STATIC_OBSTACLE
        kitti_classes[elevated_mask] = 12  # building/structure (12)

        # Dynamic detection via temporal motion signature
        if np.any(elevated_mask):
            elevated_idx = np.where(elevated_mask)[0]
            xy_elev = points[elevated_idx, :2]

            bin_size = 2.0
            grid_coords = np.floor(xy_elev / bin_size).astype(np.int32)
            unique_bins, inverse = np.unique(grid_coords, axis=0, return_inverse=True)

            for b_idx in range(len(unique_bins)):
                pts_in_bin = elevated_idx[inverse == b_idx]
                if len(pts_in_bin) < 3:
                    continue

                b_dts = dt[pts_in_bin]
                if np.ptp(b_dts) > 0.05:
                    unique_dts = np.unique(b_dts)
                    if len(unique_dts) >= 2:
                        centers = []
                        for u_dt in unique_dts:
                            m_dt = pts_in_bin[dt[pts_in_bin] == u_dt]
                            centers.append([np.mean(x[m_dt]), np.mean(y[m_dt])])
                        centers_arr = np.array(centers)
                        motion_span = np.ptp(centers_arr, axis=0)
                        if np.hypot(motion_span[0], motion_span[1]) > 0.20:
                            labels[pts_in_bin] = SemanticLabel.DYNAMIC_OBJECT
                            kitti_classes[pts_in_bin] = 0  # car (0)

        return labels, kitti_classes


def create_semantic_model(config: PerceptionConfig) -> SemanticModel:
    """Factory creating the appropriate semantic model based on configuration."""
    arch = config.model_architecture.lower()
    if arch == "spvnas":
        logger.info("Initializing SPVNAS Semantic Model (~3.3M parameters, ~20 GMACs)")
        return SPVNASSemanticModel()
    elif arch == "spvcnn":
        logger.info("Initializing SPVCNN Semantic Model (~5.5M parameters, ~30 GMACs)")
        return SPVCNNSemanticModel()
    else:
        logger.info("Initializing Mock Semantic Model for deterministic baseline")
        return MockSemanticModel()


class SemanticSegmenter:
    """Wraps a SemanticModel and produces SemanticPointCloud output."""

    def __init__(self, model: SemanticModel | None = None) -> None:
        self.model = model or MockSemanticModel()

    def segment(self, temporal_stack: np.ndarray) -> SemanticPointCloud:
        if len(temporal_stack) == 0:
            return SemanticPointCloud(
                points=np.empty((0, 4), dtype=np.float64),
                labels=np.empty(0, dtype=np.int64),
                raw_kitti_classes=np.empty(0, dtype=np.int64),
            )

        mission_labels, kitti_classes = self.model.predict(temporal_stack)

        counts = {
            "terrain": int(np.sum(mission_labels == SemanticLabel.TERRAIN)),
            "static": int(np.sum(mission_labels == SemanticLabel.STATIC_OBSTACLE)),
            "dynamic": int(np.sum(mission_labels == SemanticLabel.DYNAMIC_OBJECT)),
        }
        logger.info("Semantic segmentation: %s", counts)

        return SemanticPointCloud(
            points=temporal_stack,
            labels=mission_labels,
            raw_kitti_classes=kitti_classes,
        )

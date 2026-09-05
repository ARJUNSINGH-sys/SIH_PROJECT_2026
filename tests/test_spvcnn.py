"""Tests for SPVCNN, SPVNAS models and SemanticKITTI Mission Label Mapping."""

import numpy as np
import pytest
import torch

from perception.semantic_segmenter import (
    SPVCNNSemanticModel,
    SPVNASSemanticModel,
    map_semantickitti_to_mission,
)
from perception.spvcnn import SPVCNN, SPVNAS
from perception.types import SemanticLabel


def test_spvcnn_forward_shape():
    model = SPVCNN(in_channels=4, num_classes=19)
    points = torch.randn(128, 4)
    out = model(points)
    assert out.shape == (128, 19)
    assert model.param_count > 1_000_000


def test_spvnas_forward_shape():
    model = SPVNAS(in_channels=4, num_classes=19)
    points = torch.randn(128, 4)
    out = model(points)
    assert out.shape == (128, 19)
    assert model.param_count > 1_000_000
    # SPVNAS has fewer parameters than SPVCNN
    spvcnn = SPVCNN(in_channels=4, num_classes=19)
    assert model.param_count < spvcnn.param_count


def test_semantickitti_mission_label_mapping():
    # 0 = car -> DYNAMIC_OBJECT (Candidate)
    # 8 = road -> TERRAIN
    # 12 = building -> STATIC_OBSTACLE
    # 14 = vegetation -> STATIC_OBSTACLE
    # 16 = terrain -> TERRAIN
    kitti_preds = np.array([0, 8, 12, 14, 16, 5])  # 5 = person (dynamic)
    mission_labels = map_semantickitti_to_mission(kitti_preds)

    assert mission_labels[0] == SemanticLabel.DYNAMIC_OBJECT   # car
    assert mission_labels[1] == SemanticLabel.TERRAIN          # road
    assert mission_labels[2] == SemanticLabel.STATIC_OBSTACLE  # building
    assert mission_labels[3] == SemanticLabel.STATIC_OBSTACLE  # vegetation
    assert mission_labels[4] == SemanticLabel.TERRAIN          # terrain
    assert mission_labels[5] == SemanticLabel.DYNAMIC_OBJECT   # person


def test_spvcnn_semantic_model_wrapper():
    model = SPVCNNSemanticModel(device="cpu")
    points = np.random.uniform(-10, 10, (50, 4))
    mission_labels, kitti_classes = model.predict(points)

    assert len(mission_labels) == 50
    assert len(kitti_classes) == 50
    assert set(np.unique(mission_labels)).issubset({0, 1, 2})


def test_spvnas_semantic_model_wrapper():
    model = SPVNASSemanticModel(device="cpu")
    points = np.random.uniform(-10, 10, (50, 4))
    mission_labels, kitti_classes = model.predict(points)

    assert len(mission_labels) == 50
    assert len(kitti_classes) == 50
    assert set(np.unique(mission_labels)).issubset({0, 1, 2})

"""Tests for Module 2 — Semantic Segmenter and models."""

import numpy as np

from perception.semantic_segmenter import MockSemanticModel, SemanticSegmenter
from perception.types import SemanticLabel
from perception.validation import validate_labels


def test_mock_semantic_model_ground_vs_obstacle():
    model = MockSemanticModel(ground_threshold=0.15, obstacle_threshold=0.30)
    segmenter = SemanticSegmenter(model)

    points = np.array([
        [0.0, 0.0, 0.0, 0.0],    # Ground / terrain
        [0.0, 0.0, 0.1, 0.0],    # Ground / terrain
        [5.0, 5.0, 1.5, 0.0],    # High static obstacle
        [5.0, 5.0, 2.0, 0.0],    # High static obstacle
    ])

    cloud = segmenter.segment(points)
    assert len(cloud.labels) == 4
    validate_labels(cloud.labels)

    assert cloud.labels[0] == SemanticLabel.TERRAIN
    assert cloud.labels[1] == SemanticLabel.TERRAIN
    assert cloud.labels[2] == SemanticLabel.STATIC_OBSTACLE
    assert cloud.labels[3] == SemanticLabel.STATIC_OBSTACLE


def test_empty_semantic_point_cloud():
    segmenter = SemanticSegmenter()
    cloud = segmenter.segment(np.empty((0, 4), dtype=np.float64))
    assert cloud.n_points == 0
    assert len(cloud.labels) == 0

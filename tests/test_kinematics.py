"""Tests for Module 3 — Kinematics Engine."""

import numpy as np

from perception.kinematics_engine import KinematicsEngine
from perception.types import PerceptionConfig, SemanticLabel, SemanticPointCloud


def test_dbscan_clustering_and_velocity_estimation():
    config = PerceptionConfig(
        dbscan_eps=1.0,
        dbscan_min_samples=3,
        max_association_distance=3.0,
    )
    engine = KinematicsEngine(config)

    # Frame 1 at t=0.0: dynamic object cluster around (5.0, 2.0)
    pts1 = np.array([
        [4.9, 2.0, 0.5, 0.0],
        [5.1, 2.0, 0.5, 0.0],
        [5.0, 2.1, 0.5, 0.0],
        [5.0, 1.9, 0.5, 0.0],
    ])
    labels1 = np.full(len(pts1), SemanticLabel.DYNAMIC_OBJECT)
    cloud1 = SemanticPointCloud(pts1, labels1)

    objs1 = engine.estimate(cloud1, timestamp=0.0)
    assert len(objs1) == 1
    assert np.isclose(objs1[0].centroid_x, 5.0, atol=0.1)
    assert np.isclose(objs1[0].centroid_y, 2.0, atol=0.1)
    assert objs1[0].velocity_x == 0.0  # First frame -> 0 velocity
    assert objs1[0].velocity_y == 0.0

    # Frame 2 at t=0.1s: object moved by +0.5m in X and +0.1m in Y -> Vx = 5.0 m/s, Vy = 1.0 m/s
    pts2 = np.array([
        [5.4, 2.1, 0.5, 0.0],
        [5.6, 2.1, 0.5, 0.0],
        [5.5, 2.2, 0.5, 0.0],
        [5.5, 2.0, 0.5, 0.0],
    ])
    labels2 = np.full(len(pts2), SemanticLabel.DYNAMIC_OBJECT)
    cloud2 = SemanticPointCloud(pts2, labels2)

    objs2 = engine.estimate(cloud2, timestamp=0.1)
    assert len(objs2) == 1
    assert np.isclose(objs2[0].centroid_x, 5.5, atol=0.1)
    assert np.isclose(objs2[0].centroid_y, 2.1, atol=0.1)
    assert np.isclose(objs2[0].velocity_x, 5.0, atol=0.5)
    assert np.isclose(objs2[0].velocity_y, 1.0, atol=0.5)


def test_no_dynamic_points():
    config = PerceptionConfig()
    engine = KinematicsEngine(config)

    pts = np.array([[1.0, 2.0, 0.0, 0.0]])
    labels = np.array([SemanticLabel.TERRAIN])
    cloud = SemanticPointCloud(pts, labels)

    objs = engine.estimate(cloud, timestamp=0.0)
    assert len(objs) == 0

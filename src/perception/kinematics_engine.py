"""Module 3 — Kinematics Engine.

Estimates planar velocity (Vx, Vy) of dynamic objects BEFORE grid
quantisation.  Uses DBSCAN clustering and nearest-centroid association.

The ego-motion assumption for the prototype is zero, but the API
is designed so ego-motion compensation can be added later without
rewriting the core.

Future implementations can replace DBSCAN + nearest centroid with:
    - Kalman filter
    - Hungarian assignment
    - JPDA
    - Learned tracking
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import DBSCAN

from .types import DynamicObject, PerceptionConfig, SemanticLabel, SemanticPointCloud

logger = logging.getLogger(__name__)


class KinematicsEngine:
    """Pre-grid dynamic-object velocity estimation."""

    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config
        self._prev_clusters: list[_Cluster] | None = None
        self._prev_timestamp: float | None = None

    def estimate(
        self,
        semantic_cloud: SemanticPointCloud,
        timestamp: float,
    ) -> list[DynamicObject]:
        """Extract dynamic-object centroids and estimate Vx, Vy.

        Parameters
        ----------
        semantic_cloud : SemanticPointCloud
            Full temporal stack with semantic labels.
        timestamp : float
            Current sweep time.

        Returns
        -------
        list[DynamicObject]
        """
        dynamic_mask = semantic_cloud.mask(SemanticLabel.DYNAMIC_OBJECT)
        dynamic_points = semantic_cloud.points[dynamic_mask]

        if len(dynamic_points) == 0:
            logger.debug("No dynamic points in this frame")
            self._prev_clusters = []
            self._prev_timestamp = timestamp
            return []

        # DBSCAN clustering on (x, y) only
        xy = dynamic_points[:, :2]
        clustering = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
        ).fit(xy)

        cluster_labels = clustering.labels_
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # noise

        current_clusters: list[_Cluster] = []
        for cid in sorted(unique_labels):
            members = xy[cluster_labels == cid]
            cx = float(np.mean(members[:, 0]))
            cy = float(np.mean(members[:, 1]))
            current_clusters.append(_Cluster(cid, cx, cy, len(members)))

        # Associate current clusters with previous frame centroids
        # using nearest-centroid matching
        objects: list[DynamicObject] = []
        for idx, cur in enumerate(current_clusters):
            vx, vy = 0.0, 0.0
            conf = 0.5  # no prior → low confidence

            if self._prev_clusters and self._prev_timestamp is not None:
                dt = timestamp - self._prev_timestamp
                if dt > 1e-6:
                    best_prev, best_dist = self._find_nearest(cur)
                    if best_prev is not None and best_dist <= self.config.max_association_distance:
                        vx = (cur.cx - best_prev.cx) / dt
                        vy = (cur.cy - best_prev.cy) / dt
                        conf = max(0.0, 1.0 - best_dist / self.config.max_association_distance)

            objects.append(DynamicObject(
                object_id=idx,
                centroid_x=cur.cx,
                centroid_y=cur.cy,
                velocity_x=vx,
                velocity_y=vy,
                point_count=cur.count,
                confidence=conf,
                timestamp=timestamp,
            ))

        logger.info("Kinematics: %d dynamic objects tracked", len(objects))
        self._prev_clusters = current_clusters
        self._prev_timestamp = timestamp
        return objects

    def _find_nearest(self, cur: _Cluster) -> tuple[_Cluster | None, float]:
        """Find nearest previous cluster centroid to the current one."""
        if not self._prev_clusters:
            return None, float("inf")
        best: _Cluster | None = None
        best_dist = float("inf")
        for prev in self._prev_clusters:
            d = np.hypot(cur.cx - prev.cx, cur.cy - prev.cy)
            if d < best_dist:
                best_dist = d
                best = prev
        return best, best_dist

    def clear(self) -> None:
        self._prev_clusters = None
        self._prev_timestamp = None


class _Cluster:
    """Internal lightweight cluster representation."""
    __slots__ = ("cid", "cx", "cy", "count")

    def __init__(self, cid: int, cx: float, cy: float, count: int) -> None:
        self.cid = cid
        self.cx = cx
        self.cy = cy
        self.count = count

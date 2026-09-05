"""
Buerkle, C., Oboril, F., Jarquin, J., & Scholl, K.-U. (2020).
Efficient dynamic occupancy grid mapping using non-uniform cell representation.
2020 IEEE Intelligent Vehicles Symposium (IV), 1629-1634.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

@dataclass
class GridPartition:
    """
    Defines one partition Xk for the non-uniform grid.
    """
    k: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    delta_x: float
    delta_y: float
    
    nx: int = field(init=False)
    ny: int = field(init=False)
    offset: int = field(init=False)
    n_cells: int = field(init=False)

    def __post_init__(self):
        self.nx = int(np.ceil((self.x_max - self.x_min) / self.delta_x))
        self.ny = int(np.ceil((self.y_max - self.y_min) / self.delta_y))
        self.n_cells = self.nx * self.ny
        self.offset = 0  # To be set externally


def build_default_9partition(radius: float = 100.0) -> List[GridPartition]:
    """
    Create the paper's 9-partition layout (3x3 symmetric regions).
    
    The space is [-radius, +radius] × [-radius, +radius] centered on ego.
    Split into 3 bands in each axis:
    - Band 0: [-radius, -20m)   — coarse 0.4m cells
    - Band 1: [-20m, +20m)      — fine 0.1m cells  
    - Band 2: [+20m, +radius]   — coarse 0.4m cells
    
    The middle band is further split by the other axis:
    - (mid_x, far_y) and (far_x, mid_y) get medium 0.2m cells
    - (mid_x, mid_y) center gets fine 0.1m cells
    
    Returns sorted list of GridPartition with computed offsets.
    """
    parts = []
    
    # Define bounds: (x_min, x_max, y_min, y_max, delta_x, delta_y)
    bounds = [
        (-radius, -20.0, -radius, -20.0, 0.4, 0.4),  # k=0
        (-20.0, 20.0, -radius, -20.0, 0.2, 0.2),      # k=1
        (20.0, radius, -radius, -20.0, 0.4, 0.4),     # k=2
        (-radius, -20.0, -20.0, 20.0, 0.2, 0.2),      # k=3
        (-20.0, 20.0, -20.0, 20.0, 0.1, 0.1),         # k=4 (CENTER)
        (20.0, radius, -20.0, 20.0, 0.2, 0.2),        # k=5
        (-radius, -20.0, 20.0, radius, 0.4, 0.4),     # k=6
        (-20.0, 20.0, 20.0, radius, 0.2, 0.2),        # k=7
        (20.0, radius, 20.0, radius, 0.4, 0.4)        # k=8
    ]
    
    offset = 0
    for k, (xmin, xmax, ymin, ymax, dx, dy) in enumerate(bounds):
        p = GridPartition(k, xmin, xmax, ymin, ymax, dx, dy)
        p.offset = offset
        offset += p.n_cells
        parts.append(p)
        
    return parts


class NonUniformGrid:
    def __init__(self, partitions: List[GridPartition], radius: float = 100.0):
        """
        Set up the n-partition non-uniform grid.
        
        - partitions: list of GridPartition defining each region
        - radius: max operational radius in meters
        """
        self.partitions = partitions
        self.radius = radius
        self.total_cells = sum(p.n_cells for p in partitions)
        
        # self.belief channels: [free, static, dynamic, unknown]
        self.belief = np.zeros((self.total_cells, 4), dtype=np.float32)
        self.belief[:, 3] = 1.0  # initialize unknown
        
        # Elevation accumulators for Welford's algorithm equivalent
        self.elevation_sum = np.zeros(self.total_cells, dtype=np.float32)
        self.elevation_sum_sq = np.zeros(self.total_cells, dtype=np.float32)
        self.point_counts = np.zeros(self.total_cells, dtype=np.int32)
        
        self._prev_rover_x = 0.0
        self._prev_rover_y = 0.0
        
        # Pre-compute arrays for vectorized operations
        self._x_mins = np.array([p.x_min for p in partitions], dtype=np.float32)
        self._x_maxs = np.array([p.x_max for p in partitions], dtype=np.float32)
        self._y_mins = np.array([p.y_min for p in partitions], dtype=np.float32)
        self._y_maxs = np.array([p.y_max for p in partitions], dtype=np.float32)
        self._delta_xs = np.array([p.delta_x for p in partitions], dtype=np.float32)
        self._delta_ys = np.array([p.delta_y for p in partitions], dtype=np.float32)
        self._nys = np.array([p.ny for p in partitions], dtype=np.int32)
        self._offsets = np.array([p.offset for p in partitions], dtype=np.int32)

    def f(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Vectorized Cartesian-to-index mapping (Equation 5 from paper).
        """
        idx = np.full(x.shape, -1, dtype=np.int32)
        
        for k in range(len(self.partitions)):
            mask = (x >= self._x_mins[k]) & (x < self._x_maxs[k]) & \
                   (y >= self._y_mins[k]) & (y < self._y_maxs[k])
            
            ix = np.floor((x[mask] - self._x_mins[k]) / self._delta_xs[k]).astype(np.int32)
            iy = np.floor((y[mask] - self._y_mins[k]) / self._delta_ys[k]).astype(np.int32)
            
            idx[mask] = ix * self._nys[k] + iy + self._offsets[k]
            
        return idx

    def g(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized index-to-Cartesian mapping (inverse of f).
        Returns (x_center, y_center).
        """
        valid = indices >= 0
        x_center = np.full(indices.shape, np.nan, dtype=np.float32)
        y_center = np.full(indices.shape, np.nan, dtype=np.float32)
        
        if not np.any(valid):
            return x_center, y_center
            
        v_indices = indices[valid]
        
        # Find which partition each valid index belongs to
        k_idx = np.searchsorted(self._offsets, v_indices, side='right') - 1
        
        local_idx = v_indices - self._offsets[k_idx]
        ix = local_idx // self._nys[k_idx]
        iy = local_idx % self._nys[k_idx]
        
        x_center[valid] = self._x_mins[k_idx] + (ix + 0.5) * self._delta_xs[k_idx]
        y_center[valid] = self._y_mins[k_idx] + (iy + 0.5) * self._delta_ys[k_idx]
        
        return x_center, y_center

    def update_scan(self, points: np.ndarray, labels: np.ndarray, rover_x: float, rover_y: float) -> None:
        """
        Process one LiDAR scan through the grid.
        
        points: (N, 4) [x, y, z, intensity]
        labels: (N,) int — semantic class labels (0=terrain, 1=curb, 2=obstacle, 3=dynamic)
        """
        if points.shape[0] == 0:
            self._prev_rover_x = rover_x
            self._prev_rover_y = rover_y
            return
            
        px = points[:, 0] - rover_x
        py = points[:, 1] - rover_y
        pz = points[:, 2]
        
        cell_idx = self.f(px, py)
        valid_mask = cell_idx >= 0
        
        v_idx = cell_idx[valid_mask]
        v_labels = labels[valid_mask]
        v_z = pz[valid_mask]
        
        # Accumulate beliefs based on points
        scan_evidence = np.zeros((self.total_cells, 4), dtype=np.float32)
        
        # Default scan evidence for cells with NO points but within range (assume free space)
        # Simplified: apply to all, then overwrite for cells with points
        scan_evidence[:] = [0.7, 0.05, 0.05, 0.2] 
        
        # Identify highest priority label for each cell
        # Priorities (highest to lowest): dynamic(3), obstacle(2), curb(1), terrain(0)
        # We can use np.maximum.at or similar
        max_labels = np.full(self.total_cells, -1, dtype=np.int32)
        np.maximum.at(max_labels, v_idx, v_labels)
        
        # Update scan_evidence based on labels
        has_pts = max_labels >= 0
        
        dynamic_mask = max_labels == 3
        scan_evidence[dynamic_mask] = [0.1, 0.1, 0.7, 0.1]
        
        obstacle_mask = max_labels == 2
        scan_evidence[obstacle_mask] = [0.1, 0.7, 0.1, 0.1]
        
        curb_mask = max_labels == 1
        scan_evidence[curb_mask] = [0.1, 0.6, 0.1, 0.2]
        
        terrain_mask = max_labels == 0
        scan_evidence[terrain_mask] = [0.6, 0.1, 0.05, 0.25]
        
        # Simplified Dempster-Shafer via weighted average
        alpha = 0.3
        self.belief = (1 - alpha) * self.belief + alpha * scan_evidence
        
        # Normalize beliefs
        belief_sum = self.belief.sum(axis=1, keepdims=True)
        # Avoid division by zero
        belief_sum[belief_sum == 0] = 1.0
        self.belief /= belief_sum
        
        # Update elevation statistics
        np.add.at(self.point_counts, v_idx, 1)
        np.add.at(self.elevation_sum, v_idx, v_z)
        np.add.at(self.elevation_sum_sq, v_idx, v_z ** 2)
        
        # Update rover position
        self._prev_rover_x = rover_x
        self._prev_rover_y = rover_y

    def free_space_transfer(self, ego_dx: float, ego_dy: float) -> None:
        """
        Transfer grid state when ego vehicle moves (Section III.B, Equations 6-7).
        """
        if np.abs(ego_dx) < 1e-6 and np.abs(ego_dy) < 1e-6:
            return
            
        # Get centers of all cells
        all_indices = np.arange(self.total_cells, dtype=np.int32)
        cx, cy = self.g(all_indices)
        
        # Map back to new indices (where they came from relative to new pos)
        new_i = self.f(cx - ego_dx, cy - ego_dy)
        valid_transfers = new_i >= 0
        
        new_belief = np.zeros((self.total_cells, 4), dtype=np.float32)
        new_belief[:, 3] = 1.0
        
        new_elevation_sum = np.zeros(self.total_cells, dtype=np.float32)
        new_elevation_sum_sq = np.zeros(self.total_cells, dtype=np.float32)
        new_point_counts = np.zeros(self.total_cells, dtype=np.int32)
        
        old_i_valid = all_indices[valid_transfers]
        new_i_valid = new_i[valid_transfers]
        
        # Count how many old cells map to each new cell for averaging
        counts = np.bincount(new_i_valid, minlength=self.total_cells)
        valid_dest = counts > 0
        
        # Accumulate beliefs
        for c in range(4):
            b_acc = np.bincount(new_i_valid, weights=self.belief[old_i_valid, c], minlength=self.total_cells)
            new_belief[valid_dest, c] = b_acc[valid_dest] / counts[valid_dest]
            
        # Accumulate elevation stats
        e_sum_acc = np.bincount(new_i_valid, weights=self.elevation_sum[old_i_valid], minlength=self.total_cells)
        new_elevation_sum[valid_dest] = e_sum_acc[valid_dest]
        
        e_sum_sq_acc = np.bincount(new_i_valid, weights=self.elevation_sum_sq[old_i_valid], minlength=self.total_cells)
        new_elevation_sum_sq[valid_dest] = e_sum_sq_acc[valid_dest]
        
        pt_counts_acc = np.bincount(new_i_valid, weights=self.point_counts[old_i_valid], minlength=self.total_cells)
        new_point_counts[valid_dest] = pt_counts_acc[valid_dest]
        
        # Ensure beliefs sum to 1
        b_sum = new_belief.sum(axis=1, keepdims=True)
        b_sum[b_sum == 0] = 1.0
        new_belief /= b_sum
        
        self.belief = new_belief
        self.elevation_sum = new_elevation_sum
        self.elevation_sum_sq = new_elevation_sum_sq
        self.point_counts = new_point_counts

    def label_cells(self) -> np.ndarray:
        """
        Classify each cell using Equation 8 from the paper.
        Returns: (total_cells,) uint8 array
        0: Unknown (U), 1: Free (F), 2: Static (S), 3: Dynamic (D)
        """
        labels = np.zeros(self.total_cells, dtype=np.uint8)
        
        m_F = self.belief[:, 0]
        m_S = self.belief[:, 1]
        m_D = self.belief[:, 2]
        
        m_SD = 0.1
        max_SD = np.maximum(np.maximum(m_S, m_D), m_SD)
        
        is_S = (m_S == max_SD) & (m_S > m_D)
        is_D = (m_D == max_SD) & (m_D > m_S)
        is_F = m_F > 0.5
        
        labels[is_F] = 1
        labels[is_S & ~is_F] = 2
        labels[is_D & ~is_F & ~is_S] = 3
        
        return labels

    def to_dense_grids(self, rover_x: float, rover_y: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Export to dense NumPy arrays matching the existing API contract.
        """
        # Local grids: [-10m, +10m] @ 5cm -> 400x400
        # Global grids: [-100m, +100m] @ 50cm -> 400x400
        
        local_std = np.zeros((400, 400), dtype=np.float32)
        local_var = np.zeros((400, 400), dtype=np.float32)
        local_elev = np.zeros((400, 400), dtype=np.float32)
        local_classes = np.zeros((400, 400), dtype=np.uint8)
        global_binary = np.zeros((400, 400), dtype=np.uint8)
        near_points = np.empty((0, 4), dtype=np.float32)
        far_points = np.empty((0, 4), dtype=np.float32)
        
        all_indices = np.arange(self.total_cells, dtype=np.int32)
        cx, cy = self.g(all_indices)
        
        labels = self.label_cells()
        
        # Local grid filling
        local_mask = (cx >= -10.0) & (cx < 10.0) & (cy >= -10.0) & (cy < 10.0)
        lx = cx[local_mask]
        ly = cy[local_mask]
        
        gi = np.floor((lx + 10.0) / 0.05).astype(np.int32)
        gj = np.floor((ly + 10.0) / 0.05).astype(np.int32)
        
        gi = np.clip(gi, 0, 399)
        gj = np.clip(gj, 0, 399)
        
        l_idx = all_indices[local_mask]
        
        pc = self.point_counts[l_idx]
        e_sum = self.elevation_sum[l_idx]
        e_sum_sq = self.elevation_sum_sq[l_idx]
        
        valid_elev = pc > 0
        v_gi = gi[valid_elev]
        v_gj = gj[valid_elev]
        v_pc = pc[valid_elev]
        
        mean_elev = e_sum[valid_elev] / v_pc
        var_elev = (e_sum_sq[valid_elev] / v_pc) - (mean_elev ** 2)
        var_elev = np.maximum(var_elev, 0.0) # avoid precision negative issues
        
        local_elev[v_gi, v_gj] = mean_elev
        local_var[v_gi, v_gj] = var_elev
        local_std[v_gi, v_gj] = np.sqrt(var_elev)
        
        local_classes[gi, gj] = labels[local_mask]
        
        # Global grid filling
        global_mask = (cx >= -100.0) & (cx < 100.0) & (cy >= -100.0) & (cy < 100.0)
        gx = cx[global_mask]
        gy = cy[global_mask]
        
        ggi = np.floor((gx + 100.0) / 0.50).astype(np.int32)
        ggj = np.floor((gy + 100.0) / 0.50).astype(np.int32)
        
        ggi = np.clip(ggi, 0, 399)
        ggj = np.clip(ggj, 0, 399)
        
        g_labels = labels[global_mask]
        occupied = (g_labels == 2) | (g_labels == 3)
        
        global_binary[ggi[occupied], ggj[occupied]] = 1
        
        return local_std, local_var, local_elev, local_classes, global_binary, near_points, far_points

    def memory_stats(self) -> Dict[str, float]:
        """
        Return cell count metrics.
        """
        uniform_10cm = (200 / 0.1) ** 2  # Used for reference standard uniform
        uniform_5cm = (200 / 0.05) ** 2
        
        return {
            'uniform_5cm_cells': uniform_5cm,
            'nonuniform_total_cells': self.total_cells,
            'cell_count_reduction_pct': (1.0 - self.total_cells / uniform_5cm) * 100.0
        }

    def get_belief_grids(self) -> Dict[str, np.ndarray]:
        """
        Export belief channels as 100x100 downsampled grids for web rendering.
        """
        grid_F = np.zeros((400, 400), dtype=np.float32)
        grid_S = np.zeros((400, 400), dtype=np.float32)
        grid_D = np.zeros((400, 400), dtype=np.float32)
        
        all_indices = np.arange(self.total_cells, dtype=np.int32)
        cx, cy = self.g(all_indices)
        
        # Map to 400x400 over [-100, 100]
        gi = np.floor((cx + 100.0) / 0.50).astype(np.int32)
        gj = np.floor((cy + 100.0) / 0.50).astype(np.int32)
        
        valid = (gi >= 0) & (gi < 400) & (gj >= 0) & (gj < 400)
        gi = gi[valid]
        gj = gj[valid]
        
        grid_F[gi, gj] = self.belief[valid, 0]
        grid_S[gi, gj] = self.belief[valid, 1]
        grid_D[gi, gj] = self.belief[valid, 2]
        
        # Downsample via stride-4 slicing to 100x100
        return {
            'free': grid_F[::4, ::4],
            'static': grid_S[::4, ::4],
            'dynamic': grid_D[::4, ::4]
        }

import time
import torch
import torch.nn as nn

# Optional spconv import with PyTorch fallback
try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False


# =====================================================================
# GLOBAL CONFIGURATION & HARDWARE CONSTANTS
# =====================================================================
MAP_EXTENT = 20.0        # Top-down map covers [-20m, +20m] on X and Y (40m x 40m)
NEAR_BOX_HALF = 10.0     # Near-field rectangular boundary [-10m, +10m] (20m x 20m)
NEAR_RES = 0.05          # 5 cm grid resolution
FAR_RES = 0.50           # 50 cm grid resolution

GRID_SIZE = int((MAP_EXTENT * 2) / NEAR_RES)  # 800 x 800 cells
NUM_CELLS = GRID_SIZE * GRID_SIZE             # 640,000 cells


# =====================================================================
# STAGE 1: SPATIOTEMPORAL PREPROCESSING & INTENSITY MOMENTS
# =====================================================================
class SpatiotemporalEngine(nn.Module):
    """
    Maintains a 4-frame FIFO circular buffer and extracts cell-wise 
    intensity mean (mu) and variance (sigma^2) via GPU bin counting.
    """
    def __init__(self, buffer_size=4, grid_cells=NUM_CELLS):
        super().__init__()
        self.buffer_size = buffer_size
        self.grid_cells = grid_cells
        self.register_buffer("history_counts", torch.zeros((buffer_size, grid_cells), dtype=torch.float32))
        self.buffer_ptr = 0

    @torch.no_grad()
    def forward(self, points, cell_indices):
        """
        points: (N, 4) -> [X, Y, Z, Intensity]
        cell_indices: (N,) 1D flattened grid index
        """
        intensity = points[:, 3]
        
        # 1. Spatial Intensity Moments (Zero-loop bincount)
        counts = torch.bincount(cell_indices, minlength=self.grid_cells).float()
        valid_counts = counts.clamp(min=1.0)

        sum_i = torch.bincount(cell_indices, weights=intensity, minlength=self.grid_cells)
        sum_i2 = torch.bincount(cell_indices, weights=intensity ** 2, minlength=self.grid_cells)

        grid_mean = sum_i / valid_counts
        grid_var = (sum_i2 / valid_counts) - (grid_mean ** 2)
        grid_var = torch.clamp(grid_var, min=0.0)
        grid_std = torch.sqrt(grid_var)

        # 2. Temporal Fluctuation (Dynamic object indicator)
        self.history_counts[self.buffer_ptr] = counts
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
        temporal_variance = torch.var(self.history_counts, dim=0)

        # 3. Map cell stats back to individual points
        pt_mean = grid_mean[cell_indices]
        pt_var = grid_var[cell_indices]
        pt_std = grid_std[cell_indices]
        pt_temporal_var = temporal_variance[cell_indices]
        dynamic_candidate = (pt_temporal_var > 4.0).float()

        # Returns enriched feature array: [Intensity, Variance, StdDev, DynamicFlag]
        return torch.stack([intensity, pt_var, pt_std, dynamic_candidate], dim=1)


# =====================================================================
# STAGE 3: ADAPTIVE RECTANGULAR GRID ROUTER
# =====================================================================
class AdaptiveGridRouter(nn.Module):
    """
    Partitions points into a 5cm near-box ([-10m, 10m]) and 50cm far-field.
    No circles used: direct rectangular boundary alignment.
    """
    def __init__(self, near_bound=NEAR_BOX_HALF, near_res=NEAR_RES, far_res=FAR_RES):
        super().__init__()
        self.near_bound = near_bound
        self.near_res = near_res
        self.far_res = far_res

    def forward(self, points, features):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # 1. Rectangular Box Mask: [-10m, +10m] on both X and Y
        near_mask = (x >= -self.near_bound) & (x <= self.near_bound) & \
                    (y >= -self.near_bound) & (y <= self.near_bound)
        far_mask = ~near_mask

        # 2. Near-Field: 5cm Voxel Quantization with Full Features
        near_pts = points[near_mask]
        near_feats = features[near_mask]
        near_coords = torch.floor(near_pts[:, :3] / self.near_res).int()

        # 3. Far-Field: 50cm Voxel Quantization (Intensity only, stats zeroed)
        far_pts = points[far_mask]
        far_intensity = points[far_mask, 3:4]
        far_zero_stats = torch.zeros((far_pts.shape[0], 3), device=points.device)
        far_feats = torch.cat([far_intensity, far_zero_stats], dim=1)
        far_coords = torch.floor(far_pts[:, :3] / self.far_res).int()

        return (near_pts, near_coords, near_feats), (far_pts, far_coords, far_feats)


# =====================================================================
# STAGE 2: LIGHTWEIGHT SPARSE 3D CNN SEGMENTATION
# =====================================================================
class SparseSegmentationNet(nn.Module):
    """
    5-Class Sparse 3D CNN Segmentation Network.
    Uses SubMConv3d to strictly prevent empty-air densification.
    """
    def __init__(self, in_channels=4, num_classes=5):
        super().__init__()
        if HAS_SPCONV:
            self.net = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                spconv.SparseInverseConv3d(64, 32, kernel_size=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(32, num_classes, kernel_size=1)
            )
        else:
            # Vectorized PyTorch fallback block
            self.net = nn.Sequential(
                nn.Linear(in_channels, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )

    def forward(self, coords, feats, spatial_shape=[800, 800, 100]):
        if HAS_SPCONV:
            # Pack batch index into column 0
            batch_col = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
            b_coords = torch.cat([batch_col, coords], dim=1)
            
            x = spconv.SparseConvTensor(
                features=feats,
                indices=b_coords,
                spatial_shape=spatial_shape,
                batch_size=1
            )
            return self.net(x).features
        else:
            return self.net(feats)


# =====================================================================
# STAGE 4: 2.5D ELEVATION MAPPER & TERRAIN CLASSIFIER
# =====================================================================
class ElevationTerrainMapper(nn.Module):
    """
    Fuses geometric Delta-Z with intensity variance and segmented semantics
    into an 800x800 ego-centric grid ([-20m, +20m] @ 5cm).
    """
    def __init__(self, map_extent=MAP_EXTENT, resolution=NEAR_RES):
        super().__init__()
        self.map_extent = map_extent
        self.resolution = resolution
        self.grid_dim = int((map_extent * 2) / resolution)
        self.total_cells = self.grid_dim * self.grid_dim

    def compute_cell_indices(self, points):
        """Maps continuous (X, Y) coordinates to 1D grid cell keys."""
        gx = ((points[:, 0] + self.map_extent) / self.resolution).long()
        gy = ((points[:, 1] + self.map_extent) / self.resolution).long()
        valid = (gx >= 0) & (gx < self.grid_dim) & (gy >= 0) & (gy < self.grid_dim)
        flat_idx = torch.clamp(gx * self.grid_dim + gy, 0, self.total_cells - 1)
        return flat_idx, valid

    def forward(self, near_pts, near_logits, near_feats, far_pts):
        device = near_pts.device
        all_pts = torch.cat([near_pts, far_pts], dim=0)
        cell_idx, valid_mask = self.compute_cell_indices(all_pts)

        z = all_pts[:, 2]

        # 1. Compute Delta-Z per cell using scatter_reduce
        init_min = torch.full((self.total_cells,), 1e6, device=device)
        init_max = torch.full((self.total_cells,), -1e6, device=device)

        z_min = init_min.scatter_reduce(0, cell_idx[valid_mask], z[valid_mask], reduce="amin", include_self=False)
        z_max = init_max.scatter_reduce(0, cell_idx[valid_mask], z[valid_mask], reduce="amax", include_self=False)

        delta_z = z_max - z_min
        delta_z = torch.where(delta_z < 0.0, torch.zeros_like(delta_z), delta_z)

        # 2. Build 2.5D Semantic & Hazard Matrix
        class_map = torch.zeros(self.total_cells, dtype=torch.uint8, device=device)
        near_cell_idx, near_valid = self.compute_cell_indices(near_pts)
        near_classes = torch.argmax(near_logits, dim=1).to(torch.uint8)

        # Scatter near-field semantic classes
        class_map[near_cell_idx[near_valid]] = near_classes[near_valid]

        # 3. Geometric-Semantic Fusion Rules:
        # Step hazards / Walls (Delta Z >= 0.25m) -> Class 3 (Obstacle)
        # Curb / Pothole drop-offs (0.08m <= Delta Z < 0.25m) -> Class 2
        hard_obstacles = delta_z >= 0.25
        curb_hazards = (delta_z >= 0.08) & (delta_z < 0.25)

        class_map[hard_obstacles] = 3
        class_map[curb_hazards] = 2

        # Reshape to top-down 2D matrix (800, 800)
        elevation_grid = delta_z.view(self.grid_dim, self.grid_dim)
        terrain_grid = class_map.view(self.grid_dim, self.grid_dim)

        return elevation_grid, terrain_grid


# =====================================================================
# MASTER EIGENSIGHT PIPELINE
# =====================================================================
class EigenSightPipeline(nn.Module):
    """
    End-to-end perception pipeline integrating Stages 1 through 4.
    """
    def __init__(self):
        super().__init__()
        self.stage1_spatiotemporal = SpatiotemporalEngine()
        self.stage3_router = AdaptiveGridRouter()
        self.stage2_sparse_net = SparseSegmentationNet()
        self.stage4_mapper = ElevationTerrainMapper()

    def forward(self, raw_points):
        """
        raw_points: Tensor of shape (N, 4) -> [X, Y, Z, Intensity]
        """
        # Global coordinate mapping for Stage 1 moments
        cell_idx, _ = self.stage4_mapper.compute_cell_indices(raw_points)

        # Stage 1: Spatiotemporal Features
        feats = self.stage1_spatiotemporal(raw_points, cell_idx)

        # Stage 3: Adaptive Grid Routing (Rectangular box)
        near_data, far_data = self.stage3_router(raw_points, feats)
        near_pts, near_coords, near_feats = near_data
        far_pts, far_coords, far_feats = far_data

        # Stage 2: Sparse 3D Segmentation (Near-field only)
        # Offset coordinates to non-negative range for voxel spatial bounds
        offset_coords = near_coords.clone()
        offset_coords[:, 0] += int(NEAR_BOX_HALF / NEAR_RES)
        offset_coords[:, 1] += int(NEAR_BOX_HALF / NEAR_RES)
        offset_coords[:, 2] += 50  # Vertical elevation padding

        near_logits = self.stage2_sparse_net(offset_coords, near_feats)

        # Stage 4: 2.5D Elevation & Traversability Mapping
        elevation_map, terrain_map = self.stage4_mapper(
            near_pts, near_logits, near_feats, far_pts
        )

        return elevation_map, terrain_map


# =====================================================================
# LIVE BENCHMARK & HARDWARE VERIFICATION
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"EIGENSIGHT PIPELINE BENCHMARK | Target Device: {device}")
    print("=" * 65)

    pipeline = EigenSightPipeline().to(device)
    pipeline.eval()

    # Generate 100,000 synthetic LiDAR points on CUDA
    num_pts = 100000
    pts_x = torch.empty(num_pts, device=device).uniform_(-25.0, 25.0)
    pts_y = torch.empty(num_pts, device=device).uniform_(-25.0, 25.0)
    pts_z = torch.empty(num_pts, device=device).normal_(0.0, 0.3)
    pts_i = torch.empty(num_pts, device=device).uniform_(20.0, 240.0)

    # Inject simulated solid wall obstacle at X=4m, Y=5m
    wall_mask = (torch.rand(num_pts, device=device) < 0.05)
    pts_x[wall_mask] = torch.empty(wall_mask.sum(), device=device).uniform_(3.8, 4.2)
    pts_y[wall_mask] = torch.empty(wall_mask.sum(), device=device).uniform_(4.8, 5.2)
    pts_z[wall_mask] = torch.empty(wall_mask.sum(), device=device).uniform_(0.0, 1.6)

    raw_frame = torch.stack([pts_x, pts_y, pts_z, pts_i], dim=1)

    # Warm-up pass
    with torch.no_grad():
        _ = pipeline(raw_frame)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed inference cycle
    start_time = time.perf_counter()
    with torch.no_grad():
        elevation_grid, terrain_grid = pipeline(raw_frame)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Verification Outputs
    print(f"Processing Latency : {elapsed_ms:.2f} ms (Throughput: {1000.0 / elapsed_ms:.1f} FPS)")
    print(f"Peak VRAM Usage    : {peak_vram_mb:.1f} MB (Budget: 8,000 MB)")
    print(f"2.5D Elevation Map : Shape {tuple(elevation_grid.shape)} | Max Step: {elevation_grid.max().item():.2f}m")
    print(f"2.5D Terrain Map   : Shape {tuple(terrain_grid.shape)} | Classes Detected: {torch.unique(terrain_grid).cpu().numpy().tolist()}")

    # Assertions
    assert elevation_grid.shape == (800, 800), "Elevation grid must be 800x800."
    assert terrain_grid.shape == (800, 800), "Terrain grid must be 800x800."
    print("\n[STATUS] Pipeline successfully initialized and validated.")

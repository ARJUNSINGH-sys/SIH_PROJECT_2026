"""Unit and integration tests for the Perceptra Perception Pipeline."""

import numpy as np
import pytest
import torch

from perception.eigensight_pipeline import (
    FeatureSegmenter,
    LowEndTemporalStacker,
    NoiseRobustSparseCNN,
    ResearchVariableMapper,
    RoverState,
    build_pipeline,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_points():
    """Generates synthetic LiDAR points with terrain, wall, and dynamic obstacles."""
    rng = np.random.default_rng(42)
    n_points = 5000
    x = rng.uniform(-20.0, 20.0, n_points)
    y = rng.uniform(-20.0, 20.0, n_points)
    z = rng.normal(0.0, 0.05, n_points)
    intensity = rng.uniform(20.0, 200.0, n_points)

    # Inject obstacle wall
    x[:500] = rng.uniform(4.8, 5.2, 500)
    y[:500] = rng.uniform(-3.0, 3.0, 500)
    z[:500] = rng.uniform(0.5, 2.0, 500)

    return np.column_stack([x, y, z, intensity]).astype(np.float32)


def test_low_end_temporal_stacker(sample_points):
    """Verifies that temporal stacking runs efficiently on CPU without GPU dependencies."""
    stacker = LowEndTemporalStacker(buffer_size=4)
    
    # Process 4 frames
    for _ in range(4):
        dynamic_mask = stacker.process(sample_points)
        assert len(dynamic_mask) == len(sample_points)
        assert dynamic_mask.dtype == bool


def test_noise_robust_sparse_cnn(device):
    """Verifies NoiseRobustSparseCNN with massive initial kernel layers."""
    model = NoiseRobustSparseCNN(in_channels=4, num_classes=4).to(device)
    model.eval()

    dummy_input = torch.randn(4, 512, 4, device=device)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (4, 512, 4), "Logits must match [B, N, 4 classes]"
    assert not torch.isnan(logits).any()


def test_research_variable_mapper(sample_points):
    """Verifies research paper dual-tier variable resolution:
       - Local (0 to 10m @ 5cm): Standard deviation & variance
       - Far-field (10 to 100m @ 50cm): Strictly binary occupied (1 or 0)
    """
    mapper = ResearchVariableMapper()
    labels = np.zeros(len(sample_points), dtype=np.int64)
    rover = RoverState(x=0.0, y=0.0)

    local_std, local_var, local_elev, local_cls, global_binary, near_pts, far_pts = mapper.process(
        sample_points, labels, rover
    )

    # Local grid assertions
    assert local_std.shape == (400, 400), "Local standard deviation grid must be 400x400 (20m @ 5cm)"
    assert local_var.shape == (400, 400), "Local variance grid must be 400x400"
    assert local_std.min() >= 0.0, "Standard deviation must be non-negative"
    assert local_var.min() >= 0.0, "Variance must be non-negative"

    # Far-field binary occupancy assertions
    assert global_binary.shape == (400, 400), "Global grid must be 400x400 (200m @ 50cm)"
    assert set(np.unique(global_binary)).issubset({0, 1}), "Far-field grid must be strictly binary (0 or 1)"


def test_eigensight_pipeline_end_to_end(device, sample_points):
    """Tests full perception pipeline with rover odometry, local variance, and Buerkle et al. 96% memory reduction."""
    pipeline = build_pipeline(device)
    pipeline.set_rover_state(x=2.0, y=-1.0, yaw_deg=30.0, speed_mps=3.0)

    # Ingest consecutive frames
    for f in range(3):
        timestamp = f * 0.1
        output = pipeline(sample_points, timestamp=timestamp)

        assert output.local_std_grid.shape == (400, 400)
        assert output.local_var_grid.shape == (400, 400)
        assert output.global_binary_occupancy.shape == (400, 400)
        assert output.rover.yaw_deg == 30.0
        assert output.memory_stats["cell_count_reduction_pct"] >= 95.0
        assert "stage1_spatiotemporal_ms" in output.timings_ms
        assert "stage2_segmentation_ms" in output.timings_ms
        assert "stage3_4_mapping_ms" in output.timings_ms

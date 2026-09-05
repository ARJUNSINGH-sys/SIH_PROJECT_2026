"""Example script demonstrating programmatic usage of PerceptionPipeline."""

import sys
from pathlib import Path
import numpy as np

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perception.pipeline import PerceptionPipeline
from perception.types import PerceptionConfig


def run_demo() -> None:
    config = PerceptionConfig()
    pipeline = PerceptionPipeline(config)

    # 1. Ingest synthetic point cloud
    rng = np.random.default_rng(0)
    sweep_1 = rng.uniform(-15.0, 15.0, (1000, 3))
    dogma_1 = pipeline.process_sweep(sweep_1, timestamp=0.0)
    print(f"Sweep 1 processed: {dogma_1.total_active_cells} DOGMa cells")

    sweep_2 = rng.uniform(-15.0, 15.0, (1000, 3))
    dogma_2 = pipeline.process_sweep(sweep_2, timestamp=0.1)
    print(f"Sweep 2 processed: {dogma_2.total_active_cells} DOGMa cells (Local: {dogma_2.local_active_cells}, Global: {dogma_2.global_active_cells})")


if __name__ == "__main__":
    run_demo()

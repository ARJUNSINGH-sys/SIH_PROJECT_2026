"""Top-level training launcher for NoiseRobustSparseCNN with 7-Fold Cross Validation.

Avoids Python 3.14 package namespace collision with stdlib 'types'.
"""

import sys
from pathlib import Path

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from perception.train_7000_gpu import train_7000_pictures

if __name__ == "__main__":
    train_7000_pictures(
        k_folds=7,
        epochs_per_fold=4,
        batch_size=64,
        lr=1e-3,
        output_dir="checkpoints",
    )

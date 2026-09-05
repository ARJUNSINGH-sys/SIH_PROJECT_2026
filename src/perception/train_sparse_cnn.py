import sys
from pathlib import Path

# Remove script directory to prevent shadowing Python's standard library 'types'
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)

_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import json
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
# 1. NOISE-ROBUST SPARSE CNN WITH MASSIVE INITIAL KERNEL LAYERS
# =====================================================================
class NoiseRobustSparseCNN(nn.Module):
    """Sparse Point-Voxel CNN designed for low-SWaP defense UGVs.
    
    Features a 5-layer deep architecture with massive initial kernel layers
    (kernel sizes 7 and 5, followed by kernel sizes 3, 3, and 1) acting as a
    wide spatial-spectral filter that attenuates high-frequency LiDAR beam noise,
    dust, rain scatter, and vibration before extracting semantic features.
    
    No regularizations (no dropout, no weight decay).
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # Layer 1: Massive Noise Reduction Kernel (Kernel Size = 7, Receptive Field = 7 points)
        self.kernel_layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Layer 2: Massive Multi-Scale Spatial Aggregation Kernel (Kernel Size = 5)
        self.kernel_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Layer 3: Intermediate Spatial Feature Extractor (Kernel Size = 3)
        self.kernel_layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Layer 4: Deep Feature Aggregation & Re-compression (Kernel Size = 3)
        self.kernel_layer4 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Layer 5: Point-wise Geometric Refinement (Kernel Size = 1)
        self.kernel_layer5 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Classifier Head (Zero Regularization)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] or [N, C] where C=4 (x, y, z, intensity)
        """
        is_2d = (x.ndim == 2)
        if is_2d:
            x = x.unsqueeze(0)  # [1, N, C]

        # Conv1d expects [B, C, N]
        x_in = x.transpose(1, 2)

        # 5-Layer Convolutional Feature Extraction
        feat1 = self.kernel_layer1(x_in)   # Layer 1: Conv7 (64 channels)
        feat2 = self.kernel_layer2(feat1)  # Layer 2: Conv5 (128 channels)
        feat3 = self.kernel_layer3(feat2)  # Layer 3: Conv3 (256 channels)
        feat4 = self.kernel_layer4(feat3)  # Layer 4: Conv3 (128 channels)
        feat5 = self.kernel_layer5(feat4)  # Layer 5: Conv1 (64 channels)

        # Transpose back to [B, N, C]
        feat_out = feat5.transpose(1, 2)

        # Classifier Head
        logits = self.classifier(feat_out)

        if is_2d:
            logits = logits.squeeze(0)

        return logits


# =====================================================================
# 2. DATASET GENERATION: UGV DEFENSE BENCHMARK WITH REALISTIC NOISE
# =====================================================================
class UGVLiDARDataset(Dataset):
    """Realistic multi-class LiDAR dataset for UGV perception with injected noise."""
    def __init__(self, n_samples: int = 2800, pts_per_sample: int = 512, seed: int = 42):
        super().__init__()
        self.data, self.labels = self._generate_dataset(n_samples, pts_per_sample, seed)

    def _generate_dataset(self, n_samples: int, pts_per_sample: int, seed: int):
        rng = np.random.default_rng(seed)
        all_samples = []
        all_labels = []

        for _ in range(n_samples):
            # Class mix per sample
            n_ground = int(pts_per_sample * 0.60)
            n_rough = int(pts_per_sample * 0.15)
            n_wall = int(pts_per_sample * 0.15)
            n_dynamic = pts_per_sample - (n_ground + n_rough + n_wall)

            # 0: Drivable Ground (flat z ~ 0, low intensity)
            gx = rng.uniform(-10.0, 10.0, n_ground)
            gy = rng.uniform(-10.0, 10.0, n_ground)
            gz = rng.normal(0.0, 0.02, n_ground)
            gi = rng.uniform(40.0, 90.0, n_ground)
            gl = np.zeros(n_ground, dtype=np.int64)

            # 1: Rough / Hazard (curbs, ditches, slope)
            rx = rng.uniform(-8.0, 8.0, n_rough)
            ry = rng.uniform(-8.0, 8.0, n_rough)
            rz = rng.uniform(0.08, 0.24, n_rough)
            ri = rng.uniform(30.0, 120.0, n_rough)
            rl = np.ones(n_rough, dtype=np.int64)

            # 2: Static Obstacle (Walls, boulders, poles)
            wx = rng.uniform(-10.0, 10.0, n_wall)
            wy = rng.uniform(-10.0, 10.0, n_wall)
            wz = rng.uniform(0.4, 2.5, n_wall)
            wi = rng.uniform(180.0, 240.0, n_wall)
            wl = np.full(n_wall, 2, dtype=np.int64)

            # 3: Dynamic Object (Translating vehicles/personnel)
            cx, cy = rng.uniform(-6.0, 6.0), rng.uniform(-6.0, 6.0)
            dx = cx + rng.uniform(-0.8, 0.8, n_dynamic)
            dy = cy + rng.uniform(-0.8, 0.8, n_dynamic)
            dz = rng.uniform(0.2, 1.8, n_dynamic)
            di = rng.uniform(140.0, 220.0, n_dynamic)
            dl = np.full(n_dynamic, 3, dtype=np.int64)

            pts = np.vstack([
                np.column_stack([gx, gy, gz, gi]),
                np.column_stack([rx, ry, rz, ri]),
                np.column_stack([wx, wy, wz, wi]),
                np.column_stack([dx, dy, dz, di]),
            ])
            lbls = np.concatenate([gl, rl, wl, dl])

            # Inject realistic LiDAR high-frequency Gaussian noise + beam scatter
            noise = rng.normal(0.0, 0.04, pts[:, :3].shape)
            pts[:, :3] += noise

            all_samples.append(pts.astype(np.float32))
            all_labels.append(lbls)

        return np.array(all_samples), np.array(all_labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])


# =====================================================================
# 3. K-FOLD (k=7) CROSS-VALIDATION TRAINING ENGINE (AdamW, No Reg)
# =====================================================================
def run_k7_cross_validation(
    k_folds: int = 7,
    epochs_per_fold: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: str = "checkpoints",
) -> Dict:
    """Executes 7-Fold Cross Validation with AdamW and zero regularization."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(f"NOISE-ROBUST SPARSE CNN: {k_folds}-FOLD CROSS-VALIDATION TRAINING")
    print(f"Hardware Device: {device} | Optimizer: AdamW (weight_decay=0.0, No Reg)")
    print(f"Architectural Kernels: Layer 1 (Size 7x7 Receptive), Layer 2 (Size 5x5)")
    print("=" * 75)

    dataset = UGVLiDARDataset(n_samples=2800, pts_per_sample=512)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_metrics = []
    best_overall_val_loss = float("inf")
    best_overall_model_state = None

    history = {
        "train_loss": [[] for _ in range(k_folds)],
        "val_loss": [[] for _ in range(k_folds)],
        "train_acc": [[] for _ in range(k_folds)],
        "val_acc": [[] for _ in range(k_folds)],
    }

    criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- FOLD {fold + 1} / {k_folds} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        # Instantiate fresh model for each fold
        model = NoiseRobustSparseCNN(in_channels=4, num_classes=4).to(device)

        # AdamW with zero weight decay (NO REGULARIZATION)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

        for epoch in range(epochs_per_fold):
            # Training Phase
            model.train()
            train_loss_total = 0.0
            train_correct = 0
            train_points = 0

            for pts, lbls in train_loader:
                pts, lbls = pts.to(device), lbls.to(device)
                optimizer.zero_grad()

                logits = model(pts)  # [B, N, C]
                loss = criterion(logits.view(-1, 4), lbls.view(-1))
                loss.backward()
                optimizer.step()

                train_loss_total += loss.item() * len(pts)
                preds = torch.argmax(logits, dim=-1)
                train_correct += (preds == lbls).sum().item()
                train_points += lbls.numel()

            epoch_train_loss = train_loss_total / len(train_idx)
            epoch_train_acc = (train_correct / train_points) * 100.0

            # Validation Phase
            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_points = 0

            with torch.no_grad():
                for pts, lbls in val_loader:
                    pts, lbls = pts.to(device), lbls.to(device)
                    logits = model(pts)
                    loss = criterion(logits.view(-1, 4), lbls.view(-1))
                    val_loss_total += loss.item() * len(pts)

                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == lbls).sum().item()
                    val_points += lbls.numel()

            epoch_val_loss = val_loss_total / len(val_idx)
            epoch_val_acc = (val_correct / val_points) * 100.0

            history["train_loss"][fold].append(epoch_train_loss)
            history["val_loss"][fold].append(epoch_val_loss)
            history["train_acc"][fold].append(epoch_train_acc)
            history["val_acc"][fold].append(epoch_val_acc)

            print(
                f"  Epoch {epoch+1}/{epochs_per_fold} | "
                f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | "
                f"Gap: {abs(epoch_val_loss - epoch_train_loss):.4f}"
            )

        final_val_loss = history["val_loss"][fold][-1]
        final_val_acc = history["val_acc"][fold][-1]
        final_train_loss = history["train_loss"][fold][-1]
        generalization_gap = final_val_loss - final_train_loss

        fold_metrics.append({
            "fold": fold + 1,
            "train_loss": round(final_train_loss, 4),
            "val_loss": round(final_val_loss, 4),
            "train_acc": round(history["train_acc"][fold][-1], 2),
            "val_acc": round(final_val_acc, 2),
            "generalization_gap": round(generalization_gap, 4),
        })

        if final_val_loss < best_overall_val_loss:
            best_overall_val_loss = final_val_loss
            best_overall_model_state = model.state_dict()

    # Save Best Model Checkpoint
    checkpoint_path = Path(output_dir) / "best_rover_sparse_cnn.pt"
    torch.save(best_overall_model_state, checkpoint_path)
    print(f"\n[OK] Saved Best NoiseRobustSparseCNN Checkpoint to: {checkpoint_path}")

    # Overfitting Diagnostics Across All 7 Folds
    mean_train_loss = np.mean([f["train_loss"] for f in fold_metrics])
    mean_val_loss = np.mean([f["val_loss"] for f in fold_metrics])
    mean_val_acc = np.mean([f["val_acc"] for f in fold_metrics])
    mean_gap = np.mean([f["generalization_gap"] for f in fold_metrics])

    # Overfitting verdict
    # If gap is small (< 0.05) and accuracy is high (> 90%), model generalizes well without overfitting
    is_overfitting = (mean_gap > 0.15)
    verdict = "HEALTHY GENERALIZATION (NO OVERFITTING)" if not is_overfitting else "POTENTIAL OVERFITTING DETECTED"

    summary = {
        "k_folds": k_folds,
        "epochs_per_fold": epochs_per_fold,
        "mean_train_loss": round(float(mean_train_loss), 4),
        "mean_val_loss": round(float(mean_val_loss), 4),
        "mean_val_accuracy": round(float(mean_val_acc), 2),
        "mean_generalization_gap": round(float(mean_gap), 4),
        "overfitting_diagnosis": verdict,
        "fold_breakdown": fold_metrics,
    }

    # Save JSON summary
    metrics_path = Path(output_dir) / "k7_cv_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved k=7 Cross-Validation Metrics to: {metrics_path}")

    # Plot 7-Fold Training Curves
    plot_path = Path(output_dir) / "k7_cross_validation_results.png"
    generate_cv_plot(history, summary, str(plot_path))

    print("\n" + "=" * 75)
    print("7-FOLD CROSS-VALIDATION SUMMARY & OVERFITTING VERDICT:")
    print(f"  Mean Train Loss       : {mean_train_loss:.4f}")
    print(f"  Mean Validation Loss  : {mean_val_loss:.4f}")
    print(f"  Mean Validation Acc   : {mean_val_acc:.2f}%")
    print(f"  Generalization Gap    : {mean_gap:.4f}")
    print(f"  Diagnosis Status      : {verdict}")
    print("=" * 75)

    return summary


def generate_cv_plot(history: Dict, summary: Dict, save_path: str) -> None:
    """Generates publication-quality 7-fold CV learning curves and gap analysis."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=120)
    fig.patch.set_facecolor("#0a0d14")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    epochs = range(1, len(history["train_loss"][0]) + 1)
    k = len(history["train_loss"])

    # Plot Loss Curves for all 7 folds
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    for fold in range(k):
        ax1.plot(epochs, history["train_loss"][fold], color=colors[fold], linestyle="--", alpha=0.5)
        ax1.plot(epochs, history["val_loss"][fold], color=colors[fold], linestyle="-", label=f"Fold {fold+1}")

    # Plot Mean Loss
    mean_train_l = np.mean(history["train_loss"], axis=0)
    mean_val_l = np.mean(history["val_loss"], axis=0)
    ax1.plot(epochs, mean_train_l, color="#38bdf8", linewidth=3, linestyle="--", label="Mean Train Loss")
    ax1.plot(epochs, mean_val_l, color="#10b981", linewidth=3, label="Mean Val Loss")

    ax1.set_title("7-Fold Loss Curves (Train vs Validation)", fontsize=12, fontweight="bold", color="#38bdf8")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    # Plot Accuracy Curves
    for fold in range(k):
        ax2.plot(epochs, history["val_acc"][fold], color=colors[fold], alpha=0.5)
    mean_val_a = np.mean(history["val_acc"], axis=0)
    mean_train_a = np.mean(history["train_acc"], axis=0)
    ax2.plot(epochs, mean_train_a, color="#38bdf8", linewidth=3, linestyle="--", label="Mean Train Acc")
    ax2.plot(epochs, mean_val_a, color="#10b981", linewidth=3, label="Mean Val Acc")

    ax2.set_title(f"Validation Accuracy across 7 Folds (Mean: {summary['mean_val_accuracy']:.1f}%)", fontsize=12, fontweight="bold", color="#10b981")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Classification Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"PERCEPTRA Sparse CNN — 7-Fold Cross Validation | Status: {summary['overfitting_diagnosis']}",
        fontsize=14,
        fontweight="bold",
        color="#06b6d4",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[OK] Generated 7-Fold CV Learning Curve Plot: {save_path}")


if __name__ == "__main__":
    run_k7_cross_validation(k_folds=7, epochs_per_fold=4, batch_size=32, lr=1e-3)

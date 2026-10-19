"""
FraudMap Model Trainer

Handles GNN model training with:
- Weighted BCE loss for class imbalance
- Temporal cross-validation
- Early stopping
- Checkpoint saving
- Training metrics logging
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from src.model.fraudmap import FraudMapGNN
from src.core.config import settings
from src.core.logger import get_logger
from src.core.constants import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BATCH_SIZE,
    GRAPHSAGE_SAMPLE_SIZES,
)

logger = get_logger(__name__)


class FraudMapTrainer:
    """Trainer for the FraudMap hybrid GNN model.

    Implements the training loop with:
    - NeighborLoader for mini-batch training with neighbor sampling
    - Weighted Binary Cross-Entropy for class imbalance
    - Early stopping on validation F1 score
    - Model checkpoint saving

    Args:
        model: FraudMapGNN model instance.
        device: Training device (cuda/cpu).
        lr: Learning rate.
        weight_decay: L2 regularization.
        pos_weight: Weight for positive (fraud) class in BCE loss.
    """

    def __init__(
        self,
        model: Optional[FraudMapGNN] = None,
        device: Optional[str] = None,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        pos_weight: float = 10.0,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = (model or FraudMapGNN()).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )
        self.criterion = nn.BCELoss(
            weight=None  # We use pos_weight manually for flexibility
        )
        self.pos_weight = pos_weight

        # Training state
        self.best_f1 = 0.0
        self.patience_counter = 0

        logger.info(
            "trainer_initialized",
            device=self.device,
            model_params=sum(p.numel() for p in self.model.parameters()),
            lr=lr,
            pos_weight=pos_weight,
        )

    def train(
        self,
        data: Data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        max_epochs: int = MAX_EPOCHS,
        patience: int = EARLY_STOPPING_PATIENCE,
        checkpoint_dir: Optional[Path] = None,
    ) -> dict:
        """Train the model.

        Args:
            data: PyG Data object with x, edge_index, y.
            train_mask: Boolean mask for training nodes.
            val_mask: Boolean mask for validation nodes.
            max_epochs: Maximum training epochs.
            patience: Early stopping patience.
            checkpoint_dir: Directory to save model checkpoints.

        Returns:
            Dictionary with training history.
        """
        checkpoint_dir = checkpoint_dir or settings.model_checkpoint_dir
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        data = data.to(self.device)
        history = {"train_loss": [], "val_f1": [], "val_roc_auc": []}

        # Create NeighborLoader for mini-batch training
        train_loader = NeighborLoader(
            data,
            num_neighbors=GRAPHSAGE_SAMPLE_SIZES,
            batch_size=BATCH_SIZE,
            input_nodes=train_mask,
            shuffle=True,
        )

        logger.info(
            "training_started",
            num_train=train_mask.sum().item(),
            num_val=val_mask.sum().item(),
            max_epochs=max_epochs,
        )

        for epoch in range(1, max_epochs + 1):
            t_start = time.time()

            # --- Training ---
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # --- Validation ---
            val_metrics = self._evaluate(data, val_mask)
            history["val_f1"].append(val_metrics["f1"])
            history["val_roc_auc"].append(val_metrics["roc_auc"])

            self.scheduler.step(val_metrics["f1"])

            elapsed = time.time() - t_start
            logger.info(
                "epoch_complete",
                epoch=epoch,
                train_loss=f"{train_loss:.4f}",
                val_f1=f"{val_metrics['f1']:.4f}",
                val_roc_auc=f"{val_metrics['roc_auc']:.4f}",
                val_precision=f"{val_metrics['precision']:.4f}",
                val_recall=f"{val_metrics['recall']:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.6f}",
                elapsed=f"{elapsed:.1f}s",
            )

            # --- Early stopping ---
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self.patience_counter = 0
                self._save_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    epoch,
                    val_metrics,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        "early_stopping",
                        epoch=epoch,
                        best_f1=f"{self.best_f1:.4f}",
                    )
                    break

        logger.info("training_complete", best_f1=f"{self.best_f1:.4f}")
        return history

    def _train_epoch(self, train_loader: NeighborLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch.x, batch.edge_index)
            y = batch.y[: batch.batch_size].float().unsqueeze(1)
            pred = out[: batch.batch_size]

            # Weighted loss for class imbalance
            weight = torch.where(y == 1, self.pos_weight, 1.0)
            loss = nn.functional.binary_cross_entropy(pred, y, weight=weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self, data: Data, mask: torch.Tensor) -> dict:
        """Evaluate model on masked nodes."""
        self.model.eval()

        out = self.model(data.x, data.edge_index)
        pred_probs = out[mask].cpu().numpy().flatten()
        pred_labels = (pred_probs >= 0.5).astype(int)
        true_labels = data.y[mask].cpu().numpy()

        metrics = {
            "precision": precision_score(true_labels, pred_labels, zero_division=0),
            "recall": recall_score(true_labels, pred_labels, zero_division=0),
            "f1": f1_score(true_labels, pred_labels, zero_division=0),
            "roc_auc": roc_auc_score(true_labels, pred_probs)
            if len(set(true_labels)) > 1
            else 0.0,
        }
        return metrics

    def _save_checkpoint(
        self, path: Path, epoch: int, metrics: dict
    ) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "best_f1": self.best_f1,
            },
            path,
        )
        logger.info("checkpoint_saved", path=str(path), epoch=epoch)

    def load_checkpoint(self, path: Path) -> dict:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_f1 = checkpoint.get("best_f1", 0.0)
        logger.info(
            "checkpoint_loaded",
            path=str(path),
            epoch=checkpoint["epoch"],
            best_f1=f"{self.best_f1:.4f}",
        )
        return checkpoint["metrics"]


if __name__ == "__main__":
    from src.core.logger import setup_logging

    setup_logging(log_level="INFO")
    logger.info("FraudMap Trainer - use 'make train' or run data pipeline first")

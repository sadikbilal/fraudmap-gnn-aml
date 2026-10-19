"""
Download and prepare the Elliptic Bitcoin Dataset.

Dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- 203,769 transaction nodes
- 234,355 directed edges
- 166-dimensional feature vectors per node
- Labels: illicit (fraud), licit (legit), unknown
- 49 time steps

This script downloads the dataset from Kaggle (requires kaggle API key)
or provides instructions for manual download.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logger import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "elliptic"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "elliptic"


def download_from_kaggle():
    """Download Elliptic dataset using Kaggle API."""
    try:
        import kaggle

        logger.info("downloading_from_kaggle")
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        kaggle.api.dataset_download_files(
            "ellipticco/elliptic-data-set",
            path=str(RAW_DIR),
            unzip=True,
        )
        logger.info("download_complete", path=str(RAW_DIR))
        return True
    except Exception as e:
        logger.warning("kaggle_download_failed", error=str(e))
        return False


def check_manual_download() -> bool:
    """Check if dataset files exist (manual download)."""
    expected_files = [
        "elliptic_txs_features.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_classes.csv",
    ]
    for f in expected_files:
        # Check both directly and in subdirectory
        if not (RAW_DIR / f).exists():
            # Also check one level deeper (Kaggle sometimes nests)
            nested = list(RAW_DIR.rglob(f))
            if not nested:
                return False
    return True


def find_file(name: str) -> Path:
    """Find a file in the raw directory, checking subdirectories."""
    direct = RAW_DIR / name
    if direct.exists():
        return direct
    nested = list(RAW_DIR.rglob(name))
    if nested:
        return nested[0]
    raise FileNotFoundError(f"Cannot find {name} in {RAW_DIR}")


def preprocess_elliptic():
    """Preprocess Elliptic dataset into train-ready format.

    Creates:
    - node_features.pt: Tensor [num_nodes, 166]
    - edge_index.pt: Tensor [2, num_edges]
    - labels.pt: Tensor [num_nodes] (0=unknown, 1=illicit, 2=licit)
    - time_steps.pt: Tensor [num_nodes] (1-49)
    - node_mapping.json: txId → index mapping
    """
    logger.info("preprocessing_elliptic")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load features ---
    features_path = find_file("elliptic_txs_features.csv")
    logger.info("loading_features", path=str(features_path))
    features_df = pd.read_csv(features_path, header=None)

    # First column is txId, second is time step, rest are features
    tx_ids = features_df.iloc[:, 0].values
    time_steps = features_df.iloc[:, 1].values.astype(np.int32)
    features = features_df.iloc[:, 2:].values.astype(np.float32)

    # Create txId → index mapping
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    logger.info(
        "features_loaded",
        num_nodes=len(tx_ids),
        feature_dim=features.shape[1],
        time_steps=f"{time_steps.min()}-{time_steps.max()}",
    )

    # --- Load edges ---
    edges_path = find_file("elliptic_txs_edgelist.csv")
    logger.info("loading_edges", path=str(edges_path))
    edges_df = pd.read_csv(edges_path)

    # Map txIds to indices
    valid_edges = edges_df[
        edges_df.iloc[:, 0].isin(tx_to_idx) & edges_df.iloc[:, 1].isin(tx_to_idx)
    ]
    source = valid_edges.iloc[:, 0].map(tx_to_idx).values
    target = valid_edges.iloc[:, 1].map(tx_to_idx).values

    logger.info(
        "edges_loaded",
        num_edges=len(source),
        dropped=len(edges_df) - len(valid_edges),
    )

    # --- Load labels ---
    classes_path = find_file("elliptic_txs_classes.csv")
    logger.info("loading_labels", path=str(classes_path))
    classes_df = pd.read_csv(classes_path)

    # Map: "unknown" → 0, "1" (illicit) → 1, "2" (licit) → 2
    labels = np.zeros(len(tx_ids), dtype=np.int64)
    label_map = {"unknown": 0, "1": 1, "2": 2}

    for _, row in classes_df.iterrows():
        tx_id = row.iloc[0]
        label_str = str(row.iloc[1]).strip()
        if tx_id in tx_to_idx:
            labels[tx_to_idx[tx_id]] = label_map.get(label_str, 0)

    num_illicit = (labels == 1).sum()
    num_licit = (labels == 2).sum()
    num_unknown = (labels == 0).sum()
    logger.info(
        "labels_loaded",
        illicit=int(num_illicit),
        licit=int(num_licit),
        unknown=int(num_unknown),
        fraud_ratio=f"{num_illicit / (num_illicit + num_licit):.4f}",
    )

    # --- Save as PyTorch tensors ---
    import torch
    import json

    torch.save(torch.tensor(features), PROCESSED_DIR / "node_features.pt")
    torch.save(
        torch.tensor(np.stack([source, target]), dtype=torch.long),
        PROCESSED_DIR / "edge_index.pt",
    )
    torch.save(torch.tensor(labels), PROCESSED_DIR / "labels.pt")
    torch.save(torch.tensor(time_steps), PROCESSED_DIR / "time_steps.pt")

    # Save mapping as JSON
    with open(PROCESSED_DIR / "node_mapping.json", "w") as f:
        json.dump({str(k): int(v) for k, v in tx_to_idx.items()}, f)

    logger.info(
        "preprocessing_complete",
        output_dir=str(PROCESSED_DIR),
        files=[
            "node_features.pt",
            "edge_index.pt",
            "labels.pt",
            "time_steps.pt",
            "node_mapping.json",
        ],
    )


def main():
    """Main entry point."""
    print("=" * 60)
    print("  FraudMap - Elliptic Dataset Download & Preprocessing")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Try Kaggle API first
    if not check_manual_download():
        success = download_from_kaggle()
        if not success:
            print("\n⚠️  Kaggle API download failed!")
            print("   Please download manually from:")
            print("   https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
            print(f"\n   Extract files to: {RAW_DIR}")
            print("   Expected files:")
            print("   - elliptic_txs_features.csv")
            print("   - elliptic_txs_edgelist.csv")
            print("   - elliptic_txs_classes.csv")
            print("\n   Then run this script again.")
            sys.exit(1)

    if check_manual_download():
        preprocess_elliptic()
        print("\n✅ Elliptic dataset is ready!")
        print(f"   Processed files: {PROCESSED_DIR}")
    else:
        print(f"\n❌ Dataset files not found in {RAW_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()

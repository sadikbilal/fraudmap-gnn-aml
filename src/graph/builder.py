"""
FraudMap Graph Builder

Converts transaction records into PyTorch Geometric graph format.
Creates node features (24-dim account vectors) and edge features (4-dim).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Optional

from src.core.logger import get_logger
from src.core.constants import NODE_FEATURE_DIM, EDGE_FEATURE_DIM

logger = get_logger(__name__)


class TransactionGraphBuilder:
    """Builds PyG graph data from transaction DataFrames.

    The financial transaction network is modeled as a directed graph G = (V, E):
    - V (nodes) = accounts
    - E (edges) = directed transfers between accounts

    Each node has a 24-dimensional feature vector.
    Each edge has a 4-dimensional feature vector.
    """

    def __init__(self):
        self._account_to_idx: dict[str, int] = {}

    def build_from_dataframe(
        self,
        transactions_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> Data:
        """Build a PyG Data object from transactions.

        Args:
            transactions_df: DataFrame with columns:
                - source_account: str
                - target_account: str
                - amount: float
                - timestamp: datetime/int
                - transaction_type: str (optional)
                - channel: str (optional)
            labels_df: Optional DataFrame with columns:
                - account: str
                - label: int (0=legit, 1=fraud)

        Returns:
            PyG Data object with x, edge_index, edge_attr, and optionally y.
        """
        logger.info(
            "building_graph",
            num_transactions=len(transactions_df),
        )

        # Build node mapping
        accounts = set(transactions_df["source_account"]) | set(
            transactions_df["target_account"]
        )
        self._account_to_idx = {acc: idx for idx, acc in enumerate(sorted(accounts))}
        num_nodes = len(self._account_to_idx)

        # Build edge index (source → target directed edges)
        source_indices = transactions_df["source_account"].map(self._account_to_idx).values
        target_indices = transactions_df["target_account"].map(self._account_to_idx).values
        edge_index = torch.tensor(
            np.stack([source_indices, target_indices]), dtype=torch.long
        )

        # Build edge features (4-dim: amount, time_delta, frequency, direction)
        edge_attr = self._compute_edge_features(transactions_df)

        # Build node features (24-dim account vectors)
        node_features = self._compute_node_features(transactions_df, num_nodes)

        # Build labels if provided
        y = None
        if labels_df is not None:
            y = torch.zeros(num_nodes, dtype=torch.long)
            for _, row in labels_df.iterrows():
                if row["account"] in self._account_to_idx:
                    idx = self._account_to_idx[row["account"]]
                    y[idx] = int(row["label"])

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes,
        )

        logger.info(
            "graph_built",
            num_nodes=num_nodes,
            num_edges=edge_index.shape[1],
            node_feature_dim=node_features.shape[1],
            edge_feature_dim=edge_attr.shape[1],
            has_labels=y is not None,
        )

        return data

    def _compute_node_features(
        self, transactions_df: pd.DataFrame, num_nodes: int
    ) -> torch.Tensor:
        """Compute 24-dimensional feature vector per account.

        Features (per account):
        [0-2]   Transaction count: 24h, 7d, 30d
        [3-5]   Total amount: 24h, 7d, 30d
        [6-8]   Mean amount: 24h, 7d, 30d
        [9-11]  Std amount: 24h, 7d, 30d
        [12]    After-midnight ratio (00:00-06:00 transactions / total)
        [13]    Weekend ratio
        [14]    Account age (days since first transaction)
        [15]    Average balance proxy (mean outflow - mean inflow)
        [16-18] Unique counterparties: 24h, 7d, 30d
        [19]    Channel diversity (number of unique channels)
        [20]    Geographic spread (number of unique locations)
        [21]    Max single transaction ratio (max amount / total amount)
        [22]    In-degree (number of incoming transfers)
        [23]    Out-degree (number of outgoing transfers)
        """
        features = torch.zeros(num_nodes, NODE_FEATURE_DIM, dtype=torch.float32)

        # Group by source and target for efficient computation
        if "timestamp" in transactions_df.columns:
            df = transactions_df.copy()

            # Ensure numeric timestamp
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                max_ts = df["timestamp"].max()
                df["ts_seconds"] = (max_ts - df["timestamp"]).dt.total_seconds()
            else:
                max_ts = df["timestamp"].max()
                df["ts_seconds"] = max_ts - df["timestamp"]
        else:
            df = transactions_df.copy()
            df["ts_seconds"] = 0

        # Compute features for each account
        for account, idx in self._account_to_idx.items():
            # Outgoing transactions
            out_mask = df["source_account"] == account
            out_df = df[out_mask]

            # Incoming transactions
            in_mask = df["target_account"] == account
            in_df = df[in_mask]

            # All transactions involving this account
            all_df = pd.concat([out_df, in_df])

            if len(all_df) == 0:
                continue

            # Time-windowed features
            for w_idx, (window_name, window_sec) in enumerate(
                [("24h", 86400), ("7d", 604800), ("30d", 2592000)]
            ):
                w_mask = all_df["ts_seconds"] <= window_sec
                w_df = all_df[w_mask]
                if len(w_df) > 0:
                    features[idx, w_idx] = len(w_df)                      # count
                    features[idx, w_idx + 3] = w_df["amount"].sum()       # total amount
                    features[idx, w_idx + 6] = w_df["amount"].mean()      # mean amount
                    features[idx, w_idx + 9] = w_df["amount"].std() if len(w_df) > 1 else 0

                    # Unique counterparties
                    out_w = out_df[out_df["ts_seconds"] <= window_sec]
                    in_w = in_df[in_df["ts_seconds"] <= window_sec]
                    counterparties = set()
                    if len(out_w) > 0:
                        counterparties |= set(out_w["target_account"])
                    if len(in_w) > 0:
                        counterparties |= set(in_w["source_account"])
                    features[idx, w_idx + 16] = len(counterparties)

            # After-midnight ratio
            if "hour" in all_df.columns or "timestamp" in all_df.columns:
                try:
                    if "hour" not in all_df.columns:
                        hours = pd.to_datetime(all_df["timestamp"]).dt.hour
                    else:
                        hours = all_df["hour"]
                    night_count = ((hours >= 0) & (hours < 6)).sum()
                    features[idx, 12] = night_count / max(len(all_df), 1)
                except (TypeError, AttributeError):
                    pass

            # Weekend ratio
            try:
                if "day_of_week" not in all_df.columns:
                    days = pd.to_datetime(all_df["timestamp"]).dt.dayofweek
                else:
                    days = all_df["day_of_week"]
                weekend = (days >= 5).sum()
                features[idx, 13] = weekend / max(len(all_df), 1)
            except (TypeError, AttributeError):
                pass

            # Account age
            features[idx, 14] = all_df["ts_seconds"].max() / 86400 if len(all_df) > 0 else 0

            # Balance proxy
            total_out = out_df["amount"].sum() if len(out_df) > 0 else 0
            total_in = in_df["amount"].sum() if len(in_df) > 0 else 0
            features[idx, 15] = total_in - total_out

            # Channel diversity
            if "channel" in all_df.columns:
                features[idx, 19] = all_df["channel"].nunique()

            # Geographic spread
            if "location_code" in all_df.columns:
                features[idx, 20] = all_df["location_code"].nunique()

            # Max single transaction ratio
            total_amount = all_df["amount"].sum()
            if total_amount > 0:
                features[idx, 21] = all_df["amount"].max() / total_amount

            # Degree features
            features[idx, 22] = len(in_df)    # in-degree
            features[idx, 23] = len(out_df)   # out-degree

        # Normalize features
        features = self._normalize_features(features)

        return features

    def _compute_edge_features(self, transactions_df: pd.DataFrame) -> torch.Tensor:
        """Compute 4-dimensional edge feature vectors.

        Features per edge:
        [0] Normalized amount (log-scaled)
        [1] Time delta from reference point (normalized)
        [2] Frequency (number of transfers between same pair)
        [3] Direction indicator (1.0 for this direction)
        """
        n_edges = len(transactions_df)
        edge_attr = torch.zeros(n_edges, EDGE_FEATURE_DIM, dtype=torch.float32)

        # Amount (log-normalized)
        amounts = transactions_df["amount"].values.astype(np.float64)
        log_amounts = np.log1p(amounts)
        if log_amounts.max() > 0:
            edge_attr[:, 0] = torch.tensor(log_amounts / log_amounts.max(), dtype=torch.float32)

        # Time delta (normalized)
        if "timestamp" in transactions_df.columns:
            if pd.api.types.is_datetime64_any_dtype(transactions_df["timestamp"]):
                ts = (
                    transactions_df["timestamp"] - transactions_df["timestamp"].min()
                ).dt.total_seconds().values
            else:
                ts = transactions_df["timestamp"].values.astype(np.float64)
                ts = ts - ts.min()
            if ts.max() > 0:
                edge_attr[:, 1] = torch.tensor(ts / ts.max(), dtype=torch.float32)

        # Frequency (transfers between same pair)
        pair_counts = transactions_df.groupby(
            ["source_account", "target_account"]
        ).transform("count")["amount"].values
        max_freq = pair_counts.max() if pair_counts.max() > 0 else 1
        edge_attr[:, 2] = torch.tensor(pair_counts / max_freq, dtype=torch.float32)

        # Direction indicator
        edge_attr[:, 3] = 1.0

        return edge_attr

    @staticmethod
    def _normalize_features(features: torch.Tensor) -> torch.Tensor:
        """Normalize features to zero mean and unit variance."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        std[std == 0] = 1.0  # Avoid division by zero
        return (features - mean) / std

    @property
    def account_mapping(self) -> dict[str, int]:
        """Return the account-to-index mapping."""
        return self._account_to_idx.copy()

    @property
    def num_accounts(self) -> int:
        """Return the number of unique accounts."""
        return len(self._account_to_idx)

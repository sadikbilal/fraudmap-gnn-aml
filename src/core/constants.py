"""
FraudMap Constants

Shared constants used across the application.
"""

# =============================================================================
# Feature Engineering
# =============================================================================

# Number of features per account node (24-dimensional vector as per report)
NODE_FEATURE_DIM = 24

# Edge feature dimensions (amount, time_delta, frequency, direction)
EDGE_FEATURE_DIM = 4

# Time windows for feature computation
TIME_WINDOWS = {
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
    "30d": 30 * 24 * 3600,
}

# =============================================================================
# Model Architecture (from project report)
# =============================================================================

# GraphSAGE neighbor sampling sizes per layer
GRAPHSAGE_SAMPLE_SIZES = [25, 10, 5]  # 3 layers

# GAT attention heads
GAT_NUM_HEADS = 8

# Hidden dimensions
HIDDEN_DIM = 256
CLASSIFIER_DIMS = [256, 128]  # Two FC layers

# Dropout rate
DROPOUT_RATE = 0.3

# =============================================================================
# Training
# =============================================================================

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 512

# =============================================================================
# AML Scenario Types
# =============================================================================

AML_SCENARIOS = [
    "smurfing",        # Fan-out/fan-in (N→1 or 1→N)
    "layering",        # Chain transfers (A→B→C→...→Z)
    "structuring",     # Below-threshold periodic transfers
    "round_trip",      # Circular flow (A→B→C→A)
    "scatter_gather",  # 1→N→1 pattern
    "mutualistic",     # High-frequency mutual transfers
]

# =============================================================================
# Risk Scoring
# =============================================================================

FRAUD_THRESHOLD = 0.85          # Alert generation threshold
RISK_LEVELS = {
    "low": (0.0, 0.3),
    "medium": (0.3, 0.6),
    "high": (0.6, 0.85),
    "critical": (0.85, 1.0),
}

# =============================================================================
# Transaction Types & Channels (Turkey-specific)
# =============================================================================

TRANSACTION_TYPES = ["EFT", "havale", "odeme", "ATM", "kripto_cikis"]
CHANNELS = ["mobile", "internet", "sube", "atm"]

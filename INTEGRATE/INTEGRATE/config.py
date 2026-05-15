"""
config.py — Centralised hyper-parameter configuration
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_ROOT = "./Dataset_64"   # ← change to your dataset root
IMG_SIZE     = 64

# ── Model ─────────────────────────────────────────────────────────────────────
LATENT_CHANNELS = 16        # Controls bandwidth ratio k/n (higher → more symbols)
CHANNEL_TYPE    = "awgn"    # "awgn" | "rayleigh"
SNR_DB          = 10.0      # Training SNR in dB

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 128
EPOCHS       = 100
LR           = 1e-3
LR_MIN       = 1e-5
WEIGHT_DECAY = 0.0
NUM_WORKERS  = 4

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "models/conv_ae_64_10_tf"

# ── Evaluation ────────────────────────────────────────────────────────────────
SNR_SWEEP_RANGE = list(range(0, 21, 2))   # [0, 2, 4, ..., 20] dB
EVAL_N_IMAGES   = 8

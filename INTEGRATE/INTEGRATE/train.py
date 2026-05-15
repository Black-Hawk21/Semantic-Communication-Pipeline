"""
train.py — Launch training for Deep JSCC / CAE

Usage:
    python train.py
    python train.py --resume runs/exp_awgn_snr10/ckpt_last.pt
"""

import argparse
import torch

import config
from model   import DeepJSCC
from dataset import get_dataloader
from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume",          type=str,   default=None)
    p.add_argument("--latent_channels", type=int,   default=config.LATENT_CHANNELS)
    p.add_argument("--channel",         type=str,   default=config.CHANNEL_TYPE)
    p.add_argument("--snr_db",          type=float, default=config.SNR_DB)
    p.add_argument("--epochs",          type=int,   default=config.EPOCHS)
    p.add_argument("--batch_size",      type=int,   default=config.BATCH_SIZE)
    p.add_argument("--output_dir",      type=str,   default=config.OUTPUT_DIR)
    return p.parse_args()


def run_training(args):
    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────
    train_loader = get_dataloader(
        root        = config.DATASET_ROOT,
        split       = "train",
        batch_size  = args.batch_size,
        num_workers = config.NUM_WORKERS,
    )
    val_loader = get_dataloader(
        root        = config.DATASET_ROOT,
        split       = "val",
        batch_size  = args.batch_size,
        num_workers = config.NUM_WORKERS,
    )

    # ── Model ────────────────────────────────
    model = DeepJSCC(
        latent_channels = args.latent_channels,
        channel_type    = args.channel,
        snr_db          = args.snr_db,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Bandwidth ratio k/n = {model.bandwidth_ratio():.4f}")

    # ── Trainer ──────────────────────────────
    cfg = {
        "lr":           config.LR,
        "lr_min":       config.LR_MIN,
        "weight_decay": config.WEIGHT_DECAY,
        "epochs":       args.epochs,
    }
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
        output_dir   = args.output_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    run_training(parse_args())

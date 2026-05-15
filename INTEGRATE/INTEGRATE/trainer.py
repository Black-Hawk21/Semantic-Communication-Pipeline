"""
trainer.py — Training and evaluation engine for Deep JSCC / CAE
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model  import DeepJSCC
from losses import MSELoss, compute_psnr, compute_ssim


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Manages the full training loop for DeepJSCC.

    Args:
        model        : DeepJSCC instance.
        train_loader : DataLoader for training set.
        val_loader   : DataLoader for validation set.
        cfg          : Dict of hyper-parameters (see TrainingConfig).
        device       : 'cuda' | 'cpu' | 'mps'.
        output_dir   : Where to save checkpoints and TensorBoard logs.
    """

    def __init__(
        self,
        model:        DeepJSCC,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        device:       str  = "cuda",
        output_dir:   str  = "runs/exp",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = MSELoss()

        # Optimiser
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr           = cfg.get("lr", 1e-3),
            weight_decay = cfg.get("weight_decay", 0.0),
        )

        # LR scheduler — cosine anneal
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max  = cfg.get("epochs", 100),
            eta_min= cfg.get("lr_min", 1e-5),
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

        self.start_epoch = 0
        self.best_val_psnr = -float("inf")

    # ── Checkpoint helpers ────────────────────

    def save_checkpoint(self, epoch: int, tag: str = "last"):
        path = self.output_dir / f"ckpt_{tag}.pt"
        torch.save({
            "epoch":          epoch,
            "model":          self.model.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
            "best_val_psnr":  self.best_val_psnr,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch   = ckpt["epoch"] + 1
        self.best_val_psnr = ckpt.get("best_val_psnr", -float("inf"))
        print(f"[Trainer] Resumed from epoch {ckpt['epoch']} | best PSNR {self.best_val_psnr:.2f} dB")

    # ── One epoch ─────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
        for batch_idx, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            x_hat, _ = self.model(imgs)
            loss = self.criterion(x_hat, imgs)
            loss.backward()

            # Gradient clipping (optional but stabilising)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        elapsed  = time.time() - t0
        return {"train/loss": avg_loss, "train/time_s": elapsed}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = total_psnr = total_ssim = 0.0
        n_batches  = len(self.val_loader)

        for imgs, _ in self.val_loader:
            imgs  = imgs.to(self.device, non_blocking=True)
            x_hat, _ = self.model(imgs)

            total_loss += self.criterion(x_hat, imgs).item()
            total_psnr += compute_psnr(x_hat, imgs)
            total_ssim += compute_ssim(x_hat, imgs)

        return {
            "val/loss": total_loss / n_batches,
            "val/psnr": total_psnr / n_batches,
            "val/ssim": total_ssim / n_batches,
        }

    # ── Full training loop ────────────────────

    def fit(self, epochs: Optional[int] = None):
        epochs = epochs or self.cfg.get("epochs", 100)

        print(f"\n{'='*60}")
        print(f"  DeepJSCC Training")
        print(f"  Channel : {self.model.channel_type.upper()}  |  "
              f"SNR : {self.model.snr_db} dB  |  "
              f"k/n : {self.model.bandwidth_ratio():.4f}")
        print(f"  Epochs  : {epochs}  |  Device : {self.device}")
        print(f"{'='*60}\n")

        epoch_pbar = tqdm(range(self.start_epoch, epochs), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)
            self.scheduler.step()

            # ── Log ──────────────────────────
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["lr"] = self.scheduler.get_last_lr()[0]

            epoch_pbar.set_postfix(
                loss=f"{train_metrics['train/loss']:.4f}",
                val_loss=f"{val_metrics['val/loss']:.4f}",
                psnr=f"{val_metrics['val/psnr']:.2f}",
                ssim=f"{val_metrics['val/ssim']:.4f}",
            )

            for k, v in all_metrics.items():
                self.writer.add_scalar(k, v, global_step=epoch)

            print(
                f"Epoch [{epoch+1:03d}/{epochs}]  "
                f"Loss: {train_metrics['train/loss']:.4f}  "
                f"Val-Loss: {val_metrics['val/loss']:.4f}  "
                f"PSNR: {val_metrics['val/psnr']:.2f} dB  "
                f"SSIM: {val_metrics['val/ssim']:.4f}  "
                f"LR: {all_metrics['lr']:.2e}"
            )

            # ── Checkpoints ──────────────────
            self.save_checkpoint(epoch, tag="last")
            if val_metrics["val/psnr"] > self.best_val_psnr:
                self.best_val_psnr = val_metrics["val/psnr"]
                self.save_checkpoint(epoch, tag="best")
                print(f"  ✓ New best PSNR: {self.best_val_psnr:.2f} dB — saved.")

        self.writer.close()
        print("\nTraining complete.")

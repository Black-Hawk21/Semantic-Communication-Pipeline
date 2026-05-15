"""
evaluate.py — Evaluation utilities for Deep JSCC / CAE

Features:
  - Single-SNR evaluation (PSNR + SSIM)
  - SNR sweep (PSNR & SSIM vs SNR curve)
  - Visual comparison grid (original vs reconstructed)
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import DeepJSCC
from metrics import _per_image_psnr, _per_image_ssim


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class Evaluator:
    """
    Handles model evaluation across different SNR levels.

    Args:
        model      : Trained DeepJSCC model.
        val_loader : Validation DataLoader.
        device     : Inference device.
        output_dir : Where to save figures.
    """

    def __init__(
        self,
        model:      DeepJSCC,
        val_loader: DataLoader,
        device:     str = "cuda",
        output_dir: str = "results/conv_ae_tf",
    ):
        self.model      = model.to(device).eval()
        self.val_loader = val_loader
        self.device     = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ───────────────────────────────

    @staticmethod
    def _denorm(tensor: torch.Tensor) -> np.ndarray:
        """Convert [-1,1] tensor → uint8 (H, W, 3) numpy array."""
        img = tensor.detach().cpu().clamp(-1, 1)
        img = ((img + 1) / 2 * 255).byte()
        return img.permute(1, 2, 0).numpy()

    # ── single SNR evaluation ─────────────────

    @torch.no_grad()
    def evaluate(self, snr_db: Optional[float] = None) -> dict:
        """
        Run evaluation at a given SNR.

        Args:
            snr_db : If provided, override the model's current SNR.

        Returns:
            dict with 'psnr' and 'ssim' averaged over the validation set.
        """
        if snr_db is not None:
            self.model.set_snr(snr_db)

        total_psnr = 0.0
        total_ssim = 0.0
        n = 0

        for imgs, _ in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            x_hat, _ = self.model(imgs)

            total_psnr += _per_image_psnr(x_hat, imgs).mean().item()
            total_ssim += _per_image_ssim(x_hat, imgs).mean().item()
            n += 1

        return {
            "psnr": total_psnr / max(n, 1),
            "ssim": total_ssim / max(n, 1),
        }

    # ── SNR sweep ─────────────────────────────

    def snr_sweep(
        self,
        snr_range: List[float] = list(range(0, 21, 2)),
        save:      bool        = True,
    ) -> dict:
        """
        Evaluate PSNR and SSIM across a range of SNR values.

        Args:
            snr_range : List of SNR values in dB.
            save      : Whether to save the PSNR-vs-SNR plot.

        Returns:
            dict with 'snr_db', 'psnr', 'ssim' lists.
        """
        psnr_list, ssim_list = [], []

        print(f"\nSNR sweep ({self.model.channel_type.upper()}):")
        print(f"{'SNR (dB)':>10}  {'PSNR (dB)':>12}  {'SSIM':>8}")
        print("-" * 36)

        for snr in snr_range:
            metrics = self.evaluate(snr_db=snr)
            psnr_list.append(metrics["psnr"])
            ssim_list.append(metrics["ssim"])
            print(f"{snr:>10.1f}  {metrics['psnr']:>12.2f}  {metrics['ssim']:>8.4f}")

        results = {"snr_db": snr_range, "psnr": psnr_list, "ssim": ssim_list}

        if save:
            self._plot_snr_curve(results)

        return results

    def _plot_snr_curve(self, results: dict):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, metric, ylabel in zip(
            axes,
            ["psnr",  "ssim"],
            ["PSNR (dB)", "SSIM"],
        ):
            ax.plot(results["snr_db"], results[metric], "o-", linewidth=2)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs SNR  [{self.model.channel_type.upper()}]")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"snr_curve_{self.model.channel_type}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\nSaved SNR curve → {path}")

    # ── Visual comparison ─────────────────────

    @torch.no_grad()
    def visual_comparison(
        self,
        n_images:  int   = 8,
        snr_db:    Optional[float] = None,
        save:      bool  = True,
    ):
        """
        Plot a grid: top row = originals, bottom row = reconstructions.
        """
        if snr_db is not None:
            self.model.set_snr(snr_db)

        imgs_list, recon_list = [], []
        collected = 0

        for imgs, _ in self.val_loader:
            imgs = imgs.to(self.device)
            x_hat, _ = self.model(imgs)

            for i in range(imgs.size(0)):
                if collected >= n_images:
                    break
                imgs_list.append(self._denorm(imgs[i]))
                recon_list.append(self._denorm(x_hat[i]))
                collected += 1
            if collected >= n_images:
                break

        fig, axes = plt.subplots(
            2, n_images,
            figsize=(2 * n_images, 4),
            gridspec_kw={"hspace": 0.05, "wspace": 0.02},
        )

        for col in range(n_images):
            for row, (imgs_row, title) in enumerate(
                [(imgs_list, "Original"),
                 (recon_list, f"Recon. @ {self.model.snr_db} dB")]
            ):
                axes[row, col].imshow(imgs_row[col])
                axes[row, col].axis("off")
                if col == 0:
                    axes[row, col].set_ylabel(title, fontsize=9)

        fig.suptitle(
            f"DeepJSCC — {self.model.channel_type.upper()} | "
            f"k/n={self.model.bandwidth_ratio():.4f}",
            fontsize=11,
        )

        if save:
            path = (self.output_dir /
                    f"visual_snr{self.model.snr_db}_{self.model.channel_type}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved visual comparison → {path}")

        plt.show()
        plt.close()
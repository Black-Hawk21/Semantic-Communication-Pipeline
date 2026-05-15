"""
losses.py — Reconstruction loss functions for Deep JSCC / CAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error — standard pixel-level reconstruction loss."""

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x, reduction="mean")


class PSNRLoss(nn.Module):
    """
    Negative PSNR as a loss (maximise PSNR ↔ minimise this).

    Assumes pixel values in [-1, 1], so MAX_PIXEL = 2.0.
    """
    MAX_PIXEL = 2.0

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(x_hat, x, reduction="mean")
        psnr = 10 * torch.log10(self.MAX_PIXEL ** 2 / mse.clamp(min=1e-10))
        return -psnr


class PerceptualMSELoss(nn.Module):
    """
    Weighted combination of pixel MSE and a simple gradient (edge) loss.
    Encourages the reconstruction to preserve high-frequency structure.

    Args:
        alpha (float): Weight of the gradient term (0 → pure MSE).
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        # Sobel-like kernels for edge detection
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def _gradient(self, img: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(img, self.kx, padding=1, groups=3)
        gy = F.conv2d(img, self.ky, padding=1, groups=3)
        return (gx.pow(2) + gy.pow(2)).sqrt()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pixel_loss    = F.mse_loss(x_hat, x)
        gradient_loss = F.mse_loss(self._gradient(x_hat), self._gradient(x))
        return pixel_loss + self.alpha * gradient_loss


# ─────────────────────────────────────────────
# Metrics (non-differentiable, for logging)
# ─────────────────────────────────────────────

def compute_psnr(x_hat: torch.Tensor, x: torch.Tensor, max_pixel: float = 2.0) -> float:
    """Compute PSNR in dB. Inputs in [-1, 1]."""
    mse = F.mse_loss(x_hat.detach(), x.detach()).item()
    if mse < 1e-10:
        return float("inf")
    return 10 * torch.log10(torch.tensor(max_pixel ** 2 / mse)).item()


def compute_ssim(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """
    Simplified single-scale SSIM over a batch.
    For research use; install `torchmetrics` for a fully validated version.
    """
    # Convert [-1, 1] → [0, 1]
    x_hat = (x_hat.detach().clamp(-1, 1) + 1) / 2
    x     = (x.detach().clamp(-1, 1) + 1) / 2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x   = F.avg_pool2d(x,     3, 1, 1)
    mu_y   = F.avg_pool2d(x_hat, 3, 1, 1)
    mu_xx  = mu_x * mu_x
    mu_yy  = mu_y * mu_y
    mu_xy  = mu_x * mu_y
    sig_xx = F.avg_pool2d(x * x,         3, 1, 1) - mu_xx
    sig_yy = F.avg_pool2d(x_hat * x_hat, 3, 1, 1) - mu_yy
    sig_xy = F.avg_pool2d(x * x_hat,     3, 1, 1) - mu_xy

    num  = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den  = (mu_xx + mu_yy + c1) * (sig_xx + sig_yy + c2)
    ssim = (num / den.clamp(min=1e-8)).mean().item()
    return ssim

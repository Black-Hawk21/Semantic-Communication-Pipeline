"""
metrics_int.py — Extended metrics for Deep JSCC / CAE evaluation

Metrics implemented
────────────────────
1. BER vs SNR          — Bit Error Rate using BPSK hard-decision on the latent
2. Jitter              — Frame-to-frame reconstruction variance (temporal stability)
3. Spectral Efficiency — Shannon capacity estimate (bits/s/Hz) at the channel SNR
4. Bandwidth Usage     — Actual symbols transmitted per source pixel (k/n ratio)
                         and effective bandwidth relative to Nyquist

Extended (v2)
─────────────
- forward_fn support in jitter metrics  (analog vs encrypted pipelines)
- Digital-bridge BER measurement        (quantise → encrypt → FEC → BPSK → channel → …)
- FEC-aware bandwidth / spectral-efficiency variants

All functions accept raw PyTorch tensors and return plain Python scalars
or dicts, so they can be composed freely with your existing Evaluator.
"""

import math
from typing import Optional, List, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
import numpy as np

#### HELPERS ####

def _per_image_psnr(x_hat: torch.Tensor, x: torch.Tensor,
                    max_val: float = 2.0) -> torch.Tensor:
    """Per-image PSNR → shape (B,)."""
    mse = _per_image_mse(x_hat, x)
    return 10.0 * torch.log10(max_val ** 2 / mse.clamp(min=1e-10))


def _per_image_ssim(x_hat: torch.Tensor, x: torch.Tensor,
                    C1: float = 0.01**2 * 4, C2: float = 0.03**2 * 4,
                    window_size: int = 11) -> torch.Tensor:
    """
    Per-image SSIM for [-1,1] signals.
    C1, C2 scaled for dynamic range L=2 → (K*L)^2.
    Returns shape (B,).
    """
    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    gauss_1d = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    window = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)  # (K, K)
    window = window.unsqueeze(0).unsqueeze(0)                # (1, 1, K, K)
    window = window.expand(x.size(1), -1, -1, -1)           # (C, 1, K, K)

    pad = window_size // 2

    mu_x   = F.conv2d(x,     window, padding=pad, groups=x.size(1))
    mu_y   = F.conv2d(x_hat, window, padding=pad, groups=x.size(1))
    mu_x2  = mu_x.pow(2)
    mu_y2  = mu_y.pow(2)
    mu_xy  = mu_x * mu_y

    sigma_x2  = F.conv2d(x.pow(2),     window, padding=pad, groups=x.size(1)) - mu_x2
    sigma_y2  = F.conv2d(x_hat.pow(2), window, padding=pad, groups=x.size(1)) - mu_y2
    sigma_xy  = F.conv2d(x * x_hat,    window, padding=pad, groups=x.size(1)) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    # Mean over spatial dims & channels → per-image scalar
    return ssim_map.mean(dim=[1, 2, 3])


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BER  — Bit Error Rate  (analog latent simulation)
# ══════════════════════════════════════════════════════════════════════════════

def _encode_latent(model, imgs: torch.Tensor) -> torch.Tensor:
    """
    Get the encoder output from either DeepJSCC (.encode method) or
    DeepJSCCSecure (.encoder attribute only).
    """
    if hasattr(model, "encode") and callable(model.encode):
        return model.encode(imgs)
    return model.encoder(imgs)


def bpsk_modulate(x: torch.Tensor) -> torch.Tensor:
    """
    Map each real-valued latent element to a BPSK symbol: +1 or -1.
    This acts as a hard-decision reference transmitter.
    """
    return torch.sign(x)


def awgn_corrupt(x_bpsk: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN noise to BPSK-modulated signal."""
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std  = (1.0 / (2.0 * snr_linear)) ** 0.5   # Eb/N0 → noise variance
    noise      = torch.randn_like(x_bpsk) * noise_std
    return x_bpsk + noise


def bpsk_demodulate(y: torch.Tensor) -> torch.Tensor:
    """Hard-decision BPSK detector: sign(y) → {+1, -1}"""
    return torch.sign(y)


def compute_ber(
    z: torch.Tensor,
    snr_db: float,
    channel_type: str = "awgn",
) -> float:
    """
    Compute Bit Error Rate via BPSK hard-decision simulation.

    The latent tensor `z` is binarised (sign), modulated as BPSK,
    passed through AWGN or Rayleigh channel, and demodulated.
    BER = fraction of bits that flipped.

    Args:
        z            : Latent tensor (B, C, H, W) — pre-channel.
        snr_db       : Channel SNR in dB.
        channel_type : 'awgn' | 'rayleigh'

    Returns:
        ber (float)  : Bit error rate in [0, 0.5].
    """
    with torch.no_grad():
        z_flat   = z.detach().float()
        tx       = bpsk_modulate(z_flat)           # {+1, -1}

        if channel_type.lower() == "awgn":
            rx = awgn_corrupt(tx, snr_db)

        elif channel_type.lower() in ("rayleigh", "rician"):
            h_real   = torch.randn_like(tx)
            h_imag   = torch.randn_like(tx)
            h_mag    = (h_real.pow(2) + h_imag.pow(2)).sqrt() / math.sqrt(2)
            snr_lin  = 10 ** (snr_db / 10.0)
            noise_std = (1.0 / (2.0 * snr_lin)) ** 0.5
            rx       = h_mag * tx + torch.randn_like(tx) * noise_std
            rx       = rx / h_mag.clamp(min=1e-8)

        else:
            raise ValueError(f"Unknown channel_type '{channel_type}'")

        rx_bits  = bpsk_demodulate(rx)
        errors   = (rx_bits != tx).float().sum().item()
        total    = tx.numel()

    return errors / total


def theoretical_ber_awgn(snr_db: float) -> float:
    """
    Theoretical BPSK BER over AWGN: Q(sqrt(2·Eb/N0)).
    Useful for comparison / sanity-check plots.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    x = math.sqrt(2.0 * snr_linear)
    return 0.5 * math.erfc(x / math.sqrt(2))


def ber_vs_snr(
    model,
    val_loader,
    snr_range: List[float],
    device: str = "cuda",
    channel_type: Optional[str] = None,
    n_batches: int = 10,
) -> Dict[str, list]:
    """
    Sweep BER across SNR values using the model's encoder output
    (analog BPSK simulation on the raw latent).
    """
    ch_type = channel_type or model.channel_type
    ber_list  = []
    ber_theory = []
    model.eval()

    for snr in snr_range:
        total_ber = 0.0
        count     = 0
        for i, (imgs, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                z = _encode_latent(model, imgs)
            total_ber += compute_ber(z, snr, ch_type)
            count += 1

        ber_list.append(total_ber / max(count, 1))
        ber_theory.append(
            theoretical_ber_awgn(snr) if ch_type == "awgn" else None
        )

    return {
        "snr_db":          snr_range,
        "ber":             ber_list,
        "ber_theoretical": ber_theory,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1b.  BER  — Digital-bridge BER  (quantise → FEC → BPSK → channel → …)
# ══════════════════════════════════════════════════════════════════════════════

def compute_digital_ber(
    z: torch.Tensor,
    snr_db: float,
    channel_type: str = "awgn",
    fec_type: str = "repetition",
    fec_n: int = 5,
    quant_bits: int = 8,
) -> float:
    """
    Measure BER through the full digital sub-pipeline:
      quantise → bit-pack → FEC encode → BPSK → channel → demod → FEC decode

    This captures the *actual* post-FEC BER that the encrypted link
    experiences, whereas compute_ber() only simulates raw BPSK on the
    latent signs.

    Args:
        z            : Latent tensor (B, C, H, W).
        snr_db       : Channel SNR in dB.
        channel_type : 'awgn' | 'rayleigh'
        fec_type     : 'repetition' | 'hamming' | 'none'
        fec_n        : Repetition factor (only for fec_type='repetition').
        quant_bits   : Quantisation bit-depth.

    Returns:
        ber (float)  : Post-FEC bit error rate.
    """
    from quantize import tensor_to_bytes
    from fec import (
        RepetitionCode, HammingCode74,
        bpsk_modulate as fec_bpsk_mod,
        bpsk_demodulate_hard,
    )

    if fec_type == "repetition":
        fec = RepetitionCode(fec_n)
    elif fec_type == "hamming":
        fec = HammingCode74()
    else:
        fec = None

    total_errors = 0
    total_bits   = 0

    with torch.no_grad():
        for i in range(z.size(0)):
            z_i = z[i]
            raw_bytes, _ = tensor_to_bytes(z_i, bits=quant_bits)

            orig_bits = np.unpackbits(
                np.frombuffer(raw_bytes, dtype=np.uint8)
            ).astype(np.int32)

            coded = fec.encode(orig_bits) if fec is not None else orig_bits
            symbols = fec_bpsk_mod(coded)

            snr_lin   = 10 ** (snr_db / 10.0)
            noise_std = (1.0 / (2 * snr_lin)) ** 0.5

            if channel_type.lower() == "awgn":
                noisy = symbols + np.random.randn(len(symbols)) * noise_std
            else:  # rayleigh
                h = (np.random.randn(len(symbols)) +
                     1j * np.random.randn(len(symbols))) / np.sqrt(2)
                h_mag = np.abs(h)
                noisy = h_mag * symbols + np.random.randn(len(symbols)) * noise_std
                noisy = noisy / np.maximum(h_mag, 1e-8)

            rx_bits = bpsk_demodulate_hard(noisy)
            decoded = fec.decode_hard(rx_bits) if fec is not None else rx_bits
            decoded = decoded[:len(orig_bits)]

            total_errors += int(np.sum(orig_bits != decoded))
            total_bits   += len(orig_bits)

    return total_errors / max(total_bits, 1)


def digital_ber_vs_snr(
    model,
    val_loader,
    snr_range: List[float],
    device: str = "cuda",
    channel_type: Optional[str] = None,
    n_batches: int = 10,
    fec_type: str = "repetition",
    fec_n: int = 5,
    quant_bits: int = 8,
) -> Dict[str, list]:
    """
    Sweep post-FEC BER across SNR values using the digital bridge pipeline.
    """
    ch_type = channel_type or model.channel_type
    ber_list   = []
    ber_theory = []
    model.eval()

    for snr in snr_range:
        total_ber = 0.0
        count     = 0
        for i, (imgs, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                z = _encode_latent(model, imgs)
                if hasattr(model, "tx_transform"):
                    z = model.tx_transform(z)
            total_ber += compute_digital_ber(
                z, snr, ch_type, fec_type, fec_n, quant_bits
            )
            count += 1

        ber_list.append(total_ber / max(count, 1))
        ber_theory.append(
            theoretical_ber_awgn(snr) if ch_type == "awgn" else None
        )

    return {
        "snr_db":          snr_range,
        "ber":             ber_list,
        "ber_theoretical": ber_theory,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Jitter — Reconstruction Temporal Stability
# ══════════════════════════════════════════════════════════════════════════════

def compute_jitter(
    model,
    imgs: torch.Tensor,
    snr_db: float,
    n_passes: int = 10,
    device: str = "cuda",
    forward_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Measure jitter as the variance of reconstruction quality across multiple
    independent channel realisations for the same input batch.

    Args:
        model      : DeepJSCC or DeepJSCCSecure model.
        imgs       : Input image batch (B, 3, H, W).
        snr_db     : Channel SNR in dB.
        n_passes   : Number of independent channel realisations to sample.
        device     : Inference device.
        forward_fn : Optional callable (model, imgs) → (x_hat, z).
                     If None, calls model(imgs) directly.

    Returns:
        dict with 'mse_mean', 'mse_std', 'mse_cv', 'psnr_std'.
    """
    model.set_snr(snr_db)
    # For analog: train mode keeps batch norm stochastic + channel noise active.
    # For encrypted (custom forward_fn): eval mode is correct because the
    # digital bridge already simulates its own channel independently.
    if forward_fn is not None:
        model.eval()
    else:
        model.train()
    imgs = imgs.to(device)

    mse_vals  = []
    psnr_vals = []

    _forward = forward_fn or (lambda m, x: m(x))

    with torch.no_grad():
        for _ in range(n_passes):
            x_hat, _ = _forward(model, imgs)
            mse = F.mse_loss(x_hat, imgs).item()
            mse_vals.append(mse)
            psnr = 10 * math.log10(4.0 / (mse + 1e-10))
            psnr_vals.append(psnr)

    mse_arr  = np.array(mse_vals)
    psnr_arr = np.array(psnr_vals)

    return {
        "mse_mean": float(mse_arr.mean()),
        "mse_std":  float(mse_arr.std()),
        "mse_cv":   float(mse_arr.std() / (mse_arr.mean() + 1e-10)),
        "psnr_std": float(psnr_arr.std()),
        "n_passes": n_passes,
    }


def jitter_vs_snr(
    model,
    val_loader,
    snr_range: List[float],
    device: str = "cuda",
    n_passes: int = 10,
    n_batches: int = 3,
    forward_fn: Optional[Callable] = None,
) -> Dict[str, list]:
    """
    Compute jitter across SNR values.

    Args:
        forward_fn : Optional callable (model, imgs) → (x_hat, z).
                     Enables switching between analog and encrypted forward.
    """
    mse_std_list  = []
    psnr_std_list = []
    mse_cv_list   = []

    for snr in snr_range:
        snr_mse_std  = []
        snr_psnr_std = []
        snr_mse_cv   = []

        for i, (imgs, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            j = compute_jitter(model, imgs, snr, n_passes, device, forward_fn)
            snr_mse_std.append(j["mse_std"])
            snr_psnr_std.append(j["psnr_std"])
            snr_mse_cv.append(j["mse_cv"])

        mse_std_list.append(float(np.mean(snr_mse_std)))
        psnr_std_list.append(float(np.mean(snr_psnr_std)))
        mse_cv_list.append(float(np.mean(snr_mse_cv)))

    return {
        "snr_db":   snr_range,
        "mse_std":  mse_std_list,
        "psnr_std": psnr_std_list,
        "mse_cv":   mse_cv_list,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Spectral Efficiency — Rate-Distortion Framework
# ══════════════════════════════════════════════════════════════════════════════

def shannon_capacity(snr_db: float) -> float:
    """Shannon–Hartley: C = log2(1 + SNR)  [bits/channel-use]."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    return math.log2(1.0 + snr_linear)


def _get_latent_dims(model, img_h: int, img_w: int):
    """
    Infer the latent spatial size from the encoder architecture.

    The encoder performs 4 stride-2 downsampling stages:
        H → H/2 → H/4 → H/8 → H/16
    So latent size = (img_h/16, img_w/16) and
    k = latent_channels × (img_h/16) × (img_w/16).
    """
    latent_h = img_h // 16
    latent_w = img_w // 16
    k = model.latent_channels * latent_h * latent_w
    n = 3 * img_h * img_w   # source dimensionality (RGB pixels)
    return k, n, latent_h, latent_w


def _per_image_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-image MSE → shape (B,)."""
    return (x_hat - x).pow(2).mean(dim=[1, 2, 3])


def effective_spectral_efficiency(
    model,
    snr_db: float,
    achieved_distortion_mse: float,
    img_h: int = 64,
    img_w: int = 64,
    fec_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Compute spectral efficiency using a rate-distortion framework.

    Theory (Shannon separation theorem)
    ─────────────────────────────────────
    For an i.i.d. Gaussian source with variance σ² transmitted over an
    AWGN channel with capacity C [bits/channel-use]:

        • Source coding:  R(D) = 0.5 × log2(σ²/D)  bits/source-sample
          (rate needed to describe the source at distortion D)
        • Channel coding:  each channel use carries at most C bits
        • With k channel uses for n source samples, the system can
          deliver at most  R_max = (k/n) × C  bits/source-sample
        • Achievable distortion:  D_min = σ² × 2^(-2·R_max)

    The "effective spectral efficiency" is defined as the rate the
    model actually achieves, normalised back to bits/channel-use:

        η_achieved = R_achieved / (k/n)
                   = [0.5 × log2(σ²/D_achieved)] / (k/n)

    This is directly comparable to the Shannon capacity C.  A model
    should approach C from below; plotting η_achieved vs C on the same
    axis gives a meaningful gap-to-capacity curve.

    When fec_rate < 1 (e.g. 0.2 for repetition-5), the effective SE is
    further reduced because FEC overhead shrinks the usable throughput.

    Args:
        model                   : DeepJSCC model.
        snr_db                  : Operating SNR in dB.
        achieved_distortion_mse : Actual MSE achieved by the model at this SNR.
        img_h, img_w            : Input image spatial dimensions.
        fec_rate                : FEC code rate (1.0 = analog, <1 = FEC overhead).

    Returns:
        dict with spectral efficiency metrics.
    """
    k, n, _, _ = _get_latent_dims(model, img_h, img_w)
    kn_ratio   = k / n

    # Channel capacity [bits/channel-use]
    C_shannon  = shannon_capacity(snr_db)

    # Maximum deliverable source rate under separation theorem
    R_max      = kn_ratio * C_shannon                  # bits/source-sample

    # Source variance for [-1, 1] normalised images
    # Using σ²=1 (Gaussian upper bound) for the theoretical distortion floor
    sigma_sq = 1.0

    # Theoretical minimum distortion under separation theorem
    D_min = sigma_sq * (2.0 ** (-2.0 * R_max))

    # Achieved source coding rate via the rate-distortion function R(D)
    D_achieved = max(achieved_distortion_mse, 1e-10)
    if D_achieved < sigma_sq:
        R_achieved = 0.5 * math.log2(sigma_sq / D_achieved)
    else:
        R_achieved = 0.0  # distortion exceeds source variance → 0 useful bits

    # Effective SE: normalise back to bits/channel-use so it is on the
    # same scale as Shannon capacity C.  Apply FEC rate penalty on top.
    eta_achieved = (R_achieved / max(kn_ratio, 1e-10)) * fec_rate if kn_ratio > 0 else 0.0

    # Efficiency ratio: fraction of Shannon capacity actually used
    eta_ratio = eta_achieved / C_shannon if C_shannon > 0 else 0.0

    return {
        "snr_db":                    snr_db,
        "k_symbols":                 k,
        "n_source_samples":          n,
        "kn_ratio":                  kn_ratio,
        "fec_rate":                  fec_rate,
        "channel_capacity_bpcu":     C_shannon,
        "R_max_separation_bpss":     R_max,
        "D_min_separation":          D_min,
        "D_achieved":                D_achieved,
        "R_achieved_bpss":           R_achieved,
        "effective_SE_bpcu":         eta_achieved,
        "efficiency_ratio":          eta_ratio,
    }


def spectral_efficiency_vs_snr(
    model,
    val_loader,
    snr_range: List[float],
    img_h: int = 64,
    img_w: int = 64,
    fec_rate: float = 1.0,
    n_batches: int = 5,
    device: str = "cuda",
) -> Dict[str, list]:
    """
    Spectral efficiency sweep: measures actual distortion at each SNR,
    then computes rate-distortion-based SE.

    The model's effective_SE_bpcu is on the same scale as channel_capacity_bpcu
    (both in bits/channel-use), so they can be plotted on the same axis to
    visualise the gap to the Shannon limit.

    Set fec_rate < 1 (e.g. 0.2 for rep-5) for encrypted pipeline overhead.
    """
    results = {
        "snr_db":                [],
        "channel_capacity_bpcu": [],
        "effective_SE_bpcu":     [],
        "efficiency_ratio":      [],
        "D_achieved":            [],
        "D_min_separation":      [],
        "R_achieved_bpss":       [],
        "kn_ratio":              [],
    }

    model.eval()

    for snr in snr_range:
        model.set_snr(snr)

        # Measure actual MSE at this SNR
        total_mse = 0.0
        count = 0
        with torch.no_grad():
            for i, (imgs, _) in enumerate(val_loader):
                if i >= n_batches:
                    break
                imgs = imgs.to(device, non_blocking=True)
                x_hat, _ = model(imgs)
                total_mse += _per_image_mse(x_hat, imgs).mean().item()
                count += 1

        avg_mse = total_mse / max(count, 1)
        se = effective_spectral_efficiency(model, snr, avg_mse, img_h, img_w, fec_rate)

        results["snr_db"].append(snr)
        results["channel_capacity_bpcu"].append(se["channel_capacity_bpcu"])
        results["effective_SE_bpcu"].append(se["effective_SE_bpcu"])
        results["efficiency_ratio"].append(se["efficiency_ratio"])
        results["D_achieved"].append(se["D_achieved"])
        results["D_min_separation"].append(se["D_min_separation"])
        results["R_achieved_bpss"].append(se["R_achieved_bpss"])
        results["kn_ratio"].append(se["kn_ratio"])

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Bandwidth Usage
# ══════════════════════════════════════════════════════════════════════════════

def compute_bandwidth_usage(
    model,
    img_h: int = 64,
    img_w: int = 64,
    sample_rate_hz: float = 1.0,
    fec_rate: float = 1.0,
    quant_bits: int = 8,
) -> Dict[str, float]:
    """
    Characterise bandwidth usage of the semantic codec.

    When fec_rate < 1, the actual number of channel symbols increases by
    1/fec_rate because FEC adds redundancy.  quant_bits affects total bits
    per latent element in the encrypted pipeline.
    """
    latent_h = img_h // 16
    latent_w = img_w // 16
    k_latent = model.latent_channels * latent_h * latent_w
    n = 3 * img_h * img_w

    if fec_rate < 1.0:
        # Encrypted pipeline: each latent → quant_bits bits → FEC expansion
        k_channel_bits    = k_latent * quant_bits
        k_channel_symbols = k_channel_bits / fec_rate
        k = k_channel_symbols
    else:
        k = k_latent

    compression_ratio = n / k
    kn_ratio          = k / n

    nyquist_bw_semantic = (k * sample_rate_hz) / 2.0
    nyquist_bw_raw      = (n * sample_rate_hz) / 2.0

    return {
        "latent_channels":           model.latent_channels,
        "latent_spatial":            (latent_h, latent_w),
        "k_symbols_per_image":       k,
        "n_pixels_per_image":        n,
        "compression_ratio":         compression_ratio,
        "bandwidth_ratio_kn":        kn_ratio,
        "fec_rate":                  fec_rate,
        "quant_bits":                quant_bits,
        "nyquist_bw_semantic_hz":    nyquist_bw_semantic,
        "nyquist_bw_raw_hz":         nyquist_bw_raw,
        "bandwidth_reduction_factor": nyquist_bw_raw / max(nyquist_bw_semantic, 1e-12),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Unified metrics runner (convenience wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def run_all_metrics(
    model,
    val_loader,
    snr_range: List[float],
    device: str = "cuda",
    img_h: int = 64,
    img_w: int = 64,
    ber_n_batches: int = 10,
    jitter_n_passes: int = 10,
    jitter_n_batches: int = 3,
    verbose: bool = True,
    forward_fn: Optional[Callable] = None,
    fec_rate: float = 1.0,
    fec_type: str = "none",
    fec_n: int = 5,
    quant_bits: int = 8,
    use_digital_ber: bool = False,
) -> Dict[str, dict]:
    """
    Run all four metric groups and return a unified results dict.

    New parameters vs v1:
        forward_fn      Callable for jitter (switches analog / encrypted).
        fec_rate        FEC code rate (1.0 = analog, <1 = encrypted+FEC).
        fec_type / fec_n / quant_bits   Digital bridge config.
        use_digital_ber If True, measure BER through the digital pipeline.
    """
    results = {}

    if verbose:
        print("\n[1/4] Computing BER vs SNR …")
    if use_digital_ber:
        results["ber"] = digital_ber_vs_snr(
            model, val_loader, snr_range, device,
            n_batches=ber_n_batches,
            fec_type=fec_type, fec_n=fec_n, quant_bits=quant_bits,
        )
    else:
        results["ber"] = ber_vs_snr(
            model, val_loader, snr_range, device,
            n_batches=ber_n_batches,
        )

    if verbose:
        print("[2/4] Computing Jitter vs SNR …")
    results["jitter"] = jitter_vs_snr(
        model, val_loader, snr_range, device,
        n_passes=jitter_n_passes,
        n_batches=jitter_n_batches,
        forward_fn=forward_fn,
    )

    if verbose:
        print("[3/4] Computing Spectral Efficiency …")
    results["spectral_efficiency"] = spectral_efficiency_vs_snr(
        model, val_loader, snr_range, img_h, img_w,
        fec_rate=fec_rate, device=device,
    )

    if verbose:
        print("[4/4] Computing Bandwidth Usage …")
    results["bandwidth_usage"] = compute_bandwidth_usage(
        model, img_h, img_w,
        fec_rate=fec_rate, quant_bits=quant_bits,
    )

    model.eval()
    return results
"""
eval_metrics.py — Extended metric evaluation for Deep JSCC / CAE

Features (v2):
  - Evaluates BOTH analog and encrypted (digital bridge) pipelines
  - Plots overlay curves: Analog vs Encrypted on every metric
  - Single-model or multi-model comparison still supported

Usage
─────
    # Single model — auto-compares analog vs encrypted
    python eval_metrics.py --ckpt runs/exp_awgn_snr10/ckpt_best.pt

    # Custom FEC settings
    python eval_metrics.py --ckpt runs/exp/ckpt_best.pt \
        --fec_type repetition --fec_n 5 --quant_bits 8

    # Overlay multiple models (each gets analog + encrypted curves)
    python eval_metrics.py --compare \
        runs/awgn_k16/ckpt_best.pt \
        runs/awgn_k32/ckpt_best.pt \
        --labels "AWGN k=16" "AWGN k=32"

    # Analog only (skip encrypted evaluation)
    python eval_metrics.py --ckpt runs/exp/ckpt_best.pt --analog_only
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import config
from model import DeepJSCC
from model_integrated import DeepJSCCSecure
from dataset import get_dataloader
from metrics import (
    run_all_metrics,
    theoretical_ber_awgn,
    compute_bandwidth_usage,
    effective_spectral_efficiency,
)


# ══════════════════════════════════════════════════════════════════════════════
# Forward-function factories for analog vs encrypted evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _analog_forward(model, imgs):
    """Route through the analog channel (differentiable path)."""
    return model.forward_analog(imgs)


def _encrypted_forward(model, imgs):
    """Route through the digital bridge (quantise → encrypt → FEC → channel)."""
    return model.forward_encrypted(imgs)


# ══════════════════════════════════════════════════════════════════════════════
# FEC rate helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fec_rate(fec_type: str, fec_n: int) -> float:
    """Return the code rate for a given FEC configuration."""
    if fec_type == "repetition":
        return 1.0 / fec_n
    elif fec_type == "hamming":
        return 4.0 / 7.0
    else:
        return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "#cccccc",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
}

# Colour-blind friendly palette (solid for analog, paired with dashed for encrypted)
PALETTE = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA3377", "#33BBEE"]


def _style_ax(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.legend(fontsize=9, framealpha=0.85)


def plot_ber(results_list, labels, output_dir: Path):
    """
    BER vs SNR (log scale) with theoretical BPSK-AWGN curve.
    Each entry in results_list is a (analog_results, encrypted_results) pair.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Theoretical reference
        first_analog = results_list[0][0]
        snr_fine = np.linspace(
            first_analog["ber"]["snr_db"][0],
            first_analog["ber"]["snr_db"][-1], 200,
        )
        ber_th = [theoretical_ber_awgn(s) for s in snr_fine]
        ax.semilogy(snr_fine, ber_th, "k--", linewidth=1.2,
                    label="Theoretical BPSK-AWGN", zorder=0)

        for idx, ((r_analog, r_enc), label) in enumerate(zip(results_list, labels)):
            color = PALETTE[idx % len(PALETTE)]

            # Analog curve (solid)
            snr_a = r_analog["ber"]["snr_db"]
            ber_a = r_analog["ber"]["ber"]
            ax.semilogy(snr_a, ber_a, "o-", color=color, linewidth=2,
                        markersize=5, label=f"{label} (Analog)")

            # Encrypted curve (dashed)
            if r_enc is not None:
                snr_e = r_enc["ber"]["snr_db"]
                ber_e = r_enc["ber"]["ber"]
                ax.semilogy(snr_e, ber_e, "s--", color=color, linewidth=2,
                            markersize=5, alpha=0.8, label=f"{label} (Encrypted)")

        ax.set_ylim(bottom=1e-7)
        _style_ax(ax, "SNR (dB)", "BER",
                  "Bit Error Rate vs SNR — Analog vs Encrypted")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        fig.tight_layout()
        path = output_dir / "ber_vs_snr.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")


def plot_jitter(results_list, labels, output_dir: Path):
    """Jitter (PSNR std dev and MSE CV) vs SNR — Analog vs Encrypted."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for idx, ((r_analog, r_enc), label) in enumerate(zip(results_list, labels)):
            color = PALETTE[idx % len(PALETTE)]

            snr_a = r_analog["jitter"]["snr_db"]
            axes[0].plot(snr_a, r_analog["jitter"]["psnr_std"], "o-",
                         color=color, linewidth=2, markersize=5,
                         label=f"{label} (Analog)")
            axes[1].plot(snr_a, r_analog["jitter"]["mse_cv"], "o-",
                         color=color, linewidth=2, markersize=5,
                         label=f"{label} (Analog)")

            if r_enc is not None:
                snr_e = r_enc["jitter"]["snr_db"]
                axes[0].plot(snr_e, r_enc["jitter"]["psnr_std"], "s--",
                             color=color, linewidth=2, markersize=5, alpha=0.8,
                             label=f"{label} (Encrypted)")
                axes[1].plot(snr_e, r_enc["jitter"]["mse_cv"], "s--",
                             color=color, linewidth=2, markersize=5, alpha=0.8,
                             label=f"{label} (Encrypted)")

        _style_ax(axes[0], "SNR (dB)", "PSNR Std Dev (dB)",
                  "Jitter — PSNR Variability vs SNR")
        _style_ax(axes[1], "SNR (dB)", "MSE Coeff. of Variation",
                  "Jitter — MSE Stability vs SNR")
        fig.tight_layout()
        path = output_dir / "jitter_vs_snr.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")


def plot_spectral_efficiency(results_list, labels, output_dir: Path):
    """Effective SE vs SNR — Analog vs Encrypted, with Shannon bound."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Shannon bound
        first_analog = results_list[0][0]
        snr_arr = first_analog["spectral_efficiency"]["snr_db"]
        shannon = first_analog["spectral_efficiency"]["channel_capacity_bpcu"]
        ax.plot(snr_arr, shannon, "k--", linewidth=1.2, label="Shannon Capacity")

        for idx, ((r_analog, r_enc), label) in enumerate(zip(results_list, labels)):
            color = PALETTE[idx % len(PALETTE)]

            snr_a = r_analog["spectral_efficiency"]["snr_db"]
            se_a  = r_analog["spectral_efficiency"]["effective_SE_bpcu"]
            ax.plot(snr_a, se_a, "o-", color=color, linewidth=2,
                    markersize=5, label=f"{label} (Analog)")

            if r_enc is not None:
                snr_e = r_enc["spectral_efficiency"]["snr_db"]
                se_e  = r_enc["spectral_efficiency"]["effective_SE_bpcu"]
                ax.plot(snr_e, se_e, "s--", color=color, linewidth=2,
                        markersize=5, alpha=0.8, label=f"{label} (Encrypted)")

        _style_ax(ax, "SNR (dB)", "Spectral Efficiency (bits/channel-use)",
                  "Effective Spectral Efficiency vs SNR — Analog vs Encrypted")
        fig.tight_layout()
        path = output_dir / "spectral_efficiency_vs_snr.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")


def plot_bandwidth_usage(results_list, labels, output_dir: Path):
    """Grouped bar chart comparing bandwidth metrics: Analog vs Encrypted."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_models = len(labels)
        x = np.arange(n_models)
        bar_w = 0.35

        kn_analog  = [r[0]["bandwidth_usage"]["bandwidth_ratio_kn"] for r in results_list]
        bw_analog  = [r[0]["bandwidth_usage"]["bandwidth_reduction_factor"] for r in results_list]

        has_enc = any(r[1] is not None for r in results_list)

        # k/n ratio
        bars1 = axes[0].bar(x - bar_w / 2, kn_analog, width=bar_w,
                            color=[PALETTE[i % len(PALETTE)] for i in range(n_models)],
                            edgecolor="white", label="Analog")
        for xi, v in zip(x, kn_analog):
            axes[0].text(xi - bar_w / 2, v + 0.002, f"{v:.4f}",
                         ha="center", fontsize=8)

        if has_enc:
            kn_enc = []
            for r in results_list:
                kn_enc.append(r[1]["bandwidth_usage"]["bandwidth_ratio_kn"]
                              if r[1] is not None else 0)
            bars2 = axes[0].bar(x + bar_w / 2, kn_enc, width=bar_w,
                                color=[PALETTE[i % len(PALETTE)] for i in range(n_models)],
                                edgecolor="white", alpha=0.55, hatch="//",
                                label="Encrypted")
            for xi, v in zip(x, kn_enc):
                if v > 0:
                    axes[0].text(xi + bar_w / 2, v + 0.002, f"{v:.4f}",
                                 ha="center", fontsize=8)

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        axes[0].set_ylabel("Bandwidth Ratio k/n")
        axes[0].set_title("Channel Symbol : Source Pixel Ratio")
        axes[0].legend(fontsize=9)
        axes[0].set_ylim(0, max(kn_analog + (kn_enc if has_enc else [])) * 1.4)

        # Bandwidth reduction
        bars3 = axes[1].bar(x - bar_w / 2, bw_analog, width=bar_w,
                            color=[PALETTE[i % len(PALETTE)] for i in range(n_models)],
                            edgecolor="white", label="Analog")
        for xi, v in zip(x, bw_analog):
            axes[1].text(xi - bar_w / 2, v + 0.1, f"{v:.1f}×",
                         ha="center", fontsize=8)

        if has_enc:
            bw_enc = []
            for r in results_list:
                bw_enc.append(r[1]["bandwidth_usage"]["bandwidth_reduction_factor"]
                              if r[1] is not None else 0)
            bars4 = axes[1].bar(x + bar_w / 2, bw_enc, width=bar_w,
                                color=[PALETTE[i % len(PALETTE)] for i in range(n_models)],
                                edgecolor="white", alpha=0.55, hatch="//",
                                label="Encrypted")
            for xi, v in zip(x, bw_enc):
                if v > 0:
                    axes[1].text(xi + bar_w / 2, v + 0.1, f"{v:.1f}×",
                                 ha="center", fontsize=8)

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        axes[1].set_ylabel("Bandwidth Reduction Factor (×)")
        axes[1].set_title("Bandwidth Saved vs Raw Transmission")
        axes[1].legend(fontsize=9)

        fig.tight_layout()
        path = output_dir / "bandwidth_usage.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Printing helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_bandwidth_table(bw: dict, label: str):
    print(f"\n{'─'*56}")
    print(f"  Bandwidth Usage — {label}")
    print(f"{'─'*56}")
    print(f"  Latent channels     : {bw['latent_channels']}")
    print(f"  Latent spatial      : {bw['latent_spatial'][0]}×{bw['latent_spatial'][1]}")
    print(f"  k (channel symbols) : {bw['k_symbols_per_image']:,.0f}")
    print(f"  n (source pixels)   : {bw['n_pixels_per_image']:,}")
    print(f"  Bandwidth ratio k/n : {bw['bandwidth_ratio_kn']:.4f}")
    print(f"  Compression ratio   : {bw['compression_ratio']:.2f}× (symbols saved per pixel)")
    print(f"  FEC rate            : {bw.get('fec_rate', 1.0):.3f}")
    print(f"  Quant bits          : {bw.get('quant_bits', 'N/A')}")
    print(f"  Nyquist BW (semantic): {bw['nyquist_bw_semantic_hz']:.1f} Hz/fps")
    print(f"  Nyquist BW (raw)    : {bw['nyquist_bw_raw_hz']:.1f} Hz/fps")
    print(f"  Bandwidth reduction : {bw['bandwidth_reduction_factor']:.2f}×")


def print_ber_table(ber: dict, label: str):
    print(f"\n{'─'*56}")
    print(f"  BER vs SNR — {label}")
    print(f"{'─'*56}")
    print(f"  {'SNR (dB)':>10}  {'BER (simulated)':>18}  {'BER (theoretical)':>18}")
    for snr, b, bt in zip(ber["snr_db"], ber["ber"], ber["ber_theoretical"]):
        th_str = f"{bt:.6f}" if bt is not None else "  N/A (Rayleigh)"
        print(f"  {snr:>10.1f}  {b:>18.6f}  {th_str:>18}")


def print_jitter_table(jitter: dict, label: str):
    print(f"\n{'─'*56}")
    print(f"  Jitter vs SNR — {label}")
    print(f"{'─'*56}")
    print(f"  {'SNR (dB)':>10}  {'PSNR Std (dB)':>15}  {'MSE Std':>10}  {'MSE CV':>10}")
    for snr, ps, ms, cv in zip(
        jitter["snr_db"], jitter["psnr_std"], jitter["mse_std"], jitter["mse_cv"]
    ):
        print(f"  {snr:>10.1f}  {ps:>15.4f}  {ms:>10.6f}  {cv:>10.4f}")


def print_se_table(se: dict, label: str):
    print(f"\n{'─'*56}")
    print(f"  Spectral Efficiency — {label}")
    print(f"{'─'*56}")
    print(f"  {'SNR (dB)':>10}  {'Shannon C (b/cu)':>20}  {'Effective SE':>14}")
    for snr, sc, ese in zip(
        se["snr_db"], se["channel_capacity_bpcu"], se["effective_SE_bpcu"]
    ):
        print(f"  {snr:>10.1f}  {sc:>20.4f}  {ese:>14.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, args, device: str) -> DeepJSCC:
    """Load a trained DeepJSCC model from checkpoint."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = DeepJSCC(
        latent_channels = args.latent_channels,
        channel_type    = args.channel,
        snr_db          = args.snr_db,
    )
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"  Loaded: {ckpt_path}  "
          f"({model.channel_type.upper()}, k/n={model.bandwidth_ratio():.4f})")
    return model


def wrap_secure(
    original: DeepJSCC,
    args,
    device: str,
) -> DeepJSCCSecure:
    """Wrap a trained DeepJSCC with the encrypted digital bridge."""
    secure = DeepJSCCSecure.from_pretrained(
        original,
        quant_bits=args.quant_bits,
        fec_type=args.fec_type,
        fec_n=args.fec_n,
        key_seed=args.key_seed,
    )
    secure = secure.to(device)
    print(f"  Wrapped with encryption: FEC={args.fec_type}"
          f"{'(' + str(args.fec_n) + ')' if args.fec_type == 'repetition' else ''}"
          f"  quant={args.quant_bits}-bit  key_seed={args.key_seed}")
    return secure


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Extended metric evaluation for Deep JSCC / CAE "
                    "— Analog vs Encrypted comparison"
    )

    # Single-model mode
    p.add_argument("--ckpt",            type=str,   default=None,
                   help="Path to a single checkpoint (.pt)")

    # Multi-model comparison mode
    p.add_argument("--compare",         nargs="+",  default=None,
                   help="List of checkpoint paths for cross-model comparison")
    p.add_argument("--labels",          nargs="+",  default=None,
                   help="Display labels matching --compare checkpoints")

    # Model config
    p.add_argument("--latent_channels", type=int,   default=config.LATENT_CHANNELS)
    p.add_argument("--channel",         type=str,   default=config.CHANNEL_TYPE)
    p.add_argument("--snr_db",          type=float, default=config.SNR_DB)

    # SNR sweep
    p.add_argument("--snr_min",  type=int, default=0)
    p.add_argument("--snr_max",  type=int, default=20)
    p.add_argument("--snr_step", type=int, default=2)

    # Evaluation depth
    p.add_argument("--ber_n_batches",    type=int, default=10)
    p.add_argument("--jitter_n_passes",  type=int, default=10)
    p.add_argument("--jitter_n_batches", type=int, default=3)

    # Encryption / digital bridge config
    p.add_argument("--fec_type",    type=str, default="repetition",
                   choices=["repetition", "hamming", "none"],
                   help="FEC type for the encrypted pipeline")
    p.add_argument("--fec_n",       type=int, default=5,
                   help="Repetition factor (odd, ≥3)")
    p.add_argument("--quant_bits",  type=int, default=8, choices=[8, 16],
                   help="Quantisation bit-depth")
    p.add_argument("--key_seed",    type=int, default=12345,
                   help="Symmetric key seed for encryption")

    # Mode flags
    p.add_argument("--analog_only", action="store_true",
                   help="Skip encrypted evaluation (original v1 behaviour)")

    p.add_argument("--output_dir", type=str, default="results/extended_metrics")
    p.add_argument("--save_json",  action="store_true",
                   help="Save results as JSON for external analysis")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────
    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\nDevice : {device}")

    # ── SNR sweep ────────────────────────────
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    print(f"SNR sweep : {snr_range} dB")

    # ── Output directory ─────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Val loader ───────────────────────────
    val_loader = get_dataloader(
        root        = config.DATASET_ROOT,
        split       = "val",
        batch_size  = 64,
        num_workers = config.NUM_WORKERS,
    )

    # ── Collect checkpoints & labels ─────────
    ckpt_paths: List[str] = []
    labels:     List[str] = []

    if args.compare:
        ckpt_paths = args.compare
        labels     = args.labels or [Path(c).parent.name for c in ckpt_paths]
    elif args.ckpt:
        ckpt_paths = [args.ckpt]
        labels     = [Path(args.ckpt).parent.name]
    else:
        raise ValueError("Provide either --ckpt or --compare.")

    if len(labels) < len(ckpt_paths):
        labels += [Path(c).parent.name for c in ckpt_paths[len(labels):]]

    fec_rate = _fec_rate(args.fec_type, args.fec_n)
    do_encrypted = not args.analog_only

    # ── Run metrics for every checkpoint ─────
    # Each entry: (analog_results, encrypted_results_or_None)
    all_paired_results = []

    for ckpt_path, label in zip(ckpt_paths, labels):
        # ──────────────── ANALOG ────────────────
        print(f"\n{'═'*60}")
        print(f"  Model : {label}  —  ANALOG")
        print(f"{'═'*60}")

        original = load_model(ckpt_path, args, device)

        results_analog = run_all_metrics(
            model            = original,
            val_loader       = val_loader,
            snr_range        = snr_range,
            device           = device,
            img_h            = config.IMG_SIZE,
            img_w            = config.IMG_SIZE,
            ber_n_batches    = args.ber_n_batches,
            jitter_n_passes  = args.jitter_n_passes,
            jitter_n_batches = args.jitter_n_batches,
            verbose          = True,
            # Analog: no FEC, no digital BER, default forward
            forward_fn       = None,
            fec_rate         = 1.0,
            use_digital_ber  = False,
        )
        results_analog["label"] = f"{label} (Analog)"
        results_analog["mode"]  = "analog"

        print_bandwidth_table(results_analog["bandwidth_usage"],
                              f"{label} (Analog)")
        print_ber_table(results_analog["ber"], f"{label} (Analog)")
        print_jitter_table(results_analog["jitter"], f"{label} (Analog)")
        print_se_table(results_analog["spectral_efficiency"],
                       f"{label} (Analog)")

        # ──────────────── ENCRYPTED ─────────────
        results_enc = None
        if do_encrypted:
            print(f"\n{'═'*60}")
            print(f"  Model : {label}  —  ENCRYPTED")
            print(f"{'═'*60}")

            secure = wrap_secure(original, args, device)

            results_enc = run_all_metrics(
                model            = secure,
                val_loader       = val_loader,
                snr_range        = snr_range,
                device           = device,
                img_h            = config.IMG_SIZE,
                img_w            = config.IMG_SIZE,
                ber_n_batches    = args.ber_n_batches,
                jitter_n_passes  = args.jitter_n_passes,
                jitter_n_batches = args.jitter_n_batches,
                verbose          = True,
                # Encrypted: use digital bridge forward + BER
                forward_fn       = _encrypted_forward,
                fec_rate         = fec_rate,
                fec_type         = args.fec_type,
                fec_n            = args.fec_n,
                quant_bits       = args.quant_bits,
                use_digital_ber  = True,
            )
            results_enc["label"] = f"{label} (Encrypted)"
            results_enc["mode"]  = "encrypted"

            print_bandwidth_table(results_enc["bandwidth_usage"],
                                  f"{label} (Encrypted)")
            print_ber_table(results_enc["ber"], f"{label} (Encrypted)")
            print_jitter_table(results_enc["jitter"],
                               f"{label} (Encrypted)")
            print_se_table(results_enc["spectral_efficiency"],
                           f"{label} (Encrypted)")

        all_paired_results.append((results_analog, results_enc))

    # ── Plots ────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Generating plots …")
    print(f"{'═'*60}")
    plot_ber(all_paired_results, labels, output_dir)
    plot_jitter(all_paired_results, labels, output_dir)
    plot_spectral_efficiency(all_paired_results, labels, output_dir)
    plot_bandwidth_usage(all_paired_results, labels, output_dir)

    # ── Save JSON ────────────────────────────
    if args.save_json:
        def sanitise(obj):
            if isinstance(obj, dict):
                return {k: sanitise(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [sanitise(v) for v in obj]
            if isinstance(obj, float) and (obj != obj or obj == float("inf")):
                return None
            return obj

        # Flatten paired results into a single list for JSON
        flat_results = []
        for r_analog, r_enc in all_paired_results:
            flat_results.append(r_analog)
            if r_enc is not None:
                flat_results.append(r_enc)

        json_path = output_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(sanitise(flat_results), f, indent=2)
        print(f"\nSaved JSON results → {json_path}")

    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
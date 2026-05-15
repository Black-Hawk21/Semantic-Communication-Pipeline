"""
eval.py — Evaluation CLI for Deep JSCC / CAE

Supports evaluating the standard analog model (DeepJSCC), the encrypted
digital-bridge model (DeepJSCCSecure), or both side-by-side.

──────────────────────────────────────────────────────────────────────
Usage examples
──────────────────────────────────────────────────────────────────────

# Single-SNR eval — analog model
python eval.py --ckpt runs/exp/ckpt_best.pt --model_type analog --snr_db 10

# SNR sweep — secure model only
python eval.py --ckpt runs/exp/ckpt_best.pt --model_type secure --sweep

# Side-by-side comparison of both models (sweep + comparison plot)
python eval.py --ckpt runs/exp/ckpt_best.pt --model_type both --sweep

# BER-only sweep across both models
python eval.py --ckpt runs/exp/ckpt_best.pt --model_type both --ber_sweep

# Visual reconstruction grid comparing both models
python eval.py --ckpt runs/exp/ckpt_best.pt --model_type both --visual --snr_db 10

# Secure model with custom FEC settings
python eval.py --ckpt runs/exp/ckpt_best.pt --model_type secure \\
               --fec_type hamming --fec_n 5 --key_seed 42 --sweep
"""

import argparse
import torch

import config
from model            import DeepJSCC
from model_integrated import DeepJSCCSecure
from dataset          import get_dataloader
from evaluate         import Evaluator


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate DeepJSCC (analog) and/or DeepJSCCSecure (encrypted)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to checkpoint (.pt) from training.")

    # ── Model selection ──
    p.add_argument("--model_type", type=str, default="both",
                   choices=["analog", "secure", "both"],
                   help="Which model(s) to evaluate.")

    # ── Architecture ──
    p.add_argument("--latent_channels", type=int,   default=config.LATENT_CHANNELS)
    p.add_argument("--channel",         type=str,   default=config.CHANNEL_TYPE,
                   choices=["awgn", "rayleigh"])
    p.add_argument("--snr_db",          type=float, default=config.SNR_DB,
                   help="SNR for single-point evaluation or visual comparison.")

    # ── Secure model options ──
    p.add_argument("--quant_bits", type=int,   default=8,   choices=[8, 16],
                   help="Quantisation bit depth for the digital bridge.")
    p.add_argument("--fec_type",   type=str,   default="repetition",
                   choices=["repetition", "hamming", "none"],
                   help="FEC codec for the digital bridge.")
    p.add_argument("--fec_n",      type=int,   default=5,
                   help="Repetition factor (odd, >=3). Ignored for Hamming/none.")
    p.add_argument("--key_seed",   type=int,   default=12345,
                   help="Symmetric key seed shared between TX and RX.")

    # ── Evaluation mode ──
    p.add_argument("--sweep",     action="store_true",
                   help="Run PSNR/SSIM/BER sweep across SNR range.")
    p.add_argument("--ber_sweep", action="store_true",
                   help="Run BER-only sweep (faster; uses one batch for BER).")
    p.add_argument("--visual",    action="store_true",
                   help="Generate visual reconstruction comparison grid.")

    # ── Output ──
    p.add_argument("--output_dir", type=str, default="results/eval",
                   help="Directory for plots, CSVs, and visual grids.")
    p.add_argument("--batch_size", type=int, default=64)

    return p.parse_args()


# ─────────────────────────────────────────────
#  Model builders
# ─────────────────────────────────────────────

def build_analog(args, device: str) -> DeepJSCC:
    model = DeepJSCC(
        latent_channels=args.latent_channels,
        channel_type=args.channel,
        snr_db=args.snr_db,
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Analog]  Loaded checkpoint: {args.ckpt}")
    print(f"          Channel={args.channel.upper()}  "
          f"SNR={args.snr_db} dB  "
          f"k/n={model.bandwidth_ratio():.4f}")
    return model


def build_secure(args, device: str, analog_model: DeepJSCC = None) -> DeepJSCCSecure:
    """
    Build DeepJSCCSecure either by wrapping a pre-built analog model or by
    loading the checkpoint directly into a fresh secure model.
    """
    if analog_model is not None:
        # Wrap the already-loaded analog model (shares no weights by reference —
        # from_pretrained copies them, so eval of each is independent)
        secure = DeepJSCCSecure.from_pretrained(
            analog_model,
            quant_bits=args.quant_bits,
            fec_type=args.fec_type,
            fec_n=args.fec_n,
            key_seed=args.key_seed,
        )
    else:
        # Load the checkpoint weights directly
        ckpt = torch.load(args.ckpt, map_location=device)
        analog_tmp = DeepJSCC(
            latent_channels=args.latent_channels,
            channel_type=args.channel,
            snr_db=args.snr_db,
        )
        analog_tmp.load_state_dict(ckpt["model"])

        secure = DeepJSCCSecure.from_pretrained(
            analog_tmp,
            quant_bits=args.quant_bits,
            fec_type=args.fec_type,
            fec_n=args.fec_n,
            key_seed=args.key_seed,
        )

    secure.eval()
    print(f"[Secure]  FEC={args.fec_type}(n={args.fec_n})  "
          f"quant={args.quant_bits}-bit  "
          f"key_seed={args.key_seed}")
    return secure


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run_eval(args: argparse.Namespace):
    device = (
        "cuda" if torch.cuda.is_available()       else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\nDevice : {device}")

    # ── Dataset ──────────────────────────────
    val_loader = get_dataloader(
        root        = config.DATASET_ROOT,
        split       = "val",
        batch_size  = args.batch_size,
        num_workers = config.NUM_WORKERS,
    )

    # ── Build requested model(s) ─────────────
    models = []

    if args.model_type in ("analog", "both"):
        analog = build_analog(args, device)
        models.append(analog)
    else:
        analog = None  # still needed to build secure

    if args.model_type in ("secure", "both"):
        # Pass the already-loaded analog model to avoid double checkpoint I/O
        secure = build_secure(args, device, analog_model=analog)
        models.append(secure)

    print(f"\nEvaluating {len(models)} model(s): "
          + " | ".join(type(m).__name__ for m in models))

    # ── Evaluator ────────────────────────────
    evaluator = Evaluator(
        models     = models,
        val_loader = val_loader,
        device     = device,
        output_dir = args.output_dir,
    )

    snr_range = list(config.SNR_SWEEP_RANGE)

    # ── Dispatch ─────────────────────────────

    if args.sweep:
        if len(models) == 1:
            # Single model: individual sweep + plot
            results = evaluator.snr_sweep(models[0], snr_range=snr_range)
            _print_summary(results)
        else:
            # Multiple models: unified comparison
            all_results = evaluator.compare_sweep(snr_range=snr_range)
            for r in all_results:
                _print_summary(r)

    elif args.ber_sweep:
        evaluator.ber_sweep(snr_range=snr_range)

    elif args.visual:
        evaluator.visual_comparison(snr_db=args.snr_db, n_images=config.EVAL_N_IMAGES)

    else:
        # Default: single-SNR evaluation for all models
        print(f"\n{'─'*60}")
        print(f"  Single-SNR evaluation @ {args.snr_db} dB")
        print(f"{'─'*60}")
        print(f"  {'Model':<36}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'BER':>10}")
        print(f"  {'─'*36}  {'─'*10}  {'─'*8}  {'─'*10}")

        for model in models:
            m = evaluator.evaluate(model, snr_db=args.snr_db)
            name = type(model).__name__
            print(f"  {name:<36}  {m['psnr']:>10.2f}  {m['ssim']:>8.4f}  {m['ber']:>10.2e}")

        print(f"{'─'*60}\n")


def _print_summary(results: dict):
    print(f"\n  Summary — {results['label']}")
    print(f"  {'SNR':>6}  {'PSNR':>9}  {'SSIM':>8}  {'BER':>10}")
    for snr, psnr, ssim, ber in zip(
        results["snr_db"], results["psnr"], results["ssim"], results["ber"]
    ):
        print(f"  {snr:>6.1f}  {psnr:>9.2f}  {ssim:>8.4f}  {ber:>10.2e}")


if __name__ == "__main__":
    run_eval(parse_args())
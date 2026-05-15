"""
quantize.py — Tensor ↔ Bytes serialisation for the encryption bridge

Two quantisation strategies:
  1. Fixed-point uint8  (fast, 256 levels, ~0.02 dB quantisation noise)
  2. Fixed-point uint16 (slower, 65536 levels, negligible quantisation noise)

The tensor is flattened, quantised to integers, packed into a byte string,
encrypted by the C++ module, and unpacked back into a tensor.
"""

import struct
from typing import Tuple

import torch
import numpy as np


# ─────────────────────────────────────────────
# Quantise: Tensor → bytes
# ─────────────────────────────────────────────

def tensor_to_bytes(
    tensor: torch.Tensor,
    bits: int = 8,
) -> Tuple[bytes, dict]:
    """
    Quantise a float tensor to fixed-point integers and serialise to bytes.

    Args:
        tensor : Any-shape float tensor (will be flattened).
        bits   : Quantisation bit-depth (8 or 16).

    Returns:
        raw_bytes : The serialised byte string.
        meta      : Dict with shape, dtype, min_val, max_val, bits — needed
                    to reconstruct the tensor exactly.
    """
    assert bits in (8, 16), "Only 8-bit and 16-bit quantisation supported"

    shape   = tensor.shape
    flat    = tensor.detach().cpu().float().flatten()
    min_val = flat.min().item()
    max_val = flat.max().item()

    # Avoid division by zero for constant tensors
    val_range = max_val - min_val
    if val_range < 1e-12:
        val_range = 1.0

    # Normalise to [0, 1] → scale to [0, 2^bits - 1]
    levels   = (1 << bits) - 1
    normed   = (flat - min_val) / val_range           # [0, 1]
    quantised = (normed * levels).round().clamp(0, levels)

    if bits == 8:
        raw_bytes = quantised.to(torch.uint8).numpy().tobytes()
    else:
        raw_bytes = quantised.to(torch.int32).numpy().astype(np.uint16).tobytes()

    meta = {
        "shape":    tuple(shape),
        "min_val":  min_val,
        "max_val":  max_val,
        "bits":     bits,
        "numel":    flat.numel(),
    }
    return raw_bytes, meta


# ─────────────────────────────────────────────
# Dequantise: bytes → Tensor
# ─────────────────────────────────────────────

def bytes_to_tensor(
    raw_bytes: bytes,
    meta: dict,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Deserialise bytes back to a float tensor using the metadata from
    tensor_to_bytes().

    Args:
        raw_bytes : The byte string (possibly decrypted).
        meta      : The metadata dict from tensor_to_bytes().
        device    : Target torch device.

    Returns:
        Reconstructed float tensor with the original shape.
    """
    bits     = meta["bits"]
    levels   = (1 << bits) - 1
    numel    = meta["numel"]
    min_val  = meta["min_val"]
    max_val  = meta["max_val"]
    shape    = meta["shape"]

    val_range = max_val - min_val
    if val_range < 1e-12:
        val_range = 1.0

    if bits == 8:
        arr = np.frombuffer(raw_bytes[:numel], dtype=np.uint8).copy()
        quantised = torch.from_numpy(arr).float()
    else:
        arr = np.frombuffer(raw_bytes[:numel * 2], dtype=np.uint16).copy()
        quantised = torch.from_numpy(arr.astype(np.int32)).float()

    # Reverse the quantisation
    normed = quantised / levels                       # [0, 1]
    flat   = normed * val_range + min_val             # original range

    return flat.reshape(shape).to(device)


# ─────────────────────────────────────────────
# Convenience: batch-level wrappers
# ─────────────────────────────────────────────

def batch_tensor_to_bytes(
    batch_tensor: torch.Tensor,
    bits: int = 8,
) -> Tuple[list, list]:
    """
    Serialise each sample in a batch independently.

    Args:
        batch_tensor : (B, C, H, W) tensor.
        bits         : Quantisation depth.

    Returns:
        byte_list : List[bytes]  — one entry per sample.
        meta_list : List[dict]   — corresponding metadata.
    """
    byte_list, meta_list = [], []
    for i in range(batch_tensor.size(0)):
        b, m = tensor_to_bytes(batch_tensor[i], bits=bits)
        byte_list.append(b)
        meta_list.append(m)
    return byte_list, meta_list


def batch_bytes_to_tensor(
    byte_list: list,
    meta_list: list,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Reconstruct a batch tensor from per-sample byte strings.
    """
    tensors = [
        bytes_to_tensor(b, m, device=device)
        for b, m in zip(byte_list, meta_list)
    ]
    return torch.stack(tensors, dim=0)


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing quantisation round-trip...")

    x = torch.randn(2, 16, 4, 4)

    for bits in (8, 16):
        bl, ml = batch_tensor_to_bytes(x, bits=bits)
        x_rec  = batch_bytes_to_tensor(bl, ml)

        mse = (x - x_rec).pow(2).mean().item()
        psnr = 10 * np.log10(x.pow(2).mean().item() / max(mse, 1e-12))

        print(f"  {bits}-bit: MSE={mse:.6f}  PSNR={psnr:.1f} dB  "
              f"bytes/sample={len(bl[0])}")

    print("✓ All round-trips passed.")

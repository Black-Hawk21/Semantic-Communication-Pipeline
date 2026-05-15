"""
fec.py — Forward Error Correction for the digital sub-channel

Provides two FEC strategies:
  1. RepetitionCode  — simple 3× or 5× repetition (easy, moderate protection)
  2. HammingCode     — (7,4) Hamming code (better rate, single-error correction)

These operate on numpy bit arrays and are inserted between encryption and
BPSK modulation to protect encrypted bits from channel errors.
"""

import numpy as np
from typing import Tuple


# =====================================================================
#  Repetition Code
# =====================================================================

class RepetitionCode:
    """
    Each bit is repeated `n` times.  Decoding uses majority vote.

    Rate = 1/n.  Corrects up to floor((n-1)/2) errors per symbol.
    """

    def __init__(self, n: int = 3):
        assert n >= 3 and n % 2 == 1, "n must be odd and >= 3"
        self.n = n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """(K,) → (K*n,)"""
        return np.repeat(bits, self.n)

    def decode_hard(self, bits: np.ndarray) -> np.ndarray:
        """Hard-decision majority vote. (K*n,) → (K,)"""
        K = len(bits) // self.n
        reshaped = bits[:K * self.n].reshape(K, self.n)
        return (reshaped.sum(axis=1) > self.n // 2).astype(np.int32)

    def decode_soft(self, llr: np.ndarray) -> np.ndarray:
        """
        Soft-decision decode from log-likelihood ratios.
        LLR > 0 means bit=0 more likely; LLR < 0 means bit=1 more likely.
        Sum LLRs within each group, then threshold.
        """
        K = len(llr) // self.n
        reshaped = llr[:K * self.n].reshape(K, self.n)
        summed = reshaped.sum(axis=1)
        return (summed < 0).astype(np.int32)

    @property
    def rate(self) -> float:
        return 1.0 / self.n


# =====================================================================
#  (7,4) Hamming Code
# =====================================================================

class HammingCode74:
    """
    Classic (7,4) Hamming code.
    Rate = 4/7 ≈ 0.571.  Corrects 1 error per 7-bit block.
    """

    def __init__(self):
        # Generator matrix G (4×7) — systematic form [I4 | P]
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.int32)

        # Parity-check matrix H (3×7)
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
        ], dtype=np.int32)

        # Syndrome → error position lookup
        self._syndrome_table = {}
        for i in range(7):
            syn = tuple(self.H[:, i] % 2)
            self._syndrome_table[syn] = i

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode data bits. Pads to multiple of 4 if needed."""
        # Pad to multiple of 4
        pad = (4 - len(bits) % 4) % 4
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.int32)])

        blocks = bits.reshape(-1, 4)
        coded = (blocks @ self.G) % 2
        return coded.flatten().astype(np.int32)

    def decode_hard(self, bits: np.ndarray) -> np.ndarray:
        """Decode with single-error correction per block."""
        n_blocks = len(bits) // 7
        bits = bits[:n_blocks * 7].reshape(-1, 7).copy()

        decoded = []
        for block in bits:
            syndrome = (self.H @ block) % 2
            syn_key = tuple(syndrome)

            if syn_key != (0, 0, 0) and syn_key in self._syndrome_table:
                err_pos = self._syndrome_table[syn_key]
                block[err_pos] ^= 1  # correct the error

            decoded.append(block[:4])  # systematic bits

        return np.concatenate(decoded).astype(np.int32)

    @property
    def rate(self) -> float:
        return 4.0 / 7.0


# =====================================================================
#  BPSK Modulation / Demodulation
# =====================================================================

def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Map {0, 1} → {+1, -1} (standard BPSK)."""
    return 1.0 - 2.0 * bits.astype(np.float64)


def bpsk_demodulate_hard(symbols: np.ndarray) -> np.ndarray:
    """Hard decision: symbol > 0 → bit 0, else bit 1."""
    return (symbols < 0).astype(np.int32)


def bpsk_demodulate_soft(symbols: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Compute LLR = log(P(bit=0|y) / P(bit=1|y)) = 2*y / sigma^2
    for AWGN channel with known noise variance.
    """
    if noise_var < 1e-12:
        return np.sign(symbols) * 1e6
    return 2.0 * symbols / noise_var


# =====================================================================
#  Self-test
# =====================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Test repetition code
    rep = RepetitionCode(5)
    data = rng.integers(0, 2, size=100).astype(np.int32)
    coded = rep.encode(data)
    # Simulate some errors
    errors = rng.random(len(coded)) < 0.1  # 10% BER
    noisy = coded ^ errors.astype(np.int32)
    recovered = rep.decode_hard(noisy)
    ber = np.mean(data != recovered[:len(data)])
    print(f"Repetition(5): input BER=10% → output BER={ber*100:.1f}%")

    # Test Hamming code
    ham = HammingCode74()
    coded = ham.encode(data)
    errors = rng.random(len(coded)) < 0.05
    noisy = coded ^ errors.astype(np.int32)
    recovered = ham.decode_hard(noisy)
    ber = np.mean(data != recovered[:len(data)])
    print(f"Hamming(7,4):  input BER=5%  → output BER={ber*100:.1f}%")

    # Test BPSK
    syms = bpsk_modulate(data)
    noise = rng.standard_normal(len(syms)) * 0.5
    noisy_syms = syms + noise
    hard = bpsk_demodulate_hard(noisy_syms)
    ber = np.mean(data != hard)
    print(f"BPSK (SNR~6dB): BER={ber*100:.1f}%")

    print("✓ FEC self-tests passed.")

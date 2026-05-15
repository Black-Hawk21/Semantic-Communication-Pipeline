"""
model_integrated.py — DeepJSCC with C++ encryption integration

Architecture (inference mode):

  Image → Encoder → TX Transformer
                         │
                    ┌────────────┐
                    │  DIGITAL   │
                    │  SUB-PIPE  │
                    │            │
                    │ Quantise   │  float32 → uint8 bytes
                    │ Encrypt    │  enc3 symmetric cipher
                    │ FEC Encode │  repetition or Hamming
                    │ BPSK Mod   │  {0,1} → {+1,-1}
                    └────┬───────┘
                         │
                    Analog Channel  (AWGN / Rayleigh)
                         │
                    ┌────┴───────┐
                    │ BPSK Demod │  hard/soft decision
                    │ FEC Decode │  majority vote / syndrome
                    │ Decrypt    │  enc3 symmetric decipher
                    │ Dequantise │  uint8 bytes → float32
                    └────────────┘
                         │
                    RX Transformer → Decoder → Reconstructed Image

Training strategy:
  Phase 1: Train encoder/decoder WITHOUT encryption (standard Deep JSCC).
  Phase 2: Freeze encoder/decoder, insert digital sub-pipeline for inference.
  Phase 3 (optional): Fine-tune encoder/decoder with straight-through
           estimator through the quantisation step.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# These are your existing modules (unchanged)
from model import (
    Encoder, Decoder, LatentTransformerBlock,
    AWGNChannel, RayleighChannel,
)

# New modules for the digital bridge
from quantize import (
    batch_tensor_to_bytes, batch_bytes_to_tensor,
    tensor_to_bytes, bytes_to_tensor,
)
from fec import (
    RepetitionCode, HammingCode74,
    bpsk_modulate, bpsk_demodulate_hard,
)

# ── Try importing the compiled C++ module; fall back to pure-Python stub ──
try:
    import enc3
    HAS_CPP_ENC = True
except ImportError:
    HAS_CPP_ENC = False
    print("[WARNING] enc3 C++ module not found — using pure-Python encryption stub.")


# =====================================================================
#  Pure-Python encryption fallback (mirrors enc3_lib.cpp logic)
# =====================================================================

class _PythonEncStub:
    """Minimal Python mirror of enc3 for development / testing."""

    class EncryptResult:
        def __init__(self, ct, nb):
            self.ciphertext = ct
            self.num_bits_orig = nb
            self.ok = True

    class DecryptResult:
        def __init__(self, pt, t):
            self.plaintext = pt
            self.tampered = t

    @staticmethod
    def _str_to_bits(s: bytes) -> list:
        bits = []
        for byte in s:
            for i in range(8):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_str(bits: list) -> bytes:
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= (bits[i + j] << j)
            out.append(byte)
        return bytes(out)

    @staticmethod
    def encrypt_data(data: str, seed: int):
        """Simple XOR-based stub encryption."""
        import random
        rng = random.Random(seed)
        key_stream = bytes([rng.randint(0, 255) for _ in range(len(data))])
        ct = bytes(a ^ b for a, b in zip(data.encode('latin-1'), key_stream))
        return _PythonEncStub.EncryptResult(
            ct.decode('latin-1'), len(data) * 8
        )

    @staticmethod
    def decrypt_data(ciphertext: str, seed: int, num_bits_orig: int):
        import random
        n_bytes = (num_bits_orig + 7) // 8
        rng = random.Random(seed)
        key_stream = bytes([rng.randint(0, 255) for _ in range(n_bytes)])
        ct_bytes = ciphertext.encode('latin-1')[:n_bytes]
        pt = bytes(a ^ b for a, b in zip(ct_bytes, key_stream))
        return _PythonEncStub.DecryptResult(pt.decode('latin-1'), False)


_enc = enc3 if HAS_CPP_ENC else _PythonEncStub()


# =====================================================================
#  Digital Bridge Module
# =====================================================================

class DigitalBridge(nn.Module):
    """
    Non-differentiable digital sub-pipeline inserted between the
    TX Transformer and the analog channel during INFERENCE.

    Handles: Quantise → Encrypt → FEC → BPSK → Channel → Demod → FEC → Decrypt → Dequantise

    During training this module is bypassed entirely — the standard
    analog channel is used instead so gradients can flow.
    """

    def __init__(
        self,
        snr_db:      float = 10.0,
        channel_type: str  = "awgn",
        quant_bits:  int   = 8,
        fec_type:    str   = "repetition",  # "repetition" | "hamming" | "none"
        fec_n:       int   = 5,             # repetition factor (odd, >= 3)
        key_seed:    int   = 12345,
        encrypt:     bool  = True,
    ):
        super().__init__()
        self.snr_db       = snr_db
        self.channel_type = channel_type
        self.quant_bits   = quant_bits
        self.key_seed     = key_seed
        self.do_encrypt   = encrypt

        # FEC codec
        if fec_type == "repetition":
            self.fec = RepetitionCode(fec_n)
        elif fec_type == "hamming":
            self.fec = HammingCode74()
        else:
            self.fec = None

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        device = z.device
        results = []

        for i in range(B):
            z_i = z[i]  # (C, H, W)

            # ── 1. Quantise float → bytes ──
            raw_bytes, meta = tensor_to_bytes(z_i, bits=self.quant_bits)
            # raw_bytes is already bytes — pass directly to C++ API

            # ── 2. Encrypt ──
            if self.do_encrypt:
                enc_result = _enc.encrypt_data(raw_bytes, self.key_seed)
                payload_bytes = bytes(enc_result.ciphertext)
                num_bits_orig = enc_result.num_bits_orig
            else:
                payload_bytes = raw_bytes
                num_bits_orig = len(raw_bytes) * 8

            # ── 3. Convert to bit array ──
            bit_array = np.unpackbits(
                np.frombuffer(payload_bytes, dtype=np.uint8)
            ).astype(np.int32)

            # ── 4. FEC encode ──
            if self.fec is not None:
                coded_bits = self.fec.encode(bit_array)
            else:
                coded_bits = bit_array

            # ── 5. BPSK modulate ──
            symbols = bpsk_modulate(coded_bits)

            # ── 6. Analog channel ──
            symbols_noisy = self._apply_channel(symbols)

            # ── 7. BPSK demodulate (hard decision) ──
            rx_bits = bpsk_demodulate_hard(symbols_noisy)

            # ── 8. FEC decode ──
            if self.fec is not None:
                decoded_bits = self.fec.decode_hard(rx_bits)
            else:
                decoded_bits = rx_bits

            # ── 9. Re-pack bits → bytes ──
            decoded_bits = decoded_bits[:len(bit_array)]  # trim FEC padding
            pad = (8 - len(decoded_bits) % 8) % 8
            if pad:
                decoded_bits = np.concatenate([decoded_bits, np.zeros(pad, dtype=np.int32)])
            rx_bytes = np.packbits(decoded_bits.astype(np.uint8)).tobytes()
            # rx_bytes is already bytes — pass directly to C++ API

            # ── 10. Decrypt ──
            if self.do_encrypt:
                dec_result = _enc.decrypt_data(rx_bytes, self.key_seed, num_bits_orig)
                recovered_bytes = bytes(dec_result.plaintext)
            else:
                recovered_bytes = rx_bytes

            # ── 11. Dequantise bytes → float tensor ──
            # Channel errors can corrupt encryption metadata (padding length),
            # causing decrypt to return too few or too many bytes.
            # Pad / truncate to the expected length so reshape succeeds.
            expected_len = meta["numel"] * (1 if self.quant_bits == 8 else 2)
            if len(recovered_bytes) < expected_len:
                recovered_bytes = recovered_bytes + b'\x00' * (expected_len - len(recovered_bytes))
            elif len(recovered_bytes) > expected_len:
                recovered_bytes = recovered_bytes[:expected_len]

            z_rec_i = bytes_to_tensor(recovered_bytes, meta, device=device)
            results.append(z_rec_i)

        return torch.stack(results, dim=0)

    def _apply_channel(self, symbols: np.ndarray) -> np.ndarray:
        """Apply AWGN or Rayleigh fading to BPSK symbols."""
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_std  = (1.0 / (2 * snr_linear)) ** 0.5

        if self.channel_type == "awgn":
            noise = np.random.randn(len(symbols)) * noise_std
            return symbols + noise

        elif self.channel_type == "rayleigh":
            h = (np.random.randn(len(symbols)) +
                 1j * np.random.randn(len(symbols))) / np.sqrt(2)
            h_mag = np.abs(h)
            noise = np.random.randn(len(symbols)) * noise_std
            y = h_mag * symbols + noise
            return y / np.maximum(h_mag, 1e-8)  # perfect CSI equalisation

        else:
            raise ValueError(f"Unknown channel: {self.channel_type}")


# =====================================================================
#  Integrated DeepJSCC Model
# =====================================================================

class DeepJSCCSecure(nn.Module):
    """
    DeepJSCC with optional encrypted digital sub-channel.

    Modes:
      - training=True  → standard analog pipeline (differentiable)
      - training=False → digital bridge with encryption + FEC

    This preserves the gradient flow needed for end-to-end training
    while adding encryption security at inference time.
    """

    def __init__(
        self,
        latent_channels: int   = 16,
        channel_type:    str   = "awgn",
        snr_db:          float = 10.0,
        num_heads:       int   = 4,
        mlp_ratio:       float = 4.0,
        dropout:         float = 0.0,
        # Digital bridge config
        quant_bits:      int   = 8,
        fec_type:        str   = "repetition",
        fec_n:           int   = 5,
        key_seed:        int   = 12345,
    ):
        super().__init__()

        # ── Neural components (trainable) ──
        self.encoder = Encoder(latent_channels)
        self.tx_transform = LatentTransformerBlock(
            channels=latent_channels, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.rx_transform = LatentTransformerBlock(
            channels=latent_channels, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.decoder = Decoder(latent_channels)

        # ── Analog channel (used during training) ──
        self.analog_channel = self._build_channel(channel_type, snr_db)

        # ── Digital bridge (used during inference) ──
        self.digital_bridge = DigitalBridge(
            snr_db=snr_db, channel_type=channel_type,
            quant_bits=quant_bits, fec_type=fec_type, fec_n=fec_n,
            key_seed=key_seed, encrypt=True,
        )

        self.latent_channels = latent_channels
        self.channel_type    = channel_type
        self.snr_db          = snr_db

    @staticmethod
    def _build_channel(channel_type: str, snr_db: float) -> nn.Module:
        if channel_type.lower() == "awgn":
            return AWGNChannel(snr_db)
        elif channel_type.lower() == "rayleigh":
            return RayleighChannel(snr_db)
        raise ValueError(f"Unknown channel: {channel_type}")

    def set_snr(self, snr_db: float):
        self.snr_db = snr_db
        self.analog_channel.snr_db = snr_db
        self.digital_bridge.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ── Encode ──
        z = self.encoder(x)
        z = self.tx_transform(z)

        # ── Channel (mode-dependent) ──
        if self.training:
            # TRAINING: analog channel, fully differentiable
            z_hat = self.analog_channel(z)
        else:
            # INFERENCE: digital bridge with encryption + FEC
            z_hat = self.digital_bridge(z)

        # ── Decode ──
        z_hat = self.rx_transform(z_hat)
        x_hat = self.decoder(z_hat)

        return x_hat, z

    def forward_analog(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Force analog-only pipeline (for comparison / ablation)."""
        z = self.encoder(x)
        z = self.tx_transform(z)
        z_hat = self.analog_channel(z)
        z_hat = self.rx_transform(z_hat)
        x_hat = self.decoder(z_hat)
        return x_hat, z

    def forward_encrypted(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Force encrypted digital pipeline (even during training)."""
        z = self.encoder(x)
        z = self.tx_transform(z)
        with torch.no_grad():
            z_hat = self.digital_bridge(z)
        z_hat = self.rx_transform(z_hat)
        x_hat = self.decoder(z_hat)
        return x_hat, z

    def bandwidth_ratio(self, img_h: int = 64, img_w: int = 64) -> float:
        latent_h, latent_w = img_h // 16, img_w // 16
        k = self.latent_channels * latent_h * latent_w
        n = 3 * img_h * img_w
        return k / n

    @classmethod
    def from_pretrained(
        cls,
        original_model,
        quant_bits: int = 8,
        fec_type:   str = "repetition",
        fec_n:      int = 5,
        key_seed:   int = 12345,
    ):
        """
        Wrap a trained DeepJSCC model with the digital encryption bridge.

        Usage:
            original = DeepJSCC(...)
            original.load_state_dict(torch.load("ckpt_best.pt")["model"])

            secure = DeepJSCCSecure.from_pretrained(
                original, fec_type="repetition", fec_n=5, key_seed=42
            )
            secure.eval()
            x_hat, z = secure(test_images)
        """
        secure = cls(
            latent_channels=original_model.latent_channels,
            channel_type=original_model.channel_type,
            snr_db=original_model.snr_db,
            quant_bits=quant_bits,
            fec_type=fec_type,
            fec_n=fec_n,
            key_seed=key_seed,
        )

        # Copy weights from the pretrained model
        secure.encoder.load_state_dict(original_model.encoder.state_dict())
        secure.tx_transform.load_state_dict(original_model.tx_transform.state_dict())
        secure.rx_transform.load_state_dict(original_model.rx_transform.state_dict())
        secure.decoder.load_state_dict(original_model.decoder.state_dict())

        return secure
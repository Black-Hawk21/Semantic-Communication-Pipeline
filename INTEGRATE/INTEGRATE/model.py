"""
model.py — Convolutional Autoencoder (CAE / Deep JSCC Baseline)
Architecture: Encoder → Channel → Decoder
"""

import torch
import torch.nn as nn

import math

class PositionalEncoding2D(nn.Module):
    """
    Learned 2-D positional encoding added to flattened spatial tokens.
    Embeddings are learned, not sinusoidal, which works well for fixed spatial sizes.
    """
    def __init__(self, channels: int, h: int = 4, w: int = 4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class LatentTransformerBlock(nn.Module):
    """
    Transformer block operating on the spatial latent tensor.

    Workflow per forward pass:
        (B, C, H, W)
        → flatten → (B, H*W, C)
        → positional encoding
        → multi-head self-attention  (residual + LN)
        → feed-forward network       (residual + LN)
        → reshape → (B, C, H, W)

    Args:
        channels    (int):  Number of feature channels (= latent_channels).
        num_heads   (int):  Attention heads. Must divide channels evenly.
        mlp_ratio   (float): FFN hidden dim = channels * mlp_ratio.
        dropout     (float): Applied in attention and FFN.
        latent_h/w  (int):  Spatial size of the latent map (default 4×4 for 64-px input).
    """
    def __init__(
        self,
        channels:  int   = 16,
        num_heads: int   = 4,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
        latent_h:  int   = 4,
        latent_w:  int   = 4,
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.latent_h = latent_h
        self.latent_w = latent_w

        self.pos_enc = PositionalEncoding2D(channels, latent_h, latent_w)

        self.norm1   = nn.LayerNorm(channels)
        self.attn    = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # expects (B, T, C)
        )

        self.norm2   = nn.LayerNorm(channels)
        hidden_dim   = int(channels * mlp_ratio)
        self.ffn     = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.pos_enc(tokens)

        # Self-attention with pre-norm (more stable than post-norm)
        normed   = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens   = tokens + attn_out

        # FFN with pre-norm
        tokens   = tokens + self.ffn(self.norm2(tokens))

        # (B, H*W, C) → (B, C, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)

# ─────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ConvTBNReLU(nn.Sequential):
    """ConvTranspose2d → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1):
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Maps image (B, 3, H, W) → latent (B, latent_channels, H', W')

    Spatial downsampling:  64 → 32 → 16 → 8 → 4
    Channel progression:   3  → 64 → 128 → 256 → 1024 → latent_channels
    """
    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(3,   64,  3, stride=1, padding=1),   # 64×64
            ConvBNReLU(64,  64,  3, stride=2, padding=1),   # 32×32
            ConvBNReLU(64,  128, 3, stride=1, padding=1),
            ConvBNReLU(128, 128, 3, stride=2, padding=1),   # 16×16
            ConvBNReLU(128, 256, 3, stride=1, padding=1),
            ConvBNReLU(256, 256, 3, stride=2, padding=1),   # 8×8
            ConvBNReLU(256, 1024, 3, stride=1, padding=1),
            ConvBNReLU(1024, 1024, 3, stride=2, padding=1),  # 4x4
            # Project to latent_channels — no activation (raw channel symbols)
            nn.Conv2d(1024, latent_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Channel Models
# ─────────────────────────────────────────────

class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise channel.

    Power-normalises the signal before adding noise, so the
    SNR is well-defined: snr_db = 10 * log10(signal_power / noise_power).
    """
    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.snr_db == float("inf"):
            return x

        # Per-sample power normalisation
        power = x.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt().clamp(min=1e-8)
        x_norm = x / power

        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_std  = (1.0 / snr_linear) ** 0.5
        noise      = torch.randn_like(x_norm) * noise_std

        return (x_norm + noise) * power   # re-scale back

    def extra_repr(self) -> str:
        return f"snr_db={self.snr_db}"


class RayleighChannel(nn.Module):
    """
    Flat Rayleigh fading channel (complex fading, per symbol).
    h ~ CN(0,1), y = h*x + n
    Includes perfect CSI equalisation at the receiver (divide by |h|).
    """
    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat each spatial location / channel as one symbol pair
        shape = x.shape
        # Complex fading coefficient per element
        h_real = torch.randn_like(x)
        h_imag = torch.randn_like(x)
        h_mag  = (h_real.pow(2) + h_imag.pow(2)).sqrt() / (2 ** 0.5)

        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_std  = (1.0 / (2 * snr_linear)) ** 0.5

        y = h_mag * x + torch.randn_like(x) * noise_std
        # Perfect CSI equalisation
        return y / h_mag.clamp(min=1e-8)

    def extra_repr(self) -> str:
        return f"snr_db={self.snr_db}"


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Maps latent (B, latent_channels, H', W') → reconstructed image (B, 3, H, W)

    Spatial upsampling:    4 → 8 → 16 → 32
    Channel regression:    latent_channels → 256 → 128 → 64 → 3
    """
    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 1024, kernel_size=1, bias=True),
            ConvBNReLU(1024, 1024, 3, stride=1, padding=1),
            ConvTBNReLU(1024, 256),                         # 4 → 8
            ConvBNReLU(256, 256, 3, stride=1, padding=1),
            ConvTBNReLU(256, 128),                          # 8 → 16
            ConvBNReLU(128, 128, 3, stride=1, padding=1),
            ConvTBNReLU(128, 64),                           # 16 → 32
            ConvBNReLU(64,  64,  3, stride=1, padding=1),
            ConvTBNReLU(64,  64),                           # 32 → 64
            # Final projection — Tanh keeps output in [-1, 1]
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────
# Full CAE / Deep JSCC Model
# ─────────────────────────────────────────────

class DeepJSCC(nn.Module):
    """
    End-to-end Convolutional Autoencoder for semantic image communication.

    Pipeline:
        Image  ──► Encoder ──► Channel ──► Decoder ──► Reconstructed Image

    Args:
        latent_channels (int):  Number of feature-map channels in the latent space.
                                Controls the channel bandwidth ratio k/n.
        channel_type    (str):  'awgn' | 'rayleigh'
        snr_db          (float): Operating SNR in dB.
    """
    def __init__(
        self,
        latent_channels: int   = 16,
        channel_type:    str   = "awgn",
        snr_db:          float = 10.0,
        num_heads:       int   = 4,     # new
        mlp_ratio:       float = 4.0,   # new
        dropout:         float = 0.0,   # new
    ):
        super().__init__()
        self.encoder  = Encoder(latent_channels)
        self.tx_transform = LatentTransformerBlock(   # <── added
            channels=latent_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.channel  = self._build_channel(channel_type, snr_db)
        self.rx_transform = LatentTransformerBlock(   # <── added
            channels=latent_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.decoder  = Decoder(latent_channels)

        self.latent_channels = latent_channels
        self.channel_type    = channel_type
        self.snr_db          = snr_db

    @staticmethod
    def _build_channel(channel_type: str, snr_db: float) -> nn.Module:
        channel_type = channel_type.lower()
        if channel_type == "awgn":
            return AWGNChannel(snr_db)
        elif channel_type == "rayleigh":
            return RayleighChannel(snr_db)
        else:
            raise ValueError(f"Unknown channel_type '{channel_type}'. Choose 'awgn' or 'rayleigh'.")

    def set_snr(self, snr_db: float):
        """Hot-swap the SNR without rebuilding the model (useful during evaluation)."""
        self.snr_db         = snr_db
        self.channel.snr_db = snr_db

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent tensor before channel corruption."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from (possibly noisy) latent tensor."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat (Tensor): Reconstructed image, same shape as x.
            z     (Tensor): Latent before channel (for analysis / aux losses).
        """
        z     = self.encoder(x)
        z     = self.tx_transform(z)
        z_hat = self.channel(z)
        z_hat = self.rx_transform(z_hat)
        x_hat = self.decoder(z_hat)
        return x_hat, z

    def bandwidth_ratio(self, img_h: int = 32, img_w: int = 32) -> float:
        """
        k/n — ratio of transmitted symbols to source pixels.
        k = latent_channels * (H/8) * (W/8)
        n = 3 * H * W
        """
        latent_h = img_h // 8
        latent_w = img_w // 8
        k = self.latent_channels * latent_h * latent_w
        n = 3 * img_h * img_w
        return k / n

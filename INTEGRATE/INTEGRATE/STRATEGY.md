# Deep JSCC × C++ Encryption: Integration Strategy

## The Fundamental Tension

Your system has an inherent architectural conflict. Deep JSCC is an **analog** system — it transmits continuous-valued symbols and degrades gracefully with noise. Your C++ encryption is a **digital** system — a single flipped bit destroys the MAC and cascades into garbled output. These two paradigms cannot occupy the same wire without a bridge.

There are three ways to resolve this. I recommend **Approach A** as the primary solution and provide full code for it.

---

## Approach A: Hybrid Digital Sub-Channel (Recommended)

**Core idea:** Keep the trained neural encoder/decoder, but insert a complete digital communication pipeline between them. The encrypted bits are protected by FEC and transmitted via BPSK modulation through the analog channel.

```
                         TRAINING (differentiable)
  Image → Encoder → TxTF ──→ AWGN/Rayleigh ──→ RxTF → Decoder → Recon

                         INFERENCE (encrypted)
  Image → Encoder → TxTF ──→ ┌─────────────────────┐ ──→ RxTF → Decoder → Recon
                              │ Quantise (float→u8)  │
                              │ Encrypt  (enc3.cpp)   │
                              │ FEC Encode (Rep-5)    │
                              │ BPSK Mod  (0→+1,1→-1) │
                              │ ── Analog Channel ──  │
                              │ BPSK Demod            │
                              │ FEC Decode            │
                              │ Decrypt               │
                              │ Dequantise (u8→float) │
                              └─────────────────────┘
```

**Why this works:**
- Training uses the standard analog channel (gradients flow normally).
- At inference, the digital bridge quantises, encrypts, and protects the bits before they touch the channel.
- FEC (repetition-5 or Hamming-7,4) corrects enough errors that the MAC check passes.
- The encoder/decoder have already learned a good latent representation — we just need to transmit it reliably.

**Trade-off:** You lose the graceful analog degradation of pure Deep JSCC at inference time. At low SNR, the FEC will hit its correction limit and you get a cliff effect (works perfectly above a threshold, fails below it). This is the fundamental price of adding digital encryption.

---

## Approach B: Learned Invertible Encryption (Alternative)

Replace the C++ cipher with a differentiable, noise-tolerant "encryption" layer:

```python
class LearnedCipher(nn.Module):
    def __init__(self, channels, key_dim=64):
        super().__init__()
        # Key-conditioned mixing network
        self.key_net = nn.Linear(key_dim, channels * channels)
        self.channels = channels
    
    def forward(self, z, key_embedding):
        B, C, H, W = z.shape
        W_mat = self.key_net(key_embedding).view(B, C, C)
        z_flat = z.flatten(2)                    # (B, C, H*W)
        z_enc = torch.bmm(W_mat, z_flat)         # learned mixing
        return z_enc.view(B, C, H, W)
```

This preserves analog graceful degradation and is fully differentiable, but the security properties are those of a learned scramble — not a proven cryptographic primitive. If your threat model requires standard cryptography, use Approach A. If you need noise resilience above all, consider this.

---

## Approach C: Encrypt the Source Image (Application Layer)

Encrypt the raw pixel data *before* the neural encoder, decrypt *after* the decoder. This separates the cryptography from the channel entirely.

**Problem:** The decoder outputs a continuous approximation of the encrypted image. Even small floating-point errors in the reconstruction mean the decrypted output is garbage — the encryption amplifies reconstruction noise rather than gracefully degrading. This approach only works if the PSNR of your autoencoder is extremely high (>45 dB), which is unlikely at useful compression ratios.

---

## Delivered Code Files

| File | Purpose |
|------|---------|
| `enc3_lib.cpp` | Refactored C++ — removed `main()`, deterministic padding, pybind11 module |
| `build_enc3.sh` | Shell script to compile the Python extension |
| `quantize.py` | Float tensor ↔ byte string serialisation (8-bit and 16-bit) |
| `fec.py` | Forward error correction (Repetition-N, Hamming-7,4, BPSK mod/demod) |
| `model_integrated.py` | `DeepJSCCSecure` — integrated model with mode-switched forward pass |

---

## How to Use

### Step 1: Build the C++ extension

```bash
pip install pybind11
cd integration/
chmod +x build_enc3.sh
./build_enc3.sh
```

### Step 2: Train (Phase 1 — no encryption)

Train the standard DeepJSCC model as before. The analog channel provides the gradient signal:

```bash
python train.py --epochs 100 --snr_db 10 --channel awgn
```

### Step 3: Wrap for secure inference

```python
from model import DeepJSCC
from model_integrated import DeepJSCCSecure

# Load your trained model
original = DeepJSCC(latent_channels=16, channel_type="awgn", snr_db=10.0)
ckpt = torch.load("models/ckpt_best.pt", map_location="cpu")
original.load_state_dict(ckpt["model"])

# Wrap with encryption bridge
secure = DeepJSCCSecure.from_pretrained(
    original,
    quant_bits=8,           # 256 quantisation levels
    fec_type="repetition",  # "repetition" or "hamming"
    fec_n=5,                # 5× repetition (corrects up to 2 errors per symbol)
    key_seed=42,            # symmetric key seed (shared secret)
)
secure.eval()

# Run inference with encryption
with torch.no_grad():
    x_hat, z = secure(test_images)
```

### Step 4: Benchmark

Compare the three modes to understand the cost of encryption:

```python
secure.eval()

# Mode 1: Pure analog (no encryption, maximum PSNR)
x_hat_analog, _ = secure.forward_analog(images)

# Mode 2: Encrypted digital bridge
x_hat_secure, _ = secure.forward_encrypted(images)

# Compare PSNR across SNR sweep
for snr in range(0, 21, 2):
    secure.set_snr(snr)
    # ... measure PSNR for both modes
```

---

## Key Design Decisions Explained

### Why 8-bit quantisation?
The latent tensor has shape `(B, 16, 4, 4)` = 256 values per sample. At 8-bit, that's 256 bytes per image — a 48× compression from the 12,288 source pixels. The quantisation noise (~0.02 dB) is negligible compared to channel noise. Use 16-bit if you need higher fidelity at the cost of doubling the transmitted bits.

### Why repetition-5 over Hamming?
At moderate SNR (≥8 dB), repetition-5 with BPSK gives a post-FEC BER below 10⁻⁶, which is sufficient for the MAC to pass. Hamming-7,4 has better rate (0.57 vs 0.20) but only corrects 1 error per 7-bit block — it fails faster at low SNR. For your use case, reliability matters more than bandwidth efficiency since the latent is already heavily compressed.

### Why deterministic padding in the refactored C++?
The original `enc3.cpp` used `std::random_device` for the padding RNG, making the padding non-reproducible. The decryptor needed to independently derive `x` (the padding amount) from the 3 metadata bits embedded in the ciphertext. This works — but only if those 3 bits survive the channel. By making the padding deterministic (seeded from the key), the decryptor can independently compute the exact same padding without relying on fragile metadata bits.

### Why not a BSC instead of AWGN+BPSK?
Simulating a Binary Symmetric Channel directly would be simpler, but it decouples your system from the actual channel physics. By keeping the AWGN/Rayleigh channel and adding BPSK modulation, the BER is a function of the real channel SNR — your results are physically meaningful and comparable to the literature.

---

## SNR Operating Regions

| SNR (dB) | BPSK BER (AWGN) | Post-FEC BER (Rep-5) | MAC Status |
|----------|-----------------|----------------------|------------|
| 0        | ~0.079          | ~0.006               | Likely fails |
| 4        | ~0.012          | ~2×10⁻⁵              | Usually OK |
| 6        | ~0.0024         | ~1×10⁻⁸              | Passes |
| 8        | ~0.00019        | ~negligible           | Passes |
| 10+      | <10⁻⁵           | ~0                    | Passes |

The system is reliable above ~5 dB SNR with repetition-5 FEC. Below that, consider increasing the repetition factor to 7 or 9 (at the cost of more transmitted symbols).

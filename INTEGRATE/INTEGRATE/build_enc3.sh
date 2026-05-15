#!/usr/bin/env bash
# build_enc3.sh — Compile enc3_lib.cpp into a Python extension module
#
# Run this once from the directory containing enc3_lib.cpp:
#   chmod +x build_enc3.sh
#   ./build_enc3.sh

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# 1. Verify source file exists
# ─────────────────────────────────────────────────────────────────────
if [[ ! -f enc3_lib.cpp ]]; then
    echo "✗ enc3_lib.cpp not found in current directory ($(pwd))"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# 2. Ensure pybind11 is installed
# ─────────────────────────────────────────────────────────────────────
echo "==> Checking / installing pybind11..."
pip install pybind11 --quiet

# ─────────────────────────────────────────────────────────────────────
# 3. Resolve compiler flags
# ─────────────────────────────────────────────────────────────────────
PYBIND_INC=$(python3 -m pybind11 --includes)
PYTHON_INC=$(python3-config --includes)
EXT_SUFFIX=$(python3-config --extension-suffix)   # e.g. .cpython-312-x86_64-linux-gnu.so
OUTPUT="enc3${EXT_SUFFIX}"

echo "==> pybind11 include : ${PYBIND_INC}"
echo "==> Python include   : ${PYTHON_INC}"
echo "==> Output file      : ${OUTPUT}"

# ─────────────────────────────────────────────────────────────────────
# 4. Compile
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> Compiling enc3_lib.cpp..."

g++ -O3 -shared -std=c++17 -fPIC \
    -DPYBIND11_BUILD \
    ${PYTHON_INC} \
    ${PYBIND_INC} \
    enc3_lib.cpp \
    -o "${OUTPUT}"

echo ""
echo "✓ Build successful → ${OUTPUT}"

# ─────────────────────────────────────────────────────────────────────
# 5. Smoke-test
#
# enc3 exposes ciphertext / plaintext as py::bytes (raw binary).
# No latin-1 encoding tricks needed — pass bytes in, get bytes out.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> Running smoke-test..."

python3 - <<'PYEOF'
import sys, enc3

SEED = 12345

# ── Round-trip ──────────────────────────────────────────────────────
plaintext = b"hello deep jscc"          # bytes throughout

r = enc3.encrypt_data(plaintext, SEED)  # pass bytes directly

d = enc3.decrypt_data(r.ciphertext, SEED, r.num_bits_orig)

recovered = bytes(d.plaintext)[:len(plaintext)]
assert recovered == plaintext, f"Round-trip failed! Got: {recovered!r}"
assert not d.tampered,         "MAC tampered flag set unexpectedly!"
print("  ✓ encrypt → decrypt round-trip OK")

# ── Tamper detection ─────────────────────────────────────────────────
ct_bad = bytearray(r.ciphertext)
ct_bad[5] ^= 0x01
d2 = enc3.decrypt_data(bytes(ct_bad), SEED, r.num_bits_orig)
assert d2.tampered, "Tamper not detected — MAC check may be broken!"
print("  ✓ single-bit corruption detected by MAC")

# ── Wrong key ────────────────────────────────────────────────────────
d3 = enc3.decrypt_data(r.ciphertext, SEED + 1, r.num_bits_orig)
assert bytes(d3.plaintext)[:len(plaintext)] != plaintext or d3.tampered, \
    "Wrong key produced correct plaintext — key derivation may be broken!"
print("  ✓ wrong key yields incorrect / tampered output")

print()
print("  enc3 C++ module is ready.")
PYEOF
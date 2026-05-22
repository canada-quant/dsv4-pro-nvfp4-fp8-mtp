"""MXFP4 (group=32, E8M0 scale) → NVFP4 (group=16, FP8-E4M3 scale) conversion.

Single-tensor reference implementation in pure NumPy. CPU-only.

For each expert weight tensor in the native DeepSeek-V4-Pro release:
  - weight is I8-packed (2 FP4 e2m1 values per byte)
  - scale is F8_E8M0 (1 byte per 32-element group along the inner dim)

For the NVFP4 output:
  - weight is I8-packed (2 FP4 e2m1 values per byte, same packing)
  - scale is F8_E4M3 (1 byte per 16-element group along the inner dim)
  - per-tensor FP32 global_scale S_g absorbs out-of-E4M3-range exponents

Math:
  1. Unpack: read I8 byte → low_nibble + high_nibble → 2 FP4 e2m1 values
  2. Dequantize: each FP4 × E8M0 scale → FP32 weight (BF16-equivalent)
  3. Per-tensor S_g: choose so max(per_group_amax / S_g) fits in E4M3 range
  4. Re-group inner dim from groups of 32 to groups of 16
  5. Per-16-group: S_l = round_to_e4m3(amax_in_group / S_g / FP4_MAX)
  6. Per-element: w_q = round_to_fp4(w / (S_g * S_l))
  7. Pack: 2 FP4 values per I8 byte

Reference constants:
  FP4 e2m1 representable values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  FP4 maximum positive: 6.0
  E4M3 (OCP MX) range: subnormal 2^-9 ... finite max 1.75 * 2^8 = 448

This file is the math reference. Production conversion will run as a
Torch/CUDA kernel over full shards; this script verifies the math is
correct on one tensor at a time.
"""

from __future__ import annotations

import json
import struct
import urllib.request
import sys
from dataclasses import dataclass

import numpy as np


# ---------- FP4 e2m1 codec ----------

# FP4 e2m1 (NVFP4 / MXFP4 element format): 4-bit, 1 sign + 2 exponent + 1 mantissa.
# Lookup: index 0..7 → magnitude.
FP4_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MAX = 6.0


def fp4_unpack(packed_bytes: np.ndarray) -> np.ndarray:
    """Unpack I8-packed FP4 values to float32. Two FP4 values per byte.

    Convention (DeepSeek native + NVFP4): low nibble is the first element,
    high nibble is the second.
    Sign bit is bit 3 of the nibble; magnitude index is bits 0..2.
    """
    assert packed_bytes.dtype == np.uint8
    low = packed_bytes & 0x0F
    high = (packed_bytes >> 4) & 0x0F

    def nibble_to_float(nib: np.ndarray) -> np.ndarray:
        sign = (nib & 0x08) >> 3  # 1 if negative
        mag_idx = nib & 0x07
        mag = FP4_E2M1_MAGNITUDES[mag_idx]
        return np.where(sign.astype(bool), -mag, mag)

    out = np.empty(packed_bytes.size * 2, dtype=np.float32)
    out[0::2] = nibble_to_float(low)
    out[1::2] = nibble_to_float(high)
    return out


def fp4_pack(values: np.ndarray) -> np.ndarray:
    """Pack a flat array of FP4-quantized float values into I8 bytes.

    Inverse of fp4_unpack. Input values must already be on the FP4 grid
    (use fp4_quantize first). Length must be even.
    """
    assert values.size % 2 == 0
    flat = values.astype(np.float32).copy()
    sign = (flat < 0).astype(np.uint8) << 3
    mag = np.abs(flat)
    # Find closest magnitude index
    diffs = np.abs(mag[:, None] - FP4_E2M1_MAGNITUDES[None, :])
    mag_idx = diffs.argmin(axis=1).astype(np.uint8)
    nibbles = sign | mag_idx
    low = nibbles[0::2]
    high = nibbles[1::2]
    return ((high << 4) | low).astype(np.uint8)


def fp4_quantize(values: np.ndarray) -> np.ndarray:
    """Round each value to the nearest FP4 e2m1 magnitude (preserving sign)."""
    sign = np.sign(values).astype(np.float32)
    mag = np.abs(values).astype(np.float32)
    diffs = np.abs(mag[..., None] - FP4_E2M1_MAGNITUDES[None, :])
    mag_idx = diffs.argmin(axis=-1)
    out = sign * FP4_E2M1_MAGNITUDES[mag_idx]
    return out.astype(values.dtype)


# ---------- E8M0 codec (native DeepSeek scale format) ----------

# E8M0 is an 8-bit pure-exponent format. byte value b represents 2^(b - 127).
# byte 0 is reserved as zero in some interpretations; byte 255 as NaN.

def e8m0_to_float(bytes_in: np.ndarray) -> np.ndarray:
    """Decode E8M0 bytes to float32. byte b → 2^(b - 127)."""
    assert bytes_in.dtype == np.uint8
    return np.exp2(bytes_in.astype(np.int32) - 127).astype(np.float32)


# ---------- E4M3 codec (NVFP4 scale format, OCP MX bias=7) ----------

# E4M3 (OCP MX): 1 sign + 4 exponent (bias=7) + 3 mantissa.
# Normal: 2^(e-7) * (1 + m/8) for e in 1..15, m in 0..7
# Subnormal: 2^-6 * (m/8) for e=0, m in 1..7 — range 2^-9 .. 7*2^-9 = 7/512
# Finite max: 1.875 * 2^8 = 480 (OCP MX); some specs cap at 448 (NaN reserved)
# For our purposes we use NVFP4's convention: max 448, NaN at 0x7F/0xFF, bias=7.

E4M3_BIAS = 7
E4M3_MAX = 448.0  # NVFP4 convention; max finite positive
# Build a lookup of all 128 positive E4M3 values (sign=0).
def _build_e4m3_table() -> np.ndarray:
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 0 and m == 0:
                vals.append(0.0)
            elif e == 0:
                vals.append((2 ** -6) * (m / 8.0))
            else:
                # Skip NaN encodings (e=15, m=7 in NVFP4 spec). We treat 1.875*2^8=480 as max
                # for the table but cap to FINITE_MAX_448 = 1.75*2^8 in practice.
                vals.append((2 ** (e - E4M3_BIAS)) * (1 + m / 8.0))
    return np.array(vals, dtype=np.float32)


E4M3_POSITIVE_VALUES = _build_e4m3_table()


def f32_to_e4m3(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32 values to the nearest E4M3 representable value.

    Returns (representable_float_values, encoded_bytes).

    Values outside finite range are clamped to ±E4M3_MAX. Signs are preserved
    via bit 7 of the encoded byte.
    """
    sign = (values < 0).astype(np.uint8) << 7
    mag = np.abs(values).astype(np.float32)
    mag = np.clip(mag, 0.0, E4M3_MAX)
    diffs = np.abs(mag[..., None] - E4M3_POSITIVE_VALUES[None, :])
    code = diffs.argmin(axis=-1).astype(np.uint8)
    chosen_mag = E4M3_POSITIVE_VALUES[code]
    chosen_signed = np.where(values < 0, -chosen_mag, chosen_mag)
    encoded = sign | code
    return chosen_signed.astype(np.float32), encoded


# ---------- core conversion ----------

@dataclass
class ConversionResult:
    out_weight_bytes: np.ndarray  # I8-packed FP4
    out_scale_bytes: np.ndarray   # I8-packed E4M3 codes (one per 16-element group)
    out_global_scale: float       # F32 per-tensor scalar
    max_abs_err: float
    mean_abs_err: float
    max_rel_err: float
    mean_rel_err: float
    source_amax: float
    out_amax_after_dequant: float


def mxfp4_dequantize(
    weight_i8: np.ndarray,  # shape [out, in/2], dtype uint8
    scale_e8m0: np.ndarray,  # shape [out, in/32], dtype uint8
) -> np.ndarray:
    """Dequantize MXFP4 (native DeepSeek FP4 + E8M0) to float32.

    Returns full-precision weight of shape [out, in].
    """
    out_dim, packed_in = weight_i8.shape
    in_dim = packed_in * 2  # unpacked
    s_out, s_in = scale_e8m0.shape
    assert s_out == out_dim, f"{s_out} != {out_dim}"
    assert s_in == in_dim // 32, f"{s_in} != {in_dim // 32}"

    # Unpack FP4 magnitudes
    fp4_vals = fp4_unpack(weight_i8.reshape(-1)).reshape(out_dim, in_dim)
    # Decode scales
    scales = e8m0_to_float(scale_e8m0)  # [out, in/32]
    # Broadcast scale across each 32-element group
    scales_expanded = np.repeat(scales, 32, axis=1)  # [out, in]
    return fp4_vals * scales_expanded


def convert_mxfp4_to_nvfp4(
    weight_i8: np.ndarray,
    scale_e8m0: np.ndarray,
    *,
    group_size_in: int = 32,
    group_size_out: int = 16,
) -> ConversionResult:
    """Convert one MXFP4 tensor to NVFP4 format.

    Returns the new I8-packed weight bytes, E4M3-encoded scale bytes, the
    per-tensor FP32 global scale S_g, plus round-trip error statistics.
    """
    out_dim, packed_in = weight_i8.shape
    in_dim = packed_in * 2
    assert in_dim % group_size_out == 0, f"in_dim {in_dim} not divisible by {group_size_out}"

    # Step 1+2: dequantize source to BF16-equivalent float32
    src = mxfp4_dequantize(weight_i8, scale_e8m0)  # [out, in]
    source_amax = float(np.max(np.abs(src)))

    # Step 4: regroup into NVFP4 group=16 tiles along the inner dim
    n_groups_out = in_dim // group_size_out
    grouped = src.reshape(out_dim, n_groups_out, group_size_out)
    per_group_amax = np.max(np.abs(grouped), axis=-1)  # [out, n_groups_out]

    # Step 3: per-tensor S_g.
    # NVFP4 convention: w = S_g * S_l * fp4_val, with S_l in E4M3 (max 448),
    # fp4_val in [-6, 6]. Choose S_g so that max(per_group_amax / FP4_MAX) / S_g
    # equals 448 (E4M3 max) — i.e., S_g exactly saturates the most-extreme group.
    # That maximizes precision; outliers do NOT get clipped by S_g.
    max_required_scale = float(per_group_amax.max() / FP4_MAX)
    s_g = max_required_scale / E4M3_MAX if max_required_scale > 0 else 1.0
    # Numeric safety floor
    if s_g <= 0:
        s_g = 1.0

    # Step 5: per-group S_l in E4M3
    # S_l = per_group_amax / FP4_MAX / S_g, then quantize to E4M3
    raw_s_l = per_group_amax / FP4_MAX / s_g  # [out, n_groups_out], float32 positive
    s_l_e4m3, s_l_codes = f32_to_e4m3(raw_s_l)  # both [out, n_groups_out]
    # Avoid divide-by-zero on all-zero groups
    s_l_e4m3 = np.where(s_l_e4m3 > 0, s_l_e4m3, 1.0).astype(np.float32)

    # Step 6: quantize each weight
    # full effective scale per element is S_g * S_l_e4m3 (per group)
    full_scale = s_g * s_l_e4m3  # [out, n_groups_out]
    full_scale_expanded = np.repeat(full_scale, group_size_out, axis=1)  # [out, in]
    # Normalize, then round to FP4 grid
    normalized = src / full_scale_expanded
    quantized = fp4_quantize(normalized)
    # Reconstruct (dequant) for error measurement
    reconstructed = quantized * full_scale_expanded

    # Step 7: pack to I8
    out_weight_bytes = fp4_pack(quantized.reshape(-1)).reshape(out_dim, in_dim // 2)

    # Error metrics
    abs_err = np.abs(src - reconstructed)
    max_abs = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    # Relative error only where |src| is not tiny (avoid div-by-zero noise)
    src_abs = np.abs(src)
    mask = src_abs > 1e-6
    rel_err = np.zeros_like(abs_err)
    rel_err[mask] = abs_err[mask] / src_abs[mask]
    max_rel = float(rel_err.max())
    mean_rel = float(rel_err.mean())

    return ConversionResult(
        out_weight_bytes=out_weight_bytes,
        out_scale_bytes=s_l_codes,
        out_global_scale=s_g,
        max_abs_err=max_abs,
        mean_abs_err=mean_abs,
        max_rel_err=max_rel,
        mean_rel_err=mean_rel,
        source_amax=source_amax,
        out_amax_after_dequant=float(np.max(np.abs(reconstructed))),
    )


# ---------- safetensors fetch helper (range-read one tensor from HF) ----------

def fetch_tensor_from_hf(
    repo: str,
    shard: str,
    key_weight: str,
    key_scale: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Range-read the safetensors header + the two tensors (weight + scale)
    from a specific shard. Avoids downloading the whole shard (~13 GB)."""
    base = f"https://huggingface.co/{repo}/resolve/main"
    url = f"{base}/{shard}"

    # Get header
    req = urllib.request.Request(url, headers={"Range": "bytes=0-2097151"})
    buf = urllib.request.urlopen(req).read()
    hlen = struct.unpack("<Q", buf[:8])[0]
    if hlen > len(buf) - 8:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{hlen + 1024}"})
        buf = urllib.request.urlopen(req).read()
    header = json.loads(buf[8 : 8 + hlen])
    hdr_size = 8 + hlen

    weight_meta = header[key_weight]
    scale_meta = header[key_scale]

    def read_range(meta):
        start, end = meta["data_offsets"]
        fstart, fend = hdr_size + start, hdr_size + end - 1
        req2 = urllib.request.Request(url, headers={"Range": f"bytes={fstart}-{fend}"})
        return urllib.request.urlopen(req2).read()

    w_bytes = read_range(weight_meta)
    s_bytes = read_range(scale_meta)

    # weight is I8 (dtype "I8" in safetensors) → uint8 numpy
    w_arr = np.frombuffer(w_bytes, dtype=np.uint8).reshape(weight_meta["shape"])
    s_arr = np.frombuffer(s_bytes, dtype=np.uint8).reshape(scale_meta["shape"])
    return w_arr, s_arr, {"weight": weight_meta, "scale": scale_meta}


# ---------- CLI driver ----------

def main():
    """Range-fetch one V4-Pro expert weight+scale and run the conversion."""
    REPO = "deepseek-ai/DeepSeek-V4-Pro"
    # Layer 30 expert 0 w1 lives in shard 00032 (verified earlier)
    SHARD = "model-00032-of-00064.safetensors"
    KEY_W = "layers.30.ffn.experts.0.w1.weight"
    KEY_S = "layers.30.ffn.experts.0.w1.scale"

    print(f"Fetching {KEY_W} from {REPO}/{SHARD}...")
    w_i8, s_e8m0, meta = fetch_tensor_from_hf(REPO, SHARD, KEY_W, KEY_S)
    print(f"  weight: dtype={meta['weight']['dtype']} shape={meta['weight']['shape']} bytes={w_i8.nbytes}")
    print(f"  scale : dtype={meta['scale']['dtype']}  shape={meta['scale']['shape']} bytes={s_e8m0.nbytes}")

    print()
    print("=== Conversion ===")
    result = convert_mxfp4_to_nvfp4(w_i8, s_e8m0)
    print(f"  source amax:              {result.source_amax:.6f}")
    print(f"  output amax (dequant):    {result.out_amax_after_dequant:.6f}")
    print(f"  per-tensor S_g:           {result.out_global_scale:.6e}")
    print(f"  output weight bytes:      {result.out_weight_bytes.nbytes:,} (FP4 packed)")
    print(f"  output scale  bytes:      {result.out_scale_bytes.nbytes:,} (E4M3)")
    print()
    print("=== Round-trip error ===")
    print(f"  max abs err:  {result.max_abs_err:.6f}")
    print(f"  mean abs err: {result.mean_abs_err:.6f}")
    print(f"  max rel err:  {result.max_rel_err:.6f}")
    print(f"  mean rel err: {result.mean_rel_err:.6f}")

    # Sanity: dequant of source should match itself (idempotent)
    src = mxfp4_dequantize(w_i8, s_e8m0)
    print()
    print("=== Source statistics ===")
    print(f"  source non-zero fraction: {(np.abs(src) > 0).mean():.4f}")
    print(f"  source mean abs:          {np.abs(src).mean():.6f}")
    print(f"  source max abs:           {np.abs(src).max():.6f}")
    print(f"  unique source values:     {len(np.unique(src))}")


if __name__ == "__main__":
    main()

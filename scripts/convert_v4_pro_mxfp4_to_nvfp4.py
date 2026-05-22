"""Full-model V4-Pro MXFP4 → NVFP4 conversion pipeline.

Phase 3 deliverable. Produces canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP.

Per-tensor dispatch (matches PLAN.md Phase 3 table):

| category | source | target | action |
|---|---|---|---|
| layers.X.ffn.experts.Y.w*.{weight,scale} | MXFP4 group=32 E8M0 | NVFP4 group=16 E4M3 + per-tensor FP32 S_g | re-quantize |
| layers.X.ffn.shared_experts.w*.{weight,scale} | FP8 block 128x128 | unchanged | pass-through |
| layers.X.attn.*.{weight,scale} | FP8 block 128x128 | unchanged | pass-through |
| layers.X.hc_attn_*, hc_ffn_* | BF16 | unchanged | pass-through |
| layers.X.{attn_norm,ffn_norm}.weight | BF16 | unchanged | pass-through |
| layers.X.attn.{compressor,indexer}.* | various | unchanged | pass-through |
| layers.X.ffn.gate.{weight,bias,tid2eid} | mixed | unchanged | pass-through |
| mtp.0.ffn.experts.Y.w*.{weight,scale} | MXFP4 group=32 E8M0 | NVFP4 group=16 E4M3 + S_g | re-quantize |
| mtp.0.{e_proj,h_proj}.{weight,scale} | FP8 block 128x128 | **BF16 dequantized** | dequant (workaround for upstream ReplicatedLinear+Fp8 bug) |
| mtp.0.attn.* + mtp.0.hc_* + mtp.0.norms | various | unchanged | pass-through |
| embed.weight, head.weight, norm.weight, hc_head_* | BF16/F32 | unchanged | pass-through |

Uses torch+GPU for the FP4 quantization math. Single-GPU streaming per-shard
to avoid OOM. Writes new safetensors shards + new safetensors index +
new config.json with NVFP4 quant_config.

Run on the B300 box:
  /data/venv-serve/bin/python scripts/convert_v4_pro_mxfp4_to_nvfp4.py \\
    --source /opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp \\
    --output /opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp \\
    --shards-from 1 --shards-to 64

Per-shard time: ~30-60s on a single B300. 64 shards ≈ 30-60 min.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


# ---------- FP4 codec (torch GPU) ----------

# Magnitudes of the FP4 e2m1 grid (positive only).
FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_MAX = 6.0

# E4M3 NVFP4-style constants (OCP MX, bias=7, finite_max=448).
E4M3_MAX = 448.0


def _build_e4m3_table() -> torch.Tensor:
    """All positive E4M3 representable values (128 entries), sorted ascending."""
    vals: list[float] = []
    for e in range(16):
        for m in range(8):
            if e == 0 and m == 0:
                vals.append(0.0)
            elif e == 0:
                vals.append((2 ** -6) * (m / 8.0))
            else:
                vals.append((2 ** (e - 7)) * (1 + m / 8.0))
    return torch.tensor(sorted(vals), dtype=torch.float32)


E4M3_POSITIVE_SORTED = _build_e4m3_table()


def fp4_unpack_to_float(packed: torch.Tensor, fp4_mags: torch.Tensor) -> torch.Tensor:
    """Unpack uint8-packed FP4 (2 per byte) → float32 signed magnitudes.

    packed shape [..., N/2] uint8 → returned shape [..., N] float32.
    Low nibble is element 0, high nibble is element 1 (DeepSeek convention).
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    def nib_to_float(nib: torch.Tensor) -> torch.Tensor:
        sign = (nib & 0x08) >> 3
        mag_idx = (nib & 0x07).long()
        mag = fp4_mags[mag_idx]
        return torch.where(sign.bool(), -mag, mag)

    lo_f = nib_to_float(low)
    hi_f = nib_to_float(high)
    # Interleave low and high
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2,
                       dtype=torch.float32, device=packed.device)
    out[..., 0::2] = lo_f
    out[..., 1::2] = hi_f
    return out


def fp4_quantize_round_to_grid(values: torch.Tensor, fp4_mags: torch.Tensor) -> torch.Tensor:
    """Quantize float values to the nearest FP4 e2m1 grid value (preserving sign)."""
    sign = torch.sign(values)
    mag = torch.abs(values)
    # Use searchsorted on the positive grid: find insertion index, pick nearest of (i-1, i).
    idx = torch.searchsorted(fp4_mags, mag, right=False)
    idx = idx.clamp(max=fp4_mags.numel() - 1)
    # nearest neighbor: between fp4_mags[idx-1] and fp4_mags[idx]
    lo = fp4_mags[(idx - 1).clamp(min=0)]
    hi = fp4_mags[idx]
    pick_hi = (mag - lo) > (hi - mag)
    chosen = torch.where(pick_hi, hi, lo)
    return sign * chosen


def fp4_pack_from_float(values: torch.Tensor, fp4_mags: torch.Tensor) -> torch.Tensor:
    """Pack quantized float FP4 values back into uint8 (2 per byte).

    values shape [..., N] float32, must already be on the FP4 grid → returns shape [..., N/2] uint8.
    """
    sign_bit = (values < 0).to(torch.uint8) << 3
    mag = torch.abs(values)
    # Index lookup via approximate equality on the grid
    diffs = torch.abs(mag.unsqueeze(-1) - fp4_mags)
    idx = diffs.argmin(dim=-1).to(torch.uint8)
    nibbles = sign_bit | idx
    # Pack pairs: out_byte = high<<4 | low
    flat = nibbles.reshape(*values.shape[:-1], values.shape[-1])
    lo = flat[..., 0::2]
    hi = flat[..., 1::2]
    return (hi << 4) | lo


# ---------- E8M0 / E4M3 scale codecs ----------

def e8m0_decode(bytes_u8: torch.Tensor) -> torch.Tensor:
    """E8M0 byte b → 2^(b - 127)."""
    return torch.exp2(bytes_u8.to(torch.float32) - 127.0)


def f32_to_e4m3_code(values: torch.Tensor, sorted_table: torch.Tensor, sorted_idx_to_byte: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize positive float values to the nearest E4M3 value.

    Returns (representable_floats, e4m3_codes_uint8). Sign is NOT encoded
    here (NVFP4 block scales are positive); caller separately tracks sign.

    sorted_table and sorted_idx_to_byte must be on the same device as values.
    """
    flat = values.reshape(-1).to(torch.float32)
    flat = flat.clamp(min=0.0, max=E4M3_MAX)
    idx = torch.searchsorted(sorted_table, flat, right=False)
    idx = idx.clamp(max=sorted_table.numel() - 1)
    lo = sorted_table[(idx - 1).clamp(min=0)]
    hi = sorted_table[idx]
    pick_hi = (flat - lo) > (hi - flat)
    code_sorted = torch.where(pick_hi, idx, (idx - 1).clamp(min=0))
    chosen = sorted_table[code_sorted]
    e4m3_code = sorted_idx_to_byte[code_sorted]
    return chosen.reshape(values.shape), e4m3_code.reshape(values.shape).to(torch.uint8)


# Precompute: for each entry in E4M3_POSITIVE_SORTED, what's its E4M3 byte encoding?
def _build_sorted_to_e4m3_byte() -> torch.Tensor:
    # E4M3 byte = (e << 3) | m for positive sign.
    vals: list[tuple[float, int]] = []
    for e in range(16):
        for m in range(8):
            if e == 0 and m == 0:
                v = 0.0
            elif e == 0:
                v = (2 ** -6) * (m / 8.0)
            else:
                v = (2 ** (e - 7)) * (1 + m / 8.0)
            byte = (e << 3) | m
            vals.append((v, byte))
    vals.sort()  # ascending by value
    return torch.tensor([b for _, b in vals], dtype=torch.long)


_SORTED_INDEX_TO_E4M3_BYTE = _build_sorted_to_e4m3_byte()


# ---------- per-tensor conversion ----------

def convert_expert_mxfp4_to_nvfp4(
    weight_u8: torch.Tensor,   # [out, in/2] uint8, packed FP4
    scale_e8m0: torch.Tensor,  # [out, in/32] uint8, E8M0
    device: torch.device,
    fp4_mags: torch.Tensor,
    e4m3_sorted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Convert one expert weight tensor MXFP4 → NVFP4.

    Returns (new_weight_u8 [out, in/2], new_scale_e4m3_u8 [out, in/16], global_scale_f32).
    """
    w = weight_u8.to(device)
    s = scale_e8m0.to(device)
    # All lookup tables already moved to device by caller
    sorted_index_to_byte = _SORTED_INDEX_TO_E4M3_BYTE.to(device)

    # 1+2. Dequantize MXFP4 → float32 (unpacked, full shape)
    unpacked = fp4_unpack_to_float(w, fp4_mags)  # [out, in] float32
    src_scales = e8m0_decode(s)  # [out, in/32] float32
    scales_expanded = src_scales.repeat_interleave(32, dim=1)
    src = unpacked * scales_expanded  # [out, in] float32

    # 4. Reshape into NVFP4 group=16 tiles
    out_dim, in_dim = src.shape
    group_size = 16
    n_groups = in_dim // group_size
    grouped = src.reshape(out_dim, n_groups, group_size)
    per_group_amax = grouped.abs().amax(dim=-1)  # [out, n_groups]

    # 3. Per-tensor S_g: saturate E4M3 max on the largest per-group amax
    max_amax = float(per_group_amax.max().item())
    s_g = max_amax / (FP4_MAX * E4M3_MAX) if max_amax > 0 else 1.0
    if s_g <= 0:
        s_g = 1.0

    # 5. Per-group E4M3 scale
    raw_s_l = per_group_amax / FP4_MAX / s_g  # [out, n_groups]
    s_l_e4m3, s_l_codes = f32_to_e4m3_code(raw_s_l, e4m3_sorted, sorted_index_to_byte)
    # Avoid divide-by-zero for all-zero groups
    s_l_e4m3 = torch.where(s_l_e4m3 > 0, s_l_e4m3, torch.tensor(1.0, device=device))
    s_l_codes = s_l_codes  # uint8

    # Override (just for reproducibility — we use sorted-index-to-byte map above)
    # f32_to_e4m3_code already returned proper byte codes; trust those.
    # But to be sure: reapply the sorted_index_to_byte lookup
    # (already done inside f32_to_e4m3_code via _SORTED_INDEX_TO_E4M3_BYTE on cpu).

    # 6. Quantize each weight to FP4
    full_scale = (s_g * s_l_e4m3)  # [out, n_groups]
    full_scale_expanded = full_scale.repeat_interleave(group_size, dim=1)  # [out, in]
    normalized = src / full_scale_expanded
    quantized = fp4_quantize_round_to_grid(normalized, fp4_mags)

    # 7. Pack back to uint8
    out_u8 = fp4_pack_from_float(quantized, fp4_mags)

    # Reinterpret the scale code bytes as float8_e4m3fn so safetensors
    # writes them with F8_E4M3 dtype (matches NVFP4 spec; same bits).
    s_l_codes_e4m3 = s_l_codes.contiguous().view(torch.float8_e4m3fn)

    return out_u8.cpu(), s_l_codes_e4m3.cpu(), s_g


def dequantize_fp8_block_to_bf16(
    weight_e4m3: torch.Tensor,   # [M, N] float8_e4m3fn
    scale_e8m0: torch.Tensor,    # [M/128, N/128] uint8
    device: torch.device,
) -> torch.Tensor:
    """Dequantize FP8 block-quant weight + E8M0 block scale → BF16.

    Used for MTP e_proj/h_proj to work around upstream
    ReplicatedLinear+Fp8 scale-param-registration bug.
    """
    w = weight_e4m3.to(device)
    s = scale_e8m0.to(device)
    s_decoded = e8m0_decode(s)  # [M/128, N/128] float32
    w_bf16 = w.to(torch.float32)
    M, N = w.shape
    bm, bn = 128, 128
    assert M % bm == 0 and N % bn == 0, f"FP8 block shape mismatch: weight {w.shape} not divisible by [128, 128]"
    s_expanded = s_decoded.repeat_interleave(bm, dim=0).repeat_interleave(bn, dim=1)
    return (w_bf16 * s_expanded).to(torch.bfloat16).cpu()


# ---------- tensor dispatch ----------

EXPERT_WEIGHT_RE = re.compile(r"^(layers\.\d+|mtp\.\d+)\.ffn\.experts\.\d+\.w[123]\.weight$")
EXPERT_SCALE_RE = re.compile(r"^(layers\.\d+|mtp\.\d+)\.ffn\.experts\.\d+\.w[123]\.scale$")
MTP_EH_PROJ_WEIGHT_RE = re.compile(r"^mtp\.\d+\.(e_proj|h_proj)\.weight$")
MTP_EH_PROJ_SCALE_RE = re.compile(r"^mtp\.\d+\.(e_proj|h_proj)\.scale$")


def classify_tensor(key: str) -> str:
    if EXPERT_WEIGHT_RE.match(key):
        return "expert_weight"
    if EXPERT_SCALE_RE.match(key):
        return "expert_scale"
    if MTP_EH_PROJ_WEIGHT_RE.match(key):
        return "mtp_eh_proj_weight"
    if MTP_EH_PROJ_SCALE_RE.match(key):
        return "mtp_eh_proj_scale"
    return "passthrough"


# ---------- conversion driver ----------

def convert_shard(
    src_path: Path,
    out_path: Path,
    device: torch.device,
    log_prefix: str = "",
) -> tuple[dict[str, str], int]:
    """Convert one safetensors shard. Returns (per-key dtype-update map, byte size)."""
    t0 = time.time()
    fp4_mags = FP4_MAGNITUDES.to(device)
    e4m3_sorted = E4M3_POSITIVE_SORTED.to(device)

    # Gather all expert weight + scale pairs to convert together
    new_tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    global_scale_tensors: dict[str, torch.Tensor] = {}  # for new sidecar global_scale params

    with safe_open(src_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta_in = f.metadata() or {}
        metadata.update(meta_in)

        # First pass: collect expert weight/scale pairs
        expert_pairs: dict[str, tuple[str, str]] = {}  # base → (weight_key, scale_key)
        passthrough_keys: list[str] = []
        mtp_eh_pairs: dict[str, tuple[str, str]] = {}

        for k in keys:
            cls = classify_tensor(k)
            if cls == "expert_weight":
                base = k.removesuffix(".weight")
                expert_pairs.setdefault(base, [None, None])
                expert_pairs[base][0] = k
            elif cls == "expert_scale":
                base = k.removesuffix(".scale")
                expert_pairs.setdefault(base, [None, None])
                expert_pairs[base][1] = k
            elif cls == "mtp_eh_proj_weight":
                base = k.removesuffix(".weight")
                mtp_eh_pairs.setdefault(base, [None, None])
                mtp_eh_pairs[base][0] = k
            elif cls == "mtp_eh_proj_scale":
                base = k.removesuffix(".scale")
                mtp_eh_pairs.setdefault(base, [None, None])
                mtp_eh_pairs[base][1] = k
            else:
                passthrough_keys.append(k)

        # Pass-through tensors first
        for k in passthrough_keys:
            new_tensors[k] = f.get_tensor(k)

        # Convert expert pairs (MXFP4 → NVFP4)
        expert_count = 0
        for base, (wk, sk) in expert_pairs.items():
            if wk is None or sk is None:
                print(f"{log_prefix}WARN: incomplete expert pair at {base}: weight={wk} scale={sk}", file=sys.stderr)
                if wk: new_tensors[wk] = f.get_tensor(wk)
                if sk: new_tensors[sk] = f.get_tensor(sk)
                continue
            w = f.get_tensor(wk)
            s = f.get_tensor(sk)
            # Safetensors loads I8 dtype as torch.int8 (signed). Bit ops
            # like (x >> 4) on signed int8 do arithmetic (sign-extending)
            # shifts, which destroys FP4 nibbles in bytes with bit 7 set.
            # View as uint8 before passing to the math.
            if w.dtype == torch.int8:
                w = w.view(torch.uint8)
            # Same fix for E8M0 scales (safetensors loads as float8_e8m0fnu).
            if s.dtype == torch.float8_e8m0fnu:
                s = s.view(torch.uint8)
            new_w, new_s, s_g = convert_expert_mxfp4_to_nvfp4(
                w, s, device, fp4_mags, e4m3_sorted,
            )
            new_tensors[wk] = new_w
            new_tensors[sk] = new_s
            # Store per-tensor global scale as a sidecar BF16 scalar
            new_tensors[f"{base}.weight_scale_2"] = torch.tensor([s_g], dtype=torch.float32)
            expert_count += 1

        # Dequantize MTP e_proj/h_proj FP8 → BF16
        mtp_eh_count = 0
        mtp_eh_dropped_scales = []
        for base, (wk, sk) in mtp_eh_pairs.items():
            if wk is None:
                continue
            w_e4m3 = f.get_tensor(wk)  # float8_e4m3fn
            if sk is None or sk not in keys:
                # No scale — just pass through weight as-is
                new_tensors[wk] = w_e4m3
                continue
            s_e8m0 = f.get_tensor(sk)
            new_w_bf16 = dequantize_fp8_block_to_bf16(w_e4m3, s_e8m0, device)
            new_tensors[wk] = new_w_bf16
            # Drop the scale — destination is unquantized BF16
            mtp_eh_dropped_scales.append(sk)
            mtp_eh_count += 1

    # Add metadata
    metadata["format"] = "pt"
    metadata.setdefault(
        "v4_pro_nvfp4_conversion_version",
        "0.1.0 (2026-05-21)",
    )

    # Write the new shard
    save_file(new_tensors, str(out_path), metadata={k: str(v) for k, v in metadata.items()})

    dt = time.time() - t0
    print(
        f"{log_prefix}{src_path.name} → {out_path.name}: "
        f"{expert_count} expert pairs converted, "
        f"{mtp_eh_count} MTP eh_proj dequantized, "
        f"{len(passthrough_keys)} passthrough, "
        f"{dt:.1f}s",
        file=sys.stderr,
    )

    return {
        "expert_count": expert_count,
        "mtp_eh_count": mtp_eh_count,
        "passthrough_count": len(passthrough_keys),
        "elapsed_s": dt,
        "out_size_bytes": out_path.stat().st_size,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards-from", type=int, default=1)
    p.add_argument("--shards-to", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Read the input index
    idx_path = args.source / "model.safetensors.index.json"
    with idx_path.open() as f:
        in_idx = json.load(f)

    # Determine which shards to process
    shard_list = [
        f"model-{i:05d}-of-00064.safetensors"
        for i in range(args.shards_from, args.shards_to + 1)
    ]

    # Copy non-safetensors files (config, tokenizer, etc.) on first invocation
    if args.shards_from == 1:
        for fn in (args.source.iterdir()):
            if fn.name.endswith(".safetensors") or fn.name == "model.safetensors.index.json":
                continue
            dst = args.output / fn.name
            if not dst.exists():
                if fn.is_dir():
                    shutil.copytree(fn, dst)
                else:
                    shutil.copy2(fn, dst)
        print(f"copied non-safetensors auxiliary files to {args.output}", file=sys.stderr)

    # Convert each shard
    stats = {}
    for shard in shard_list:
        src = args.source / shard
        dst = args.output / shard
        if not src.exists():
            print(f"missing source {src}; skipping", file=sys.stderr)
            continue
        if dst.exists():
            print(f"output {dst} already exists; skipping (delete to regenerate)", file=sys.stderr)
            continue
        s = convert_shard(src, dst, device, log_prefix=f"[{shard}] ")
        stats[shard] = s

    # Write per-run stats to output dir
    stats_path = args.output / f".conversion_stats_{args.shards_from}_{args.shards_to}.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote {stats_path}", file=sys.stderr)

    # If we did the full range, regenerate the index + write config.json
    if args.shards_from == 1 and args.shards_to == 64:
        regenerate_index(args.output)
        rewrite_config(args.source, args.output)


def regenerate_index(out_dir: Path) -> None:
    """Walk the output shards, build a fresh model.safetensors.index.json."""
    weight_map: dict[str, str] = {}
    total_size = 0
    shards = sorted(out_dir.glob("model-*-of-00064.safetensors"))
    for shard in shards:
        with shard.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        for k, meta in header.items():
            if k == "__metadata__":
                continue
            weight_map[k] = shard.name
            offsets = meta.get("data_offsets")
            if offsets:
                total_size += offsets[1] - offsets[0]
    idx = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    out_path = out_dir / "model.safetensors.index.json"
    with out_path.open("w") as f:
        json.dump(idx, f, indent=2)
    print(f"wrote index: {len(weight_map)} keys, {total_size:,} total bytes ({total_size/1e9:.2f} GB)", file=sys.stderr)


def rewrite_config(src_dir: Path, out_dir: Path) -> None:
    """Update config.json to mark the artifact as NVFP4."""
    src_cfg = json.load((src_dir / "config.json").open())
    # Update quantization_config to reflect NVFP4 conversion
    # Keep expert_dtype="fp4" so vLLM's DeepseekV4FP8Config.get_quant_method
    # still routes through the "fp4" branch, but set moe_quant_algo="NVFP4"
    # to trigger ModelOptNvFp4FusedMoE (vLLM PR #42209). This is the upstream
    # routing convention.
    src_cfg["expert_dtype"] = "fp4"
    src_cfg["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",  # for FP8 attention; preserved as-is
        "weight_block_size": [128, 128],
        # Triggers ModelOptNvFp4FusedMoE in vllm/models/deepseek_v4/quant_config.py
        # per PR #42209 (https://github.com/vllm-project/vllm/pull/42209).
        "moe_quant_algo": "NVFP4",
        # Metadata for downstream readers
        "expert_format": {
            "name": "nvfp4",
            "weight_dtype": "fp4_e2m1",
            "weight_packing": "uint8_2_per_byte",
            "group_size": 16,
            "scale_dtype": "fp8_e4m3",
            "weight_scale_2_dtype": "fp32",
        },
        "mtp_e_proj_h_proj_dequant": "bf16",
    }
    out_cfg = out_dir / "config.json"
    with out_cfg.open("w") as f:
        json.dump(src_cfg, f, indent=2)
    print(f"wrote {out_cfg}", file=sys.stderr)


if __name__ == "__main__":
    main()

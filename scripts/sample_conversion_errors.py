"""Sample MXFP4 → NVFP4 conversion errors across many experts.

Phase 2 step 2-3 deliverable. Replaces the invented "max-rel < 5%" gate
with a characterization of the actual error distribution across a
representative sample of V4-Pro experts.

Samples:
  - Layer indices: 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60
  - Expert IDs per layer: 0, 96, 192, 288, 383 (5 each)
  - All three of w1, w2, w3
  - 10 MTP-block experts at layer 'mtp.0'

Output: docs/findings/conversion_math_validation.md with distribution stats.
"""

from __future__ import annotations

import json
import statistics
import struct
import sys
import urllib.request

import numpy as np

# Reuse the math reference
sys.path.insert(0, "/home/paul/dsv4-pro-nvfp4-fp8-mtp/scripts")
from convert_mxfp4_tensor_to_nvfp4 import (
    convert_mxfp4_to_nvfp4,
    fetch_tensor_from_hf,
)


# Layer-to-shard mapping is needed since shards are content-addressed.
# Pull the index once.
REPO = "deepseek-ai/DeepSeek-V4-Pro"
print("fetching index...", file=sys.stderr)
idx_buf = urllib.request.urlopen(
    f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"
).read()
INDEX = json.loads(idx_buf)
WM = INDEX["weight_map"]


# Build the sample list.
SAMPLE_LAYERS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
SAMPLE_EXPERTS = [0, 96, 192, 288, 383]
SAMPLE_WS = ["w1", "w2", "w3"]
MTP_SAMPLE_EXPERTS = [0, 50, 100, 150, 200, 250, 300, 350, 383]


def make_sample_keys() -> list[tuple[str, str, str]]:
    """Return list of (weight_key, scale_key, label)."""
    out: list[tuple[str, str, str]] = []
    for L in SAMPLE_LAYERS:
        for E in SAMPLE_EXPERTS:
            for W in SAMPLE_WS:
                wk = f"layers.{L}.ffn.experts.{E}.{W}.weight"
                sk = f"layers.{L}.ffn.experts.{E}.{W}.scale"
                if wk in WM and sk in WM:
                    out.append((wk, sk, f"L{L:02d}/E{E:03d}/{W}"))
    # MTP experts (layer "mtp.0")
    for E in MTP_SAMPLE_EXPERTS:
        for W in SAMPLE_WS:
            wk = f"mtp.0.ffn.experts.{E}.{W}.weight"
            sk = f"mtp.0.ffn.experts.{E}.{W}.scale"
            if wk in WM and sk in WM:
                out.append((wk, sk, f"MTP/E{E:03d}/{W}"))
    return out


def main() -> None:
    samples = make_sample_keys()
    print(f"Sampling {len(samples)} expert weight tensors...", file=sys.stderr)
    rows: list[dict] = []
    for i, (wk, sk, label) in enumerate(samples):
        shard = WM[wk]
        if WM[sk] != shard:
            print(f"  SKIP {label}: weight and scale on different shards", file=sys.stderr)
            continue
        try:
            w_i8, s_e8m0, _ = fetch_tensor_from_hf(REPO, shard, wk, sk)
            result = convert_mxfp4_to_nvfp4(w_i8, s_e8m0)
            row = {
                "label": label,
                "shape_out": list(w_i8.shape),
                "source_amax": result.source_amax,
                "s_g": result.out_global_scale,
                "max_abs": result.max_abs_err,
                "mean_abs": result.mean_abs_err,
                "max_rel": result.max_rel_err,
                "mean_rel": result.mean_rel_err,
            }
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(
                    f"  [{i+1:>3}/{len(samples)}] {label}: "
                    f"max_abs={result.max_abs_err:.5f} mean_abs={result.mean_abs_err:.5f} "
                    f"max_rel={result.max_rel_err:.3f} mean_rel={result.mean_rel_err:.4f}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"  ERROR {label}: {e}", file=sys.stderr)

    # Distribution stats
    def stats(metric: str) -> dict:
        vals = [r[metric] for r in rows]
        if not vals:
            return {}
        return {
            "n": len(vals),
            "min": float(min(vals)),
            "p50": float(statistics.median(vals)),
            "p90": float(statistics.quantiles(vals, n=10)[-1]),
            "p99": float(statistics.quantiles(vals, n=100)[-1]) if len(vals) >= 100 else float(max(vals)),
            "max": float(max(vals)),
            "mean": float(statistics.mean(vals)),
        }

    summary = {
        "n_tensors_sampled": len(rows),
        "max_abs_err": stats("max_abs"),
        "mean_abs_err": stats("mean_abs"),
        "max_rel_err": stats("max_rel"),
        "mean_rel_err": stats("mean_rel"),
        "source_amax": stats("source_amax"),
        "s_g": stats("s_g"),
    }

    out_path = "/home/paul/dsv4-pro-nvfp4-fp8-mtp/docs/findings/conversion_math_validation.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)

    # Pretty summary to stdout
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

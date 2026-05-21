#!/usr/bin/env python3
"""Phase 4 gate — confirm MTP is PRESENT and BF16 (Option Y invariant).

REWRITTEN 2026-05-21 for the NVFP4-FP8-MTP artifact's actual design.

Earlier W4A16 framing (sibling repo) expected MTP to be QUANTIZED — every
expert and attention projection should have a `.weight_scale`. That gate
was designed when "MTP quantized like the rest of the model" was the
target.

The NVFP4 path hit a `RuntimeError: Inplace update to inference tensor
outside InferenceMode is not allowed` during the MTP block's qparam
writeback (the shared-embedding lookup in the vendored upstream model
creates tensors under `inference_mode()`, which traps the in-place
`weight_scale` write that fires at `sequential_epoch_end`). Both this
workstream and the H200 sibling workstream independently converged on
"Option Y": MTP block is sequenced through the pipeline but **excluded
from the quantization recipe** (via `r"re:.*mtp\\..*"` in the
QuantizationModifier `ignore` list). MTP weights stay BF16 in the
artifact, are still PRESENT (not dropped like RedHat's transformers-stock
loader does), and load cleanly in vLLM with `--speculative_config
method=mtp`. The differentiator vs RedHat is presence, not precision.

So the gate this script enforces is:

  1. MTP block PRESENT — at least 6 unconditional MTP weight tensors
     (lower bound; the dryrun's actual mtp.* count was 799).
  2. ALL sampled MTP `.weight` tensors are bfloat16 — no FP8 / FP4 /
     packed quantization on MTP modules. Quantized MTP would mean the
     recipe leaked through the `ignore` list, which is a bug we want to
     catch.
  3. NO MTP quantization-scale tensors — no `.weight_scale`,
     `.weight_scale_inv`, `.weight_packed`, `.weight_global_scale`,
     `.input_global_scale`, `.weight_zero_point` on any `mtp.*` key. (The
     `mtp.0.hc_*_scale` keys are architecture-native — head-compressed
     scales — and excluded by the `.weight_scale` / `.weight_packed` /
     etc. specificity here.)
  4. Key MTP modules present by name — `e_proj.weight`, `h_proj.weight`,
     `emb.tok_emb.weight` (the renamed embedding; see
     postprocess_for_vllm.py for the rename), plus the attention
     projections (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`).
"""
import argparse
import json
import re
import sys
from pathlib import Path

from safetensors import safe_open

# Tensor-name suffixes that indicate a key is a QUANTIZATION ARTIFACT (not
# an architecture-native scale like the `hc_*_scale` tensors in DSV4's
# head-compressed layers).
QUANT_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_inv",
    ".weight_packed",
    ".weight_global_scale",
    ".input_global_scale",
    ".weight_zero_point",
)

# Required MTP key patterns that must be present (post-postprocess rename).
REQUIRED_MTP_KEYS = (
    re.compile(r"^mtp\.\d+\.e_proj\.weight$"),
    re.compile(r"^mtp\.\d+\.h_proj\.weight$"),
    re.compile(r"^mtp\.\d+\.emb\.tok_emb\.weight$"),
    re.compile(r"^mtp\.\d+\.attn\.wq_a\.weight$"),
    re.compile(r"^mtp\.\d+\.attn\.wq_b\.weight$"),
    re.compile(r"^mtp\.\d+\.attn\.wkv\.weight$"),
    re.compile(r"^mtp\.\d+\.attn\.wo_a\.weight$"),
    re.compile(r"^mtp\.\d+\.attn\.wo_b\.weight$"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="post-calibration NVFP4-FP8-MTP model dir")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        sys.exit(f"FATAL: {idx_path} not found")

    weight_map = json.loads(idx_path.read_text()).get("weight_map", {})
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
    mtp_weight_keys = [k for k in mtp_keys if k.endswith(".weight")]
    mtp_quant_keys = [
        k for k in mtp_keys if any(k.endswith(suf) for suf in QUANT_SUFFIXES)
    ]

    print(f"total tensors in artifact: {len(weight_map)}")
    print(f"MTP tensors total:         {len(mtp_keys)}")
    print(f"  of which .weight:        {len(mtp_weight_keys)}")
    print(f"  of which quant scales:   {len(mtp_quant_keys)} (expect 0)")

    # Check 1: presence
    if len(mtp_weight_keys) < 6:
        sys.exit(
            f"FAIL: only {len(mtp_weight_keys)} MTP .weight tensors found; "
            "expected ≥6 (e_proj, h_proj, attn.{wq_a,wq_b,wkv,wo_a,wo_b}). "
            "The MTP block may have been dropped at save time."
        )

    # Check 2: no quantization on MTP
    if mtp_quant_keys:
        print(
            "FAIL: MTP modules carry quantization-scale tensors — recipe "
            "leaked through the `ignore` list:"
        )
        for k in mtp_quant_keys[:10]:
            print(f"  {k}")
        if len(mtp_quant_keys) > 10:
            print(f"  ... and {len(mtp_quant_keys) - 10} more")
        sys.exit(1)

    # Check 3: required keys present
    missing = []
    for pat in REQUIRED_MTP_KEYS:
        if not any(pat.match(k) for k in mtp_keys):
            missing.append(pat.pattern)
    if missing:
        sys.exit(
            "FAIL: required MTP keys missing:\n  " + "\n  ".join(missing)
        )

    # Check 4: dtype audit — every quantize-candidate MTP Linear's .weight
    # must be bfloat16. The Option Y invariant only applies to tensors
    # that WOULD have been quantized had the recipe included them
    # (Linear projections matching the recipe's targets). Norms, heads,
    # and embeds are legitimately float32 in the DSV4 architecture and
    # would never be quantized regardless.
    quant_candidate_pat = re.compile(
        r"^mtp\.\d+\."
        r"(attn\.(wq_a|wq_b|wkv|wo_a|wo_b)|e_proj|h_proj"
        r"|ffn\.experts\.\d+\.(w1|w2|w3))"
        r"\.weight$"
    )
    candidates = [k for k in mtp_weight_keys if quant_candidate_pat.match(k)]
    print()
    print(
        f"checking dtypes of {len(candidates)} quantize-candidate MTP Linears "
        "(sampling head + tail) ..."
    )
    samples = candidates[:8] + candidates[-4:]  # head + tail
    bad_dtype = []
    for k in samples:
        shard = weight_map[k]
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
        dtype_str = str(t.dtype)
        marker = "OK" if dtype_str == "torch.bfloat16" else "BAD"
        print(f"  [{marker}] {k}: dtype={dtype_str}, shape={tuple(t.shape)}")
        if dtype_str != "torch.bfloat16":
            bad_dtype.append((k, dtype_str))
    if bad_dtype:
        sys.exit(
            "FAIL: MTP quantize-candidate Linears are not bfloat16 — "
            "Option Y invariant violated:\n  "
            + "\n  ".join(f"{k}: {dt}" for k, dt in bad_dtype)
        )

    print()
    print("MTP gate PASSED — block present, bfloat16, unquantized.")
    print(f"({len(mtp_weight_keys)} MTP weights ready for vLLM "
          "`--speculative_config method=mtp` load.)")


if __name__ == "__main__":
    main()

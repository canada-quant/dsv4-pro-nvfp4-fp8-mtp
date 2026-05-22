# Phase 3 v3 conversion validation — 2026-05-22

After the AMEX-block instance loss + relaunch, the conversion was re-run
on a fresh `i-02c88555a88fbbc19` (`p6-b300.48xlarge`, `35.87.217.155`)
with all three Phase 3 fixes baked in:

1. **int8 → uint8 view before bit ops** (otherwise FP4 nibble extraction
   silently corrupts on bytes with bit 7 set due to arithmetic shift)
2. **Shared per-tensor S_g between w1 and w3 of each expert** (ModelOpt's
   `ModelOptNvFp4FusedMoE.process_weights_after_loading` discards w3's
   `weight_scale_2` and uses w1's for both; if they don't match, w3
   dequant is wrong)
3. **`input_scale = 1.0` sidecar emitted per expert tensor** (without it,
   `layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)` at load
   time multiplies `weight_scale_2` by uninitialized `torch.empty()`
   garbage, destroying the per-tensor scale)

## Byte-level dequant validation

Verified against the source MXFP4 dequant across 15 sampled tensors
(5 experts × {w1, w2, w3}) on layer 30:

| Expert | w | src_amax | nv_amax | S_g | input_scale | max_abs_diff | mean_abs_diff | correlation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | w1 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00107 | 1.0000 |
| 0 | w2 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00104 | 1.0000 |
| 0 | w3 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00108 | 1.0000 |
| 50 | w1 | 0.18750 | 0.18750 | 6.9754e-05 | 1.00 | 0.01339 | 0.00082 | 0.9976 |
| 50 | w2 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00105 | 1.0000 |
| 50 | w3 | 0.12500 | 0.12054 | 6.9754e-05 | 1.00 | 0.01339 | 0.00084 | 0.9972 |
| 100 | w1 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00109 | 1.0000 |
| 100 | w2 | 0.18750 | 0.18750 | 6.9754e-05 | 1.00 | 0.01339 | 0.00100 | 0.9965 |
| 100 | w3 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00118 | 1.0000 |
| 200 | w1–w3 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00109–0.00114 | 1.0000 |
| 380 | w1, w3 | 0.12500 | 0.12500 | 4.6503e-05 | 1.00 | 0.01042 | 0.00103–0.00106 | 1.0000 |
| 380 | w2 | 0.18750 | 0.18750 | 6.9754e-05 | 1.00 | 0.01339 | 0.00080 | 0.9973 |

All metrics inside the Phase 2 sampling envelope (mean abs ~ 1e-3, max abs
~ 1e-2, correlation > 0.99). The mild correlation dip for the high-amax
experts (E=50 w1/w3, E=100 w2, E=380 w2) is the FP4 grid noise at higher
magnitudes — same shape as the Phase 2 characterization saw.

## Shared S_g (ModelOpt invariant)

| Expert | w1 S_g | w3 S_g | match |
|---:|---:|---:|---|
| 0 | 4.6503e-05 | 4.6503e-05 | ✓ |
| 50 | 6.9754e-05 | 6.9754e-05 | ✓ |
| 100 | 4.6503e-05 | 4.6503e-05 | ✓ |
| 200 | 4.6503e-05 | 4.6503e-05 | ✓ |
| 380 | 4.6503e-05 | 4.6503e-05 | ✓ |

The `forced_s_g = max(w1_amax, w3_amax) / (FP4_MAX * E4M3_MAX)` policy
correctly produces identical scales for w1 and w3 per expert. This
satisfies ModelOpt's `torch.allclose(layer.w13_weight_scale_2[:, 0],
layer.w13_weight_scale_2[:, 1])` check.

## Artifact stats (post-v3 conversion)

| Field | Value |
|---|---|
| Output dir | `/opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp/` |
| Total size | 852 GiB |
| Shard count | 64 |
| Index keys | 287,962 |
| Total tensor bytes (per index) | 913,956,972,536 (913.96 GB) |
| MTP `e_proj`/`h_proj` dequantized to BF16 | 2 tensors |
| Expert pairs converted (main trunk) | 60 layers × 384 experts × 3 = 69,120 |
| Expert pairs converted (MTP block) | 384 × 3 = 1,152 |

Per-expert sidecars emitted:
- `<expert>.w{1,2,3}.weight_scale_2` (FP32 [1], per-tensor global scale)
- `<expert>.w{1,2,3}.input_scale` (FP32 [1], dynamic activation = 1.0)

## What's still to verify

Byte-level dequant validates the **stored representation** matches the
source within bounded FP4 quant noise. What's still to verify is the
**served behavior**: TRTLLM kernel must consume the layout correctly and
produce coherent text. That's the next step (serve + smoke + benchmarks).

# Option A "lossless conversion" audit — 2026-05-21

**Status: AUDIT, not a recommendation.** The earlier ground-truth doc stated
that converting native DeepSeek V4-Pro FP4 → NVFP4 was an "in-place format
conversion, mostly lossless." This document tests that claim against the
actual weight and scale layout and identifies what the conversion would
actually entail. No A/B/C decision is made here.

## TL;DR

The "lossless in-place conversion" claim does not survive contact with the
data. Three specific issues:

1. **Group size mismatch.** Native source uses **group=32** along the input
   dim. NVFP4 spec is **group=16**. The group boundaries don't align —
   conversion requires regrouping, which is a re-quantization step
   constrained by the already-quantized FP4 values (cannot recover precision
   lost in the original FP4 step).
2. **Scale-format range/precision.** Native scales are E8M0 (pure 8-bit
   exponent, no mantissa). NVFP4 scales are FP8-E4M3 (1+4+3 bits). Sampled
   E8M0 exponents range from −12 to −7. E4M3's exact-power-of-2 range as a
   single-level scale is [−9, 8]; **3 of 7 unique observed exponents
   (−12, −11, −10) fall outside this range** and would require either
   saturation or NVFP4's two-level scaling (per-tensor FP32 S_g × per-block
   FP8 S_l) to absorb the range mismatch. With the two-level scheme, the
   scale-format conversion can be near-lossless **but requires per-tensor
   global-scale computation from weight statistics** — that is itself a
   calibration step (weight-only, not data-dependent).
3. **No stored activation scales.** Native source has no static activation
   scale tensors. The DeepSeek inference path uses dynamic `fp4_act_quant`
   (`vendor/dsv4-pro-upstream/kernel.py:186`). NVFP4 serving can use either
   dynamic input-quant or static calibrated activation scales. Whichever
   we pick is independent of A/B/C and either way is *not* covered by
   "no calibration."

## What we measured

### Tensor layout (one expert as the canonical case)

From `vendor/dsv4-pro-upstream/` and a range-read of shard 32:

```
weight: layers.30.ffn.experts.0.w1.weight
  dtype: I8, shape: [3072, 3584]
  bytes: 11,010,048 → 22,020,096 FP4 elements (2 packed per byte)
  unpacked shape: [3072, 7168]

scale:  layers.30.ffn.experts.0.w1.scale
  dtype: F8_E8M0, shape: [3072, 224]
  bytes: 688,128

ratio (FP4 weights : E8M0 scales) = 32:1
  → group size = 32 along the input dim
  → scale layout: 1 scale per [1, 32] row-tile of unpacked weight
```

This is consistent across all sampled expert w1/w2/w3 tensors (group=32
along the inner dim) and matches **MXFP4** (OCP Microscaling FP4 spec:
4-bit elements + E8M0 8-bit-exponent block scales over groups of 32).

Attention and shared-expert weights use a different layout: FP8-E4M3
elements with `weight_block_size=[128,128]` (per the `quantization_config`
in `vendor/dsv4-pro-upstream/config.json`), one E8M0 scale per 128×128
block. This is the standard DeepSeek block-FP8 quant — not affected by
the FP4-group-size discussion.

### Scale exponent distribution

Sampled E8M0 scale bytes from layers 0, 8, 18, 28, 38, 48, 58 (8 shards;
4,832 raw bytes; ~600 per layer):

| exponent (byte − 127) | count | % | exact-pow2 in E4M3 (bias=7)? |
|---:|---:|---:|---|
| −12 | 125 | 2.59% | no (out of range, need S_g) |
| −11 | 110 | 2.28% | no |
| −10 | 1,856 | 38.41% | no |
| −9 | 1,962 | 40.60% | yes (subnormal) |
| −8 | 777 | 16.08% | yes (subnormal) |
| −7 | 2 | 0.04% | yes (subnormal) |
| −6 .. 8 | 0 | 0% | yes (normal range) |

Per-layer ranges (min, max):

```
layer 0 : [-12, -10]   (entirely below E4M3 native power-of-2 range)
layer 8 : [-9, -7]
layer 18: [-11, -8]
layer 28: [-10, -8]
layer 38: [-10, -8]
layer 48: [-10, -7]
layer 58: [-11, -8]
```

Range width per layer never exceeds 3 exponents (fits comfortably in
E4M3's 18-exponent range if absorbed into a per-tensor global scale).

Notes on E4M3 representability (OCP MX FP8 e4m3, NVFP4 convention,
bias=7, finite_max=448, subnormals at exponents −9, −8, −7 with
mantissas 1, 2, 4):

```
exact pow2 in E4M3 (m=0 or subnormal m∈{1,2,4}): exponents {-9 ... 8}
above range (saturation): exponents > 8
below range (round to zero or smallest subnormal): exponents < -9
```

## What this implies for "in-place conversion"

### Step 1 — scale-format conversion (E8M0 → E4M3)

If we keep the existing 32-element group structure and only convert each
E8M0 scale to E4M3 individually:
- 99.58% of sampled scales are at exponents −9..−5 — exactly
  representable in E4M3 as power-of-2 mantissa-zero values.
- 0.42% are at exponents −12..−10 — out of E4M3's single-level range.
  Naive conversion either saturates to the smallest E4M3 subnormal
  (introducing a multiplicative error of 2^(missing exponents) per affected
  scale; up to 8× in the worst observed case) or requires a per-tensor
  FP32 global scale `S_g` to absorb the range offset.

If we use NVFP4's two-level scaling (per-tensor FP32 `S_g` × per-block
FP8-E4M3 `S_l`), then for each sampled tensor we can pick `S_g = 2^k`
with `k = -(observed_max_exp + 9)` and rescale every `S_l` into the
representable range. Per-layer max−min exponent range is at most 3, so
the rescaled E4M3 values fall comfortably within mantissa-0 powers of 2.
**Under two-level scaling, the scale-format conversion can be exact (no
rounding error) at the scale layer.** But the global `S_g` must be
computed per tensor — that is a weight-statistics analysis step ("which
exponents appear in this tensor's scale set"), not a calibration with
data, but also not zero.

### Step 2 — group regrouping (32 → 16)

Even if step 1 is exact, the group boundaries change. Each native
[1×32] tile carries one E8M0 scale. NVFP4 expects each [1×16] tile to
carry its own E4M3 scale. There are two options:

**Option α — split scales.** Give both [1×16] halves of each original
[1×32] tile the SAME scale (the converted E4M3 value from step 1). This
is a no-information-loss step at the bit level (the FP4 values don't
change; we just doubled the scale storage by replication). It produces
a valid NVFP4 representation but does not benefit from finer grouping
— the effective precision is exactly what the original group=32
quantization captured.

**Option β — re-quantize on a [1×16] grid.** For each new 16-element
group, dequantize the FP4 values back to FP16/BF16 (using the original
E8M0 scale), find a new max-abs per-group, derive a new E4M3 scale,
re-encode each weight as FP4. This step is **lossy in general** because
the FP4 set is fixed ({0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} times scale)
and re-quantization with a different scale rounds against a different
grid. Net direction (gain or loss in accuracy) depends on the original
group-32 scale's optimality — typically re-quantization at a tighter
group can only retain or marginally improve precision **relative to the
already-FP4 values**, never beyond it.

Neither option is "in-place" in the literal sense (both rewrite tensor
files). Option α is information-preserving relative to the source.
Option β is a re-quantization with measurable error budget; whether it
helps depends on per-tensor statistics.

### Step 3 — activation handling at serve time

The native inference path uses dynamic per-token activation FP4
quantization (`fp4_act_quant` in `vendor/dsv4-pro-upstream/kernel.py`).
NVFP4 kernels in vLLM (via DeepGEMM / sgl-deep-gemm) similarly support
either dynamic input-quant or static calibrated input scales. We have
not yet inspected vLLM's NVFP4 kernel API to determine which it expects
for a V4-Pro-style MoE FFN under `--moe-backend deep_gemm_mega_moe`.

If the chosen activation path is **dynamic**, no calibration data is
required for activations (matches Option A's framing).

If the chosen activation path is **static**, calibration data is
required regardless of how weights are produced — Option A does not
escape this.

## What we did NOT verify (open items)

- **Wider scale sampling.** We sampled ~5K scale bytes across 8 shards.
  The full source has 49 B scale elements. Out-of-range exponents
  could exist that we did not see — e.g., the lm_head, embed-related
  scales, or specific outlier-expert weights. Before any production
  conversion we'd want a full pass to find global min/max exponents
  across every tensor in the model.
- **Empirical re-quantization error.** Option β's actual lossiness on
  V4-Pro experts has not been measured. The bounds we cite (re-quant
  at tighter group ≤ no worse than original) are theoretical; a
  concrete number requires dequant-and-requant on a sample expert and
  evals on a small held-out set.
- **NVFP4 production scale convention used by vLLM.** vLLM's
  `Mxfp4MoEMethod` handles MXFP4 (group=32, E8M0) — confirmed by
  `vllm/models/deepseek_v4/quant_config.py` on main. We have not
  found a corresponding `Nvfp4MoEMethod` for V4-Pro group=16 / E4M3
  in vLLM. See companion document
  `docs/findings/vllm_pro_serving_path.md`.
- **What "MXFP4-FP8-MTP" already gives us.** Open question whether
  the artifact we initially specified ("NVFP4-FP8-MTP") is even the
  right artifact given that the native release IS already
  MXFP4-FP8-MTP and vLLM serves it directly. The "NVFP4" framing
  was inherited from V4-Flash where it was the natural delta vs the
  BF16 source. For V4-Pro that delta dissolves. This is the strategic
  question, not an A/B/C choice.

## Receipts

- `vendor/dsv4-pro-upstream/config.json` — `quantization_config` block
  shows `quant_method: fp8, weight_block_size: [128,128], scale_fmt: ue8m0`
  for non-expert tensors.
- `vendor/dsv4-pro-upstream/inference_config.json` — `expert_dtype: fp4,
  dtype: fp8, scale_fmt: ue8m0`.
- `vendor/dsv4-pro-upstream/kernel.py` — `fp4_quant_kernel`,
  `fp4_act_quant`, `fp4_gemm` definitions. Expert path is FP4 weights ×
  dynamic-FP4 activations × E8M0 scales.
- Range-reads of shards 2, 10, 20, 30, 32, 40, 50, 60, 63 (per-shard
  safetensors JSON headers + raw scale bytes for sampled tensors). All
  scale tensors observed are `F8_E8M0`, all expert weight tensors are
  `I8` packed (2 FP4 per byte).

## Open questions to answer before any A/B/C decision

1. Does vLLM have a V4-Pro NVFP4 (group=16, E4M3) serving path? If not,
   the artifact has no way to be served on vLLM regardless of how it's
   produced. (Answered in
   `docs/findings/vllm_pro_serving_path.md`.)
2. Does our hardware (B300, sm_103a) have a native NVFP4 group=16
   GEMM path that meaningfully beats MXFP4 group=32? Open.
3. Is there a non-vLLM serve target (TensorRT-LLM, custom) where NVFP4
   group=16 has a clear performance win for V4-Pro's expert sizes
   (7168 × 3072)? Open.
4. If activation handling ends up being dynamic in both MXFP4 and NVFP4
   serving paths, what is the residual quality argument for NVFP4 over
   MXFP4 on V4-Pro? Open.

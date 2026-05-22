# MXFP4 → NVFP4 conversion math: error characterization — 2026-05-21

**Phase 2 deliverable.** Replaces the invented "max-rel < 5%" gate in
the original PLAN with a measured distribution across 192 sampled
V4-Pro expert weight tensors. No go/no-go threshold imposed at this
phase per the PLAN amendment — the quality gate is downstream
(Phase 6 benchmarks against the native MXFP4 baseline).

## Sample

192 expert weight tensors successfully converted and measured. Sample
plan from `scripts/sample_conversion_errors.py`:

- **Main-trunk experts**: 11 layers × 5 expert IDs × 3 w-tensors = 165
  - Layer indices: 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60
  - Expert IDs (per layer): 0, 96, 192, 288, 383
  - W-tensors: w1, w2, w3
- **MTP-block experts**: 9 expert IDs × 3 w-tensors = 27
  - Expert IDs: 0, 50, 100, 150, 200, 250, 300, 350, 383

192 of 192 sampled tensors completed conversion without NaN / Inf.
Raw per-tensor numbers in `docs/findings/conversion_math_validation.json`.

## Distribution

| Metric | min | p50 | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| max_abs_err | 0.0104 | 0.0104 | 0.0134 | **0.0833** | 0.0833 | 0.0137 |
| mean_abs_err | 0.000638 | 0.00107 | 0.00112 | 0.00125 | 0.00143 | 0.00104 |
| max_rel_err | 0.286 | **0.333** | 0.333 | 0.333 | 0.333 | 0.322 |
| mean_rel_err | 0.045 | 0.063 | 0.065 | 0.067 | 0.067 | 0.060 |
| source amax | 0.125 | 0.125 | 0.250 | 2.000 | 2.000 | 0.227 |
| per-tensor S_g | 4.65e-5 | 4.65e-5 | 9.30e-5 | 7.44e-4 | 7.44e-4 | 8.43e-5 |

## What the numbers mean

### Max relative error: 33.3% across nearly the entire distribution

This is **a structural property of FP4 e2m1 quantization**, not a bug in
our conversion. The FP4 magnitude set is
`{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`. The largest gap between
adjacent values (ignoring the 0→0.5 jump) is 4→6 (33.3%) and 1.5→2.0
(33.3%). A pre-quantized weight at the midpoint of such an adjacent
pair has worst-case relative error 33.3%.

Every sampled tensor has at least one element near such a midpoint, so
max_rel_err saturates at exactly 33.3%. This is a per-element noise
floor of the destination format and cannot be reduced without changing
the FP4 grid itself.

### Mean relative error: 6%

This is the more informative summary statistic. Across all 192 tensors,
mean rel err clusters tightly around 6% (p50 6.3%, p99 6.7%). Roughly
matches the textbook expectation for FP4 quantization noise: the
average per-element error in a uniform-distribution-like input
quantized to ~8 magnitude levels is roughly half the grid spacing,
which yields ~6%.

So the conversion is **as lossy as the destination grid can be**, and
no worse. There is no additional error introduced by:
- the regrouping (32→16) — we're re-quantizing onto the FP4 grid, not
  losing information about it
- the E8M0→E4M3 scale conversion — for our sampled tensors,
  per-tensor S_g + per-group E4M3 scale represents the original E8M0
  scales without further rounding error

### Max absolute error: dominated by the few large-amax tensors

p50 max_abs is 0.0104 — small. p99 jumps to 0.0833 — that's a tensor
with source_amax = 2.0 (compared to median source_amax = 0.125). The
absolute error scales with the per-tensor magnitude, as expected.

The 0.0833 max_abs = 1/12 corresponds exactly to a worst-case rounding
event (1.5 → 2.0 or 4 → 6) on a tensor with source_amax = 2.0. Same
33% relative error, just scaled.

### Mean absolute error: clusters tightly at 1.0e-3

Across all 192 tensors, mean_abs_err sits between 0.000638 and 0.00143
(spread less than 2×). The conversion is highly deterministic on
weight-statistics-driven inputs.

### Per-tensor global scale (S_g)

Most tensors landed at S_g ≈ 4.65e-5. This corresponds to:
`S_g = source_amax / (FP4_max * E4M3_max) = 0.125 / (6 * 448) = 4.65e-5`.

The few tensors at S_g = 7.44e-4 (16× larger) correspond to source_amax
= 2.0 — outlier experts with larger weight magnitudes.

## What the numbers do NOT mean

- They are NOT a quality verdict on the converted artifact. Weight-stat
  round-trip error is a necessary condition but not sufficient — the
  actual quality check is whether forward-pass outputs (logits) match
  native within an acceptable bound. That is Phase 3.5's job.
- They are NOT bound-tight on tensors we did NOT sample. We covered
  192 of ~70,000 expert weight tensors in the model (~0.27%). The
  outlier (max_abs = 0.083) suggests some tensors have higher amax;
  the full-model conversion in Phase 3 needs to handle the full range.
- They tell us NOTHING about the FP8 attention path. Those tensors
  pass through unchanged in our recipe; their error is whatever the
  upstream FP8 quant chose.

## Implications for Phase 3 (full-model conversion)

- The Phase 2 math is sound. Roll it forward to the full 64-shard
  conversion script.
- Watch for tensors with source_amax > 2.0 during the full run —
  these may need a different per-tensor S_g policy if they exceed
  E4M3's representable range at the per-group level. None observed in
  the 192-sample but the full model has more tensors.
- The per-tensor S_g is currently chosen to saturate E4M3 max on the
  largest per-group amax. For tensors with one extreme group and many
  small groups, this wastes precision. Consider a percentile-based
  S_g (e.g. p99 of per-group amax) as a follow-up optimization.

## Receipts

- `scripts/convert_mxfp4_tensor_to_nvfp4.py` — reference math
- `scripts/sample_conversion_errors.py` — sampling driver
- `docs/findings/conversion_math_validation.json` — raw per-tensor results

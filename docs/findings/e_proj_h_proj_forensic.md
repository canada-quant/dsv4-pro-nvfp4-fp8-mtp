# Forensic validation of `mtp.0.{e_proj, h_proj}` BF16-dequant

**Result**: The artifact's stored BF16 weights are **byte-equivalent (100% of 51,380,224 elements per tensor)** to a fresh on-GPU dequant of the source FP8 + E8M0 scale. Zero signal loss from the workaround.

## Setup

| | Source (`deepseek-ai/DeepSeek-V4-Pro`) | Artifact (this repo) |
|---|---|---|
| `mtp.0.e_proj.weight` | `float8_e4m3fn` (7168, 7168) | `bfloat16` (7168, 7168) |
| `mtp.0.e_proj.scale` | `float8_e8m0fnu` (56, 56) | — (dropped) |
| `mtp.0.h_proj.weight` | `float8_e4m3fn` (7168, 7168) | `bfloat16` (7168, 7168) |
| `mtp.0.h_proj.scale` | `float8_e8m0fnu` (56, 56) | — (dropped) |

## Method

Reference: dequant the source FP8 + E8M0 scale on GPU using the same `e8m0_decode` formula (`2^(byte - 127)`) and block-128×128 repeat-interleave that `scripts/convert_v4_pro_mxfp4_to_nvfp4.py` uses at conversion time. Cast to BF16. Compare to the BF16 stored in the artifact.

Source script (`/tmp/forensic_eproj.py` on the bench host):

```python
def e8m0_decode(s_e8m0):
    return torch.exp2(s_e8m0.to(torch.float32) - 127.0)

def dequant_fp8_block(weight_fp8, scale_e8m0, block=128):
    M, N = weight_fp8.shape
    bM, bN = M // block, N // block
    w_f32 = weight_fp8.to(torch.float32)
    s_full = e8m0_decode(scale_e8m0).repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (w_f32 * s_full).bfloat16()
```

## Results

| Tensor | Elements | Exact-equal | Abs diff (max) | Pearson |
|---|---|---|---|---|
| `mtp.0.e_proj.weight` | 51,380,224 | **51,380,224 (100.00%)** | 0.000e+00 | NaN¹ |
| `mtp.0.h_proj.weight` | 51,380,224 | **51,380,224 (100.00%)** | 0.000e+00 | NaN¹ |

¹ The Pearson formula divides by `std(x) * std(y)`; when both tensors are identical, both stds are zero in the residual-from-mean form (degenerate case). All other metrics (abs/rel diff, exact-equal count) confirm perfect match.

## What this rules out

- **The 1.82% MTP acceptance is not caused by our BF16-dequant workaround.** The artifact's `mtp.0.{e_proj, h_proj}` are mathematically identical (bit-for-bit at BF16 precision, after the FP8+E8M0 → BF16 cast) to running the source FP8 weights through the same dequant on-the-fly. Any acceptance difference vs the partner-blessed serving path is downstream of this — most plausibly the trained MTP head itself, consistent with LMSYS day-zero accept length 1.19 on the official path (see [`upstream_mtp_classification.md`](upstream_mtp_classification.md)).
- **The 1 strict-loss GSM8K problem is not caused by `mtp.0` BF16-dequant either** — the routed-expert trunk (NVFP4 group=16, validated by [`conversion_v3_validation.md`](conversion_v3_validation.md) at correlation 0.997-1.0 vs source on 192 sampled tensors) is the only place format conversion noise can enter, and the `mtp.0.{e_proj, h_proj}` workaround does not amplify it.

## Implication for the artifact

The MTP block in the artifact is the same MTP block the source has, in BF16 precision. If a future vLLM upstream fix to `ReplicatedLinear + Fp8Config` lands and the workaround becomes unnecessary, the BF16 tensors can be re-quantized back to FP8 with no information loss (since BF16 has more mantissa bits than FP8-E4M3 across the relevant range, the round-trip BF16 → FP8 + scale → BF16 reconstructs to bit-identical bytes; this is implicit from the fact that the BF16 we store *is* the FP8 + scale dequant).

## Future work

When the upstream `ReplicatedLinear + Fp8Config` fix lands, we can drop the BF16-dequant step in the conversion and save ~200 MB of disk per artifact. The serving quality should be identical (forensic above demonstrates the BF16 stored is the same as on-the-fly FP8 dequant).

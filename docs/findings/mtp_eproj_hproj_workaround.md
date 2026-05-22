# MTP `e_proj` / `h_proj` BF16-dequant workaround

## What we did

In the converted artifact, the two MTP-block linear projections `mtp.0.e_proj.weight` and `mtp.0.h_proj.weight` are stored as **BF16** rather than as FP8 block-quantized weights. The native V4-Pro source has these two tensors at FP8 with `.scale` sidecars; we dequantize them at conversion time using their on-disk FP8 + scale pair, write the BF16 result, and drop the `.scale` sidecar.

## Why

vLLM mainline (at the SHA we pin, `39910f2b25`, and the current HEAD `vllm/models/deepseek_v4/nvidia/mtp.py`) defines `e_proj` and `h_proj` as:

```python
self.e_proj = ReplicatedLinear(
    config.hidden_size, config.hidden_size, bias=False,
    return_bias=False, quant_config=quant_config,
)
self.h_proj = ReplicatedLinear(...)  # same shape
```

When `quant_config` is the main model's `Fp8Config`, `ReplicatedLinear + Fp8Config` does not consistently register the `weight_scale_inv` parameter under the name the MTP loader looks for. The loader transforms on-disk `mtp.0.e_proj.scale` to `model.layers.61.e_proj.weight_scale_inv` (where layer index 61 is the MTP block's reserved index, since the trunk uses 0-60), and `params_dict[name]` raises `KeyError: 'model.layers.61.e_proj.weight_scale_inv'`.

We confirmed this failure mode:

1. On `--tensor-parallel-size 8 --enable-expert-parallel` (the upstream `single_node_tep` default)
2. On `--data-parallel-size 8 --enable-expert-parallel` (the alternative `single_node_dep` strategy)

The error is identical in both topologies. It is a loader-path issue (parameter-registration name mismatch between `ReplicatedLinear + Fp8Config` and the MTP loader's expected key), not a topology-routing issue.

Our PR [#43319](https://github.com/vllm-project/vllm/pull/43319) attempts a candidate-list resolution: it tries `.weight_scale_inv`, `.weight_scale`, and the `.mtp_block.` variants in order. But none of those candidates resolve when `e_proj` / `h_proj` register *no* scale parameter at all — which is what's happening: the loader-side `params_dict` simply does not contain a scale slot for these two modules.

The proper upstream fix is for `ReplicatedLinear + Fp8Config` to register `weight_scale` (or `weight_scale_inv`) as a named parameter so the MTP loader can resolve it. That's the right place — across all callers of `ReplicatedLinear`, not just for V4-Pro MTP. We have not authored that fix yet.

## Why we picked dequant-at-conversion

Alternatives we considered:

1. **Fork vLLM and patch `ReplicatedLinear + Fp8Config`**. Would fix the root cause, but expanding beyond our 4 already-filed patches into core `ReplicatedLinear` modifications widens the change surface and the maintenance burden for our artifact's installer. Reserved for v0.2 or as a separate upstream PR not gated on this artifact's release.
2. **Strip MTP entirely** (the V4-Flash + RedHat default path via HF transformers' `_keys_to_ignore_on_load_unexpected`). Would lose the MTP retention story.
3. **Re-quantize `e_proj` / `h_proj` to NVFP4** along with the MoE experts. Would work in principle but: (a) the upstream MTP `ReplicatedLinear` was not designed for NVFP4 routing, so this would need its own loader patch, and (b) `e_proj`/`h_proj` are not MoE — they're per-token linear projections — so they're outside the NVFP4-on-MoE invariant that this conversion is built around.
4. **Dequant to BF16 and let mainline serve it as unquantized BF16**. Costs ~200 MB extra disk vs FP8 (each linear is hidden×hidden = 7168×7168 ≈ 100 MB at BF16 vs ~50 MB at FP8 + scale). Loads cleanly through the default-BF16 path in `ReplicatedLinear` with no special treatment. Compatible with all serving topologies, all backends.

(4) is what we picked. The disk cost (~200 MB out of 852 GB ≈ 0.024%) is negligible; the load reliability is total.

## Numerical impact

The BF16 → FP8 round trip at conversion is the only place numerical accuracy could be lost compared to the original FP8 versions. We dequantize using the source FP8 + per-block scales, producing BF16 weights that are byte-equivalent (up to the unavoidable FP8 quant-error of the source) to the original. We do **not** re-quantize back to FP8.

This means the BF16 weights we store are *more* precise than the original FP8 versions (BF16 has ~3-4 more bits of mantissa than FP8 E4M3 at the same exponent range), not less.

The only concern is whether the slight precision *gain* at `e_proj`/`h_proj` interacts oddly with the rest of the MTP forward pass — but the MTP forward consumes BF16 hidden-state input regardless, and the immediate downstream consumer is a RMSNorm, so the precision differential is reabsorbed within one op.

## Future work

- File a focused vLLM PR for `ReplicatedLinear + Fp8Config` to register `weight_scale_inv` properly. This would unblock the native MXFP4 + MTP serve path for everyone.
- Offline comparison: dequant the original `mtp.0.e_proj.weight` (FP8) to BF16 outside vLLM, compare element-wise to what we wrote in the artifact, confirm byte-equivalence. (We have the source on disk; this is a 5-minute check that should ship in v0.1.1.)
- Once the upstream fix lands and propagates to a release, the BF16-dequant of these two tensors becomes pointless; the artifact can be regenerated with FP8 `e_proj`/`h_proj` and save ~200 MB.

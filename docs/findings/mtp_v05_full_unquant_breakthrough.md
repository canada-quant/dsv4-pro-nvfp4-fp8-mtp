# MTP gap — v5 patch hypothesis tested, falsified

**Date:** 2026-05-23
**Status:** Closed (hypothesis falsified — root cause still unknown)

## Problem statement

Through the v0.2 / v0.3 / v0.4-bisect bisection we showed:

| Artifact | Trunk | mtp.0 | MTP acceptance (n=1) on our build |
|---|---|---|---|
| v0.2 | NVFP4 | NVFP4 + FP8 + BF16(e_proj/h_proj) | 3.07% |
| v0.3 | NVFP4 | BF16 (full) | 3.33% |
| v0.4-bisect | MXFP4 (passthrough) | BF16 (full) | 2.65% |
| Native MXFP4 + fork docker | MXFP4 | FP8 + MXFP4 (native) | **91.14%** |
| Native MXFP4 + our build | MXFP4 | FP8 + MXFP4 (native) | **fails to load** |

All artifacts we converted give the same ~3% on our build regardless of trunk format or MTP block format. The cause is in our build path, not the conversion artifact.

## Hypothesis (v5)

Mainline vLLM's `vllm/models/deepseek_v4/nvidia/mtp.py` has a `_mtp_block_is_quantized_on_disk` detection (from our own PR #43319). When MTP weights are BF16 on disk, it sets `quant_config = None` LOCALLY — used for `e_proj`, `h_proj`, `shared_head` construction. **But the inner `DeepseekV4DecoderLayer` reads `vllm_config.quant_config` directly** and constructs its `attn` and `ffn` (FusedMoE) with the trunk's NVFP4 quant_config. Since on-disk MTP weights are BF16 (no scale sidecars), the scale parameters never get loaded → MTP forward produces near-zero outputs → ~3% acceptance.

## v5 patch (applied, then reverted)

```python
if _mtp_block_is_quantized_on_disk(vllm_config):
    quant_config = vllm_config.quant_config
else:
    quant_config = None
    # v5: also null vllm_config.quant_config globally so the inner
    # DeepseekV4DecoderLayer (attn + ffn) builds unquantized.
    vllm_config.quant_config = None
    logger.info_once(...)
```

Both patches (v3 + v5) verified to fire on all 8 workers via log:
- `DeepseekV4MoE: prefix='model.layers.61.ffn' is MTP layer; mtp.* is BF16 on disk -> quant_config=None + moe_backend=triton`
- `MTP block weights are BF16 on disk ... Constructing the MTP draft tower without quantization`

## Result

| Artifact | MTP n=1 (no v5) | MTP n=1 (v5 applied) | Verdict |
|---|---|---|---|
| v0.4-bisect (MXFP4 trunk + BF16 mtp.0) | 2.65% | **2.65%** | Identical — v5 has no effect |

**v5 has no effect on MTP acceptance.** The hypothesis — that MTP block's attn/ffn quant_config mismatch is the cause — is falsified.

Patch reverted to keep mainline clean.

## Why v5 didn't help

Two non-mutually-exclusive possibilities:

1. **The MTP block's attn was already loading correctly even with NVFP4 quant_config**: maybe ReplicatedLinear with FP8Config registers parameters that default to identity-like values when no scale is loaded (so the matmul produces approximately correct BF16 output even without scales). Then v5 doesn't change the forward behavior.

2. **The bug is in the MTP forward path itself, not in load-time quant_config**: the MTP block math (hc_pre/hc_post/mhc_fused_post_pre, norm_gate refactoring) might differ semantically between mainline (May 22) and fork (Apr 25) in a way that breaks MTP draft hidden-state computation.

## What's actually different between mainline and fork

The fork's DSV4 code at `vllm/model_executor/models/deepseek_v4{,_mtp}.py` (Apr 25 2026) differs from mainline's `vllm/models/deepseek_v4/nvidia/{model,mtp}.py` (May 22 2026) in several substantive ways:

1. **DecoderLayer return signature**: fork returns `hidden_states`; mainline returns 4-tuple `(x, residual, post_mix, res_mix)` and defers the final `hc_post` to the caller.

2. **`mhc_fused_post_pre`**: mainline has a NEW fused kernel that combines the post-mix of one layer with the pre-mix of the next layer in a single op. Fork has separate `hc_pre` + `hc_post` calls per layer.

3. **`norm_gate` refactoring**: mainline integrates `ffn_norm` and the router gate matmul into a `NormGatedLinear` inside `DeepseekV4MoE`. Fork keeps them separate.

4. **MTP weight-name remapping**: mainline adds 3 new mappings for `ffn_norm.weight → ffn.norm_gate.norm.weight`, `ffn.gate.weight → ffn.norm_gate.gate.weight`, `ffn.gate.tid2eid → ffn.norm_gate.tid2eid`. Fork has none of these.

5. **mhc kernel organization**: fork has tilelang kernels inline in `vllm/model_executor/layers/mhc.py`. Mainline splits them into `vllm/model_executor/kernels/mhc/{tilelang,torch,triton,aiter}.py` with CustomOp dispatch.

Mathematically, these should be equivalent — the fused op should produce the same result as separate ops, and the norm_gate refactoring is just where the matmul lives. But ONE of these (most likely the fused op or the norm_gate refactoring) appears to break MTP semantically.

## Stacked-attn FP8 scale loader gap

Separate finding from testing native MXFP4 load on our build (with all 4 + v3 patches applied):

```
KeyError: 'model.layers.61.mtp_block.attn.fused_wqa_wkv.weight_scale_inv'
```

Native checkpoint has `mtp.0.attn.wq_a.scale` and `mtp.0.attn.wkv.scale` (per-key FP8 block scales). Mainline's loader stacks `wq_a` + `wkv` into `fused_wqa_wkv`, but the scale renaming + stacking path doesn't combine the scales — the loader tries to load `attn.wq_a.scale` → renamed to `attn.fused_wqa_wkv.weight_scale_inv` via stacked_params_mapping, which doesn't exist in params_dict.

This is a separate fix that would unblock our build from serving the native checkpoint. Filed as follow-up to PR #43319.

## Conclusion

The MTP gap on our build is real, reproducible, and not yet root-caused. The hypothesis tested in this experiment (`quant_config` propagation bug) is falsified. The remaining suspect surface is the mainline-vs-fork divergence in `mhc_fused_post_pre` + `norm_gate` semantics — but isolating and fixing that requires either:

(a) Mainline SHA bisection between Apr 25 and May 22 (multi-hour build cycles).
(b) Porting fork's loader + DecoderLayer code wholesale to mainline (multi-day effort).
(c) Filing the gap as a vLLM issue and waiting for the maintainers (who shipped both versions) to investigate.

**Filed:** vLLM issue [#43472](https://github.com/vllm-project/vllm/issues/43472) covering both findings (the stacked-attn loader gap and the mainline-vs-fork MTP acceptance gap).

The shipping decision for the canada-quant v0.3 artifact: ship as the **clean NVFP4 trunk quality** result (HumanEval 95.1%, MMLU-Pro 81.6%, IFEval 84.8%) with the **MTP gap honestly documented** as an open mainline-vs-fork mystery. Native MXFP4 + fork docker remains the recipe for working MTP at 91%.

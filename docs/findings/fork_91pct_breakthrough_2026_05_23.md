# Fork build at e8e38e16 — 91.07% MTP confirmed on our box

**Date:** 2026-05-23 (session 4 — afternoon)
**Status:** Decisive — fork's DSV4 + MTP path works at 91% on our box; mainline doesn't.

## What we did

Built `zyongye/vllm@e8e38e16` (the partner-blessed dsv4 branch tip) from source on our B300 box, with `TORCH_CUDA_ARCH_LIST=10.3a`. Resulting vllm version: `0.1.dev50+ge8e38e168.cu130`.

Served `/opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp` (native deepseek-ai V4-Pro checkpoint, MXFP4 trunk + FP8 attn + MXFP4 mtp.0) with `--moe-backend flashinfer_trtllm --speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.

Ran the same 20-prompt MTP probe used for all other measurements.

## Result

```
MTP: 3060/3360 = 91.07% acceptance rate over 20 prompts
```

Same artifact, same prompts, same hardware. Fork build at greedy temp=0: **91.07%**. Mainline build at greedy temp=0: **3%**.

## What this proves

1. The MTP head works correctly. The artifact data is fine.
2. The 91% upper bound is not docker-specific. It's a property of the fork's DSV4 implementation.
3. Mainline's DSV4 (`vllm/models/deepseek_v4/`) and the fork's (`vllm/model_executor/models/deepseek_v4{_mtp,}.py`) diverge in a way that costs ~88pt of MTP acceptance.

## What's still blocked

Loading the canada-quant NVFP4 artifacts on the fork build is more involved than expected. The fork's `deepseek_v4_attention.py` has FP8-specific `wo_a` einsum that fails when `wo_a` is unquantized (BF16). Patching that requires re-deriving the BF16 attention output path from scratch (fork's o-projection is FP8 throughout). Also, the fork doesn't have NVFP4 expert support (PR #42209 was post-fork).

Attempted but not yet working:
- v10 patch set: add `_mtp_block_is_quantized_on_disk` detection to fork's `deepseek_v4_mtp.py` and v3-equivalent moe_backend=triton override in fork's `deepseek_v4.py`. Patches fire correctly. But the FP8 attention forward path fails at first inference (`AttributeError: 'ColumnParallelLinear' object has no attribute 'weight_scale_inv'`).
- BF16 fallback for `wo_a` in fork's `deepseek_v4_attention.py`: shape-mismatch in the bf16 einsum reshape — the recipe needs more careful work to handle V4-Pro dims correctly.

## Next-session actions (research-grounded)

The cleanest path forward is one of:

1. **Author a CORRECT BF16 fallback for fork's attention.py.** The wo_a fp8 einsum path needs a proper BF16 sibling that handles V4-Pro's specific dims (n_local_groups, o_lora_rank, head_dim). Multi-hour but isolated.

2. **Make a new canada-quant artifact that the fork can load natively** — i.e., MXFP4 trunk passthrough + native FP8/MXFP4 mtp.0 (not BF16-dequant of mtp). This is essentially "native + the canada-quant tokenizer/config tweaks" — minimal differentiation vs native, but at least it would serve at 91% on the fork build.

3. **Port the fork's DSV4 attention forward to mainline.** The fork's deepseek_v4_attention.py has the FP8 einsum path that works with V4-Pro's MTP. Mainline's deepseek_v4/nvidia/attention.py might have a regression in the equivalent code. This is multi-day audit work.

4. **Wait for vLLM maintainer to act on issue #43472.** They have full reproduction context now (our 4 surgical patches + SHA bisect + fork-vs-mainline 91% vs 3% measurement).

## Build artifacts left on the box

- `/opt/dlami/nvme/src/vllm_fork` — the fork tree, currently with v10 patches applied (mtp.py BF16 detection, deepseek_v4.py v3-equivalent moe_backend override, deepseek_v4_attention.py wo_a BF16 fallback **with a shape bug**).
- `/opt/dlami/nvme/src/vllm` — mainline tree, restored to canada-quant patched May 22 base.
- `/opt/pytorch` venv currently holds the fork build (`vllm-0.1.dev50+ge8e38e168.cu130`). To restore the mainline build, re-run `cd /opt/dlami/nvme/src/vllm && pip install -e . --no-build-isolation`.

## Measurement table — complete

| Build | Artifact | MTP n=1 | Notes |
|---|---|---|---|
| Mainline 39910f2b25 + 4 patches | v0.2 (NVFP4 trunk + NVFP4/FP8 mtp.0) | 3.07% | baseline |
| Mainline + 5 patches (v3 BF16-mtp dispatch) | v0.3 (NVFP4 trunk + BF16 mtp.0) | 3.33% | |
| Mainline + 5 patches | v0.4-bisect (MXFP4 trunk + BF16 mtp.0) | 2.65% | |
| Mainline + v5 patch (full unquant MTP block) | v0.4-bisect | 2.65% | falsified |
| Mainline + v6 (revert mhc_post_pre kernel) | v0.4-bisect | 2.85% | falsified |
| Mainline + v7 (revert topk buffer share) | v0.4-bisect | 2.65% | falsified |
| Mainline + v8 (ignored_layers config) | v0.3 + ignored_layers | 2.96-3.41% | falsified |
| Mainline 0ce6613b9 (pre-#41035) + patches | v0.4-bisect | 2.77% | falsified |
| Mainline 879a8c318 (pre-#41536) + patches | v0.4-bisect | 2.76% | falsified |
| **Fork e8e38e16 + base patches** | **native MXFP4** | **91.07%** | ✓ confirmed on our box |

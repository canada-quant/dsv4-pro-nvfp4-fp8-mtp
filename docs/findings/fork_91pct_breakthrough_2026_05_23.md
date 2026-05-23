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

After this session's deep dive, the realistic paths are:

1. **Author CORRECT BF16 fallback for fork's attention.py** — *attempted in this session*. The wo_a BF16 einsum was actually math-correct (counting outputs work, "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" produced cleanly). But MTP still 2.51% — the bug is not in wo_a, it's that the fork's BF16 MoE forward doesn't implement V4-Pro's `sqrtsoftplus + renormalize` routing. With BF16 MoE backend = triton, routing falls back to Default/Unspecified → wrong experts selected → garbage drafts. **Fixing this requires either implementing sqrtsoftplus+renormalize in triton MoE (deep) or keeping mtp.0 quantized so the FP8 MoE path stays active (which means no BF16 dequant — see option 2).**

2. **Make canada-quant artifact with native mtp.0 passthrough.** Skip the BF16 dequant of mtp.0 entirely. Trunk: MXFP4 passthrough. mtp.0: native FP8/MXFP4 passthrough. Result: artifact is byte-identical to native on the trunk + mtp. Fork loads it at 91%. But this provides ~zero differentiation from `deepseek-ai/DeepSeek-V4-Pro` — the canada-quant value-add disappears.

3. **Combine NVFP4 trunk + native mtp.0 passthrough.** Trunk: NVFP4 (our differentiation). mtp.0: native passthrough (so MTP MoE routing works). For this to serve, we need NVFP4 trunk support — mainline has it (PR #42209) but its MTP is broken (3%); fork doesn't have it. So this combination is **unservable on either build today**.

4. **Wait for vLLM maintainer to fix mainline DSV4 MTP** — issue #43472 has full reproduction context (4 surgical patches + SHA bisect + fork-vs-mainline 91% vs 3% measurement on same hardware).

5. **Port NVFP4 expert support (PR #42209) into the fork build.** Multi-day. The fork is older infrastructure; PR #42209 was designed against modern mainline. Backporting would require resolving conflicts in compressed-tensors, ModelOptNvFp4FusedMoE registration, the routing dispatch in `trtllm_nvfp4_moe.py`, and possibly the loader. Probably 2-3 days of focused work.

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
| **Fork e8e38e16 (no patches)** | **native MXFP4** | **91.07%** | ✓ first confirm on our box |
| **Fork e8e38e16 (no patches, reconfirm)** | **native MXFP4** | **91.94%** | ✓ reconfirmed |
| Fork e8e38e16 + v10 patches (BF16-mtp loader) | v0.4-bisect | 2.51% | tested; loads, low MTP. V4-Pro sqrtsoftplus+renormalize routing isn't implemented for unquantized BF16 MoE in fork — triton backend silently does wrong routing → garbage drafts. |

# v0.3 BF16-mtp.0 experiment — hypothesis falsified (2026-05-23)

## TL;DR

Re-converted V4-Pro with **the entire `mtp.0` block dequantized to BF16** (mirrors V4-Flash recipe which achieves 80%+ MTP acceptance) instead of NVFP4 experts + BF16 e_proj/h_proj. Result: **3.33% MTP acceptance at n=1**, essentially unchanged from v0.2's 3.07%. The hypothesis that NVFP4 quantization of `mtp.0.ffn.experts.*` was the cause of our MTP weakness is **falsified**.

This rules out one major suspect and points the root cause at either (a) the trunk's NVFP4 activation distribution being incompatible with V4-Pro's trained MTP head, or (b) a divergence in mainline vLLM's MTP forward path vs zyongye's fork.

## Setup

**v0.3 artifact**: identical to v0.2 except mtp.0 block fully BF16 (1170 weight keys, 0 quant-scale sidecars):
- `mtp.0.ffn.experts.*.w{1,2,3}.weight` — BF16 (dequanted from source MXFP4)
- `mtp.0.attn.{wkv, wo_a, wo_b, wq_a, wq_b}.weight` — BF16 (dequanted from source FP8 block)
- `mtp.0.ffn.shared_experts.{w1, w2, w3}.weight` — BF16 (dequanted from source FP8 block)
- `mtp.0.{e_proj, h_proj}.weight` — BF16 (same as v0.2)
- `mtp.0.attn.{attn_sink, kv_norm, q_norm, attn_norm, enorm, hnorm}.weight` — BF16 (passthrough)
- Trunk experts (`layers.0..60.ffn.experts.*`): NVFP4 (same as v0.2)
- Trunk attention: FP8 block 128×128 (same as v0.2)

Total size: **950 GB** (vs v0.2's 852 GB — the +98 GB is mostly the BF16 mtp experts where v0.2 had NVFP4-packed).

**vLLM modifications** required to load the BF16 mtp.0 block on mainline + 4 patches:

1. **v3 patch** in `vllm/models/deepseek_v4/nvidia/model.py`: detect MTP layer by `prefix` matching the regex `\.layers\.(\d+)\.` and layer index ≥ `num_hidden_layers`. For MTP layers, check the safetensors index for absence of quant-scale sidecars under `mtp.*`; if absent, set `quant_config = None` so the FusedMoE inside the MTP block allocates BF16 slots instead of NVFP4-packed slots.
2. **v4 patch** in the same file: after setting `quant_config = None`, override `vllm_config.kernel_config.moe_backend = 'triton'` for the MTP layer's FusedMoE construction. The default `flashinfer_trtllm` unquantized backend doesn't support V4-Pro's routing method 100; triton handles arbitrary routing.

Both patches together let the v0.3 artifact load and serve under MTP.

**Serve config**: TP=8 + EP + indexer_cache + FULL_AND_PIECEWISE cuda graphs + `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, same as the v0.2 measurement.

## Measurement

| Run | Artifact | mtp.0 format | MTP n | Drafts | Accepted | Per-token | Accept length |
|---|---|---|---|---|---|---|---|
| Baseline v0.2 | v0.2 (852 GB) | NVFP4 experts + BF16 e_proj/h_proj | 1 | 6,030 | 185 | 3.07% | 1.031 |
| Baseline v0.2 | v0.2 (852 GB) | NVFP4 experts + BF16 e_proj/h_proj | 2 | 13,180 | 240 | 1.82% | 1.036 |
| **v0.3** | v0.3 (950 GB) | **BF16 entire mtp.0** | 1 | 13,586 | 452 | **3.33%** | 1.033 |
| Native (fork docker) | upstream native | MXFP4 experts + FP8 e_proj/h_proj | 1 | 3,330 | 3,035 | **91.14%** | 1.911 |
| Native (fork docker) | upstream native | MXFP4 experts + FP8 e_proj/h_proj | 2 | 4,886 | 3,936 | **80.56%** | 2.611 |

Same 20-prompt chat workload across all runs. v0.3 at n=1 (3.33%) is essentially unchanged from v0.2 at n=1 (3.07%) — the small bump is within sampling noise on 6k-13k draft tokens.

## What this rules out

- ❌ **NVFP4 quantization of `mtp.0.ffn.experts.*` damages the MTP head's outputs.** Falsified — BF16 mtp experts give the same acceptance.
- ❌ **The BF16 dequant of `mtp.0.{e_proj, h_proj}` interacts poorly with FP8-block `mtp.0.attn.*`.** Falsified — the all-BF16 mtp.0 in v0.3 doesn't help.
- ❌ **The patch #43319 `_mtp_block_is_quantized_on_disk` detector misroutes the MTP draft tower's quant_config.** Falsified — even with the explicit v3 patch correctly identifying the MTP layer and setting `quant_config = None`, MTP stays at ~3%.

## What's still on the table

- ⚠️ **Trunk NVFP4 quantization shifts the activation distribution flowing into the MTP draft head**, in a way V4-Flash's NVFP4 calibration didn't (V4-Flash got 81%+ acceptance with NVFP4 trunk + BF16 mtp). V4-Pro's MTP head may be more sensitive to activation drift than V4-Flash's at its larger scale (61 layers × 7168 hidden vs 43 × 4096). The only fix is to either (a) re-train the MTP head against NVFP4-trunk activations (out of scope), or (b) keep the trunk in a format the MTP head was trained for (MXFP4 — but that defeats the purpose of NVFP4 conversion).
- ⚠️ **Mainline vLLM's MTP forward path has a semantic divergence from zyongye's fork that affects acceptance.** Could be in the MTP `compute_logits` path, the speculative-decoder's `RejectionSampler`, the routing-method dispatch, or interactions between FlashInfer NVFP4 trunk and triton BF16 MTP MoE outputs. Conclusively bisecting this needs a vLLM build from zyongye's fork @bc34b25e/e8e38e1 + cherry-pick of PR #42209's NVFP4 routing — a multi-hour rebuild that's out of scope for this session.

## Ship decision

v0.2 remains the public artifact. Reasons:

- v0.3 doesn't deliver the MTP fix it was hypothesized to provide.
- v0.3 is 98 GB larger and requires 2 additional vLLM patches (v3 + v4) to load.
- The trunk quality is unchanged in both (GSM8K 96.89%, MMLU-Pro 81.64%, HumanEval 95.1%) — the only thing affected is MTP speculative-decoding acceptance.
- Users who care about throughput / batched serving get the same product from v0.2 with fewer patches.

The MODEL_CARD now honestly acknowledges: V4-Pro MTP head is healthy (91% on native + fork docker), but our (NVFP4-trunk + mainline-patched serve) pipeline drops it to ~3%. Settings and mtp.0 format don't recover it. Recommend serving without MTP for maximum throughput on this artifact.

## Reproducibility

The v3/v4 vLLM patches and the modified `scripts/convert_v4_pro_mxfp4_to_nvfp4.py` (with `mtp_expert_*` and `mtp_fp8_block_*` BF16-dequant routing) are preserved in the repo for anyone who wants to attempt the experiment themselves. The raw JSON is at `docs/benchmarks/matrix/mtp_v03_bf16mtp_n1.json`.

## Open issue for v0.4 or community

A vLLM upstream PR could combine:
- The v3 + v4 patches above (correctly handle BF16 MTP block on mainline)
- A cleaner version of patch #43319's logic
- Possibly the fork's MTP forward semantics ported to mainline (the conclusive test)

That work is non-trivial — likely needs cooperation with zyongye/maintainers and a day of focused build/test cycles.

# Native MXFP4 + MTP vs our NVFP4 + MTP — corrected picture (2026-05-23)

## TL;DR

The native `deepseek-ai/DeepSeek-V4-Pro` checkpoint served via the partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image (zyongye fork build, vLLM `v0.1.dev15833+g62d441ee8`) shows **91.14% MTP per-token acceptance at n=1** and **80.56% at n=2** on the same 20-prompt workload we used for our 1.82% NVFP4 measurement. That settles the question raised by zyongye on issue #43455 — V4-Pro MTP is NOT structurally weak. The earlier "`opt_in_features` matches structural V4-Pro MTP capability" framing in `upstream_mtp_classification.md` was wrong, and is being retracted here.

## 2×2 matrix

| Build | Format | MTP n | Draft tokens | Accepted | Per-token acceptance | Accept length |
|---|---|---|---|---|---|---|
| Fork docker (`v0.1.dev15833+g62d441ee8`) | Native MXFP4 | 1 | 3,330 | 3,035 | **91.14%** | 1.911 |
| Fork docker (`v0.1.dev15833+g62d441ee8`) | Native MXFP4 | 2 | 4,886 | 3,936 | **80.56%** | 2.611 |
| Mainline @ `39910f2b25` + 4 patches | Our NVFP4 | 1 | 6,030 | 185 | **3.07%** | 1.031 |
| Mainline @ `39910f2b25` + 4 patches | Our NVFP4 | 2 | 13,180 | 240 | **1.82%** | 1.036 |

Gap analysis:

| n | Native | Ours | Ratio |
|---|---|---|---|
| 1 | 91.14% | 3.07% | **29.7×** |
| 2 | 80.56% | 1.82% | **44.3×** |

Both configs degrade similarly between n=1 and n=2 (native: -10.6pt; ours: -1.2pt absolute). The multi-forward chain affects both paths in the same direction — it's not the dominant factor in the gap.

The vLLM warning ("Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate") applies to both paths and explains the n=1 → n=2 degradation, but at the head-quality level it's not the cause of the 29-44× gap.

## What can't be cleanly isolated

The 2×2 isn't a pure on-checkpoint test because each (format × build) cell requires a different combination:

- **Native MXFP4 + mainline + 4 patches**: cannot load. Hits `KeyError: 'model.layers.61.e_proj.weight_scale_inv'` at weight load on the `ReplicatedLinear + Fp8Config` scale-param registration gap on `mtp.0.{e_proj, h_proj}`. Documented at [`mtp_eproj_hproj_workaround.md`](mtp_eproj_hproj_workaround.md). Our PR #43319's candidate-list mitigation doesn't cover it.
- **Our NVFP4 + fork docker**: cannot load. The image predates PR #42209's NVFP4 MoE routing merge (image's vLLM is from ~April 24; #42209 merged 2026-05-22). Documented at [`lambda_docker_portability.md`](lambda_docker_portability.md).

So we can only measure the two corners that work. We can't separate "our NVFP4 conversion damages MTP" from "our mainline-patched build serves MTP wrong".

## Hypotheses for the 29-44× gap

1. **NVFP4 quantization of `mtp.0.ffn.experts.*`** could damage the draft head's outputs even though the trunk experts measured correlation 0.997-1.0 vs source on 192 sampled tensors ([`conversion_v3_validation.md`](conversion_v3_validation.md)). The MTP draft head consumes the trunk's hidden state and projects through `mtp.0.{e_proj, h_proj, attn, ffn}`; small per-tensor noise can compound through this chain in a way that doesn't show up on the trunk's per-tensor sampling test.
2. **`mtp.0.{e_proj, h_proj}` BF16-dequant interaction** — the BF16 weights are byte-equivalent to source FP8 dequant ([`e_proj_h_proj_forensic.md`](e_proj_h_proj_forensic.md), 100% of 51M elements per tensor), but the FORWARD path through unquantized BF16 `ReplicatedLinear` (no FP8 scale-mult, no activation requant) is different from the FP8-block forward in the fork docker. The activation distribution flowing into `mtp.0.ffn` may end up slightly off the calibration the FlashInfer NVFP4 trunk experts were trained for.
3. **PR #43319's `_mtp_block_is_quantized_on_disk` detector** in our patched build returns True for our artifact (because MTP experts have `weight_scale*` sidecars), correctly applying `quant_config` to the MTP draft tower. But the detector's quantization-suffix list is `(.weight_scale, .weight_scale_inv, .weight_packed, .weight_global_scale, .input_global_scale, .weight_zero_point)` — it does NOT include `.scale` (without `weight_` prefix). For the native MXFP4 source the MTP scales end in `.scale`, so the detector misclassifies native MTP as BF16-on-disk and bypasses `quant_config`. This doesn't affect our artifact, but it's a defect in our patch worth fixing as part of #43319.

## Implications for v0.2+

- The MODEL_CARD's "MTP retained but limited by upstream V4-Pro MTP maturity" framing is being retracted. The native MTP works at ~91% at n=1; whatever drops ours to 3% lives inside our conversion + serve pipeline.
- Issue #43455 is updated with this comparison.
- Open work: try to bisect (1) vs (2) vs (3) by partial-conversion experiments. The fork-docker path can't help us measure since it can't load our NVFP4. Possible angle: produce a "transitional" artifact that keeps MTP in native MXFP4 + uses NVFP4 only for the trunk, serve on the fork docker (no — fork docker doesn't have #42209 to dispatch trunk NVFP4 either). So the bisection probably needs to happen inside our build, via toggling patch #43319's detector behavior and the BF16 vs FP8 e_proj/h_proj choice.

## Raw artifacts

- `docs/benchmarks/matrix/mtp_native_mxfp4_n1_2026_05_23.json`
- `docs/benchmarks/matrix/mtp_native_mxfp4_n2_2026_05_23.json`
- `docs/benchmarks/matrix/mtp_ours_nvfp4_n1_2026_05_23.json`
- `docs/benchmarks/matrix/mtp_A_nvfp4_flashinfer_mtp_2026_05_22.json` (the original 1.82% measurement at n=2)

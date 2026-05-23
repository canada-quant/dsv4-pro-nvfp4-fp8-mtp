# Findings index — V4-Pro NVFP4-FP8-MTP

## Methodology and measurements

- [`backend_format_matrix.md`](findings/backend_format_matrix.md) — NVFP4 × MXFP4 × flashinfer × deep_gemm matrix + c=16 batched extension + matched GSM8K-300. Headline: NVFP4 +41% aggregate at c=16, +8% at c=1.
- [`throughput_scaling.md`](findings/throughput_scaling.md) — concurrency sweep c=1 → c=128. Headline: peak aggregate 1606 tok/s at c=64; throughput drops past c=64 → c=128 (production sweet spot c=32-64).
- [`aime30_full_2026_05_22.md`](findings/aime30_full_2026_05_22.md) — full 30-problem AIME 2024 thinking=high run. Headline: 20/30 raw = 66.67%, 20/29 non-truncated = 68.97%, Wilson CI [0.49, 0.81] includes DeepSeek's reported V4-Pro range.
- [`gsm8k_scoring_correction.md`](findings/gsm8k_scoring_correction.md) — corrected bench string-match scorer to numeric. NVFP4 GSM8K-full goes 94.09% → 96.89%, matched-300 NVFP4 vs MXFP4 goes from -2.33pt apparent gap to -0.33pt real gap (within Wilson CI overlap, 1 strict-loss problem).
- [`e_proj_h_proj_forensic.md`](findings/e_proj_h_proj_forensic.md) — offline forensic confirms artifact's `mtp.0.{e_proj, h_proj}` BF16 is 100% byte-equivalent to source FP8 dequant (51M elements per tensor, zero noise). The MTP weakness is not caused by the workaround.
- [`mmlu_pro_2026_05_22.md`](findings/mmlu_pro_2026_05_22.md) — MMLU-Pro 5-shot (full 12,032 test set). Headline: 0.8164 ± 0.0034, +0.5pt vs V4-Flash predecessor. Per-subject same pattern (law lowest 0.61, math highest 0.92).
- [`ifeval_2026_05_22.md`](findings/ifeval_2026_05_22.md) — IFEval (zero-shot, chat template). Headline: prompt_level_strict 0.8484 ± 0.0154, slightly below V4-Flash NVFP4 (-0.6pt) but +2.8pt above RedHat V4-Flash NVFP4.
- [`humaneval_2026_05_22.md`](findings/humaneval_2026_05_22.md) — HumanEval / HumanEval+ via EvalPlus (greedy, n=1). Headline: HumanEval **0.951**, HumanEval+ **0.896** — beats V4-Flash NVFP4 by +3.6pt / +4.2pt. Strongest quality lift measured.
- [`lambda_docker_portability.md`](findings/lambda_docker_portability.md) — Partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image cannot load our artifact (same `KeyError: w13_input_scale` as the deep_gemm path on mainline). The image predates PR #42209's NVFP4 routing. Use our mainline + 4-patches build path instead.
- [`mtp_native_vs_ours_2026_05_23.md`](findings/mtp_native_vs_ours_2026_05_23.md) — Same V4-Pro MTP head measured on native + fork docker (91% at n=1, 80.56% at n=2) vs our v0.2 NVFP4 build (3.07%/1.82%). 2×2 matrix isolates this as a build+conversion-path issue, not a structural V4-Pro MTP weakness.
- [`mtp_v03_bf16_block_bisection.md`](findings/mtp_v03_bf16_block_bisection.md) — v0.3 re-conversion with entire `mtp.0.*` as BF16 (matches V4-Flash recipe). Result: 3.33% — falsifies the "NVFP4 mtp experts cause the gap" hypothesis.
- [`mtp_v05_full_unquant_breakthrough.md`](findings/mtp_v05_full_unquant_breakthrough.md) — Three surgical patches tested (v5 = full-block unquant, v6 = revert PR #41536 fused kernel, v7 = revert PR #42538 `topk_indices_buffer` sharing). All falsified the corresponding hypothesis. Remaining suspect surface: V1 spec_decode rejection-sampler commits (#41035, #40269, #40651). Filed as [vLLM #43472](https://github.com/vllm-project/vllm/issues/43472).
- [`overnight_2026_05_23_session.md`](findings/overnight_2026_05_23_session.md) — Session log for the overnight MTP-recovery attempt: 4 surgical experiments + 1 native-on-our-build load test, all negative. Documents what's ruled out, what's left to bisect, and the next-session pickup path.
- [`mtp_sha_bisect_2026_05_23.md`](findings/mtp_sha_bisect_2026_05_23.md) — Session 2 SHA bisect: built mainline at pre-#41035 (May 12) and pre-#41536 (May 10) SHAs. Both give ~2.76% MTP. **The fork at e8e38e16 is NOT a mainline ancestor** — it's a separate Apr 13 branch with independent DSV4 work. Mainline's DSV4 (via #40860) has the bug intrinsically; SHA bisection cannot isolate it.
- [`fork_91pct_breakthrough_2026_05_23.md`](findings/fork_91pct_breakthrough_2026_05_23.md) — Session 4 (afternoon): built `zyongye/vllm@e8e38e16` from source on our B300 box. Served native MXFP4 V4-Pro with `--speculative-config method=mtp num_speculative_tokens=1`. **Result: 91.07% MTP acceptance** on the SAME 20-prompt probe that gives 3% on mainline. Confirms the 91% is not docker-specific — it's a property of the fork's DSV4 implementation. Loading the canada-quant NVFP4 artifact on the fork is still blocked by the fork's FP8-only `wo_a` einsum path that fails when wo_a is unquantized BF16.
- [`upstream_mtp_classification.md`](findings/upstream_mtp_classification.md) — vLLM/LMSYS evidence that V4-Pro MTP is upstream-known to be weak. Used to contextualize our 1.82% acceptance measurement.
- [`mtp_eproj_hproj_workaround.md`](findings/mtp_eproj_hproj_workaround.md) — why `mtp.0.{e_proj, h_proj}` are stored as BF16 in this artifact (vLLM `ReplicatedLinear + Fp8Config` scale-param-registration gap).
- [`conversion_math_validation.md`](findings/conversion_math_validation.md) — initial 4-tensor byte-level dequant verification.
- [`conversion_v3_validation.md`](findings/conversion_v3_validation.md) — full 192-sampled-tensor validation (correlation 0.997-1.0 vs source).
- [`phase_0_closeout.md`](findings/phase_0_closeout.md) — early-stage closeout of the loader/MTP-load discovery, mostly superseded by `mtp_eproj_hproj_workaround.md` and `upstream_mtp_classification.md`.
- [`phase_6_quality_summary.md`](findings/phase_6_quality_summary.md) — interim quality-summary doc, since superseded by the matrix's matched GSM8K-300 results.

## Upstream research

- [`upstream_research_v4_pro_ground_truth.md`](findings/upstream_research_v4_pro_ground_truth.md) — V4-Pro architecture + tokenizer + config audit before conversion.
- [`v4_pro_active_params.md`](findings/v4_pro_active_params.md) — active-param math vs DeepSeek's published numbers.
- [`vllm_pro_serving_path.md`](findings/vllm_pro_serving_path.md) — vLLM model-class location + how the V4-Pro subpackage routes.
- [`v4_flash_patches_after_pro_subpackage.md`](findings/v4_flash_patches_after_pro_subpackage.md) — which of the V4-Flash patches still apply to V4-Pro and which are subsumed.

## vLLM serve issues

See [`VLLM_SETUP_ISSUES.md`](VLLM_SETUP_ISSUES.md) for the 4 patches + setup gotchas catalog.

## Benchmark raw artifacts

All under [`benchmarks/`](benchmarks/) — per-benchmark JSON outputs. The backend-format matrix and extension cells are in [`benchmarks/matrix/`](benchmarks/matrix/).

## Predecessor reference

Predecessor V4-Flash findings: `https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/tree/main/docs/findings/`. Several patterns carry over (the 4 vLLM patches, the MTP-retention pattern, the bench harness structure); the V4-Pro additions are the format conversion (vs calibration), the `mtp.0.{e_proj,h_proj}` workaround, the deep_gemm + NVFP4 gap, and the upstream-MTP classification context.

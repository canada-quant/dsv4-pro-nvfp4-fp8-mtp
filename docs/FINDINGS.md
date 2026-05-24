# Findings index — V4-Pro NVFP4-FP8-MTP

## Start here

- [`v12_nvfp4_mtp_working_2026_05_24.md`](findings/v12_nvfp4_mtp_working_2026_05_24.md) — **The full 5-step recipe** for native-parity MTP on mainline vLLM with cuda graphs ON. Conversion (NVIDIA-recipe `mtp.*` byte-passthrough) + four vLLM PR patches (#43248, #43288, #43290, #43319 with the `.scale` detector fix, #43467) + one local DSV4FP8Config per-layer MoE routing patch + flashinfer 0.6.8 pin.
- [`v12_nvidia_recipe_2026_05_24.md`](findings/v12_nvidia_recipe_2026_05_24.md) — Conversion-side alignment with NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe.
- [../MODEL_CARD.md](../MODEL_CARD.md) — HF-friendly summary with headline numbers.
- [../docs/QUICKSTART.md](QUICKSTART.md) — end-to-end serve recipe.
- [../docs/VLLM_SETUP_ISSUES.md](VLLM_SETUP_ISSUES.md) — 5 patches + setup gotchas catalog.

## Best-of-best v12 measurements (single-node 8× B300, TP=8 + EP, MTP n=1 + cuda graphs ON, `--max-model-len 65536`)

| Metric | Value | Detail |
|---|---|---|
| MTP draft acceptance (n=1, focused) | **91.21%** | 20-prompt probe; 3010/3300 |
| MTP draft acceptance (cumulative, MTP + AIME thinking=high) | **92.83%** | 37,341/40,225 |
| GSM8K full n=1319 | **96.59%** | CI [95.47%, 97.44%], 0 truncations |
| AIME 2024 thinking=high | **21/30 = 70.00%** | `max_tokens=60000`, 0 truncations |
| HumanEval pass@1 (EvalPlus greedy) | **0.951** | |
| HumanEval+ pass@1 (EvalPlus greedy) | **0.902** | |
| Throughput c=1 (MTP n=1 + cuda graphs) | **139.3 tok/s** | |
| Throughput c=16 aggregate | **672.6 tok/s** | |
| Throughput c=64 aggregate | **1,927.3 tok/s** | |
| Throughput c=128 aggregate | **3,004.8 tok/s** | |

## Quality benchmarks (additional)

- [`humaneval_2026_05_22.md`](findings/humaneval_2026_05_22.md) — HumanEval / HumanEval+ via EvalPlus.
- [`mmlu_pro_2026_05_22.md`](findings/mmlu_pro_2026_05_22.md) — MMLU-Pro 5-shot (predecessor measurement; v12 run queued).
- [`ifeval_2026_05_22.md`](findings/ifeval_2026_05_22.md) — IFEval zero-shot (predecessor measurement; v12 run queued).
- [`aime30_full_2026_05_22.md`](findings/aime30_full_2026_05_22.md) — AIME 2024 thinking=high methodology + v12 best-of-best result.
- [`gsm8k_scoring_correction.md`](findings/gsm8k_scoring_correction.md) — numeric-match scorer fix (predecessor doc; matters for any rerun).

## Throughput

- [`throughput_scaling.md`](findings/throughput_scaling.md) — concurrency sweep c=1 → c=128.
- [`backend_format_matrix.md`](findings/backend_format_matrix.md) — NVFP4 × MXFP4 × flashinfer × deep_gemm matrix. Now used as reference for "why `flashinfer_trtllm` is required for NVFP4 artifacts."

## Forensic + math validation

- [`conversion_v3_validation.md`](findings/conversion_v3_validation.md) — 192-sampled-tensor dequant validation (correlation 0.997-1.0 vs source).
- [`conversion_math_validation.md`](findings/conversion_math_validation.md) — initial 4-tensor byte-level dequant verification.
- [`e_proj_h_proj_forensic.md`](findings/e_proj_h_proj_forensic.md) — historical artifact from the v0.3 BF16-dequant era. Now moot since v12's MTP block is byte-passthrough.

## Upstream architecture research

- [`upstream_research_v4_pro_ground_truth.md`](findings/upstream_research_v4_pro_ground_truth.md) — V4-Pro architecture + tokenizer + config audit before conversion.
- [`v4_pro_active_params.md`](findings/v4_pro_active_params.md) — active-param math vs DeepSeek's published numbers.
- [`vllm_pro_serving_path.md`](findings/vllm_pro_serving_path.md) — vLLM DSV4 model-class location.
- [`v4_flash_patches_after_pro_subpackage.md`](findings/v4_flash_patches_after_pro_subpackage.md) — V4-Flash patch inheritance.
- [`upstream_mtp_classification.md`](findings/upstream_mtp_classification.md) — vLLM/LMSYS classification of DSV4 MTP head behavior (early-debug context).

## Historical MTP debug arc (kept for receipts)

These document the dead-ends, falsifications, and SHA bisections that ran for ~3 days before v12's NVIDIA-recipe alignment unblocked the work. Each is correct as a snapshot of its session. They are NOT load-bearing for current operation of the artifact.

- [`mtp_native_vs_ours_2026_05_23.md`](findings/mtp_native_vs_ours_2026_05_23.md)
- [`mtp_v03_bf16_block_bisection.md`](findings/mtp_v03_bf16_block_bisection.md)
- [`mtp_v05_full_unquant_breakthrough.md`](findings/mtp_v05_full_unquant_breakthrough.md)
- [`overnight_2026_05_23_session.md`](findings/overnight_2026_05_23_session.md)
- [`mtp_sha_bisect_2026_05_23.md`](findings/mtp_sha_bisect_2026_05_23.md)
- [`fork_91pct_breakthrough_2026_05_23.md`](findings/fork_91pct_breakthrough_2026_05_23.md)
- [`mtp_eproj_hproj_workaround.md`](findings/mtp_eproj_hproj_workaround.md)
- [`option_a_lossless_audit.md`](findings/option_a_lossless_audit.md)
- [`lambda_docker_portability.md`](findings/lambda_docker_portability.md)
- [`phase_0_closeout.md`](findings/phase_0_closeout.md), [`phase_6_quality_summary.md`](findings/phase_6_quality_summary.md)

## Predecessor reference

V4-Flash findings: <https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/tree/main/docs/findings/>. The V4-Pro additions over V4-Flash are: format conversion (vs calibration), NVIDIA-recipe MTP passthrough, the `.scale` detector fix, and per-layer MoE routing for the hybrid NVFP4-trunk + MXFP4-MTP shape.

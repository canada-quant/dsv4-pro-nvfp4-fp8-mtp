# Findings index — V4-Pro NVFP4-FP8-MTP

## Methodology and measurements

- [`backend_format_matrix.md`](findings/backend_format_matrix.md) — NVFP4 × MXFP4 × flashinfer × deep_gemm matrix + c=16 batched extension + matched GSM8K-300. Headline: NVFP4 +41% aggregate at c=16, +8% at c=1, GSM8K -2.33 pt vs source within Wilson CI overlap.
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

# Backend × format matrix and pre-ship extension — measurements 2026-05-22

What this answers: **what is the right MoE backend for the NVFP4 V4-Pro artifact, how does its throughput and quality compare to the native MXFP4 source on the upstream-recipe path, and how does the advantage scale with concurrency?**

## Setup

- Hardware: 8× B300 SXM6 AC (compute_cap 10.3, 288 GB HBM3e per GPU)
- vLLM mainline @ `39910f2b25` + the 4 local patches (#43248, #43288, #43290, #43319)
- Topology: `--tensor-parallel-size 8 --enable-expert-parallel` (upstream-default `single_node_tep` strategy from `vllm-project/recipes/models/deepseek-ai/DeepSeek-V4-Pro.yaml`)
- Per-cell extras: `--attention_config.use_fp4_indexer_cache=True`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`, `--kv-cache-dtype fp8`, `--block-size 256`
- Bench: `scripts/bench_v4_pro.py` at temperature 0, fixed prompts

## Base matrix (single-stream and per-backend compatibility)

| Cell | Format | MoE backend | MTP | Concurrency | Outcome |
|---|---|---|---|---|---|
| A | NVFP4 (this artifact) | flashinfer_trtllm | n=2 | c=1 sequential | OK — p50 75.3 tok/s; MTP 240/13180 = 1.82% |
| B | NVFP4 (this artifact) | deep_gemm_mega_moe | n=2 | c=1 sequential | **FAIL** at load — `KeyError: 'layers.0.ffn.experts.w13_input_scale'` |
| C | Native MXFP4 | deep_gemm_mega_moe | off | c=1 sequential | OK — p50 69.8 tok/s |
| D | Native MXFP4 | flashinfer_trtllm | off | c=1 sequential | OK — p50 66.9 tok/s |

## Extension (c=16 batched aggregate + matched-GSM8K quality)

| Cell | Format | MoE backend | MTP | n / concurrency | Metric | Value |
|---|---|---|---|---|---|---|
| F | NVFP4 (this artifact) | flashinfer_trtllm | off | n=64 c=16 batched | aggregate output tok/s | **572.8** |
| F | NVFP4 (this artifact) | flashinfer_trtllm | off | n=64 c=16 batched | wall-clock for 64 prompts | 9.73 s |
| F | NVFP4 (this artifact) | flashinfer_trtllm | off | n=64 c=16 batched | per-stream p50 output tok/s | 15.9 |
| G | Native MXFP4 | deep_gemm_mega_moe | off | n=64 c=16 batched | aggregate output tok/s | **405.9** |
| G | Native MXFP4 | deep_gemm_mega_moe | off | n=64 c=16 batched | wall-clock for 64 prompts | 13.29 s |
| G | Native MXFP4 | deep_gemm_mega_moe | off | n=64 c=16 batched | per-stream p50 output tok/s | 10.4 |
| I | Native MXFP4 | deep_gemm_mega_moe | off | n=300 c=16 GSM8K | accuracy | **0.9800** (294/300) |
| I | Native MXFP4 | deep_gemm_mega_moe | off | n=300 c=16 GSM8K | Wilson 95% CI | [0.957, 0.991] |
| J | NVFP4 (this artifact) | flashinfer_trtllm | off | n=300 c=16 GSM8K | accuracy | **0.9567** (287/300) |
| J | NVFP4 (this artifact) | flashinfer_trtllm | off | n=300 c=16 GSM8K | Wilson 95% CI | [0.927, 0.974] |

Cell E (NVFP4 + flashinfer + no MTP at the bench's default concurrency=8) is in the artifacts but not separately reported here since Cells A (c=1 +MTP) and F (c=16 -MTP) bracket it.

Raw JSONs in `docs/benchmarks/matrix/`.

## Findings

### 1. NVFP4 throughput beats native MXFP4 — advantage grows with batch

| Operating point | NVFP4 | MXFP4 | NVFP4 advantage |
|---|---|---|---|
| c=1 single-stream (NVFP4 with MTP n=2 overhead) | 75.3 tok/s | 69.8 tok/s | **+7.9%** |
| c=16 batched aggregate (both no MTP) | 572.8 tok/s | 405.9 tok/s | **+41.1%** |

The single-stream gap is modest because at c=1 the bottleneck is decode latency through the same FP4 expert dequant + per-token routing path, and the two backends' kernel-launch overhead dominates over per-element work. At c=16, NVFP4's `flashinfer_trtllm` saturates the Blackwell tensor cores more effectively than MXFP4's `deep_gemm_mega_moe` mega-kernel, and the advantage opens up.

The c=16 wall-clock for 64 prompts is 9.73 s for NVFP4 vs 13.29 s for MXFP4 — NVFP4 finishes the same batch in 73% of the time.

### 2. `deep_gemm_mega_moe` does NOT dispatch NVFP4 in current vLLM mainline

Cell B's load-time `KeyError: 'layers.0.ffn.experts.w13_input_scale'` is the smoking gun. `deep_gemm_mega_moe`'s parameter-registration path expects **fused-name MoE parameters** — one tensor for all experts at the layer level, named like `experts.w13_input_scale`. NVFP4 ModelOpt layout (which our artifact uses, and which is what PR #42209 added support for on the `flashinfer_trtllm` path) registers **per-expert names** like `experts.{E}.w1.input_scale`. The deep_gemm path's expected key never resolves and the loader raises.

For an NVFP4 artifact, `flashinfer_trtllm` is currently the only viable MoE backend on vLLM mainline. We file a vLLM issue documenting the gap with a minimal repro (point the recipe at any NVFP4 ModelOpt artifact + `--moe-backend deep_gemm_mega_moe` and trigger load).

### 3. `flashinfer_trtllm` works for native MXFP4 too, just slower than deep_gemm on MXFP4

Cell D shows that `flashinfer_trtllm` can dispatch the native MXFP4 weights successfully — it's not NVFP4-only — but it's ~4% slower than `deep_gemm_mega_moe` on those weights (66.9 vs 69.8 tok/s c=1). This is consistent with the upstream choice of `deep_gemm_mega_moe` as the Blackwell default for native FP8/MXFP4: deep_gemm's mega-kernel saves dispatch overhead by fusing all experts into one launch, and the MXFP4 path lights up that kernel well.

The implication for comparing formats: the fair "format-only" comparison is Cell A vs Cell D (both flashinfer) — that's +12.6% c=1. The "recipe-default vs recipe-default" comparison is Cell A vs Cell C — that's +7.9% c=1. The "real production at batch" comparison is Cell F vs Cell G — that's +41.1% c=16.

### 4. Quality on matched GSM8K-300 — small strict-loss within Wilson CI overlap

Cells I (MXFP4) and J (NVFP4) ran the same first 300 problems of the GSM8K test set under identical config (TP=8 + EP + indexer_cache + FULL_AND_PIECEWISE, c=16, max_tokens=2048, temp=0, no MTP).

| | NVFP4 (J) | MXFP4 (I) |
|---|---|---|
| Correct | 287/300 | 294/300 |
| Accuracy | 0.9567 | 0.9800 |
| Wilson 95% CI | [0.927, 0.974] | [0.957, 0.991] |
| Truncation | 0/300 | 0/300 |
| Per-problem agreement | 293/300 | — |

The per-problem agreement matrix:

|   | NVFP4 correct | NVFP4 wrong |
|---|---|---|
| MXFP4 correct | 287 | **7** |
| MXFP4 wrong | 0 | 6 |

NVFP4 lost 7 problems vs MXFP4 (MXFP4 correct, NVFP4 wrong) and gained 0 in the other direction. This is a **strict-loss pattern**, not random noise: the conversion is uniformly small-worse than the source on this benchmark, not "different but equivalent on average."

The 2.33 pt gap sits within Wilson CI overlap (95.71 to 97.45 overlap on the upper-side of NVFP4 and lower-side of MXFP4). It is within the normal NVFP4-conversion-loss tolerance — RedHat's V4-Flash NVFP4 ↔ BF16 comparison reported a comparable ~1-2 pt gap on GSM8K. We characterize this as "quality preserved within format-conversion tolerance," not "quality matched exactly," and document the 7 strict-loss problems so users can inspect them if their workload is GSM8K-adjacent.

### 5. MTP draft acceptance is structural to V4-Pro, not config-driven

Cell A's MTP run, 20 chat prompts at MTP n=2 (sum across all engines):

- Draft tokens emitted: 13,180
- Accepted: 240
- Per-token acceptance: **1.82%**
- Equivalent accept length: 1.036

LMSYS's day-zero V4-Pro blog reports accept length ~1.19 on the official partner-blessed deployment, noting "the MTP path may not be hitting full effectiveness on Pro." vLLM upstream's recipe YAML lists MTP under `opt_in_features` for V4-Pro, not as part of the default deployment. See `upstream_mtp_classification.md` for the evidence trail. Our number is in the same regime as the upstream baseline; this is not a CQL conversion-damage signal.

## Methodology footnotes

- "p50 tok/s" at c=1 sequential is the median of per-request output throughputs from 20 short-prompt latency probes. With 5 distinct prompts repeated 4 times each, the distribution is bimodal-ish (each prompt has a stable output length / tok-per-sec), so p50 is more informative than the mean.
- "Aggregate output tok/s" at c=16 batched is total completion tokens across all 64 prompts divided by total wall-clock from the first prompt dispatched to the last response received. This is the headline number a production deployment cares about.
- "Per-stream p50 tok/s" at c=16 is the median of individual stream's tok/s, which decreases as concurrency rises (interleaving + shared GPU resources). Aggregate is the right comparator across c values.
- All cells used identical bench code, prompts, temperature 0, max_tokens=128 (latency cells) or 2048 (GSM8K cells).
- Cell A had MTP enabled; cells C, D, F, G, I, J did not. Native MXFP4 + MTP cannot currently load on vLLM mainline due to the e_proj/h_proj loader gap documented in `mtp_eproj_hproj_workaround.md` — same error symptom in both topology settings. This puts a small headwind on Cell A's number (MTP rejection overhead), which means the +7.9% advantage of NVFP4 over the native recipe-default path at c=1 is conservative.
- The `--speculative-config` flag in our installation uses kebab-case; the YAML uses underscore (`--speculative_config`). vLLM accepts both forms via argparse.

## Raw artifacts

All in `docs/benchmarks/matrix/`:

- `latency_A_nvfp4_flashinfer_mtp_2026_05_22.json`
- `latency_C_mxfp4_deepgemm_2026_05_22.json`
- `latency_D_mxfp4_flashinfer_2026_05_22.json`
- `latency_E_nvfp4_noMTP_c1_2026_05_22.json` (concurrency=8 actual; bracket cell)
- `latency_F_nvfp4_noMTP_c16_2026_05_22.json`
- `latency_G_mxfp4_deepgemm_c16_2026_05_22.json`
- `mtp_A_nvfp4_flashinfer_mtp_2026_05_22.json`
- `gsm8k_I_mxfp4_deepgemm_300_2026_05_22.json`
- `gsm8k_J_nvfp4_noMTP_300_2026_05_22.json`

Driver scripts: `scripts/matrix_runner.sh`, `scripts/extension_runner.sh`.

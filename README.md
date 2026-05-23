# dsv4-pro-nvfp4-fp8-mtp

Source repo for the DeepSeek-V4-Pro NVFP4-FP8 conversion artifact at [`canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP).

A format conversion of `deepseek-ai/DeepSeek-V4-Pro` from its native MXFP4+FP8 mixed-precision checkpoint into NVFP4 (group=16, FP8 E4M3 block scales + FP32 per-tensor `weight_scale_2`) on routed MoE experts, while preserving FP8 block 128×128 on attention and shared experts. The MTP block (`mtp.0.*`) is kept in the saved weights so vLLM can load the artifact with `--speculative-config method=mtp`.

The conversion is byte-level and deterministic; V4-Pro shipped natively as FP4+FP8 with no public BF16 source, so no fresh activation calibration was required.

## Measurements (8× B300 SXM6 AC, TP=8 + EP, upstream-default `single_node_tep` strategy)

All measurements on the same TP=8 + EP topology, `--moe-backend flashinfer_trtllm`, `--attention_config.use_fp4_indexer_cache=True`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`. MTP off unless noted.

### Quality

| Benchmark | This artifact (NVFP4) | V4-Flash NVFP4 predecessor | RedHat V4-Flash NVFP4 |
|---|---|---|---|
| GSM8K strict 8-shot (full n=1319, 0 truncation) | **0.9689** | 0.9181 | 0.910 (self-report) |
| GSM8K matched n=300, NVFP4 vs source MXFP4 | 0.9867 vs 0.9900 (1 strict-loss) | n/a | n/a |
| AIME 2024 thinking=high (n=30, 0 length-truncation) | 0.6667 raw / 0.6897 non-truncated | 0.8333 / 0.9600 | 0.9000 |
| MMLU-Pro 5-shot (full n=12,032) | **0.8164 ± 0.0034** | 0.8113 | not reported |
| HumanEval pass@1 (EvalPlus, greedy) | **0.951** | 0.915 | 0.896 |
| HumanEval+ pass@1 (EvalPlus, greedy) | **0.896** | 0.854 | 0.860 |
| IFEval prompt_level_strict | 0.8484 ± 0.0154 | 0.8540 | 0.8207 |
| MTP draft acceptance (chat workload, N=2) | 1.82% (240/13180 tokens) | n/a | n/a |

GSM8K full set under proper numeric scoring is **96.89%** (the older 94.09% number was a bench string-match artifact — `75.00 vs 75`; see [`docs/findings/gsm8k_scoring_correction.md`](docs/findings/gsm8k_scoring_correction.md)). On the matched-300 subset under identical config NVFP4 loses only 1 strict-loss problem vs the native MXFP4 source. HumanEval is the strongest quality lift (+3.6 / +4.2pt vs V4-Flash NVFP4). MTP acceptance is in the LMSYS-reported V4-Pro regime (their accept length 1.19 on the partner-blessed deployment); upstream classifies V4-Pro MTP as `opt_in_features` not default. The artifact's `mtp.0.{e_proj, h_proj}` BF16 weights are **100% byte-equivalent** to source FP8 dequant — the MTP weakness is not caused by the conversion.

### Throughput

| Operating point | This artifact (NVFP4) | Native MXFP4 + deep_gemm | Δ |
|---|---|---|---|
| c=16 batched aggregate (64 prompts, output tok/s) | **572.8** | 405.9 | **+41.1%** |
| c=64 batched aggregate (peak, 128 prompts) | **1606.3** | not measured at c=64 | — |
| c=128 batched aggregate (256 prompts) | 1151.1 | not measured at c=128 | — |
| c=1 single-stream + MTP n=2 | 75.3 | 69.8 (no MTP) | +7.9% |

NVFP4 lead vs the upstream-recipe-default native MXFP4 path is modest at c=1 (+8%) and widens to **+41% aggregate at c=16**. Throughput peaks around c=64 (1606 tok/s aggregate) and drops past c=64 — **production sweet spot is c=32-64** on this 8-GPU node. Full backend × format matrix + scaling curve in [`docs/findings/backend_format_matrix.md`](docs/findings/backend_format_matrix.md) and [`docs/findings/throughput_scaling.md`](docs/findings/throughput_scaling.md).

Full per-benchmark write-ups in [`docs/findings/`](docs/findings/) and raw JSONs in [`docs/benchmarks/matrix/`](docs/benchmarks/matrix/).

## Quick start

One-line install (builds vLLM mainline + the 4 patches, ~15 min on a fresh DLAMI):

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

Serve (recommended upstream-default config):

```bash
export PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda

vllm serve canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

Add `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` to enable MTP (opt-in; ~1.8% acceptance per upstream-known V4-Pro MTP limits).

Full setup in [`docs/QUICKSTART.md`](docs/QUICKSTART.md). The 4 patches + gotcha catalog in [`docs/VLLM_SETUP_ISSUES.md`](docs/VLLM_SETUP_ISSUES.md).

## Docker portability

The partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image does NOT load this artifact (as of 2026-05-23) — same `KeyError: w13_input_scale` as the deep_gemm path on mainline. The image's vLLM build predates PR #42209's NVFP4 MoE routing merge (2026-05-22). Use our mainline + 4-patches build path. Full repro in [`docs/findings/lambda_docker_portability.md`](docs/findings/lambda_docker_portability.md).

## Recipe summary

| Group | Modules | Source | Target |
|---|---|---|---|
| routed experts | `layers.X.ffn.experts.Y.w{1,2,3}` and `mtp.0.ffn.experts.*` | MXFP4 group=32 + E8M0 | NVFP4 group=16 + E4M3 + per-tensor FP32 S_g + `input_scale=1.0` |
| attention | `wq_a, wq_b, wkv, wo_a, wo_b` and fused variants | FP8 block 128×128 | unchanged |
| shared experts | `shared_experts.w*` | FP8 block 128×128 | unchanged |
| hc/norms/indexer/compressor | various | BF16 / mixed | unchanged |
| **`mtp.0.{e_proj, h_proj}.weight`** | FP8 block 128×128 | **BF16 (dequantized)** — 100% byte-equivalent to source dequant | upstream-loader workaround — see `docs/findings/mtp_eproj_hproj_workaround.md` |
| embeddings / head / hc_head | various | BF16/FP32 | unchanged |

Per-expert NVFP4 sidecars: `<expert>.w{1,2,3}.weight_scale_2` (FP32 [1] global scale, shared between w1/w3 per ModelOpt invariant) and `<expert>.w{1,2,3}.input_scale` (FP32 [1] activation scale, value 1.0).

## Upstream contributions

Patches and issues extracted from this work and filed upstream:

| PR / Issue | Description | Status |
|---|---|---|
| vLLM [#42209](https://github.com/vllm-project/vllm/pull/42209) (sychen52, NVIDIA) | NVFP4 MoE support for DSV4 (ModelOptNvFp4FusedMoE + trtllm_nvfp4_moe + oracle/nvfp4.py) | **MERGED** 2026-05-22 |
| vLLM [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` (compressed_tensors) | open |
| vLLM [#43288](https://github.com/vllm-project/vllm/pull/43288) | `.get("scale_fmt", "ue8m0")` + BF16 `getattr` wrap | open |
| vLLM [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback (attention) | open |
| vLLM [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP loader: candidate-list scale resolution + BF16-on-disk detect | open |
| vLLM issue [#43454](https://github.com/vllm-project/vllm/issues/43454) | `deep_gemm_mega_moe` doesn't dispatch NVFP4 (per-expert vs fused param naming) — `KeyError: 'layers.0.ffn.experts.w13_input_scale'` | open |
| vLLM issue [#43455](https://github.com/vllm-project/vllm/issues/43455) | V4-Pro MTP acceptance 1.82% reproduces LMSYS day-zero ~1.19 accept length; `opt_in_features` classification matches | open (informational) |

The installer script applies all 4 open patches automatically. PR #42209 is mainline; the installer's cherry-pick of those 3 commits becomes a no-op if you base the build on a SHA after the merge.

## Repo layout

```
MODEL_CARD.md                         — HF model card (the HF README)
LICENSE                               — MIT (matches upstream DeepSeek-V4-Pro)
PLAN.md                               — phase-by-phase plan with gates
README.md                             — this file
docs/
  QUICKSTART.md                       — end-to-end serve recipe
  VLLM_SETUP_ISSUES.md                — the 4 patches + setup gotchas
  FINDINGS.md                         — index of findings docs
  benchmarks/                         — per-benchmark JSON outputs
    matrix/                           — backend×format matrix + extension cells outputs
  findings/                           — methodology + diagnostic notes
    backend_format_matrix.md          — NVFP4×MXFP4 × flashinfer×deep_gemm matrix + matched-GSM8K-300
    throughput_scaling.md             — c=1 → c=128 batched concurrency sweep
    aime30_full_2026_05_22.md         — AIME-30 thinking=high full run
    mmlu_pro_2026_05_22.md            — MMLU-Pro 5-shot full 12k
    humaneval_2026_05_22.md           — HumanEval / HumanEval+ via EvalPlus
    ifeval_2026_05_22.md              — IFEval zero-shot
    gsm8k_scoring_correction.md       — bench string-match → numeric-match fix
    e_proj_h_proj_forensic.md         — mtp.0 BF16 byte-equivalence proof
    lambda_docker_portability.md      — partner-blessed docker image does not load this artifact
    mtp_eproj_hproj_workaround.md     — why mtp.0.{e_proj,h_proj} are BF16 in this artifact
    upstream_mtp_classification.md    — vLLM/LMSYS evidence on V4-Pro MTP weakness
    conversion_math_validation.md     — initial byte-level dequant validation
    conversion_v3_validation.md       — 192-tensor sampled validation
  recipes/
    nvfp4_fp8_mtp_replication.md      — full conversion + serve replication recipe
patches/
  patch_43248_ct_bool_wrap.diff
  patch_43288_scale_fmt_get.diff
  patch_43290_weight_scale_fallback.diff
  patch_43319_mtp_quant_detect.diff
  VERSIONS.md
scripts/
  install_vllm_with_patches.sh        — one-line installer (CUDA toolkit + Rust + 4 patches)
  convert_v4_pro_mxfp4_to_nvfp4.py    — GPU-accelerated conversion (~17 min on 1× B300)
  bench_v4_pro.py                     — unified GSM8K / AIME / latency / MTP harness (numeric-match GSM8K scorer)
  matrix_runner.sh                    — backend×format matrix driver
  extension_runner.sh                 — c=16 batched + matched-GSM8K extension driver
vendor/dsv4-pro-upstream/             — vendored upstream model.py / kernel.py / config.json
memory/MEMORY.md                      — pointers to load-bearing facts for future sessions
```

## Predecessor

V4-Flash predecessor at [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) (shipped 2026-05-21). V4-Pro is **not** a straight re-application of that recipe — the format conversion (MXFP4 → NVFP4, since V4-Pro shipped natively as FP4+FP8 rather than as BF16) and the serving path (FlashInfer NVFP4 backend + PR #42209) are V4-Pro-specific. The MTP retention pattern carries over.

## License

MIT, inherited from `deepseek-ai/DeepSeek-V4-Pro`. See [`LICENSE`](LICENSE).

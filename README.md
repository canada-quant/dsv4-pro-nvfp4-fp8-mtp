# dsv4-pro-nvfp4-fp8-mtp

Source repo for the DeepSeek-V4-Pro NVFP4-FP8 conversion artifact at [`canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP).

A format conversion of `deepseek-ai/DeepSeek-V4-Pro` from its native MXFP4+FP8 mixed-precision checkpoint into NVFP4 (group=16, FP8 E4M3 block scales + FP32 per-tensor `weight_scale_2`) on routed MoE experts, while preserving FP8 block 128×128 on attention and shared experts. The MTP block (`mtp.0.*`) is kept in the saved weights so vLLM can load the artifact with `--speculative-config method=mtp`.

The conversion is byte-level and deterministic; V4-Pro shipped natively as FP4+FP8 with no public BF16 source, so no fresh activation calibration was required.

## Measurements (8× B300 SXM6 AC, TP=8 + EP, upstream-default `single_node_tep` strategy)

All measurements on the same TP=8 + EP topology, `--attention_config.use_fp4_indexer_cache=True`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`. NVFP4 dispatches via `flashinfer_trtllm` (the only NVFP4-aware MoE backend in current vLLM mainline); native MXFP4 dispatches via `deep_gemm_mega_moe` (upstream-recipe default for native).

### Throughput

| Operating point | This artifact (NVFP4 + flashinfer) | Native MXFP4 + deep_gemm | Δ |
|---|---|---|---|
| **c=16 batched aggregate (64 prompts, output tok/s)** | **572.8** | 405.9 | **+41.1%** |
| c=16 batched wall-clock (64 prompts) | 9.73 s | 13.29 s | 0.73× |
| c=1 single-stream (p50 output tok/s, +MTP n=2) | 75.3 | 69.8 | +7.9% |

The single-stream c=1 advantage is modest (+8%); the batched c=16 advantage opens up to +41% — NVFP4's tensor-core utilization on Blackwell scales better with batch than MXFP4's mega-kernel path. Full backend × format matrix in [`docs/findings/backend_format_matrix.md`](docs/findings/backend_format_matrix.md).

### Quality (matched GSM8K-300, same config, no MTP, c=16, max_tokens=2048, temp=0)

| | NVFP4 (this artifact) | Native MXFP4 | Δ |
|---|---|---|---|
| Correct | 287/300 | 294/300 | -7 |
| Accuracy | 0.9567 | 0.9800 | -2.33 pt |
| Wilson 95% CI | [0.927, 0.974] | [0.957, 0.991] | overlap [0.957, 0.974] |
| Per-problem agreement | 293/300 agree; **NVFP4 lost 7**, gained 0 | — | strict-loss pattern |
| Truncation | 0 | 0 | — |

The 2.33 pt gap is within Wilson CI overlap and within the normal NVFP4-conversion-loss tolerance (comparable to the V4-Flash NVFP4 ↔ BF16 gap reported by RedHat).

Other measurements on this artifact:

| Benchmark | This artifact |
|---|---|
| GSM8K strict 8-shot (full n=1319) | 0.9409 (1241/1319), 0 truncation |
| AIME 2024 thinking=high (partial n=25/30) | 0.7600 (19/25), 0 truncation on captured set |
| MTP draft acceptance (chat workload, N=2) | 1.82% (240/13180 tokens) |

MTP acceptance is in the same regime as the LMSYS day-zero V4-Pro report (accept length ~1.19), and vLLM upstream itself classifies V4-Pro MTP as an `opt_in_features` entry — not the default deployment. The low rate reflects V4-Pro's trained MTP head, not the conversion. See [`docs/findings/upstream_mtp_classification.md`](docs/findings/upstream_mtp_classification.md).

Full per-benchmark write-ups in [`docs/findings/`](docs/findings/) and raw JSONs in [`docs/benchmarks/`](docs/benchmarks/).

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

## Recipe summary

| Group | Modules | Source | Target |
|---|---|---|---|
| routed experts | `layers.X.ffn.experts.Y.w{1,2,3}` and `mtp.0.ffn.experts.*` | MXFP4 group=32 + E8M0 | NVFP4 group=16 + E4M3 + per-tensor FP32 S_g + `input_scale=1.0` |
| attention | `wq_a, wq_b, wkv, wo_a, wo_b` and fused variants | FP8 block 128×128 | unchanged |
| shared experts | `shared_experts.w*` | FP8 block 128×128 | unchanged |
| hc/norms/indexer/compressor | various | BF16 / mixed | unchanged |
| **`mtp.0.{e_proj, h_proj}.weight`** | FP8 block 128×128 | **BF16 (dequantized)** | upstream-loader workaround — see `docs/findings/mtp_eproj_hproj_workaround.md` |
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
| vLLM issue (to file) | `deep_gemm_mega_moe` does not dispatch NVFP4 expert weights (per-expert vs fused param naming) | pending |

The installer script applies all 4 open patches automatically. PR #42209 is now mainline and no longer requires cherry-pick.

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
    matrix/                           — backend×format matrix outputs (2026-05-22)
  findings/                           — methodology + diagnostic notes
    backend_format_matrix.md          — NVFP4×MXFP4 × flashinfer×deep_gemm matrix
    conversion_math_validation.md     — byte-level dequant validation
    conversion_v3_validation.md       — 192-tensor sampled validation
    mtp_eproj_hproj_workaround.md     — why mtp.0.{e_proj,h_proj} are BF16 in this artifact
    upstream_mtp_classification.md    — vLLM/LMSYS evidence on V4-Pro MTP weakness
    upstream_research_v4_pro_ground_truth.md
    phase_0_closeout.md
    phase_6_quality_summary.md
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
  bench_v4_pro.py                     — unified GSM8K / AIME / latency / MTP harness
  matrix_runner.sh                    — backend×format matrix driver
  extension_runner.sh                 — c=16 batched + matched-GSM8K extension driver
vendor/dsv4-pro-upstream/             — vendored upstream model.py / kernel.py / config.json
memory/MEMORY.md                      — pointers to load-bearing facts for future sessions
```

## Predecessor

V4-Flash predecessor at [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) (shipped 2026-05-21). V4-Pro is **not** a straight re-application of that recipe — the format conversion (MXFP4 → NVFP4, since V4-Pro shipped natively as FP4+FP8 rather than as BF16) and the serving path (FlashInfer NVFP4 backend + PR #42209) are V4-Pro-specific. The MTP retention pattern carries over.

## License

MIT, inherited from `deepseek-ai/DeepSeek-V4-Pro`. See [`LICENSE`](LICENSE).

---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Pro
tags:
  - compressed-tensors
  - nvfp4
  - fp8
  - vllm
  - deepseek
  - mtp
  - speculative-decoding
  - mixture-of-experts
library_name: vllm
---

# canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP

An NVFP4-FP8 conversion of `deepseek-ai/DeepSeek-V4-Pro` that retains the MTP (multi-token-prediction) block in the saved weights, so vLLM can load it with `--speculative-config method=mtp`.

## What this is

- 852 GiB across 64 safetensors shards (vs ~864 GB on-disk native FP4+FP8+BF16 source — close to parity because the format change is a transcoding, not a re-quantization).
- 1,598.84 B total parameters / 49.60 B active per token — verified by summing tensor element counts across all 64 shards.
- Routed FFN experts converted **MXFP4 group=32 → NVFP4 group=16**: per-block E8M0 → E4M3 scales + per-tensor FP32 `weight_scale_2` (shared between `w1`/`w3` per ModelOpt invariant) + per-tensor FP32 `input_scale=1.0` sidecars.
- Attention (`wq_a/wq_b/wkv/wo_a/wo_b` and fused variants), shared experts, indexer, compressor: **FP8 block 128×128, preserved verbatim**.
- MTP block (`mtp.0.*`): NVFP4 experts (same as trunk) + BF16 `e_proj`/`h_proj` (dequantized from FP8 at conversion time as an upstream-loader workaround — see [Recipe](#quantization-recipe) below).
- Hardware target: 8× B300 SXM6 AC (compute_cap 10.3), TP=8 + expert-parallel.

## Headline measurements

All numbers measured 2026-05-22 on 8× B300 SXM6 AC (288 GB HBM3e per GPU, sm_103a) with the upstream-default `single_node_tep` strategy: TP=8, `--enable-expert-parallel`, `--attention_config.use_fp4_indexer_cache=True`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`. vLLM mainline @ `39910f2b25` + 4 local patches (see below).

### Throughput vs native MXFP4 source

Both configs on their preferred MoE backend per upstream guidance — NVFP4 on `flashinfer_trtllm` (the only NVFP4-aware backend in current mainline), native MXFP4 on `deep_gemm_mega_moe` (the upstream-recipe default for native). Same TP=8 + EP topology, same prompts, same temperature 0.

| Operating point | This artifact (NVFP4 + flashinfer) | Native MXFP4 + deep_gemm | Speedup |
|---|---|---|---|
| **c=16 batched (64 prompts, aggregate output tok/s)** | **572.8** | 405.9 | **+41.1%** |
| **c=16 batched (wall-clock for 64 prompts)** | **9.73 s** | 13.29 s | **0.73×** |
| c=1 single-stream (p50 output tok/s, no MTP) | per-stream 16.5 / aggregate 211 | 69.8 (c=1 sequential) | see notes |
| c=1 single-stream (p50, with MTP n=2) | 75.3 | 69.8 (no MTP) | +7.9% |

The single-stream advantage at c=1 is modest (+8% with MTP overhead included). At c=16 batched the advantage opens up to **+41% aggregate throughput** — NVFP4's tensor-core utilization on Blackwell scales better with batch than MXFP4's mega-kernel path. Full backend × format matrix in [`docs/findings/backend_format_matrix.md`](docs/findings/backend_format_matrix.md).

### Quality

Same TP=8 + EP + indexer_cache + FULL_AND_PIECEWISE config on both configs, MTP off on both for the matched comparison, c=16, temperature 0, max_tokens=2048, same first 300 problems of GSM8K test set:

| Benchmark | This artifact (NVFP4) | Native MXFP4 source | Δ |
|---|---|---|---|
| GSM8K matched n=300 | **0.9567** (287/300) | 0.9800 (294/300) | -2.33 pt |
| Wilson 95% CI | [0.927, 0.974] | [0.957, 0.991] | overlap [0.957, 0.974] |
| Per-problem agreement | 293/300 agree; **NVFP4 lost 7**, gained 0 | — | strict-loss pattern |

The 2.33 pt gap is within Wilson CI overlap and within the normal NVFP4-conversion-loss tolerance (RedHat's V4-Flash NVFP4 artifact showed a comparable ~1-2 pt gap vs BF16 on GSM8K). Zero truncation on both sides rules out methodology contamination.

Other measurements on this artifact (single-config, NVFP4):

| Benchmark | This artifact | Notes |
|---|---|---|
| GSM8K strict 8-shot (full n=1319) | 0.9409 (1241/1319) | Earlier full-set run, 0 truncation, max_tokens=2048 |
| AIME 2024 thinking=high (partial n=25/30) | 0.7600 (19/25) | 65K max_tokens, 0 truncation on captured set, Wilson 95% CI [0.56, 0.89]; original bench hit a network error at problem 26 — bench has since been patched |

The AIME 25-problem partial sits within DeepSeek's published V4-Pro AIME range (~80-85%) at a binomial CI lower bound of ~56%.

### MTP draft acceptance

Measured under headline config (NVFP4 + flashinfer + MTP n=2), 20 chat-style prompts, summed across all engine counters:

| Metric | Value |
|---|---|
| Draft tokens emitted | 13,180 |
| Tokens accepted | 240 |
| **Per-token acceptance rate** | **1.82%** |
| Equivalent average accept length (N=2) | 1.036 |

This is in the same regime as the LMSYS day-zero V4-Pro report (accept length ~1.19 on the officially partner-blessed fork-built deployment, with the explicit note "the MTP path may not be hitting full effectiveness on Pro"). vLLM upstream itself classifies V4-Pro MTP as an `opt_in_features` entry in the official recipe YAML — not the default deployment. The low rate reflects V4-Pro's trained MTP head, not the conversion. See [`docs/findings/upstream_mtp_classification.md`](docs/findings/upstream_mtp_classification.md) for the evidence trail.

MTP is retained on disk so users can opt in (rejection-sample overhead is small at this acceptance rate) or benefit automatically if upstream V4-Pro MTP improves. It is not the throughput driver of this artifact.

## Recommended serving config

Single-node 8× B300, upstream-default `single_node_tep` strategy:

```bash
vllm serve canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'

# Add for MTP spec-decode (opt-in; ~1.8% acceptance per above):
#   --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

`flashinfer_trtllm` is the only MoE backend in current vLLM mainline that dispatches NVFP4 expert weights. `deep_gemm_mega_moe` (the default for native MXFP4) raises `KeyError: 'layers.0.ffn.experts.w13_input_scale'` on NVFP4 inputs because the mega-kernel path expects fused-name MoE parameters while NVFP4 ModelOpt layout uses per-expert names. A vLLM issue documenting this gap is filed (links below).

`--attention_config.use_fp4_indexer_cache=True` is the Blackwell-specific override from the upstream recipe and applies to the V4-Pro sparse attention indexer regardless of expert format.

## Quick start

```bash
# 1. Build vLLM with the 4 required patches (~15 min)
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash

# 2. Download the artifact (852 GiB, ~5-10 min with HF Xet + token)
hf auth login
hf download canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP --local-dir /scratch/v4-pro-nvfp4

# 3. Serve (see "Recommended serving config" above)
```

Full setup in [`docs/QUICKSTART.md`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/QUICKSTART.md). The 4 patches + setup gotchas are catalogued in [`docs/VLLM_SETUP_ISSUES.md`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/VLLM_SETUP_ISSUES.md).

## Quantization recipe

This is a **format conversion** (MXFP4 → NVFP4), not a fresh calibration. V4-Pro shipped natively as FP4+FP8 — there is no public BF16 source — so no activation statistics had to be re-collected. The conversion is deterministic, byte-level, on the source tensors.

| Tensor category | Source format | Target format | Action |
|---|---|---|---|
| `layers.X.ffn.experts.Y.w{1,2,3}` (routed) | MXFP4 group=32 + E8M0 block scale | NVFP4 group=16 + E4M3 block scale + FP32 per-tensor S_g + FP32 `input_scale=1.0` | Re-quantize: dequant → regroup → per-tensor `S_g = max_amax / (FP4_max × E4M3_max)` (shared between `w1` and `w3` per ModelOpt invariant; independent for `w2`) → per-block E4M3 → FP4 grid quantize |
| `mtp.0.ffn.experts.Y.w{1,2,3}` | MXFP4 group=32 | NVFP4 group=16 | Re-quantize (same as trunk) |
| `layers.X.attn.{wq_a, wq_b, wkv, wo_a, wo_b}` | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| `layers.X.ffn.shared_experts.w*` | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| `layers.X.{hc_attn_*, hc_ffn_*, attn_norm, ffn_norm}` | BF16 | BF16 | Passthrough |
| `layers.X.attn.{compressor, indexer}.*` | mixed FP8/BF16 | unchanged | Passthrough |
| `mtp.0.{e_proj, h_proj}.weight` | FP8 block 128×128 | **BF16 (dequantized)** | See [`docs/findings/mtp_eproj_hproj_workaround.md`](docs/findings/mtp_eproj_hproj_workaround.md) |
| `mtp.0.attn.*`, `mtp.0.hc_*`, MTP norms | mixed | unchanged | Passthrough |
| `embed.weight`, `head.weight`, `norm.weight`, `hc_head_*` | BF16/FP32 | unchanged | Passthrough |

**Why `mtp.0.e_proj`/`h_proj` are dequantized to BF16**: vLLM mainline's `ReplicatedLinear` + `Fp8Config` path does not currently register the `weight_scale_inv` parameter slot for these two `mtp.0` modules in a way the MTP loader can resolve. Loading the native FP8 versions of `e_proj`/`h_proj` against this loader produces `KeyError: 'model.layers.61.e_proj.weight_scale_inv'` — measured on both TP=8 + EP and DP=8 + EP topologies. Dequantizing to BF16 at conversion time costs ~200 MB extra disk vs FP8 but eliminates the load failure entirely. Documented in detail at [`docs/findings/mtp_eproj_hproj_workaround.md`](docs/findings/mtp_eproj_hproj_workaround.md) and partially addressed by our [vLLM patch #43319](https://github.com/vllm-project/vllm/pull/43319).

The full conversion script is [`scripts/convert_v4_pro_mxfp4_to_nvfp4.py`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/scripts/convert_v4_pro_mxfp4_to_nvfp4.py) (GPU-accelerated, ~17 min for 64 shards on 1× B300). Byte-level dequant validation (correlation 0.997-1.0 vs source on 192 sampled tensors) is in [`docs/findings/conversion_v3_validation.md`](docs/findings/conversion_v3_validation.md).

## vLLM patches required

The artifact loads on vLLM mainline + the 4 open patches below. PR #42209 (the NVFP4 MoE support for DSV4) merged 2026-05-22 and is now in mainline directly. The installer script applies the 4 remaining patches automatically.

| PR | Purpose | Status |
|---|---|---|
| [#42209](https://github.com/vllm-project/vllm/pull/42209) (sychen52, NVIDIA) | NVFP4 MoE support for DSV4 (ModelOptNvFp4FusedMoE + trtllm_nvfp4_moe + oracle/nvfp4.py) | **MERGED** 2026-05-22 |
| [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` (compressed_tensors) | open |
| [#43288](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` defensive `.get()` + BF16 `getattr` wrap | open |
| [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback (attention) | open |
| [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP loader: candidate-list scale resolution + BF16-on-disk detect | open |

Upstream issues filed from this work (no installer-side action; tracking + docs only):

| Issue | Subject | Status |
|---|---|---|
| [#43454](https://github.com/vllm-project/vllm/issues/43454) | `deep_gemm_mega_moe` doesn't dispatch NVFP4 (per-expert vs fused param naming) — `KeyError: 'layers.0.ffn.experts.w13_input_scale'` | open |
| [#43455](https://github.com/vllm-project/vllm/issues/43455) | V4-Pro MTP acceptance 1.82% on vLLM mainline reproduces LMSYS day-zero ~1.19 accept length — `opt_in_features` classification correctly reflects current MTP head capability | open (informational) |

If any merge after this writing, the installer script's patch list should shrink to match.

## Differences vs `RedHatAI/DeepSeek-V4-Pro-NVFP4-FP8`

As of 2026-05-22, RedHat has not shipped a V4-Pro NVFP4 artifact. If one ships later, the structural difference will mirror the V4-Flash predecessor: this artifact retains the MTP block, RedHat's would strip it via the HF transformers default `_keys_to_ignore_on_load_unexpected`.

## Files in the artifact

- 64 sharded `model-*.safetensors` files + `model.safetensors.index.json` (852 GiB total)
- `config.json` — vLLM-compatible quantization_config with NVFP4 routing for experts + FP8 block for attention/shared
- `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` — upstream V4-Pro
- `chat_template.jinja` — upstream V4-Pro three-tier reasoning template
- `README.md` — this file (the HF render of `MODEL_CARD.md`)

## Reproduction

The end-to-end conversion + serve recipe is in [`docs/recipes/nvfp4_fp8_mtp_replication.md`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/recipes/nvfp4_fp8_mtp_replication.md). Hardware: 1× B300 (288 GB HBM3e) for conversion, 8× B300 for serving.

## Predecessor

V4-Flash predecessor: [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) (shipped 2026-05-21). V4-Pro is **not** a straight re-application of that recipe — the format conversion (MXFP4 → NVFP4, since V4-Pro shipped natively as FP4+FP8) and the serving path (FlashInfer NVFP4 backend + PR #42209) are V4-Pro-specific. The MTP retention pattern carries over.

## Citation

```bibtex
@misc{canada-quant-dsv4-pro-nvfp4-fp8-mtp-2026,
  title  = {DeepSeek-V4-Pro NVFP4-FP8 with MTP preserved for vLLM speculative decoding},
  author = {Canada Quant},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP}
}
```

## License

MIT, inherited from `deepseek-ai/DeepSeek-V4-Pro`.

## Acknowledgments

- DeepSeek for V4-Pro and the MTP architecture.
- The V4-Flash predecessor recipe that established the MTP-retention pattern.
- vLLM, llm-compressor, compressed-tensors, and FlashInfer maintainers.
- PR #42209 contributors (sychen52, xinli-sw, pavanimajety, zyongye) for the DSV4 NVFP4 MoE kernel work that made serving this artifact possible.

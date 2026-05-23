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
- MTP block (`mtp.0.*`): NVFP4 experts (same as trunk) + BF16 `e_proj`/`h_proj` (dequantized from FP8 at conversion time, **verified 100% byte-equivalent** to on-the-fly source FP8 dequant — see [forensic doc](docs/findings/e_proj_h_proj_forensic.md)). A v0.3 experiment with the entire `mtp.0.*` block re-dequanted to BF16 was tried (see [bisection doc](docs/findings/mtp_v03_bf16_block_bisection.md)) and did **not** recover MTP acceptance; v0.2 remains the shipped artifact.
- Hardware target: 8× B300 SXM6 AC (compute_cap 10.3), TP=8 + expert-parallel.

## Headline measurements

All numbers measured 2026-05-22/23 on 8× B300 SXM6 AC (288 GB HBM3e per GPU, sm_103a) with the upstream-default `single_node_tep` strategy: TP=8, `--enable-expert-parallel`, `--moe-backend flashinfer_trtllm`, `--attention_config.use_fp4_indexer_cache=True`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`. vLLM mainline @ `39910f2b25` + 5 local patches (see below).

### Quality

All numbers MTP-off, greedy / temperature 0, chat template applied:

| Benchmark | This artifact (NVFP4) | V4-Flash NVFP4 predecessor (no spec) | RedHat V4-Flash NVFP4 (no MTP) |
|---|---|---|---|
| GSM8K strict 8-shot (full n=1319) | **0.9689** (1278/1319, 0 truncation) | 0.9181 | 0.910 (self-report) |
| GSM8K matched n=300 vs source MXFP4 | **0.9867** (NVFP4 296/300) vs **0.9900** (MXFP4 297/300) | n/a | n/a |
| AIME 2024 thinking=high (full n=30) | 0.6667 raw / 0.6897 non-truncated | 0.8333 raw / 0.9600 non-trunc | 0.9000 raw |
| MMLU-Pro 5-shot (full n=12,032) | **0.8164 ± 0.0034** | 0.8113 | not reported |
| HumanEval pass@1 (EvalPlus, greedy) | **0.951** | 0.915 | 0.896 |
| HumanEval+ pass@1 (EvalPlus, greedy) | **0.896** | 0.854 | 0.860 |
| IFEval prompt_level_strict | 0.8484 ± 0.0154 | 0.8540 | 0.8207 |
| IFEval prompt_level_loose | 0.8780 ± 0.0141 | 0.8928 | 0.8466 |
| IFEval inst_level_strict | 0.8945 | 0.9005 | 0.8765 |
| IFEval inst_level_loose | 0.9149 | 0.9293 | 0.8945 |

**Quality summary**: Strong on GSM8K (96.89% full, 98.67% matched-300 vs native source 99.00% — only 1 strict-loss problem out of 300 on identical config), MMLU-Pro (81.64%, +0.5pt vs V4-Flash), HumanEval (95.1% / 89.6%, +3.6 / +4.2pt vs V4-Flash). AIME-30 raw at 66.67% — within Wilson CI [0.49, 0.81] at n=30 which contains DeepSeek's reported V4-Pro range (~80-85%) at the upper bound; non-truncated rate at 69%. IFEval slightly below V4-Flash (-0.6 to -1.5pt) but ahead of RedHat V4-Flash NVFP4 (+2.8 to +3.1pt on the like-comparison rows).

Per-benchmark methodology and per-subject / per-problem detail in `docs/findings/`. The historical "94.09%" GSM8K number in earlier drafts was a bench-scorer string-match artifact ("75.00" vs gold "75"); the numeric-match rescoring restores the +3pt difference uniformly. See [`docs/findings/gsm8k_scoring_correction.md`](docs/findings/gsm8k_scoring_correction.md).

### Throughput

Same config, MTP off unless noted. Single-stream c=1 numbers reflect sequential per-stream throughput; multi-stream aggregate is the production-relevant number.

| Operating point | This artifact (NVFP4 + flashinfer) | Native MXFP4 + deep_gemm | Δ |
|---|---|---|---|
| **c=16 batched aggregate (64 prompts, output tok/s)** | **572.8** | 405.9 | **+41.1%** |
| **c=64 batched aggregate (128 prompts)** | **1606.3 (peak)** | not measured at c=64 | — |
| **c=128 batched aggregate (256 prompts)** | 1151.1 | not measured at c=128 | — |
| c=1 single-stream (p50, with MTP n=2 overhead) | 75.3 | 69.8 (no MTP) | +7.9% |

**Throughput pattern**: NVFP4 lead vs native MXFP4 is modest at c=1 (+8%) and widens to **+41% aggregate at c=16**. Beyond c=64 aggregate throughput plateaus then declines (compute/memory contention dominates over parallelism gains on this 8-GPU node). **Production sweet spot is c=32-64** for this artifact. Full backend × format matrix + concurrency scaling in [`docs/findings/backend_format_matrix.md`](docs/findings/backend_format_matrix.md) and [`docs/findings/throughput_scaling.md`](docs/findings/throughput_scaling.md).

### MTP draft acceptance

Measured under headline config + `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`, 20 chat-style prompts, summed across all engine counters:

| Metric | Value |
|---|---|
| Draft tokens emitted | 13,180 |
| Tokens accepted | 240 |
| **Per-token acceptance rate** | **1.82%** |
| Equivalent average accept length (N=2) | 1.036 |

**Update 2026-05-23**: The MTP acceptance gap was investigated in two stages and the framing has been corrected.

**Stage 1** — measured the *same V4-Pro MTP head* on the native MXFP4 checkpoint via the partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image (zyongye fork): **91.14% at n=1, 80.56% at n=2** on the same 20-prompt workload. The MTP head itself is healthy — the earlier "MTP is structurally weak on V4-Pro" framing in this card was retracted (see vLLM issue [#43455](https://github.com/vllm-project/vllm/issues/43455)).

**Stage 2** — bisected whether the gap was from the NVFP4 quantization of `mtp.0.ffn.experts.*`. Built a v0.3 candidate artifact (NOT shipped) with the **entire `mtp.0.*` block as BF16** (matches V4-Flash's predecessor recipe exactly; +98 GB on disk vs v0.2). MTP acceptance moved 3.07% → **3.33%** — within noise. The mtp.0 quant choice is *not* the dominant cause. v0.3 stays as a documented research artifact; v0.2 remains the shipped product.

The remaining suspect after the bisection is the **trunk's NVFP4 quantization perturbing the activation distribution that flows into the MTP head**. The V4-Pro MTP head was trained against the native MXFP4 trunk's outputs; our trunk is NVFP4 (group=16 + E4M3 + per-tensor FP32) which gives subtly different quant noise per layer. After 61 trunk layers the activation distribution diverges enough that the head's drafts no longer align with the trunk's verifier, and drafts get rejected. Recovering MTP would require **re-training the MTP head against the NVFP4-trunk activation distribution** — which needs a BF16 V4-Pro source to do cleanly, and no such source is publicly available.

The artifact now keeps `mtp.0.*` BF16 on disk:
- 100% byte-equivalent to source FP8/MXFP4 dequant (verified per-tensor — see [`docs/findings/e_proj_h_proj_forensic.md`](docs/findings/e_proj_h_proj_forensic.md))
- Forward-compatible if upstream V4-Pro MTP recalibration or mainline-vLLM MTP-forward fixes ship later — recovery happens at serve time without re-converting
- Costs +98 GB vs v0.2's hybrid layout

Full bisection writeup at [`docs/findings/mtp_v03_bf16_block_bisection.md`](docs/findings/mtp_v03_bf16_block_bisection.md). 2×2 matrix at [`docs/findings/mtp_native_vs_ours_2026_05_23.md`](docs/findings/mtp_native_vs_ours_2026_05_23.md).

**Production recommendation**: serve this artifact **without `--speculative-config`** until upstream resolves V4-Pro MTP for NVFP4-trunk artifacts. Trunk quality is unchanged from v0.2 (GSM8K 96.89%, MMLU-Pro 81.64%, HumanEval 95.1% / 89.6% — all unaffected by the mtp.0 block change since MTP is not used at decode time). Batched throughput stays at the v0.2 figures.

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

# MTP spec-decode is currently NOT recommended on this artifact (~3% acceptance
# due to the trunk-NVFP4 ↔ MTP-head distribution mismatch documented above).
# When upstream V4-Pro MTP recalibration ships or mainline-vLLM MTP forward is
# fixed for NVFP4-trunk artifacts, the line below becomes worth enabling.
# Until then, leave MTP off and serve only the trunk (full +41% c=16 advantage
# vs native MXFP4 still holds without MTP).
#   --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

`flashinfer_trtllm` is the only MoE backend in current vLLM mainline that dispatches NVFP4 expert weights. `deep_gemm_mega_moe` (the default for native MXFP4) raises `KeyError: 'layers.0.ffn.experts.w13_input_scale'` on NVFP4 inputs because the mega-kernel path expects fused-name MoE parameters while NVFP4 ModelOpt layout uses per-expert names. A vLLM issue ([#43454](https://github.com/vllm-project/vllm/issues/43454)) documents the gap.

`--attention_config.use_fp4_indexer_cache=True` is the Blackwell-specific override from the upstream recipe and applies to the V4-Pro sparse attention indexer regardless of expert format.

## Quick start

```bash
# 1. Build vLLM with the 5 required patches (~15 min)
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
| `mtp.0.ffn.experts.Y.w{1,2,3}` | MXFP4 group=32 | NVFP4 group=16 | Re-quantize (same as trunk). v0.3 BF16 alternative was tested and falsified — see [`docs/findings/mtp_v03_bf16_block_bisection.md`](docs/findings/mtp_v03_bf16_block_bisection.md) |
| `layers.X.attn.{wq_a, wq_b, wkv, wo_a, wo_b}` | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| `layers.X.ffn.shared_experts.w*` | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| `layers.X.{hc_attn_*, hc_ffn_*, attn_norm, ffn_norm}` | BF16 | BF16 | Passthrough |
| `layers.X.attn.{compressor, indexer}.*` | mixed FP8/BF16 | unchanged | Passthrough |
| `mtp.0.{e_proj, h_proj}.weight` | FP8 block 128×128 | **BF16 (dequantized)** | See note below |
| `mtp.0.attn.*`, `mtp.0.hc_*`, MTP norms | mixed | unchanged | Passthrough |
| `embed.weight`, `head.weight`, `norm.weight`, `hc_head_*` | BF16/FP32 | unchanged | Passthrough |

**Why `mtp.0.e_proj`/`h_proj` are dequantized to BF16**: vLLM mainline's `ReplicatedLinear` + `Fp8Config` path does not currently register the `weight_scale_inv` parameter slot for these two `mtp.0` modules in a way the MTP loader can resolve. Loading the native FP8 versions of `e_proj`/`h_proj` against this loader produces `KeyError: 'model.layers.61.e_proj.weight_scale_inv'` — measured on both TP=8 + EP and DP=8 + EP topologies. Dequantizing to BF16 at conversion time costs ~200 MB extra disk vs FP8 but eliminates the load failure entirely. **Forensic confirms the BF16 stored is 100% byte-equivalent (51M elements per tensor) to the source FP8 dequant** — see [`docs/findings/e_proj_h_proj_forensic.md`](docs/findings/e_proj_h_proj_forensic.md). Documented in detail at [`docs/findings/mtp_eproj_hproj_workaround.md`](docs/findings/mtp_eproj_hproj_workaround.md) and partially addressed by our [vLLM patch #43319](https://github.com/vllm-project/vllm/pull/43319).

The full conversion script is [`scripts/convert_v4_pro_mxfp4_to_nvfp4.py`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/scripts/convert_v4_pro_mxfp4_to_nvfp4.py) (GPU-accelerated, ~17 min for 64 shards on 1× B300). Byte-level dequant validation (correlation 0.997-1.0 vs source on 192 sampled tensors) is in [`docs/findings/conversion_v3_validation.md`](docs/findings/conversion_v3_validation.md).

## vLLM patches required

The artifact loads on vLLM mainline + the 5 open patches below. PR #42209 (the NVFP4 MoE support for DSV4) merged 2026-05-22 and is now in mainline directly. The installer script applies the 5 remaining patches automatically.

| PR | Purpose | Status |
|---|---|---|
| [#42209](https://github.com/vllm-project/vllm/pull/42209) (sychen52, NVIDIA) | NVFP4 MoE support for DSV4 (ModelOptNvFp4FusedMoE + trtllm_nvfp4_moe + oracle/nvfp4.py) | **MERGED** 2026-05-22 |
| [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` (compressed_tensors) | open |
| [#43288](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` defensive `.get()` + BF16 `getattr` wrap | open |
| [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback (attention) | open |
| [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP loader: candidate-list scale resolution + BF16-on-disk detect | open |
| [#43467](https://github.com/vllm-project/vllm/pull/43467) | DSV4 MegaMoE early-fail for NVFP4 (deep_gemm + NVFP4 incompatible) | open |
| **v0.3 BF16-MTP dispatch** (`patches/patch_v0p3_dsv4_mtp_bf16_dispatch.diff`) | DSV4 MoE: detect MTP layer (idx >= num_hidden_layers) and route as unquantized + override moe_backend to triton when on-disk MTP block is BF16. Required to load v0.3+ artifacts. Bundled with #43467's deep_gemm+NVFP4 guard. | local patch; upstream PR pending |

Upstream issues filed from this work (no installer-side action; tracking + docs only):

| Issue | Subject | Status |
|---|---|---|
| [#43454](https://github.com/vllm-project/vllm/issues/43454) | `deep_gemm_mega_moe` doesn't dispatch NVFP4 (per-expert vs fused param naming) — `KeyError: 'layers.0.ffn.experts.w13_input_scale'` | open |
| [#43455](https://github.com/vllm-project/vllm/issues/43455) | V4-Pro MTP acceptance ~1.82% on our mainline-patched build vs ~80% on the native checkpoint + fork docker image (corrected — original `opt_in_features` framing was wrong) | open (investigating) |

If any merge after this writing, the installer script's patch list should shrink to match.

## Docker portability

The partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image **does NOT load this artifact** as of 2026-05-23. The image's vLLM build (`v0.1.dev15833+g62d441ee8`, ~April 24) predates PR #42209's NVFP4 MoE routing merge (2026-05-22), so the loader has no place to write the per-expert NVFP4 scales and raises `KeyError: 'layers.0.ffn.experts.w13_input_scale'` — same failure mode as the deep_gemm path on mainline. Use the mainline + 4-patches recipe from [`docs/QUICKSTART.md`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/QUICKSTART.md). Once `vllm/vllm-openai:deepseekv4-cu130` (or any successor tag) is rebuilt from a vLLM mainline that includes PR #42209, the docker path should also work. Full repro: [`docs/findings/lambda_docker_portability.md`](docs/findings/lambda_docker_portability.md).

## Differences vs `RedHatAI/DeepSeek-V4-Pro-NVFP4-FP8`

As of 2026-05-23, RedHat has not shipped a V4-Pro NVFP4 artifact. If one ships later, the structural difference will mirror the V4-Flash predecessor: this artifact retains the MTP block, RedHat's would strip it via the HF transformers default `_keys_to_ignore_on_load_unexpected`.

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

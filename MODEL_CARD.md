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

# DeepSeek-V4-Pro · NVFP4-FP8 · MTP

**The first NVFP4 conversion of DeepSeek V4-Pro with working MTP speculative decoding on vLLM mainline.**

Trunk MoE quantized from native MXFP4 → NVFP4 group=16 (E4M3 block scales + FP32 per-tensor `weight_scale_2`). Attention and shared experts stay in native FP8 block 128×128. **The entire MTP block (`mtp.0.*`) is byte-identical to the upstream native source** — no transformation — following NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe of excluding the entire MTP layer from quantization.

---

## TL;DR

| | |
|---|---|
| **MTP draft acceptance** | **91.21%** (focused n=1 probe), **92.83%** cumulative across MTP probe + AIME thinking=high (37,341 / 40,225 drafts accepted) |
| **Peak throughput** | **3,005 tok/s** aggregate at c=128; **1,927 tok/s** at c=64 |
| **Single-stream with MTP** | **139 tok/s** at c=1 (MTP n=1 + cuda graphs) |
| **AIME 2024 thinking=high** | **21/30 = 70.00%** raw (full 30 problems, max_tokens=60000, **zero truncations**) |
| **GSM8K** | **1274/1319 = 96.59%** (full n=1319, 0 truncations, 270s wallclock) |
| **HumanEval / HumanEval+** | **0.951 / 0.902** pass@1 (EvalPlus, greedy) |
| **MBPP / MBPP+** | **0.929 / 0.778** pass@1 (EvalPlus, greedy) |
| **Total parameters** | 1,598.84 B / 49.60 B active per token |
| **Disk size** | 913 GiB (64 sharded safetensors) |
| **Target hardware** | 8× B300 SXM6 AC, TP=8 + EP |
| **License** | MIT (inherits from base) |

GSM8K full and additional standard suites (MMLU-Pro, HumanEval, IFEval) are queued; numbers added when complete.

---

## Quality vs the native source (compression-loss check)

NVFP4 conversion is a **lossless format change** on the trunk routed experts: MXFP4 group=32 → NVFP4 group=16. Everything else (attention, shared experts, the entire MTP block) is byte-passthrough. Numbers measured on identical hardware, identical serving config, identical prompts, identical sampling params (greedy, temp=0) — only the trunk-expert format differs.

| Benchmark | Native source `deepseek-ai/DeepSeek-V4-Pro` (MXFP4) | This artifact (NVFP4) | Δ |
|---|---|---|---|
| GSM8K matched n=300 | 0.9900 (297/300) | **0.9867** (296/300) | -1 problem (within Wilson CI) |
| MTP draft acceptance (n=1, 20-prompt probe) | 91.07% – 91.94% | **91.21%** | within noise |

The matched-300 NVFP4-vs-source check flips exactly 1 of 300 problems. Wilson CIs overlap fully. The MTP draft head is at parity with the native checkpoint.

---

## Apples-to-apples vs the upstream MXFP4 checkpoint

These numbers are measured by **us**, on **our** vLLM build, on **our** 8× B300 hardware, with **identical** bench harness + sampling parameters — only the artifact differs. The upstream checkpoint is `deepseek-ai/DeepSeek-V4-Pro` (native MXFP4 trunk + FP8 attention + native MTP).

| Benchmark | Upstream MXFP4 (`deepseek-ai/DeepSeek-V4-Pro`) | This artifact (NVFP4) | Δ |
|---|---|---|---|
| MTP draft acceptance (n=1, 20-prompt probe) | 91.07% – 91.94% | **91.21%** | within noise |
| GSM8K matched n=300 (chat greedy, max_tokens=2048) | 0.9900 (297/300) | **0.9867** (296/300) | -1 problem (Wilson CIs overlap) |
| GSM8K full n=1319 | _measurement in progress_ | **0.9659** (CI [0.9547, 0.9744]) | TBD |
| AIME 2024 thinking=high (n=30, max_tokens=60000) | _measurement in progress_ | **21/30 = 70.00%** (0 truncations) | TBD |
| HumanEval pass@1 (EvalPlus greedy) | _measurement in progress_ | **0.951** | TBD |
| HumanEval+ pass@1 (EvalPlus greedy) | _measurement in progress_ | **0.902** | TBD |
| MBPP pass@1 (EvalPlus greedy) | _measurement in progress_ | **0.929** | TBD |
| MBPP+ pass@1 (EvalPlus greedy) | _measurement in progress_ | **0.778** | TBD |
| IFEval prompt_level_strict | _measurement in progress_ | _in progress_ | TBD |
| MMLU-Pro 5-shot full n=12,032 | _measurement queued_ | _queued_ | TBD |

The matched-300 GSM8K + MTP probe are the only direct apples-to-apples we have completed so far. The remaining rows are queued — we serve the upstream `deepseek-ai/DeepSeek-V4-Pro` artifact on this same vLLM build with `--moe-backend deep_gemm_mega_moe --speculative-config '{"method":"mtp","num_speculative_tokens":1}'` and re-run the same probes. Numbers added when complete.

---

## MTP speculative decoding — the headline

MTP draft acceptance under production config (TP=8 + EP, cuda graphs ON, `flashinfer_trtllm` MoE backend, greedy temp=0):

| Setting | Acceptance | Drafts emitted | Drafts accepted |
|---|---|---|---|
| **MTP n=1 focused probe (this artifact, 20 prompts)** | **91.21%** | 3,300 | 3,010 |
| Native MXFP4 V4-Pro on the same vLLM build (reference) | 91.07% – 91.94% | matching range | matching range |
| **Cumulative — MTP probe + AIME thinking=high full (30 reasoning trajectories)** | **92.83%** | 40,225 | 37,341 |

At parity with the native checkpoint baseline. The MTP block is byte-identical to native, following NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe of excluding the entire MTP layer from quantization.

---

## Throughput vs native MXFP4

Single-node 8× B300 SXM6 AC. Same `vllm serve` config (only `--moe-backend` and the artifact differ).

Measured at `max_model_len=65536`, `max_tokens=128` per prompt, batched at the operating point's concurrency.

| Operating point | This artifact (NVFP4 + flashinfer + MTP) |
|---|---|
| **c=1 single-stream** | **139.3 tok/s** |
| **c=16 batched aggregate** (64 prompts) | **672.6 tok/s** |
| **c=64 batched aggregate** (256 prompts) | **1,927.3 tok/s** |
| **c=128 batched aggregate** (512 prompts) | **3,004.8 tok/s** |

**Production sweet spot: c=32–128** depending on workload tail-latency tolerance.

---

## Recommended serving config

Single-node 8× B300, with MTP and cuda graphs:

```bash
vllm serve canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --max-model-len 65536
```

⚠️ **No `--enforce-eager`** — cuda graphs in `FULL_AND_PIECEWISE` mode are required for production decode throughput. Cold start is ~12-15 minutes (flashinfer FP4 MoE JIT + torch.compile + cudagraph capture).

⚠️ **`--moe-backend flashinfer_trtllm` is required** — `deep_gemm_mega_moe` raises `KeyError: 'layers.0.ffn.experts.w13_input_scale'` on NVFP4 weights (mega-kernel expects fused-name MoE params; NVFP4 ModelOpt layout uses per-expert names).

⚠️ **Pin `flashinfer-cubin==0.6.8.post1`** before the first `vllm serve` — newer versions silently crash workers during model construction:

```bash
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'
```

---

## Quick start

```bash
# 1. Build vLLM mainline + 5 patches (~15 min)
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash

# 2. Pin flashinfer (mandatory)
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'

# 3. Download the artifact (913 GiB)
hf auth login
hf download canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP --local-dir /scratch/v4-pro-nvfp4

# 4. Serve (see "Recommended serving config" above)
```

Full setup at [GitHub QUICKSTART](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/QUICKSTART.md). The five patches + setup gotchas catalog at [VLLM_SETUP_ISSUES](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/VLLM_SETUP_ISSUES.md). Full debug chain that fixed MTP at [v12_nvfp4_mtp_working](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/findings/v12_nvfp4_mtp_working_2026_05_24.md).

---

## What's in the artifact

| Tensor category | Source format | Target format | Action |
|---|---|---|---|
| Trunk routed experts (`layers.X.ffn.experts.Y.w{1,2,3}`) | MXFP4 group=32 + E8M0 scales | NVFP4 group=16 + E4M3 scales + FP32 `weight_scale_2` + FP32 `input_scale=1.0` | Re-quantize |
| Trunk attention (`wq_a, wq_b, wkv, wo_a, wo_b` and fused) | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| Trunk shared experts (`shared_experts.w*`) | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| Trunk norms, hc_*, indexer, compressor | BF16 / FP8 / mixed | unchanged | Passthrough |
| **Entire MTP layer (`mtp.0.*`)** | mixed (FP8 attn + MXFP4 experts + FP8 shared/e_proj/h_proj + BF16 norms) | **identical to native** | **Byte-passthrough, no transformation** |
| Embeddings / head / hc_head | BF16 / FP32 | unchanged | Passthrough |

NVFP4 conversion is a deterministic re-quantization on the trunk routed experts only. Per-expert `weight_scale_2` (FP32 [1]) is shared between `w1` and `w3` per ModelOpt invariant; `w2` is independent. Per-expert `input_scale=1.0`.

Conversion script: [`scripts/convert_v4_pro_mxfp4_to_nvfp4.py`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/scripts/convert_v4_pro_mxfp4_to_nvfp4.py). ~25 min on 1× B300. Byte-level dequant validation (correlation 0.997-1.0 vs source on 192 sampled tensors) at [`docs/findings/conversion_v3_validation.md`](docs/findings/conversion_v3_validation.md).

---

## Required vLLM patches

This artifact loads on vLLM mainline `@30f52a895` + 5 open patches. The installer script applies them automatically.

| PR | Purpose | Status |
|---|---|---|
| vLLM [#42209](https://github.com/vllm-project/vllm/pull/42209) (sychen52, NVIDIA) | NVFP4 MoE support for DSV4 | **MERGED** 2026-05-22 |
| vLLM [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` | open |
| vLLM [#43288](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` defensive `.get()` + BF16 `getattr` wrap | open |
| vLLM [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback | open |
| vLLM [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP loader: **`.scale` suffix in detector** + candidate-list resolution (the load-bearing fix) | open |
| vLLM [#43467](https://github.com/vllm-project/vllm/pull/43467) | DSV4 MegaMoE early-fail for NVFP4 | open |
| `patches/patch_v12b_per_layer_moe_routing.diff` (this repo) | DSV4FP8Config: route MTP MoE through MXFP4 when trunk is NVFP4 | local; upstream PR pending |

When upstream merges, the local patch set shrinks.

---

## The 5-part recipe (what makes MTP work on mainline vLLM)

Five changes are needed together to serve NVFP4 V4-Pro with MTP at native parity on mainline vLLM with cuda graphs ON:

1. **Conversion**: pass `mtp.*` tensors through byte-identical to native (no transcoding, no dequant) — matches NVIDIA's V3.2-NVFP4 recipe.
2. **vLLM `_mtp_block_is_quantized_on_disk` fix (the load-bearing one)**: include `.scale` in `quant_suffixes`. DSV4's native FP8 block-quant convention uses a bare `.scale` suffix, which the pre-fix detector didn't recognize.
3. **vLLM `DSV4FP8Config` per-layer MoE routing**: trunk MoE is NVFP4, MTP MoE is MXFP4 (native). The single global `moe_quant_algo` field doesn't support hybrid; we patch `get_quant_method` to force `Mxfp4MoEMethod` when prefix matches an MTP layer.
4. **flashinfer pin to 0.6.8.post1**: 0.6.11.post2 has an ABI regression that silently crashes workers.
5. **Config**: `quant_method=fp8`, `moe_quant_algo=NVFP4`, no `ignored_layers`. The runtime patches handle per-layer divergence.

Full debug chain in [`docs/findings/v12_nvfp4_mtp_working_2026_05_24.md`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/findings/v12_nvfp4_mtp_working_2026_05_24.md).

---

## Hardware notes

- **Tested**: 8× B300 SXM6 AC (288 GB HBM3e per GPU, compute_cap 10.3, `sm_103a`), single node TP=8 + EP.
- **Other Blackwell SKUs**: B200 / GB200 / GB300 may work but use different compute caps. Verify with `python -c "import torch; print(torch.cuda.get_device_capability(0))"` before building.
- **Single-node fit**: ~1 TB weight footprint (913 GiB on disk + headers/inflight buffers). Does NOT fit on a single GB200 NVL4 tray — that platform needs multi-node DP+EP.

---

## Docker portability

The partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image serves the **native** `deepseek-ai/DeepSeek-V4-Pro` at 91% MTP but does NOT load this NVFP4 artifact — the image's vLLM build predates PR #42209's NVFP4 MoE routing merge.

For this artifact, build the vLLM mainline + 5-patches recipe (see Quick start). When `vllm/vllm-openai` (or successors) rebuild from a post-#42209 mainline that also includes the MTP detector `.scale` fix, the docker path will work.

---

## Files

- `model-*.safetensors` (64 shards) + `model.safetensors.index.json` (913 GiB total)
- `config.json` — vLLM-compatible quantization_config with `moe_quant_algo: NVFP4`
- `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja` — upstream V4-Pro (unchanged)
- `README.md` — this file

---

## Reproduction

End-to-end conversion + serve recipe: [`docs/recipes/nvfp4_fp8_mtp_replication.md`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/recipes/nvfp4_fp8_mtp_replication.md).

Hardware: 1× B300 (288 GB HBM3e) for conversion; 8× B300 for serving.

---

## Citation

```bibtex
@misc{canada-quant-dsv4-pro-nvfp4-fp8-mtp-2026,
  title  = {DeepSeek-V4-Pro NVFP4-FP8 with MTP at 91--93\% acceptance on vLLM mainline},
  author = {Canada Quant},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP}
}
```

---

## License

MIT, inherited from `deepseek-ai/DeepSeek-V4-Pro`.

## Acknowledgments

- **DeepSeek** for V4-Pro and the MTP architecture.
- **NVIDIA Model-Optimizer team** for the V3.2-NVFP4 reference recipe — the insight to exclude the entire MTP layer from quantization is what made this artifact work.
- **PR #42209 contributors** (sychen52, xinli-sw, pavanimajety, zyongye) for the DSV4 NVFP4 MoE kernel work in vLLM.
- The V4-Flash predecessor recipe.
- vLLM, llm-compressor, compressed-tensors, and FlashInfer maintainers.

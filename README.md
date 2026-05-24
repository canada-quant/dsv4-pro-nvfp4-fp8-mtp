# dsv4-pro-nvfp4-fp8-mtp

Reproduction repo for [`canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) — the first NVFP4 conversion of DeepSeek V4-Pro with **working MTP speculative decoding** on vLLM mainline.

V4-Pro shipped natively as a mixed FP4+FP8+BF16 checkpoint, so this conversion does not save disk (artifact is 913 GiB vs upstream 864 GiB). **The win is throughput** — on the same vLLM build, same 8× B300 hardware, same bench, MTP n=1 + cuda graphs ON, **NVFP4 runs +25% to +37% faster than upstream MXFP4** at production concurrencies, while preserving MTP at native parity.

## Family / related repos

| Repo | HF model card | Role |
|---|---|---|
| **this repo** (`dsv4-pro-nvfp4-fp8-mtp`) | [Pro NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) | V4-Pro NVFP4 + MTP, B300-only |
| [`canada-quant/dsv4-flash-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp) | [Flash NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) | smaller sibling — V4-Flash same recipe (B300 / RTX PRO 6000) |
| [`canada-quant/dsv4-flash-w4a16-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp) | [Flash W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) | smaller sibling — V4-Flash W4A16 + MTP (Hopper-compatible) |
| [`canada-quant/dsv4-flash-w4a16-fp8`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8) | [Flash W4A16-FP8](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) | smaller sibling — no-MTP baseline |

## Headline — throughput speedup vs upstream MXFP4

Same vLLM build (mainline `30f52a895` + 5 PR patches + 1 local), same 8× B300 SXM6 AC TP=8+EP, same bench harness, MTP n=1 + cuda graphs ON, `max_model_len=65536`, max_tokens=128. Only artifact + recommended MoE backend differ.

| Concurrency | Upstream MXFP4 + `deep_gemm` + MTP | **NVFP4 (this artifact) + `flashinfer` + MTP** | **Speedup** |
|---|---|---|---|
| bs=1 single-stream | 110.8 tok/s | **139.3 tok/s** | **+25.7%** |
| bs=16 batched (64 prompts) | 491.4 tok/s | **672.6 tok/s** | **+36.9%** |
| bs=64 batched (256 prompts) | 1,699.2 tok/s | **1,927.3 tok/s** | **+13.4%** |
| bs=128 batched (512 prompts) | 2,806.7 tok/s | **3,004.8 tok/s** | **+7.1%** |

NVFP4 wins at every concurrency, peaking +37% at bs=16. The gain narrows past bs=64 as both formats saturate the GPUs. **Production sweet spot bs=16–64.**

### MTP speculative decoding

| Metric | Value |
|---|---|
| MTP n=1 acceptance, focused probe (20 prompts) | **91.21%** (3010/3300 drafts) |
| Cumulative, MTP probe + AIME Non-Think greedy (full reasoning) | **92.83%** (37,341/40,225) |
| Reference: native MXFP4 (`deepseek-ai/DeepSeek-V4-Pro`), same vLLM build, same 20-prompt probe | **90.92%** (2905/3195 drafts) |

At parity with the upstream checkpoint baseline.

### Quality (chat greedy temperature 0)

| Benchmark | This artifact (NVFP4) | Notes |
|---|---|---|
| GSM8K full n=1319 | **0.9659** (CI [0.9547, 0.9744]) | chat greedy, max_tokens=2048, 0 truncations |
| GSM8K matched n=300 vs upstream MXFP4 | **0.9867** (296/300) vs upstream 0.9900 (297/300) | 1 strict-loss problem, Wilson CIs overlap |
| AIME 2024 Non-Think greedy (n=30) | **21/30 = 70.00%** | max_tokens=60000, 0 truncations |
| HumanEval pass@1 | **0.951** | EvalPlus greedy |
| HumanEval+ pass@1 | **0.902** | EvalPlus greedy |
| MBPP pass@1 | **0.929** | EvalPlus greedy |
| MBPP+ pass@1 | **0.778** | EvalPlus greedy |
| IFEval prompt_level_strict | _chat-eval rerun queued_ | initial completions-mode pass measured 0.244 — not a fair number (V4-Pro Instruct needs chat template) |
| MMLU-Pro 5-shot full n=12,032 | _queued_ | lm-eval-harness via OpenAI-completions backend |

Apples-to-apples upstream-checkpoint comparison rows (running the same harness on `deepseek-ai/DeepSeek-V4-Pro` via this same vLLM build) are queued; numbers added as they land.

## Quick start

```bash
# 1. Build vLLM mainline + 5 patches (~15 min on a fresh DLAMI)
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash

# 2. Pin flashinfer to 0.6.8.post1 (0.6.11 has ABI regression that silently crashes workers)
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'

# 3. Download the artifact (913 GiB)
hf auth login
hf download canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP --local-dir /scratch/v4-pro-nvfp4

# 4. Serve with MTP + cuda graphs
vllm serve /scratch/v4-pro-nvfp4 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --max-model-len 65536
```

No `--enforce-eager`. Cold start (flashinfer FP4 MoE JIT + torch.compile + cudagraph capture) is ~12-15 minutes. Steady-state MTP: ~91% acceptance.

Full setup in [`docs/QUICKSTART.md`](docs/QUICKSTART.md). 5 patches + setup gotchas catalog in [`docs/VLLM_SETUP_ISSUES.md`](docs/VLLM_SETUP_ISSUES.md). Full debug chain that fixed MTP: [`docs/findings/v12_nvfp4_mtp_working_2026_05_24.md`](docs/findings/v12_nvfp4_mtp_working_2026_05_24.md).

## Conversion recipe summary

| Group | Modules | Source | Target |
|---|---|---|---|
| Trunk routed experts | `layers.X.ffn.experts.Y.w{1,2,3}` | MXFP4 group=32 + E8M0 | NVFP4 group=16 + E4M3 + per-tensor FP32 `weight_scale_2` + FP32 `input_scale=1.0` |
| Trunk attention | `wq_a, wq_b, wkv, wo_a, wo_b` and fused | FP8 block 128×128 | unchanged |
| Trunk shared experts | `shared_experts.w*` | FP8 block 128×128 | unchanged |
| Trunk hc/norms/indexer/compressor | various | BF16 / mixed | unchanged |
| **Entire MTP layer `mtp.0.*`** | mixed (FP8 + MXFP4 + BF16) | **byte-identical to native** — passthrough, no transformation |
| Embeddings / head / hc_head | various | BF16 / FP32 | unchanged |

NVIDIA's [`nvidia/DeepSeek-V3.2-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) reference recipe excludes the entire MTP layer (`model.layers.61*` for V4-Pro) from quantization. v12 follows this. Conversion script: [`scripts/convert_v4_pro_mxfp4_to_nvfp4.py`](scripts/convert_v4_pro_mxfp4_to_nvfp4.py) (~25 min on 1× B300).

## Upstream contributions

| PR / Issue | Description | Status |
|---|---|---|
| [`vllm-project/vllm#42209`](https://github.com/vllm-project/vllm/pull/42209) (sychen52, NVIDIA) | NVFP4 MoE support for DSV4 | **MERGED** 2026-05-22 |
| [`vllm-project/vllm#43248`](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` | open |
| [`vllm-project/vllm#43288`](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` defensive `.get()` + BF16 `getattr` wrap | open |
| [`vllm-project/vllm#43290`](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback (attention) | open |
| [`vllm-project/vllm#43319`](https://github.com/vllm-project/vllm/pull/43319) | MTP loader: **add `.scale` to detector + candidate-list scale resolution** (the load-bearing fix) | open |
| [`vllm-project/vllm#43467`](https://github.com/vllm-project/vllm/pull/43467) | DSV4 MegaMoE early-fail for NVFP4 (deep_gemm + NVFP4 incompatible) | open |
| `patches/patch_v12b_per_layer_moe_routing.diff` | DSV4FP8Config: route MTP MoE through MXFP4 when global `moe_quant_algo=NVFP4` (hybrid NVFP4-trunk + MXFP4-MTP) | local; upstream PR pending |

The installer script applies all open patches automatically. When they merge upstream the patch set shrinks.

## Repo layout

```
MODEL_CARD.md                         — HF model card (the HF README)
LICENSE                               — MIT (inherits from upstream)
README.md                             — this file
docs/
  QUICKSTART.md                       — end-to-end serve recipe
  VLLM_SETUP_ISSUES.md                — the 5 patches + 14 setup gotchas
  FINDINGS.md                         — index of findings docs
  benchmarks/                         — per-benchmark JSON outputs
  findings/                           — methodology + diagnostic notes
    v12_nvfp4_mtp_working_2026_05_24.md   — how we got to ~91% MTP
    backend_format_matrix.md          — NVFP4×MXFP4 × flashinfer×deep_gemm matrix
    throughput_scaling.md             — bs=1 → bs=128 batched concurrency sweep
    aime30_full_2026_05_22.md         — AIME-30 Non-Think greedy
    mmlu_pro_2026_05_22.md            — MMLU-Pro 5-shot full 12k
    humaneval_2026_05_22.md           — HumanEval / HumanEval+ via EvalPlus
    ifeval_2026_05_22.md              — IFEval zero-shot
    gsm8k_scoring_correction.md       — numeric-match scorer fix
    e_proj_h_proj_forensic.md         — byte-equivalence proof
    conversion_v3_validation.md       — 192-tensor sampled validation
  recipes/
    nvfp4_fp8_mtp_replication.md      — full conversion + serve replication
patches/
  patch_43248_ct_bool_wrap.diff
  patch_43288_scale_fmt_get.diff
  patch_43290_weight_scale_fallback.diff
  patch_43319_mtp_quant_detect.diff   (now includes .scale fix)
  patch_v12b_per_layer_moe_routing.diff
scripts/
  install_vllm_with_patches.sh        — one-line installer
  convert_v4_pro_mxfp4_to_nvfp4.py    — GPU conversion (~25 min on 1× B300)
  bench_v4_pro.py                     — GSM8K / AIME / latency / MTP harness
vendor/dsv4-pro-upstream/             — vendored upstream config + reference
```

## License

MIT, inherited from `deepseek-ai/DeepSeek-V4-Pro`. See [`LICENSE`](LICENSE).

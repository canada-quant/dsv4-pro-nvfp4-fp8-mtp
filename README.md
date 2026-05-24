# dsv4-pro-nvfp4-fp8-mtp

Source repo for [`canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) — the first NVFP4 conversion of DeepSeek V4-Pro with **working MTP speculative decoding** on vLLM mainline.

Routed MoE experts converted MXFP4 group=32 → NVFP4 group=16 (E4M3 block scales + FP32 per-tensor `weight_scale_2`). All other tensors passthrough from native — most importantly, the **entire MTP block (`mtp.0.*`) is byte-identical** to the native source. Following NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe of excluding the entire MTP layer from quantization.

## Headline (8× B300 SXM6 AC, TP=8 + EP, vLLM mainline + 5 patches, cuda graphs ON)

### MTP speculative decoding

| Metric | Value |
|---|---|
| **MTP n=1 acceptance, focused probe (20 prompts)** | **91.21%** (3010/3300 drafts) |
| **Cumulative, MTP probe + AIME thinking=high (full reasoning)** | **92.83%** (37,341/40,225) |
| Reference: fork docker + native MXFP4 | 91.07% – 91.94% |
| Earlier conversion attempts (v0.2/v0.3/v0.4) | 2.65% – 3.33% |

v12 is at parity with the native checkpoint baseline. Earlier attempts perturbed `mtp.0` in some way (NVFP4 experts in v0.2; BF16 dequant in v0.3/v0.4) and tripped a separate `_mtp_block_is_quantized_on_disk` detector bug in vLLM. v12 fixes both.

### Quality (greedy temperature 0)

| Benchmark | This artifact (NVFP4) | V4-Flash NVFP4 predecessor | RedHat V4-Flash NVFP4 |
|---|---|---|---|
| **AIME 2024 thinking=high** (n=30, max_tokens=60000, 0 truncations) | **21/30 = 70.00%** | 0.8333 | 0.9000 |
| GSM8K strict 8-shot (full n=1319, 0 truncations) | **0.9659** (CI [0.9547, 0.9744]) | 0.9181 | 0.910 (self-report) |
| GSM8K matched n=300, NVFP4 vs source MXFP4 | 0.9867 vs 0.9900 (1 strict-loss) | n/a | n/a |
| HumanEval pass@1 (EvalPlus, greedy) | **0.951** | 0.915 | 0.896 |
| HumanEval+ pass@1 (EvalPlus, greedy) | **0.902** | 0.854 | 0.860 |
| MBPP pass@1 (EvalPlus, greedy) | **0.929** | not reported | not reported |
| MBPP+ pass@1 (EvalPlus, greedy) | **0.778** | not reported | not reported |
| IFEval prompt_level_strict | _in progress_ | 0.8540 | 0.8207 |
| MMLU-Pro 5-shot (full n=12,032) | _queued_ | 0.8113 | not reported |

### Throughput (MTP n=1 + cuda graphs, `max_model_len=65536`)

| Operating point | Aggregate tok/s |
|---|---|
| c=1 single-stream | **139.3** |
| c=16 batched (64 prompts) | **672.6** |
| c=64 batched (256 prompts) | **1,927.3** |
| c=128 batched (512 prompts) | **3,004.8** |

Throughput scales smoothly through c=128. **Production sweet spot c=32–128** depending on tail-latency tolerance.

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

Full setup in [`docs/QUICKSTART.md`](docs/QUICKSTART.md). The 5 patches + setup gotchas catalog in [`docs/VLLM_SETUP_ISSUES.md`](docs/VLLM_SETUP_ISSUES.md). Full debug chain that fixed MTP: [`docs/findings/v12_nvfp4_mtp_working_2026_05_24.md`](docs/findings/v12_nvfp4_mtp_working_2026_05_24.md).

## Recipe summary

| Group | Modules | Source | Target |
|---|---|---|---|
| Trunk routed experts | `layers.X.ffn.experts.Y.w{1,2,3}` | MXFP4 group=32 + E8M0 | NVFP4 group=16 + E4M3 + per-tensor FP32 `weight_scale_2` + FP32 `input_scale=1.0` |
| Trunk attention | `wq_a, wq_b, wkv, wo_a, wo_b` and fused | FP8 block 128×128 | unchanged |
| Trunk shared experts | `shared_experts.w*` | FP8 block 128×128 | unchanged |
| Trunk hc/norms/indexer/compressor | various | BF16 / mixed | unchanged |
| **Entire MTP layer `mtp.0.*`** | mixed (FP8 + MXFP4 + BF16) | **byte-identical to native** — passthrough, no transformation |
| Embeddings / head / hc_head | various | BF16 / FP32 | unchanged |

NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe excludes the entire MTP layer (`model.layers.61*` for V4-Pro) from quantization. v12 follows this. Conversion script: [`scripts/convert_v4_pro_mxfp4_to_nvfp4.py`](scripts/convert_v4_pro_mxfp4_to_nvfp4.py) (~25 min on 1× B300).

## Upstream contributions

| PR / Issue | Description | Status |
|---|---|---|
| vLLM [#42209](https://github.com/vllm-project/vllm/pull/42209) (sychen52, NVIDIA) | NVFP4 MoE support for DSV4 | **MERGED** 2026-05-22 |
| vLLM [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` | open |
| vLLM [#43288](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` defensive `.get()` + BF16 `getattr` wrap | open |
| vLLM [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback (attention) | open |
| vLLM [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP loader: **add `.scale` to detector + candidate-list scale resolution** (the load-bearing fix) | open |
| vLLM [#43467](https://github.com/vllm-project/vllm/pull/43467) | DSV4 MegaMoE early-fail for NVFP4 (deep_gemm + NVFP4 incompatible) | open |
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
    throughput_scaling.md             — c=1 → c=128 batched concurrency sweep
    aime30_full_2026_05_22.md         — AIME-30 thinking=high
    mmlu_pro_2026_05_22.md            — MMLU-Pro 5-shot full 12k
    humaneval_2026_05_22.md           — HumanEval / HumanEval+ via EvalPlus
    ifeval_2026_05_22.md              — IFEval zero-shot
    gsm8k_scoring_correction.md       — numeric-match scorer fix
    e_proj_h_proj_forensic.md         — byte-equivalence proof
    conversion_v3_validation.md       — 192-tensor sampled validation
    (older findings retained for the historical debug record)
  recipes/
    nvfp4_fp8_mtp_replication.md      — full conversion + serve replication
patches/
  patch_43248_ct_bool_wrap.diff
  patch_43288_scale_fmt_get.diff
  patch_43290_weight_scale_fallback.diff
  patch_43319_mtp_quant_detect.diff   (now includes .scale fix)
  patch_v12b_per_layer_moe_routing.diff
  patch_v12a_detector_scale_suffix.diff   (folded into #43319)
  (older v0p3/v0p5/v0p6/v0p7 retained as reference; not in install)
scripts/
  install_vllm_with_patches.sh        — one-line installer
  convert_v4_pro_mxfp4_to_nvfp4.py    — GPU conversion (~25 min on 1× B300)
  bench_v4_pro.py                     — GSM8K / AIME / latency / MTP harness
vendor/dsv4-pro-upstream/             — vendored upstream config + reference
memory/MEMORY.md                      — load-bearing facts for future sessions
```

## License

MIT, inherited from `deepseek-ai/DeepSeek-V4-Pro`. See [`LICENSE`](LICENSE).

# dsv4-pro-nvfp4-fp8-mtp

Source repo for the [`canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) artifact (private during validation; flipped public after benchmark sign-off).

## What this is

A converted NVFP4 quantization of `deepseek-ai/DeepSeek-V4-Pro`:

- **Source**: `deepseek-ai/DeepSeek-V4-Pro` (native MXFP4 experts + FP8 block attention + BF16 hc_*/norms — **the public release IS already FP4+FP8; no public BF16 source exists**)
- **Target**: NVFP4 group=16, FP8-E4M3 block scales + FP32 per-tensor `weight_scale_2`, FP8 block attention preserved, MTP block retained and serve-loadable
- **Total params**: 1,598.84 B (~1.6 T) — verified by summing tensor element counts across all 64 safetensors shards
- **Active params per token**: 49.60 B (matches NVIDIA's published "49 B active" within rounding)
- **Output artifact size**: ~852 GiB across 64 shards + sidecars
- **Hardware target**: 8× B300 SXM6 AC (sm_103a), TP=8

## Recipe (what we changed vs the native source)

| Tensor category | Source format | Target format | Action |
|---|---|---|---|
| `layers.X.ffn.experts.Y.w{1,2,3}` | MXFP4 group=32 + E8M0 block scale | NVFP4 group=16 + E4M3 block scale + FP32 per-tensor S_g | Re-quantize. Math: dequant → regroup → per-tensor `S_g = max_amax/(FP4_max × E4M3_max)` (shared between w1/w3 per ModelOpt requirement) → per-block E4M3 S_l → FP4 grid quantize |
| `layers.X.attn.*` | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| `layers.X.ffn.shared_experts.w*` | FP8 block 128×128 | FP8 block 128×128 | Passthrough |
| `layers.X.hc_attn_*`, `hc_ffn_*`, `attn_norm`, `ffn_norm` | BF16 | BF16 | Passthrough |
| `layers.X.attn.{compressor,indexer}.*` | mixed FP8/BF16 | mixed FP8/BF16 | Passthrough |
| `mtp.0.ffn.experts.Y.w*` | MXFP4 group=32 | NVFP4 group=16 | Re-quantize (same as main trunk) |
| **`mtp.0.{e_proj,h_proj}.weight`** | FP8 block 128×128 | **BF16 (dequantized)** | Workaround for upstream vLLM `ReplicatedLinear+Fp8Config` scale-param-registration bug — see `docs/findings/phase_0_closeout.md` |
| `mtp.0.attn.*`, `hc_*`, `norms` | mixed | unchanged | Passthrough |
| `embed.weight`, `head.weight`, `norm.weight`, `hc_head_*` | BF16/F32 | unchanged | Passthrough |

Per-expert sidecars emitted alongside each MoE expert weight (required by ModelOpt's NVFP4 loader):

- `<expert>.w{1,2,3}.weight_scale_2`: FP32 [1] per-tensor global scale (w1.S_g == w3.S_g per ModelOpt invariant)
- `<expert>.w{1,2,3}.input_scale`: FP32 [1] dynamic activation scale, value 1.0

## Quick start (private repo — token required)

```bash
# 1. Install vLLM with our patches (~15 min build)
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash

# 2. Download the artifact (852 GiB, ~3-5 min with HF Xet + token)
hf auth login
hf download canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP --local-dir /scratch/v4-pro-nvfp4

# 3. Serve (TP=8 on 8× B300; FlashInfer NVFP4 backend autotunes ~2 min on first run)
export PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda
vllm serve /scratch/v4-pro-nvfp4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --moe-backend flashinfer_trtllm \
  --enforce-eager \
  --port 8089
```

## Patches required

| PR | Repo | Purpose | Status |
|---|---|---|---|
| `vllm-project/vllm#42209` (sychen52, NVIDIA) | vllm | NVFP4 MOE support for DSV4 (ModelOptNvFp4FusedMoE + trtllm_nvfp4_moe + oracle/nvfp4.py) | OPEN — cherry-picked into our branch |
| `vllm-project/vllm#43248` | vllm | `bool()` wrap on `is_static_input_scheme` (compressed_tensors) | OPEN |
| `vllm-project/vllm#43288` | vllm | `scale_fmt` defensive `.get()` + BF16 `getattr` wrap | OPEN |
| `vllm-project/vllm#43290` (revised) | vllm | `weight_scale_inv`-or-`weight_scale` fallback, cached at init to avoid dynamo data-dependent branching | OPEN |
| `vllm-project/vllm#43319` (revised) | vllm | MTP loader candidate-list for `.scale` resolution (handles e_proj/h_proj naming variants) | OPEN |

All five are applied automatically by `scripts/install_vllm_with_patches.sh`.

## Repo layout

```
PLAN.md                                — 8-phase plan with gates
MODEL_CARD.md                          — HF-uploaded model card (filled with measurements)
README.md                              — this file
CLAUDE.md                              — agent handoff notes (private, not pushed to HF)
patches/                               — 4 .diff files for the vLLM patches
  patch_43248_ct_bool_wrap.diff
  patch_43288_scale_fmt_get.diff
  patch_43290_weight_scale_fallback.diff
  patch_43319_mtp_quant_detect.diff
  VERSIONS.md
scripts/
  install_vllm_with_patches.sh         — one-line installer (CUDA toolkit + Rust + 4 patches + #42209 cherry-pick)
  convert_v4_pro_mxfp4_to_nvfp4.py     — GPU-accelerated conversion pipeline (~17 min for 64 shards on 1× B300)
  bench_v4_pro.py                      — unified GSM8K / AIME / latency / MTP harness
  run_all_benchmarks.sh                — fires all 4 benches sequentially
  run_native_baseline.sh               — switches the serve to the native MXFP4 source for baseline comparison
  compare_bench_results.py             — emits Markdown comparison from the JSON results
docs/
  findings/                            — per-phase methodology + diagnostic notes
    upstream_research_v4_pro_ground_truth.md
    option_a_lossless_audit.md
    vllm_pro_serving_path.md
    v4_pro_active_params.md
    v4_flash_patches_after_pro_subpackage.md
    conversion_math_validation.md
    conversion_v3_validation.md
    phase_0_closeout.md
  benchmarks/                          — per-benchmark JSON outputs + write-ups
vendor/dsv4-pro-upstream/              — vendored upstream model.py / kernel.py / config.json (Apache-2.0)
memory/MEMORY.md                       — pointers to load-bearing facts for future sessions
```

## Predecessor

V4-Flash predecessor at [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) (shipped 2026-05-21). V4-Pro is **not** a straight re-application of that recipe — the format conversion (MXFP4 → NVFP4) and serving path (FlashInfer NVFP4 backend + PR #42209) are V4-Pro-specific.

## License

MIT, inherited from upstream `deepseek-ai/DeepSeek-V4-Pro`.

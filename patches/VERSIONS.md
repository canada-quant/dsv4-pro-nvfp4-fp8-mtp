# Pinned dependency versions for V4-Pro NVFP4-FP8-MTP serve + build

## vLLM build

- **Upstream base**: vLLM mainline at SHA `30f52a895` (2026-05-23) or newer.
  This SHA already includes the merged PR #42209 (NVFP4 MoE support for DSV4).
- **Local patch stack**: 5 PR patches + 1 local-only diff (see "Patches" below)
- **Build flags**:
  - `TORCH_CUDA_ARCH_LIST=10.3a` (B300 sm_103a; NOT `10.0a` which is sm_100a
    for B200 and won't run on B300)
  - `CUDA_HOME=/usr/local/cuda` (full toolkit; DLAMI's `/opt/pytorch/cuda` is
    runtime-only)
  - `pip install -e . --no-build-isolation`
- **Build prerequisites**:
  - `setuptools-rust>=1.9.0` — required by vLLM PR #43283 (Rust frontend)
  - Rust toolchain via rustup
  - `cuda-toolkit-13-0` system package (DLAMI bundled CUDA is runtime-only)
  - `nvidia-fabricmanager-595` running (`sudo systemctl start nvidia-fabricmanager`)
  - Clean `.deps/` + `build/` between commits that switch cmake generator

## Patches applied (5 PR + 1 local)

| Patch file | Upstream PR / issue | Site | Status |
|---|---|---|---|
| `patches/patch_43248_ct_bool_wrap.diff` | [vllm#43248](https://github.com/vllm-project/vllm/pull/43248) | `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:679,700` | open |
| `patches/patch_43288_scale_fmt_get.diff` | [vllm#43288](https://github.com/vllm-project/vllm/pull/43288) | `vllm/models/deepseek_v4/nvidia/model.py:909` | open |
| `patches/patch_43290_weight_scale_fallback.diff` | [vllm#43290](https://github.com/vllm-project/vllm/pull/43290) | `vllm/models/deepseek_v4/attention.py:334` | open |
| `patches/patch_43319_mtp_quant_detect.diff` | [vllm#43319](https://github.com/vllm-project/vllm/pull/43319) | `vllm/models/deepseek_v4/nvidia/mtp.py` — **`.scale` suffix in detector + candidate-list scale resolution** (the load-bearing one) | open |
| `patches/patch_43467_megamoe_nvfp4_guard.diff` | [vllm#43467](https://github.com/vllm-project/vllm/pull/43467) | `vllm/models/deepseek_v4/nvidia/model.py:711` — DSV4 MegaMoE early-fail when NVFP4 artifact is loaded with `--moe-backend deep_gemm_mega_moe` | open |
| `patches/patch_v12b_per_layer_moe_routing.diff` | (local — upstream PR pending) | `vllm/models/deepseek_v4/quant_config.py` `DSV4FP8Config.get_quant_method` — when trunk `moe_quant_algo=NVFP4`, route MTP layer MoE through `Mxfp4MoEMethod` (because the entire MTP layer is excluded from quantization in v12, following NVIDIA's V3.2-NVFP4 reference recipe) | local |

## Already-merged

- **vLLM [#42209](https://github.com/vllm-project/vllm/pull/42209)** (sychen52, xinli-sw, pavanimajety, zyongye / NVIDIA) — NVFP4 MoE support for DSV4. **MERGED 2026-05-22.** Present in mainline at SHA `30f52a895` and later.

## Archived (superseded, kept under `patches/archive/`)

These were intermediate attempts during the MTP-recovery debug arc. v12 + the
five live patches above superseded all of them. They are kept in
`patches/archive/` for the historical record only — not applied by the
installer.

- `patch_v0p3_dsv4_mtp_bf16_dispatch.diff` — Original combined patch. Part (a)
  BF16 mtp.0 dispatch is superseded by v12's NVIDIA-recipe full passthrough +
  #43319's `.scale` detector fix. Part (b) MegaMoE NVFP4 early-fail is split
  out as `patches/patch_43467_megamoe_nvfp4_guard.diff`.
- `patch_v0p5_mtp_unquant_full_block.diff` — full-block MTP unquant experiment.
- `patch_v0p6_fork_pattern_hc_post.diff` — fork-pattern hc_post experiment.
- `patch_v0p7_disable_topk_buffer_share.diff` — topk_indices_buffer experiment.
- `patch_v12a_detector_scale_suffix.diff` — folded into `patch_43319_mtp_quant_detect.diff`.

## flashinfer pin (required)

```bash
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'
```

0.6.11.post2 has an ABI regression that silently crashes vLLM workers during
model construction (`RuntimeError: cancelled` with no stack trace).

## Upstream PR status (refresh at session start)

```bash
for n in 43248 43288 43290 43319 43467; do
  gh pr view $n --repo vllm-project/vllm --json state,mergedAt --jq '"#\($n): state=\(.state) mergedAt=\(.mergedAt)"'
done
```

If any of the five PRs merges, drop the corresponding `patches/patch_NNNN_*.diff` from `scripts/install_vllm_with_patches.sh` and bump the pinned base SHA.

## Serve env summary

| Package | Version | Source |
|---|---|---|
| vllm | mainline `30f52a895` + 5 PR patches + 1 local | editable install |
| torch | 2.11.0+cu130 | DLAMI bundled |
| flashinfer | 0.6.8.post1 (pinned) | required |
| setuptools-rust | ≥1.9.0 | required by vLLM Rust frontend |
| cargo / rustc | stable | rustup user install |
| CUDA toolkit | 13.0 | `/usr/local/cuda` |
| evalplus | 0.3.1 | HumanEval / MBPP |
| lm_eval | 0.4.12 | MMLU-Pro / IFEval |

## Serve flags (production)

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

No `--enforce-eager` — cuda graphs in `FULL_AND_PIECEWISE` mode are required for production decode throughput.

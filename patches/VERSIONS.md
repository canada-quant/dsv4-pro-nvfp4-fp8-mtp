# Pinned dependency versions for V4-Pro NVFP4-FP8-MTP serve + build

## vLLM build

- **Upstream base**: `vllm-project/vllm@39910f2b25` (2026-05-22)
  ([Rust Frontend] Move code from `vllm-frontend-rs` (#43283))
- **Branch on box**: `canada-quant-v4pro-nvfp4` at `0a78a466d9` (= base +
  4 patch commits)
- **Resulting `vllm.__version__`**: `0.21.1rc1.dev196+g0a78a466d.d20260521`
- **Build flags**:
  - `TORCH_CUDA_ARCH_LIST=10.3a` (B300 sm_103a; NOT `10.0a` which is sm_100a
    for B200 and won't run on B300)
  - `CUDA_HOME=/usr/local/cuda` (full toolkit; DLAMI's `/opt/pytorch/cuda`
    is runtime-only)
  - `pip install -e . --no-build-isolation`
- **Build prerequisites added since V4-Flash**:
  - `setuptools-rust>=1.9.0` — required by vLLM PR #43283 (Rust frontend).
    Without it, `pyproject.toml` metadata fails with
    `ModuleNotFoundError: No module named 'setuptools_rust'`.
  - Rust toolchain (`cargo`, `rustc`) — install via rustup
    (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal`)
  - **Clean `.deps/` + `build/` before re-build** when switching commits
    that use a different cmake generator (Ninja vs Make). vLLM main builds
    on Ninja today; stale Make caches from a prior build will hard-fail.

## Patches applied (4)

All target lines on current main are un-fixed; none of the 4 PRs have merged.

| Patch file | Upstream PR | Reason kept | Site |
|---|---|---|---|
| `patches/patch_43248_ct_bool_wrap.diff` | [vllm#43248](https://github.com/vllm-project/vllm/pull/43248) | Generic compressed-tensors; V4-Pro FP8 attention path passes through this | `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:679,700` |
| `patches/patch_43288_scale_fmt_get.diff` | [vllm#43288](https://github.com/vllm-project/vllm/pull/43288) | Defensive scale_fmt access; BF16-serving safety | `vllm/models/deepseek_v4/nvidia/model.py:909` |
| `patches/patch_43290_weight_scale_fallback.diff` | [vllm#43290](https://github.com/vllm-project/vllm/pull/43290) | V4-Pro attention stores `weight_scale` not `weight_scale_inv`; load-blocking otherwise | `vllm/models/deepseek_v4/attention.py:334` |
| `patches/patch_43319_mtp_quant_detect.diff` | [vllm#43319](https://github.com/vllm-project/vllm/pull/43319) | V4-Pro MTP is MXFP4-quantized on disk; MTP loader must detect and route through quant_config | `vllm/models/deepseek_v4/nvidia/mtp.py:61` |

Dropped from the V4-Flash inheritance:
- `helpers.py.diff` (llm-compressor sequential pipeline calibration tracer) —
  irrelevant; V4-Pro recipe is format conversion, not gradient calibration.
- `modeling_deepseek_v4.py.diff` ≈ [transformers#46127](https://github.com/huggingface/transformers/pull/46127)
  — irrelevant; vLLM owns the V4-Pro loader, stock transformers V4 class is
  not in our forward path.

## Upstream PR status (refresh at session start)

```bash
for n in 43248 43288 43290 43319 42209; do
  gh pr view $n --repo vllm-project/vllm --json state,merged --jq '"#\($n): state=\(.state) merged=\(.merged)"'
done
```

| PR | What it does | Status as of 2026-05-21 | Action if merged |
|---|---|---|---|
| vllm-project/vllm#43248 | `bool()` wrap on `is_static_input_scheme` | open | drop patch_43248_ct_bool_wrap.diff |
| vllm-project/vllm#43288 | scale_fmt defensive read + BF16 getattr | open | drop patch_43288_scale_fmt_get.diff |
| vllm-project/vllm#43290 | weight_scale_inv-or-weight_scale fallback | open | drop patch_43290_weight_scale_fallback.diff |
| vllm-project/vllm#43319 | MTP quant detection from safetensors | open | drop patch_43319_mtp_quant_detect.diff |
| vllm-project/vllm#42209 | **Add NVFP4 MOE support for DeepSeek V4** (sychen52) | open, ready, base=main, branch=nvfp4_dsv4 | Phase 4 design must coordinate with this — see findings doc |

## Calibration env (`/data/venv-calib`)

V4-Pro recipe does not use gradient-based calibration (source is already FP4),
so this venv is not load-bearing for V4-Pro work. Inherited from V4-Flash:

| Package | Version | Notes |
|---|---|---|
| torch | 2.11.0+cu130 | DLAMI bundled |
| compressed-tensors | 0.15.1a20260515 | not exercised in V4-Pro recipe |
| llmcompressor | `f2aa32e2` SHA | not exercised in V4-Pro recipe |
| py-spy | 0.4.2 | retained for diagnostics |

## Serve env (`/data/venv-serve`)

| Package | Version | Source |
|---|---|---|
| vllm | `0.21.1rc1.dev196+g0a78a466d.d20260521` | editable install from `/data/src/vllm` on branch `canada-quant-v4pro-nvfp4` (base `39910f2b25` + 4 patches) |
| torch | 2.11.0+cu130 | DLAMI bundled |
| setuptools-rust | 1.12.1 | required by vLLM Rust frontend |
| cargo / rustc | 1.95.0 | rustup user install at `~/.cargo/bin` |
| flashinfer | 0.6.8.post1 | inherited |
| CUDA toolkit | 13.0 | `/usr/local/cuda` |
| evalplus | latest | for HumanEval phase |
| lm_eval | 0.4.12 | for IFEval / MMLU-Pro phase |

## Serve flags

For native V4-Pro MXFP4-FP8-MTP (Phase 0 smoke):

```bash
CUDA_HOME=/usr/local/cuda \
  vllm serve /path/to/DeepSeek-V4-Pro \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --tensor-parallel-size 8 \
  --moe-backend deep_gemm_mega_moe \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

`--enforce-eager` may be needed if cuda-graph capture fails — PR #42604
(merged 2026-05-15) added cuda-graph mode for V4-Pro but verify
empirically before claiming it works without the flag.

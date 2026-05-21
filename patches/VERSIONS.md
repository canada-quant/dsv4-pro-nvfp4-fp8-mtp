# Pinned dependency versions for V4-Pro calibration + serve

Inherited from V4-Flash. Update as new pins are verified for V4-Pro.

## Calibration env (`/data/venv-calib`)

| Package | Version | Source |
|---|---|---|
| torch | 2.11.0+cu130 | DLAMI bundled |
| compressed-tensors | 0.15.1a20260515 | pypi |
| llmcompressor | `f2aa32e2` (SHA) | git install from main |
| py-spy | 0.4.2 | pypi (for stuck-process diagnostics) |
| transformers | 5.8.1 | pypi — **patched** with `patches/modeling_deepseek_v4.py.diff` to remove the `mtp.*` strip on load |
| safetensors | latest | (auto via torch/compressed-tensors) |
| datasets | latest | (auto) |

## Serve env (`/data/venv-serve`)

| Package | Version | Source |
|---|---|---|
| vllm | `0.21.1rc1.dev164+gd05d52059.d20260521` | git source build, mainline + PR #42209 cherry-pick + 5 local patches (see `docs/VLLM_SETUP_ISSUES.md`) |
| torch | 2.11.0+cu130 | DLAMI bundled |
| flashinfer | 0.6.8.post1 | pypi |
| CUDA toolkit | 13.0 | `/usr/local/cuda` (system install via `sudo apt install cuda-toolkit-13-0`) |
| evalplus | latest | pypi |
| lm_eval | 0.4.12 | pypi |

## Build flags

- `TORCH_CUDA_ARCH_LIST=10.3a` for B300 sm_103a (NOT 10.0a — that's sm_100a for B200, won't run on B300)
- `CUDA_HOME=/usr/local/cuda` at both build time and serve time (Tilelang invokes nvcc at runtime; DLAMI's `/opt/pytorch/cuda` lacks headers)
- `VLLM_TEST_FORCE_FP8_MARLIN=1` at serve time (DeepGemm's sm_103a FP8 kernels are partial as of 2026-05; Marlin FP8 is the safe default)

## Upstream PR status (refresh at session start)

| PR | What it fixes | Status as of 2026-05-21 |
|---|---|---|
| vllm-project/vllm#43248 | `bool()` wrap on `is_static_input_scheme` | open |
| vllm-project/vllm#43288 | `.get("scale_fmt", "ue8m0")` + BF16 `getattr` follow-up | open |
| vllm-project/vllm#43290 | `weight_scale_inv`-or-`weight_scale` fallback | open |
| vllm-project/vllm#43319 | MTP-quant-detect + BF16 `wo_a` fallback | open |
| vllm-project/vllm#43297 | `(1,)`-shape `global_scale` loader broadcast (issue) | open |
| vllm-project/vllm#43304 | MTP draft inherits main quant scheme (issue) | partially addressed by #43319 |
| huggingface/transformers#46127 | adds `DeepseekV4NextNPredictor` (sibling-filed) | open, awaits `forward()` + tests |
| vllm-project/llm-compressor#2734 | Observer.synchronize NCCL desync | open |
| vllm-project/llm-compressor#2743 | Multi-rank cache-offload deadlock | open |
| vllm-project/llm-compressor#2745 | MTP inference-tensor crash | open |
| vllm-project/compressed-tensors#711 | sharded-module load path | open |

Refresh with:

```bash
for n in 43248 43288 43290 43319 43297 43304; do
  gh pr view $n --repo vllm-project/vllm --json state,merged --jq '"#\($n): state=\(.state) merged=\(.merged)"'
done
```

If any are merged, drop the corresponding patch from `scripts/install_vllm_with_patches.sh`.

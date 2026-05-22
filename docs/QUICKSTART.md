# Quick start — serve `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` on vLLM

End-to-end recipe to get the artifact serving with the recommended upstream-default config on a Blackwell-class node.

## Hardware

- **GPU**: 8× NVIDIA B300 SXM6 AC (288 GB HBM3e each). Compute capability **10.3** (`sm_103a`).
- **System RAM**: ≥512 GB recommended for the vLLM load path (the artifact is 852 GiB on disk, loaded across 8 ranks).
- **Disk**: ~900 GB free for the artifact + scratch.

Other Blackwell SKUs (B200, GB200, GB300) may work — they have different compute caps. **Run `python -c "import torch; print(torch.cuda.get_device_capability(0))"` before building.** B300 SXM6 AC is what we tested.

The artifact's ~960 GB weight footprint (when including the safetensors header overhead and inflight buffers) does not fit on a single GB200 NVL4 tray; that platform needs multi-node DP+EP. Single-node deployment is validated on 8× B300 only.

## Step 1 — Base environment

A fresh AWS DLAMI Ubuntu 24.04 image already has the right Python (3.13) + PyTorch (2.11.0+cu130) + CUDA 13 runtime at `/opt/pytorch`. We install vLLM directly into the `/opt/pytorch` venv (do NOT use `--system-site-packages` from another venv — that has a long history of breakage on torch propagation).

If you're not on the DLAMI, build yourself a Python 3.13 venv with torch 2.11.0+cu130 first.

CUDA 13 source-build dependencies (the bundled DLAMI CUDA is runtime-only):

```bash
sudo apt update && sudo apt install -y cuda-toolkit-13-0 nvidia-fabricmanager-595
sudo systemctl start nvidia-fabricmanager   # otherwise CUDA Error 802 on multi-GPU
```

Confirm hardware + capabilities:

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
# Expect: 8× B300, 288 GB each, compute_cap 10.3

/opt/pytorch/bin/python -c "import torch; print(torch.cuda.get_device_capability(0))"
# Expect: (10, 3)
```

## Step 2 — Build vLLM with the 4 local patches

One-line installer (clones mainline at the pinned SHA, applies 4 patches, installs setuptools-rust + rustup + ninja, builds with `TORCH_CUDA_ARCH_LIST=10.3a`):

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

This pins vLLM mainline @ `39910f2b25` (2026-05-22 snapshot, includes the now-merged PR #42209 for NVFP4 MoE support) and applies the 4 open patches. Total build time ~15 min on a fresh DLAMI. The patches are extracted into `patches/` for inspection.

If `cargo` is not on your worker PATH, prepend `$HOME/.cargo/bin` to PATH before running `vllm serve`. The vLLM frontend now has a Rust component (PR #43283) and the build/run requires Rust ≥ 1.78.

## Step 3 — Download the artifact

```bash
hf auth login                                           # HF token with read access to the private repo
hf download canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  --local-dir /scratch/v4-pro-nvfp4
# 852 GiB. With HF Xet protocol + auth, ~5-10 min on a 10 Gbps NIC.
```

## Step 4 — Serve

The recommended config (upstream-default `single_node_tep` strategy + the V4-Pro Blackwell extras):

```bash
export PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda

vllm serve /scratch/v4-pro-nvfp4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8089
```

Notes:

- **`--moe-backend flashinfer_trtllm`** is required for this NVFP4 artifact. Using `--moe-backend deep_gemm_mega_moe` raises `KeyError: 'layers.0.ffn.experts.w13_input_scale'` at load — the mega-kernel path expects fused-name MoE params, NVFP4 ModelOpt layout uses per-expert names. (See [`docs/findings/backend_format_matrix.md`](findings/backend_format_matrix.md).)
- **`--attention_config.use_fp4_indexer_cache=True`** is the Blackwell-specific override from the upstream recipe; applies to the V4-Pro sparse attention indexer regardless of expert format.
- **`--compilation-config FULL_AND_PIECEWISE`** is the upstream `single_node_tep` setting. Without cuda-graphs, decode collapses to ~5 tok/s on this artifact — make sure you don't have `--enforce-eager` set anywhere.
- Cold-start including FlashInfer autotune is ~5-6 min. First request after ready may be slow (autotune warm-up); subsequent requests stabilize at full throughput.

### MTP (opt-in)

Add:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Expected acceptance: ~1.8% per-token. V4-Pro's trained MTP head is structurally weak today (consistent with the LMSYS day-zero report of accept length ~1.19 and with upstream classifying V4-Pro MTP as `opt_in_features` rather than default). MTP retention here means you *can* serve it; not that it's a big win at this point in the upstream curve. See [`docs/findings/upstream_mtp_classification.md`](findings/upstream_mtp_classification.md).

## Step 5 — Smoke test

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/scratch/v4-pro-nvfp4","messages":[{"role":"user","content":"What is 17*19? Return only the final integer."}],"max_tokens":40,"temperature":0}'
```

Expected: `"content":"323"` (sometimes "323 " with whitespace; the chat template's stop tokens handle this).

## Step 6 — Throughput probes

`scripts/bench_v4_pro.py` is the harness used to produce the measurements in [`MODEL_CARD.md`](../MODEL_CARD.md). Examples:

```bash
# c=1 single-stream output throughput (20 short prompts, max_tokens=128)
python scripts/bench_v4_pro.py latency \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4 \
  --n 20 --max-tokens 128 --concurrency 1

# c=16 batched aggregate throughput (64 prompts, gives aggregate_tps + wall-clock)
python scripts/bench_v4_pro.py latency \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4 \
  --n 64 --max-tokens 128 --concurrency 16

# MTP acceptance over a 20-prompt chat workload
python scripts/bench_v4_pro.py mtp \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4

# GSM8K-300 quality probe
python scripts/bench_v4_pro.py gsm8k \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4 \
  --limit 300 --concurrency 16 --max-tokens 2048
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA Error 802 (system not yet initialized)` | fabric manager not running | `sudo systemctl start nvidia-fabricmanager` |
| `KeyError: 'layers.0.ffn.experts.w13_input_scale'` at load | `--moe-backend deep_gemm_mega_moe` on NVFP4 artifact | use `--moe-backend flashinfer_trtllm` |
| `KeyError: 'model.layers.61.e_proj.weight_scale_inv'` | trying to load *native* MXFP4 artifact with `--speculative-config method=mtp` | not a thing this artifact does (we BF16-dequant e_proj/h_proj); this only happens on the native source artifact |
| `ModuleNotFoundError: setuptools_rust` | vLLM Rust frontend not installed | `pip install setuptools-rust>=1.9.0` + rustup install |
| 5 tok/s decode rate | `--enforce-eager` is set somewhere | drop the flag; FULL_AND_PIECEWISE cuda-graph mode is required for full speed |
| `ninja` not found on worker | PATH not propagated to subprocess | prepend `/opt/pytorch/bin` to PATH before `vllm serve` |
| Build fails on `cuda-toolkit-13-0` missing | source build needs system CUDA, not the DLAMI runtime | `sudo apt install -y cuda-toolkit-13-0` |

More gotchas catalogued in [`VLLM_SETUP_ISSUES.md`](VLLM_SETUP_ISSUES.md).

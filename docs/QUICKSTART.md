# Quick start — serve `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` with MTP

End-to-end recipe to get the artifact serving with **~91% focused / ~93% cumulative MTP draft acceptance** + **cuda graphs ON** on a Blackwell-class node.

## Hardware

- **GPU**: 8× NVIDIA B300 SXM6 AC (288 GB HBM3e each). Compute capability **10.3** (`sm_103a`).
- **System RAM**: ≥512 GB recommended for the vLLM load path (913 GB artifact loaded across 8 ranks).
- **Disk**: ~1 TB free for the artifact + scratch.

Other Blackwell SKUs (B200, GB200, GB300) may work but use different compute caps. **Run `python -c "import torch; print(torch.cuda.get_device_capability(0))"` before building.** B300 SXM6 AC is what we tested.

The artifact's ~1 TB weight footprint (with safetensors header overhead) does not fit on a single GB200 NVL4 tray; that platform needs multi-node DP+EP. Single-node deployment is validated on 8× B300 only.

## Step 1 — Base environment

A fresh AWS DLAMI Ubuntu 24.04 image already has the right Python (3.13) + PyTorch (2.11.0+cu130) + CUDA 13 runtime at `/opt/pytorch`. Install vLLM directly into the `/opt/pytorch` venv (do NOT use `--system-site-packages` from another venv — long history of breakage on torch propagation).

If you're not on the DLAMI, build a Python 3.13 venv with torch 2.11.0+cu130 first.

CUDA 13 source-build dependencies (the bundled DLAMI CUDA is runtime-only):

```bash
sudo apt update && sudo apt install -y cuda-toolkit-13-0 nvidia-fabricmanager-595 ninja-build
sudo systemctl start nvidia-fabricmanager   # otherwise CUDA Error 802 on multi-GPU
```

Confirm hardware + capabilities:

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
# Expect: 8× B300, 288 GB each, compute_cap 10.3

/opt/pytorch/bin/python -c "import torch; print(torch.cuda.get_device_capability(0))"
# Expect: (10, 3)
```

## Step 2 — Build vLLM with the 5 local patches

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

Pins vLLM mainline @ `30f52a895` (a SHA that includes the now-merged PR #42209 for NVFP4 MoE support) and applies the 5 open patches. Total build time ~15 min on a fresh DLAMI. Patches extracted into `patches/` for inspection.

If `cargo` is not on your worker PATH, prepend `$HOME/.cargo/bin` to PATH before running `vllm serve`. The vLLM frontend has a Rust component (PR #43283) and requires Rust ≥ 1.78.

## Step 3 — Pin flashinfer to 0.6.8.post1

```bash
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'
```

flashinfer 0.6.11.post2 (which `pip install -e .` of vLLM pulls in) has an ABI regression that silently crashes vLLM workers during model construction. 0.6.8.post1 is stable. If you skip this step, you'll see `RuntimeError: cancelled` and `WorkerProc initialization failed due to an exception in a background process` with no stack trace.

## Step 4 — Download the artifact

```bash
hf auth login                                           # HF token with read access to the repo
hf download canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  --local-dir /scratch/v4-pro-nvfp4
# 913 GiB. With HF Xet protocol + auth, ~5-10 min on a 10 Gbps NIC.
```

## Step 5 — Serve (with MTP + cuda graphs)

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
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --max-model-len 65536 \
  --port 8089
```

Notes:

- **No `--enforce-eager`.** Cuda graphs in `FULL_AND_PIECEWISE` mode are the default and required for production decode throughput.
- **`--moe-backend flashinfer_trtllm`** is the required backend for this NVFP4 artifact. Using `--moe-backend deep_gemm_mega_moe` raises `KeyError: 'layers.0.ffn.experts.w13_input_scale'` at load — the mega-kernel expects fused-name MoE params; NVFP4 ModelOpt layout uses per-expert names. (Filed as [vLLM #43454](https://github.com/vllm-project/vllm/issues/43454); fix PR [#43467](https://github.com/vllm-project/vllm/pull/43467).)
- **MTP n=1** is the production sweet spot (V4-Pro has a 1-layer MTP block; n=2 reuses the same layer twice with degraded acceptance).
- **Cold start: ~12-15 min** (flashinfer FP4 MoE JIT compile + torch.compile + cudagraph capture). The serve will sit in "No available shared memory broadcast block found in 60 seconds" notices during the compile phase — this is normal.

## Step 6 — Smoke test

After "Application startup complete":

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/scratch/v4-pro-nvfp4","messages":[{"role":"user","content":"What is 17*19? Return only the final integer."}],"max_tokens":40,"temperature":0}'
```

Expected: `"content":"323"`.

## Step 7 — Verify MTP acceptance

```bash
python scripts/bench_v4_pro.py mtp \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4
```

Expected: **~91% acceptance** on the default 20-prompt probe. The bench scrapes `/metrics` for `vllm:spec_decode_num_drafts_total` and `vllm:spec_decode_num_accepted_tokens_total` before + after the workload and reports the delta.

If you see <10% acceptance, something in the patch stack is missing — check [`docs/findings/v12_nvfp4_mtp_working_2026_05_24.md`](findings/v12_nvfp4_mtp_working_2026_05_24.md) for the full fix chain that's required.

## Step 8 — Throughput probes (optional)

`scripts/bench_v4_pro.py` is the harness used to produce the measurements in [`MODEL_CARD.md`](../MODEL_CARD.md):

```bash
# c=1 single-stream output throughput (20 short prompts, max_tokens=128)
python scripts/bench_v4_pro.py latency \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4 \
  --n 20 --max-tokens 128 --concurrency 1

# c=16 batched aggregate throughput (64 prompts)
python scripts/bench_v4_pro.py latency \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4 \
  --n 64 --max-tokens 128 --concurrency 16

# c=64 peak aggregate throughput
python scripts/bench_v4_pro.py latency \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4 \
  --n 128 --max-tokens 128 --concurrency 64

# GSM8K full quality probe (1319 problems)
python scripts/bench_v4_pro.py gsm8k \
  --base-url http://localhost:8089 \
  --model /scratch/v4-pro-nvfp4 \
  --concurrency 32 --max-tokens 2048
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA Error 802 (system not yet initialized)` | fabric manager not running | `sudo systemctl start nvidia-fabricmanager` |
| `KeyError: 'layers.0.ffn.experts.w13_input_scale'` at load | `--moe-backend deep_gemm_mega_moe` on NVFP4 | use `--moe-backend flashinfer_trtllm` |
| `KeyError: 'model.layers.61.e_proj.weight_scale_inv'` | vLLM `_mtp_block_is_quantized_on_disk` missing `.scale` suffix | patch #43319 (or install script) |
| `RuntimeError: cancelled` + silent worker crash | flashinfer 0.6.11.post2 ABI regression | `pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'` |
| `ValueError: Unquantized MoE backend FlashInfer TRTLLM does not support routing method` | trying to load MTP MoE unquantized | normally avoided automatically with v12 patches; if you see this, your `_mtp_block_is_quantized_on_disk` detector is misidentifying the MTP block as BF16 |
| `ModuleNotFoundError: setuptools_rust` | vLLM Rust frontend not installed | `pip install setuptools-rust>=1.9.0` + rustup |
| 5 tok/s decode rate | `--enforce-eager` set | drop the flag; cuda graphs required |
| `ninja` not found on worker | PATH not propagated to subprocess | prepend `/usr/bin:/opt/pytorch/bin` to PATH; install system ninja: `sudo apt install ninja-build` |
| Build fails on `cuda-toolkit-13-0` missing | source build needs system CUDA | `sudo apt install -y cuda-toolkit-13-0` |
| ~3% MTP acceptance instead of ~91% | conversion perturbed `mtp.0` (e.g. dequant'd to BF16) | rebuild with the v12 `classify_tensor` that passes `mtp.*` through byte-identical |

More gotchas catalogued in [`VLLM_SETUP_ISSUES.md`](VLLM_SETUP_ISSUES.md).

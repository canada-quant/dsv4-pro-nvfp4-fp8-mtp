# Quick start — serve `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` on vLLM

End-to-end recipe to get the artifact serving with MTP speculative decoding on a Blackwell-class node.

## Hardware

- **GPU**: 4× NVIDIA B300 SXM6 AC (288 GB HBM3e each). Compute capability **10.3** (`sm_103a`).
- **System RAM**: ≥256GB recommended for the vLLM load path.
- **Disk**: ~200 GB free for the 172 GB artifact + scratch.

Other Blackwell SKUs (B200, B100) may work with the right `TORCH_CUDA_ARCH_LIST` — they have different compute caps (e.g. `sm_100a`). **Run `python -c "import torch; print(torch.cuda.get_device_capability(0))"` to confirm before building.**

> ⚠️ Not validated on H100 / H200 / consumer Blackwell (RTX 50-series, sm_120). NVFP4 tensor-core support varies; verify before deploying.

## Step 1 — Python + CUDA environment

```bash
# Ubuntu 24.04 + system CUDA toolkit (for source builds — the bundled DLAMI CUDA is runtime-only)
sudo apt install -y cuda-toolkit-13-0  # adjust for your CUDA version

# Python 3.13 venv (or use the DLAMI's /opt/pytorch venv)
python3.13 -m venv /data/venv-serve
source /data/venv-serve/bin/activate
pip install --upgrade pip wheel
```

> Do NOT use `--system-site-packages` if your OS Python is 3.12 or older — `pyo3_runtime.PanicException` on `cryptography` import.

## Step 2 — Build vLLM from source with the 5 local patches

The 5 patches are needed until upstream merges them — see [VLLM_SETUP_ISSUES.md](VLLM_SETUP_ISSUES.md) for the full diff and rationale.

```bash
git clone https://github.com/vllm-project/vllm /data/src/vllm
cd /data/src/vllm

# Cherry-pick PR #42209 (NVFP4 MoE for DSV4) if not yet on main
gh pr checkout 42209 || true

# Apply our 5 local patches (see VLLM_SETUP_ISSUES.md for the exact lines):
# - PR #43248: bool() wrap on is_static_input_scheme
# - PR #43288: .get("scale_fmt", "ue8m0") + getattr-wrapped quantization_config for BF16 load
# - PR #43290: weight_scale_inv-or-weight_scale fallback
# - PR #43319: MTP-quant-detect + BF16 wo_a fallback path
# - bool() wrap also at 5 sites in compressed_tensors.py (PR #43248 expanded)

# Build for sm_103a (the B300 — NOT sm_100a)
export TORCH_CUDA_ARCH_LIST=10.3a
pip install -e . --no-build-isolation

# Optional: install dependencies the bench harness needs
pip install langdetect immutabledict nltk evalplus openai
```

> **Don't build for sm_100a on B300.** The DLAMI docs may say B300 is sm_100; this is wrong as of 2026-05. `nvidia-smi --query-gpu=compute_cap --format=csv` and `torch.cuda.get_device_capability(0)` are authoritative. `sm_100a` binaries crash silently with no kernel found on `sm_103a`.

## Step 3 — Download the artifact

```bash
huggingface-cli login  # or set HF_TOKEN

# Option A: download to scratch (HF cache)
huggingface-cli download canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --local-dir /scratch/weights/v4-flash-nvfp4-fp8-mtp

# Option B: serve directly from HF cache (vllm handles download lazily)
# Skip this step; pass the HF repo ID directly to vllm serve below.
```

## Step 4 — Serve

```bash
# With MTP speculative decoding (the differentiator):
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --tensor-parallel-size 4 \
    --port 8089 \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

# Plain serve (no spec-decode), for the no-MTP comparison:
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --tensor-parallel-size 4 \
    --port 8089 \
    --kv-cache-dtype fp8
```

**Why TP=4 (not TP=8)?** Measured: TP=4 is faster than TP=8 at c≥4 by up to 21.6% on this artifact. Per-rank MoE expert shards on TP=8 are small enough to underutilize NVFP4 tensor cores. TP=4 is the right operating point for production serving on this artifact.

**Why `CUDA_HOME=/usr/local/cuda`?** vLLM's Tilelang backend invokes `nvcc` at runtime; the DLAMI bundles a runtime-only CUDA at `/opt/pytorch/cuda`, which lacks the headers Tilelang needs. Pointing to a full toolkit install fixes it.

**Why `VLLM_TEST_FORCE_FP8_MARLIN=1`?** DeepGemm's sm_103a FP8 kernels are partial; the Marlin FP8 path is the safe default until DeepGemm catches up.

## Step 5 — Smoke test

```bash
curl -X POST http://localhost:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP",
    "messages": [{"role":"user","content":"Write a Python function to reverse a string"}],
    "max_tokens": 100
  }'
```

Expect a code response in under a second. If the serve is running with `--speculative-config method=mtp`, the Prometheus endpoint at `http://localhost:8089/metrics` will show non-zero `vllm:spec_decode_num_accepted_tokens_total` after this request.

## Step 6 — Thinking-mode requests

DSV4-Flash supports a thinking mode via `chat_template_kwargs`. Three effective levels: off (default), `"high"`, `"max"`.

```bash
curl -X POST http://localhost:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP",
    "messages": [{"role":"user","content":"Solve: integral of x^2 sin(x) from 0 to pi"}],
    "max_tokens": 32768,
    "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}
  }'
```

OpenAI Python SDK: pass via `extra_body`:

```python
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:8089/v1", api_key="dummy")
resp = await client.chat.completions.create(
    model="canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=32768,
    extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
)
```

The SDK rejects `chat_template_kwargs` as a direct kwarg; `extra_body` forwards it as a top-level body field which vLLM picks up.

## Step 7 — Splitting reasoning from final answer (optional)

If you want `message.reasoning` separated from `message.content` (instead of both concatenated in `content`), start the serve with the DSV4 reasoning parser:

```bash
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --tensor-parallel-size 4 \
    --port 8089 \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --reasoning-parser deepseek_v4
```

Then chat responses will populate `message.reasoning` with the `<think>...</think>` content and `message.content` with the post-thinking final response. Either form scores the same in lm_eval-style harnesses since they extract via regex.

## Troubleshooting

See [VLLM_SETUP_ISSUES.md](VLLM_SETUP_ISSUES.md) for the 14 gotchas encountered while bringing this artifact up, including:

- `sm_103a` vs `sm_100a` (wrong arch silently dies)
- `quantization_config` attribute missing on BF16 configs
- `weight_scale_inv` vs `weight_scale` naming mismatch
- MTP block double-quantization in the draft-model load path
- The 5 PRs and what each fixes

## Verifying the artifact loaded correctly

```bash
# Check MTP weights present
curl -s http://localhost:8089/v1/models | jq

# Check Prometheus is reporting spec-decode metrics
curl -s http://localhost:8089/metrics | grep spec_decode

# Send a chat request, then re-check metrics — counters should increment
```

If `vllm:spec_decode_num_accepted_tokens_total` increments per request, MTP is loaded and working. If it stays 0, the spec-decode path isn't wired up (check `--speculative-config` flag is present).

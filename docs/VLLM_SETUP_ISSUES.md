# vLLM setup issues + the 5 patches needed to serve this artifact

Comprehensive list of every gotcha encountered bringing this artifact up on vLLM mainline, plus the exact diff for each local patch (with corresponding upstream PR where applicable).

## The 5 local patches

Until upstream merges, you'll need these applied to your local vLLM checkout. The first 4 are already filed as PRs against `vllm-project/vllm`; the 5th is being prepared.

### Patch 1 — `bool()` wrap on `is_static_input_scheme` ([PR #43248](https://github.com/vllm-project/vllm/pull/43248))

**File**: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`

**Sites** (5 occurrences as of mainline `d05d52059`):

```python
# Before
is_static_input_scheme = input_quant and not input_quant.dynamic

# After
is_static_input_scheme = bool(input_quant and not input_quant.dynamic)
```

**Why**: `input_quant and not input_quant.dynamic` evaluates to `input_quant` (a `QuantizationArgs` object) when truthy, not `True`. Downstream code uses it as a boolean in `if`-tests, but when stored/serialized it's the object reference. The `bool()` wrap is defensive and idempotent.

**Symptom without patch**: `TypeError: object is not subscriptable` at unrelated downstream sites that try to index into what they expected to be a `bool`.

### Patch 2 — `.get("scale_fmt", "ue8m0")` ([PR #43288](https://github.com/vllm-project/vllm/pull/43288))

**File**: `vllm/models/deepseek_v4/nvidia/model.py:909`

**Original**:
```python
self.scale_fmt = config.quantization_config["scale_fmt"]
```

**Patched**:
```python
self.scale_fmt = config.quantization_config.get("scale_fmt", "ue8m0")
```

**Why**: RedHat's artifact (and ours) don't include `scale_fmt` as an explicit key in `quantization_config` because `ue8m0` is the implicit default for FP8_BLOCK. The original `[...]` indexing crashes with `KeyError`.

### Patch 3 — BF16 load: `getattr(config, "quantization_config", None) or {}` (PR #43288 follow-up)

**File**: `vllm/models/deepseek_v4/nvidia/model.py:909` (same line as patch 2)

**After patch 2**:
```python
self.scale_fmt = config.quantization_config.get("scale_fmt", "ue8m0")
```

**Needs to become** (so BF16 models load too):
```python
_qc = getattr(config, "quantization_config", None) or {}
self.scale_fmt = _qc.get("scale_fmt", "ue8m0")
```

**Why**: BF16 (unquantized) models have **no `quantization_config` attribute at all** on the HF config object — not even an empty dict. Patch 2 alone crashes on `AttributeError: 'DeepseekV4Config' object has no attribute 'quantization_config'` when loading the BF16 reference. The `getattr ... or {}` form handles both:
- BF16 (`quantization_config` attribute missing): `getattr` returns `None`, the `or {}` substitutes empty dict, `.get("scale_fmt", "ue8m0")` returns the default `"ue8m0"`.
- Quantized without explicit `scale_fmt` (like ours and RedHat's): `getattr` returns the dict, `.get` returns the default.

This was discovered when bringing up BF16 DSV4-Flash at TP=8 for the reference baseline benchmark.

### Patch 4 — `weight_scale_inv`-or-`weight_scale` fallback ([PR #43290](https://github.com/vllm-project/vllm/pull/43290))

**File**: `vllm/models/deepseek_v4/attention.py:334`

**Original**:
```python
weight_scale_inv = self.wo_a.weight_scale_inv
```

**Patched**:
```python
weight_scale_inv = getattr(self.wo_a, "weight_scale_inv", None) or self.wo_a.weight_scale
```

**Why**: Different llm-compressor versions emit the attention scale tensor under different attribute names. Our artifact (compressed-tensors `0.15.1a20260515`) uses `weight_scale`; the vLLM DSV4 model hardcodes `weight_scale_inv`. The fallback handles both.

**Symptom without patch**: `AttributeError: 'CompressedLinearMethod' object has no attribute 'weight_scale_inv'`

### Patch 5 — MTP-quant-detect + BF16 `wo_a` fallback ([PR #43319](https://github.com/vllm-project/vllm/pull/43319))

**Files**:
1. `vllm/models/deepseek_v4/nvidia/mtp.py` (`DSV4MultiTokenPredictorLayer.__init__`)
2. `vllm/models/deepseek_v4/attention.py` (forward branch)

**Issue**: vLLM's MTP draft-model construction inherits the main model's `quant_config`. For our artifact, the MTP block is **unquantized BF16 on disk**, but the construction path tries to apply NVFP4-FP8 quantization to it, which crashes the attention forward because the `wo_a` weights are BF16, not FP8.

**Fix in mtp.py**:
```python
def _mtp_block_is_quantized_on_disk(vllm_config):
    """Return True if the safetensors index lists FP8/NVFP4 dtypes under mtp.0.*"""
    import json
    try:
        idx_path = os.path.join(vllm_config.model_config.model, "model.safetensors.index.json")
        idx = json.load(open(idx_path))
        # Walk all keys under mtp.0 — if any are FP8/INT4/packed, MTP IS quantized
        ...  # full impl in the PR
    except Exception:
        return True  # safe default: assume quantized (matches old behavior)

class DSV4MultiTokenPredictorLayer:
    def __init__(self, vllm_config, ...):
        if not _mtp_block_is_quantized_on_disk(vllm_config):
            quant_config = None  # override: MTP is BF16 on disk, don't quantize
        ...
```

**Fix in attention.py forward**:
```python
class DeepseekV4Attention:
    def __init__(self, ...):
        # Cache at init so torch.compile doesn't trace hasattr() at runtime
        self._wo_a_is_unquantized = not isinstance(self.wo_a.quant_method, FP8_BLOCK_methods)
        ...

    def forward(self, ...):
        if current_platform.is_rocm() or self._wo_a_is_unquantized:
            # BF16 wo_a path — route through rocm_inv_rope_einsum (it's BF16-compatible)
            return self._rocm_inv_rope_einsum_path(...)
        # FP8 quantized path (the original)
        ...
```

**Why**: Without this, the MTP draft model crashes on first forward because `self.wo_a` is BF16 but the forward path expects FP8 scales.

## Other gotchas (no patch needed, just configuration)

### 1. `TORCH_CUDA_ARCH_LIST=10.3a` for B300 (NOT `10.0a`)

B300 SXM6 AC has compute capability **10.3 (`sm_103a`)**, not 10.0. Verify with:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
python3 -c "import torch; print(torch.cuda.get_device_capability(0))"
# Expect: (10, 3)
```

Building vLLM with `TORCH_CUDA_ARCH_LIST=10.0a` produces `sm_100a` binaries that silently fail to find kernels at runtime on `sm_103a`. The `a` suffix is non-portable arch-family-specific.

### 2. `CUDA_HOME=/usr/local/cuda` at serve time

The AWS DLAMI bundles a runtime-only CUDA at `/opt/pytorch/cuda` (no headers). vLLM's Tilelang backend invokes `nvcc` at runtime, which fails on missing headers. Point at a full CUDA toolkit install:

```bash
sudo apt install cuda-toolkit-13-0  # or matching your CUDA version
export CUDA_HOME=/usr/local/cuda
```

### 3. `VLLM_TEST_FORCE_FP8_MARLIN=1`

DeepGemm's `sm_103a` FP8 kernels are partial as of 2026-05. The Marlin FP8 path is the safe default until DeepGemm catches up. Set `VLLM_TEST_FORCE_FP8_MARLIN=1` at serve startup.

### 4. Don't use `--system-site-packages`

If your system Python is older than the venv Python (e.g. system 3.12, venv 3.13), inheriting `dist-packages` causes `pyo3_runtime.PanicException` on `cryptography` import. Use a clean venv without `--system-site-packages`.

### 5. NCCL collectives on B300

We hit NCCL+NVLink hangs on the GPTQ Hessian-reduce path during calibration on B300. Symptoms: workers stuck at `cudaStreamSynchronize`, NVRM "Failed to send inband data" in `dmesg`. Workarounds tried (none helped): `NCCL_P2P_DISABLE=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_DEBUG=INFO`.

Resolution for this artifact: used `QuantizationModifier` (RTN-style, weight-only) instead of `GPTQModifier`. Has **zero `dist.*` calls in its main path** — no NCCL collectives. May still apply to other calibration recipes; verify your modifier doesn't hit the same hang.

### 6. Transformers `_keys_to_ignore_on_load_unexpected` strips MTP

`DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` (transformers 5.8.1, still on `main` as of 2026-05-20) silently drops `mtp.*` keys at load time. To preserve MTP during calibration, apply [`patches/modeling_deepseek_v4.py.diff`](../patches/modeling_deepseek_v4.py.diff) which removes that line.

This is **for calibration only** — serving the final artifact on vLLM doesn't go through transformers' load path.

**Upstream fix in flight (sibling workstream)**: [huggingface/transformers#46127](https://github.com/huggingface/transformers/pull/46127) — adds a `DeepseekV4NextNPredictor` class so `mtp.*` keys load into real submodules instead of being filtered out. Currently waiting on a `forward()` implementation + tests per maintainer feedback; the artifact in this repo demonstrates the load-path side works end-to-end (saved weights round-trip through calibration → save → vLLM load → spec-decode serve with measured 81.6% MTP acceptance on AIME 2024). When #46127 lands, this gotcha and the `patches/modeling_deepseek_v4.py.diff` patch become obsolete.

### 7. compressed-tensors `from_accelerate` AttributeError on sharded modules

`offload/dispatch.py:95` raises `AttributeError: 0 is not an nn.Module` when running multi-rank with expert sharding. Workaround: monkey-patch `Observer.synchronize` to short-circuit on rank > 0. Documented in `docs/findings/multirank_observer_sync_hang.md`.

### 8. llm-compressor inference-mode tensor crash on MTP

llm-compressor's calibration loop wraps in `torch.inference_mode()` which crashes on the MTP block's hand-written tensor operations. Filed as llm-compressor [#2745](https://github.com/vllm-project/llm-compressor/issues/2745). Workaround in `scripts/calibration_model.py`: explicit `torch.no_grad()` instead of `inference_mode()`.

### 9. vLLM `(1,)`-shape `global_scale` loader broadcast

vLLM's FusedMoE loader for NVFP4 expects scalar (`shape=()`) `global_scale` tensors, but llm-compressor emits `shape=(1,)`. Squeeze in postprocess: `scripts/squeeze_global_scales.py`. Filed as vLLM [#43297](https://github.com/vllm-project/vllm/issues/43297).

### 10. MTP draft model inherits main model's quant scheme

vLLM's DSV4 MTP draft construction copies `quant_config` from the main model. If main is NVFP4-quantized but MTP is BF16 on disk, the draft model crashes at first forward. Filed as vLLM [#43304](https://github.com/vllm-project/vllm/issues/43304) — our patch 5 above addresses this on the load + forward paths.

### 11. EvalPlus is the right HumanEval harness for chat-mode models

lm_eval's HumanEval scoring is broken on chat-mode-only models (it produces gibberish answers because it can't stop generation correctly). Use EvalPlus instead:

```bash
pip install evalplus
evalplus.codegen humaneval --base-url http://localhost:8089/v1 \
    --model canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP --backend openai
evalplus.evaluate humaneval --samples <output.jsonl>
```

We measured **91.5% pass@1** with EvalPlus vs ~6-20% with lm_eval on the same artifact — the gap is the harness, not the model.

### 12. `--apply_chat_template` does NOT inject thinking-mode kwargs

lm_eval's `--apply_chat_template` applies the chat template but doesn't pass `chat_template_kwargs.thinking=true`. To benchmark thinking-mode, you need a custom harness that calls vLLM's chat endpoint with `extra_body={"chat_template_kwargs": {...}}`. See `scripts/aime_bench.py` in our repo for a reference implementation.

### 13. OpenAI Python SDK rejects `chat_template_kwargs` as direct kwarg

```python
# WRONG — raises "AsyncCompletions.create() got an unexpected keyword argument"
client.chat.completions.create(model=..., chat_template_kwargs={"thinking": True})

# RIGHT — vLLM extension via extra_body
client.chat.completions.create(model=..., extra_body={"chat_template_kwargs": {"thinking": True}})
```

### 14. `max_tokens=16384` is too low for AIME thinking=high

Some AIME problems use up to ~25K tokens of reasoning at `thinking=high`. With `max_tokens=16384`, ~25% of responses truncate; with `max_tokens=65536`, only ~10% truncate (and those are reasoning-loop pathological — bumping further doesn't help). Set `max_tokens=65536` for any thinking-mode reasoning benchmark. Report **both raw pass@1 AND non-truncated pass@1** to disambiguate truncation from reasoning failure.

## Verifying the patches are applied

```bash
cd /data/src/vllm
grep -n "bool(input_quant and not input_quant.dynamic)" \
    vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
# Expect: 5 matches

grep -n 'getattr(config, "quantization_config"' vllm/models/deepseek_v4/nvidia/model.py
# Expect: 1 match (around line 909)

grep -n 'weight_scale_inv.*or.*weight_scale' vllm/models/deepseek_v4/attention.py
# Expect: 1 match (around line 334)

grep -n '_mtp_block_is_quantized_on_disk' vllm/models/deepseek_v4/nvidia/mtp.py
# Expect: 2+ matches
```

If any are missing, the corresponding load or forward will crash.

## Upstream PR status

| PR | Title | Status |
|---|---|---|
| [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` | filed 2026-05-21, open |
| [#43288](https://github.com/vllm-project/vllm/pull/43288) | `.get("scale_fmt", "ue8m0")` defensive | filed 2026-05-21, open |
| [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback | filed 2026-05-21, open |
| [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP-quant-detect + BF16 `wo_a` fallback | filed 2026-05-21, open |
| [#43297](https://github.com/vllm-project/vllm/issues/43297) | `(1,)`-shape `global_scale` loader broadcast (issue) | filed 2026-05-21, open |
| [#43304](https://github.com/vllm-project/vllm/issues/43304) | MTP draft inherits main quant scheme (issue) | filed 2026-05-21, open |
| [llm-compressor #2745](https://github.com/vllm-project/llm-compressor/issues/2745) | MTP inference-mode crash | filed earlier, open |

When these merge upstream, you can drop the corresponding local patches.

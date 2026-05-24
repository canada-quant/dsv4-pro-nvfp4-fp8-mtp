# vLLM setup issues + the 5 patches needed to serve this artifact

Comprehensive list of every gotcha encountered bringing this artifact up on vLLM mainline with **MTP at 91.45%**, plus the exact diff for each local patch (with corresponding upstream PR) and the calibration/environment quirks that don't require a patch.

## V4-Pro-specific findings

These items are V4-Pro-specific and not in the V4-Flash predecessor's setup-issues doc:

1. **The MTP block must be byte-passthrough from native.** Any transformation of `mtp.0.*` weights (NVFP4 transcode of experts, BF16 dequant of attn / e_proj / h_proj) drops MTP acceptance from ~91% to ~3%. NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe excludes the entire MTP layer (`model.layers.61*` for V4-Pro) from quantization. v12 follows this; v0.2 / v0.3 / v0.4 did not. Full debug chain: [`findings/v12_nvfp4_mtp_working_2026_05_24.md`](findings/v12_nvfp4_mtp_working_2026_05_24.md).

2. **The load-bearing vLLM bug**: `_mtp_block_is_quantized_on_disk` was missing `".scale"` from its `quant_suffixes` list. DSV4 native checkpoints store FP8 block scales with a raw `.scale` suffix (e.g. `mtp.0.attn.wq_a.scale`), which the runtime renamer converts to `.weight_scale_inv`. But the detector runs BEFORE the rename, scans the raw on-disk keys, sees no matches, returns `False` → MTP block built with `quant_config=None` → `e_proj`/`h_proj` registers `weight` only (no `weight_scale_inv`) → loader raises `KeyError: 'model.layers.61.e_proj.weight_scale_inv'`. **Fix is a single line in `quant_suffixes`**. Patch [#43319](https://github.com/vllm-project/vllm/pull/43319) (this repo's `patches/patch_43319_mtp_quant_detect.diff`).

3. **Per-layer MoE quant dispatch for hybrid NVFP4-trunk + MXFP4-MTP**. v12 has NVFP4 trunk experts but MXFP4 MTP experts. `DSV4FP8Config.get_quant_method` has a single global `moe_quant_algo` branch — when it's `"NVFP4"`, all MoE layers go through `ModelOptNvFp4FusedMoE` including MTP. The MTP MoE then tries to load NVFP4-packed params, finds the on-disk MXFP4 layout instead, crashes. Patch: detect MTP layer by prefix and force `Mxfp4MoEMethod` regardless of global `moe_quant_algo`. Local patch `patches/patch_v12b_per_layer_moe_routing.diff`; upstream PR pending.

4. **flashinfer 0.6.11.post2 silent worker crash**. `pip install -e .` of vLLM mainline pulls in `flashinfer-cubin==0.6.11.post2` which silently crashes workers during model construction (no stack trace, just `RuntimeError: cancelled` and `WorkerProc initialization failed due to an exception in a background process`). Pin to 0.6.8.post1: `pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'`. Likely ABI regression between 0.6.8 and 0.6.11.

5. **`deep_gemm_mega_moe` does not dispatch NVFP4 in current mainline**. Loading our NVFP4 artifact with `--moe-backend deep_gemm_mega_moe` raises `KeyError: 'layers.0.ffn.experts.w13_input_scale'`. The mega-kernel path expects fused-name MoE parameters (one tensor for all experts); NVFP4 ModelOpt layout uses per-expert names. Use `--moe-backend flashinfer_trtllm` for NVFP4. Filed as [vLLM #43454](https://github.com/vllm-project/vllm/issues/43454) + fix PR [#43467](https://github.com/vllm-project/vllm/pull/43467).

6. **`--moe-backend flashinfer_trtllm` is the only path for NVFP4 today**. On native MXFP4, `deep_gemm_mega_moe` is ~4% faster than `flashinfer_trtllm`. The honest NVFP4-vs-MXFP4 throughput comparison is each format on its preferred backend.

## The 5 local patches

Until upstream merges, you'll need these applied to your vLLM checkout. All are filed as PRs against `vllm-project/vllm`. PR #42209 (NVFP4 MoE support for DSV4) merged 2026-05-22 and is now in mainline — no cherry-pick needed.

### Patch 1 — `bool()` wrap on `is_static_input_scheme` ([PR #43248](https://github.com/vllm-project/vllm/pull/43248))

**File**: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`

**Sites** (5 occurrences as of mainline `d05d52059`):

```python
# Before
is_static_input_scheme = input_quant and not input_quant.dynamic
# After
is_static_input_scheme = bool(input_quant and not input_quant.dynamic)
```

**Why**: `input_quant and not input_quant.dynamic` evaluates to `input_quant` (a `QuantizationArgs` object) when truthy, not `True`. Downstream code expects `bool`. `bool()` wrap is defensive and idempotent.

**Symptom without patch**: `TypeError: object is not subscriptable` at unrelated downstream sites.

### Patch 2 — `.get("scale_fmt", "ue8m0")` ([PR #43288](https://github.com/vllm-project/vllm/pull/43288))

**File**: `vllm/models/deepseek_v4/nvidia/model.py:909`

**Original**:
```python
self.scale_fmt = config.quantization_config["scale_fmt"]
```

**Patched**:
```python
_qc = getattr(config, "quantization_config", None) or {}
self.scale_fmt = _qc.get("scale_fmt", "ue8m0")
```

**Why**: Some quantized artifacts (and ours) don't include `scale_fmt` as an explicit key. BF16 reference models have no `quantization_config` attribute at all. Original `[...]` indexing crashes with `KeyError` / `AttributeError`.

### Patch 3 — `weight_scale_inv`-or-`weight_scale` fallback ([PR #43290](https://github.com/vllm-project/vllm/pull/43290))

**File**: `vllm/models/deepseek_v4/attention.py:334`

**Original**:
```python
weight_scale_inv = self.wo_a.weight_scale_inv
```

**Patched**:
```python
weight_scale_inv = getattr(self.wo_a, "weight_scale_inv", None) or self.wo_a.weight_scale
```

**Why**: Different llm-compressor versions emit the attention scale tensor under different attribute names. Some artifacts (compressed-tensors `0.15.1a20260515`) use `weight_scale`; the vLLM DSV4 model hardcodes `weight_scale_inv`. Fallback handles both.

### Patch 4 — MTP loader: `.scale` detector + candidate-list ([PR #43319](https://github.com/vllm-project/vllm/pull/43319))

**THE LOAD-BEARING PATCH for MTP.**

**Files**:
1. `vllm/models/deepseek_v4/nvidia/mtp.py` (`_mtp_block_is_quantized_on_disk` detector + `.scale` candidate-list resolution in `load_weights`)

**Fix in detector** (the actual root cause of every "3% MTP" measurement):

```python
quant_suffixes = (
    ".scale",   # ← THIS ONE. DSV4 native FP8 block scale convention.
    ".weight_scale",
    ".weight_scale_inv",
    ".weight_packed",
    ".weight_global_scale",
    ".input_global_scale",
    ".weight_zero_point",
)
```

Without `.scale` in the list, the detector scanned the raw on-disk `mtp.*` keys, found nothing matching `.weight_scale*`, returned False → MTP block built unquantized → `e_proj.weight_scale_inv` never registered → `KeyError` at load → all earlier attempts worked around by **dequantizing mtp.0 to BF16**, which broke MTP acceptance entirely.

**Fix in `load_weights`**: candidate-list scale resolution that tries both `.weight_scale_inv` AND `.weight_scale` suffixes for non-expert scales, with optional `.mtp_block.` prefix variants. Handles the spec layer's mtp_block name rewrite.

### Patch 5 — DSV4 MegaMoE early-fail for NVFP4 ([PR #43467](https://github.com/vllm-project/vllm/pull/43467))

**File**: `vllm/models/deepseek_v4/nvidia/model.py` (`DeepseekV4MoE.__init__`)

**Adds**:
```python
if self.use_mega_moe:
    _qc = getattr(config, "quantization_config", None) or {}
    _algo = (_qc.get("moe_quant_algo") if isinstance(_qc, dict) else None)
    if isinstance(_algo, str) and _algo.upper() == "NVFP4":
        raise NotImplementedError(
            "DeepSeek V4 MegaMoE does not currently dispatch NVFP4 expert "
            "layout. Use --moe-backend flashinfer_trtllm for NVFP4 MoE "
            "artifacts on Blackwell."
        )
```

**Why**: Clear error instead of `KeyError: 'layers.0.ffn.experts.w13_input_scale'` when users try `--moe-backend deep_gemm_mega_moe` on an NVFP4 artifact. Closes [vLLM #43454](https://github.com/vllm-project/vllm/issues/43454).

### Patch v12b — Per-layer MoE routing (local, upstream PR pending)

**File**: `vllm/models/deepseek_v4/quant_config.py` (`DSV4FP8Config.get_quant_method`)

**Adds**: prefix-based MTP layer detection. When prefix matches an MTP layer (index ≥ `num_hidden_layers`), force `Mxfp4MoEMethod` regardless of global `moe_quant_algo='NVFP4'`. Trunk MoE dispatch unchanged.

```python
import re as _re
_is_mtp_layer = False
try:
    from vllm.config import get_current_vllm_config
    hf_cfg = get_current_vllm_config().model_config.hf_config
    n_hidden = int(getattr(hf_cfg, 'num_hidden_layers', 10**9))
    _m = _re.search(r'\.layers\.(\d+)\.', prefix or '')
    _is_mtp_layer = bool(_m and int(_m.group(1)) >= n_hidden)
except Exception:
    pass
if self.expert_dtype == "fp4":
    if self.moe_quant_algo == "NVFP4" and not _is_mtp_layer:
        return ModelOptNvFp4FusedMoE(...)
    # MTP layer OR moe_quant_algo!='NVFP4' → MXFP4 path
    return Mxfp4MoEMethod(layer.moe_config)
```

**Why**: A hybrid NVFP4-trunk + MXFP4-MTP artifact (which is what v12 ships, following NVIDIA's V3.2-NVFP4 recipe) needs per-layer MoE dispatch. The single global `moe_quant_algo` is too restrictive.

Saved as [`patches/patch_v12b_per_layer_moe_routing.diff`](../patches/patch_v12b_per_layer_moe_routing.diff).

## Other gotchas (no patch needed, just configuration)

### 1. `TORCH_CUDA_ARCH_LIST=10.3a` for B300 (NOT `10.0a`)

B300 SXM6 AC has compute capability **10.3 (`sm_103a`)**, not 10.0. Verify with:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
python3 -c "import torch; print(torch.cuda.get_device_capability(0))"
# Expect: (10, 3)
```

Building vLLM with `TORCH_CUDA_ARCH_LIST=10.0a` produces `sm_100a` binaries that fail at runtime on `sm_103a`. The `a` suffix is non-portable arch-family-specific.

### 2. `CUDA_HOME=/usr/local/cuda` at serve time

The AWS DLAMI bundles a runtime-only CUDA at `/opt/pytorch/cuda` (no headers). vLLM's Tilelang backend invokes `nvcc` at runtime, which fails on missing headers. Point at a full CUDA toolkit install:

```bash
sudo apt install cuda-toolkit-13-0
export CUDA_HOME=/usr/local/cuda
```

### 3. `ninja` system-wide (not just venv)

vLLM worker subprocesses don't inherit the venv PATH. If `ninja` is only in `/opt/pytorch/bin`, the workers' JIT compile of flashinfer's FP4 MoE module fails with `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`. Install system ninja:

```bash
sudo apt install ninja-build
```

### 4. Pin flashinfer to 0.6.8.post1

`flashinfer-cubin==0.6.11.post2` silently crashes vLLM workers (see V4-Pro finding #4 above). After the install script, force:

```bash
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'
```

### 5. Don't use `--system-site-packages`

If your system Python is older than the venv Python (e.g. system 3.12, venv 3.13), inheriting `dist-packages` causes `pyo3_runtime.PanicException` on `cryptography` import. Use a clean venv.

### 6. Transformers `_keys_to_ignore_on_load_unexpected` strips MTP

`DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` (transformers 5.8.1) silently drops `mtp.*` keys at load time. For calibration only — apply [`patches/modeling_deepseek_v4.py.diff`](../patches/modeling_deepseek_v4.py.diff). Serving on vLLM doesn't hit this. Upstream fix in flight at [huggingface/transformers#46127](https://github.com/huggingface/transformers/pull/46127).

### 7. compressed-tensors `from_accelerate` AttributeError on sharded modules

Calibration-only issue. Workaround in predecessor repo: monkey-patch `Observer.synchronize` to short-circuit on rank > 0. Not exercised in V4-Pro's byte-level conversion.

### 8. llm-compressor inference-mode tensor crash on MTP

llm-compressor's calibration loop wraps in `torch.inference_mode()` which crashes on the MTP block's hand-written tensor operations. Filed as llm-compressor [#2745](https://github.com/vllm-project/llm-compressor/issues/2745). Workaround: explicit `torch.no_grad()` instead of `inference_mode()`. Not exercised in V4-Pro's format-conversion pipeline.

### 9. vLLM `(1,)`-shape `global_scale` loader broadcast

vLLM's FusedMoE loader for NVFP4 expects scalar (`shape=()`) `global_scale` tensors, but llm-compressor emits `shape=(1,)`. Filed as vLLM [#43297](https://github.com/vllm-project/vllm/issues/43297). Workaround: squeeze in postprocess. Not exercised in V4-Pro's format-conversion pipeline (our conversion writes shape `(1,)` and vLLM handles it after our local patches; if you observe the issue, use `scripts/squeeze_global_scales.py`).

### 10. EvalPlus is the right HumanEval harness for chat-mode models

lm_eval's HumanEval scoring is broken on chat-mode-only models. Use EvalPlus instead:

```bash
pip install evalplus
evalplus.codegen humaneval --base-url http://localhost:8089/v1 \
    --model canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP --backend openai
evalplus.evaluate humaneval --samples <output.jsonl>
```

We measured **95.1% pass@1** with EvalPlus vs ~6-20% with lm_eval — the gap is the harness, not the model.

### 11. `--apply_chat_template` does NOT inject thinking-mode kwargs

lm_eval applies the chat template but doesn't pass `chat_template_kwargs.thinking=true`. For thinking-mode benchmarks, write a custom harness that calls vLLM's chat endpoint with `extra_body={"chat_template_kwargs": {...}}`. See `scripts/aime_bench.py` for a reference.

### 12. OpenAI Python SDK rejects `chat_template_kwargs` as direct kwarg

```python
# WRONG — raises "AsyncCompletions.create() got an unexpected keyword argument"
client.chat.completions.create(model=..., chat_template_kwargs={"thinking": True})

# RIGHT — vLLM extension via extra_body
client.chat.completions.create(model=..., extra_body={"chat_template_kwargs": {"thinking": True}})
```

### 13. `max_tokens=16384` is too low for AIME thinking=high

Some AIME problems use up to ~25K tokens of reasoning at `thinking=high`. With `max_tokens=16384`, ~25% of responses truncate; with `max_tokens=65536`, only ~10% truncate. Set `max_tokens=65536` for any thinking-mode reasoning benchmark. Report **both raw pass@1 AND non-truncated pass@1**.

### 14. Cold start is ~15 min — that's expected

Cuda-graph capture + flashinfer FP4 MoE JIT + torch.compile dynamo + AOT autograd takes ~12-15 minutes from process spawn to "Application startup complete". During the wait, the engine logs `No available shared memory broadcast block found in 60 seconds` warnings — these are not errors. If 30 minutes passes with no further progress, check for actual worker crashes via `grep ERROR /var/log/vllm.log`.

## Verifying the patches are applied

```bash
cd /opt/dlami/nvme/src/vllm

grep -n "bool(input_quant and not input_quant.dynamic)" \
    vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
# Expect: 5 matches

grep -n 'getattr(config, "quantization_config"' vllm/models/deepseek_v4/nvidia/model.py
# Expect: 1 match (around line 909)

grep -n 'weight_scale_inv.*or.*weight_scale\|getattr.*weight_scale_inv' vllm/models/deepseek_v4/attention.py
# Expect: 1+ match

grep -n '"\.scale",' vllm/models/deepseek_v4/nvidia/mtp.py
# Expect: 1 match in _mtp_block_is_quantized_on_disk's quant_suffixes tuple

grep -n '_is_mtp_layer' vllm/models/deepseek_v4/quant_config.py
# Expect: 3+ matches (the v12b patch)
```

If any are missing, MTP load will fail or fall back to ~3% acceptance.

## Upstream PR status

| PR | Title | Status |
|---|---|---|
| [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` | open |
| [#43288](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` defensive `.get()` | open |
| [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback | open |
| [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP loader: **`.scale` detector** + candidate-list scale resolution | open |
| [#43467](https://github.com/vllm-project/vllm/pull/43467) | DSV4 MegaMoE early-fail for NVFP4 | open |
| v12b per-layer MoE routing | DSV4FP8Config: MTP-layer dispatch override | local; upstream PR pending |
| [#43297](https://github.com/vllm-project/vllm/issues/43297) | `(1,)`-shape `global_scale` (issue) | open |

When these merge upstream, drop the corresponding local patches from the install script.

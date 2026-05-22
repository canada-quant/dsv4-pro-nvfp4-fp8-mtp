# Phase 0 close-out — 2026-05-21

## Status

| Step | Status | Notes |
|---|---|---|
| 1 — hardware verify | ✅ | 8× B300 sm_103a, 275 GB HBM each, /opt/dlami/nvme = 23 TB free |
| 2 — V4-Flash patch diff | ✅ | All 4 vLLM patches still apply line-for-line; 2 non-vLLM patches dropped |
| 3 — clean rebuild | ✅ | canada-quant-v4pro-nvfp4 branch on upstream/main@39910f2b25 + 4 named patch commits; vllm 0.21.1rc1.dev196+g0a78a466d |
| 4 — stage native MXFP4 | ✅ | 92 files / 806 GiB at /opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp/ via hf_xet (4m15s, 3.3 GB/s sustained) |
| 5 — native serve smoke | ✅ | Without MTP — vllm serve up at port 8089 in 6m06s; prompt "Write one short sentence about a green apple" → coherent response: "A green apple sits crisp and bright, promising a sharp, refreshing bite." |
| 6 — MTP counter | ⚠️ deferred to Phase 3 postprocess | Upstream vLLM bug surfaced (details below); workaround built into our conversion pipeline |
| 7 — per-token active params | ✅ | **49.60 B** — matches NVIDIA's published "49 B" within rounding |

Floor established: V4-Pro serves natively on our build for the main forward.

## What worked the first time

- Hardware: unchanged from V4-Flash, no surprises.
- vLLM patch survival: the supervisor framing expected the V4-Pro
  subpackage to supersede most patches; in reality all 4 still apply
  line-for-line because the V4-Pro PRs landing in May haven't touched
  those specific call sites. Keep all 4.
- Download: HF Xet via authenticated token clocked **3.3 GB/s
  sustained** to /opt/dlami/nvme. 4m15s for 806 GiB.

## What broke and required iteration

### Build-time

- **setuptools-rust missing** — vLLM main HEAD now includes a Rust
  frontend (PR #43283). Without `setuptools-rust>=1.9.0` + rustup,
  `pip install -e .` fails at metadata generation. Fix: install both
  (added to `scripts/install_vllm_with_patches.sh`).
- **CMake generator mismatch** — leaving the previous build's `.deps/`
  in place (Ninja-generated) caused the new build to refuse with
  "Does not match the generator used previously". Fix: `rm -rf
  .deps build` before re-running pip install.

### Run-time

- **Patch #43290 used `or`-on-tensor at attention.py:346.** The
  original sed-applied form was
  `wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None) or self.wo_a.weight_scale`
  which triggers `torch._dynamo.exc.Unsupported: Data-dependent
  branching` even with `--enforce-eager` (dynamo capture runs
  regardless of cuda-graph mode). Revised to cache
  `_wo_a_uses_inv_scale = hasattr(...)` at init and branch on the
  Python bool at forward time (resolved at trace time).
- **`ninja` not on worker subprocess PATH.** vLLM workers JIT-compile
  DeepGEMM kernels at runtime; ninja is installed at
  `/data/venv-serve/bin/ninja` but workers spawned by nohup didn't
  inherit that path. Fix: `export PATH=/data/venv-serve/bin:$PATH`
  before `vllm serve`.

## The MTP serve issue (deferred to Phase 3)

**Root cause**: when the V4-Pro MTP block is constructed,
`DeepSeekV4MultiTokenPredictorLayer.__init__` builds `e_proj` and
`h_proj` as `ReplicatedLinear(quant_config=quant_config)`. Even
though `quant_config` is `DeepseekV4FP8Config` (FP8 block-quant), the
resulting ReplicatedLinear ends up with **only `.weight`** registered
in `params_dict` — no `.weight_scale_inv` or `.weight_scale` companion.

Verified by adding a diagnostic print at the failing
`params_dict[name]` lookup:

```
PATCH-DIAG: looked up 'model.layers.61.e_proj.weight_scale_inv'; not in params_dict
PATCH-DIAG: keys matching 'model.layers.61.e_proj':
PATCH-DIAG:   model.layers.61.e_proj.weight
```

The actual ReplicatedLinear-with-Fp8Config code path silently fails
to register the scale parameter. This is a vLLM upstream bug,
distinct from PR #43319 (which addresses _whether_ MTP gets a
quant_config). Even with PR #43319 routing the quant_config in,
ReplicatedLinear doesn't honor it for scale-param registration.

Our scale-name fallback patch (committed as the candidate-list
revision of PR #43319) still passes through to the primary name
when no candidates are in params_dict — so the underlying issue is
not in the loader but in the model construction.

### Why deferring to Phase 3 is correct, not a punt

The artifact we ship is the OUTPUT of our Phase 3 conversion
pipeline, not the native source. In Phase 3 we control the on-disk
layout of MTP `e_proj` and `h_proj`. By dequantizing FP8 + E8M0
scale to BF16 during postprocess, the converted artifact's MTP
`e_proj` / `h_proj` become BF16 — matching what vLLM's
unquantized ReplicatedLinear actually expects.

Cost: ~51 M × 2 (e_proj + h_proj) × 2 bytes = 200 MB added to the
artifact size (out of 800-900 GB). Negligible. Precision cost:
near-zero, since these are linear layers operating on hidden states
that themselves carry FP8/BF16 precision.

If the upstream ReplicatedLinear+Fp8 bug gets fixed before we ship,
we can revisit and keep `e_proj`/`h_proj` as native FP8. But the
artifact doesn't gate on that fix.

This issue should be filed upstream as a follow-up to PR #43319
documenting the construction-side gap.

## Next phases

Phase 0 is closed for the work it can produce without MTP. Phase 1
(NVFP4 vs MXFP4 microbenchmark on B300 V4-Pro expert shapes) and
Phase 3 (full-model conversion with MTP `e_proj`/`h_proj` BF16
dequant) run next. Phase 2 conversion math is already validated
across 192 sampled tensors.

## Receipts

- vLLM version: `0.21.1rc1.dev196+g0a78a466d` (editable at
  `/data/src/vllm` branch `canada-quant-v4pro-nvfp4`)
- Native source: `/opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp/`
  (806 GiB, 64 safetensors shards + inference/* + config.json + tokenizer)
- Working serve command (no MTP):
  ```bash
  export PATH=/data/venv-serve/bin:$HOME/.cargo/bin:$PATH
  export CUDA_HOME=/usr/local/cuda
  vllm serve /opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp \
    --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
    --enable-expert-parallel --tensor-parallel-size 8 \
    --moe-backend deep_gemm_mega_moe --enforce-eager \
    --port 8089
  ```
- Smoke prompt response logged at
  `system_fingerprint: vllm-0.21.1rc1.dev196+g0a78a466d-tp8-ep-58e7f98c`,
  16 completion tokens, finish_reason: stop, sane output.

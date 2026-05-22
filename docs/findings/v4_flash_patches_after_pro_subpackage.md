# V4-Flash patches: disposition for V4-Pro work — 2026-05-21

**Phase 0 step 2 deliverable. Report-back-to-supervisor checkpoint.**

## TL;DR

The supervision brief expected the V4-Pro subpackage rewrite to supersede
most of the V4-Flash patches. **It didn't.** All 4 vLLM patches target
lines that are still un-fixed on current `vllm-project/vllm@main`, and
all 4 are still OPEN with none merged. Net disposition: **keep 4 vLLM
patches, drop 2 non-vLLM patches** (the two that targeted llm-compressor
and transformers are calibration-specific and irrelevant to a format-
conversion recipe).

Equally important and unanticipated: the existing box state
(`/data/src/vllm` editable + `/data/venv-serve`) **already has the
V4-Pro subpackage and all 4 patches applied**. The venv-rebuild step
in the plan may be unnecessary — see "implications" at the end.

## Box state snapshot (2026-05-21)

Hardware (verified):
- 8× B300 SXM6 AC, 275040 MiB each, compute_cap 10.3
- `/opt/dlami/nvme`: 28 TB / 23 TB free (the `/scratch` equivalent on
  this DLAMI — the `/scratch/...` paths in V4-Flash CLAUDE.md should
  resolve here)
- `/data`: 295 GB / 113 GB free
- `/data/venv-serve/bin/vllm --version`: `0.21.1rc1.dev164+gd05d52059.d20260521`

vLLM source (`/data/src/vllm`):
- Editable install location (referenced from `/data/venv-serve` via
  `pip install -e .`)
- Remotes: `upstream=vllm-project/vllm`, `origin=jasl/vllm` (a fork
  used during V4-Flash work), `canada-quant=canada-quant/vllm`
- Current branch: `pr-42209`
- Current HEAD: `d05d52059f` — "Merge branch 'main' into nvfp4_dsv4"
  - 32 commits behind `vllm-project/vllm@main` (currently
    `39910f2b25`, 2026-05-22T00:21Z)
  - 4 commits ahead (the nvfp4_dsv4 branch work that's on the
    canada-quant fork)
- Working tree has uncommitted edits to 4 files (the patches —
  see disposition below)
- Untracked: `vllm/models/deepseek_v4/nvidia/model.py.bak_bf16`
  (V4-Flash agent's BF16-experiment leftover; safe to delete)

V4-Pro subpackage present at d05d52059 and at current main (identical
file listing):
```
vllm/models/deepseek_v4/
  __init__.py
  amd/
  attention.py
  common/
  compressor.py
  nvidia/
  quant_config.py
```

The `nvfp4_dsv4` branch is on `canada-quant/vllm`, **not** on
`vllm-project/vllm`. Upstream vLLM does not have a `nvfp4_dsv4` branch
(searched `gh api repos/vllm-project/vllm/branches?per_page=100`).

## Per-patch disposition

### vLLM patches (4)

**Patch #43248 — `bool()` wrap on `is_static_input_scheme`**

| | |
|---|---|
| Target | `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` |
| Upstream state | OPEN, last activity in window |
| Lines un-fixed on current main | 2 sites: `is_static_input_scheme = input_quant and not input_quant.dynamic` (current main lines ~676, 697) |
| Reason still needed | Generic compressed-tensors codepath, not V4-Pro-specific. Affects any FP8/INT4 quant route through CompressedTensorsW8A16Fp8 / W4A8Int. V4-Pro's FP8 attention path passes through compressed-tensors → this matters at load. |
| **Decision** | **Keep.** Apply on next rebuild against newer main. |

**Patch #43288 — `scale_fmt` defensive `.get()` + BF16 `getattr` wrap**

| | |
|---|---|
| Target | `vllm/models/deepseek_v4/nvidia/model.py:909` |
| Upstream state | OPEN |
| Line un-fixed on current main | `self.scale_fmt = config.quantization_config["scale_fmt"]` (hard subscript) |
| Reason still needed | V4-Pro's HF config DOES include `scale_fmt: ue8m0`, so the `.get()` form is no-op for the native release. The BF16 getattr wrap matters if anyone serves a BF16 V4-Pro (no `quantization_config` attr) — relevant to our converted NVFP4 artifact's config.json if we structure it as a top-level quant section vs nested. **Keep for defensive value, prevents KeyError on edge cases.** |
| **Decision** | **Keep.** |

**Patch #43290 — `weight_scale_inv`-or-`weight_scale` fallback**

| | |
|---|---|
| Target | `vllm/models/deepseek_v4/attention.py:334` |
| Upstream state | OPEN |
| Line un-fixed on current main | `wo_a_scale = self.wo_a.weight_scale_inv` (no fallback) |
| Reason still needed | V4-Pro's attention stores FP8 weights with `weight_scale` (not `weight_scale_inv`). Without this patch, the `wo_a` quant path raises AttributeError at serve time. **This is load-blocking for V4-Pro.** |
| **Decision** | **Keep — required for V4-Pro to serve.** |

**Patch #43319 — MTP-quant-detect from safetensors header + BF16 wo_a fallback**

| | |
|---|---|
| Target | `vllm/models/deepseek_v4/nvidia/mtp.py` |
| Upstream state | OPEN |
| Line un-fixed on current main | No `_mtp_block_is_quantized_on_disk()` function; the loader uses `quant_config = vllm_config.quant_config` directly without disk-detection |
| Reason still needed | V4-Pro's MTP IS quantized on disk (MXFP4 experts). Without this patch, the MTP loader path may incorrectly assume BF16 and KeyError. Our converted NVFP4 artifact's MTP block is also fully quantized — needs the same detection logic. **Required for V4-Pro MTP serving.** |
| **Decision** | **Keep — required for V4-Pro MTP serving.** |

### Non-vLLM patches (2)

**Patch — `helpers.py.diff` against llm-compressor**

| | |
|---|---|
| Target | `src/llmcompressor/pipelines/sequential/helpers.py:208` |
| Purpose | Calibration tracer: handle transformers Cache subclasses as fx-encodable args, so V4-Flash sequential calibration could trace through the past_key_values path |
| Relevance to V4-Pro | NONE. V4-Pro recipe is format conversion (MXFP4 → NVFP4), not gradient-based calibration. No sequential pipeline traversal needed. |
| **Decision** | **Drop.** |

**Patch — `modeling_deepseek_v4.py.diff` against transformers (≈ PR #46127)**

| | |
|---|---|
| Target | `transformers/models/deepseek_v4/modeling_deepseek_v4.py` |
| Purpose | (a) Remove `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` so MTP keys aren't silently dropped on `from_pretrained`. (b) Stop auto-creating `DynamicCache` when `past_key_values is None` (calibration-specific issue). |
| Upstream state | OPEN at transformers#46127 |
| Relevance to V4-Pro | NONE for serving — vLLM's V4-Pro subpackage owns the loader; stock transformers' V4 model class is not in our forward path. Relevant only if we ever load V4-Pro through transformers for evals or debugging, which the plan does not call for. |
| **Decision** | **Drop.** |

## What the working tree currently has applied

```
$ git -C /data/src/vllm status
On branch pr-42209
Changes to be committed:
  modified:   vllm/models/deepseek_v4/nvidia/mtp.py            (#43319)

Changes not staged for commit:
  modified:   vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py  (#43248)
  modified:   vllm/models/deepseek_v4/attention.py             (#43290)
  modified:   vllm/models/deepseek_v4/nvidia/model.py          (#43288)

Untracked:
  vllm/models/deepseek_v4/nvidia/model.py.bak_bf16             (V4-Flash leftover)
```

So all 4 vLLM patches are present in the local checkout, applied as
uncommitted edits on `pr-42209` (which itself sits on the merge commit
`d05d52059` that integrated upstream main into our `nvfp4_dsv4` branch).

## Implications for Phase 0 step 3 (venv rebuild)

Two viable paths, with non-obvious tradeoffs. **Surfacing for
supervisor decision, not picking.**

### Path A — keep the existing venv, just commit the patches cleanly

What's needed:
1. Commit the 4 patch edits on `pr-42209` as a clean named commit
   ("Apply 4 V4-Flash patches: #43248, #43288, #43290, #43319")
2. Delete the `nvidia/model.py.bak_bf16` stray
3. Verify the existing venv still imports cleanly (it should — it was
   built against this exact tree)

Pros:
- No rebuild time (saves 10–20 min)
- Known-good state from V4-Flash ship day
- All 4 patches already in place

Cons:
- 32 commits behind upstream main → missing whatever V4-Pro
  bugfixes / perf work has landed in the last day
- `pr-42209` is a weird branch name to be on; suggests we were
  tracking some specific PR (which PR? we don't know without
  investigating)
- The merge state ("Merge main into nvfp4_dsv4") may have left
  conflict-resolution artifacts we haven't audited

### Path B — fresh rebuild against current upstream main

What's needed:
1. Fetch upstream → checkout `upstream/main` (or pin to a specific
   recent commit like `39910f2b25`)
2. Reapply the 4 vLLM patches (sed for #43248, #43288, #43290;
   cherry-pick from canada-quant fork for #43319)
3. Rebuild: `TORCH_CUDA_ARCH_LIST=10.3a pip install -e .` (10–20 min
   on this box)
4. Verify import + V4-Flash smoke before V4-Pro work

Pros:
- Clean, predictable state
- Includes any V4-Pro fixes that landed in the 32-commit gap
- Documented commit SHA for the build going forward

Cons:
- 10–20 min rebuild
- Risk: one of the 32 newer commits has refactored a code path
  one of the patches targets, requiring rebase (likely on
  attention.py given recent activity per PR list)

## Recommendation (your call, not the agent's)

Path B is cleaner. Path A is faster. The argument for Path B is that
the 32-commit gap is small (1 day) but not zero, and a known commit
SHA in the install script beats "whatever d05d52059 happened to be."
The argument for Path A is that we have a working state with all 4
patches in place; rebuilding risks regression we don't need.

If you pick Path A, the next concrete step is `git add -A &&
git commit -m "Apply 4 V4-Flash patches for V4-Pro continuity"` and
the `model.py.bak_bf16` deletion, then we move on to Phase 0 step 4
(stage native MXFP4 source).

If you pick Path B, we rebuild against `vllm-project/vllm@39910f2b25`
(current main HEAD) and update `scripts/install_vllm_with_patches.sh`
in this repo to pin to that SHA + the 4 patch applications.

## Receipts

- `gh pr view 43248 43288 43290 43319 --repo vllm-project/vllm`:
  all 4 OPEN, none merged, all created/updated within the last week
- `gh pr view 46127 --repo huggingface/transformers`: OPEN
- `ssh ... 'cd /data/src/vllm && git log/status'`: branch state above
- `gh api repos/vllm-project/vllm/contents/vllm/models/deepseek_v4?ref=d05d52059`:
  V4-Pro subpackage present at our venv commit
- `gh api repos/vllm-project/vllm/contents/vllm/models/deepseek_v4?ref=main`:
  V4-Pro subpackage present at current main (same file listing)
- `gh api repos/vllm-project/vllm/compare/main...d05d52059`:
  `behind_by: 32, ahead_by: 4, status: diverged`
- Current vLLM main `compressed_tensors.py` lines ~676/697: still un-wrapped
- Current vLLM main `vllm/models/deepseek_v4/nvidia/model.py:909`:
  `self.scale_fmt = config.quantization_config["scale_fmt"]` (unchanged)
- Current vLLM main `vllm/models/deepseek_v4/attention.py:334`:
  `wo_a_scale = self.wo_a.weight_scale_inv` (no fallback)
- Current vLLM main `vllm/models/deepseek_v4/nvidia/mtp.py`:
  no `_mtp_block_is_quantized_on_disk` function

## STOPPING HERE per Phase 0 execution discipline (resolved)

Plan said "report back after step 2, before venv rebuild in step 3."
Supervisor chose **Path B** — clean rebuild against upstream main with
the 4 patches as named commits on a `canada-quant-v4pro-nvfp4` branch.
Step 3 is now in progress; see appendix below.

---

# Appendix — Path B execution log (step 3 in progress)

## Resolved: PR #42209 ("weird branch name" mystery)

The previous local branch `pr-42209` was tracking
[vllm-project/vllm#42209](https://github.com/vllm-project/vllm/pull/42209):
**"Add NVFP4 MOE support for Deepseek V4."** by `sychen52` (NVIDIA),
OPEN with `ready`, `ci/build`, `deepseek`, `nvidia` labels, last
updated 2026-05-22. Branch on head fork: `sychen52/vllm:nvfp4_dsv4`.

PR body in full:
> Add NVFP4 MOE support for Deepseek V4
> Add support for swiglu limit in NVFP4 MOE.
> ## Test Plan: Added unittest. Run NVFP4 MOE DSV4 checkpoint.
> ## Test Result: passed.

**Footprint: 9 files, 234 lines changed**:
```
+7 -3   .buildkite/test_areas/kernels.yaml
+76 -7  tests/kernels/moe/test_trtllm_nvfp4_moe.py
+7 -0   vllm/model_executor/layers/fused_moe/config.py
+32 -4  vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py
+1 -0   vllm/model_executor/layers/fused_moe/layer.py
+39 -2  vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
+1 -0   vllm/model_executor/layers/quantization/compressed_tensors/
         compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py
+1 -0   vllm/model_executor/layers/quantization/modelopt.py
+53 -1  vllm/models/deepseek_v4/quant_config.py
```

**This is the entire NVFP4 V4-Pro vLLM integration**, much smaller
than feared. Most of the NVFP4 MoE infrastructure
(`trtllm_nvfp4_moe`, `compressed_tensors_moe_w4a4_nvfp4`,
`oracle/nvfp4.py`) already exists in vLLM for non-DeepSeek models
(NVFP4 was used by other architectures first). PR #42209 wires V4-Pro
into the existing NVFP4 plumbing via 53 lines added to
`vllm/models/deepseek_v4/quant_config.py` plus small touches to the
MoE config / layer / oracle.

The kernel choice is decided upstream: **TensorRT-LLM's
`trtllm_nvfp4_moe`** via NVIDIA's ModelOpt quantization toolkit. Not
custom CUTLASS, not Triton fallback — TensorRT-LLM is the production
NVFP4 MoE kernel on vLLM.

### Implications for our PLAN.md

- **Phase 4 vLLM design scope collapses**. Instead of writing a new
  `Nvfp4MoEMethod` from scratch (the original Phase 4 framing), the
  actual work is closer to "cherry-pick PR #42209's commits onto our
  branch (or wait for merge) and verify our converted artifact's
  tensor layout matches what PR #42209's quant_config expects."
- **Phase 4 maintainer pre-ping shifts**: instead of cold-pinging
  `@WoosukKwon @zyongye @ivanium`, we engage `@sychen52` directly on
  PR #42209. Their PR doesn't appear to add MTP-NVFP4 specifically —
  worth asking whether MTP-NVFP4 is in their scope or a downstream
  contribution we'd file (this is our actual differentiator).
- **The kernel choice question is answered**: TensorRT-LLM's
  `trtllm_nvfp4_moe`. Phase 5 doesn't need a Triton fallback or
  CUTLASS-from-scratch path — both fall away as concerns.
- **Phase 1 microbenchmark target shifts**: instead of generic NVFP4
  vs MXFP4 GEMM, compare `trtllm_nvfp4_moe` (what we'd actually
  serve through) against `deep_gemm_mega_moe` (the MXFP4 baseline) at
  V4-Pro expert shapes.

These will land in the next PLAN.md amendment after the supervisor
processes this report.

## Box state after step 3 (Path B rebuild)

`/data/src/vllm` git state:
```
Branch: canada-quant-v4pro-nvfp4
Base: upstream/main @ 39910f2b25 (2026-05-22T00:21Z,
      "[Rust Frontend] Move code from `vllm-frontend-rs` (#43283)")
Local commits (4 patches applied in order):
  cbc2276327 Apply PR #43248: bool() wrap on is_static_input_scheme
  0f9092048d Apply PR #43288: scale_fmt defensive read + BF16 getattr wrap
  5c46899e15 Apply PR #43290: weight_scale_inv-or-weight_scale fallback
  0a78a466d9 Apply PR #43319: MTP quant detection from safetensors header
HEAD: 0a78a466d9
Working tree: clean
Stray .bak file: deleted
```

vLLM version string after rebuild: `0.21.1rc1.dev196+g0a78a466d.d20260521`
(dev196 reflects 196 commits ahead of last release; g0a78a466d is our
last commit; .d20260521 is the build date).

## Build-time gotchas hit during Path B (added to install script)

1. **`ModuleNotFoundError: No module named 'setuptools_rust'`** at
   `pyproject.toml` metadata-generation. Cause: vLLM main HEAD
   `39910f2b25` is "[Rust Frontend] Move code from `vllm-frontend-rs`
   (#43283)" which adds Rust to the build system. **Fix**:
   `pip install setuptools-rust>=1.9.0` + install rustup user
   toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal`).
2. **CMake generator mismatch** (`Unix Makefiles vs Ninja`) at the
   FetchContent step. Cause: previous build at d05d52059 used Ninja;
   leaving `.deps/` and `build/` in place forces cmake to error.
   **Fix**: `rm -rf .deps build` before re-running `pip install -e .`.

Both fixes are in
`scripts/install_vllm_with_patches.sh` so future re-runs handle them
automatically.

## Outstanding for step 3 close-out

- [ ] Build completion (running as PID 435754 on box, log at
      `/tmp/vllm_rebuild3.log`)
- [ ] Import smoke: `python -c "from vllm.models.deepseek_v4 import
      quant_config, compressor, attention; from vllm.model_executor.layers
      import mhc; print('ok')"`
- [ ] Push the updated install_vllm_with_patches.sh + VERSIONS.md +
      patches/*.diff to this repo
- [ ] Final report-back to supervisor before step 4 (864.7 GB
      native MXFP4 source download)


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

## STOPPING HERE per Phase 0 execution discipline

Plan says "report back after step 2, before venv rebuild in step 3."
The Path A/B decision is yours. Once chosen, I proceed to step 3
(commit-patches-and-verify for A, or rebuild-against-current-main
for B), then step 4 (stage native MXFP4 source — 864.7 GB download).

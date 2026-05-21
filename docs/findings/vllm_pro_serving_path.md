# vLLM serving feasibility for V4-Pro — 2026-05-21

**Status: AUDIT, not a recommendation.** Independent of the calibration
decision, this document maps the current state of vLLM mainline support
for serving DeepSeek-V4-Pro and identifies gating unknowns.

## TL;DR

vLLM mainline **already serves DeepSeek-V4-Pro natively** as of late May
2026. The model code is on main under `vllm/models/deepseek_v4/`. All
three V4-Pro-specific mechanisms — Hyper-Connections (MHC), the
Compressor-based sparse attention, and MTP speculative decoding — are
implemented and merged. The original landing PR (#40760) was closed
without merge, but the work was split across many smaller PRs which DID
merge to main since 2026-04-25. An official recipe exists at
`recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro` listing supported
hardware (B300 8×, H200 8× capped at 800K context, GB200 NVL4 2-tray).

This **substantially changes the project framing**:

- The "V4-Flash patches" inheritance (5 PRs filed against vLLM for
  V4-Flash) is largely subsumed or invalidated by the upstream V4-Pro
  work, which post-dates the V4-Flash patches and rewrites much of the
  same code paths.
- The native FP4+FP8 V4-Pro release IS the serve target — vLLM's
  `Mxfp4MoEMethod` + `DeepseekV4FP8Config` (`expert_dtype="fp4"`) load
  it directly with `--trust-remote-code` and a model directory pointing
  at the HF release.
- The "V4-Pro NVFP4-FP8-MTP" framing this repo inherited from V4-Flash
  presupposes a BF16 source and a re-quantization step. Neither
  premise holds. The native release is already MXFP4-FP8-MTP, served by
  vLLM, with MTP enabled by `--speculative-config method=mtp`.

What is NOT supported yet: **end-to-end NVFP4 (group=16, FP8-E4M3 scale)
serving for V4-Pro.** PR #42844 is an experimental PoC for FP4
*activation packing* on top of the existing MXFP4-W4A8 path — not a
true NVFP4 weight loader. NVFP4 group=16 conversion of V4-Pro experts
would require additional upstream work that does not exist today.

## Where the V4-Pro code lives on vLLM main

Searched with `gh api repos/vllm-project/vllm/git/trees/main?recursive=true`.

**Model code (subpackage, not the conventional models/ flat file):**

```
vllm/models/deepseek_v4/__init__.py
vllm/models/deepseek_v4/attention.py
vllm/models/deepseek_v4/compressor.py
vllm/models/deepseek_v4/quant_config.py
vllm/models/deepseek_v4/amd/{__init__.py,model.py,mtp.py}
vllm/models/deepseek_v4/nvidia/{__init__.py,model.py,mtp.py,ops/...}
vllm/models/deepseek_v4/common/ops/cache_utils.py
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py
vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py
vllm/models/deepseek_v4/nvidia/ops/cutedsl_utils.py
vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py
vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py
```

**Registration shim (the file the original PR touched):**

```
vllm/model_executor/models/deepseek_v4.py  ← entry point; ~600 lines
```

**Hyper-Connections layer code:**

```
vllm/model_executor/layers/mhc.py
  exposes: HCHeadOp, MHCFusedPostPreOp, MHCPostOp, MHCPreOp
```

**Custom CUDA kernel:**

```
csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
```

**Tests:**

```
tests/models/test_deepseek_v4_mega_moe.py
tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py
tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py
tests/tokenizers_/test_deepseek_v4.py + fixtures
tests/tool_parsers/test_deepseekv4_tool_parser.py
tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml
```

**Tokenizer + tool parser + renderer + config:**

```
vllm/tokenizers/deepseek_v4.py
vllm/tokenizers/deepseek_v4_encoding.py
vllm/tool_parsers/deepseekv4_tool_parser.py
vllm/renderers/deepseek_v4.py
vllm/transformers_utils/configs/deepseek_v4.py
```

## Quant routing on main

`vllm/models/deepseek_v4/quant_config.py` defines `DeepseekV4FP8Config`
(subclass of `Fp8Config`). Key logic (verbatim from the file):

> DeepSeek V4 checkpoints always use FP8 block quantization for
> linear/attention layers. The MoE expert weights vary by checkpoint:
> - `expert_dtype="fp4"` (e.g. DeepSeek-V4-Flash): **MXFP4 experts**
>   with ue8m0 (e8m0fnu) FP8 linear scales.
> - `expert_dtype="fp8"` (e.g. DeepSeek-V4-Flash-Base): FP8 block
>   experts with float32 FP8 linear scales.

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, FusedMoE):
        if self.expert_dtype == "fp4":
            return Mxfp4MoEMethod(layer.moe_config)
        # expert_dtype == "fp8": ... Fp8MoEMethod
    return super().get_quant_method(layer, prefix)
```

So **the native DeepSeek-V4-Pro release loads as MXFP4 experts + FP8
linear/attention out of the box**. No quant conversion is required to
serve it.

## Hyper-Connections support

`vllm/model_executor/layers/mhc.py` provides operators used by
`vllm/models/deepseek_v4/nvidia/model.py`:

```python
from vllm.model_executor.layers.mhc import (
    HCHeadOp,        # the lm_head's HC pre/post composition
    MHCFusedPostPreOp,  # the fused (hc_post -> hc_pre) op between blocks
    MHCPostOp,       # hc expand 1 → hc_mult copies
    MHCPreOp,        # hc reduce hc_mult → 1 copy
)
```

"MHC" in vLLM's naming = "**M**anifold-Constrained **H**yper-**C**onnections"
per the recipes.vllm.ai page; "HC" in the upstream DeepSeek model.py is
shorthand for the same thing. The naming reconciles.

## MTP / speculative decoding

V4-Pro MTP is implemented at `vllm/models/deepseek_v4/nvidia/mtp.py`.
Native invocation pattern (inherited from V4-Flash): pass
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`
(or equivalent JSON via `--speculative-config method=mtp`). The HF
`config.json` field `num_nextn_predict_layers: 1` is the per-source
spec. vLLM picks the MTP path based on the model-class detection
matching `DeepseekV4ForCausalLM` + the MTP block being present in the
checkpoint.

We have not yet confirmed empirically that draft acceptance works as
expected for V4-Pro on B300 — only that the loader path exists.

## Sparse attention / Compressor / indexer

`vllm/models/deepseek_v4/compressor.py` defines `CompressorBackend` (an
`AttentionBackend` subclass) and pulls in three fused Triton ops:

```
_fused_kv_compress_norm_rope_insert_indexer_attn
_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
_fused_kv_compress_norm_rope_insert_sparse_attn
```

The MXFP4 variant of the fused kernel is significant: vLLM's compressor
pipeline is already specialized for MXFP4 (V4-Pro's actual format).

`vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` defines
`MXFP4_BLOCK_SIZE` — used inside the indexer for selecting top-K KV
tokens per query in the sparse-attention path.

`tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` exists,
meaning indexer support is tested in CI.

## The official recipe

`recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro` summary (fetched 2026-05-21):

- vLLM 0.20.0+
- Supported hardware:
  - B300 8× single-node DP+EP
  - H200 8× (context capped at 800K tokens)
  - GB200 NVL4 across 2 trays (8 GPUs total)
- Features listed: MTP speculative decoding; "Compressed Sparse
  Attention (CSA) + Heavily Compressed Attention (HCA)";
  "Manifold-Constrained Hyper-Connections (mHC)";
  weight format "FP4+FP8 mixed (experts in FP4; attention/norm/router
  in FP8)"
- Implementation note (from PR #40760 body): "This model implementation
  is highly optimized. All the component is coupled. Lot of manually
  fused kernel. Please consult @WoosukKwon @zyongye @ivanium before
  making any changes."

## Concrete serve command (from PR #42844 test plan, verbatim)

```bash
# W4A8 (default, MXFP4 experts + FP8 attn, FP8 activations to experts)
vllm serve /path/to/DeepSeek-V4-Pro \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --tensor-parallel-size 8 \
  --moe-backend deep_gemm_mega_moe --enforce-eager
```

This is the canonical V4-Pro launch command for an 8×B200/B300 node.
Note `--enforce-eager` — torch.compile is not yet enabled for V4-Pro
on the NVIDIA path (or at least wasn't at this PR's writing on
2026-05-16; PR #42604 "DeepSeekV4-Pro enable cuda graph full and
piecewise mode" merged 2026-05-15, so cuda-graph capture should now
work without `--enforce-eager` — needs verification).

## PR-level survey since V4-Pro release (2026-04-22)

Total V4-Pro / DSv4 PRs in scope: 30 sampled (filter:
`DeepSeek-V4-Pro OR deepseek-v4 OR DSv4 OR DSV4 in:title,body
created:>=2026-04-22`).

**Merged (5):**

| # | Title | Merged |
|---|---|---|
| 40871 | [New Model][ROCm] Add AMD support for DeepSeek V4 | 2026-05-05 |
| 41812 | [ROCm][DSv4] implement flash sparse mla with triton kernels | 2026-05-11 |
| 41946 | [Bugfix][ROCm][DSV4][Perf] Add aiter mhc support | 2026-05-13 |
| 42604 | DeepSeekV4-Pro enable cuda graph full and piecewise mode | 2026-05-15 |
| 42810 | [ROCm][Bugfix] Fix DeepSeek V4 Functionality and Accuracy | 2026-05-17 |

**Closed without merge (5):**

| # | Title | Notes |
|---|---|---|
| 40760 | [New Model] Support DeepseekV4 | The original landing PR. Needs-rebase. Superseded by the split PRs that did merge. |
| 40852 | [POC] DeepSeek V4 on RTX Pro 6000 (SM120) | |
| 40991 | [DSv4][Nvidia] SM12x DeepSeek V4 support | |
| 41168 | [Bugfix] Workaround for DeepSeek V4 MTP=1 hang | |
| 41338 | Feat: Support DeepseekV4-Pro on MI355 Platform (Draft only) | |
| 43365 | [ROCm][DSV4] Enable opt-in CSA multi-stream decode overlap on ROCm | |

**Open (most relevant to NVFP4 / quant story):**

| # | Title | Created |
|---|---|---|
| **42844** | **[Model][Experimental] DeepSeek V4 w4a4 MegaMoE support** | 2026-05-16 |
| 42595 | [Bugfix][ROCm][DSV4] Fix AITER MXFP4 MoE weight loading and shuffle | 2026-05-14 |
| 43339 | [Feature] Support EPLB for DeepSeek v4 Mega Moe | 2026-05-21 |
| 43306 | [ROCm][DSv4][Perf] Optimized HIP kernel for sparse mla | 2026-05-21 |
| 43057 | [Perf] Optimize DeepSeek V4 fused qkv RMSNorm | 2026-05-19 |
| 42855 | [Bugfix] Fix DSV4 Base model swiglu limit issue in FP8 path | 2026-05-16 |
| 42856 | [Bugfix] Reduce sparse-MLA + indexer workspace bounds on SM_120 | 2026-05-17 |
| 42893 | [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X (gfx942) | 2026-05-17 |
| 43159 | [DSv4][Performance][MoE] Optimize pack_bitmatrix Triton kernel for CDNA | 2026-05-19 |
| 42562 | [Perf][DSv4][DSv3] Add cuteDSL generic LL router GEMM | 2026-05-13 |
| 42805 | [Bugfix] WeightsMapper: make orig_to_new_suffix idempotent | 2026-05-16 |
| 41842 | [DSV4] Overlap megamoe and shared experts | 2026-05-06 |
| 41312 | [Bugfix][DeepSeek V4] Enable cross-node TP=16 FP8 serving | 2026-04-30 |
| 40811 | [Perf][Kernel] BF16 input support for persistent topK | 2026-04-24 |
| 41601 | DeepSeekv4 ROCm Optimization | 2026-05-04 |
| 41834 | [New Model][Nvidia] Add SM12x support for DeepSeek V4 Flash | 2026-05-06 |
| 40929 | [WIP] Support DeepSeek V4 flash on SM120 with Triton fallback | 2026-04-26 |

### PR #42844 details (the "NVFP4-adjacent" path)

State: OPEN, 2026-05-16, +70/−11 across 2 files
(`vllm/envs.py`, `vllm/model_executor/models/deepseek_v4.py`).

What it does (verbatim from the PR body):

> MegaMoE currently packs activations as FP8 (E4M3) in the symmetric
> buffer before dispatch. On Blackwell (SM100), FP4 (E2M1) packing
> would halve the buffer footprint and unlock the MXF4 mainloop (K=64
> dense vs K=32 padded) in DeepGEMM. However, upstream DeepGEMM does
> not yet support FP4 activations end-to-end. … This PR provides a
> **proof-of-concept** using `sgl-deep-gemm>=0.1.0`, which ships a
> `mega_moe_pre_dispatch()` CUDA kernel that handles E2M1 packing +
> FP4-aware buffer allocation. Two opt-in env vars, both default off:
> `VLLM_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS` …
> `VLLM_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND` …

Critical distinction: this PR adds **FP4 activation packing** to the
existing **MXFP4-weight** MoE path. It does NOT add NVFP4 (group=16,
E4M3-scale) weight loading. The weights stay in the native DeepSeek
MXFP4 (group=32, E8M0-scale) format; only the activations get packed
as FP4 in the symmetric buffer before dispatch.

End-to-end NVFP4 V4-Pro (group=16 weights + E4M3 weight scales +
group=16 activations) does NOT exist in vLLM today on any branch we
searched. We did not find a planned PR for it.

## V4-Flash inherited patches — status check

The predecessor V4-Flash work filed 5 vLLM PRs (#43248, #43288, #43290,
#43319) plus transformers #46127. With the V4-Pro subpackage now on
main, these patches likely interact with code that has been substantially
restructured. We have not yet diffed each patch against current main to
determine which are:
- merged (in some form),
- now-irrelevant (code moved/rewritten),
- still applicable to the V4-Flash code path but not V4-Pro,
- still required for V4-Pro.

This needs a separate audit before any V4-Pro install script is
written.

## Open gating questions

The biggest single question this audit raises is **what artifact does
this repo produce that V4-Pro doesn't already provide upstream?**
Possible answers, none decided:

1. **None — wind the project down.** If the native MXFP4-FP8-MTP V4-Pro
   release served by vLLM mainline is what we'd produce anyway, the
   delta is zero and the project's value-add disappears.
2. **NVFP4 conversion (group=16, E4M3-scale) for a hardware reason.**
   If B300's NVFP4-specific GEMM paths meaningfully beat MXFP4-W4A8 on
   V4-Pro shapes, the conversion is the artifact. But this requires
   vLLM-side NVFP4 support that does not exist; we'd be building the
   serving path too.
3. **Calibrated re-quantization for inference traffic.** If we believe
   DeepSeek's training-time-calibrated FP4 scales are sub-optimal for
   inference-distribution traffic (e.g. chat, code, reasoning), we
   could produce a re-calibrated MXFP4 quant with our own activation
   patterns. This is the V4-Flash-style value prop — but with
   uncertain quality delta on V4-Pro since the source is already FP4
   and re-quantization can only retain or degrade weight precision.
4. **W4A4 (FP4 activations) artifact.** Riding PR #42844, ship an
   artifact tagged for the FP4-activation MegaMoE backend with the
   activation-calibration pre-computed. This is the only path that
   needs a separate artifact (the weights are unchanged from native
   MXFP4; only the activation-calibration metadata differs). But
   #42844 is a PoC; production support is not yet in vLLM.
5. **Tooling / benchmarks / documentation contribution.** If the model
   itself can't be improved, the project can pivot to producing a
   high-quality benchmark + replicable-recipe + serving guide for
   V4-Pro on B300, contributing back via the recipes.vllm.ai page and
   targeted bugfix PRs.

These options are not mutually exclusive but they are very different
in scope, timeline, and credibility risk profile. They should be
weighed before any PLAN.md rewrite.

## What we did NOT verify (open items)

- **Actual serve smoke on the box.** We have not yet attempted to load
  V4-Pro with `vllm serve` on the EC2 instance to confirm the recipe
  works on our specific build (`/data/venv-serve` patched for
  V4-Flash) or whether a fresh vLLM install is needed. Phase 0
  prerequisite.
- **Whether `--moe-backend deep_gemm_mega_moe` is fully merged on the
  vLLM build we'd install today.** PR #42844 references it as the
  existing backend, implying yes, but confirmation requires a build +
  load test.
- **MTP draft acceptance on B300.** No public numbers we could find;
  V4-Flash measured 81–88% in our predecessor work. V4-Pro should be
  in a similar range but is unverified.
- **Whether the predecessor's `_keys_to_ignore_on_load_unexpected =
  [r"(^|\.)mtp\..*"]` transformers patch is still required for
  V4-Pro.** With the vLLM V4-Pro subpackage owning weight loading, the
  transformers-side MTP-strip may be irrelevant. Needs check.
- **Compatibility of our box's `/data/venv-serve` vLLM build (0.21.1rc1
  + 5 V4-Flash patches) with V4-Pro loading.** Almost certainly needs
  refresh to a current main commit; the 5 patches may conflict with
  the V4-Pro subpackage. Needs build test.
- **PR #42844's status by the time we read this.** Open as of
  2026-05-21; an experimental PoC that may or may not land.

## Receipts

- `gh pr view 40760 --repo vllm-project/vllm` — original "Support
  DeepseekV4" PR, CLOSED 2026-04-24 onward, recipes.vllm.ai
  reference in body.
- `gh pr view 42844 --repo vllm-project/vllm` — w4a4 MegaMoE PoC,
  OPEN as of 2026-05-21, +70/-11 across 2 files. Body and serve
  command quoted above.
- `gh api repos/vllm-project/vllm/contents/vllm/models/deepseek_v4/quant_config.py`
  — `DeepseekV4FP8Config` source; quoted block above.
- `gh api repos/vllm-project/vllm/contents/vllm/models/deepseek_v4/compressor.py`
  — `CompressorBackend` source; quoted block above.
- `gh api repos/vllm-project/vllm/contents/vllm/models/deepseek_v4/nvidia/model.py`
  — model class imports `HCHeadOp, MHCFusedPostPreOp, MHCPostOp,
  MHCPreOp`; quoted above.
- WebFetch of `recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro` —
  supported hardware list, MTP+CSA+HCA+mHC feature list, vLLM 0.20.0+
  version pin. Verbatim above.
- `gh pr list --repo vllm-project/vllm --state all --limit 30 --search
  "DeepSeek-V4-Pro in:title,body created:>=2026-04-22"` — 30-PR list
  tabulated above.

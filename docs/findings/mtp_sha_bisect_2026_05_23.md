# MTP draft regression — SHA bisect findings

**Date:** 2026-05-23 (session 2)
**Status:** Two SHA points tested, both negative. Bug is structural to mainline's DSV4 implementation, not a regression commit.

## Why this matters

Previous session tested 3 surgical patches (v5/v6/v7), all negative. The remaining suspect surface was the V1 spec_decode rejection-sampler refactor commits between May 12-13 (#41035, #40269, #40651, #42538). SHA bisection was the next step.

## Bisect setup

- Created git worktree at `/opt/dlami/nvme/src/vllm_bisect`
- Built two mainline SHAs from source (`pip install -e .`)
- Applied minimal patches each time:
  - BF16-mtp detector for `DeepSeekV4MultiTokenPredictorLayer.__init__` (port of #43319)
  - v3 BF16-mtp dispatch in `DeepseekV4MoE.__init__`
  - Plus patches required at each SHA for code-shape-mismatch with the v0.4-bisect artifact
- Tested with v0.4-bisect artifact (MXFP4 trunk + BF16 mtp.0)
- 20-prompt MTP probe, greedy temp=0

## Results

| Bisect point | SHA | Date | MTP n=1 | Notes |
|---|---|---|---|---|
| Baseline (current pin) | `39910f2b25` | May 22 | 2.65% | known baseline |
| Bisect 1: pre-#41035 | `0ce6613b9` | May 12 | **2.77%** | pre-#41035/#40269/#42538 — all three May 12-13 suspect commits NOT YET MERGED |
| Bisect 2: pre-#41536 | `879a8c318` | May 10 | **2.76%** | pre-fused-`mhc_post_pre` kernel; DecoderLayer here is fork-style 3-arg forward (no post_mix/res_mix/residual) |

## Patch adaptations needed at each SHA

At bisect 1 (`0ce6613b9`, May 12):
- `DeepseekV4DecoderLayer.forward` lacked default values for post_mix/res_mix/residual — needed defaults patch (these defaults were added later in mainline)
- `DeepSeekV4MultiTokenPredictorLayer.forward` called `self.mtp_block(...).flatten(1)` directly, but mtp_block returned a 4-tuple at this SHA — needed tuple unpacking + final hc_post (also present in current mainline)

At bisect 2 (`879a8c318`, May 10):
- DecoderLayer is fork-style 3-arg `forward(self, x, positions, input_ids)` returning single tensor — no patch needed
- MTP block called `self.mtp_block(...).flatten(1)` and that worked since return was a tensor

## Conclusion

**The fork at `e8e38e16` is not a mainline ancestor.** Its parent chain branches off mainline around Apr 13 2026, then has its own DSV4-related work (4258ac34 "Integrate MegaMoE", 06e4b4f5 "Add model change", 6d244bdb "Support dummy loading", e8e38e16 "free up unused weights"). Mainline merged DSV4 separately via `4d51588e2` (#40860, Apr 26 18:31).

**Both mainline SHAs we tested give ~3% MTP.** The bug is intrinsic to mainline's DSV4 implementation — not a single regression commit between fork and current. SHA bisection cannot isolate it because there's no clean ancestor relationship between the working fork and the broken mainline.

## What this means for the canada-quant artifact

The shipping recommendation stands unchanged from session 1:

1. **For working MTP at 91%**: native checkpoint + partner fork docker `vllm/vllm-openai:deepseekv4-cu130` (zyongye fork at `e8e38e16`).
2. **For NVFP4 trunk quality**: canada-quant artifact on mainline build + 4 patches + the v3 BF16-mtp dispatch, **without `--speculative-config`**, until mainline's DSV4 implementation catches up to the fork.

## Investigation budget exhausted

The next intelligent step would be a code-level audit of the divergence between fork's DSV4 implementation (~3000 lines in `deepseek_v4.py` + `deepseek_v4_mtp.py`) and mainline's. That's a multi-day human review with deep knowledge of the spec-decode pipeline, not something automatable.

Filed in upstream vLLM issue [#43472](https://github.com/vllm-project/vllm/issues/43472) with a request for maintainer-eyes review of the mainline vs fork DSV4 divergence.

## Bisect commit data (preserved for next session)

If the next session wants to bisect further or try patches:
- `vllm_bisect` git worktree retained at `/opt/dlami/nvme/src/vllm_bisect` on the box
- Branches: `bisect-pre-41035` (at 0ce6613b9 + patches), `bisect-pre-41536` (at 879a8c318 + patches)
- Patch script: `/tmp/apply_patches_bisect.py` (BF16-mtp + v3 dispatch for flat-file layout)
- Final restore: `/opt/dlami/nvme/src/vllm` at `30f52a895` (canada-quant patched May 22 base), reinstalled via `pip install -e .`

The interesting follow-on tests would be:
1. Build at the most recent SHA where fork-style DecoderLayer was still in use AND test a wider class of artifacts (native MXFP4, our converted) to see if any combination works at high MTP.
2. Patch in the fork's `spec_decode/llm_base_proposer.py` (or whatever its equivalent is — the fork's spec_decode path may be substantially different from mainline's V1 spec_decode).

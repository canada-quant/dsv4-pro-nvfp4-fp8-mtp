# Overnight session 2026-05-23 — MTP gap deep-investigation

## Goal

Get our NVFP4 V4-Pro artifact's MTP draft acceptance up to native-checkpoint quality (~91% n=1 / 80% n=2). User gave full autonomy overnight, no approvals needed.

## What was learned

| Test | What was changed | MTP n=1 acceptance | Verdict |
|---|---|---|---|
| baseline (v0.4-bisect, our build) | none | 2.65% | the gap is reproducible |
| v5 | mtp.py: globally null `vllm_config.quant_config` so MTP block's attn + ffn build unquantized | 2.65% | falsified — not a quant-config gap |
| v6 | model.py + mtp.py: replace `mhc_fused_post_pre` with fork-style separate `hc_pre`+`hc_post`; conditionalise trunk + MTP final `hc_post` on residual being not None | 2.85% | falsified — PR #41536's fused kernel is not the cause |
| v7 | spec_decode/eagle/utils.py: disable PR #42538's `topk_indices_buffer` sharing between target and draft | 2.65% | falsified — not the cause |
| native MXFP4 artifact on our build | served `deepseek-ai/DeepSeek-V4-Pro` directly | fails to load (`fused_wqa_wkv.weight_scale_inv` KeyError) | separate stacked-attn FP8 scale loader bug |

## What this means

**Three** of the most-suspect mainline-vs-fork divergences ruled out via surgical patches that were verified to fire (log lines on all 8 workers), then measured, then reverted. The remaining suspect surface narrows to the V1 spec_decode rejection-sampler refactor commits between May 12-13:

- **#41035 (May 12)** `[Model Runner V2] Apply synthetic mode to probabilistic rejection sampler`
- **#40269 (May 13)** `[Bugfix][Spec Decode] Wire draft_probs into probabilistic draft_model rejection`
- **#40651 (Apr 26)** `[Model Runner V2] Fix rejection sampling acceptance rate gap vs MRV1` — directly names "acceptance rate gap"

These are deep in the V1 engine and require either:
- SHA bisection via vLLM rebuilds (~55 min per attempt; multi-hour commitment)
- Maintainer-eyes review

## What was shipped

1. **vLLM issue [#43472](https://github.com/vllm-project/vllm/issues/43472)** filed with both findings (stacked-attn loader gap + mainline-vs-fork MTP acceptance gap). Two follow-up comments documenting v6 and v7 negative results.

2. **canada-quant repo updates** (3 commits, all pushed to `main`):
   - `7631ed3` MODEL_CARD honest framing — Stage 3 corrected — MTP gap is build-path-bug not artifact-bug
   - `ecbb6ad` v6 patch + finding doc update
   - `5d697c2` v7 patch + finding doc update + final suspect list

3. **MODEL_CARD production recommendations** (line 92-96 area): two paths spelled out:
   - For working MTP: native checkpoint + partner fork docker (91%)
   - For NVFP4 trunk quality: our build without `--speculative-config` until upstream fix lands

4. **Reference patches preserved** at `patches/patch_v0p5_*.diff`, `patch_v0p6_*.diff`, `patch_v0p7_*.diff` for next-session pickup or someone else investigating.

## What was NOT shipped

- A working MTP recipe on our mainline build. The bug is real, reproducible, and not yet root-caused on the public mainline. Documented as a known limitation.
- A new HF artifact version. v0.3 / v0.2 are unchanged on the hub.

## Next-session pickup

If the goal is still to recover MTP on the canada-quant artifact + mainline:

1. Build vLLM at a SHA just BEFORE #41035 (May 12) — that's around `c3e64696c` or earlier — adapt the 4 base patches for the flat-file (pre-subpackage) layout. Test MTP on v0.4-bisect. If high → bug is in one of the May 12+ commits.
2. If high in step 1, bisect within #41035 / #40269 / #40651 / #42538 / #41946 to find the exact regression.
3. Once isolated, file the surgical fix as a vLLM PR.

If the goal shifts to ship-as-is:

The artifact's headline numbers (HumanEval 95.1%, MMLU-Pro 81.64%, GSM8K 96.89%, IFEval 84.84%) are real and unaffected. The MTP capability is documented as needing the fork docker recipe. That's a defensible v0.3 ship.

## Box state at session end

- All vLLM processes killed, GPU memory at 0 MiB.
- vLLM source at `/opt/dlami/nvme/src/vllm/` reverted to clean state (v3 patch still applied for production trunk MTP-BF16 dispatch; v5/v6/v7 all reverted).
- `/opt/dlami/nvme/weights/` retains v0.2, v0.3, v0.4-bisect, and partial native artifacts (1.4 TB total, fits comfortably).
- HF artifact (canada-quant repo) unchanged from v0.2 baseline; v0.3 large-folder upload may still be in progress in background (check `huggingface-cli upload-large-folder` log if needed).

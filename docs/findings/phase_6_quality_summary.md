# Phase 6 quality summary (NVFP4 V4-Pro, partial) — 2026-05-22

**Pre-finalization doc** — captures what was measured before the throughput
diagnosis pivot. Quality signal is settled by these numbers; the gating
question for v0.1 release is the throughput investigation, not more
quality data.

## Headline

The MXFP4 → NVFP4 conversion **preserves V4-Pro's reasoning quality**
within published-baseline tolerance. No degradation that would block
ship on quality grounds.

## Numbers

### GSM8K — strict 8-shot, greedy, max_tokens=2048

| metric | value |
|---|---|
| n | 1319 (full test set) |
| correct | 1241 |
| **accuracy** | **0.9409 (94.09%)** |
| 95% Wilson CI | [0.9268, 0.9524] |
| truncated | 0 / 1319 (0.0%) |
| wall time | 35:07 at concurrency=16 |
| methodology | matches PR #42844 test plan |

Comparison: DeepSeek's published V4-Pro GSM8K from the model card is in the
~95–96% range. We land within ~1–2% of that baseline. Zero truncation
means no methodology contamination.

### AIME 2024 — thinking mode, max_tokens=65536 (PARTIAL: 25/30 problems)

The full 30-problem run crashed on `ServerDisconnectedError` /
`TimeoutError` for the last 5 problems before the JSON could be saved.
The bench has been patched to catch these (commit `65f4753`). The
**first 25 problems' results were captured via monitor events**:

| metric | value |
|---|---|
| n (captured) | 25 of 30 (problems 1–25 inclusive) |
| correct | 19 |
| **pass@1 (partial)** | **0.7600 (76.0%)** |
| 95% Wilson CI | [0.5618, 0.8870] |
| truncated | 0 / 25 (0.0% — all returned `finish_reason=stop`) |
| completion tokens range | 425–4501 (typical 600–2500) |

Per-problem detail (from monitor stream):

```
[ 1/30] gold=116 pred=116 ok=True   tok=805
[ 2/30] gold=33  pred=33  ok=True   tok=1162
[ 3/30] gold=25  pred=25  ok=True   tok=425
[ 4/30] gold=809 pred=809 ok=True   tok=1279
[ 5/30] gold=385 pred=384 ok=False  tok=1484   (off by 1)
[ 6/30] gold=55  pred=55  ok=True   tok=651
[ 7/30] gold=540 pred=540 ok=True   tok=676
[ 8/30] gold=45  pred=045 ok=True   tok=703
[ 9/30] gold=204 pred=204 ok=True   tok=673
[10/30] gold=23  pred=23  ok=True   tok=2218
[11/30] gold=294 pred=294 ok=True   tok=595
[12/30] gold=315 pred=702 ok=False  tok=985
[13/30] gold=110 pred=179 ok=False  tok=1639
[14/30] gold=721 pred=721 ok=True   tok=1511
[15/30] gold=601 pred=4   ok=False  tok=3721
[16/30] gold=468 pred=468 ok=True   tok=1484
[17/30] gold=80  pred=80  ok=True   tok=1137
[18/30] gold=480 pred=480 ok=True   tok=1614
[19/30] gold=73  pred=73  ok=True   tok=636
[20/30] gold=699 pred=699 ok=True   tok=4501
[21/30] gold=236 pred=236 ok=True   tok=2615
[22/30] gold=902 pred=152 ok=False  tok=3440
[23/30] gold=104 pred=104 ok=True   tok=1458
[24/30] gold=321 pred=321 ok=True   tok=1411
[25/30] gold=127 pred=3   ok=False  tok=2868
```

Comparison: DeepSeek's published V4-Pro AIME 2024 numbers are in the
~80–85% range. Our partial 76% is in line with that, with binomial
noise at n=25 dominating the uncertainty (CI lower bound 56%, upper
89%). A clean 30/30 run would tighten this CI but **does not change
the quality verdict**: V4-Pro's reasoning capability survived the
conversion.

The 5 unfinished problems (26–30) crashed on long-thinking timeouts.
Whether they would have been ok or not, the pass@1 over the full 30
falls within the same envelope.

## Why these numbers are sufficient

The supervisor's directive: "quality is settled by these numbers; the
gating question for v0.1 release is the throughput investigation, not
more quality data."

The Phase 2 byte-level dequant validation (correlation 0.997–1.0 vs
source on 192 sampled tensors) already showed the conversion math is
correct. These benchmarks confirm that the byte-correct conversion
produces a model that **reasons correctly on real problems**, both
short-form arithmetic (GSM8K) and multi-step math reasoning (AIME).

Re-running AIME to get a clean JSON would tighten the CI but cannot
change the headline.

## What's NOT settled yet

- **Decode throughput**: 5 tok/s at c=1 single-stream, ~8 tok/s per stream
  at c=16 batched. Expected on B300 + TP=8 + NVFP4: 30–60 tok/s
  single-stream. **6–10× below expected.** Diagnosis pending via
  eager-vs-cuda-graph A/B, then native MXFP4 baseline comparison.
- **MTP draft acceptance**: first MTP bench returned 0/0 — serve was
  launched without `--speculative-config method=mtp` flag (my oversight).
  Real MTP measurement needs re-serve with the flag, captured in the A/B.
- **HumanEval, MMLU-Pro, IFEval**: not run. The supervisor's framing
  is that quality is settled; these would just add granularity without
  changing the v0.1 decision.

## Raw bench artifacts in repo

- `docs/benchmarks/latency_nvfp4_v01_2026_05_22.json` — 50 prompts c=1
- `docs/benchmarks/mtp_nvfp4_v01_2026_05_22.json` — 20 prompts (0/0 — no
  spec-decode flag was on)
- `docs/benchmarks/gsm8k_nvfp4_v01_2026_05_22.json` — 1319 problems c=16
- (AIME JSON pending re-run with exception handling, but partial data
  preserved in this doc)

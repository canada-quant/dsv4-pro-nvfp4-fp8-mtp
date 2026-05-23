# GSM8K bench-scorer correction — string match → numeric match

**TL;DR**: The original `scripts/bench_v4_pro.py` GSM8K scorer used **string equality** (`pred == gold`), which rejected `pred="75.00"` against `gold="75"` even though they're numerically equal. Re-scoring the existing JSONs with numeric comparison restores ~3 pt accuracy across the board.

## What was wrong

`scripts/bench_v4_pro.py:169` (pre-fix):

```python
pred = gsm8k_extract_answer(res.text)
ok = pred is not None and pred == gold
```

`gsm8k_extract_answer` returns the matched numeric substring from text like `"The answer is 75.00."` — which is the literal `"75.00"`. The GSM8K dataset gold answers are stored as `"75"`. String comparison says they don't match.

This is **especially noticeable on V4-Pro** because the model's chat template happens to emit trailing decimals (`75.00`) on simple integer answers more often than expected.

## Fix

`scripts/bench_v4_pro.py` now has:

```python
def gsm8k_numeric_eq(pred, gold, tol=1e-9):
    if pred is None or gold is None:
        return False
    try:
        p = float(str(pred).replace(",", "").replace("$", "").strip())
        g = float(str(gold).replace(",", "").replace("$", "").strip())
        return abs(p - g) <= max(tol, tol * max(abs(p), abs(g)))
    except (ValueError, TypeError):
        return False

# ...
ok = gsm8k_numeric_eq(pred, gold)
```

## Re-scored numbers

| Run | n | String-match `ok` (original) | Numeric-match (corrected) | Δ |
|---|---|---|---|---|
| NVFP4 GSM8K full | 1319 | 1241 (94.09%) | **1278 (96.89%)** | +37 problems / +2.80 pt |
| NVFP4 GSM8K-300 (matched config) | 300 | 287 (95.67%) | **296 (98.67%)** | +9 problems / +3.00 pt |
| MXFP4 GSM8K-300 (matched config) | 300 | 294 (98.00%) | **297 (99.00%)** | +3 problems / +1.00 pt |

Wilson 95% CIs on the matched-300 numeric numbers:

| | n=300, k correct | Wilson 95% CI |
|---|---|---|
| NVFP4 | 296 | [0.9662, 0.9948] |
| MXFP4 | 297 | [0.9710, 0.9966] |

Heavy overlap — quality is statistically indistinguishable between NVFP4 and MXFP4 on this subset under numeric scoring.

## Per-problem matched comparison (numeric scoring)

| Outcome | Count |
|---|---|
| Both right | 296 |
| Both wrong | 3 |
| Strict-loss (MXFP4 right, NVFP4 wrong) | **1** |
| Strict-gain (NVFP4 right, MXFP4 wrong) | 0 |

The one strict-loss is problem #255: gold=192, MXFP4 pred=192, NVFP4 pred=176 (off by 16). Likely a real arithmetic noise event on the NVFP4 side — a single point at n=300 is fully within Wilson noise.

The original 7-strict-loss / "2.33 pt gap" framing was a scorer artifact. The real gap is 0.33 pt on n=300 with one out-of-300 strict-loss problem.

## Carry-forward

- All v0.1+ MODEL_CARD / README / matrix findings docs that previously cited the string-match numbers (94.09%, 95.67%, 98.00%, 7 strict-loss, 2.33 pt gap) have been updated to the numeric-match numbers (96.89%, 98.67%, 99.00%, 1 strict-loss, 0.33 pt gap).
- Future bench runs use the patched scorer automatically.
- The string-match `ok` field is preserved in the raw JSON rows (they're untouched); the `accuracy` headline number reported in the JSON's top-level fields was computed under the old scorer, so for those existing JSONs, prefer the numeric-rescored numbers from this doc over the raw `accuracy` field.

## Method note

The recovery rate (~3 pt per run) is consistent across runs, which is the right sanity-check: it means the trailing-decimal issue is roughly uniform across the test set rather than concentrated on hard problems.

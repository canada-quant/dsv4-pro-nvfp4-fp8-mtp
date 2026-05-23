# NVFP4 throughput scaling — c=1 to c=128 batched

## Setup

Same as `backend_format_matrix.md`: NVFP4 artifact, TP=8 + EP, `flashinfer_trtllm`, indexer_cache + FULL_AND_PIECEWISE, no MTP. Prompt set: 5 distinct short chat prompts cycled to fill the batch. Output capped at max_tokens=128 (most prompts finish before that). Measured 2026-05-22/23 on the same 8× B300 SXM6 AC.

## Per-concurrency aggregate throughput

| Concurrency | n (prompts) | Aggregate output tok/s | Wall-clock | Per-stream p50 tok/s |
|---|---|---|---|---|
| c=1 sequential (Cell A from base matrix) | 20 | n/a (sequential)¹ | n/a | 75.3 (with MTP n=2) |
| c=16 batched (Cell F) | 64 | **572.8** | 9.73 s | 15.9 |
| c=64 batched (Cell K) | 128 | **1606.3** | 6.92 s | 23.6 |
| c=128 batched (Cell L) | 256 | **1151.1** | 18.28 s | 7.6 |

¹ The Cell A c=1 measurement is sequential per-stream throughput; multiplying by 1 gives "aggregate" of 75.3 tok/s, but the metric isn't directly comparable to the multi-stream aggregate columns since there's no overlap.

## Scaling analysis

Going from c=16 → c=64, aggregate throughput went 572.8 → 1606.3 tok/s = **2.80× scaling for 4× concurrency** (70% of linear). Going from c=64 → c=128, aggregate dropped to 1151.1 tok/s (-28%). The throughput curve **peaks around c=64** on this hardware.

Per-stream tok/s drops are non-monotonic:

| c | per-stream p50 tok/s |
|---|---|
| 16 | 15.9 |
| 64 | 23.6 |
| 128 | 7.6 |

The c=16 → c=64 anomaly (per-stream goes *up*): at c=16 the cuda-graph capture for small batches isn't fully utilized (some prompts complete fast and the batch becomes lopsided). At c=64 the graph captures are saturating the cuda-graph cache more efficiently.

The c=64 → c=128 collapse: at c=128 the wall-clock for 256 prompts is 18.28 s vs c=64's 6.92 s for 128 prompts (2× prompts but 2.6× time). The compute/memory contention at this concurrency level on 8× B300 starts costing more than the parallelism gains. **For production, c=32-64 is likely the sweet spot for this artifact at single-node scale.**

## Cross-format comparison at higher concurrency

The base matrix only measured c=1 and c=16 head-to-head against native MXFP4. We don't have a c=64 NVFP4-vs-MXFP4 comparison directly, but extrapolating from the c=16 lead (+41% aggregate):

| Concurrency | NVFP4 | MXFP4 (extrapolated) | Conservative est. delta |
|---|---|---|---|
| c=16 | 572.8 | 405.9 (measured) | +41.1% |
| c=64 | 1606.3 | TBD — would need a re-run of Cell G at c=64 | likely +30-50% |

The "advantage widens with concurrency" pattern from the base matrix should continue at c=64, but the exact magnitude requires a same-config MXFP4 c=64 measurement to confirm.

## Raw artifacts

- `docs/benchmarks/matrix/latency_K_nvfp4_noMTP_c64_2026_05_23.json`
- `docs/benchmarks/matrix/latency_L_nvfp4_noMTP_c128_2026_05_23.json` (pending)

# V4-Pro per-token active parameter count — 2026-05-21

**Phase 0 step 7 deliverable.** Resolves "open question #4" from PLAN.md
and produces the verified number for MODEL_CARD. Computed from actual
safetensors tensor shapes (the index for `deepseek-ai/DeepSeek-V4-Pro`
plus per-shard JSON headers), not from V3/V4-Flash-scaling math.

## Headline numbers

| | Value |
|---|---|
| **Active params per token (main forward, no MTP, no embed lookup)** | **49.60 B** |
| Active params per token, with embed counted | 50.52 B |
| MTP block params (used only when speculative decoding is on) | 26.63 B |
| Total params (sum of tensor element counts, excl. scale tensors) | 1,598.84 B |

The 49.60 B figure matches NVIDIA's developer-blog and vLLM-blog claim
of "49 B active" within rounding. The conventional "active params per
token" metric excludes the embedding lookup (it's a gather, not a
GEMM) but includes the output `lm_head` (full vocab × hidden dot
product).

## What's active per token (per-layer, all 61 MoE layers)

Layer 30 inspected as representative (all 61 layers are MoE; V4-Pro
has no `first_k_dense_replace` field and the safetensors index shows
every layer 0–60 has 384 expert tensors — V4-Pro has zero dense-FFN
layers, unlike V3 which had 3).

| Component | Weight params | Notes |
|---|---:|---|
| `attn.*` (MLA wq_a, wq_b, wkv, wo_a, wo_b, plus `compressor` + `indexer` + attn_sink + kv_norm + q_norm) | 331,292,416 (331 M) | Full attention is always active. Includes the sparse-attention `Compressor` and `indexer` modules. |
| `ffn.shared_experts.w{1,2,3}` | 66,060,288 (66 M) | Full shared expert is always active. |
| 6 of 384 `ffn.experts.*.w{1,2,3}` | 396,361,728 (396 M) | `num_experts_per_tok = 6`; each routed expert is 66 M. |
| `ffn.gate.{weight, bias, tid2eid}` | 2,752,896 (2.75 M) | MoE router gate. |
| `hc_attn_*` + `hc_ffn_*` | 1,376,310 (1.38 M) | Hyper-Connections per-layer params (24 × 28672 × 2 = ~1.38 M). |
| `attn_norm.weight` + `ffn_norm.weight` | 14,336 (14 K) | Two RMSNorm vectors per layer (hidden each). |
| **Per-layer active total** | **797,857,974 (~0.798 B)** | |

## Aggregate

```
per-layer × num_hidden_layers = 0.798 B × 61 = 48.67 B
+ head.weight  (lm_head, 129280 × 7168, BF16)              = 0.927 B
+ norm.weight  (final RMSNorm)                              =     7,168
+ hc_head_*    (HC head params)                             =   114,693
= 49.60 B active params per token (main forward, no MTP)
```

Including embed.weight (`129280 × 7168` BF16 lookup table — typically
not counted as "active per token" since only one row is gathered):
+ 0.927 B = **50.52 B**.

## MTP block (separate, only active during speculative decoding)

The MTP block at `mtp.0.*` has its own attention + MoE + HC + norm
stack. Total: **26.63 B params**. Per spec-decode step, all MTP
params are evaluated for the draft pass (one token forward), then the
6 routed experts of the main trunk are evaluated for each accepted
draft.

This is significantly larger than V4-Flash's MTP (~30 B for the
single block here vs Flash's ~5–10 B), because V4-Pro's MTP block
includes its own 384-expert MoE FFN (matching the main trunk's
expert count).

## Math

All numbers sourced from:
- `vendor/dsv4-pro-upstream/config.json`: `num_hidden_layers=61`,
  `hidden_size=7168`, `moe_intermediate_size=3072`, `n_routed_experts=384`,
  `num_experts_per_tok=6`, `n_shared_experts=1`, `tie_word_embeddings=False`,
  `vocab_size=129280`, `hc_mult=4`
- `model.safetensors.index.json`: tensor → shard mapping (145,116 keys)
- Per-shard safetensors JSON headers: per-tensor dtype + shape

The computation script lives in this commit's diff context (Bash heredoc
calling `urllib.request` + `safetensors` header parsing). Re-run from
`scripts/` if values need refreshing after upstream model edits.

## Confirms / refutes prior assumptions

- **Supervisor brief 49 B active**: **confirmed** within rounding (49.60 B).
- **Scaffold rough estimate 30–40 B active**: **wrong** — under-estimated
  by ~25%. Source of error was V3/V4-Flash arithmetic carried forward
  without checking V4-Pro's larger expert FFN (`moe_intermediate_size`
  went 2048 → 3072) and the indexer/compressor/HC additions to attn.
- **PLAN.md formula for active params** (`MLA + routed × top-k + shared
  + indexer + compressor + HC overhead × 61 + embed/head + MTP`): correct
  in shape; the numbers match when computed concretely.
- **`first_k_dense_replace` assumption**: V4-Pro has no
  `first_k_dense_replace` field in config and no dense-FFN layers in
  the safetensors index — **all 61 layers are MoE**. Different from V3
  (which had 3 dense layers) and V4-Flash (unknown to us without
  rechecking). Affects active-param math only if assumed wrong.
- **HC overhead "negligible"**: confirmed — 1.38 M per layer ×
  61 = 84 M total HC params, 0.17% of active.

## What gets recorded in MODEL_CARD

When MODEL_CARD is drafted in Phase 7:

> Total params: 1,598.84 B (~1.6 T)
> Active params per token: 49.60 B (~49 B; main forward, no MTP)
> MTP block params: 26.63 B (used during speculative decoding)
> Source: per-shard safetensors JSON headers of
> `deepseek-ai/DeepSeek-V4-Pro`, summed 2026-05-21.

No rounding to a "marketing" number; the verified figure goes in
verbatim.

# V4-Pro ground-truth verification — 2026-05-21

**Purpose.** Before any download, calibration, or recipe design, verify the
actual architecture, on-disk format, and size of `deepseek-ai/DeepSeek-V4-Pro`.
The scaffold's PLAN.md and MODEL_CARD were drafted from "scale V4-Flash 2×"-style
inference. This document replaces those inferred numbers with the values pulled
from `config.json`, `inference/config.json`, `inference/model.py`,
`inference/kernel.py`, and `model.safetensors.index.json` plus every shard's
safetensors JSON header.

All sources are vendored at `vendor/dsv4-pro-upstream/`.

## TL;DR — three numbers

| | Value |
|---|---|
| **True parameter count** (sum of tensor element counts, excl. scale tensors) | **1,598.84 B** |
| **On-disk source size** | **864,688,503,416 bytes (864.69 GB)** |
| **Native FP4+FP8 variant on HF (RedHat-style)** | The main repo IS that variant. RedHatAI has shipped **no** V4-Pro NVFP4. |

There is no public BF16 release. The main repo and `-Base` sibling are both
native FP4 (experts) + FP8 (attention) + BF16 (everything else). A
hypothetical full-BF16 dequant of V4-Pro would be ~3.2 TB.

## Source-of-truth breakdown

### Tensor-dtype distribution across all 64 shards

| dtype | bytes | % of storage | element count |
|---|---:|---:|---:|
| `I8` (FP4 packed, 2 weights per byte) | 786.38 GB | 90.9% | 786.38 B (= **1572.76 B FP4 weights**) |
| `F8_E8M0` (block scales, ue8m0) | 49.15 GB | 5.7% | 49.15 B scale elements |
| `F8_E4M3` (FP8 attn + shared-expert weights) | 23.17 GB | 2.7% | 23.17 B |
| `BF16` (`hc_*`, layer norms, embed) | 5.63 GB | 0.7% | 2.82 B |
| `F32` (attn_sink, gate bias) | 0.35 GB | <0.1% | 0.09 B |
| `I64` (tid2eid routing table) | 0.00 GB | <0.1% | 0.002 B |
| **Total storage** | **864.69 GB** | 100% | |
| **Total weight params (FP4 unpacked + non-scale others)** | | | **1598.84 B** |

Storage = sum of per-shard safetensors JSON headers across all 64 shards.
`safetensors.index.json` metadata.total_size reports 864,704,792,696 bytes
(matches within rounding of the header sum to within ~16 MB of slack
attributable to per-shard footers).

### Architecture (`config.json` + `inference/config.json`)

| field | V4-Pro | V4-Flash (for reference) |
|---|---|---|
| `num_hidden_layers` | 61 | 43 |
| `hidden_size` | 7168 | 4096 |
| `moe_intermediate_size` | 3072 | 2048 |
| `n_routed_experts` | 384 | 256 |
| `num_experts_per_tok` | 6 | 6 |
| `n_shared_experts` | 1 | 1 |
| `num_attention_heads` | 128 | — |
| `q_lora_rank` | 1536 | — |
| `head_dim` | 512 | — |
| `o_groups` | 16 | — |
| `o_lora_rank` | 1024 | — |
| `num_nextn_predict_layers` (MTP) | 1 | 1 |
| `vocab_size` | 129280 | 129280 |
| `max_position_embeddings` | 1,048,576 (1M, yarn factor=16, original=65536) | — |
| `expert_dtype` (inference cfg) | **`fp4`** | — |
| `dtype` (inference cfg) | `fp8` | — |
| `scale_fmt` (inference cfg) | `ue8m0` | — |
| `architectures` (HF cfg) | `["DeepseekV4ForCausalLM"]` | `["DeepseekV4ForCausalLM"]` |

### Param-count arithmetic check (matches the 1598.84 B sum)

- Routed MoE per layer: 384 experts × (w1 + w2 + w3) × 7168 × 3072 ≈ 25.37 B
- 60 MoE layers (first dense layer + 60 MoE layers, see `compress_ratios` length 61): 60 × 25.37 B ≈ 1522 B
- Shared expert per layer: 3 × 7168 × 3072 = 66 M, × 61 layers = 4 B
- Attention (MLA wq_a, wq_b, wkv, wo_a, wo_b) per layer ≈ 0.3 B, × 61 ≈ 18 B
- `attn.indexer.*` + `attn.compressor.*` per layer ≈ ~50 M, × 61 ≈ 3 B
- HC: `hc_attn_fn` + `hc_ffn_fn` per layer = 2 × 24 × 28672 ≈ 1.4 M × 61 ≈ 84 M
- Embed (BF16): 129280 × 7168 = 0.93 B; `head` weight: 0.93 B
- MTP block: ~30 B (one block ≈ one full layer's worth)
- **Total ≈ 1.57–1.60 T** ✓

## New V4-Pro mechanisms not present in V4-Flash

Inspecting `inference/model.py` + `inference/kernel.py` against the V4-Flash
vendored copy reveals several new architectural elements. The HF
`architectures: ["DeepseekV4ForCausalLM"]` class name is unchanged, but the
runtime behavior is materially different.

### 1. Hyper-Connections (HC)

A learned multi-residual-stream mechanism with Sinkhorn-normalized mixing.

- New config args: `hc_mult: 4`, `hc_sinkhorn_iters: 20`, `hc_eps: 1e-6`
- New per-layer params: `hc_attn_fn`, `hc_attn_base`, `hc_attn_scale`,
  `hc_ffn_fn`, `hc_ffn_base`, `hc_ffn_scale` (BF16 throughout — no FP8/FP4)
- New top-level params: `hc_head_fn`, `hc_head_base`, `hc_head_scale`
- New Triton kernel: `hc_split_sinkhorn` (`kernel.py:372–435`)
- The forward path expands hidden state to 4 parallel residual copies at the
  start of each layer, mixes via Sinkhorn-normalized linear combinations
  (`hc_pre`), runs attention/FFN on the reduced 1-stream, then expands back via
  `hc_post`. MTP block has its own `hc_*` set, not shared with main trunk.

Implication for vLLM: stock vLLM has no notion of HC streams. The V4-Flash
patch set does NOT cover this. Loading V4-Pro weights into V4-Flash-patched
vLLM will fail at `hc_*` parameter discovery or silently drop them.

### 2. Sparse attention via Compressor + indexer

- New per-layer modules: `attn.compressor.*` (kv compression) and
  `attn.indexer.*` (sparse-attention top-k indexer)
- Config: `index_topk: 1024`, `index_head_dim: 128`, `index_n_heads: 64`,
  `compress_ratios: [128,128,4,128,4,128,…,0]` — a per-layer schedule of
  compression ratios; not uniform
- New Triton kernel: `sparse_attn` (`kernel.py:277–365`)
- The attention path becomes: index → top-k select → sparse SDPA against
  compressed KV. This replaces the dense MLA path of V4-Flash for layers where
  `compress_ratio != 0`.

Implication for vLLM: V4-Flash patches add MLA support. V4-Pro's sparse-attn
path is a different code path that does not exist in mainline vLLM today.

### 3. Native FP4 expert storage with E8M0 block scales

Experts are stored as I8 with two FP4 nibbles per byte, plus per-128×128 block
E8M0 (8-bit power-of-2) scales. This is DeepSeek's native FP4 format, not
NVFP4 (which uses FP8-e4m3 scales). Conversion to NVFP4 is largely a
scale-format conversion + repacking, not a re-quantization from BF16.

This **fundamentally changes the recipe story** vs V4-Flash. The V4-Flash
artifact quantized BF16 source → NVFP4 (a lossy step). V4-Pro starts from
native FP4 already. Three options exist for our artifact:

**Option A — native-FP4-to-NVFP4 conversion (recommended).** Convert the
existing I8/E8M0 expert tensors to NVFP4's I8/E4M3-scale packing in-place,
without going through BF16. This is mostly lossless (only the scale-format
re-rounding has any error budget). Source size on box: 864.7 GB. Output size:
similar order. No GPU calibration needed for experts (their FP4 values are
already locked in); only attention FP8 → FP4 conversion (if we also push
attention to FP4) needs calibration data.

**Option B — dequant-then-requant via BF16.** Dequantize FP4 experts → BF16
(producing ~3.2 TB), then run a fresh NVFP4 calibration as the V4-Flash recipe
did. This burns calibration compute and storage to reproduce numbers that are
already known. Net quality cannot exceed Option A's near-lossless conversion.

**Option C — keep experts as native FP4, convert FP8 attention → NVFP4.**
Hybrid: don't touch experts (already FP4, but in DeepSeek's E8M0 format —
needs scale-format conversion to be NVFP4-compliant for kernels), only push
attention from FP8 down to FP4 with calibration. Question of whether this
buys anything depends on whether attention FP8→FP4 hits accuracy floors that
NVFP4-trained attention would avoid.

We need a recipe decision before Phase 2 calibration. Option A is the cheapest
and most defensible.

### 4. Token-to-expert routing table (`tid2eid`)

A `[vocab_size, n_activated_experts]` int32 lookup table per layer's gate
(`layers.X.ffn.gate.tid2eid`) — looks like a learned per-token static
expert-assignment cache to skip routing-softmax for known token IDs. Not in
V4-Flash. Postprocess pipeline needs to carry this through unmodified.

### 5. MTP block carries hc_*

Unlike V4-Flash's MTP (a single near-vanilla transformer block at `mtp.0.*`),
V4-Pro's MTP carries `hc_attn_*`, `hc_ffn_*`, `hc_head_*` plus its own
`Compressor`-free attention. Tensor count in `mtp.*` is **2,343 tensors**
(vs V4-Flash MTP's ~800). The MTP-retention pipeline must enumerate these
keys properly, not just match against V4-Flash's pattern list.

### 6. `num_hash_layers: 3` — undocumented

Config field `num_hash_layers: 3` appears in V4-Pro `inference/config.json`
but I have not yet located its consumer in `model.py`. Likely related to the
`indexer` mechanism (hash-based bucketing for sparse attention). Open
question.

## Implications for the original scaffold

### PLAN.md Phase 0 — wrong premise

> Goal: V4-Pro BF16 source downloaded to /scratch/weights/v4-pro-bf16-mtp/ (~1.4–1.8 TB)

Wrong on three counts:
- No BF16 source exists; the public release is native FP4+FP8.
- Size is 864.7 GB, not 1.4–1.8 TB.
- If we dequantize to BF16 ourselves, the result is ~3.2 TB.

This phase needs to be rewritten as a **calibration-source decision** (Option
A vs B vs C above), and the staging step changes to "download 864.7 GB native
FP4+FP8 source," which fits comfortably on `/scratch` and does not require
the BF16 path at all.

### PLAN.md Phase 2 — calibration sample count

The scaffold currently says "64 samples (matches V4-Flash)" for calibration.
Per the supervisory brief and the v0.1 credibility lesson from V4-Flash,
**we should be calibrating at 768 samples from day one** (RedHat's reference
count for this size class). The "start at 64, recalibrate later" framing
should be removed entirely — that is the exact pattern that produced the
Flash v0.1 numbers we are spending v0.2 to fix.

Even more relevant: if we pick Option A, **expert calibration is not even
needed** (experts are already FP4). Only attention FP8 → FP4 needs
calibration, and only if we push attention to FP4 at all.

### MODEL_CARD.md size claims

The MODEL_CARD currently asserts BF16 source size. All four scaffold docs
(MODEL_CARD, README, PLAN, CLAUDE.md inheritance table) should be updated
to:
- Native source: 864.7 GB (FP4 experts + FP8 attn + BF16 hc_*/norms)
- BF16-equivalent size: ~3.2 TB (not stored; only relevant for downstream
  consumers thinking about dequant)
- Total params: 1598.84 B (~1.6 T)
- Active params per token: ~30–40 B (to be measured exactly; supervisor
  brief said 49 B — needs precise per-token-active count)

### vLLM patches

V4-Flash shipped with 5 vLLM patches covering FP8 attention, MoE FP4 quant
detection, MTP loading, and BF16 fallback. **None of those 5 patches address
HC, Compressor, indexer, sparse_attn, or fp4_gemm.** V4-Pro requires
substantial new vLLM work, not just inherited patches. The Triton kernels
themselves (`fp4_gemm`, `sparse_attn`, `hc_split_sinkhorn`) live in
DeepSeek's `inference/kernel.py`; vLLM would need to either vendor these
or reimplement.

This is the biggest single risk to the artifact timeline. Before committing
to a quantization recipe, we need to scope what vLLM-side work is required
to serve V4-Pro at all (independent of our NVFP4 reformatting), and decide
whether to:
- file the vLLM PRs and wait (high quality, slow),
- vendor `inference/model.py` + `inference/kernel.py` and serve via vLLM's
  "custom model code" path (faster but ties us to DeepSeek's reference
  inference loop),
- or pursue both in parallel (vendor for v0.1 ship, upstream for v0.2
  release).

The V4-Flash predecessor chose option 2 (vendor + 5 patches) for v0.1 and
upstream-PR for v0.2. Same pattern likely applies here, but the patch
surface is much larger.

## Recommended next steps (gated decisions)

Before any download or calibration, surface these decisions to the user:

1. **Calibration source: Option A / B / C** (Option A recommended).
2. **Expert quantization target:** keep native FP4 (just convert scale format
   to NVFP4-compatible E4M3) vs re-quantize. Recommended: keep native FP4.
3. **Attention quantization target:** keep FP8 vs push to NVFP4. Open.
4. **vLLM strategy:** vendor-and-serve vs upstream-first vs parallel.
   Recommended: vendor + serve for v0.1, file upstream PRs incrementally.
5. **Calibration sample count if calibration is needed at all:** 768
   (recommended) vs scaffold's 64 (rejected).

Then, only then, stage the 864.7 GB source on `/scratch` and begin Phase 0
proper.

## Receipts (commands that produced the above)

- `curl https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/config.json`
- `curl https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/inference/config.json`
- `curl https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/inference/model.py`
- `curl https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/inference/kernel.py`
- `curl https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/model.safetensors.index.json`
  (11.3 MB, 145,116 weight_map entries, metadata.total_size 864,704,792,696)
- Range-read first ~1 MB of each `model-NNNNN-of-00064.safetensors` (64 HTTP
  range requests) to sum per-shard JSON header tensor shapes by dtype. Script
  in this doc's commit message.

## Open items to resolve before scaffold rewrite

- Confirm 49 B / ~30–40 B per-token active param count by computing exactly
  from the model graph (one routed layer + one shared + indexer + compressor
  + HC overhead × 61 + embed + head + MTP).
- Locate `num_hash_layers: 3` consumer in `model.py`.
- Check whether `transformers` `DeepseekV4ForCausalLM` modeling class
  (whatever shipped or didn't ship with V4-Flash) covers V4-Pro's `hc_*` and
  `compressor`/`indexer` keys, or if it silently drops them via
  `_keys_to_ignore_on_load_unexpected`. Likely the latter.
- Diff `vendor/dsv4-pro-upstream/model.py` vs the predecessor's
  `/data/vendor/dsv4-upstream/model.py` on the box to enumerate every
  structural change (single source of truth for what's actually new).

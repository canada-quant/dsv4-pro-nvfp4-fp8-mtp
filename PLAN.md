# PLAN.md — V4-Pro NVFP4-FP8-MTP

## Mission

Produce **`canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP`** — an NVFP4 (group=16,
FP8-E4M3 scales) conversion of DeepSeek-V4-Pro's MoE experts from the native
MXFP4 (group=32, E8M0 scales) source, with FP8 block-quant attention preserved,
hc_* / norms preserved at BF16, and the MTP block fully retained and active
under vLLM `--speculative-config method=mtp`.

This is **not** a re-application of the V4-Flash recipe. The V4-Flash recipe
calibrated from BF16. The V4-Pro source is already FP4 — the recipe here is
format conversion (MXFP4 → NVFP4) plus the vLLM-side serving path that does
not yet exist upstream.

## Source-of-truth references (read before each phase)

- `docs/findings/upstream_research_v4_pro_ground_truth.md` — 1598.84 B params,
  864.69 GB on-disk MXFP4+FP8+BF16, no public BF16 source.
- `docs/findings/option_a_lossless_audit.md` — group-size mismatch (32 vs 16);
  E8M0→E4M3 scale conversion exact with two-level S_g but requires per-tensor
  weight-stats analysis; regrouping is a re-quantization step with bounded
  loss against the source's already-FP4 values.
- `docs/findings/vllm_pro_serving_path.md` — vLLM mainline serves native
  MXFP4-FP8-MTP V4-Pro today via `vllm/models/deepseek_v4/` subpackage +
  `Mxfp4MoEMethod`. No NVFP4 path exists upstream. The official recipe is
  `recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro`.
- `vendor/dsv4-pro-upstream/` — `config.json`, `inference_config.json`,
  `model.py`, `kernel.py` from `deepseek-ai/DeepSeek-V4-Pro` for offline
  reference.

## Phase 0 — Box readiness + native MXFP4 serve smoke (1 session)

**Goal**: confirm the box can serve the native V4-Pro release before we change
anything. This is the floor — if we can't serve native, we can't validate the
converted artifact either.

Steps:

1. SSH the box, verify hardware unchanged from V4-Flash session:
   `ssh -i ~/.ssh/qwenv4-quant.pem ubuntu@35.161.108.205 'nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv'`
   Expect 8× B300, 288 GB each, compute_cap 10.3.
2. Diff the 5 V4-Flash vLLM patches against current `vllm-project/vllm@main`:
   - `gh pr view 43248 43288 43290 43319 --repo vllm-project/vllm --json state,merged,mergedAt`
   - For each: still applicable? merged? superseded by the V4-Pro subpackage
     rewrite? Result drives whether we keep, drop, or rebase each patch.
   - Output: `docs/findings/v4_flash_patches_after_pro_subpackage.md`.
3. Rebuild `/data/venv-serve` against a current vLLM main commit that includes
   the V4-Pro subpackage. Build with `TORCH_CUDA_ARCH_LIST=10.3a` for sm_103a.
   Document the commit SHA in `scripts/install_vllm_with_patches.sh`.
4. Stage native MXFP4 source: `hf download deepseek-ai/DeepSeek-V4-Pro --local-dir /scratch/weights/v4-pro-native-mxfp4-mtp/`.
   ~864.7 GB; verify `df -h /scratch` has room (free V4-Flash artifacts first
   if needed; the V4-Flash artifact is published, can be re-pulled from HF).
5. Native serve smoke with the recipe command:
   ```bash
   /data/venv-serve/bin/vllm serve /scratch/weights/v4-pro-native-mxfp4-mtp \
     --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
     --enable-expert-parallel --tensor-parallel-size 8 \
     --moe-backend deep_gemm_mega_moe \
     --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
     --port 8089
   ```
   Try without `--enforce-eager` first (PR #42604 merged); add it back if
   cuda-graph capture fails.
6. Single-prompt round-trip:
   `curl http://localhost:8089/v1/chat/completions -d '{...}'` returns a coherent
   answer. Check Prometheus `vllm:spec_decode_num_accepted_tokens_total`
   increments — confirms MTP draft path is active.
7. **Compute per-token active params** from `vendor/dsv4-pro-upstream/config.json`
   directly, do not infer from V4-Flash scaling. Math:
   - Attention (MLA): `q_lora_rank * hidden + hidden * (head_dim + qk_rope_head_dim) * num_heads + wo_a + wo_b` per active layer
   - Routed experts active: `num_experts_per_tok * 3 * hidden * moe_intermediate_size`
   - Shared expert: `3 * hidden * moe_intermediate_size`
   - Indexer + compressor: read from `config.json` (`index_n_heads * index_head_dim * 2`)
   - HC overhead: `2 * 24 * (4 * hidden)` per layer (negligible)
   - Multiply per-MoE-layer figure by 60 (61 layers with `first_k_dense_replace=1` per DSV3 convention; verify exactly), add MTP layer, attention path, embed/head
   Output: a single verified number to record in MODEL_CARD. This resolves the
   "30–40 B vs 49 B" open question before any later phase needs it.

**Patch disposition rule** (explicit to override the agent's preservation
instinct): any of the 5 V4-Flash patches superseded by the V4-Pro subpackage
work is **dropped, not rebased**. The default action when in doubt is "drop
and let the V4-Pro subpackage own the code path" — carrying obsolete patches
forward is the failure mode here, not under-preservation.

Gates (proceed to Phase 1 only when ALL pass):

- [ ] Hardware confirmed 8× B300 sm_103a
- [ ] vLLM venv rebuilt against post-V4-Pro-subpackage main
- [ ] V4-Flash patch disposition documented; superseded patches dropped, not
      rebased
- [ ] Native MXFP4 source staged
- [ ] `vllm serve` reaches "Application startup complete"
- [ ] One round-trip prompt returns sane output
- [ ] MTP acceptance counter increments (>0 accepted tokens)
- [ ] Per-token active param count computed from config.json and recorded

Risk: V4-Flash patches conflict with the V4-Pro subpackage rewrite. Mitigation:
treat them as legacy; only retain the ones still applicable to V4-Pro per the
audit. File fresh PRs for anything V4-Pro-specific that surfaces.

**Execution discipline**: report back to the supervisor after step 2 (the
V4-Flash patch diff against current main), before doing the venv rebuild in
step 3. The patch-disposition decision is the highest-uncertainty step of
Phase 0 and the one where "drop, not rebase" discipline most matters —
catching drift before a rebuild is cheaper than catching it after.

## Phase 1 — NVFP4 vs MXFP4 perf microbenchmark on B300 V4-Pro shapes (0.5–1 session)

**Goal**: measure the perf delta of NVFP4 (group=16, E4M3 scale) versus MXFP4
(group=32, E8M0 scale) GEMM on B300 sm_103a at the exact expert tensor shapes
V4-Pro uses. This is a measurement, not a gate — informs how loudly the
artifact's perf claims can be stated.

Shapes (per expert w1/w3 and w2):
- w1 / w3: `[3072, 7168]` (FP4 weight), input `[batch, 7168]`, output `[batch, 3072]`
- w2: `[7168, 3072]` (FP4 weight), input `[batch, 3072]`, output `[batch, 7168]`

Steps:

1. Identify available GEMM kernels:
   - MXFP4 (group=32): vendored `fp4_gemm` in `vendor/dsv4-pro-upstream/kernel.py`,
     vLLM's `Mxfp4MoEMethod` path through DeepGEMM `mega_moe`
   - NVFP4 (group=16): NVIDIA cuDNN, CUTLASS NVFP4 path, vLLM internal NVFP4
     (used by other architectures), or sgl-deep-gemm NVFP4 kernels if present
2. Write `scripts/bench_nvfp4_vs_mxfp4_b300.py` running both at batch=1, 4,
   16, 64, 256, sequence=512, 2048. Report TFLOPS, latency, memory bandwidth.
3. Output: `docs/benchmarks/nvfp4_vs_mxfp4_b300_v4pro_shapes.md`.

**Scope discipline**: using a CUTLASS NVFP4 kernel in the microbenchmark
is fine — Phase 1 is measurement, not implementation. **Do not let a
CUTLASS call written for Phase 1 drift into Phase 5 implementation.**
Phase 5's kernel choice is still Triton-fallback-only; any CUTLASS work
that's not measurement is Phase 8 only if Phase 6 numbers justify it.

Gates (none; measurement only). Findings inform:
- The perf claim language in MODEL_CARD ("NVFP4 is X% faster than MXFP4 on
  B300" with the actual X, or "NVFP4 matches MXFP4 within noise — ecosystem
  reach is the differentiator")
- Whether to invest in additional NVFP4-specific kernel optimization in
  Phase 5

## Phase 2 — Conversion engine: single-tensor validation (1 session)

**Goal**: prove the MXFP4 → NVFP4 conversion math on one expert weight tensor
end-to-end, with measured error.

Math (per expert weight, native FP4 group=32 → NVFP4 group=16):

1. **Unpack and dequantize**: read I8 weight tensor (shape `[out, in/2]`) plus
   E8M0 scale tensor (shape `[out, in/32]`). Each I8 byte → 2 FP4 e2m1 values
   {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Multiply each FP4 by its E8M0 scale
   (power of 2) and the per-tensor S_g if any — produces BF16-equivalent
   weight matrix of shape `[out, in]`.
2. **Reshape into NVFP4 groups**: `[out, in]` → `[out, in/16, 16]`.
3. **Per-tensor global scale**: compute `tensor_amax = max(abs(weight))`. Choose
   `S_g = tensor_amax / (FP4_max * E4M3_max)` where FP4_max=6, E4M3_max=448.
   Store `S_g` as FP32. (If `S_g < 1`, set `S_g = 1.0` and accept saturation
   for outliers — to be measured.)
4. **Per-group E4M3 scale**: for each `[16]` group, compute `group_amax = max(abs(group))`.
   Then `S_l = group_amax / (FP4_max * S_g)`, quantized to FP8-E4M3 (round to
   nearest representable E4M3 value with bias=7).
5. **Quantize each weight**: `w_quant = round(w / (S_g * S_l))` clipped to the
   FP4 e2m1 grid.
6. **Repack** as I8 with 2 FP4 per byte.
7. **Round-trip validation**: dequantize the NVFP4 result back to BF16 and
   compare against the BF16-equivalent dequant from step 1. Compute
   max-absolute-error, mean-absolute-error, max-relative-error, KL divergence
   on a softmaxed slice if applicable.

Steps:

1. Write `scripts/convert_mxfp4_tensor_to_nvfp4.py` — pure-Python +
   `torch` implementation of the above, no CUDA dependency. Run on one
   expert weight on CPU to validate correctness first.
2. Run the conversion across **100 sampled expert weight tensors** spanning:
   - Different layers (sample every 6th layer: 0, 6, 12, …, 60)
   - Different expert IDs within a layer (sample at quartile boundaries)
   - All three of w1, w2, w3 per sampled expert
   - At least 10 MTP-block experts (`mtp.0.ffn.experts.*`)
3. For each: record max-abs-err, mean-abs-err, max-rel-err, mean-rel-err
   against the BF16-equivalent dequant from step 1 of the math.
4. Output: `docs/findings/conversion_math_validation.md` with the per-tensor
   numbers and the distribution: p50, p99, max for each error metric, broken
   down by layer index and by w1/w2/w3.

Gates (proceed to Phase 3 only when ALL pass):

- [ ] Conversion script runs deterministically (same input → same output bits)
- [ ] All 100 sampled tensors complete without NaN / Inf in the output
- [ ] Error distribution is documented (p50, p99, max per metric, per
      layer-band, per w-index)

**No numeric pass/fail threshold at this phase.** This is characterization,
not a quality gate. The error budget for an MXFP4→NVFP4 format conversion
is unknown a priori; setting a number here would be inventing precision we
don't have. The forward-pass quality check is Phase 3.5; the user-facing
quality decision is read off Phase 6's benchmark numbers.

## Phase 3 — Conversion engine: full-model production (1–2 sessions)

**Goal**: convert the full V4-Pro model and write the output as a
serve-ready safetensors directory.

Tensor disposition by layer category:

| Tensor category | Source dtype | Target dtype | Action |
|---|---|---|---|
| `layers.X.ffn.experts.Y.w{1,2,3}.weight` | MXFP4 group=32 (I8 packed) | NVFP4 group=16 (I8 packed) | Convert (Phase 2 math) |
| `layers.X.ffn.experts.Y.w{1,2,3}.scale` | E8M0 (F8_E8M0) | E4M3 (F8_E4M3) | Convert per the new group=16 scales |
| `layers.X.ffn.shared_experts.w{1,2,3}.{weight,scale}` | FP8 block | FP8 block | Preserve as-is |
| `layers.X.ffn.gate.{weight,bias,tid2eid}` | BF16 / F32 / I32 | unchanged | Preserve |
| `layers.X.attn.*.{weight,scale}` | FP8 block | FP8 block | Preserve |
| `layers.X.hc_attn_*`, `layers.X.hc_ffn_*` | BF16 | BF16 | Preserve |
| `layers.X.{attn_norm,ffn_norm,input_layernorm,post_attention_layernorm}.weight` | BF16 | BF16 | Preserve |
| `layers.X.attn.{compressor,indexer}.*` | FP8 / BF16 mix | unchanged | Preserve |
| `mtp.0.ffn.experts.Y.w{1,2,3}.{weight,scale}` | MXFP4 | NVFP4 | **Convert** (consistent with main trunk; MTP fully NVFP4) |
| `mtp.0.{attn,hc_*,norm,enorm,hnorm,e_proj,h_proj}.*` | FP8 / BF16 mix | unchanged | Preserve |
| `embed.weight`, `head.weight`, `norm.weight`, `hc_head_*` | BF16 / FP32 | unchanged | Preserve |

Per-tensor S_g production strategy:

- For each weight tensor independently, compute `S_g` from weight statistics
  in step 3 of Phase 2's math.
- Store the resulting per-tensor `S_g` as a sidecar tensor (e.g.,
  `layers.X.ffn.experts.Y.w1.global_scale` of dtype F32, shape `[]`).
- Naming convention follows NVFP4's two-level scale convention used elsewhere
  in vLLM. Verify the exact suffix name once Phase 4 reads Mxfp4MoEMethod.

Steps:

1. Write `scripts/convert_v4_pro_mxfp4_to_nvfp4.py` — iterates shards, applies
   per-tensor dispatch per the table, writes new shards under
   `/scratch/weights/v4-pro-nvfp4-fp8-mtp/`.
2. Run on 1 shard first (sanity), then all 64.
3. Write a fresh `model.safetensors.index.json` summing the new shards.
4. Write the output `config.json`:
   - Copy `deepseek-ai/DeepSeek-V4-Pro/config.json` as base
   - Update `quantization_config` to a new `nvfp4` entry (exact structure
     determined by vLLM's NVFP4 config schema — fill in during Phase 4
     design read)
   - Set `expert_dtype: nvfp4` (or whatever vLLM's
     `DeepseekV4FP8Config` expects for the NVFP4 branch — we'll be adding it)
   - Keep `num_hidden_layers: 61`, `num_nextn_predict_layers: 1`,
     `hc_mult: 4`, `hc_sinkhorn_iters: 20`, all V4-Pro fields verbatim
5. Write postprocess `packed_modules_mapping` if needed for vLLM loader.
6. Verify MTP keys preserved:
   ```bash
   python -c "import json; d=json.load(open('.../model.safetensors.index.json')); print(sum(1 for k in d['weight_map'] if k.startswith('mtp.')))"
   ```
   Expect 2343.
7. Compute and record the **actual byte total** of the output directory
   (`du -sb /scratch/weights/v4-pro-nvfp4-fp8-mtp/`). The ~910 GB estimate
   in this plan is rough; the measured number is what goes in MODEL_CARD
   with no rounding.

Gates (proceed to Phase 3.5 only when ALL pass):

- [ ] All 64 output shards produced
- [ ] `model.safetensors.index.json` sums to the measured byte total
- [ ] 2343 `mtp.*` keys present
- [ ] 384 unique expert IDs per layer × 61 layers + 384 in MTP block all present
- [ ] No `quantization_config: {quant_method: fp8}` legacy keys in config.json
- [ ] Spot-check: dequant one converted expert, compare against the same
      expert dequantized from native source — distribution consistent with
      Phase 2's characterized error distribution (i.e., no order-of-magnitude
      outlier)
- [ ] Measured output byte total recorded for MODEL_CARD

Risk: conversion bug silently corrupts a subset of experts. Mitigation:
deterministic per-shard hashing of the input + output; spot-check every Nth
expert against native dequant. The Phase 3.5 forward-pass parity check is the
backstop against a silent corruption that survives spot-checks.

## Phase 3.5 — Local forward-pass parity check (1 session)

**Goal**: catch conversion bugs that don't show up as weight-stat errors but
do show up as wrong logits. Bridges the gap between "artifact exists on disk"
(Phase 3) and "artifact serves under vLLM" (Phase 5). Without this, a
serve-time logits bug surfaces only as garbled output from vLLM and is
hard-to-impossible to debug from inside the serve stack.

**HBM feasibility note**: native ~864 GB + converted ~910 GB = ~1.77 TB
of weights. On 8× B300 = 2.24 TB total HBM, that leaves ~470 GB across
8 GPUs for activations + the comparison machinery. Probably fits for a
32-token prompt at TP=8 with no KV cache, but is not guaranteed. The
agent should size feasibility against actual measured weight load
footprint (Phase 0 output) and pick one of:
- **Pattern A — simultaneous load**: both models resident, single forward
  pass each, immediate logits comparison. Faster, requires the math to
  fit.
- **Pattern B — sequential load**: load native, run forward, save logits
  to disk (BF16 tensor, ~vocab×seq×2 bytes = a few MB), unload, load
  converted, run forward, compare against disk-cached native logits.
  Slower, no OOM risk.
Decide which pattern based on the actual measured weight footprint, not
assumed footprint. Document the choice in the Phase 3.5 output.

Steps:

1. Load both models as plain torch modules using `vendor/dsv4-pro-upstream/model.py`
   (not vLLM): native MXFP4 from `/scratch/weights/v4-pro-native-mxfp4-mtp/`
   and converted NVFP4 from `/scratch/weights/v4-pro-nvfp4-fp8-mtp/`. Both
   in eval mode. Layout per the feasibility pattern chosen above.
2. Feed one short prompt (e.g., 32 tokens of "The quick brown fox") through
   both. Capture the final-layer logits as BF16 tensors.
3. Compute logits diff statistics: max-abs-diff, mean-abs-diff, max-rel-diff,
   KL(softmax(native) || softmax(converted)) over the vocab.
4. Also capture the per-token argmax for the next 8 generated tokens
   (greedy). Native and converted should agree on at least the first few
   tokens for a sane conversion.
5. Output: `docs/findings/conversion_forward_pass_parity.md` with the
   diff numbers + the first-8-token comparison.

Gates (proceed to Phase 4 only when ALL pass):

- [ ] Both models load without error
- [ ] Both models produce non-NaN logits on the same prompt
- [ ] Logits diff numbers documented (no pass/fail threshold; characterization)
- [ ] The greedy first 8 tokens are reported; mismatches are flagged for
      analysis but do not block — partial divergence is expected for a
      quantization conversion

If logits are wildly different (e.g., max-abs-diff order-of-magnitude beyond
the per-expert weight-error scale would predict), the conversion has a bug.
Diagnose before any vLLM-side work, since serving a broken artifact wastes
the Phase 4–5 design-and-implement effort.

## Phase 4 — vLLM Nvfp4MoEMethod for V4-Pro — design (1 session)

**Goal**: design the vLLM serving path for the converted artifact. No code
yet; just the architectural plan and reading of existing patterns.

Steps:

1. Read `vllm/model_executor/layers/quantization/mxfp4.py` (or equivalent):
   how does `Mxfp4MoEMethod` integrate with the V4-Pro MoE backend? What's
   the weight loading signature, what's the forward signature?
2. Read whether vLLM already has any NVFP4 quant method (it does for other
   models — search `Nvfp4MoEMethod` / `nvfp4`). Determine reusability for
   the V4-Pro MoE path.
3. Read `vllm/models/deepseek_v4/quant_config.py` `get_quant_method` —
   identify where to add the `expert_dtype="nvfp4"` branch.
4. Read `vllm/models/deepseek_v4/compressor.py` — the MXFP4-fused-kernel
   variant of `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`
   must have an NVFP4 sibling if attention is touched in NVFP4 mode.
   Since we keep attention FP8 unchanged, this hook MAY not be required —
   verify.
5. Read DeepGEMM's NVFP4 GEMM API (vLLM's vendored `vllm.third_party.deep_gemm`
   or external `deep_gemm` / `sgl-deep-gemm`). Identify whether the
   `mega_moe` kernel supports NVFP4 weights today or needs a new path.
6. Output: `docs/findings/vllm_nvfp4_v4pro_design.md` — the design doc.
   Sections: weight loading contract, scale-tensor naming, forward kernel
   choice, compatibility with `--speculative-config method=mtp`, list of
   files added/modified with a **line-count estimate** per file. ~2000
   net new lines is a rough sanity-check anchor (based on comparable
   quant-method additions to vLLM), not a tripwire — the supervisor
   reviews proportionality. State the estimate, what's in it, what's
   deferred to Phase 8, and let the supervisor call whether Phase 5
   begins as scoped or gets rescoped.
7. **Maintainer pre-ping** (intel, not a gate): post a short discussion
   on the vLLM repo (or a short note on PR #40760) tagging
   `@WoosukKwon @zyongye @ivanium` asking whether an `Nvfp4MoEMethod` for
   V4-Pro experts is in-progress internally, would be welcomed as a
   contribution, or whether they have a different design in mind. Cost
   is five minutes; value is potentially saving weeks of Phase 5 work
   that goes in the wrong direction. Phase 5 proceeds regardless of
   their response — silence = proceed.

Gates (proceed to Phase 5 only when ALL pass):

- [ ] Design doc reviewed and accepted (by user)
- [ ] New code organized to mirror existing quant-method layout (new
      `Nvfp4MoEMethod` lives in clearly-named files; no cramming into
      unrelated modules)
- [ ] Design does not conflict with paths actively being modified by
      currently-open V4-Pro PRs (re-run `gh pr list` for any DSV4/V4-Pro
      PRs created or updated since Phase 0 to confirm)
- [ ] NVFP4 GEMM choice locked (DeepGEMM, vLLM-vendored Triton, or external
      `deep_gemm` / `sgl-deep-gemm`; **no CUTLASS-from-scratch path at
      this phase** — see Phase 5 fall-back)
- [ ] Line-count estimate recorded in design doc with breakdown
      (what's in it; what's deferred to Phase 8); supervisor reviews
      proportionality before Phase 5 begins
- [ ] Scale-tensor naming convention finalized; Phase 3 conversion script
      regenerates artifact if naming changes
- [ ] Maintainer pre-ping sent (response not required to proceed)

## Phase 5 — vLLM Nvfp4MoEMethod implementation + serve smoke (2–3 sessions)

**Goal**: implement the design and load the converted artifact.

Steps:

1. Implement `Nvfp4MoEMethod` per the design.
2. Add `expert_dtype="nvfp4"` branch in `DeepseekV4FP8Config.get_quant_method`.
3. If a new Compressor variant is needed, implement.
4. Build the patched vLLM into `/data/venv-serve` (or a sibling
   `/data/venv-serve-nvfp4`).
5. Serve smoke:
   ```bash
   /data/venv-serve-nvfp4/bin/vllm serve /scratch/weights/v4-pro-nvfp4-fp8-mtp \
     --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
     --enable-expert-parallel --tensor-parallel-size 8 \
     --moe-backend deep_gemm_mega_moe \
     --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
     --port 8090
   ```
6. Round-trip prompt verifies coherent output.
7. MTP draft counter increments.

Gates (proceed to Phase 6 only when ALL pass):

- [ ] Converted artifact loads without errors
- [ ] Round-trip prompt sane (coherent grammar, on-topic response)
- [ ] MTP `vllm:spec_decode_num_accepted_tokens_total > 0`

**No perf gate at Phase 5.** Throughput is a Phase 6 measurement, not a
Phase 5 gate. Even a slow NVFP4 path proves correctness here; perf framing
is decided after the Phase 6 numbers are in. (The earlier "within 2×" gate
was both too generous to be meaningful and wrong-direction as a gate.)

Risk: NVFP4 GEMM kernel doesn't exist for the exact V4-Pro expert shapes on
sm_103a. Mitigation at Phase 5: **Triton fallback only**. Use a Triton NVFP4
GEMM at whatever TFLOPS it lands — correct-but-slow is acceptable for Phase 5
gating. If Phase 6 shows the Triton fallback is unacceptably slow for the
ship case, that becomes a Phase 8 follow-up (custom kernel work), not a
Phase 5 task. Writing CUTLASS or hand-tuned PTX kernels at Phase 5 expands
scope by multiple weeks and is the wrong place for that work.

## Phase 6 — Benchmarks (1–2 sessions)

**Goal**: produce the measurement table for MODEL_CARD. Same methodology as
V4-Flash but applied to V4-Pro.

Benchmarks (lock `max_tokens=65536` for all thinking-mode runs; report raw
AND non-truncated pass@1 per the V4-Flash lesson):

- [ ] AIME 2024 thinking=high, 3-way: ours-NVFP4-MTP / ours-NVFP4-no-spec /
      native-MXFP4-MTP
- [ ] GSM8K strict 8-shot (matches PR #42844 test plan format)
- [ ] MMLU-Pro 5-shot
- [ ] HumanEval EvalPlus pass@1
- [ ] IFEval prompt-strict + loose
- [ ] Chat-template coding sweep c=1, 4, 8, 16 — capture MTP acceptance
      per concurrency cell
- [ ] Wall-clock decode latency: NVFP4 vs MXFP4 baseline at c=1, batch=1,
      median over 100 prompts (report median + p95 + std)
- [ ] MTP draft acceptance rate: NVFP4 vs MXFP4 baseline on AIME-reasoning
      and chat-coding workloads

Gates (proceed to Phase 7 only when ALL pass):

- [ ] All listed benchmarks completed without harness errors
- [ ] Numbers reported with confidence intervals where applicable
      (n, mean, 95% CI for pass@1 metrics; median + p95 + std for
      throughput/latency)
- [ ] NVFP4 vs native-MXFP4 deltas tabulated per benchmark
- [ ] Per-benchmark methodology documented (sample count, max_tokens,
      temperature, thinking-mode discipline) in
      `docs/benchmarks/<benchmark>_<date>.md`

**No numeric quality threshold gates at Phase 6.** This is characterization,
not a pass/fail. The ship decision in Phase 7 is read off the data by the
user — auto-triggering a ship at "within 3%" is the v0.1 Flash failure mode
re-templated. If the numbers come in worse than expected, the response is
to surface them honestly in MODEL_CARD and let the user decide whether the
artifact is ship-ready as-is, ship-ready with a noted gap, or held for a
v0.2 iteration. No threshold the agent invented should auto-decide that.

## Phase 7 — Upstream PRs + HF publish (parallel, multi-session)

Upstream-side, file in order (vLLM PR first because everything else
references a working serve target on mainline):

1. **vLLM PR**: `Nvfp4MoEMethod` + `expert_dtype="nvfp4"` branch in
   `DeepseekV4FP8Config`. Body carries the Phase 6 benchmark numbers as
   evidence. Tag `@WoosukKwon @zyongye @ivanium` per the PR #40760
   maintainer note (and acknowledge the prior pre-ping if any). Land this
   first — the rest of the contributions depend on a working serve target
   that's referenceable from mainline.
2. **Recipe-page contribution**: only after the vLLM PR has review traction
   (not necessarily merged). Add an "NVFP4 V4-Pro" section to
   `recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro` referencing the PR + the
   HF artifact path + the serve command.
3. **DeepGEMM upstream**: only if a new NVFP4 mainloop kernel was actually
   written in Phase 5 (Triton fallback alone doesn't qualify). File in
   `deepseek-ai/DeepGEMM` per the #42845 / #338 precedent.
4. **transformers**: revisit `transformers#46127` (V4-Flash MTP-strip).
   Likely no V4-Pro-specific addition needed since vLLM owns the V4-Pro
   loader; verify and close the loop.

HF publish, gated on user authorization (separate decision from upstream
PR status — the artifact ships when the user reads the Phase 6 numbers and
calls it):

- [ ] MODEL_CARD.md filled with measurements (numbers verbatim from Phase 6
      output, no rounding)
- [ ] README.md mirrors MODEL_CARD voice
- [ ] License `mit` (matches upstream DSV4-Pro)
- [ ] Tags: `nvfp4 fp8 vllm deepseek mtp speculative-decoding mixture-of-experts compressed-tensors deepseek-v4-pro`
- [ ] Clean download from HF + serve from downloaded path produces working
      smoke
- [ ] Public flip of `canada-quant/dsv4-pro-nvfp4-fp8-mtp` GitHub repo

## Phase 8 — Custom kernel optimization (optional follow-up)

**Goal**: only triggered if Phase 6 shows the Triton NVFP4 fallback is
unacceptably slow for the production use case. Not gated, not scheduled —
opened as a follow-up only if measurement justifies it.

If pursued, scope is:
- Write CUTLASS-based or hand-tuned PTX NVFP4 GEMM kernels for V4-Pro
  expert shapes specifically (`w1`/`w3` `[3072, 7168]`, `w2` `[7168, 3072]`)
- Target sm_103a (B300) first; sm_100a (B200) as a fast follow
- File as a separate vLLM contribution; this is a multi-week project that
  must not bleed back into Phase 5's correctness scope

Out-of-scope unless Phase 6 numbers force it.

Voice rules:
- No "first to..." framing — be factual.
- No emojis.
- Lead with what the artifact IS and the measurements table; positioning
  follows.
- No mention of any related V4-Pro-class work outside this repo's scope.

## Risk register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| V4-Flash patches conflict with V4-Pro subpackage on rebuild | Phase 0 blocked | high | Diff first, retire patches that are obsolete. Net work likely smaller than carrying 5 patches forward. |
| NVFP4 GEMM kernel for V4-Pro shapes does not exist on sm_103a | Phase 5 perf is poor | medium | Triton fallback at Phase 5 (correctness-only; perf is not gated at Phase 5). CUTLASS-based custom kernel is Phase 8 only, if Phase 6 numbers justify it. Ship Phase 7 with documented perf gap; don't block the artifact on a custom kernel. |
| Conversion math has hidden bug that corrupts a subset of experts | Phase 6 quality drops | medium | Per-shard hashing + spot-check against native dequant; full BF16-dequant comparison before publish. |
| Native MXFP4 V4-Pro doesn't serve on our box's vLLM build | Phase 0 blocked | low | Rebuild against current main; this is expected work, not a true risk. |
| MTP draft acceptance drops in NVFP4 vs native MXFP4 | speedup story weakens | low–medium | NVFP4 weights are conversion of the same FP4 values; acceptance should match within noise. If not, investigate per-layer error and adjust per-tensor S_g policy. |
| TP=8 doesn't fit NVFP4 V4-Pro KV cache at full context | can't serve long contexts | low | NVFP4 weight footprint ~ same as MXFP4 (extra scales offset by no other delta) = ~108 GB/GPU. 288 GB - 108 GB = 180 GB / GPU for KV + activations. Should be ample. |
| Recipes.vllm.ai maintainers won't accept an "NVFP4-V4-Pro" recipe | Phase 7 upstream blocked | medium | Ship the artifact + own documentation regardless; upstream is a parallel contribution, not a gate. |
| zyongye / Inferact pushback on Nvfp4MoEMethod design | vLLM PR delayed | medium | Phase 4 maintainer pre-ping is intel for exactly this. Engage early with the design doc, not the PR. Their model expertise improves the design; their goodwill is needed for the merge. If pre-ping reveals they have a competing internal design, pause and coordinate before writing Phase 5 code. |
| Phase 4 design line-count estimate balloons past the ~2000-line sanity anchor | scope creep silently expands | medium | Phase 4 gate requires explicit estimate with breakdown (what's in it, what's deferred to Phase 8). Supervisor reviews proportionality; ~2000 is an anchor, not an auto-tripwire. |
| NVFP4 perf delta vs MXFP4 turns out flat or negative on B300 | claim language constrained | medium | Already anticipated. Phase 1 microbenchmark settles this. Artifact still ships; framing shifts from "perf win" to "ecosystem reach / NVIDIA-native format compatibility". |
| Phase 7 HF upload reveals safetensors metadata mismatch | publish delayed | low | Validate locally via fresh HF download + load before pushing public. |

## Out of scope

- W4A16 GPTQ recipe for V4-Pro (separate repo if needed)
- Other DeepSeek architectures
- Pre-training-time recalibration (we don't have the BF16 master)
- Building a brand-new vLLM-from-scratch fork (we contribute to mainline)
- "First to" / "the only" framing in any public copy

## Open questions outstanding through all phases

1. NVFP4 vs MXFP4 perf on B300 sm_103a (answer in Phase 1)
2. Exact NVFP4 scale-tensor naming convention used by vLLM (answer in Phase 4)
3. Whether MTP draft acceptance is preserved across MXFP4 → NVFP4 conversion
   (answer in Phase 6)
4. Whether `num_hash_layers: 3` (V4-Pro config field) affects conversion
   (answer in Phase 2 — likely no, it's an indexer parameter not a weight
   storage parameter)

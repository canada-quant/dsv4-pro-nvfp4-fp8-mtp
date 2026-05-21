# PLAN.md — V4-Pro NVFP4-FP8 with MTP retention

Phase-by-phase plan with measurable gates. Derived from the V4-Flash plan that shipped 2026-05-21; deltas noted.

## Phase 0 — Pre-flight (1–2 hours)

**Goal**: confirm hardware, software, and source weights are ready. Do NOT proceed until every check passes.

Gates:
- [ ] B300 compute_cap = 10.3 confirmed
- [ ] `/data/venv-calib` exists and `compressed-tensors==0.15.1a20260515`, `llmcompressor` at the right SHA
- [ ] `/data/venv-serve` vllm version matches the V4-Flash-validated build OR is rebuilt with patches
- [ ] `deepseek-ai/DeepSeek-V4-Pro` config.json fields match `CLAUDE.md` table (no surprise changes from upstream since 2026-05-21)
- [ ] V4-Pro BF16 source downloaded to `/scratch/weights/v4-pro-bf16-mtp/` (~1.4–1.8 TB). If `/scratch` doesn't have room, free up V4-Flash's `bf16-mtp` first (we have the artifact; the BF16 source can be re-staged from S3 if needed)
- [ ] V4-Pro vendored model.py and kernel.py in `vendor/dsv4-pro-upstream/` — verify against `deepseek-ai/DeepSeek-V4-Pro`'s `inference/` dir; may be identical to V4-Flash's vendored copy

**Surface area for new gotchas**: V4-Pro's larger hidden_size (7168 vs 4096) and routed-expert count (384 vs 256) may exercise code paths V4-Flash didn't. Watch for OOM, kernel-tiling mismatches, sharded-MoE save coordination issues.

## Phase 1 — 1-layer + 4-rank dryrun (THE GATE) (30 min – 2 hours)

**Goal**: prove the recipe survives one layer of V4-Pro calibration before committing to the full multi-hour run.

```bash
python scripts/quantize_v4_pro_nvfp4_fp8_mtp.py \
    --dry-run-one-layer --layer-idx 5 \
    --samples 16 --max-seq-len 512 --batch-size 1
```

Gates:
- [ ] Layer 5 calibrates without crashing
- [ ] No NaN / Inf in the saved scales
- [ ] Same `_keys_to_ignore_on_load_unexpected` patch from V4-Flash still strips MTP correctly during the load path
- [ ] 4-rank torchrun completes the layer without hangs (verifies multi-rank-MoE save coordination still works at V4-Pro's expert count of 384)

Failure modes to expect: out-of-memory at 4-rank with V4-Pro's larger experts (may need to pivot to 1-rank like V4-Flash did), Observer.synchronize hang (V4-Flash workaround was monkey-patch), expert-sharding coordination edge cases.

## Phase 2 — Full calibration (3–8 hours, depending on rank count)

**Goal**: produce the calibrated NVFP4-FP8 artifact with MTP block preserved at BF16.

```bash
# 1-rank baseline (most reliable, slowest):
python scripts/quantize_v4_pro_nvfp4_fp8_mtp.py \
    --output-dir /scratch/weights/v4-pro-nvfp4-fp8-mtp \
    --samples 64 --max-seq-len 512 --batch-size 1
# Expect: 4–8 hours on B300 1-rank

# Or 8-rank multi-rank (3–6× faster if it doesn't deadlock; gated on llm-compressor#2743 fix):
torchrun --nproc-per-node 8 scripts/quantize_v4_pro_nvfp4_fp8_mtp.py \
    --output-dir /scratch/weights/v4-pro-nvfp4-fp8-mtp \
    --samples 64
```

**Calibration sample count**: 64 matches V4-Flash for consistency. Open question: bump to 768 (RedHat's reference) to address the AIME truncation-rate concern from V4-Flash? **Recommendation**: start at 64 (faster bring-up, can iterate later); if AIME truncation rate is bad, recalibrate at 768 for v0.2.

Gates:
- [ ] Artifact directory has ~50–100 safetensors shards (V4-Pro is bigger than V4-Flash's 35)
- [ ] `model.safetensors.index.json` has expected total key count (will be much higher than V4-Flash's 134k due to more layers + more experts)
- [ ] 384 unique expert IDs present
- [ ] MTP keys preserved: `grep mtp model.safetensors.index.json | wc -l` should be ≥ 800
- [ ] `config.json` has `quantization_config.format = "mixed-precision"` and both group regexes present

## Phase 3 — Postprocess for vLLM (10–30 min)

```bash
python scripts/postprocess_for_vllm.py /scratch/weights/v4-pro-nvfp4-fp8-mtp
python scripts/squeeze_global_scales.py /scratch/weights/v4-pro-nvfp4-fp8-mtp
```

Gates:
- [ ] `num_hidden_layers: 61` in postprocessed `config.json` (NOT 43 like V4-Flash, NOT 60 — verify against `deepseek-ai/DeepSeek-V4-Pro` config)
- [ ] `num_nextn_predict_layers: 1`
- [ ] `expert_dtype: fp4`
- [ ] `scale_fmt: ue8m0` injected (until our PR #43288 merges upstream)
- [ ] `packed_modules_mapping` present
- [ ] `re:.*\.layers\.61\..*` in the `ignore` list so MTP draft model construction doesn't apply NVFP4 quant to the MTP block (61 here, not 43 — adjust for V4-Pro's layer count)

## Phase 4 — Verify MTP retention

```bash
python scripts/verify_mtp_keys.py /scratch/weights/v4-pro-nvfp4-fp8-mtp
python scripts/verify_mtp_quantized.py /scratch/weights/v4-pro-nvfp4-fp8-mtp
```

Gates:
- [ ] ≥ 800 keys matching `re:^mtp\..*` present
- [ ] All MTP keys are BF16 or FP32 — **zero** FP8/NVFP4 dtypes
- [ ] Attention weight scale shapes match (hidden_size / 128, k_dim / 128) per FP8_BLOCK 128×128

## Phase 5 — vLLM serve smoke

```bash
# Adjust TP based on Phase 0 memory test result. V4-Pro likely TP=8 due to size.
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve /scratch/weights/v4-pro-nvfp4-fp8-mtp \
  --tensor-parallel-size 8 --port 8089 --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Gates:
- [ ] `Application startup complete` in log
- [ ] Chat smoke (curl `/v1/chat/completions`) returns a sensible response
- [ ] Prometheus `vllm:spec_decode_num_accepted_tokens_total` increments after a request

## Phase 6 — Benchmarks (4–8 hours total)

Mirror the V4-Flash methodology. Lock `max_tokens=65536` for any thinking-mode benchmark. Report raw + non-truncated pass@1.

- [ ] AIME 2024 thinking=high, 3-way (ours-MTP / ours-no-spec / BF16-MTP if memory allows)
  - Expect higher absolute pass@1 than V4-Flash (V4-Pro is more capable)
  - Truncation rate should be ≤ V4-Flash's 5/30 if calibration coverage is adequate
- [ ] GSM8K strict + flexible (8-shot)
- [ ] MMLU-Pro (5-shot)
- [ ] HumanEval EvalPlus pass@1
- [ ] IFEval prompt-strict + loose
- [ ] Chat-template coding sweep c=1, 4, 8, 16 — capture MTP acceptance per cell
- [ ] BF16 + MTP reference on the same hardware (TP=8 needed for V4-Pro BF16 — may be too large to fit even at TP=8, in which case defer)

## Phase 7 — HF upload (gated on user authorization)

- [ ] MODEL_CARD.md complete with all measurements
- [ ] README.md mirrors MODEL_CARD voice
- [ ] License = `mit` (matches upstream DSV4-Pro)
- [ ] Tags include `nvfp4 fp8 vllm deepseek mtp speculative-decoding mixture-of-experts compressed-tensors`
- [ ] Sanity check: clean download from HF + serve from downloaded path produces working smoke

## Risk register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| V4-Pro BF16 source doesn't fit in /scratch | can't calibrate | high | Free V4-Flash bf16-mtp dir first; or use S3 streaming |
| Calibration OOM at 1-rank on V4-Pro | can't even produce single-rank artifact | medium | Per-layer offload; reduce samples; sequential pipelining |
| 8-rank torchrun deadlocks (same as V4-Flash hit) | back to slower 1-rank | high | Use 1-rank as baseline; investigate `propagate_error` fix from llm-compressor #2008/#2743 only if performance gate matters |
| V4-Pro modeling class has new MTP layout | postprocess fails on key naming | medium | Verify with `python -c "import safetensors; ..." | grep mtp` before postprocess |
| vLLM mainline doesn't support V4-Pro's larger hidden_size in NVFP4 MoE kernel | serve fails | low–medium | Verify by trying the same vLLM build that works for V4-Flash; if it fails, the kernel issue surfaces a new vLLM PR opportunity |
| TP=8 still doesn't fit V4-Pro (~1.5 TB BF16 → ~370 GB at NVFP4-FP8, 8× B300 = 2.3 TB so should fit) | can't serve | low | Verify with vllm serve smoke; pivot to TP=4 + offload if it fails |
| Acceptance rate on V4-Pro reasoning drops vs V4-Flash | speedup story weakens | low | V4-Pro should match or exceed V4-Flash acceptance (more capable model + same recipe) |

## Out of scope for this repo

- W4A16 GPTQ recipe — sibling repo if needed
- Other architectures
- vLLM-side kernel patches — already validated on V4-Flash; if V4-Pro surfaces a new one, file as a fresh PR

## Differentiator (final positioning, when artifact ships)

Same structure as V4-Flash, just at V4-Pro scale. Lead with what the artifact IS:

- Same NVFP4-FP8 quantization math as RedHat's V4-Pro NVFP4 (if they've shipped one)
- MTP block (`mtp.0.*`, ~800 tensors at BF16) retained
- Serve-loadable with `--speculative-config method=mtp`
- Measured quality parity vs BF16 reference
- Measured wall-clock speedup vs no-MTP

**No "first to..." framing.** No emojis. Lead with measurements.

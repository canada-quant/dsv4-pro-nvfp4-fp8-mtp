---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Pro
tags:
  - compressed-tensors
  - nvfp4
  - fp8
  - vllm
  - deepseek
  - mtp
  - speculative-decoding
  - mixture-of-experts
library_name: vllm
---

# canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP

**TEMPLATE — fill in measurements as phases complete. Mirror the predecessor [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) MODEL_CARD voice exactly. No emojis, no "first to..." framing, lead with what the artifact IS.**

A DeepSeek-V4-Pro NVFP4-FP8 quantization that retains the MTP (multi-token-prediction) block in the saved weights, so vLLM can load it with `--speculative-config method=mtp`.

## What this is

- **TBD** GB across **TBD** safetensors shards (vs ~1.4–1.8 TB BF16 source, MTP block included).
- Same quantization scheme as the V4-Flash predecessor: NVFP4 (group=16, FP8 e4m3 scales) on routed FFN experts, FP8_BLOCK 128×128 on attention.
- MTP block (`mtp.0.*`, ~800 tensors) kept at BF16 — not dropped at load time, not double-quantized when the MTP draft model is constructed.

That last point is the only structural difference from any V4-Pro NVFP4 artifact that runs through stock HF transformers' load path (which strips `mtp.*` keys via `_keys_to_ignore_on_load_unexpected`). We patched the modeling class during calibration so MTP made it through.

## Headline measurements (TBD)

All numbers measured on **TBD** × B300 SXM6 AC. Quant configs at TP=**TBD**, BF16 reference at TP=**TBD**. Same prompts, same temperature 0, chat template applied server-side.

| Benchmark | This artifact | BF16 + MTP reference | RedHat V4-Pro NVFP4 (if shipped) |
|---|---|---|---|
| AIME 2024 raw pass@1 (thinking=high, max_tokens=65536) | TBD | TBD | TBD |
| AIME 2024 non-truncated pass@1 | TBD | TBD | TBD |
| AIME 2024 wall-clock (30 problems, c=8) | TBD | TBD | TBD |
| MTP draft acceptance, AIME reasoning | TBD | TBD | n/a |
| GSM8K strict-match (8-shot) | TBD | TBD | TBD |
| MMLU-Pro (5-shot) | TBD | TBD | TBD |
| HumanEval pass@1 (EvalPlus) | TBD | TBD | TBD |
| IFEval prompt-strict | TBD | TBD | TBD |

(Methodology note for the eventual AIME writeup: equalize `max_tokens` across all configs being compared, report raw + non-truncated pass@1 separately. Lesson from V4-Flash.)

## Wall-clock vs RedHat (TBD)

(Mirror the V4-Flash table structure once measurements are in.)

## MTP draft acceptance per workload (TBD)

| Workload | Acceptance |
|---|---|
| Random prompts (1024 in / 512 out) | TBD |
| Raw code completion (HumanEval `/v1/completions`) | TBD |
| Chat-templated code (HumanEval `/v1/chat/completions`, c=1) | TBD |
| Chat-templated code, c=4 / c=8 / c=16 | TBD |
| Instruction following (IFEval) | TBD |
| AIME 2024 reasoning (thinking=high) | TBD |

## Recommended serving config (TBD — verify with load test)

TP=**TBD** on **TBD**× B300 SXM6 (288 GB HBM3e per GPU). Per-rank load test in Phase 0 should establish whether TP=4 or TP=8 is the right operating point. V4-Flash's TP=4 finding was MoE-saturation-driven; V4-Pro's larger experts may invert the result.

## Quick start

See [`docs/QUICKSTART.md`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/QUICKSTART.md) once the source repo is public, or use the one-line installer:

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

Serving:

```bash
# With MTP spec-decode
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  --tensor-parallel-size <TBD> \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

## Quantization recipe

| Group | Modules | Scheme | Format |
|---|---|---|---|
| attention | `wq_a, wq_b, wkv, wo_a, wo_b` (and fused variants) | FP8_BLOCK 128×128, weight static + input dynamic FP8 group=128 | `float-quantized` |
| experts | `w1, w2, w3` per expert | NVFP4 group=16, weight static + input dynamic="local" FP4 group=16 | `nvfp4-pack-quantized` |
| ignored | `lm_head`, `embed_tokens`, norms, `ffn.gate`, `ffn.shared_experts`, attn `compressor`, attn `indexer`, `attn_sink`, `hc_*` | unquantized (BF16) | n/a |
| MTP block (`mtp.0.*`) | all ~800 keys | unquantized (BF16, preserved verbatim) | n/a |

Calibration corpus: HuggingFaceH4/ultrachat_200k train_sft, **TBD samples** × max_seq_len 512 × batch_size 1, seed 42. (Decide 64 vs 768 in Phase 2 — see PLAN.md.)

## vLLM patches required

Same 5 patches as the V4-Flash predecessor, applied automatically by the one-line installer. If any have merged upstream by the time V4-Pro ships, this list shrinks.

1. vLLM [#43248](https://github.com/vllm-project/vllm/pull/43248) — `bool()` wrap on `is_static_input_scheme`
2. vLLM [#43288](https://github.com/vllm-project/vllm/pull/43288) — `.get("scale_fmt", "ue8m0")` + BF16 `getattr` follow-up
3. vLLM [#43290](https://github.com/vllm-project/vllm/pull/43290) — `weight_scale_inv`-or-`weight_scale` fallback
4. vLLM [#43319](https://github.com/vllm-project/vllm/pull/43319) — MTP-quant-detect + BF16 `wo_a` fallback path
5. transformers [#46127](https://github.com/huggingface/transformers/pull/46127) — sibling PR for `DeepseekV4NextNPredictor`

## License

MIT, inherited from `deepseek-ai/DeepSeek-V4-Pro`.

## Acknowledgments

- DeepSeek for V4-Pro and the MTP architecture.
- The V4-Flash predecessor recipe and its measured 81.6% / 88% MTP acceptance numbers that established the pattern this artifact extends.
- vLLM, llm-compressor, compressed-tensors maintainers.
- PR #42209 contributors (sychen52, xinli-sw, pavanimajety, zyongye) for the DSV4 NVFP4 MoE kernel work.

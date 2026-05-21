# dsv4-pro-nvfp4-fp8-mtp

Source repo for the upcoming DeepSeek-V4-Pro NVFP4-FP8 quantization artifact at `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` (target HF repo, not yet created).

This is the V4-Pro scale-up of the V4-Flash recipe shipped at [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) on 2026-05-21. Same architecture class (`DeepseekV4ForCausalLM`), same quantization scheme (NVFP4 experts + FP8_BLOCK 128×128 attention + BF16 MTP block), scaled up to V4-Pro's 61 layers / 7168 hidden / 384 routed experts.

**Status**: Phase 0 (pre-flight) — no measurements yet. See [`PLAN.md`](PLAN.md) for the phase plan and [`CLAUDE.md`](CLAUDE.md) for the inherited context.

## What this will produce

The same kind of artifact as V4-Flash, at V4-Pro scale:

- NVFP4 (group=16, FP8 e4m3 scales) on routed FFN experts
- FP8_BLOCK 128×128 on attention projections
- MTP block (`mtp.0.*`) preserved at BF16, not stripped at load time, not double-quantized
- Loadable via vLLM with `--speculative-config method=mtp` for measured spec-decode speedup

## V4-Pro vs V4-Flash architecture

| Field | V4-Flash | V4-Pro |
|---|---|---|
| num_hidden_layers | 43 | **61** |
| hidden_size | 4096 | **7168** |
| n_routed_experts | 256 | **384** |
| num_experts_per_tok | 6 | 6 |
| moe_intermediate_size | 2048 | **3072** |
| num_nextn_predict_layers | 1 | 1 |

Approximate sizes:
- V4-Flash: ~284 B parameters, 543 GB BF16 source, 172 GB NVFP4 artifact
- V4-Pro: ~700–900 B parameters (estimate), ~1.4–1.8 TB BF16 source expected, ~370–470 GB NVFP4 artifact expected

## Repo layout

```
CLAUDE.md                        — session notes / agent handoff (gitignored from public push)
PLAN.md                          — phase plan with measurable gates
SYSTEM_PROMPT.md                 — starter prompt for the next agent
MODEL_CARD.md                    — fill in as phases complete
patches/
  modeling_deepseek_v4.py.diff   — transformers patch (removes MTP key strip during load)
scripts/
  install_vllm_with_patches.sh   — one-line installer (carries the 5 V4-Flash patches)
  quantize_v4_pro_nvfp4_fp8_mtp.py  — calibration entry (adapt num_hidden_layers, MoE shape)
  postprocess_for_vllm.py        — config + key surgery for vLLM compatibility
  verify_mtp_keys.py             — confirm MTP keys present
  verify_mtp_quantized.py        — confirm MTP weights are NOT quantized (BF16 pass-through)
docs/
  QUICKSTART.md                  — end-to-end serve recipe (template)
  VLLM_SETUP_ISSUES.md           — 5 patches + 14 gotchas catalog (carries from V4-Flash)
  FINDINGS.md                    — index of methodology/diagnostic notes
  benchmarks/                    — per-benchmark write-ups (empty until measured)
  findings/                      — methodology notes (empty)
  recipes/
    nvfp4_fp8_mtp_replication.md — the architecture-agnostic recipe doc
vendor/dsv4-pro-upstream/        — vendored upstream model.py / kernel.py / config.json (TBD)
memory/MEMORY.md                 — pointers to inherited findings
```

## Predecessor reference

Everything in this repo is the V4-Pro re-application of the V4-Flash recipe. When in doubt, **read the predecessor**:

- Source: https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp
- Artifact: https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP
- Local checkout: `/home/paul/dsv4-flash-nvfp4-fp8-mtp/`

## License

MIT, inherited from upstream `deepseek-ai/DeepSeek-V4-Pro`.

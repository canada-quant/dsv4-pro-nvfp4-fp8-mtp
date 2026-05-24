# v12 conversion: NVIDIA V3.2-NVFP4 recipe applied + critical loader-detector bug fixed

**Date:** 2026-05-24
**Status:** Conversion + fixes complete; runtime serving blocked by silent worker crash in rebuilt mainline (affects both v0.2 and v12 — independent of our patches)

## Research findings (Session 5)

### NVIDIA's V3.2-NVFP4 reference recipe (critical insight)

`nvidia/DeepSeek-V3.2-NVFP4`'s `hf_quant_config.json` exposes the canonical NVFP4 + MTP pattern. Key entries from their `exclude_modules` list:

- For every layer 0-60: `self_attn.indexer*`, `self_attn.kv_a_proj_with_mqa`, `self_attn.kv_b_proj`, `self_attn.q_a_proj`, `self_attn.q_b_proj` — ALL attention submodules excluded.
- **`model.layers.61*` — entire MTP layer excluded, no NVFP4 transformation at all.**

The conversion is **MoE-experts-only NVFP4**. Trunk attention stays at FP8. MTP block stays at its native format (FP8 attn + MXFP4 experts + FP8 e_proj/h_proj). No dequantization of MTP weights to BF16.

Reference: `https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4/raw/main/hf_quant_config.json`

### Our prior conversions were all wrong vs this recipe

| Artifact | Trunk experts | MTP transformation | Note |
|---|---|---|---|
| v0.2 | NVFP4 (transcoded) | MTP experts MXFP4→NVFP4 (transcoded) + e_proj/h_proj BF16 (dequanted) | Perturbs MTP head's training-time activation distribution |
| v0.3 | NVFP4 (transcoded) | ENTIRE mtp.0 dequanted to BF16 | Same perturbation, plus loss of FP8 precision in mtp attn |
| v0.4-bisect | MXFP4 passthrough | ENTIRE mtp.0 dequanted to BF16 | Same MTP perturbation |
| **v12 (this session)** | **NVFP4 (transcoded)** | **ZERO transformation — every mtp.* tensor byte-passthrough from native** | **Matches NVIDIA recipe** |

### Bug found and fixed in our `_mtp_block_is_quantized_on_disk` (PR #43319)

The detector scanned the safetensors index for keys ending in `.weight_scale`, `.weight_scale_inv`, `.weight_packed`, etc. — i.e. the POST-rename forms.

But native DSV4 artifacts (and our v12 passthrough) ship with `mtp.0.attn.wq_a.scale` — the RAW `.scale` suffix. The detector never matched it → returned False → `quant_config = None` for the entire MTP block → `ReplicatedLinear` for `e_proj`/`h_proj` constructed as `UnquantizedLinearMethod` → NO `weight_scale_inv` parameter registered → `KeyError: 'model.layers.61.e_proj.weight_scale_inv'` at load.

**Fix:** added `.scale` to the suffix list in `_mtp_block_is_quantized_on_disk`. Now the detector returns True for native and v12-style passthrough artifacts.

Verified via instrumentation: after the fix, the MTP detection log shifts from `"MTP block weights are BF16 on disk"` (incorrect for native) to silent-pass (correct — `quant_config = global FP8`), and `model.layers.61.e_proj.weight_scale_inv` IS now registered (`MTP draft model loaded: 39 params` vs prior 29 BF16-only).

This is a real upstream bug worth filing once we can demonstrate the loader path completes end-to-end.

### v12 conversion artifact

Built `/opt/dlami/nvme/weights/v4-pro-v12-strict`:
- Total size: 913 GB across 64 shards (vs 806 GB native; +13% from NVFP4 sidecars on trunk)
- All trunk routed experts MXFP4→NVFP4 transcoded (1152 pairs per shard × 56 shards ≈ 64512 expert tensor pairs)
- Zero MTP transformation — 2343 mtp.* keys passed through byte-for-byte
- config.json: `quant_method=fp8 + moe_quant_algo=NVFP4`, no `ignored_layers` (per-layer routing handled by quant_config dispatch)

Per-layer MoE routing patch in `vllm/models/deepseek_v4/quant_config.py`: when prefix matches MTP layer index (≥ num_hidden_layers), force `Mxfp4MoEMethod` even though global `moe_quant_algo=NVFP4`. This handles the v12 mismatch where trunk experts are NVFP4 but mtp experts stay MXFP4.

## What's currently blocking deployment

After rebuilding mainline vLLM today (`pip install -e .` against `vllm/` at SHA `30f52a895`), serving any artifact (v12 OR previously-working v0.2) crashes silently:

- TP loads complete cleanly (~110 GB per GPU)
- `MTP draft model loaded: 39 params` confirms our `_mtp_block_is_quantized_on_disk` fix took effect
- ~12 min later (during warmup / cuda graph capture, even with `--enforce-eager`), ONE worker dies with no Python traceback
- All other workers cancel via `shm_broadcast`, engine init fails

The dying worker has no Python error in its log — pattern matches SIGKILL or SIGSEGV from CUDA-side.

Different workers die in different runs (TP0, TP1, TP2) — suggests timing/memory pressure rather than a deterministic code bug. Likely caused by a flashinfer/compressed-tensors version mismatch that the today's rebuild pulled in:

- This rebuild: `flashinfer-cubin-0.6.11.post2 flashinfer-python-0.6.11.post2 compressed-tensors-0.15.0.1`
- Previous (working) rebuild: `flashinfer-cubin-0.6.8.post1 flashinfer-python-0.6.8.post1`

The flashinfer minor-version bump (0.6.8 → 0.6.11) is the most likely culprit.

## Open questions / next-session pickup

1. **Pin flashinfer to 0.6.8.post1** and rebuild. If v0.2 then works (sanity), retest v12.
2. If v12 then loads AND runs trunk inference cleanly, run MTP probe at n=1. Expected from NVIDIA's recipe: **>= 33%** acceptance (matches recipes.vllm.ai's reported V4-Pro MTP figure); aspirational: **80%+** (close to fork's 91% on native).
3. If MTP is still ~3% on v12 + mainline, the bug is intrinsic to mainline's V1 spec_decode pipeline (consistent with everything else we've ruled out). At that point, our research-grade contribution is the proper v12 artifact + the `.scale`-detector fix; the MTP gap stays as a documented mainline-vs-fork limitation.

## What was committed this session

- `scripts/convert_v4_pro_mxfp4_to_nvfp4.py` — simplified `classify_tensor` to passthrough all `mtp.*` keys (NVIDIA recipe)
- New finding doc (this file)
- Conversion artifact at `/opt/dlami/nvme/weights/v4-pro-v12-strict` (913 GB, local only)
- Loader detector fix in `vllm/models/deepseek_v4/nvidia/mtp.py` (`.scale` added to quant suffix list)
- Per-layer MoE routing patch in `vllm/models/deepseek_v4/quant_config.py` (reverted at session end — kept as reference patch)

## Standing references

- vLLM issue [#43472](https://github.com/vllm-project/vllm/issues/43472) — our umbrella mainline-vs-fork DSV4 MTP gap report
- NVIDIA Model-Optimizer [#750](https://github.com/NVIDIA/Model-Optimizer/issues/750) — modelopt's own MTP-exclude bug on GLM-4.7
- `nvidia/DeepSeek-V3.2-NVFP4` — the canonical NVFP4 + DSV4-family recipe we're now aligned with

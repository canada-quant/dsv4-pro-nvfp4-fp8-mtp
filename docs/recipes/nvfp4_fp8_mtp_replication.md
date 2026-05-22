# NVFP4-FP8-MTP Replication Recipe — DeepSeek-V4-Pro

**Purpose**: end-to-end recipe to reproduce `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` from `deepseek-ai/DeepSeek-V4-Pro`. This is a **format conversion** (MXFP4 group=32 → NVFP4 group=16 on routed experts), not a calibration — V4-Pro shipped natively as FP4+FP8+BF16 mixed-precision with no public BF16 source, so no fresh activation calibration is needed.

Document conventions:
- "Source artifact" = `deepseek-ai/DeepSeek-V4-Pro` on HF (864 GB on disk, native FP4+FP8+BF16).
- "This artifact" = `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` (852 GB on disk, NVFP4 experts + FP8 attention + BF16 norms + BF16 `mtp.0.{e_proj,h_proj}`).
- "Predecessor" = `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` (V4-Flash version of the same pattern). Different model, different conversion (BF16 → NVFP4 calibration), but same MTP-retention idea. All dates ISO 8601.

---

## 1. What this recipe does, in one paragraph

Reads V4-Pro's native shard set, transcodes each routed FFN expert weight (MXFP4 group=32 + E8M0 block scales) into NVFP4 (group=16 + E4M3 block scales + FP32 per-tensor `weight_scale_2` + FP32 `input_scale=1.0`). Preserves attention (`wq_a / wq_b / wkv / wo_a / wo_b` and fused variants), shared experts, and the indexer/compressor at FP8 block 128×128 (passthrough). Preserves all norms / hc_* / embeddings / heads at BF16 / FP32 (passthrough). For the MTP block, transcodes experts the same as the trunk, but **dequantizes `mtp.0.{e_proj, h_proj}.weight` from FP8 to BF16** as a one-time workaround for a vLLM mainline `ReplicatedLinear + Fp8Config` loader gap (full reasoning in [`findings/mtp_eproj_hproj_workaround.md`](../findings/mtp_eproj_hproj_workaround.md)). Writes 64 new shards plus a vLLM-compatible `config.json` with NVFP4 routing in `quantization_config.moe_quant_algo`.

The math is deterministic and byte-level. No GPU calibration step is needed for the conversion. The conversion itself runs on a single B300 in ~17 minutes for the 64 shards.

---

## 2. Hardware target

Conversion (write side): **1× B300 SXM6 AC** (288 GB HBM3e). The expert-weight tensors load one at a time, dequantize via DeepGEMM kernels, regroup, requantize, and write — never holding more than one shard in GPU memory. Can also run on B200 (sm_100a) with `TORCH_CUDA_ARCH_LIST=10.0a` for the build — kernel paths exist.

Serving (read side): **8× B300 SXM6 AC** (288 GB HBM3e per GPU). The artifact's ~960 GB weight footprint (including safetensors headers + inflight buffers) needs ~120 GB per rank at TP=8 + EP plus the FlashInfer autotune cache. Single-node B300 is what we tested. The upstream-default `single_node_tep` strategy (TP + expert parallel) is what's blessed.

---

## 3. Software prerequisites

### 3.1 Base venv

Use the AWS DLAMI Ubuntu 24.04 `/opt/pytorch` venv (Python 3.13, torch 2.11.0+cu130). Confirm:

```bash
/opt/pytorch/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
# Expect: 2.11.0+cu130 (10, 3)
```

### 3.2 Conversion-side dependencies

```bash
/opt/pytorch/bin/pip install --upgrade safetensors compressed-tensors
# tested with safetensors >= 0.6.2, compressed-tensors >= 0.15.1
```

That's it for conversion. DeepGEMM kernels are pulled by vLLM and accessible at runtime — the conversion script imports the dequant kernel directly from `vllm._C` after the vLLM install in step 5.

### 3.3 Serve-side dependencies

Build vLLM with the 4 local patches (see `docs/QUICKSTART.md` and `scripts/install_vllm_with_patches.sh`).

---

## 4. Source artifact retrieval

```bash
hf auth login
hf download deepseek-ai/DeepSeek-V4-Pro --local-dir /scratch/weights/v4-pro-source
# 864 GB. ~10-15 min with HF Xet + token on 10 Gbps.
```

Confirm structure:

```bash
ls /scratch/weights/v4-pro-source/ | head
# Should include: config.json, generation_config.json, model.safetensors.index.json,
#                 model-00001-of-00064.safetensors ... model-00064-of-00064.safetensors,
#                 tokenizer.json, tokenizer_config.json, chat_template.jinja
```

Check the safetensors index for expected layer count and tensor dtypes:

```bash
python -c "
import json
idx = json.load(open('/scratch/weights/v4-pro-source/model.safetensors.index.json'))
keys = list(idx['weight_map'])
print('total tensors:', len(keys))
import re
layer_nums = {int(m.group(1)) for k in keys for m in [re.match(r'.*layers\\.(\\d+)\\.', k)] if m}
print('trunk layer range:', min(layer_nums), 'to', max(layer_nums))
print('mtp.* keys:', sum(1 for k in keys if k.startswith('mtp.')))
"
# Expect: total ~6000+ tensors, trunk layers 0-60 (61 hidden layers), mtp.* ~2300
```

---

## 5. vLLM build (needed for the dequant kernel)

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

~15 min on a fresh DLAMI. After the build, `from vllm._C import dsmoe_dequant_mxfp4` is importable from the `/opt/pytorch` venv — the conversion script uses this.

---

## 6. Run the conversion

```bash
/opt/pytorch/bin/python scripts/convert_v4_pro_mxfp4_to_nvfp4.py \
  --src /scratch/weights/v4-pro-source \
  --dst /scratch/weights/v4-pro-nvfp4 \
  --num-shards 64 \
  --device cuda:0
# ~17 min on 1× B300
```

The script:
- Iterates each of the 64 source shards
- For each tensor in the shard:
  - If the key matches the routed-expert pattern (`layers.*.ffn.experts.*.w{1,2,3}.weight`): dequant MXFP4 group=32 → BF16, regroup to NVFP4 group=16, compute per-tensor `S_g = max_amax / (FP4_max × E4M3_max)` (shared across `w1`/`w3` per ModelOpt invariant; independent for `w2`), quantize to per-block E4M3 + FP4 grid, emit + companion `weight_scale_2` (FP32 [1]) and `input_scale` (FP32 [1], value 1.0) sidecars.
  - If the key matches `mtp.0.ffn.experts.*.w{1,2,3}.weight`: same MXFP4→NVFP4 conversion.
  - If the key matches `mtp.0.{e_proj, h_proj}.weight`: dequant FP8 block 128×128 → BF16, write BF16. The companion `.scale` key is dropped.
  - Otherwise: passthrough — read raw bytes, write raw bytes, no GPU touch.
- Writes shard-by-shard to `dst`, with a fresh `model.safetensors.index.json` built from accumulated key-to-shard mapping.
- Writes a fresh `config.json` with `quantization_config.moe_quant_algo = "NVFP4"` so vLLM routes routed experts through `ModelOptNvFp4FusedMoE`.

### Byte-level validation

Spot-check the math against the source on 192 sampled tensors:

```bash
/opt/pytorch/bin/python scripts/sample_conversion_errors.py \
  --src /scratch/weights/v4-pro-source \
  --dst /scratch/weights/v4-pro-nvfp4 \
  --n 192
# Expect: correlation 0.997-1.0 vs source on all sampled tensors
```

Output is captured in `docs/findings/conversion_v3_validation.md`.

---

## 7. MTP key verification

Confirm the MTP block is present and structured correctly:

```bash
/opt/pytorch/bin/python scripts/verify_mtp_keys.py \
  --artifact /scratch/weights/v4-pro-nvfp4
# Expect: ~2343 mtp.* keys, e_proj/h_proj at BF16, experts at NVFP4
```

Confirm MTP experts are NVFP4 (not BF16, not FP8) — this is the load-bearing invariant:

```bash
/opt/pytorch/bin/python scripts/verify_mtp_quantized.py \
  --artifact /scratch/weights/v4-pro-nvfp4
# Expect: routed experts under mtp.0.ffn.experts.* — all NVFP4 packed dtype
#         e_proj and h_proj — BF16 (the workaround), no .scale sidecar
```

---

## 8. Serve smoke

```bash
export PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda

/opt/pytorch/bin/vllm serve /scratch/weights/v4-pro-nvfp4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8089
```

Cold-start including FlashInfer autotune: ~5-6 min. Then:

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/scratch/weights/v4-pro-nvfp4","messages":[{"role":"user","content":"What is 17*19?"}],"max_tokens":40,"temperature":0}' \
| jq -r '.choices[0].message.content'
# Expect: "323"
```

For MTP, add `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`. Acceptance is ~1.8% per-token consistent with upstream V4-Pro MTP weakness (see `findings/upstream_mtp_classification.md`).

---

## 9. Reproducing the measurements

Same bench harness as `MODEL_CARD.md`:

```bash
# Latency probes (matches MODEL_CARD numbers)
python scripts/bench_v4_pro.py latency --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4 --n 20 --max-tokens 128 --concurrency 1

python scripts/bench_v4_pro.py latency --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4 --n 64 --max-tokens 128 --concurrency 16

# MTP acceptance probe
python scripts/bench_v4_pro.py mtp --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4

# GSM8K-300 matched-config quality probe (used for the matched comparison)
python scripts/bench_v4_pro.py gsm8k --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4 \
  --limit 300 --concurrency 16 --max-tokens 2048
```

For the full backend × format matrix (4 cells), use `scripts/matrix_runner.sh`. For the c=16 batched + matched-GSM8K extension (4 more cells), use `scripts/extension_runner.sh`.

---

## 10. Upload to HF

The artifact is private during validation. After internal sign-off:

```bash
hf upload canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  /scratch/weights/v4-pro-nvfp4 \
  --commit-message "v0.1: NVFP4 conversion of DeepSeek-V4-Pro with MTP retained"
# 852 GiB. ~15-30 min with HF Xet on a 10 Gbps NIC.
```

Then upload the model card:

```bash
hf upload canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  MODEL_CARD.md --path-in-repo README.md \
  --commit-message "v0.1: model card"
```

---

## 11. What's different from the V4-Flash predecessor recipe

| Step | V4-Flash recipe | V4-Pro recipe |
|---|---|---|
| Quantization method | Calibration: load BF16, run llm-compressor `QuantizationModifier` with 64 samples from ultrachat_200k | **Conversion**: load native MXFP4, transcode to NVFP4 — no calibration, no GPU forward pass on calibration data |
| Source format on HF | BF16 ~600 GB | Native FP4+FP8+BF16 ~864 GB (no public BF16 exists) |
| Output format on routed experts | NVFP4 group=16 + FP8 E4M3 block scales | Same |
| Output format on attention | FP8 block 128×128 | Same |
| MTP block in output | BF16 (verbatim, never quantized — calibration recipe used `ignore=[r"re:.*mtp\..*"]`) | NVFP4 experts (same as trunk) + BF16 `e_proj`/`h_proj` (dequant workaround) |
| Patches required at serve | 5 (4 in vllm, 1 in transformers) | 4 (all in vllm; transformers patch obsolete since V4-Pro modeling uses different `_keys_to_ignore_on_load_unexpected`) |
| Hardware for conversion | 4× B300 (calibration forward needs MoE saturation) | 1× B300 (per-shard transcoding) |
| Time to artifact | ~6-8 hours (calibration loop + checkpointing + postprocess) | ~17 min (just the transcoding) |

The reason V4-Pro is simpler: DeepSeek already shipped it quantized, so we just convert formats. V4-Flash shipped BF16, so we had to calibrate from scratch.

---

## 12. Open questions for v0.2

- Replace BF16-dequant of `mtp.0.{e_proj, h_proj}` with proper FP8 once `ReplicatedLinear + Fp8Config` is fixed upstream (saves ~200 MB disk; doesn't affect quality).
- File the vLLM issue documenting `deep_gemm_mega_moe`'s NVFP4 incompatibility so future builds can dispatch NVFP4 through the mega-kernel path.
- Investigate whether the 7 NVFP4-strict-loss problems on GSM8K-300 (see `findings/backend_format_matrix.md`) are concentrated on certain operator categories — could inform a v0.2 conversion with per-tensor S_g per-w1/w2/w3 (we currently share S_g between w1 and w3 per ModelOpt invariant; independent w2). The strict-loss is small (-2.33pt, within CI overlap) so v0.1 ships as-is.
- AIME 2024 full 30-problem run with the patched bench (the 25/30 partial was captured before the network-resilience fix).

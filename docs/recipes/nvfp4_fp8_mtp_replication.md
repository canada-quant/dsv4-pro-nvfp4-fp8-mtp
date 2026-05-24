# NVFP4-FP8-MTP Replication Recipe — DeepSeek-V4-Pro (v12)

**Purpose**: end-to-end recipe to reproduce `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` from `deepseek-ai/DeepSeek-V4-Pro`. This is a **format conversion** (MXFP4 group=32 → NVFP4 group=16 on trunk routed experts only), not a calibration — V4-Pro shipped natively as FP4+FP8+BF16 mixed-precision with no public BF16 source.

Document conventions:
- "Source artifact" = `deepseek-ai/DeepSeek-V4-Pro` on HF (864 GB on disk, native FP4+FP8+BF16).
- "This artifact" = `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` v12 (913 GiB on disk; NVFP4 trunk routed experts + native FP8 attention + native MXFP4-FP8-BF16 MTP block, full passthrough).
- "Predecessor" = `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` (V4-Flash version of the same pattern).

---

## 1. What this recipe does

The conversion script reads V4-Pro's native shard set and transcodes **only the trunk routed FFN experts** (`layers.X.ffn.experts.Y.w{1,2,3}` for X in 0..60) from MXFP4 group=32 + E8M0 block scales into NVFP4 group=16 + E4M3 block scales + FP32 per-tensor `weight_scale_2` + FP32 `input_scale=1.0`. Every other tensor — attention (FP8 block 128×128), shared experts (FP8 block 128×128), norms (BF16), hc_/indexer/compressor sub-modules, embeddings, head, and **the entire MTP layer `mtp.0.*` (mixed MXFP4 + FP8 + BF16, native)** — is byte-passthrough.

This follows NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe: excluding the entire MTP layer from quantization is what preserves MTP draft-head accuracy. See [`findings/v12_nvidia_recipe_2026_05_24.md`](../findings/v12_nvidia_recipe_2026_05_24.md).

The math is deterministic and byte-level. No GPU calibration step is needed. The conversion runs on a single B300 in **~25 minutes**.

---

## 2. Hardware target

**Conversion (write side)**: 1× B300 SXM6 AC (288 GB HBM3e). The trunk expert weights load one at a time, dequantize via DeepGEMM kernels, regroup, requantize, and write — never holding more than one shard in GPU memory.

**Serving (read side)**: 8× B300 SXM6 AC (288 GB HBM3e per GPU). The artifact's ~1 TB weight footprint (913 GiB on disk + safetensors headers + inflight buffers) needs ~120 GB per rank at TP=8 + EP plus FlashInfer autotune cache.

---

## 3. Software prerequisites

### 3.1 Base venv

AWS DLAMI Ubuntu 24.04 `/opt/pytorch` venv (Python 3.13, torch 2.11.0+cu130):

```bash
/opt/pytorch/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
# Expect: 2.11.0+cu130 (10, 3)
```

### 3.2 Conversion-side dependencies

```bash
/opt/pytorch/bin/pip install --upgrade safetensors compressed-tensors
```

### 3.3 Serve-side dependencies

Build vLLM mainline `@30f52a895` + 5 PR patches + 1 local patch (see `docs/QUICKSTART.md` and `scripts/install_vllm_with_patches.sh`).

Pin flashinfer:

```bash
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'
```

---

## 4. Source artifact retrieval

```bash
hf auth login
hf download deepseek-ai/DeepSeek-V4-Pro --local-dir /scratch/weights/v4-pro-source
# 864 GB.
```

Sanity check:

```bash
python -c "
import json, re
idx = json.load(open('/scratch/weights/v4-pro-source/model.safetensors.index.json'))
keys = list(idx['weight_map'])
print('total tensors:', len(keys))
layer_nums = {int(m.group(1)) for k in keys for m in [re.match(r'.*layers\\.(\\d+)\\.', k)] if m}
print('trunk layer range:', min(layer_nums), 'to', max(layer_nums))
print('mtp.* keys:', sum(1 for k in keys if k.startswith('mtp.')))
"
# Expect: ~6000+ tensors, trunk layers 0-60 (61 hidden layers), mtp.* ~2300
```

---

## 5. Run the conversion

```bash
/opt/pytorch/bin/python scripts/convert_v4_pro_mxfp4_to_nvfp4.py \
  --src /scratch/weights/v4-pro-source \
  --dst /scratch/weights/v4-pro-nvfp4 \
  --num-shards 64 \
  --device cuda:0
# ~25 min on 1× B300
```

What `classify_tensor()` does in v12:

```python
def classify_tensor(key: str) -> str:
    # v12 recipe: ZERO mtp transformation. Per NVIDIA/DeepSeek-V3.2-NVFP4,
    # exclude entire MTP layer from quantization.
    if key.startswith("mtp."):
        return "passthrough"
    if EXPERT_WEIGHT_RE.match(key):
        return "expert_weight"
    if EXPERT_SCALE_RE.match(key):
        return "expert_scale"
    return "passthrough"
```

The script then:
- For each key classified `expert_weight`: dequant MXFP4 → BF16 → regroup to group=16 → compute per-tensor `S_g = max_amax / (FP4_max × E4M3_max)` (shared between `w1` and `w3` per ModelOpt invariant; independent for `w2`) → quantize to per-block E4M3 + FP4 grid → emit `weight_packed` + `weight_scale` + companion `weight_scale_2` (FP32 [1]) + `input_scale` (FP32 [1], value 1.0).
- For each `expert_scale`: regenerated above; drop the source `.scale` key.
- For each `passthrough` key: read raw bytes, write raw bytes. **All `mtp.*` keys take this path.**

Writes a fresh `config.json` with `quantization_config.moe_quant_algo = "NVFP4"` so vLLM routes trunk routed experts through `ModelOptNvFp4FusedMoE`. **Do not set `ignored_layers`** — the local DSV4FP8Config per-layer routing patch (`patch_v12b_per_layer_moe_routing.diff`) handles the MTP layer routing at construction time.

### Byte-level validation

```bash
/opt/pytorch/bin/python scripts/sample_conversion_errors.py \
  --src /scratch/weights/v4-pro-source \
  --dst /scratch/weights/v4-pro-nvfp4 \
  --n 192
# Expect: correlation 0.997-1.0 vs source
```

See [`findings/conversion_v3_validation.md`](../findings/conversion_v3_validation.md).

---

## 6. MTP block verification

```bash
/opt/pytorch/bin/python scripts/verify_mtp_keys.py \
  --artifact /scratch/weights/v4-pro-nvfp4
# Expect: ~2343 mtp.* keys, all byte-identical to source
```

The MTP block in v12 is **fully native** — same MXFP4 routed experts as the source, same FP8 attention sub-modules (compressor, indexer, wq_a, wq_b, wkv, wo_a, wo_b), same FP8 block 128×128 `e_proj` / `h_proj` / shared experts, same BF16 norms. The conversion script does not touch any `mtp.*` key.

---

## 7. Serve smoke test

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
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --max-model-len 65536 \
  --port 8089
```

Cold start ~12-15 min (FlashInfer FP4 MoE JIT + torch.compile + cudagraph capture). Then:

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/scratch/weights/v4-pro-nvfp4","messages":[{"role":"user","content":"What is 17*19?"}],"max_tokens":40,"temperature":0}' \
| jq -r '.choices[0].message.content'
# Expect: "323"
```

MTP probe:

```bash
python scripts/bench_v4_pro.py mtp --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4
# Expect: ~91% acceptance on the 20-prompt probe
```

If acceptance is <10%, one of the five fix-stack pieces is missing. See [`findings/v12_nvfp4_mtp_working_2026_05_24.md`](../findings/v12_nvfp4_mtp_working_2026_05_24.md).

---

## 8. Reproducing the headline measurements

```bash
# MTP probe (~30 sec)
python scripts/bench_v4_pro.py mtp --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4

# AIME 2024 thinking=high full (~25 min, max_tokens=60000, 0 truncations)
python scripts/bench_v4_pro.py aime --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4 --max-tokens 60000 --concurrency 8

# GSM8K full n=1319 (~5 min with MTP + c=32)
python scripts/bench_v4_pro.py gsm8k --base-url http://localhost:8089 \
  --model /scratch/weights/v4-pro-nvfp4 --concurrency 32 --max-tokens 2048

# Latency / throughput sweep
for c in 1 16 64 128; do
  n=$((c*4))
  python scripts/bench_v4_pro.py latency --base-url http://localhost:8089 \
    --model /scratch/weights/v4-pro-nvfp4 --n $n --max-tokens 128 --concurrency $c
done

# HumanEval / HumanEval+ via EvalPlus (~5 min sequential codegen + ~10 sec eval)
evalplus.codegen /scratch/weights/v4-pro-nvfp4 humaneval \
  --backend openai --base_url http://localhost:8089/v1 --greedy \
  --root /tmp/evalplus_humaneval
evalplus.evaluate --dataset humaneval \
  --samples /tmp/evalplus_humaneval/humaneval/*.jsonl
```

---

## 9. Upload to HF

```bash
hf upload canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  /scratch/weights/v4-pro-nvfp4 \
  --commit-message "v12: NVFP4 trunk + native MTP passthrough per NVIDIA V3.2 recipe"
# 913 GiB; ~15-30 min with HF Xet on a 10 Gbps NIC.

hf upload canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  MODEL_CARD.md --path-in-repo README.md
```

---

## 10. What's different from the V4-Flash predecessor recipe

| Step | V4-Flash recipe | V4-Pro v12 recipe |
|---|---|---|
| Quantization method | Calibration: BF16 source → llm-compressor `QuantizationModifier` with 64 ultrachat_200k samples | **Conversion**: load native MXFP4, transcode to NVFP4 — no calibration |
| Source format on HF | BF16 ~600 GB | Native FP4+FP8+BF16 ~864 GB (no public BF16) |
| Output trunk routed experts | NVFP4 group=16 + E4M3 block scales | Same |
| Output attention | FP8 block 128×128 | Same |
| **MTP block** | BF16 (`ignore=[r"re:.*mtp\..*"]` in calibration recipe) | **Native passthrough** (MXFP4 + FP8 + BF16 mixed, NVIDIA-recipe aligned) |
| vLLM patches | 5 (4 vllm + 1 transformers) | 5 + 1 local (all vllm; transformers patch obsolete) |
| Hardware for conversion | 4× B300 (calibration forward) | 1× B300 (per-shard transcoding) |
| Time to artifact | ~6-8 hours | ~25 min |
| MTP draft acceptance | 81.60% (AIME) / 87.92% (chat) | **91.21%** focused / **92.83%** cumulative |

---

## 11. The five vLLM patches + one local diff (v12)

| Patch file | Upstream PR | Where it fixes things |
|---|---|---|
| `patches/patch_43248_ct_bool_wrap.diff` | [#43248](https://github.com/vllm-project/vllm/pull/43248) | compressed-tensors `is_static_input_scheme` `bool()` wrap |
| `patches/patch_43288_scale_fmt_get.diff` | [#43288](https://github.com/vllm-project/vllm/pull/43288) | DSV4 `scale_fmt` defensive `.get()` + BF16 `getattr` |
| `patches/patch_43290_weight_scale_fallback.diff` | [#43290](https://github.com/vllm-project/vllm/pull/43290) | DSV4 attention `weight_scale_inv`-or-`weight_scale` fallback |
| `patches/patch_43319_mtp_quant_detect.diff` | [#43319](https://github.com/vllm-project/vllm/pull/43319) | **`.scale` suffix in MTP detector** (load-bearing) + candidate-list scale resolution |
| `patches/patch_43467_megamoe_nvfp4_guard.diff` | [#43467](https://github.com/vllm-project/vllm/pull/43467) | DSV4 MegaMoE early-fail for NVFP4 |
| `patches/patch_v12b_per_layer_moe_routing.diff` | (local) | `DSV4FP8Config.get_quant_method` per-layer routing for hybrid NVFP4-trunk + MXFP4-MTP |

PR #42209 (NVFP4 MoE support for DSV4, sychen52 / NVIDIA) is **merged in mainline** (2026-05-22) and does not need to be applied.

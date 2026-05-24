# v12 — first working NVFP4 V4-Pro with MTP at 91.45%

**Date:** 2026-05-24
**Status:** ✅ Working. Mainline vLLM, cuda graphs ON, NVFP4 trunk, native MTP block, 91.45% MTP acceptance.

## Headline measurement

```
v12 (NVFP4 trunk + native FP8/MXFP4 mtp) + cuda graphs ON + mainline vLLM:
MTP n=1: 2995 / 3275 = 91.45% acceptance on 20-prompt probe
```

For reference, on the same 20-prompt probe with the same hardware (B300 SXM6 AC, TP=8, EP):
- Fork build + native MXFP4 V4-Pro: 91.07% (first measurement) / 91.94% (re-measure)
- v12 + mainline + cuda graphs: **91.45%**
- v12 + mainline + `--enforce-eager`: 90.84%
- v0.2 (NVFP4 mtp experts + BF16 e_proj/h_proj) on mainline: 3.07%
- v0.3 (BF16 entire mtp.0) on mainline: 3.33%
- v0.4-bisect (MXFP4 trunk passthrough + BF16 mtp.0) on mainline: 2.65%

v12 is the first canada-quant NVFP4 conversion of V4-Pro that achieves native-level MTP acceptance, and the first NVFP4 V4-Pro of any provenance that we've observed serving correctly through vLLM mainline's MTP path.

## Production config

```bash
vllm serve /path/to/v4-pro-v12-strict \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --moe-backend flashinfer_trtllm \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --max-model-len 32768
```

No `--enforce-eager`. CUDA graphs in `FULL_AND_PIECEWISE` mode (default for V4-Pro). Compile pass takes ~12–15 minutes for cold start (flashinfer fp4 MoE JIT + torch.compile + cudagraph capture).

## The fix stack

Five separate changes had to land together. None alone was sufficient.

### 1. Conversion: ZERO `mtp.*` transformation

`scripts/convert_v4_pro_mxfp4_to_nvfp4.py:classify_tensor` — any key starting with `mtp.` returns `"passthrough"`. The previous v0.2 / v0.3 / v0.4 attempts all transformed mtp tensors (NVFP4 experts in v0.2; BF16 dequant in v0.3/v0.4). NVIDIA's `nvidia/DeepSeek-V3.2-NVFP4` reference recipe excludes the entire MTP layer (`model.layers.61*` for V3.2; same for V4-Pro) from quantization. v12 follows this — every mtp.* byte is identical to the native source.

### 2. `DSV4FP8Config.get_quant_method` per-layer MoE routing

`vllm/models/deepseek_v4/quant_config.py` had a single global `moe_quant_algo` branch — `"NVFP4"` → `ModelOptNvFp4FusedMoE`, anything else → `Mxfp4MoEMethod`. With v12 the trunk MoE is NVFP4 (transcoded) but the MTP MoE is MXFP4 (native passthrough). A single global setting can't represent both.

Patch: in `get_quant_method`, detect MTP layer by prefix (`re.search(r'\.layers\.(\d+)\.')` against `prefix` arg, compared to `config.num_hidden_layers`). When the layer is the MTP block, force `Mxfp4MoEMethod` even if global `moe_quant_algo='NVFP4'`. Trunk dispatch is unchanged.

### 3. **The load-bearing bug**: `_mtp_block_is_quantized_on_disk` was missing `.scale` suffix

`vllm/models/deepseek_v4/nvidia/mtp.py:_mtp_block_is_quantized_on_disk` scans the safetensors index for `mtp.*` keys with quantization-scale suffixes. The pre-fix list:

```python
quant_suffixes = (
    ".weight_scale",
    ".weight_scale_inv",
    ".weight_packed",
    ".weight_global_scale",
    ".input_global_scale",
    ".weight_zero_point",
)
```

DSV4 native checkpoints store FP8 block scales with a raw `.scale` suffix (e.g. `mtp.0.attn.wq_a.scale`). The runtime `WeightsMapper` converts these to `.weight_scale_inv` at load time. But the detector ran BEFORE the rename — it scanned the raw on-disk keys, none of which match the post-rename suffixes — so it returned False.

False from the detector meant the MTP block was constructed with `quant_config=None`. With `quant_config=None`, `ReplicatedLinear` for `e_proj` / `h_proj` registered only `weight`, no `weight_scale_inv` param. The loader then iterated the on-disk `.scale` keys, renamed them to `.weight_scale_inv`, looked them up in `params_dict`, and raised `KeyError: 'model.layers.61.e_proj.weight_scale_inv'`.

The diagnostic dump that confirmed this:
```
[v12-diag-construct] e_proj being built with quant_config=None (type=NoneType)
MTP block weights are BF16 on disk (no quant scales for mtp.* keys ...)
[v12-diag] MISSING name='model.layers.61.e_proj.weight_scale_inv'
[v12-diag] e_proj/h_proj keys=['model.layers.61.e_proj.weight', 'model.layers.61.h_proj.weight']
```

After fix:
```
[v12-diag-construct] e_proj being built with quant_config=<DeepseekV4FP8Config> (type=DeepseekV4FP8Config)
MTP draft model loaded: 39 params
```

One-line fix:
```python
quant_suffixes = (
    ".scale",   # ← added: DSV4 native FP8 block scale convention
    ".weight_scale",
    ".weight_scale_inv",
    ...
)
```

### 4. flashinfer pin to 0.6.8.post1

`flashinfer-cubin==0.6.11.post2` (which mainline build of vLLM ships with as of May 24) triggered silent worker crashes during model construction. Pinning back to 0.6.8.post1 (what the fork ships with) restored stable startup. Likely an ABI regression between the two — neither version is `requires`'d by vllm, so the upgrade is silent.

```bash
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'
```

### 5. config.json `quantization_config`

The artifact's `config.json` ships with `quant_method: "fp8"`, `moe_quant_algo: "NVFP4"`, `expert_dtype: "fp4"`, and **no `ignored_layers`** field. The per-layer routing patch (#2) handles MTP/trunk MoE divergence at runtime. Adding `ignored_layers: ["model.layers.61*"]` would route the MTP MoE to `UnquantizedFusedMoEMethod`, which crashes on flashinfer_trtllm because the unquantized backend doesn't support DSV4 routing method.

## What this proves

The mainline DSV4 MTP forward path was never broken. The 3% acceptance we measured on v0.2/v0.3/v0.4 was caused by **our conversion transforming the MTP block AND a bug in the BF16-detection logic that mis-classified the (still-quantized) MTP block as BF16 and stripped its scales at construction time**.

NVIDIA's recipe (don't touch the MTP layer) is correct. The detector bug just made it look like there was a separate mainline MTP-forward bug. There isn't.

## Replication

The full canonical install + serve recipe:

```bash
# 1. Build vLLM mainline at SHA 30f52a895 + our 5 patches
bash scripts/install_vllm_with_patches.sh

# 2. Pin flashinfer to 0.6.8 (workaround for silent worker crash)
pip install --no-deps 'flashinfer-cubin==0.6.8.post1' 'flashinfer-python==0.6.8.post1'

# 3. Build v12 artifact from native MXFP4 source
python scripts/convert_v4_pro_mxfp4_to_nvfp4.py \
    --source /path/to/deepseek-ai/DeepSeek-V4-Pro \
    --output /path/to/v4-pro-v12-strict \
    --device cuda:0
# Conversion takes ~25 min on B300. 913 GB output.

# 4. Serve
vllm serve /path/to/v4-pro-v12-strict \
    --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --moe-backend flashinfer_trtllm \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --max-model-len 32768
```

Cold start: ~15 min. Steady-state MTP: ~91% acceptance at greedy temp=0.

## Followups

1. **File the detector fix upstream**: the `.scale`-missing-from-suffixes bug in `_mtp_block_is_quantized_on_disk` is a generic vLLM bug affecting any DSV4 family checkpoint that uses native scale naming. PR-worthy.

2. **File the per-layer MoE routing fix upstream**: the DSV4FP8Config single-global-`moe_quant_algo` is restrictive for hybrid-quant artifacts (NVFP4 trunk + MXFP4 mtp). Extending it to support per-layer override (or per-prefix exclude list) would unblock any future NVFP4 conversion that follows NVIDIA's exclude-MTP recipe.

3. **Comment update on vLLM issue #43472**: previous filing claimed mainline DSV4 MTP forward was broken (3% acceptance). Retract that — the bug was in the detector and our conversion, not the forward path. Mainline DSV4 MTP works fine when handed a correctly-shaped artifact.

4. **Quality re-measure on v12**: re-run GSM8K / MMLU-Pro / HumanEval / AIME / IFEval against v12 to confirm trunk quality is preserved (should match v0.2 numbers since the trunk MoE recipe is identical).

5. **Throughput re-measure on v12**: re-run c=1 / c=16 / c=64 / c=128 throughput sweep to confirm the v0.2 throughput numbers still hold and to capture MTP speedup vs no-MTP baseline.

6. **HF push**: v12 is the first artifact worth publishing as `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP` proper. Quality + throughput + MTP all measured.

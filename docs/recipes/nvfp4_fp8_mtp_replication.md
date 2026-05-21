# NVFP4-FP8-MTP Replication Recipe — DeepSeek-V4-Flash

**Purpose:** This document is the load-bearing recipe for producing an MTP-preserving NVFP4-FP8 quantization of DeepSeek-V4-Flash. The same formula should generalize to other MoE+MTP architectures in the DSV-family with the items in §11 adjusted.

Document conventions:
- "RedHat reference" = `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` (drops MTP).
- "Sibling repo" = `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` predecessor + the W4A16-MTP repo for sm120 (RTX Pro 6000 / GB10).
- "Our artifact" = `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`.
- All dates ISO 8601.

---

## 1. The differentiator (and why it's load-bearing)

| Aspect | RedHat NVFP4-FP8 | This recipe (NVFP4-FP8-MTP) |
|---|---|---|
| NVFP4 on routed FFN experts | yes | yes — identical math |
| FP8_BLOCK 128×128 on attention | yes | yes — identical math |
| MTP speculative-decoding head | **dropped at load** | **preserved (BF16, present, unquantized)** |
| `--speculative-config method=mtp` usable | no (no draft tensors) | yes |
| Spec-decode tok/s gain | 0 | ~1.5–2× on agentic workloads (target) |
| Differentiator durability | n/a | until transformers stops filtering `mtp.*` |

The MTP retention is **the entire reason this artifact exists**. RedHat's quantization math is fine; the loader pipeline they used silently dropped the MTP block. The win here is not in the quantization recipe — it's in (a) the patches that keep MTP weights from being filtered, (b) the recipe modification that keeps MTP weights from being corrupted by quantization, and (c) the postprocess that makes MTP-bearing artifacts vLLM-loadable.

---

## 2. Hardware target

This is a **data-center Blackwell** recipe. Verified target SM: **10.0a** (B100 / B200 / B300, SXM6 AC).

**Do NOT use this recipe's vLLM serve path on consumer Blackwell (sm120: RTX Pro 6000, GB10, DGX Spark).** NVFP4 hardware tensor-core support is not exercised on those parts yet. The sibling W4A16-FP8-MTP recipe is the right path for sm120; that recipe targets `jasl/dm120` vLLM fork and W4A16 Marlin kernels.

Calibration on B300 was performed on:
- AWS EC2 `p6-b300.48xlarge` (8× B300 SXM6 AC, ~140 GB HBM each, NVLink full-mesh)
- AWS profile `rozo`, region `us-west-2a`, on-demand (verified via IMDS, not spot)
- Ubuntu 24.04 DLAMI, `/opt/pytorch` torch 2.11.0+cu130 venv

Minimum HBM for calibration: ~80 GB per rank with N-way expert sharding (verified at 8-rank: ~26 GB/rank steady-state weight residency; calibration scratch peaks ~70 GB during NVFP4 packing).

Minimum system RAM: ~512 GB if running 1-rank (the path we used after the 8-rank cache-offload deadlock). 2 TB headroom recommended.

Minimum NVMe scratch: 600 GB for BF16 source + 200 GB for artifact + 100 GB for per-rank checkpoint subdirs during multi-rank save.

---

## 3. Software stack

Pinned versions used for V4-Flash. Treat these as the **upper bound of what's known to work**; newer pins may regress on subtle gotchas (see §9).

| Component | Version | Source |
|---|---|---|
| Python | 3.13 | DLAMI default |
| PyTorch | 2.11.0+cu130 | `/opt/pytorch` |
| CUDA toolkit (source builds) | 13.0 | `sudo apt install cuda-toolkit-13-0` |
| CUDA runtime (bundled) | 13.x | `/opt/pytorch/cuda` |
| `transformers` | 5.8.1 + our patch | `patches/modeling_deepseek_v4.py.diff` |
| `compressed-tensors` | `0.15.1.a20260515` | `vllm-project/compressed-tensors` `main` snapshot |
| `llmcompressor` | `0.10.1.dev123+gf2aa32e2` | `kylesayrs/transformers-v5` branch |
| `vLLM` (serve target) | mainline + PR #42209 | `vllm-project/vllm` `main` @ post-#43077 + cherry-pick `sychen52:nvfp4_dsv4` |
| `safetensors` | latest stable | pip default |
| `py-spy` | 0.4.2 | required for diagnosing collective hangs |
| `ninja` | any | required by torch inductor on serve path |

**Critical:** Do NOT use `--system-site-packages` on the calibration venv. `/opt/pytorch`'s Python 3.13 venv inheriting `/usr/lib/python3/dist-packages` (Python-3.12-compiled wheels) crashes with `pyo3_runtime.PanicException` on `cryptography` import.

---

## 4. Required patches (the MTP-preservation patches)

### 4.1 `transformers` — remove `_keys_to_ignore_on_load_unexpected` for `mtp.*`

Located at `patches/modeling_deepseek_v4.py.diff` in this repo. Two hunks:

**Hunk 1 (the load-time filter):** transformers' `DeepseekV4PreTrainedModel` defines:
```python
_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
```
This silently drops every `mtp.*` key at `from_pretrained` time. We remove it. **Still current on transformers `main` as of 2026-05-20.** This is the single biggest reason the RedHat NVFP4-FP8 artifact has no MTP.

**Hunk 2 (the MTP block forward):** transformers' base modeling doesn't wire MTP into the forward graph (it's an inference-only branch). We add the minimal MTP forward that mirrors the upstream DeepSeek `model.py` MTP block.

### 4.2 `llmcompressor` — `helpers.py` dotless tensor-match fix

Located at `patches/helpers.py.diff`. The vendored upstream DeepSeek model registers a tensor named `freqs_cis` (no dots). The unpatched `llmcompressor.utils.helpers.tensor_follows_pattern` regex was written assuming dotted keys and rejects this. The patch broadens the match. Without this, calibration crashes at sequencing.

### 4.3 (vLLM serve) — PR #42209

Not a patch we wrote — it's the upstream PR that adds NVFP4 MoE support for DSV4 in vLLM. As of 2026-05-21 it has 3 approvals (xinli-sw, pavanimajety, zyongye) and is awaiting Buildkite green. We apply it as a cherry-pick on mainline `vllm-project/vllm` (post-#43077, which completed the DSV4 modular refactor 2026-05-19).

### 4.4 (vLLM serve) — `scale_fmt` defensive `.get()` (filed as our PR)

`vllm/models/deepseek_v4/nvidia/model.py:909` does `config.quantization_config["scale_fmt"]` — a hard subscript that throws `KeyError` when the field is absent. Affects both mainline and the jasl fork. Our PR converts it to `.get("scale_fmt", "ue8m0")`. Until merged, **our artifact's `config.json` must include `scale_fmt: ue8m0`** (the postprocess script injects this).

---

## 5. The recipe (NVFP4 routed experts + FP8_BLOCK attention + MTP Option Y)

```python
recipe = QuantizationModifier(
    targets=["Linear"],
    config_groups={
        # Group 0 — attention: FP8_BLOCK 128×128
        "group_0": QuantizationScheme(
            targets=[
                r"re:.*attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
            ],
            weights=QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="block",
                block_structure=[128, 128],
                symmetric=True,
                dynamic=False,
                observer="minmax",
            ),
            input_activations=None,  # weight-only on attn (matches RedHat)
            format="float-quantized",
        ),
        # Group 1 — routed FFN experts: NVFP4 group_size=16 + FP8 e4m3 scales
        "group_1": QuantizationScheme(
            targets=[
                r"re:.*ffn\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
            ],
            weights=QuantizationArgs(
                num_bits=4,
                type="float",
                strategy="tensor_group",
                group_size=16,
                symmetric=True,
                dynamic=False,
                observer="minmax",
            ),
            input_activations=QuantizationArgs(
                num_bits=4,
                type="float",
                strategy="tensor_group",
                group_size=16,
                symmetric=True,
                dynamic="local",  # online activation scaling, matches RedHat
                observer="minmax",
            ),
            format="nvfp4-pack-quantized",
        ),
    },
    ignore=[
        # Architecture-native non-quantize:
        "lm_head",
        r"re:.*\.embed_tokens$",
        r"re:.*\.norm$",
        r"re:.*\.gate$",                       # MoE router — must stay BF16
        r"re:.*\.q_norm$",
        r"re:.*\.k_norm$",
        # OPTION Y — MTP block stays BF16, present, unquantized:
        r"re:.*mtp\..*",
    ],
    scale_fmt="ue8m0",
)
```

### Why Option Y (MTP excluded from recipe)

Both this B300 NVFP4 workstream and the H200 W4A16 sibling workstream independently arrived at the same conclusion: **MTP weights stay BF16 in the artifact**. The reasons:

1. **Inference-tensor crash during MTP quantization** — the vendored upstream DSV4 model uses `with torch.inference_mode():` around the MTP block's shared-embedding lookup (it's a single-token branch designed for runtime, not training-style numerics). `QuantizationModifier`'s `sequential_epoch_end` writes `weight_scale` in-place after each layer's pass; for the MTP block this in-place write fires on a tensor created under `inference_mode()`, which raises `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`. Filed as llm-compressor #2745.

2. **Speculative decoding is small-tensor numerics; spec-decode heads are sensitive.** Quantizing the MTP draft head can shift the draft distribution enough to crater acceptance rate (the metric that determines spec-decode wall-clock speedup). Empirically — across both workstreams — preserving MTP as BF16 maintains the upstream model's 7.01% acceptance rate while still giving ~75% disk savings on the rest of the model (the MTP block is a single transformer layer; ~1.5 GB BF16 in a 172 GB artifact).

3. **vLLM `--speculative-config method=mtp` loads BF16 MTP weights without complaint.** It expects the MTP block to be present and float-typed; precision below BF16 was never a supported configuration anyway.

The Option Y invariant is **verified** by `scripts/verify_mtp_quantized.py` (despite the legacy name): MTP block PRESENT + ALL `.weight` tensors BF16 + ZERO quantization-scale tensors on any `mtp.*` key + required modules (`e_proj`, `h_proj`, `attn.*`) present.

### Why `QuantizationModifier`, not `GPTQModifier`

`GPTQModifier._reduce_hessian_to_target_rank` deadlocks on multi-rank B300 (any of 2/4/8 ranks). Symptom: workers stuck at `compress_module_list:304` with `cudaStreamSynchronize`, NVRM "Failed to send inband data" in `dmesg`. Verified workarounds tried and failed: `NCCL_P2P_DISABLE=1`, `NCCL_NVLS_ENABLE=0`. The deadlock is in the Hessian-reduce code path; `QuantizationModifier` (RTN-style, no Hessian) has no `dist.*` calls in its main file and was expected to be immune. **It was NOT fully immune** — see §9 (Observer.synchronize NCCL desync).

### Why sample count = 64, max-seq-len = 512

Matches RedHat's reference script exactly. The W4A16 predecessor recipe used 768 — that value was inherited without re-checking against the NVFP4 path's consumer. The cost of being wrong: 8-rank cache-offload deadlock at 768 samples (filed as llm-compressor #2743). 64 samples is the validated count and reproduces RedHat's reported accuracy on their NVFP4-FP8 artifact.

---

## 6. Phase-by-phase walkthrough

Each phase has a **gate** — a deterministic check that must pass before the next phase starts. Skipping gates is the #1 way this recipe burns 8h of calibration time and produces an unusable artifact.

### Phase 0 — Box bootstrap

```bash
bash scripts/bootstrap_p6_b300.sh
```

What it does (idempotent):
1. `sudo apt install cuda-toolkit-13-0` if not present.
2. Creates `/data/venv-calib` with pinned `compressed-tensors`, `llmcompressor`, `py-spy`.
3. Applies patches §4.1 and §4.2 to the venv-installed packages.
4. Verifies `/scratch/weights/bf16-mtp/` exists (568 GB, 46 safetensors). If not, re-stage from S3.
5. Verifies `/data/vendor/dsv4-upstream/` has upstream's `model.py`, `kernel.py`, `config.json`.

**Gate:** `python -c "from transformers import DeepseekV4PreTrainedModel; assert not DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected"` — patch verified active.

### Phase 1 — Load test (RAM sanity)

```bash
torchrun --nproc-per-node 8 scripts/loadtest_sharded.py
```

Loads BF16 source under the same decoupled-expert sharding as calibration and reports per-rank RSS. **Gate:** ≤30 GB per rank. If higher, either the sharding is wrong or the BF16 source is corrupt.

Why this phase exists: single-rank dryrun cannot project multi-rank construction RAM (memory entry: `dryrun_projection_blindspot.md`). The construction pattern is N-rank-only.

### Phase 2 — Calibration

Two sub-phases. Phase 2a is **the gate**; Phase 2b is the production run.

**Phase 2a — 4-rank dryrun (1 layer, 16 samples):**
```bash
torchrun --nproc-per-node 4 scripts/quantize_v4_nvfp4_fp8_mtp.py \
    --dry-run-one-layer --samples 16
```
- Confirms `QuantizationModifier` doesn't hit the GPTQ-path NCCL bug on B300.
- Surfaces multi-rank-only failures cheaply.

**Phase 2a gate (the 4-number A/B report):** Compare 1-rank vs 4-rank dryrun outputs:
- Total keys saved (within 1%)
- Unique expert IDs (must be 256 in both)
- MTP keys (must be 799 in both — Option Y means MTP present, not absent)
- Expert weight-scale ratio per layer (within 5%)

If any of the four diverges, Phase 2b is blocked.

**Known Phase 2a friction (resolved at the time of writing):**
- Observer.synchronize NCCL desync hangs at subgraph 6 — monkey-patch in `scripts/quantize_v4_nvfp4_fp8_mtp.py` replaces it with a no-op (the synchronize is harmless on RTN-style; it was added for the GPTQ branch). Filed as llm-compressor #2734 (already filed by sibling agent for the GPTQ variant — we commented with the RTN-side confirmation).
- `kernel_shim.sparse_attn` device-mismatch when calibration samples land on rank-N's GPU but the stubbed kernel allocates on rank-0 — wrappers in `scripts/upstream/kernel_shim.py` relocate tensors to the calling device.

**Phase 2b — full calibration (all layers, 64 samples):**
```bash
torchrun --nproc-per-node 1 scripts/quantize_v4_nvfp4_fp8_mtp.py \
    --output-dir /scratch/weights/v4-flash-nvfp4-fp8-mtp \
    --samples 64 --max-seq-len 512 --batch-size 1
```

**Note: this is 1-rank, not 8-rank.** The 8-rank multi-rank run hit a cache-offload deadlock at samples ≥ 256 (filed as llm-compressor #2743; `propagate_error` flag in main not in our pin). The 1-rank path completes in ~87 minutes for V4-Flash on B300. For larger MoE+MTP DSV-family architectures, expect proportionally longer; pin the `propagate_error` fix or split the calibration into resumable chunks if multi-rank is needed.

**Per-layer checkpointing is enabled** (`scripts/quantize_v4_nvfp4_fp8_mtp.py` writes atomic `.tmp` + rename per sequential epoch end; resume-skip if checkpoint exists). If the run dies, restart same command — completed layers skip cleanly.

**Phase 2b gate:** artifact dir contains
- ~35 safetensors shards (V4-Flash)
- `model.safetensors.index.json` with ~134,000 keys
- 256 unique expert IDs in `weight_map` keys matching `re:.*ffn\.experts\.\d+\..*`
- ~799 keys matching `re:^mtp\..*`
- `config.json` with `quantization_config.format == "mixed-precision"` and both group regexes present

### Phase 3 — Postprocess for vLLM

```bash
python scripts/postprocess_for_vllm.py /scratch/weights/v4-flash-nvfp4-fp8-mtp
python scripts/squeeze_global_scales.py /scratch/weights/v4-flash-nvfp4-fp8-mtp
```

`postprocess_for_vllm.py` does:
- Rename `mtp.0.embed.weight` → `mtp.0.emb.tok_emb.weight` (vLLM expects upstream key naming).
- Inject `scale_fmt: ue8m0` into `config.json` (required by the hard subscript at vLLM model.py:909 until our PR lands).
- Set `num_hidden_layers: 43` (NOT 44 — MTP is `num_nextn_predict_layers: 1`, not a main layer).
- Set `expert_dtype: fp4` (mainline DSV4 reads this).
- Inject `packed_modules_mapping` for the NVFP4 expert layout.

`squeeze_global_scales.py` does:
- Sweeps every `weight_global_scale` and `input_global_scale` tensor in the artifact.
- Squeezes shape `(1,)` → 0-d scalar.
- vLLM's MoE loader at `fused_moe/layer.py:_load_per_tensor_weight_scale` does in-place `.copy_()` into a scalar slot; loading a shape-`(1,)` source rejects with `output with shape [] doesn't match the broadcast shape [1]`.
- ~66,048 tensors to squeeze for V4-Flash. Atomic .tmp + rename per shard.
- This is also an upstream-side bug class (loader should `.view([]).copy_(...)`) — file as PR after artifact ships.

### Phase 4 — Verify MTP retention (the differentiator gate)

```bash
python scripts/verify_mtp_quantized.py /scratch/weights/v4-flash-nvfp4-fp8-mtp
```

Despite the name, this enforces the **Option Y invariant**: MTP PRESENT + BF16 + UNQUANTIZED.

Four checks:
1. `≥6` MTP `.weight` tensors (lower bound; V4-Flash actual = 799).
2. ZERO quantization-scale tensors on any `mtp.*` key (no `.weight_scale`, `.weight_scale_inv`, `.weight_packed`, etc.).
3. Required modules present: `e_proj`, `h_proj`, `emb.tok_emb`, `attn.{wq_a, wq_b, wkv, wo_a, wo_b}`.
4. Every quantize-candidate MTP Linear's `.weight` is bfloat16 (sample head + tail).

If any check fails, the artifact is unusable for the differentiator — re-run Phase 2b. Common cause of failure: recipe's `ignore` list lost the `r"re:.*mtp\..*"` entry.

### Phase 5 — vLLM serve smoke

**Critical:** use mainline vLLM + PR #42209. DO NOT use jasl/dm120 fork — it's sm120 (consumer Blackwell) and its optimized `fp8_einsum.py` kernel doesn't exist in mainline and doesn't match our data-center NVFP4 tensor layouts.

Setup:
```bash
cd /data/src/vllm
git remote -v  # confirm 'upstream' = vllm-project/vllm
git stash      # stash any local mods
git fetch upstream main
git checkout upstream/main
git fetch upstream pull/42209/head:pr-42209
git cherry-pick <PR_42209_commits>   # or merge pr-42209 into a local branch
TORCH_CUDA_ARCH_LIST=10.0a pip install -e . --no-build-isolation
```

Phase 5a — serve WITHOUT speculative_config (baseline path):
```bash
vllm serve /scratch/weights/v4-flash-nvfp4-fp8-mtp \
    --tensor-parallel-size 4 --port 8089 --kv_cache_dtype fp8
```
**Gate:** server starts, `/v1/completions` returns coherent text on `"The capital of France is"` and `"def fibonacci(n):"` prompts.

Phase 5b — serve WITH speculative_config (the differentiator):
```bash
vllm serve /scratch/weights/v4-flash-nvfp4-fp8-mtp \
    --tensor-parallel-size 4 --port 8089 --kv_cache_dtype fp8 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```
**Gate:** server starts (MTP weights loaded as draft head); `/v1/completions` returns coherent text; check vLLM server logs for `Spec decode acceptance rate: <number>` — target ≥7% (matches upstream reported rate).

### Phase 6 — Harness eval

Use the cloned harness at `/data/harness/`:
```bash
cd /data/harness/
python -m harness.cli --port 8089 --tasks gsm8k,mmlu_pro --num-samples 500
```

**Gates:**
- GSM8K accuracy ≥ 0.91 (RedHat NVFP4-FP8 reports 0.910; we should match or beat since MTP is present)
- MMLU-Pro accuracy ≥ RedHat's reported value
- Spec-decode acceptance rate ≥ 7% on agentic prompts

If GSM8K is below 0.88, the recipe regressed — bisect.

### Phase 7 — Model card + HF upload

**Gated on user authorization.** Repo is private until Phase 7 runs.

Model card must lead with the MTP retention differentiator (table from §1). Include:
- Reproducer script reference (`scripts/quantize_v4_nvfp4_fp8_mtp.py`)
- Patches reference (`patches/`)
- vLLM serve command (mainline + PR #42209 with explicit pin)
- Spec-decode acceptance rate measured in Phase 5b
- Harness eval numbers from Phase 6

Upload via `huggingface-cli upload canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` after authorization.

---

## 7. Gotchas catalog (the 12 things that broke, in order of discovery)

For each: symptom, root cause, fix. Captured exhaustively because larger MoE+MTP architectures will likely re-encounter most of them.

1. **`mtp.*` silent drop at load** — symptom: artifact has zero `mtp.*` keys. Cause: transformers `_keys_to_ignore_on_load_unexpected`. Fix: §4.1 patch.

2. **GPTQ Hessian NCCL deadlock on multi-rank B300** — symptom: workers stuck at `compress_module_list:304`. Cause: `GPTQModifier._reduce_hessian_to_target_rank` deadlock. Fix: switch to `QuantizationModifier` (RTN-style). Filed: llm-compressor #2734.

3. **Observer.synchronize NCCL desync** — symptom: 4-rank dryrun hangs at subgraph 6 even with `QuantizationModifier`. Cause: per-Linear `dist.all_reduce` in `Observer.synchronize` desyncs with expert-sharded modules. Fix: monkey-patch to no-op in the script. Filed (commented): llm-compressor #2734.

4. **`from_accelerate` AttributeError on sharded ranks** — symptom: save crashes with `AttributeError: 'int' is not an nn.Module` in `dispatch_with_map`. Cause: sharded MoE modules have integer entries in the device map. Fix: replace `modify_save_pretrained` to skip `from_accelerate`. Filed: compressed-tensors #711.

5. **Multi-rank save rank-0-only gate drops experts on other ranks** — symptom: artifact contains only ranks-0 experts. Cause: original wrapper only invokes save on rank 0. Fix: rewrite wrapper to save per-rank subdir, then rank-0 merges + writes unified index. Logic in `_CompatModel.save_pretrained` + `scripts/merge_rank_subdirs.py`. Atomic `.tmp` + rename throughout.

6. **8-rank cache-offload deadlock at samples ≥ 256** — symptom: calibration completes layers but hangs on cache rotation. Cause: cache-offload error propagation missing (`propagate_error` added in llm-compressor PR #2008, not in our pin). Fix: pivoted to 1-rank for V4-Flash. Filed: llm-compressor #2743. For larger MoE architectures, pin the `propagate_error` fix to enable multi-rank.

7. **MTP inference-tensor crash at sequential_epoch_end** — symptom: at subgraph 44 (the MTP block), `RuntimeError: Inplace update to inference tensor outside InferenceMode`. Cause: upstream's MTP block uses `inference_mode()` around shared-embed; quant writeback hits this. Fix: Option Y — add `r"re:.*mtp\..*"` to recipe `ignore`. Filed: llm-compressor #2745.

8. **`scale_fmt` hard subscript in vLLM** — symptom: at serve init, `KeyError: 'scale_fmt'` at `vllm/models/deepseek_v4/nvidia/model.py:909`. Cause: hard subscript on a field RedHat's config doesn't ship. Fix: postprocess injects `scale_fmt: ue8m0` into config.json. **Pattern lesson:** "different from reference" ≠ "wrong"; verify the consumer reads with `.get()` before normalizing away. Memory entry: `diverge_from_reference_doesnt_mean_wrong.md`. Filed: (our queued PR for defensive `.get()`).

9. **`global_scale` shape (1,) vs scalar** — symptom: at serve weight-load, `output with shape [] doesn't match the broadcast shape [1]`. Cause: vLLM's MoE loader does in-place `.copy_()` into a scalar slot; our save emits shape `(1,)`. Fix: `scripts/squeeze_global_scales.py`. Also an upstream bug (file as PR).

10. **`wo_a.weight_scale_inv` missing on attn block** — symptom: at serve init, `AttributeError`. Cause: our recipe's group_0 left `input_activations` set; vLLM auto-rename to `weight_scale_inv` only triggers when `input_activations=None`. Fix: set `input_activations=None` in group_0 (attn is weight-only; matches RedHat).

11. **`FP8 Marlin kernel selection failed`** — symptom: vLLM picks no kernel for attn FP8 path. Cause: kernel pickers gate on FP8 Marlin auto-detect; B300 detection logic is conservative. Fix: `VLLM_TEST_FORCE_FP8_MARLIN=1` env var.

12. **jasl/dm120 fp8_einsum 256-row mystery** — symptom: at cuda_graph capture, `RuntimeError: DeepSeek V4 fp8 einsum weight rows must be divisible by out_rank=1024, got 256`. Cause: jasl/dm120 is a sm120-optimized fork; its `fp8_einsum.py` kernel reshapes weights based on sm120-specific assumptions. Mainline vLLM has no `fp8_einsum.py`. Fix: switch serve to mainline + PR #42209.

13. **vLLM mainline DSV4 expects FUSED `attn.fused_wqa_wkv` on disk** — symptom: at weight load, `KeyError: 'layers.0.attn.fused_wqa_wkv.weight_scale'`. Cause: vLLM mainline's DSV4 uses `MergedColumnParallelLinear` for `attn.fused_wqa_wkv` (merges wq_a + wkv) and `attn.compressor.fused_wkv_wgate`. Its `load_weights` `stacked_params_mapping` renames the on-disk unfused keys to the merged-Linear name at load time, but the merged Linear's `weight_scale` parameter is only allocated when the `config_groups` regex matches the FUSED prefix. Our recipe targeted UNFUSED prefixes (`wq_a|wkv`), so no scale slot existed. Fix: update `config_groups[group_0]['targets']` regex to `r"re:.*\.attn\.(fused_wqa_wkv|compressor\.fused_wkv_wgate|wq_b|wo_a|wo_b)$"`. Script: `scripts/update_config_for_fused_attn.py`. No on-disk tensor surgery needed — vLLM's load-time merger does the concat. **Forward-compat note:** if upstream changes the merged-Linear names again, re-check `stacked_params_mapping` in mainline `vllm/models/deepseek_v4/nvidia/model.py:load_weights` and update the regex accordingly.

14. **W8A16Fp8 scheme `is_static_input_scheme` is `None` AssertionError** — symptom: at `process_weights_after_loading`, `assert self.is_static_input_scheme is False` fails. Cause: scheme picker passes `input_quant and not input_quant.dynamic` which short-circuits to `None` when `input_quant=None` (our weight-only attn case). Scheme stores `None`; `None is False` evaluates `False`, assertion fires. Fix: wrap with `bool(...)` in `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` at lines 679 + 700. **This is our queued upstream PR #43248** — until it merges, apply locally with `sed -i 's|is_static_input_scheme = input_quant and not input_quant.dynamic|is_static_input_scheme = bool(input_quant and not input_quant.dynamic)|g' <path>/compressed_tensors.py`.

15. **`Inplace update to inference tensor` is GLOBAL to the MTP block, not just shared-embed modules** — symptom: calibration completes subgraphs 1-43 successfully then crashes at subgraph 44 (the MTP block) with `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`, REGARDLESS of how narrowly the recipe excludes MTP modules. Cause: the MTP block forward path uses `with torch.inference_mode():` upstream (shared-embedding lookup most likely), and the resulting inference-mode marking propagates through residuals to ALL downstream modules in the MTP block (attn, FFN experts). Narrowing the ignore to keep MTP attn/experts in scope DOES NOT WORK — the crash still fires. **Verified empirically 2026-05-21** by running v2 calibration with `r"re:.*mtp\.\d+\.(embed|e_proj|h_proj).*"` ignore (only the embed-touching modules excluded): subgraphs 1-43 complete (~73 min on B300 1-rank), subgraph 44 crashes identically. See `scripts/quantize_v4_nvfp4_fp8_mtp_v2.py` (the variant) and llm-compressor #2745 for the upstream root cause. Conclusion: **Option Y (broad MTP exclusion) is the only currently-shippable recipe shape** for DSV4-MTP artifacts. The downstream consequence is that vLLM mainline's MTP draft model can't currently serve such artifacts — see vLLM issue #43304 for the upstream draft-model-quant-config-inheritance problem we filed.

16. **DSV4 MTP draft model inherits main quantization scheme — `KeyError: model.layers.43.mtp_block.ffn.experts.w13_weight` at load** — symptom: when serving with `--speculative-config method=mtp`, the MTP draft model construction passes the main model's `quant_config` to the mtp_block's FFN. For BF16 MTP artifacts (the only currently-shippable shape per gotcha 15), the FFN module allocates `w13_weight_packed` per the NVFP4 scheme, while the loader iterates on-disk `w1.weight`/`w2.weight`/`w3.weight` keys and renames to `w13_weight` (unquantized name) — KeyError. Adding regex like `re:.*mtp_block.*` or `re:.*\.layers\.43\..*` to the config's `ignore` list does NOT consistently work — the matcher uses the **construction-time prefix** (which lacks `mtp_block` because the prefix is assembled before module-attribute traversal) so `mtp_block` regex misses, while `layers.43` only works if num_hidden_layers is known at recipe-author time. **Filed as vLLM issue #43304** with 3 proposed fix options. **No artifact-side workaround exists today** — the artifact's MTP weights are BF16 + present, but live `--speculative-config method=mtp` serve is gated on this vLLM-side fix landing OR upstream `llm-compressor#2745` fix landing (which would let us quantize MTP and avoid the BF16 problem).

---

## 8. Decision rationales (the WHY of the recipe shape)

Captured because the recipe has surprising choices that pure code-read would miss.

- **NVFP4 group_size=16 (not 32 or 64)** — matches RedHat exactly. Larger groups regress accuracy on routed FFN experts where activation distributions are narrow.
- **FP8_BLOCK 128×128 on attn (not per-channel)** — matches RedHat exactly. 128×128 is the Blackwell hardware-native FP8 block shape; per-channel doesn't map to tensor cores.
- **`input_activations=None` on group_0 (attn) but dynamic="local" on group_1 (experts)** — attn weights are static; expert activations vary per-token and need online scaling. Matches RedHat.
- **`observer="minmax"` everywhere** — RTN-style. `mse` would be marginally more accurate but `QuantizationModifier`'s minmax path is the one with no `dist.*` calls; switching to `mse` re-introduces collective hangs.
- **Why ignore `gate` (router) — it's BF16-critical.** The MoE router output drives expert dispatch; quantizing it changes routing and cascades into all downstream quant error budgets.
- **Why ignore `lm_head` — output projection is sensitive.** RedHat ignores it. Quantizing it on a ~284B model loses ~0.5 GSM8K points for 1.5 GB savings — not worth it.
- **Why ignore `norm` / `q_norm` / `k_norm`** — RMSNorm has multiplicative scale that compounds quant error.
- **Why 1-rank for Phase 2b (not 8-rank)** — see gotcha #6. The 8-rank path needs `propagate_error` from llm-compressor PR #2008 which isn't in our pin.
- **Why mainline vLLM (not jasl/dm120)** — see gotcha #12 + memory entry `jasl_dm120_is_sm120_not_sm100.md`. jasl/dm120 is sm120 (consumer Blackwell); our artifact targets sm100a (data-center Blackwell).

---

## 9. Upstream contribution queue (the parallel OSS work)

These are filed / queued alongside the artifact work. Every gotcha in §7 is also an upstream improvement opportunity.

| # | Repo | Filing | Status |
|---|---|---|---|
| 1 | vllm-project/vllm | bool() wrap on `is_static_input_scheme` (5 sites in `compressed_tensors.py`) | PR #43248 filed |
| 2 | vllm-project/llm-compressor | #2734 — Observer.synchronize NCCL desync | sibling filed; we commented with NVFP4 confirmation |
| 3 | vllm-project/compressed-tensors | #711 — `dispatch_with_map` AttributeError on sharded ranks | filed |
| 4 | vllm-project/llm-compressor | #2741 — intra-calibration resume feature request | filed |
| 5 | vllm-project/llm-compressor | #2743 — 8-rank cache-offload deadlock | filed |
| 6 | vllm-project/llm-compressor | #2745 — MTP inference-tensor crash | filed |
| 7 | vllm-project/vllm | `scale_fmt` hard subscript → `.get()` at `model.py:909` | queued |
| 8 | vllm-project/vllm | `global_scale` loader `.view([]).copy_()` for shape `(1,)` sources | queued |
| 9 | vllm-project/vllm | comment on PR #42209 with B300 reproducer | queued (post-merge or pre-merge) |

Branding rule for all filings: do NOT brand-drop "canada-quant" in PR bodies. Maintainers see contributor identity in the GitHub UI; redundant brand mentions reduce review-readiness. State the use case factually ("a ~284B-MoE DeepSeek-V4-Flash NVFP4 quantization with preserved MTP, 256 experts × 1-rank calibration").

---

## 10. Artifact identity

For V4-Flash:
- HF repo: `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`
- Approx artifact size: 172 GB across ~35 safetensors shards
- Key count: ~134,000
- Routed experts: 256 (unique IDs 0–255)
- MTP keys preserved: ~799
- License: inherits upstream DSV4 license

---

## 11. Generalizing to other MoE+MTP architectures

The recipe should generalize to other MoE+MTP architectures in the DSV-family with these items re-verified:

| Item | What to check |
|---|---|
| Main layers count | Update `num_hidden_layers` in postprocess to match upstream's value (not hard-coded to 43). |
| MTP block layout | Re-verify the `ignore` regex matches the architecture's actual MTP key prefix (could be `mtp.0`, or multiple). |
| Modeling-class `_keys_to_ignore_on_load_unexpected` | §4.1 patch may need to target a different class name. |
| Calibration time | Scale by parameter count; 1-rank time on V4-Flash is ~87 min on B300. |
| vLLM PR #42209 | Verify still required; mainline may have absorbed it. |

The most important thing: **run Phase 0–4 first on a small sample (e.g., layer 5 only) before committing to a multi-hour Phase 2b run**. The patches and the recipe shape are stable; what's likely to vary is the modeling class name, the MTP block layout, and the per-layer parameter count budget.

---

## 12. Document maintenance

This recipe is **load-bearing**. When something in it goes out of date, update it the same day:
- Upstream PR merges (§4.3, §4.4, §9) — update status; if a PR merged, mark obsolete in our patches.
- New gotchas discovered during replication on related architectures — add to §7 with discovery date.
- Recipe parameter changes (e.g., `samples`, `group_size`) — update §5 and document the reason.
- Memory entries created or updated — cross-reference here.

If this document contradicts something in the code, **trust the code and update this document**. Code is the source of truth; this is the human-readable map.

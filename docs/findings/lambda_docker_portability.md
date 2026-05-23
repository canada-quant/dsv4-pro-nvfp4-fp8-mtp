# Lambda docker image portability test — `vllm/vllm-openai:deepseekv4-cu130`

**Result**: This artifact does NOT load on the partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image as of 2026-05-23. The image's pre-#42209 vLLM build fails with the same `KeyError: 'layers.0.ffn.experts.w13_input_scale'` we see on mainline when NVFP4 dispatch doesn't include PR #42209's routing.

Use the mainline + 4-patches build path from [`docs/QUICKSTART.md`](../QUICKSTART.md) instead.

## Setup

- Hardware: same 8× B300 SXM6 AC as the matrix benches
- Image: `vllm/vllm-openai:deepseekv4-cu130` (digest `2e05966d0557`, content size 8.16 GB)
- Image's vLLM build: `v0.1.dev15833+g62d441ee8` (pre-mainline, from zyongye fork branch — predates PR #42209's merge on 2026-05-22)
- Test command (matching our local serve config, except `flashinfer-autotune` disabled to speed up startup):

```bash
docker run --rm \
  --gpus all --ipc host \
  -v /opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp:/model:ro \
  -p 8090:8000 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  vllm/vllm-openai:deepseekv4-cu130 \
  --model /model \
  --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --max-model-len 65536 \
  --no-enable-flashinfer-autotune
```

## Failure

The container starts cleanly through engine init, recognizes the model as `DeepseekV4ForCausalLM`, and starts loading shards. At ~1% load progress (after first shard) each of the 8 workers raises:

```
Worker_TPx_EPx ERROR multiproc_executor.py:879 KeyError: 'layers.0.ffn.experts.w13_input_scale'
EngineCore failed to start.
RuntimeError: Engine core initialization failed.
```

Full failure trace in `phaseF_docker.log` on the bench host.

The KeyError is at `params_dict[name_mapped]` during the weight load's expert-mapping step. The on-disk per-expert key `experts.{E}.w1.input_scale` is being mapped to a fused name `experts.w13_input_scale` that the image's MoE method does not register.

## Why this happens

The image's vLLM (`v0.1.dev15833+g62d441ee8`, ~April 24) predates the merge of PR #42209 (NVFP4 MoE support for DSV4) on 2026-05-22. Without #42209's `ModelOptNvFp4FusedMoE` registering `w13_input_scale` etc., the loader has nowhere to put the per-expert input-scale tensors and raises.

This is the same failure mode that issue [#43454](https://github.com/vllm-project/vllm/issues/43454) documents for `--moe-backend deep_gemm_mega_moe` on mainline — but on the docker image it appears even with `--moe-backend flashinfer_trtllm` because #42209's NVFP4 routing isn't installed at all.

## What works

The recipe we ship in [`docs/QUICKSTART.md`](../QUICKSTART.md):
- vLLM mainline pinned at `39910f2b25` (2026-05-22 snapshot)
- Cherry-pick of PR #42209 (merged 2026-05-22 14:21 UTC — after our pinned SHA 00:21 UTC, so cherry-pick adds it explicitly)
- 4 local patches (#43248, #43288, #43290, #43319)
- Build with `TORCH_CUDA_ARCH_LIST=10.3a` for sm_103a (B300)

Installer script: `scripts/install_vllm_with_patches.sh`.

## Implication for the MODEL_CARD

We claim users should serve via the installer + mainline build, NOT via `vllm/vllm-openai:deepseekv4-cu130`. The docker image will be a viable path once it's rebuilt from a vLLM mainline that includes PR #42209 (or a successor docker tag based on a post-#42209 fork build).

This is consistent with the upstream YAML's `min_vllm_version: "0.20.0"` constraint being more permissive than the *actual* compatibility surface for NVFP4-quantized artifacts — the YAML pre-dates the NVFP4 routing landing. We document the gap so users don't waste time pulling 8 GB of image to discover the incompatibility.

## Future work

When `vllm/vllm-openai` ships an updated `deepseekv4-cu130` tag (or any other tag) built from a post-#42209 vLLM mainline, repeat this test. If it then loads cleanly, update this doc and the MODEL_CARD's portability section.

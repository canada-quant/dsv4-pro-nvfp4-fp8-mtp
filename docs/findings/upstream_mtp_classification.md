# Upstream classification of V4-Pro MTP — evidence summary

This doc captures the upstream-side evidence that V4-Pro MTP is a known-weak area today, used to contextualize our 1.82% acceptance measurement.

## Evidence 1: vLLM official recipe lists MTP under `opt_in_features`

From [`vllm-project/recipes/models/deepseek-ai/DeepSeek-V4-Pro.yaml`](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Pro.yaml) (page last updated 2026-04-24):

```yaml
features:
  spec_decoding:
    description: "Multi-Token Prediction speculative decoding with 2 speculative tokens."
    args:
      - "--speculative_config"
      - '{"method":"mtp","num_speculative_tokens":2}'

opt_in_features:
  - spec_decoding
```

The `opt_in_features` list is the YAML's explicit non-default flag set. The default-strategy is `single_node_tep` with no MTP. Tool calling and reasoning-parser, by contrast, are *also* in `features` but not in `opt_in_features` — meaning the maintainers blessed them as default-on. MTP is the only feature in the opt-in list. The upstream-maintained recipe says: "MTP is here in the checkpoint, you can turn it on, but we don't think it's the right default."

## Evidence 2: LMSYS day-zero V4-Pro blog reports accept length ~1.19

From the LMSYS day-zero benchmark of V4-Pro on the officially partner-blessed `vllm/vllm-openai:deepseekv4-cu130` docker image:

> MTP-3 on B200 Pro (accept ~1.19). The per-position breakdown is heavily skewed -- positions 0/1/2 accept 2226 / 354 / 55 tokens respectively -- so the spec path looks like it is mostly accepting only position 0. This suggests the MTP path may not be hitting full effectiveness on Pro; we did not investigate further.

Accept length 1.19 at MTP-3 means an average of 0.19 extra tokens per draft step, which translates to a per-token acceptance rate of 0.19 / 3 ≈ 6.3%. This is on the officially partner-blessed deployment with the docker image LMSYS rebuilt from `zyongye/vllm@bc34b25e` (the V4-Pro fork branch).

Our measurement (1.82% per-token at MTP n=2; accept length 1.036) is in the same regime as LMSYS's number — both fall well short of V4-Flash's ~80% acceptance on chat-templated coding and ~81.6% on AIME reasoning, both of which were measured on essentially the same vLLM MTP RejectionSampler path. The MTP code itself is generic; the difference is the trained MTP head.

## Evidence 3: V4-Pro `mtp.0.*` is structurally different from V4-Flash's

| Field | V4-Flash | V4-Pro |
|---|---|---|
| num_nextn_predict_layers | 1 | 1 |
| Trained MTP loss curve at end of training (per DeepSeek tech report) | converged steady | reported as "weaker than main model" with note that V4-Pro postponed full MTP optimization to focus on the larger trunk |
| Hidden size routed into MTP draft | 4096 | 7168 |
| Number of routed experts in MTP | 256 | 384 |

The MTP block at the V4-Pro scale appears to be undertrained relative to the trunk. This is consistent with both LMSYS's "not hitting full effectiveness on Pro" framing and our measurement.

## Implication for this artifact

The 1.82% acceptance we measure is **not** a sign that our MXFP4 → NVFP4 conversion damaged the MTP signal. It is consistent with the known upstream baseline. The artifact is still useful with MTP off (it's the throughput-leader cell — see `backend_format_matrix.md`), and MTP is retained on disk so users can:

1. Turn it on as a low-cost extra (the rejection-sample overhead is small at this acceptance rate),
2. Benefit automatically if a future MTP-improvement PR/checkpoint update raises the acceptance.

The cost of retaining MTP in the artifact is small: the MTP block is ~1% of total weights (~10 GB across 64 shards), and it's BF16-pinned for `e_proj`/`h_proj` (200 MB) for the loader-workaround reason documented separately. Net cost of MTP retention vs strip: ~10 GB out of 852 GB.

## What this does NOT say

This doc does not assert that the conversion is *perfect* on MTP semantics — only that the measured acceptance is in the upstream-known range and that the dominant cause is the trained head, not the conversion. An offline mtp.0 forward-pass comparison between our BF16-dequanted `e_proj`/`h_proj` and the original FP8 versions would tighten this conclusion. That's listed as future work in `mtp_eproj_hproj_workaround.md`.

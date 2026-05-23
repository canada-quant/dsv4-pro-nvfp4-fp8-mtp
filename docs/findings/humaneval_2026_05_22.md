# HumanEval / HumanEval+ EvalPlus (NVFP4, no MTP) — 2026-05-22

## Headline

| Benchmark | This artifact (NVFP4) | V4-Flash NVFP4 (no spec) | RedHat V4-Flash NVFP4 (no MTP) |
|---|---|---|---|
| HumanEval base pass@1 | **0.951** | 0.915 | 0.896 |
| HumanEval+ pass@1 | **0.896** | 0.854 | 0.860 |

V4-Pro NVFP4 beats V4-Flash NVFP4 on both metrics:
- HumanEval base: +3.6pt (0.915 → 0.951)
- HumanEval+: +4.2pt (0.854 → 0.896)

And ahead of RedHat V4-Flash NVFP4 by +5.5pt / +3.6pt respectively. Coding is the strongest quality differentiator we've measured.

## Setup

- Hardware: 8× B300 SXM6 AC
- vLLM mainline @ `39910f2b25` + 4 patches
- Serve config: `--tensor-parallel-size 8 --enable-expert-parallel --moe-backend flashinfer_trtllm --attention_config.use_fp4_indexer_cache=True --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`, max-model-len 131072
- Tooling: `evalplus 0.3.1` codegen + evaluate
- Backend: OpenAI-compat protocol against our vLLM serve (`--backend openai --base-url http://localhost:8089/v1`)
- Sampling: **greedy** (`--greedy True`), `n_samples=1`
- MTP off

## Notes on EvalPlus invocation

- The `evalplus.codegen` CLI fails with `AssertionError: Temperature must be positive for sampling` when called without `--greedy`; EvalPlus's default is sampling mode but its `openai.py` asserts `temperature > 0`. Set `--greedy True` for pass@1 evaluation (which expects greedy decoding).
- The `evalplus.evaluate` CLI auto-locates the samples file under the codegen output root.

Raw results: `docs/benchmarks/matrix/humaneval/`.

## Reproduction

```bash
# With serve up at port 8089 (see docs/QUICKSTART.md):
pip install evalplus
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8089/v1

evalplus.codegen \
  canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP \
  humaneval \
  --backend openai \
  --base-url http://localhost:8089/v1 \
  --bs 1 \
  --greedy True \
  --root humaneval_out

evalplus.evaluate --dataset humaneval --samples humaneval_out/<the-jsonl-file>
```

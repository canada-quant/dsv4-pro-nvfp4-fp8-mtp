#!/usr/bin/env bash
# Run all V4-Pro benchmarks sequentially against an already-up vLLM serve.
# Outputs go to docs/benchmarks/<bench>_<label>_<date>.json.

set -e

MODEL="${1:?usage: $0 <model_path> <label> [base_url]}"
LABEL="${2:?usage: $0 <model_path> <label> [base_url]}"
BASE_URL="${3:-http://localhost:8089}"
DATE=$(date -u +%Y_%m_%d)

PY="${PY:-/opt/pytorch/bin/python}"
BENCH="${BENCH:-/tmp/bench_v4_pro.py}"

echo "==== Smoke prompt ===="
curl -s "$BASE_URL/v1/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Write one short sentence about a green apple.\"}],
  \"max_tokens\": 64,
  \"temperature\": 0
}" | $PY -c 'import sys, json; r = json.load(sys.stdin); print("response:", r["choices"][0]["message"]["content"]); print("finish_reason:", r["choices"][0]["finish_reason"])'
echo

echo "==== Latency (50 prompts, c=1) ===="
$PY $BENCH latency --base-url "$BASE_URL" --model "$MODEL" --label "$LABEL" --n 50 --max-tokens 256

echo
echo "==== MTP acceptance (20 prompts) ===="
$PY $BENCH mtp --base-url "$BASE_URL" --model "$MODEL" --label "$LABEL"

echo
echo "==== GSM8K (1319 problems, c=16, max_tokens=2048) ===="
$PY $BENCH gsm8k --base-url "$BASE_URL" --model "$MODEL" --label "$LABEL" --concurrency 16 --max-tokens 2048

echo
echo "==== AIME 2024 (30 problems, c=8, max_tokens=65536, thinking) ===="
$PY $BENCH aime --base-url "$BASE_URL" --model "$MODEL" --label "$LABEL" --concurrency 8 --max-tokens 65536 --timeout 1800

echo
echo "==== DONE — outputs at docs/benchmarks/*_${LABEL}_${DATE}.json ===="

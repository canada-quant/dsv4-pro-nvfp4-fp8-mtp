#!/usr/bin/env bash
# Eager vs CUDA-graph A/B on NVFP4 V4-Pro.
# Kills any running serve, runs N=10 latency prompts in eager mode, then
# N=10 in cuda-graph mode (no --enforce-eager). Diff tells us whether the
# `--enforce-eager` flag is throughput-killing.
#
# Usage:
#   scripts/eager_vs_graph_ab.sh

set -e

MODEL=/opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp

run_serve() {
    local label="$1"; shift
    local extra_args="$@"
    echo "==== starting serve [$label] with: $extra_args ===="
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 5
    nohup bash -c "
export PATH=/opt/pytorch/bin:\$HOME/.cargo/bin:\$PATH
export CUDA_HOME=/usr/local/cuda
/opt/pytorch/bin/vllm serve $MODEL \\
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \\
  --enable-expert-parallel --tensor-parallel-size 8 \\
  --moe-backend flashinfer_trtllm $extra_args \\
  --port 8089 2>&1
" > /tmp/v4pro_serve_${label}.log 2>&1 &
    SPID=$!
    echo "$SPID" > /tmp/v4pro_serve_${label}.pid

    while ps -p $SPID > /dev/null 2>&1; do
        if grep -qE "Application startup complete" /tmp/v4pro_serve_${label}.log 2>/dev/null; then
            echo "  [$label] READY"
            return 0
        fi
        if grep -qE "WorkerProc hit|RuntimeError: Engine|EngineCore failed" /tmp/v4pro_serve_${label}.log 2>/dev/null; then
            echo "  [$label] FAILED"
            tail -20 /tmp/v4pro_serve_${label}.log
            return 1
        fi
        sleep 15
    done
    echo "  [$label] serve died unexpectedly"
    return 1
}

SPEC_CFG='--speculative-config {"method":"mtp","num_speculative_tokens":1}'

# ---- A: eager mode + MTP ----
run_serve "eager_mtp" "--enforce-eager $SPEC_CFG"
/opt/pytorch/bin/python /tmp/bench_v4_pro.py latency \
    --base-url http://localhost:8089 --model "$MODEL" \
    --label "nvfp4_eager_mtp_ab" --n 10 --max-tokens 128 \
    --output docs/benchmarks/latency_nvfp4_eager_mtp_ab_$(date -u +%Y_%m_%d).json
/opt/pytorch/bin/python /tmp/bench_v4_pro.py mtp \
    --base-url http://localhost:8089 --model "$MODEL" \
    --label "nvfp4_eager_mtp_ab" \
    --output docs/benchmarks/mtp_nvfp4_eager_mtp_ab_$(date -u +%Y_%m_%d).json

# ---- B: cuda-graph mode + MTP ----
run_serve "graph_mtp" "$SPEC_CFG"
/opt/pytorch/bin/python /tmp/bench_v4_pro.py latency \
    --base-url http://localhost:8089 --model "$MODEL" \
    --label "nvfp4_graph_mtp_ab" --n 10 --max-tokens 128 \
    --output docs/benchmarks/latency_nvfp4_graph_mtp_ab_$(date -u +%Y_%m_%d).json
/opt/pytorch/bin/python /tmp/bench_v4_pro.py mtp \
    --base-url http://localhost:8089 --model "$MODEL" \
    --label "nvfp4_graph_mtp_ab" \
    --output docs/benchmarks/mtp_nvfp4_graph_mtp_ab_$(date -u +%Y_%m_%d).json

echo "==== A/B DONE ===="
echo "compare:"
ls -la docs/benchmarks/latency_nvfp4_*_ab_*.json

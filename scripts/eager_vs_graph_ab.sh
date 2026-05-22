#!/usr/bin/env bash
# Eager vs CUDA-graph A/B on NVFP4 V4-Pro (both with MTP enabled).
# Kills any running serve, starts run-A (eager+MTP), benchmarks 10 latency
# + 20 MTP probes, kills, starts run-B (cuda-graph + MTP), same probes.
#
# Outputs:
#   docs/benchmarks/latency_nvfp4_eager_mtp_ab_<date>.json
#   docs/benchmarks/mtp_nvfp4_eager_mtp_ab_<date>.json
#   docs/benchmarks/latency_nvfp4_graph_mtp_ab_<date>.json
#   docs/benchmarks/mtp_nvfp4_graph_mtp_ab_<date>.json
#
# Usage:
#   scripts/eager_vs_graph_ab.sh

set -e

MODEL=/opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp
DATE=$(date -u +%Y_%m_%d)

run_serve_eager() {
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 5
    nohup bash -c '
export PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda
/opt/pytorch/bin/vllm serve /opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --tensor-parallel-size 8 \
  --moe-backend flashinfer_trtllm --enforce-eager \
  --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":1}'"'"' \
  --port 8089 2>&1
' > /tmp/v4pro_serve_eager_mtp.log 2>&1 &
    echo $! > /tmp/v4pro_serve_eager_mtp.pid
}

run_serve_graph() {
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 5
    nohup bash -c '
export PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda
/opt/pytorch/bin/vllm serve /opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --enable-expert-parallel --tensor-parallel-size 8 \
  --moe-backend flashinfer_trtllm \
  --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":1}'"'"' \
  --port 8089 2>&1
' > /tmp/v4pro_serve_graph_mtp.log 2>&1 &
    echo $! > /tmp/v4pro_serve_graph_mtp.pid
}

wait_for_ready() {
    local label="$1"
    local logfile="/tmp/v4pro_serve_${label}.log"
    local pidfile="/tmp/v4pro_serve_${label}.pid"
    local pid=$(cat "$pidfile")
    echo "[$label] waiting on PID $pid..."
    while ps -p "$pid" > /dev/null 2>&1; do
        if grep -qE "Application startup complete" "$logfile" 2>/dev/null; then
            echo "[$label] READY"
            return 0
        fi
        if grep -qE "WorkerProc hit|RuntimeError: Engine|EngineCore failed" "$logfile" 2>/dev/null; then
            echo "[$label] FAILED"
            tail -30 "$logfile"
            return 1
        fi
        sleep 20
    done
    echo "[$label] serve died unexpectedly"
    tail -30 "$logfile"
    return 1
}

probe() {
    local label="$1"
    /opt/pytorch/bin/python /tmp/bench_v4_pro.py latency \
        --base-url http://localhost:8089 --model "$MODEL" \
        --label "${label}_ab" --n 10 --max-tokens 128 \
        --output "docs/benchmarks/latency_${label}_ab_${DATE}.json"
    /opt/pytorch/bin/python /tmp/bench_v4_pro.py mtp \
        --base-url http://localhost:8089 --model "$MODEL" \
        --label "${label}_ab" \
        --output "docs/benchmarks/mtp_${label}_ab_${DATE}.json"
}

echo "==== A: eager + MTP ===="
run_serve_eager
wait_for_ready "eager_mtp"
probe "nvfp4_eager_mtp"

echo "==== B: cuda-graph + MTP ===="
run_serve_graph
wait_for_ready "graph_mtp"
probe "nvfp4_graph_mtp"

echo "==== A/B DONE ===="
ls -la docs/benchmarks/latency_nvfp4_*_ab_*.json docs/benchmarks/mtp_nvfp4_*_ab_*.json

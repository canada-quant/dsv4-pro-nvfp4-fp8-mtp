#!/usr/bin/env bash
# Backend × format matrix for V4-Pro on B300 + TP=8 + EP (upstream default `single_node_tep`)
#   Cells:
#     A) NVFP4 + flashinfer_trtllm + indexer_cache + FULL_AND_PIECEWISE + MTP n=2
#     B) NVFP4 + deep_gemm_mega_moe + indexer_cache + FULL_AND_PIECEWISE + MTP n=2
#     C) Native MXFP4 + deep_gemm_mega_moe + indexer_cache + FULL_AND_PIECEWISE (no MTP)
#     D) Native MXFP4 + flashinfer_trtllm + indexer_cache + FULL_AND_PIECEWISE (no MTP)

set -u
DATE=$(date -u +%Y_%m_%d)
NVFP4=/opt/dlami/nvme/weights/v4-pro-nvfp4-fp8-mtp
MXFP4=/opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp
OUT=/tmp/matrix_out
mkdir -p "$OUT"

kill_serve() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
    pkill -9 -f "VLLM::" 2>/dev/null
    pkill -9 -f "vllm serve" 2>/dev/null
    sleep 6
}

wait_ready() {
    local logfile="$1"
    local pid="$2"
    local label="$3"
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q "Application startup complete" "$logfile" 2>/dev/null; then
            echo "[$label] READY after ${elapsed}s"
            return 0
        fi
        if grep -qE "EngineCore failed|WorkerProc hit|KeyError" "$logfile" 2>/dev/null; then
            echo "[$label] FAILED after ${elapsed}s — error tail:"
            tail -40 "$logfile" | grep -E "Error|KeyError|Traceback" | head -10
            return 1
        fi
        sleep 15
        elapsed=$((elapsed + 15))
        if [ $elapsed -gt 900 ]; then
            echo "[$label] TIMEOUT after ${elapsed}s"
            return 1
        fi
    done
    echo "[$label] process died"
    tail -20 "$logfile"
    return 1
}

run_serve() {
    local label="$1"; shift
    local model="$1"; shift
    local logfile="$OUT/serve_${label}.log"
    echo "==== Launching $label ===="
    kill_serve
    nohup env PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda \
        /opt/pytorch/bin/vllm serve "$model" "$@" > "$logfile" 2>&1 &
    local pid=$!
    echo "[$label] PID=$pid"
    wait_ready "$logfile" "$pid" "$label"
    return $?
}

probe() {
    local label="$1"
    local model="$2"
    /opt/pytorch/bin/python /tmp/bench_v4_pro.py latency \
        --base-url http://localhost:8089 \
        --model "$model" \
        --label "$label" --n 20 --max-tokens 128 \
        --output "$OUT/latency_${label}_${DATE}.json" 2>&1 | tee "$OUT/probe_${label}.log"
}

probe_mtp() {
    local label="$1"
    local model="$2"
    /opt/pytorch/bin/python /tmp/bench_v4_pro.py mtp \
        --base-url http://localhost:8089 \
        --model "$model" \
        --label "$label" \
        --output "$OUT/mtp_${label}_${DATE}.json" 2>&1 | tee "$OUT/mtp_probe_${label}.log"
}

# -------- Cell A: NVFP4 + flashinfer_trtllm + MTP --------
run_serve "A_nvfp4_flashinfer_mtp" "$NVFP4" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8089 \
&& probe "A_nvfp4_flashinfer_mtp" "$NVFP4" \
&& probe_mtp "A_nvfp4_flashinfer_mtp" "$NVFP4"

# -------- Cell B: NVFP4 + deep_gemm_mega_moe + MTP --------
run_serve "B_nvfp4_deepgemm_mtp" "$NVFP4" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8089 \
&& probe "B_nvfp4_deepgemm_mtp" "$NVFP4" \
&& probe_mtp "B_nvfp4_deepgemm_mtp" "$NVFP4"

# -------- Cell C: Native MXFP4 + deep_gemm_mega_moe (NO MTP — native+MTP can't load) --------
run_serve "C_mxfp4_deepgemm" "$MXFP4" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8089 \
&& probe "C_mxfp4_deepgemm" "$MXFP4"

# -------- Cell D: Native MXFP4 + flashinfer_trtllm (NO MTP) --------
run_serve "D_mxfp4_flashinfer" "$MXFP4" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8089 \
&& probe "D_mxfp4_flashinfer" "$MXFP4"

echo "==== MATRIX DONE ===="
ls -la "$OUT"

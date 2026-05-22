#!/usr/bin/env bash
# Pre-ship extension matrix: 4 cells to firm up the headline.
#   E) NVFP4 + flashinfer NO MTP c=1 latency (rest of headline overhead removed)
#   F) NVFP4 + flashinfer NO MTP c=16 batched latency
#   G) Native MXFP4 + deep_gemm c=16 batched latency
#   I) Native MXFP4 + deep_gemm GSM8K 300-problem subset (quality reference)

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
    local logfile="$1"; local pid="$2"; local label="$3"
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q "Application startup complete" "$logfile" 2>/dev/null; then
            echo "[$label] READY after ${elapsed}s"; return 0
        fi
        if grep -qE "EngineCore failed|WorkerProc hit|KeyError" "$logfile" 2>/dev/null; then
            echo "[$label] FAILED"; tail -20 "$logfile" | grep -E "Error|KeyError|Traceback" | head -10; return 1
        fi
        sleep 15; elapsed=$((elapsed + 15))
        if [ $elapsed -gt 900 ]; then echo "[$label] TIMEOUT"; return 1; fi
    done
    echo "[$label] process died"; tail -20 "$logfile"; return 1
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
}

# -------- E: NVFP4 + flashinfer NO MTP c=1 --------
run_serve "E_nvfp4_flashinfer_noMTP" "$NVFP4" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend flashinfer_trtllm \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8089 \
&& /opt/pytorch/bin/python /tmp/bench_v4_pro.py latency \
    --base-url http://localhost:8089 --model "$NVFP4" \
    --label "E_nvfp4_noMTP_c1" --n 20 --max-tokens 128 \
    --output "$OUT/latency_E_nvfp4_noMTP_c1_${DATE}.json" \
&& /opt/pytorch/bin/python /tmp/bench_v4_pro.py latency \
    --base-url http://localhost:8089 --model "$NVFP4" \
    --label "F_nvfp4_noMTP_c16" --n 64 --max-tokens 128 --concurrency 16 \
    --output "$OUT/latency_F_nvfp4_noMTP_c16_${DATE}.json"

# -------- G: Native MXFP4 + deep_gemm c=16 + GSM8K --------
run_serve "G_mxfp4_deepgemm_c16" "$MXFP4" \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention_config.use_fp4_indexer_cache=True \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8089 \
&& /opt/pytorch/bin/python /tmp/bench_v4_pro.py latency \
    --base-url http://localhost:8089 --model "$MXFP4" \
    --label "G_mxfp4_deepgemm_c16" --n 64 --max-tokens 128 --concurrency 16 \
    --output "$OUT/latency_G_mxfp4_deepgemm_c16_${DATE}.json" \
&& /opt/pytorch/bin/python /tmp/bench_v4_pro.py gsm8k \
    --base-url http://localhost:8089 --model "$MXFP4" \
    --label "I_mxfp4_deepgemm_gsm8k" --limit 300 --concurrency 16 --max-tokens 2048 \
    --output "$OUT/gsm8k_I_mxfp4_deepgemm_300_${DATE}.json"

echo "==== EXTENSION MATRIX DONE ===="
ls -la "$OUT"/{latency_E,latency_F,latency_G,gsm8k_I}_*.json 2>&1

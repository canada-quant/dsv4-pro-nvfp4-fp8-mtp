#!/usr/bin/env bash
# Kill the NVFP4 serve, start the native MXFP4 serve (same config),
# run all benchmarks, save with label `native_mxfp4`.

set -e

echo "==== killing current serve ===="
pkill -9 -f "VLLM::" 2>/dev/null || true
sleep 5

echo "==== starting native MXFP4 serve ===="
nohup bash -c '
export PATH=/opt/pytorch/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda
/opt/pytorch/bin/vllm serve /opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --moe-backend deep_gemm_mega_moe \
  --enforce-eager \
  --port 8089 2>&1
' > /tmp/v4pro_native_serve.log 2>&1 &
echo $! > /tmp/v4pro_native_serve.pid

echo "==== waiting for native serve ready ===="
PID=$(cat /tmp/v4pro_native_serve.pid)
while ps -p $PID > /dev/null 2>&1; do
  if grep -qE "Application startup complete|Uvicorn running on" /tmp/v4pro_native_serve.log 2>/dev/null; then
    echo "SERVE READY"
    break
  fi
  if grep -qE "WorkerProc hit|RuntimeError: Engine|EngineCore failed" /tmp/v4pro_native_serve.log 2>/dev/null; then
    echo "SERVE FAILED:"
    grep -A 3 -E "WorkerProc hit|EngineCore failed|KeyError|AttributeError" /tmp/v4pro_native_serve.log | head -20
    exit 1
  fi
  sleep 20
done

echo "==== running benchmarks ===="
/tmp/run_all_benchmarks.sh /opt/dlami/nvme/weights/v4-pro-native-mxfp4-mtp native_mxfp4 http://localhost:8089

echo "==== done ===="

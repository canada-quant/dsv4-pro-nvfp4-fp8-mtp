#!/usr/bin/env bash
# Build vLLM from source with the 5 local patches needed to serve
# canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP until those patches merge upstream.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
#
# Environment overrides (all optional):
#   VLLM_SRC_DIR       Where to clone vLLM (default: $HOME/src/vllm)
#   VLLM_REF           vLLM ref to base on (default: main)
#   TORCH_CUDA_ARCH    Compute capability for build (default: auto-detect)
#   SKIP_BUILD         If "1", patch only — don't run pip install
#   SKIP_DEPS          If "1", don't install bench/eval deps (langdetect, evalplus, etc.)
#
# Detects compute capability via torch and picks the right arch suffix.
# Refuses to build for the wrong arch on B300 (sm_103a, not sm_100a).

set -euo pipefail

# ---------- config ----------
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$HOME/src/vllm}"
VLLM_REF="${VLLM_REF:-main}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"

echo "==> install_vllm_with_patches.sh — preparing vLLM build for canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP"
echo "    VLLM_SRC_DIR=$VLLM_SRC_DIR"
echo "    VLLM_REF=$VLLM_REF"

# ---------- detect compute capability ----------
if [ -z "${TORCH_CUDA_ARCH:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    DETECTED_CC="$(python3 -c "
import torch
if torch.cuda.is_available():
    c = torch.cuda.get_device_capability(0)
    print(f'{c[0]}.{c[1]}a')
else:
    print('')
" 2>/dev/null || echo '')"
    if [ -n "$DETECTED_CC" ]; then
      TORCH_CUDA_ARCH="$DETECTED_CC"
      echo "    detected GPU compute capability: $TORCH_CUDA_ARCH"
    fi
  fi
fi
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-10.3a}"

case "$TORCH_CUDA_ARCH" in
  10.3a|10.0a|12.0a|9.0a)
    echo "    building for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH"
    ;;
  *)
    echo "    WARNING: unusual TORCH_CUDA_ARCH='$TORCH_CUDA_ARCH'"
    echo "    Common values: 10.3a (B300), 10.0a (B200), 12.0a (sm_120 consumer), 9.0a (H100/H200)"
    ;;
esac

# ---------- prerequisites ----------
echo "==> checking prerequisites"
for cmd in git python3 pip; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not found in PATH"; exit 1; }
done

# CUDA_HOME (needs full toolkit, not just runtime — Tilelang invokes nvcc at runtime)
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
    echo "    setting CUDA_HOME=/usr/local/cuda"
  else
    echo "    WARNING: CUDA_HOME not set and /usr/local/cuda not found"
    echo "    If this is a DLAMI with /opt/pytorch/cuda only (runtime, no headers), install:"
    echo "      sudo apt install cuda-toolkit-13-0"
    echo "    then re-run with: CUDA_HOME=/usr/local/cuda bash $0"
  fi
fi

# ---------- clone / fetch vLLM ----------
if [ ! -d "$VLLM_SRC_DIR/.git" ]; then
  echo "==> cloning vllm-project/vllm to $VLLM_SRC_DIR"
  git clone https://github.com/vllm-project/vllm "$VLLM_SRC_DIR"
fi
cd "$VLLM_SRC_DIR"
git fetch origin --quiet
git checkout "$VLLM_REF" --quiet
git pull --ff-only --quiet || true

# ---------- apply the 4 patches ----------
echo "==> applying 4 local patches (PRs #43248, #43288, #43290, #43319)"

# Patch 1 — bool() wrap on is_static_input_scheme (PR #43248)
# Sites: 5 occurrences in compressed_tensors.py
CT_FILE="vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
if grep -q "is_static_input_scheme = input_quant and not input_quant.dynamic" "$CT_FILE"; then
  echo "    patching #43248: bool() wrap in $CT_FILE (5 sites)"
  sed -i 's|is_static_input_scheme = input_quant and not input_quant.dynamic|is_static_input_scheme = bool(input_quant and not input_quant.dynamic)|g' "$CT_FILE"
else
  echo "    skipping #43248 (already applied or merged)"
fi

# Patch 2 + 3 — scale_fmt defensive .get + getattr wrapping (PR #43288 + BF16 follow-up)
M_FILE="vllm/models/deepseek_v4/nvidia/model.py"
if grep -q 'self.scale_fmt = config.quantization_config\["scale_fmt"\]' "$M_FILE"; then
  echo "    patching #43288 (original .get): $M_FILE"
  sed -i 's|self.scale_fmt = config.quantization_config\["scale_fmt"\]|self.scale_fmt = config.quantization_config.get("scale_fmt", "ue8m0")|g' "$M_FILE"
fi
if grep -q 'self.scale_fmt = config.quantization_config.get("scale_fmt", "ue8m0")' "$M_FILE"; then
  echo "    patching #43288 BF16 follow-up: getattr wrap in $M_FILE"
  python3 - <<PYEOF
p = "$M_FILE"
src = open(p).read()
old = '        self.scale_fmt = config.quantization_config.get("scale_fmt", "ue8m0")'
new = '        _qc = getattr(config, "quantization_config", None) or {}\n        self.scale_fmt = _qc.get("scale_fmt", "ue8m0")'
if old in src:
    src = src.replace(old, new)
    open(p, "w").write(src)
    print("      wrote BF16 getattr wrap")
else:
    print("      no exact match for BF16 wrap (already applied or moved)")
PYEOF
else
  echo "    skipping #43288 (already applied or merged)"
fi

# Patch 4 — weight_scale_inv-or-weight_scale fallback (PR #43290)
A_FILE="vllm/models/deepseek_v4/attention.py"
if grep -q 'weight_scale_inv = self.wo_a.weight_scale_inv' "$A_FILE"; then
  echo "    patching #43290: weight_scale_inv fallback in $A_FILE"
  sed -i 's|weight_scale_inv = self.wo_a.weight_scale_inv|weight_scale_inv = getattr(self.wo_a, "weight_scale_inv", None) or self.wo_a.weight_scale|g' "$A_FILE"
else
  echo "    skipping #43290 (already applied or merged)"
fi

# Patch 5 — MTP-quant-detect (PR #43319)
# More complex than sed — cherry-pick from canada-quant fork if not yet on main
if ! grep -q "_mtp_block_is_quantized_on_disk" vllm/models/deepseek_v4/nvidia/mtp.py 2>/dev/null; then
  echo "    patching #43319: cherry-picking from canada-quant:fix/dsv4-mtp-draft-quant-detect"
  git remote add canada-quant https://github.com/canada-quant/vllm.git 2>/dev/null || true
  git fetch canada-quant fix/dsv4-mtp-draft-quant-detect --quiet
  if ! git cherry-pick --no-commit FETCH_HEAD; then
    echo "    cherry-pick had conflicts — falling back to PR #43319 diff manually"
    git cherry-pick --abort 2>/dev/null || true
    echo "    NOTE: PR #43319 must be applied manually until cleanly cherry-pickable"
    echo "    See docs/VLLM_SETUP_ISSUES.md in this repo for the full diff."
  fi
else
  echo "    skipping #43319 (already applied or merged)"
fi

# ---------- build ----------
if [ "$SKIP_BUILD" = "1" ]; then
  echo "==> SKIP_BUILD=1, leaving build to caller"
else
  echo "==> building vLLM (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH, this takes ~10-20 min on a fast machine)"
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH" pip install -e . --no-build-isolation
fi

# ---------- optional dependencies ----------
if [ "$SKIP_DEPS" != "1" ]; then
  echo "==> installing bench/eval dependencies (set SKIP_DEPS=1 to skip)"
  pip install --quiet langdetect immutabledict nltk evalplus openai datasets || true
fi

# ---------- summary ----------
cat <<EOM

==> Done.

Quick smoke:

    CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \\
      vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \\
      --tensor-parallel-size 4 \\
      --kv-cache-dtype fp8 \\
      --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

For full instructions, see:
    https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/QUICKSTART.md

For the rationale on each patch:
    https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/VLLM_SETUP_ISSUES.md

EOM

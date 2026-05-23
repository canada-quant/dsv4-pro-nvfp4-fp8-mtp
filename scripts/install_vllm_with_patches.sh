#!/usr/bin/env bash
# Build vLLM from source pinned at an upstream commit + apply the 4 open
# patches needed to serve canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP until
# they merge upstream.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
#
# Environment overrides (all optional):
#   VLLM_SRC_DIR       Where to clone vLLM (default: $HOME/src/vllm)
#   VLLM_REF           vLLM ref to base on (default: pinned SHA below)
#   PATCHES_REPO_RAW   Raw URL prefix for the 4 .diff files (default: this repo's main)
#   TORCH_CUDA_ARCH    Compute capability for build (default: auto-detect)
#   SKIP_BUILD         If "1", patch only — don't run pip install
#   SKIP_DEPS          If "1", don't install bench/eval deps (langdetect, evalplus, etc.)
#
# Detects compute capability via torch and picks the right arch suffix.
# Refuses to build for the wrong arch on B300 (sm_103a, not sm_100a).

set -euo pipefail

# ---------- pinned upstream commit ----------
# This commit is the vLLM main HEAD verified on 2026-05-22 to serve native
# DeepSeek-V4-Pro (MXFP4 experts + FP8 block attention + MTP) with the 4
# applied patches below. The V4-Pro subpackage at vllm/models/deepseek_v4/
# is present at this commit. Update the SHA when bumping to a newer base.
VLLM_PINNED_SHA="${VLLM_PINNED_SHA:-39910f2b25}"
PATCHES_REPO_RAW="${PATCHES_REPO_RAW:-https://raw.githubusercontent.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/main/patches}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$HOME/src/vllm}"
VLLM_REF="${VLLM_REF:-$VLLM_PINNED_SHA}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"

echo "==> install_vllm_with_patches.sh — preparing vLLM build for"
echo "    canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP"
echo "    VLLM_SRC_DIR=$VLLM_SRC_DIR"
echo "    VLLM_REF=$VLLM_REF  (pinned upstream: $VLLM_PINNED_SHA)"
echo "    PATCHES_REPO_RAW=$PATCHES_REPO_RAW"

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
for cmd in git python3 pip curl; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not found in PATH"; exit 1; }
done

# CUDA_HOME — needs full toolkit, not just runtime (Tilelang invokes nvcc at runtime).
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

# setuptools-rust + Rust toolchain — required since vLLM PR #43283 introduced
# the Rust frontend. Without these, pyproject.toml metadata generation fails
# with "ModuleNotFoundError: No module named 'setuptools_rust'".
echo "==> installing setuptools-rust + Rust toolchain if absent"
if ! python3 -c "import setuptools_rust" >/dev/null 2>&1; then
  echo "    installing setuptools-rust"
  pip install --quiet "setuptools-rust>=1.9.0"
fi
if ! command -v cargo >/dev/null 2>&1; then
  if [ -d "$HOME/.cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "    installing rustup (user install, no sudo)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
fi
cargo --version
rustc --version

# ---------- clone / fetch vLLM ----------
if [ ! -d "$VLLM_SRC_DIR/.git" ]; then
  echo "==> cloning vllm-project/vllm to $VLLM_SRC_DIR"
  git clone https://github.com/vllm-project/vllm "$VLLM_SRC_DIR"
fi
cd "$VLLM_SRC_DIR"

# Make sure upstream remote points at vllm-project (in case this repo was set
# up with a different "origin").
if ! git remote get-url upstream >/dev/null 2>&1; then
  git remote add upstream https://github.com/vllm-project/vllm.git
fi
git fetch upstream --quiet

# Check out the pinned base. If $VLLM_REF is a SHA, this lands at a detached
# HEAD; we make a clean branch from it so commits land somewhere named.
git reset --hard HEAD 2>/dev/null || true
git checkout "$VLLM_REF" --quiet
HEAD_SHA="$(git rev-parse --short HEAD)"
echo "==> base commit: $HEAD_SHA ($(git log -1 --format='%s'))"

BRANCH_NAME="canada-quant-v4pro-nvfp4"
if git rev-parse --verify "$BRANCH_NAME" >/dev/null 2>&1; then
  git checkout "$BRANCH_NAME" --quiet
  git reset --hard "$HEAD_SHA" --quiet
else
  git checkout -b "$BRANCH_NAME" --quiet
fi

# ---------- clean stale cmake caches ----------
# .deps from a prior build at a different generator (Ninja vs Make) causes
# cmake to refuse with "Does not match the generator used previously".
echo "==> cleaning stale cmake caches"
rm -rf .deps build
find . -name "CMakeCache.txt" -delete 2>/dev/null
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null

# ---------- apply the 4 patches ----------
echo "==> applying 4 patches (PRs #43248, #43288, #43290, #43319)"

PATCH_TMPDIR="$(mktemp -d)"
trap "rm -rf $PATCH_TMPDIR" EXIT

apply_patch () {
  local name="$1"
  local pr="$2"
  local url="$PATCHES_REPO_RAW/$name"
  local msg="$3"
  local local_path="$PATCH_TMPDIR/$name"
  echo "--- $name (PR #$pr) ---"
  if ! curl -sSL "$url" -o "$local_path"; then
    echo "    ERROR: failed to fetch $url"
    exit 1
  fi
  if ! git apply --check "$local_path" 2>/dev/null; then
    # Already applied (line context matches the patched state, not the original)
    if git apply --check --reverse "$local_path" 2>/dev/null; then
      echo "    already applied — skipping"
      return 0
    fi
    echo "    WARNING: patch does not apply cleanly; attempting 3-way merge"
    if ! git apply --3way "$local_path"; then
      echo "    ERROR: $name failed to apply. Inspect $local_path."
      exit 1
    fi
  else
    git apply "$local_path"
  fi
  git add -A
  git -c user.email="quant@canada-quant.io" -c user.name="canada-quant" \
    commit --quiet -m "$msg"
}

apply_patch "patch_43248_ct_bool_wrap.diff" 43248 \
  "Apply PR #43248: bool() wrap on is_static_input_scheme

Wraps the truthy expression in bool() at 2 sites in
compressed_tensors.py so input_quant=None (which is legal for some
schemes) returns False instead of None.
Pending upstream PR: https://github.com/vllm-project/vllm/pull/43248"

apply_patch "patch_43288_scale_fmt_get.diff" 43288 \
  "Apply PR #43288: scale_fmt defensive read + BF16 getattr wrap

deepseek_v4/nvidia/model.py:909 was hard-subscripting
config.quantization_config[\"scale_fmt\"]; crashes if (a) the config
lacks scale_fmt or (b) quantization_config is absent (BF16 serving).
Pending upstream PR: https://github.com/vllm-project/vllm/pull/43288"

apply_patch "patch_43290_weight_scale_fallback.diff" 43290 \
  "Apply PR #43290: weight_scale_inv-or-weight_scale fallback

deepseek_v4/attention.py wo_a access raised AttributeError when
weights store weight_scale (the FP8 block convention used by V4-Pro
attention). Load-blocking for V4-Pro without this patch.
Pending upstream PR: https://github.com/vllm-project/vllm/pull/43290"

apply_patch "patch_43319_mtp_quant_detect.diff" 43319 \
  "Apply PR #43319: MTP quant detection from safetensors header

deepseek_v4/nvidia/mtp.py defaulted to assuming MTP is BF16 and
skipped quant_config wiring. V4-Pro MTP IS quantized on disk (MXFP4
experts); MTP loader inspects safetensors header for
.experts.*.w[123].scale and routes through quant_config when present.
Required for V4-Pro MTP serving.
Pending upstream PR: https://github.com/vllm-project/vllm/pull/43319"

# ---------- cherry-pick PR #42209 (sychen52, NVIDIA): NVFP4 MOE for V4 ----------
# This is the NVFP4 V4-Pro serving path (Phase 5 from this repo's PLAN.md).
# Adds ModelOptNvFp4FusedMoE routing to DeepseekV4FP8Config when
# quantization_config.moe_quant_algo == "NVFP4", plus the
# trtllm_nvfp4_moe expert kernel.
# PR #42209 merged to upstream main 2026-05-22T14:21:51Z. Our pinned
# SHA 39910f2b25 is dated 2026-05-22T00:21:48Z — pre-merge — so the
# cherry-pick is still active at the pin. If you bump VLLM_REF to a
# SHA after the merge, git merge-base --is-ancestor below will detect
# the commits are already in HEAD and skip them (no-op).
echo "==> cherry-picking PR #42209 (NVFP4 MOE for V4-Pro)"
git remote add sychen52 https://github.com/sychen52/vllm.git 2>/dev/null || true
git fetch sychen52 nvfp4_dsv4 --quiet
for sha in e2e6d12b7b dc950c7342 af11a8480c; do
  if git merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
    echo "    $sha already in branch — skipping"
    continue
  fi
  echo "    cherry-picking $sha"
  if ! git -c user.email="quant@canada-quant.io" -c user.name="canada-quant" \
       cherry-pick "$sha"; then
    echo "    ERROR: $sha conflicts; manual resolution needed"
    git cherry-pick --abort 2>/dev/null || true
    exit 1
  fi
done

echo "==> final branch state"
git log --oneline -7

# ---------- build ----------
if [ "$SKIP_BUILD" = "1" ]; then
  echo "==> SKIP_BUILD=1, leaving build to caller"
else
  echo "==> building vLLM (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH, takes ~10-20 min)"
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH" pip install -e . --no-build-isolation
fi

# ---------- optional dependencies ----------
if [ "$SKIP_DEPS" != "1" ]; then
  echo "==> installing bench/eval dependencies (set SKIP_DEPS=1 to skip)"
  pip install --quiet langdetect immutabledict nltk evalplus openai datasets || true
fi

# ---------- import smoke ----------
echo "==> import smoke test"
python3 -c "
from vllm.models.deepseek_v4 import quant_config, compressor, attention
from vllm.model_executor.layers import mhc
print('vllm.models.deepseek_v4 + mhc import: ok')
import vllm
print('vllm version:', vllm.__version__)
"

# ---------- summary ----------
cat <<EOM

==> Done.

Quick smoke (native V4-Pro MXFP4-FP8-MTP):

    CUDA_HOME=/usr/local/cuda \\
      vllm serve /path/to/DeepSeek-V4-Pro \\
      --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \\
      --enable-expert-parallel --tensor-parallel-size 8 \\
      --moe-backend deep_gemm_mega_moe \\
      --speculative-config '{"method":"mtp","num_speculative_tokens":1}'

(Recipe page: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)

For full instructions:
    https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/QUICKSTART.md

For the rationale on each patch:
    https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp/blob/main/docs/VLLM_SETUP_ISSUES.md

EOM

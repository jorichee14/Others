#!/usr/bin/env bash
# Build ONE virtualenv that runs the whole pipeline: merge, dynamic removal,
# colorize, YOLO detection and meshing.
#
#     ./setup_env.sh [env_dir]          # default ~/lidar-env
#
# Why a venv rather than pip install --user:
#   the previous breakage came from layering pip wheels into ~/.local on top of
#   apt's python3-numpy / python3-scipy. pip upgraded numpy to 2.x for torch,
#   while apt's scipy and the Open3D wheel are compiled against the numpy 1.x
#   ABI, so both stopped importing. A venv WITHOUT --system-site-packages owns
#   its entire stack, so no apt package can be half-upgraded underneath it.
#
# CUDA wheels are chosen from the driver's reported CUDA version, not from
# nvcc: nvcc is the toolkit you compile with, the driver is what actually has
# to run the kernels, and on this class of machine they routinely differ.
set -euo pipefail

ENV_DIR="${1:-$HOME/lidar-env}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== detecting CUDA ==="
CUDA_MAJOR=""
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    # "CUDA Version: 12.4" from the nvidia-smi banner = the highest CUDA
    # runtime this driver can run, which is the number that matters for wheels
    CUDA_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1 || echo "")
    CUDA_MAJOR="${CUDA_VER%%.*}"
    echo "driver ${DRIVER:-unknown}, supports CUDA up to ${CUDA_VER:-unknown}"
else
    echo "nvidia-smi not found -> CPU-only install"
fi

# CuPy JIT-compiles every kernel, so it needs NVRTC at RUNTIME -- the cupy
# wheel does not bundle it. Without these the GPU reports healthy and then
# dies on the first kernel compile with "libnvrtc.so.NN: cannot open shared
# object file". torch's wheels usually drag most of them in, but not
# dependably, so ask for them explicitly.
case "$CUDA_MAJOR" in
    12|13) CUPY_PKG="cupy-cuda12x"; TORCH_IDX="https://download.pytorch.org/whl/cu121"
           CUDA_LIBS="nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-nvjitlink-cu12" ;;
    11)    CUPY_PKG="cupy-cuda11x"; TORCH_IDX="https://download.pytorch.org/whl/cu118"
           CUDA_LIBS="nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cublas-cu11" ;;
    *)     CUPY_PKG="";             TORCH_IDX="https://download.pytorch.org/whl/cpu"
           CUDA_LIBS="" ;;
esac
[ -n "$CUPY_PKG" ] && echo "selected $CUPY_PKG and torch from ${TORCH_IDX##*/}" \
                   || echo "no CUDA detected: installing CPU torch, skipping cupy"

echo
echo "=== creating $ENV_DIR ==="
# deliberately NOT --system-site-packages: that would let apt's numpy/scipy
# leak in and reintroduce the exact ABI conflict this is meant to prevent
python3 -m venv "$ENV_DIR"
PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"
"$PIP" install -q -U pip wheel setuptools

echo
echo "=== core stack (numpy pinned by constraints.txt) ==="
"$PIP" install -r "$HERE/requirements.txt" -c "$HERE/constraints.txt"

echo
echo "=== torch ==="
# constraints still apply, so pip picks a torch build that runs on numpy 1.x
"$PIP" install torch torchvision --index-url "$TORCH_IDX" -c "$HERE/constraints.txt"

echo
echo "=== ultralytics (--no-deps: its deps are already pinned above) ==="
"$PIP" install ultralytics --no-deps

if [ -n "$CUPY_PKG" ]; then
    echo
    echo "=== cupy + CUDA runtime libraries ==="
    "$PIP" install "$CUPY_PKG" -c "$HERE/constraints.txt"
    # shellcheck disable=SC2086
    "$PIP" install $CUDA_LIBS -c "$HERE/constraints.txt" || \
        echo "WARNING: could not install $CUDA_LIBS -- if the GPU fails on a "\
             "kernel compile, install nvidia-cuda-nvrtc-cu* manually"
fi

echo
echo "=== verifying ==="
"$PY" "$HERE/verify_env.py"

cat <<EOF

Environment ready. Use it for every stage:

    $PY 01_build_map.py pipeline_config.json
    $PY 02_pcd_to_mesh_sionna_v9.py out/static.pcd mesh.ply

or activate it once per shell:

    source $ENV_DIR/bin/activate

Do NOT pip install into this env without -c constraints.txt -- that is how the
numpy ceiling gets lifted and Open3D stops importing again.
EOF

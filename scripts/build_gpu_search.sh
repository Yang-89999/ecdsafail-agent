#!/usr/bin/env bash
# Build tools/gpu_search/gpu_search: with nvcc for this machine's GPU, or (--cpu, or no nvcc) with g++ for the CPU
# path only (the CPU path is what the GPU is checked against, and what the cloud tests compare with nonce_search).
#
#   scripts/build_gpu_search.sh [--cpu]        env GPU_ARCH=sm_120 overrides the detected architecture
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd -P)"
src="$here/tools/gpu_search/gpu_search.cu"
out="$here/tools/gpu_search/gpu_search"
if [[ "${1:-}" != "--cpu" ]]; then
  nvcc="$(command -v nvcc || true)"
  if [[ -z "$nvcc" && -x /usr/local/cuda/bin/nvcc ]]; then nvcc=/usr/local/cuda/bin/nvcc; fi
  if [[ -n "$nvcc" ]]; then
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)"
    arch="${GPU_ARCH:-}"
    if [[ -z "$arch" && -n "$cap" ]]; then arch="sm_$cap"; fi
    arch="${arch:-native}"
    ver="$("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    echo "nvcc $ver, target $arch"
    if ! "$nvcc" -O3 -std=c++17 -arch="$arch" -Xcompiler -fopenmp -Xptxas -v -o "$out" "$src" -lgomp 2> "$out.build.log"; then
      cat "$out.build.log" >&2
      echo "!! nvcc build failed (an RTX 5090, sm_120, needs CUDA 12.8 or newer; this machine has $ver)" >&2
      exit 1
    fi
    grep -A2 "_Z5k_sim" "$out.build.log" | grep -E "registers|spill" || true
    echo "built $out (GPU)"
    exit 0
  fi
  echo "no nvcc found: building the CPU path only" >&2
fi
g++ -O3 -march=native -fopenmp -std=c++17 -x c++ -o "$out" "$src"
echo "built $out (CPU)"

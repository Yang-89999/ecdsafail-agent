#!/usr/bin/env bash
# GPU benchmark for the nonce search, for a freshly rented machine (e.g. one RTX 5090 for an hour). It builds the
# public head's op stream (commit 5ba2d2d, whose official TAIL_NONCE 12009835509 is known to be clean), checks the
# GPU tester three ways, then measures the search rate at several lane counts. Nothing here is ours or secret: it
# uses only the public challenge. Output also goes to ~/gpu-bench/bench.log.
#
#   bash scripts/gpu_bench.sh [seconds per lane setting, default 120]
#
# Needs an NVIDIA GPU with CUDA 12.8+ (nvcc), git, curl, python3 and a C compiler; installs Rust if missing.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd -P)"
secs="${1:-120}"
work="${GPU_WORK:-$HOME/gpu-bench}"
head_commit="5ba2d2d"
head_full="5ba2d2d4e6fcee9f2f5786d02f26414dafaa53e3"
head_nonce="12009835509"
mkdir -p "$work"
exec > >(tee -a "$work/bench.log") 2>&1
echo "== $(date -u +%FT%TZ) GPU nonce-search benchmark =="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || { echo "!! no NVIDIA GPU visible"; exit 1; }
echo "CPU cores: $(nproc)"

missing=()
for c in git curl python3 cc; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
if ((${#missing[@]})); then
  echo "installing: ${missing[*]}"
  if command -v apt-get >/dev/null 2>&1; then
    sudo_=""; [[ "$(id -u)" -ne 0 ]] && sudo_="sudo"
    $sudo_ apt-get update -qq && $sudo_ apt-get install -y -qq git curl python3 build-essential >/dev/null
  else
    echo "!! please install: ${missing[*]}"; exit 1
  fi
fi
if ! command -v cargo >/dev/null 2>&1; then
  if [[ ! -f "$HOME/.cargo/env" ]]; then curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain none; fi
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

echo "-- the public challenge at $head_commit --"
ch="$work/ecdsafail-challenge"
# only this one commit (25 MB): a full clone carries thousands of submission branches
if ! git -C "$ch" cat-file -e "$head_full^{commit}" 2>/dev/null; then
  rm -rf "$ch"; mkdir -p "$ch"
  git -C "$ch" init -q
  git -C "$ch" remote add origin https://github.com/ecdsafail/ecdsafail-challenge.git
  git -C "$ch" fetch -q --depth 1 origin "$head_full"
fi
git -C "$ch" checkout -q "$head_full"
chan="$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$ch/rust-toolchain")"
rustup toolchain install "$chan" --profile minimal >/dev/null
(cd "$ch" && RUSTUP_TOOLCHAIN="$chan" cargo fetch --locked -q)
if [[ ! -s "$work/head/ops.bin" ]]; then "$here/scripts/build_ops.sh" "$ch" "$work/head"; fi
echo "-- nonce_search (CPU reference) and the job file --"
"$here/scripts/build_nonce_search.sh" "$ch"
(cd "$work/head" && "$here/tools/nonce_search/target/release/nonce_search" export "$work/head.job")
echo "-- gpu_search --"
"$here/scripts/build_gpu_search.sh"
gs="$here/tools/gpu_search/gpu_search"

echo "-- check 1: the head's official nonce must be clean on the GPU --"
"$gs" "$work/head.job" check "$head_nonce" | tee "$work/check.txt"
grep -q "clean=true" "$work/check.txt" || { echo "!! the official nonce did not come out clean: the GPU path is wrong; stop here"; exit 1; }
echo "-- check 2: GPU against this program's CPU path (4 CRN draws, 64 early-stop nonces) --"
"$gs" "$work/head.job" selftest 4 64 > "$work/selftest_full.txt" || true
grep -v "^nonce \|^crn \|^{" "$work/selftest_full.txt" | tee "$work/selftest.txt"
grep -q "SELFTEST OK" "$work/selftest.txt" || { echo "!! selftest failed; stop here"; exit 1; }
echo "-- check 3: the GPU's 64 nonces against the Rust reference (nonce_search compare) --"
(cd "$work/head" && "$here/tools/nonce_search/target/release/nonce_search" compare 1000 64 > "$work/rust_compare.txt" 2>&1) || true
grep "passed_batches" "$work/rust_compare.txt" | sort > "$work/rust_nonces.txt" || true
grep "^nonce " "$work/selftest_full.txt" | sort > "$work/gpu_nonces.txt" || true
if [[ -s "$work/rust_nonces.txt" ]] && cmp -s "$work/rust_nonces.txt" "$work/gpu_nonces.txt"; then
  echo "Rust reference and GPU agree on all 64 nonces"
else
  echo "!! Rust reference and GPU differ (see $work/rust_nonces.txt vs $work/gpu_nonces.txt)"
fi

echo "-- benchmark: $secs s per lane setting --"
"$gs" "$work/head.job" bench "$secs" | tee "$work/bench.txt"
echo "== done $(date -u +%FT%TZ); paste everything from the line 'GPU nonce-search benchmark' into the thread =="

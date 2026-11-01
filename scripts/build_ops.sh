#!/usr/bin/env bash
# Build build_circuit from a worktree and run it in a scratch directory (bubblewrap-confined when available, like
# benchmark.sh), leaving the op stream at <outdir>/ops.bin. Shared by win_score.sh.
#
#   scripts/build_ops.sh <worktree> <outdir>
set -euo pipefail
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true
wt="$(cd "${1:?worktree}" && pwd -P)"
out="${2:?outdir}"
mkdir -p "$out"
out="$(cd "$out" && pwd -P)"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$wt/rust-toolchain")}"
export CARGO_NET_OFFLINE=true
cd "$wt"
echo "-- build (worktree $wt) --"
set +e
cargo build --release --locked --offline --bin build_circuit > "$out/cargo.log" 2>&1
rc=$?
set -e
grep -E "^(error|warning: unused)|Finished|Compiling quantum_ecc" "$out/cargo.log" | tail -5 || true
if [[ $rc -ne 0 ]]; then
  grep -E "^error" -A8 "$out/cargo.log" | head -40 || true
  echo "!! cargo build failed (exit $rc)"
  exit 1
fi

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
chmod 1777 "$scratch"
cp target/release/build_circuit "$scratch/build_circuit"
chmod 0555 "$scratch/build_circuit"
echo "-- build_circuit --"
t0=$(date +%s)
set +e
if command -v bwrap >/dev/null 2>&1; then
  bw=( bwrap )
  if [[ "$(id -u)" -ne 0 ]] && command -v setpriv >/dev/null 2>&1; then bw=( setpriv --no-new-privs bwrap ); fi
  "${bw[@]}" --ro-bind / / --dev /dev --ro-bind /proc /proc \
    --bind "$scratch" "$scratch" --chdir "$scratch" --setenv TMPDIR "$scratch" \
    --unshare-user --unshare-net --unshare-ipc --unshare-uts --unshare-cgroup \
    --cap-drop ALL --new-session --die-with-parent --uid 65534 --gid 65534 \
    -- "$scratch/build_circuit" > "$scratch/build.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 && ! -s "$scratch/ops.bin" ]] && grep -q "bwrap:" "$scratch/build.log"; then
    # no user namespaces here (some containers): run unconfined in the scratch dir, as fast_score.sh does without bwrap
    ( cd "$scratch" && TMPDIR="$scratch" ./build_circuit > "$scratch/build.log" 2>&1 )
    rc=$?
  fi
else
  ( cd "$scratch" && TMPDIR="$scratch" ./build_circuit > "$scratch/build.log" 2>&1 )
  rc=$?
fi
set -e
tail -3 "$scratch/build.log"
echo "build_circuit: exit $rc, $(( $(date +%s) - t0 ))s"
if [[ $rc -ne 0 || ! -s "$scratch/ops.bin" ]]; then
  echo "!! build_circuit failed (exit $rc) or produced no ops.bin"
  exit 1
fi
mv "$scratch/ops.bin" "$out/ops.bin"

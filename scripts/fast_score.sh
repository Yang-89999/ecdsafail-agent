#!/usr/bin/env bash
# Cheap score for one worktree: build build_circuit from the worktree's
# src/point_add, run it in a scratch dir (bubblewrap-confined when available,
# like benchmark.sh), then evaluate the op stream with EVAL_SHOTS shots using
# tools/fast_eval. Prints the evaluator's report; the last line is JSON.
#
#   scripts/fast_score.sh <worktree> [shots=64] [challenge checkout]
#
# This is an approximation for ranking candidates. The claim is always the
# official benchmark.sh run.
set -euo pipefail
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true
wt="$(cd "${1:?worktree}" && pwd -P)"
shots="${2:-64}"
here="$(cd "$(dirname "$0")/.." && pwd -P)"
challenge="${3:-$(dirname "$wt")/../ecdsafail-challenge}"
fast_eval="$here/tools/fast_eval/target/release/fast_eval"
[[ -x "$fast_eval" ]] || bash "$here/scripts/build_fast_eval.sh" "$challenge"

export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$wt/rust-toolchain")}"
export CARGO_NET_OFFLINE=true
cd "$wt"
echo "-- build (worktree $wt) --"
cargo build --release --locked --offline --bin build_circuit 2>&1 | grep -E "^(error|warning: unused)|Finished|Compiling quantum_ecc" | tail -5

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
chmod 1777 "$scratch"
cp target/release/build_circuit "$scratch/build_circuit"
chmod 0555 "$scratch/build_circuit"

echo "-- build_circuit --"
t0=$(date +%s)
if command -v bwrap >/dev/null 2>&1; then
  bw=( bwrap )
  if [[ "$(id -u)" -ne 0 ]] && command -v setpriv >/dev/null 2>&1; then bw=( setpriv --no-new-privs bwrap ); fi
  set +e
  "${bw[@]}" --ro-bind / / --dev /dev --ro-bind /proc /proc \
    --bind "$scratch" "$scratch" --chdir "$scratch" --setenv TMPDIR "$scratch" \
    --unshare-user --unshare-net --unshare-ipc --unshare-uts --unshare-cgroup \
    --cap-drop ALL --new-session --die-with-parent --uid 65534 --gid 65534 \
    -- "$scratch/build_circuit" > "$scratch/build.log" 2>&1
  rc=$?
  set -e
else
  set +e
  ( cd "$scratch" && TMPDIR="$scratch" ./build_circuit > "$scratch/build.log" 2>&1 )
  rc=$?
  set -e
fi
tail -3 "$scratch/build.log"
echo "build_circuit: exit $rc, $(( $(date +%s) - t0 ))s"
if [[ $rc -ne 0 || ! -s "$scratch/ops.bin" ]]; then
  echo "!! build_circuit failed (exit $rc) or produced no ops.bin"
  echo '{"ok": false, "reason": "build_circuit failed"}'
  exit 1
fi

echo "-- fast_eval ($shots shots) --"
t1=$(date +%s)
set +e
( cd "$scratch" && EVAL_SHOTS="$shots" "$fast_eval" --note "fast" )
rc=$?
set -e
echo "fast_eval: exit $rc, $(( $(date +%s) - t1 ))s"
if [[ $rc -ne 0 || ! -s "$scratch/score.json" ]]; then
  echo '{"ok": false, "reason": "fast_eval rejected the circuit (see report above)"}'
  exit 1
fi
python3 -c "import json,sys; d=json.load(open('$scratch/score.json')); d['ok']=True; d['shots']=$shots; print(json.dumps(d))"

#!/usr/bin/env bash
# Fresh-inputs check for one worktree: build the circuit once, then test the SAME op stream on DRAWS independent
# sets of SHOTS inputs (fast_eval with EVAL_SALT=1..DRAWS mixes a salt into the Fiat-Shamir hash), plus the
# official draw (no salt). The official harness tests only one hash-derived draw; a circuit that is wrong on a
# small share of inputs, or whose average Toffoli count depends on the inputs, can look better on that one draw
# than it is. This check shows the circuit's real failure rate and its average Toffoli over many draws.
#
#   scripts/fresh_check.sh <worktree> [draws=8] [shots=9024] [challenge checkout]
#
# "Correct" here means the standard of the published reference circuits (Babbush et al. 2026, Schrottenloher
# 2026, both approximate): an error rate of at most 2^-13.3 (about 1e-4) on independent inputs. MAX_ERROR_RATE
# overrides it. Prints one line per draw; the last line is JSON:
#   {"ok": <error_rate <= max>, "error_rate": failing/tested, "error_rate_ucb95": ..., "failing_shots": wrong or
#    phase-flagged shots over all fresh draws, "failed_shots": wrong results only, "draws": N, "passed": draws with
#    no failure, "tested_shots": N*shots, "mean_toffoli": ..., "sd_toffoli": ..., "qubits": ..., "mean_score": ...,
#    "official": {"ok": ..., "toffoli": ..., "score": ...}}
# The same JSON is appended to <worktree>/.agentlab/fresh.jsonl.
set -euo pipefail
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true
wt="$(cd "${1:?worktree}" && pwd -P)"
draws="${2:-8}"
shots="${3:-9024}"
here="$(cd "$(dirname "$0")/.." && pwd -P)"
challenge="${4:-$(dirname "$wt")/../ecdsafail-challenge}"
fast_eval="$here/tools/fast_eval/target/release/fast_eval"
stamp="$here/tools/fast_eval/.agentlab-version"
if [[ ! -x "$fast_eval" || "$(cat "$stamp" 2>/dev/null)" != "4" ]]; then
  bash "$here/scripts/build_fast_eval.sh" "$challenge" >&2
fi

export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$wt/rust-toolchain")}"
export CARGO_NET_OFFLINE=true
cd "$wt"
echo "-- build (worktree $wt) --" >&2
cargo build --release --locked --offline --bin build_circuit 2>&1 | grep -E "^error|Finished" | tail -3 >&2

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
chmod 1777 "$scratch"
cp target/release/build_circuit "$scratch/build_circuit"
chmod 0555 "$scratch/build_circuit"
set +e
if command -v bwrap >/dev/null 2>&1; then
  bw=( bwrap )
  if [[ "$(id -u)" -ne 0 ]] && command -v setpriv >/dev/null 2>&1; then bw=( setpriv --no-new-privs bwrap ); fi
  "${bw[@]}" --ro-bind / / --dev /dev --ro-bind /proc /proc \
    --bind "$scratch" "$scratch" --chdir "$scratch" --setenv TMPDIR "$scratch" \
    --unshare-user --unshare-net --unshare-ipc --unshare-uts --unshare-cgroup \
    --cap-drop ALL --new-session --die-with-parent --uid 65534 --gid 65534 \
    -- "$scratch/build_circuit" > "$scratch/build.log" 2>&1
else
  ( cd "$scratch" && TMPDIR="$scratch" ./build_circuit > "$scratch/build.log" 2>&1 )
fi
rc=$?
set -e
if [[ $rc -ne 0 || ! -s "$scratch/ops.bin" ]]; then
  tail -3 "$scratch/build.log" >&2
  out='{"ok": false, "reason": "build_circuit failed"}'
  mkdir -p "$wt/.agentlab"; echo "$out" >> "$wt/.agentlab/fresh.jsonl"; echo "$out"; exit 1
fi

for salt in official $(seq 1 "$draws"); do
  rm -f "$scratch/draw.json"
  if [[ $salt == official ]]; then
    ( cd "$scratch" && EVAL_SHOTS="$shots" "$fast_eval" --note fresh >/dev/null 2>&1 ) || true
  else
    ( cd "$scratch" && EVAL_SALT="$salt" EVAL_SHOTS="$shots" "$fast_eval" --note fresh >/dev/null 2>&1 ) || true
  fi
  if [[ -s "$scratch/draw.json" ]]; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d['draw']=sys.argv[2]; print(json.dumps(d))" \
      "$scratch/draw.json" "$salt" | tee -a "$scratch/draws.jsonl" >&2
  else
    echo "{\"draw\": \"$salt\", \"ok\": false, \"error\": \"evaluator produced no draw.json\"}" | tee -a "$scratch/draws.jsonl" >&2
  fi
done

out="$(MAX_ERROR_RATE="${MAX_ERROR_RATE:-0.0001}" python3 - "$scratch/draws.jsonl" <<'PY'
import json, math, os, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
off = next((r for r in rows if r.get("draw") == "official"), {})
fresh = [r for r in rows if r.get("draw") != "official"]
tofs = [r["avg_tof"] for r in fresh if "avg_tof" in r]
q = next((r["qubits"] for r in rows if "qubits" in r), None)
mean = sum(tofs) / len(tofs) if tofs else None
sd = math.sqrt(sum((t - mean) ** 2 for t in tofs) / (len(tofs) - 1)) if len(tofs) > 1 else None
passed = sum(1 for r in fresh if r.get("ok"))
tested = sum(r.get("shots", 0) for r in fresh)
failing = sum(r.get("failing_shots", r.get("classical_failures", 0)) for r in fresh)
rate = failing / tested if tested else None
# one-sided 95 % upper bound (Wilson score) on the per-input error rate
ucb = None
if tested:
    z = 1.645; n = tested; ph = failing / n
    ucb = (ph + z * z / (2 * n) + z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)
mx = float(os.environ.get("MAX_ERROR_RATE", "0.0001"))
res = {
    "ok": bool(fresh) and rate is not None and rate <= mx and not any(r.get("ancilla_garbage_batches") for r in fresh),
    "error_rate": rate, "error_rate_ucb95": ucb, "max_error_rate": mx, "failing_shots": failing,
    "draws": len(fresh), "passed": passed,
    "failed_shots": sum(r.get("classical_failures", 0) for r in fresh),
    "garbage_batches": sum(r.get("phase_garbage_batches", 0) + r.get("ancilla_garbage_batches", 0) for r in fresh),
    "tested_shots": sum(r.get("shots", 0) for r in fresh),
    "mean_toffoli": round(mean, 3) if mean is not None else None,
    "sd_toffoli": round(sd, 3) if sd is not None else None,
    "qubits": q,
    "mean_score": round(mean) * q if mean is not None and q else None,
    "official": {"ok": off.get("ok"), "toffoli": off.get("avg_tof"), "failed_shots": off.get("classical_failures"),
                 "score": round(off["avg_tof"]) * q if off.get("avg_tof") is not None and q else None},
}
print(json.dumps(res))
PY
)"
mkdir -p "$wt/.agentlab"
python3 -c "import json,sys,time; d=json.loads(sys.argv[1]); d['ts']=time.time(); print(json.dumps(d))" "$out" >> "$wt/.agentlab/fresh.jsonl" 2>/dev/null || true
echo "$out"

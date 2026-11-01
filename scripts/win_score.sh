#!/usr/bin/env bash
# Win-mode score for one worktree: build its op stream, run it on shared input draws with the official simulator and
# checks (tools/nonce_search crnfast), and compare draw by draw with our base and the public head (agentlab/winscore.py).
# Prints a short report; the last line is JSON (ok, score, metrics). Also recorded to <worktree>/.agentlab/scores.jsonl.
#
#   scripts/win_score.sh <worktree> [draws=24] [challenge checkout] [first salt=0] [reference json]
#
# draws: more draws = smaller error bars (about 3-6 s per draw per core). The reference json (our base's and the
# public head's results on the same draws) is written by agentlab (work/win/crn_ref.json); without it the score is raw.
set -euo pipefail
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true
wt="$(cd "${1:?worktree}" && pwd -P)"
here="$(cd "$(dirname "$0")/.." && pwd -P)"
draws="${2:-${AGENTLAB_WIN_DRAWS:-24}}"
[[ -n "$draws" ]] || draws=24
challenge="${3:-${AGENTLAB_CHALLENGE:-$(dirname "$wt")/../ecdsafail-challenge}}"
salt="${4:-0}"
ref="${5:-${AGENTLAB_WIN_REF:-$here/work/win/crn_ref.json}}"
budget="${AGENTLAB_WIN_BUDGET:-0.0}"
rate="${AGENTLAB_WIN_RATE:-1000}"
record() { mkdir -p "$wt/.agentlab"; python3 -c "import json,sys,time; d=json.loads(sys.argv[1]); d.pop('per_draw',None); d['ts']=time.time(); print(json.dumps(d))" "$1" >> "$wt/.agentlab/scores.jsonl" 2>/dev/null || true; echo "$1"; }

ns="$here/tools/nonce_search/target/release/nonce_search"
if [[ ! -x "$ns" || "$(cat "$here/tools/nonce_search/.agentlab-version" 2>/dev/null)" != "3" ]]; then
  bash "$here/scripts/build_nonce_search.sh" "$challenge" >&2
fi

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
if ! bash "$here/scripts/build_ops.sh" "$wt" "$scratch"; then
  record '{"ok": false, "reason": "build failed (see the report above)"}'
  exit 1
fi

threads="${AGENTLAB_WIN_THREADS:-$(nproc 2>/dev/null || echo 4)}"
echo "-- crn evaluation ($draws draws from salt $salt) --"
t1=$(date +%s)
set +e
( cd "$scratch" && "$ns" crnfast "$salt" "$draws" "$threads" > "$scratch/crn.log" 2> "$scratch/crn.err" )
rc=$?
set -e
grep "^CRN" "$scratch/crn.log" || tail -5 "$scratch/crn.err"
echo "crn: exit $rc, $(( $(date +%s) - t1 ))s"
set +e
out="$(PYTHONPATH="$here" python3 -m agentlab.winscore "$scratch/crn.log" "$ref" "$budget" "$rate")"
set -e
echo "$out" | sed '$d'
record "$(echo "$out" | tail -1)"
echo "$out" | tail -1 | python3 -c "import json,sys; sys.exit(0 if json.loads(sys.stdin.read()).get('ok') else 1)"

#!/usr/bin/env bash
# Hand merge of our win base onto a public head that ends apply_merged_policy with a hash-bound source_edits pass
# (39b7cd5 onward: disc50_followup.txt). A plain replay does not build there: that pass inserts ops at position 0
# and our position-bound passes, which run after it, then miss their sites. Here our commits are replayed, the final
# pass is moved after our passes and applied by rebind_edits (each site found by its ops and 24 ops of context), the
# result is built, and only if it builds does agentlab/win-base move. Run with win mode stopped:
#
#   docker/run.sh bash tools/merge_head/merge_head.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd -P)"
ch="${AGENTLAB_CHALLENGE_DIR:-/root/ecdsafail-challenge}"
base=agentlab/win-base
G=(git -c user.name=agentlab -c user.email=agentlab@localhost -c "safe.directory=*")
cd "$ch"
"${G[@]}" fetch -q origin main
up="$(git rev-parse origin/main)"; old="$(git rev-parse "$base")"
if git merge-base --is-ancestor "$up" "$old"; then echo "our base ${old:0:8} already contains the public head ${up:0:8}; nothing to do"; exit 0; fi
fork="$(git merge-base "$old" "$up")"
mapfile -t ours < <(git rev-list --reverse --no-merges "$fork..$old")
echo "== public head ${up:0:8}; our base ${old:0:8} = ${#ours[@]} commit(s) on ${fork:0:8}"
"${G[@]}" branch -f "agentlab/win-base-before-${up:0:8}" "$old"
echo "   (our base as it was is kept as branch agentlab/win-base-before-${up:0:8})"
wt="$here/work/win/worktrees/merge-head"
"${G[@]}" worktree remove --force "$wt" 2>/dev/null || true
"${G[@]}" worktree add -q --detach "$wt" "$up"
cd "$wt"
if ! "${G[@]}" cherry-pick -x --keep-redundant-commits "${ours[@]}" > /dev/null; then
  "${G[@]}" cherry-pick --abort || true
  echo "!! our commits do not apply on ${up:0:8}; base left at ${old:0:8}"; exit 1
fi
python3 "$here/tools/merge_head/patch.py" "$wt"
"${G[@]}" add -A src/point_add
"${G[@]}" commit -qm "merge our win base onto ${up:0:8}: its final source_edits pass runs after our passes (rebind_edits)"
new="$(git rev-parse HEAD)"
out="$here/work/win/crn_ops/merge-${new:0:8}"
mkdir -p "$out"
echo "== fetching crates (a fresh container has none) and building ${new:0:8} (a few minutes)"
(cd "$wt" && RUSTUP_TOOLCHAIN="$(sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' rust-toolchain)" cargo fetch --locked -q)
set +e
bash "$here/scripts/build_ops.sh" "$wt" "$out" > "$out.log" 2>&1
rc=$?
set -e
grep -E "rebind|panicked|left:|right:|!!" "$out.log" | head -30 || true
rm -f "$out/ops.bin"
cd "$ch"
if [[ $rc -ne 0 ]]; then
  "${G[@]}" worktree remove --force "$wt" || true
  echo "!! the merged base does not build; base left at ${old:0:8} (log: work/win/crn_ops/merge-${new:0:8}.log)"; exit 1
fi
"${G[@]}" branch -f "$base" "$new"
"${G[@]}" worktree remove --force "$wt" || true
rm -f "$here/work/win/replay_failed.json"
echo "== OK: our base is now ${new:0:8} on the public head ${up:0:8}. Start win mode again; its first round measures the lead."

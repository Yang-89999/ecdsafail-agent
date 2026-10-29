---
name: claim-audit
description: How the Auditor turns a promising solver result into a claim that can be submitted: clean-room re-run on the newest upstream head, protected-file diff, determinism check, official benchmark, comparison to the best.
---

# Claim audit

A solver's number is a hypothesis. A claim needs all of:

1. **Protected files untouched.** `git diff --name-only <base>` and untracked files must lie entirely inside the editable paths.
2. **Applies on the newest upstream head.** Fetch, new worktree at `origin/main`, cherry-pick the experiment's commits. A conflict or a stale base is a rejection; the board moves daily.
3. **Deterministic.** Build and run `build_circuit` twice; `sha256(ops.bin)` must match. Non-determinism would mean the submitted circuit is not the one measured.
4. **Official checker passes.** `bash benchmark.sh` in the clean worktree; parse the printed metrics and `score.json`. The agent's own fast score is discarded.
5. **Beats the bar.** Compare with the public best (the upstream head measured by the same checker) and the best verified local result. Policy: submit only when strictly below the bar by at least `min_relative_gain`.

Record the verdict in the ledger with every number and log path. A rejected experiment keeps its logs; the Planner learns from them.

The audit never edits the candidate. If it fails, it goes back to the backlog as a lead with the failure reason attached.

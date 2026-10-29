---
name: experiment-ledger
description: Conventions for the shared SQLite ledger that holds ideas, experiments, verified bests and approvals; what every role must record and how the Planner ranks from it.
---

# Experiment ledger

Tables: `ideas`, `experiments`, `bests`, `approvals` (see `agentlab/ledger.py`).

Rules:

- Every solver run is an `experiments` row from start to finish, with worktree, base and head commits, model, shots, metrics, score, relative gain, note, and a log path. A crashed run is still a row (`failed`).
- `gain = (best_at_start - score) / best_at_start`; negative means worse; `-1.0` means invalid.
- `bests` holds only numbers printed by the official checker (9024 shots). Fast scores never enter it.
- `ideas.tries / gain_sum / best_gain` feed the Planner's bandit. An idea with four or more tries and a mean gain below −50 % is retired as `exhausted`.
- Follow-up ideas from solver reports enter with a low prior and `source=solver:<exp>`; duplicates by title are ignored.
- `approvals` is the human gate. `agentlab submit` refuses without an approval row for that experiment and an `audited` status.

`agentlab ledger` prints the summary used in planning. Public notes may quote only what the ledger holds.

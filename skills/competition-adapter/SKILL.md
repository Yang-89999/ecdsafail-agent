---
name: competition-adapter
description: How to add a new verifiable-benchmark board (another Yukon challenge, sig.golf) to agentlab by implementing the adapter interface and a domain skill pack.
---

# Competition adapter

Implement `agentlab/adapters/<name>.py` as a `CompetitionAdapter` subclass:

- `editable_paths`, `protected_paths`
- `ensure_ready()`: toolchain, deps, prebuilt binaries, idempotent
- `current_best()`: the public bar, measured or fetched; cache by upstream commit
- `fast_score(worktree, shots)`: cheap approximation; never a claim
- `full_verify(worktree, note)`: the official checker exactly as CI runs it
- `rules_text()`: the rules the solver prompt quotes
- `submit(worktree, note)`: official channel only; the CLI enforces the approval gate

Register it in `agentlab/adapters/__init__.py`, add `skills/<domain>/SKILL.md` packs, and seed ideas in a planner module for that board. The orchestrator, ledger, auditor and CLI do not change.

sig.golf sketch: editable `submission/`; fast score = local RV64IM emulator cycles × signature bytes; full verify = `python3 scripts/run.py` (needs Linux with Landlock and user systemd, hours); extra solver split into scheme engineer and Lean prover.

"""agentlab: an evolutionary-search agent system for verifiable research benchmarks.

Roles (see docs/design.md):
  Scout      -> turns the public board and merged submissions into ideas
  Planner    -> bandit over the idea backlog, allocates solver runs
  Solver     -> one hypothesis per git worktree, edits only the allowed folder
  Scorer     -> cheap metric (few-shot evaluation)
  Verifier   -> the official checker, full shots
  Auditor    -> clean-room re-run, protected-file diff, determinism check
  Human gate -> nothing is submitted without an explicit approval record
"""

__version__ = "0.1.0"

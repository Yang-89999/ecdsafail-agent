# Design

Full design note (with the competition analysis): the Claude Doc "Signature-competition research agent: design v0".

## Roles

| Role | Implementation | Trust |
|---|---|---|
| Scout | `roles/scout.py`: diffs of merged upstream submissions → ideas | leads only |
| Literature Scout | `roles/lit_scout.py`: arXiv queries and post URLs → `papers` → model extraction → ideas with status `candidate`, source `lit:<paper>` | reads only; quotes a paper's numbers only as the paper's claim |
| Idea Reviewer | `roles/idea_reviewer.py`: scores candidates on relevance, novelty (vs the code, the backlog, the profile), feasibility, expected gain → `open` with a prior, or `rejected` with the reason kept in `ideas.review` | runs before each round when candidates exist |
| Planner | `roles/planner.py`: UCB bandit over `ideas`, optimistic prior for untried ideas | — |
| Solver | `roles/solver.py`: one idea, one worktree, one `codex exec` / API run, then independent re-score | numbers discarded |
| Fast scorer | `scripts/fast_score.sh` + `tools/fast_eval` (64 shots) | ranking only |
| Full verifier | `benchmark.sh` in the worktree (9024 shots) | the claim |
| Auditor | `roles/auditor.py`: protected diff, clean room on upstream head, determinism, full verify, compare | gate to submit |
| Human | `agentlab approve` row in the ledger | required |

## Loop (one round)

1. fetch upstream; `current_best()` = official score of the head, cached by commit
2. Planner picks `solvers` ideas by UCB (`mean gain + c·sqrt(ln N / n)`)
3. each Solver: worktree at upstream head → agent run → protected diff → fast score → ledger row, idea reward, follow-up ideas
4. any run whose fast gain > threshold → Auditor → `audited` + verified best row
5. retire ideas with ≥4 tries and mean gain < −50 %

## Timing on this host (4 vCPU, 2026-10-29)

- cargo build of build_circuit: ~2 min cold, seconds warm
- build_circuit (the generator, with its simplification passes): ~5 min
- eval 9024 shots: ~1 min; 64 shots: seconds
- upstream head e4076ae: 740,969 Toffoli × 1,234 qubits = 914,355,746

So one score cycle is ~6 minutes and the generator dominates; a solver doing 10 cycles runs about an hour.

## Open items

- `ecdsafail submit` needs the CLI and an API key on the host that submits.
- `current_best()` trusts the repo head; the board may be ahead of the repo (Scout should compare against the page).
- sig.golf adapter: not started.

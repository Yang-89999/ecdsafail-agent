# Design

Full design note (with the competition analysis): the Claude Doc "Signature-competition research agent: design v0".

## Roles

| Role | Implementation | Trust |
|---|---|---|
| Scout | `roles/scout.py`: diffs of merged upstream submissions → ideas | leads only |
| Literature Scout | `roles/lit_scout.py`: arXiv queries and post URLs → `papers` → model extraction → ideas with status `candidate`, source `lit:<paper>` | reads only; quotes a paper's numbers only as the paper's claim |
| Strategist | `roles/strategist.py`: profile + backlog + reports → `work/problem.md`: formulation (min T×W, T by stage, W one peak), sub-problems with ceilings and closed/open status, hypotheses with predicted gain + refutation test; sets priors = min(prediction, ceiling), prunes, tags ideas by sub-problem; runs first and every 6 experiments | the deep-thinking pass of the planning block |
| Coordinator | `roles/coordinator.py`: the main agent of the planning block. Each time an executor slot is free it reads the runs in progress (with their live `scores.jsonl`), the results that came back since its last decision, the backlog and the bandit's ranking; lists its options; decides `new` (an idea, with one concrete instruction), `continue` (a finished run from its own final tree and notes), `combine` (cherry-pick several finished branches onto the upstream head), `retire` (with a reason) or `wait`; appends the decision in plain language to `work/coordinator.md` | the human reads `work/coordinator.md`; falls back to the bandit's picks when the model gives no answer |
| Brainstorm | `roles/brainstorm.py`: profile + recent reports + backlog → candidates, plus up to 2 open questions sent to `ask` | runs when the backlog is low |
| Ask (on-demand literature) | `lit_scout.ask`: one specific question → arXiv search → cited answer memo (cached) + candidate ideas; exposed to solvers as `scripts/ask_lit.sh` | question-driven, not bulk |
| Idea Reviewer | `roles/idea_reviewer.py`: scores candidates on relevance, novelty (vs the code, the backlog, the profile), feasibility, expected gain → `open` with a prior, or `rejected` with the reason kept in `ideas.review` | runs before each round when candidates exist |
| Bandit | `roles/planner.py`: UCB over `ideas` (untried first by prior, gain-scaled exploration, spread across sub-problems) | a hint to the Coordinator and its fallback |
| Solver (executor) | `roles/solver.py`: one brief, one worktree, one `codex exec` / API run, then independent re-score; modes `new` (worktree at the upstream head), `continue` (worktree from branch `agentlab/exp<parent>`, parent's base commit kept so the audit's cherry-pick range covers both runs), `combine` (upstream head, the brief lists the branches to cherry-pick); every `fast_score.sh` call appends to `<worktree>/.agentlab/scores.jsonl` | numbers discarded; no clock, stops on a verified gain or no hope (8 h safety cap) |
| Fast scorer | `scripts/fast_score.sh` + `tools/fast_eval` (64 shots) | ranking only |
| Full verifier | `benchmark.sh` in the worktree (9024 shots) | the claim |
| Auditor | `roles/auditor.py`: protected diff, clean room on upstream head, determinism, full verify, compare | gate to submit |
| Human | `agentlab approve` row in the ledger | required |

## Loop (asynchronous, `orchestrator.py`)

K executor slots run in parallel and independently. The loop wakes up whenever a run finishes or every minute:

1. a slot is free → fetch upstream, `current_best()`; replenish the backlog if low; Strategist revision if 6 runs finished since the last; **Coordinator decides** what the free slots do (it sees what is running, what finished since its last decision, the backlog, the bandit's hint) → launches
2. a run finishes → protected diff → independent fast score → ledger row, idea reward, notes harvested → the result waits for the Coordinator's next decision; fast gain > threshold → Auditor → `audited` + verified best row
3. retire ideas with ≥4 tries and mean gain < −50 %; model backend failing fast → exponential backoff on launches (running runs continue)

`--hours H` is a launch window, not a kill switch: after H hours nothing new starts and the loop waits for the running runs, then writes the unit summary (`work/units/`). `--rounds N` = N×K launches. `work/status.md` (rewritten every minute) shows the running runs with elapsed time, scored variants, best so far and last activity, the Coordinator's latest decision, and anything waiting for approval.

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

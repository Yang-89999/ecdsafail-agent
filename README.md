# agentlab: autoresearch agents for verifiable benchmarks

An evolutionary-search agent system for Yukon-style research boards. First target:
[ECDSA.fail](https://ecdsa.fail) (reversible secp256k1 point addition, score = Toffoli × qubits).
Second: sig.golf, through the same core and a new adapter.

The design is in `docs/design.md`. Short version: the official checker is the reward function.
A planning block runs the search: the Strategist formulates the task as an optimisation problem and
decomposes it into sub-problems with measured ceilings; the Coordinator (the main agent) assigns
sub-tasks to parallel executor agents (Solvers, one git worktree each), reads every result as it comes
back, and decides the next move: a new idea, continuing a run from its own tree, combining several
branches, retiring an idea, or waiting; it writes each decision in plain language to
`work/coordinator.md`. A bandit ranking is its hint and its fallback. Runs are asynchronous and have
no clock: a slow run keeps its slot while the others turn over, and a run ends when its executor finds
a gain or no hope. The Auditor re-runs anything promising in a clean room with the official checker,
and nothing is submitted without a recorded human approval. Reading roles feed the backlog: the
Literature Scout turns papers and posts into cited candidate ideas, the Ask tool answers a designer's
question with citations, and the Idea Reviewer scores each candidate before it can reach a solver.

## Layout

```
agentlab/            the system (config, ledger, llm backends, adapters, roles, orchestrator, cli)
skills/              SKILL.md packs the agents are prompted with (core + per-board)
scripts/             fast_score.sh (cheap metric), build_fast_eval.sh
tools/fast_eval/     the harness's trusted evaluator, built with a configurable shot count
work/                ledger.sqlite, worktrees/, runs/<exp>/ (prompt, agent log, report)  [git-ignored]
agentlab.toml        configuration
```

## Run in Docker (recommended on a Mac)

The container is the sandbox for the untrusted circuit code from the shared repo and for the
commands Codex runs, and it is Linux like the organisers' scorer.

```bash
git clone https://github.com/Yang-89999/ecdsafail-agent && cd ecdsafail-agent
docker/run.sh                        # first start: builds the image, clones the harness, builds it (~15 min)
codex login --device-auth            # inside the container: sign in with ChatGPT on your Mac's browser
agentlab best                        # official score of the upstream head (~6-10 min)
agentlab run --rounds 1 --solvers 1  # first solver round
```

The harness checkout, ledger/worktrees and Codex login live in Docker volumes and survive restarts.
Edits to this repo on the Mac are visible inside the container immediately.

### Units of work (5 h each) or a week unattended

```bash
docker/start.sh                 # one unit = agentlab run --hours 5 --solvers 2, detached; run again for the next unit
docker/start.sh --days 7        # or one long run
docker/logs.sh                  # follow the log
docker/shell.sh                 # shell inside: agentlab status | agentlab coordinator | agentlab watch <exp> | agentlab approve <exp> --by Yang
docker/stop.sh
```

A unit starts with the Strategist (formulation, decomposition into sub-problems with measured ceilings,
priors from predicted gains), revised after every six finished runs. Then the Coordinator fills the two
executor slots and refills each one as soon as it frees up, feeding every result back into its next
decision (`work/coordinator.md`). The 5 hours are a launch window: after it nothing new starts, but a run
in progress is never cut off; the unit ends when the last run finishes (`work/units/`). Between units
the Mac may sleep; the ledger, worktrees, notes and problem statement carry over, and the next unit's
Coordinator can continue any earlier run from its tree. The loop also refills its own backlog (upstream
scout + brainstorm when fewer than 3 ideas are open, one arXiv refresh per day, every candidate through
the reviewer), waits with exponential backoff when the model backend fails fast (usage limits), and
rewrites `work/status.md` every minute: the running runs (elapsed, scored variants, best so far, last
activity), the Coordinator's latest decision, and anything waiting for approval. Keep the Mac plugged
in with the lid open during a unit; `start.sh` runs `caffeinate` so it does not sleep.

## Setup (Linux or macOS, without Docker)

macOS note: the harness's `setup.sh` sources `~/.cargo/env` and the system bash 3.2 aborts if that file is
missing, so on a fresh Mac run `mkdir -p ~/.cargo && touch ~/.cargo/env` once before `agentlab setup`.
Homebrew, Node (`npm i -g @openai/codex`) and Python 3 are needed first.

```bash
git clone https://github.com/ecdsafail/ecdsafail-challenge.git ../ecdsafail-challenge
pip install -e .
agentlab setup            # runs the harness's setup.sh (Rust 1.93, bubblewrap) and builds tools/fast_eval
agentlab best             # measures the upstream head with the official checker (~6 min), caches it
agentlab ideas            # backlog (seeded with research directions on first run)
```

Model access, one of:

- Codex CLI: `npm i -g @openai/codex`, then `codex login` (or `printf '%s' "$KEY" | codex login --with-api-key`); `backend = "codex"`.
- Responses API: `export OPENAI_API_KEY=...`; `backend = "openai"`, set `model`.

## Running

```bash
agentlab strategy                        # Strategist: formulate the problem, decompose with ceilings, allocate -> work/problem.md
agentlab lit-scout                       # arXiv queries (+ --url for posts) -> papers table -> candidate ideas with citations
agentlab review                          # score candidates: accept into the backlog (with a prior) or reject with a reason
agentlab ideas                           # the backlog, with status, prior and the reviewer's note
agentlab run --rounds 3 --solvers 2      # 6 executor runs on 2 parallel slots, the Coordinator deciding each launch
agentlab coordinator                     # the Coordinator's last decisions (options, decision, launches, retirements)
agentlab lessons                         # what the system has learned so far (revised after every run, read by every agent)
agentlab trend -n 30                     # leaderboard history: measured score per merged submission, what each step changed, synthesis -> work/trend.md
agentlab watch [exp]                     # readable live view of a solver run (second terminal)
agentlab ledger                          # what happened
agentlab audit 12                        # clean-room + official benchmark for experiment 12
agentlab approve 12 --by Yang            # human gate
agentlab submit 12                       # refuses unless audited, approved, and strictly better than the bar
```

Every solver run leaves `work/runs/exp<N>/{prompt.md,agent.log,report.json}`, its notes under
`work/notes/exp<N>/`, and a worktree under `work/worktrees/exp<N>` with its commits on branch
`agentlab/exp<N>`; a later `continue` run starts from that branch and a `combine` run cherry-picks several.

Designers can also ask the literature a question mid-run (`bash scripts/ask_lit.sh "..."` inside a solver,
`agentlab ask "..."` by hand): it searches arXiv for that question, answers with citations, caches the memo
under `work/lit/`, and files any transferable ideas as candidates. The brainstorm role asks up to two open
questions per call. Literature queries are in `agentlab/roles/lit_scout.py` (`DEFAULT_QUERIES`); override them with
`lit_queries = [...]` in `agentlab.toml` or `--query` on the command line. The model used by every
role is `model` in `agentlab.toml` (or `AGENTLAB_MODEL`); leave it empty for the Codex default or set
the exact name shown by `/model` inside `codex`. The ledger records the model for each run.

## Rules the system enforces

- Solvers can only change the editable paths; the Auditor diffs against the base commit.
- Agent-reported numbers are discarded; the orchestrator re-scores the tree it is handed.
- `bests` holds only numbers printed by the official checker.
- No submission without `audited` status, an approval row, and a strict improvement.

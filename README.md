# agentlab: autoresearch agents for verifiable benchmarks

An evolutionary-search agent system for Yukon-style research boards. First target:
[ECDSA.fail](https://ecdsa.fail) (reversible secp256k1 point addition, score = Toffoli × qubits).
Second: sig.golf, through the same core and a new adapter.

The design is in `docs/design.md`. Short version: the official checker is the reward function,
parallel Solver agents mutate one hypothesis each in their own git worktree, a bandit Planner
allocates runs across the idea backlog, the Auditor re-runs everything in a clean room with the
official checker, and nothing is submitted without a recorded human approval. Two reading roles feed
the backlog: the Literature Scout turns papers and posts into cited candidate ideas, and the Idea
Reviewer scores each candidate before it can reach a solver.

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
docker/start.sh                 # one unit = agentlab run --hours 5 --solvers 1, detached; run again for the next unit
docker/start.sh --days 7        # or one long run
docker/logs.sh                  # follow the log
docker/shell.sh                 # shell inside: agentlab status | agentlab watch | agentlab approve <exp> --by Yang
docker/stop.sh
```

A long run starts with the Strategist (formulation, decomposition into sub-problems with measured ceilings,
priors from predicted gains) and revises it after every six experiments; solvers get the formulation and
their sub-problem in the brief, and parallel solvers are spread over different sub-problems. It also refills its own backlog (upstream scout + brainstorm when fewer than 3 ideas are open, one
arXiv refresh per day, every candidate through the reviewer), waits with exponential backoff when the
model backend fails fast (usage limits), survives a crashed round, and rewrites `work/status.md` after
every round with anything waiting for approval. Keep the Mac plugged in with the lid open; `start.sh`
runs `caffeinate` so it does not sleep.

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
agentlab run --rounds 3 --solvers 2      # each round: review pending candidates, pick ideas, run solvers, re-score, audit winners
agentlab watch                           # readable live view of the latest solver run (second terminal)
agentlab ledger                          # what happened
agentlab audit 12                        # clean-room + official benchmark for experiment 12
agentlab approve 12 --by Yang            # human gate
agentlab submit 12                       # refuses unless audited, approved, and strictly better than the bar
```

Every solver run leaves `work/runs/exp<N>/{prompt.md,agent.log,report.json}` and a worktree under
`work/worktrees/exp<N>` with its commits on branch `agentlab/exp<N>`.

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

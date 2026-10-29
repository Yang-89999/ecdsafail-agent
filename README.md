# agentlab: autoresearch agents for verifiable benchmarks

An evolutionary-search agent system for Yukon-style research boards. First target:
[ECDSA.fail](https://ecdsa.fail) (reversible secp256k1 point addition, score = Toffoli × qubits).
Second: sig.golf, through the same core and a new adapter.

The design is in `docs/design.md`. Short version: the official checker is the reward function,
parallel Solver agents mutate one hypothesis each in their own git worktree, a bandit Planner
allocates runs across the idea backlog, the Auditor re-runs everything in a clean room with the
official checker, and nothing is submitted without a recorded human approval.

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
agentlab run --rounds 3 --solvers 2      # each round: pick ideas, run solvers in parallel, re-score, audit winners
agentlab ledger                          # what happened
agentlab audit 12                        # clean-room + official benchmark for experiment 12
agentlab approve 12 --by Yang            # human gate
agentlab submit 12                       # refuses unless audited, approved, and strictly better than the bar
```

Every solver run leaves `work/runs/exp<N>/{prompt.md,agent.log,report.json}` and a worktree under
`work/worktrees/exp<N>` with its commits on branch `agentlab/exp<N>`.

## Rules the system enforces

- Solvers can only change the editable paths; the Auditor diffs against the base commit.
- Agent-reported numbers are discarded; the orchestrator re-scores the tree it is handed.
- `bests` holds only numbers printed by the official checker.
- No submission without `audited` status, an approval row, and a strict improvement.

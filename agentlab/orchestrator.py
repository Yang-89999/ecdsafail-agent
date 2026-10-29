"""The loop.

round:
  best   <- adapter.current_best()           (public bar)
  review <- idea_reviewer on pending candidates
  idea   <- planner.ucb_pick(ledger.ideas)   (one per solver slot)
  exp    <- solver.run_solver(...)            (parallel, one worktree each)
  if exp beats best on the fast score        -> auditor.audit(exp)   (full official run)
  if audited and beats best                  -> status "audited": waits for a human approval
submission is a separate, explicit command (cli: submit) that checks the approval record.

Long runs (`--hours`): the loop replenishes the backlog when it runs low (scout, brainstorm,
daily literature refresh, all through the reviewer), backs off when the model backend fails
fast (usage limits), survives a crashed round, and writes work/status.md after every round.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .adapters import load_adapter
from .config import Config
from .ledger import Ledger
from .roles import auditor, planner, solver

DAY = 24 * 3600


def _fmt(seconds: float) -> str:
    h = int(seconds) // 3600
    return f"{h // 24}d {h % 24}h" if h >= 24 else f"{h}h {(int(seconds) % 3600) // 60}m"


def replenish(cfg: Config, adapter, ledger: Ledger, state: dict, force_lit: bool = False) -> None:
    """Refill the backlog: upstream scout, brainstorm from our own evidence, daily literature refresh."""
    from .roles.brainstorm import brainstorm
    from .roles.idea_reviewer import review
    from .roles.lit_scout import extract, harvest
    from .roles.scout import scout

    if cfg.backend == "dry":
        return
    if force_lit or time.time() - state.get("last_lit", 0) > DAY:
        try:
            n = harvest(cfg, ledger, None, None, 10, recent=True)
            print(f"  [replenish] literature refresh: {n} new papers", flush=True)
            if n:
                extract(cfg, ledger)
        except Exception as e:  # noqa: BLE001
            print(f"  [replenish] literature refresh failed: {e}")
        state["last_lit"] = time.time()
    if len(ledger.ideas("open")) < max(3, cfg.solvers):
        try:
            print(f"  [replenish] backlog low: scouting upstream and brainstorming", flush=True)
            scout(cfg, adapter, ledger, 10, summarize=True)
            brainstorm(cfg, adapter, ledger)
        except Exception as e:  # noqa: BLE001
            print(f"  [replenish] failed: {e}")
    if ledger.ideas("candidate"):
        print(f"  reviewing {len(ledger.ideas('candidate'))} candidate ideas: {review(cfg, adapter, ledger)}", flush=True)


def write_status(cfg: Config, ledger: Ledger, state: dict) -> None:
    p = Path(cfg.runs_dir).parent / "status.md"
    lines = [f"# agentlab status  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", "",
             f"rounds done: {state['rounds']}   elapsed: {_fmt(time.time() - state['t0'])}"
             + (f"   remaining: {_fmt(max(0, state['deadline'] - time.time()))}" if state.get("deadline") else ""),
             f"backend failures in a row: {state['fails']}", "", "```", ledger.summary_text(25), "```"]
    audited = ledger.experiments(5, "audited")
    if audited:
        lines += ["", "## Waiting for your approval"] + [
            f"- exp #{e['id']} score {e['score']:,}  →  `agentlab approve {e['id']} --by Yang` then `agentlab submit {e['id']}`"
            for e in audited]
    p.write_text("\n".join(lines) + "\n")


def run_rounds(cfg: Config, rounds: int, solvers: int | None = None, audit_threshold: float = 0.0,
               hours: float | None = None) -> None:
    adapter = load_adapter(cfg)
    adapter.ensure_ready()
    ledger = Ledger(cfg.ledger_path)
    planner.seed_ideas(ledger)
    rng = random.Random()
    solvers = solvers or cfg.solvers
    base_ref = f"{cfg.upstream_remote}/{cfg.upstream_branch}"
    state = {"t0": time.time(), "deadline": time.time() + hours * 3600 if hours else None,
             "rounds": 0, "fails": 0, "last_lit": time.time()}
    _maybe_strategize(cfg, adapter, ledger, state, force=True)
    if hours:
        print(f"running for {_fmt(hours * 3600)} (until {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(state['deadline']))}); "
              f"status in {Path(cfg.runs_dir).parent / 'status.md'}", flush=True)

    while True:
        if rounds and state["rounds"] >= rounds:
            break
        if state["deadline"] and time.time() >= state["deadline"]:
            print("time is up", flush=True)
            break
        try:
            _one_round(cfg, adapter, ledger, rng, solvers, audit_threshold, base_ref, state)
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print("  round crashed; waiting 5 min before the next one", flush=True)
            time.sleep(300)
        state["rounds"] += 1
        write_status(cfg, ledger, state)
        if state["fails"]:
            wait = min(2 * 3600, 20 * 60 * 2 ** (state["fails"] - 1))
            print(f"  model backend failed fast {state['fails']}x in a row (usage limit?); waiting {wait // 60} min", flush=True)
            time.sleep(wait)
        time.sleep(1)
    print(ledger.summary_text())


def _maybe_strategize(cfg, adapter, ledger, state, force: bool = False, every: int = 6) -> None:
    """Formulate/decompose/allocate before the loop, then after every `every` finished experiments."""
    from .roles.strategist import problem_path, strategize
    if cfg.backend == "dry":
        return
    n_exp = len(ledger.experiments(10000))
    due = force and not problem_path(cfg).exists() or n_exp - state.get("last_strat_exp", 0) >= every
    if not due:
        return
    try:
        print(f"  [strategist] revising the problem statement ({n_exp} experiments so far)", flush=True)
        strategize(cfg, adapter, ledger)
    except Exception as e:  # noqa: BLE001
        print(f"  [strategist] failed: {e}")
    state["last_strat_exp"] = n_exp


def _one_round(cfg, adapter, ledger, rng, solvers, audit_threshold, base_ref, state) -> None:
    subprocess.run(["git", "fetch", "-q", cfg.upstream_remote, cfg.upstream_branch], cwd=adapter.root, check=False)
    best = adapter.current_best()
    if best:
        ledger.record_best(best["score"], best["toffoli"], best["qubits"], f"upstream:{best['commit']}", verified=True)
    hdr = f"=== round {state['rounds'] + 1}" + (f"  elapsed {_fmt(time.time() - state['t0'])}" if state.get("deadline") else "")
    print(f"\n{hdr}  public best {best['score'] if best else '?'} ===", flush=True)
    replenish(cfg, adapter, ledger, state)
    _maybe_strategize(cfg, adapter, ledger, state)
    picks = []
    ideas = ledger.ideas("open")
    for _ in range(solvers):
        taken_ids = {p.id for p in picks}
        taken_sub = {getattr(p, "subproblem", None) for p in picks} - {None}
        pool = [i for i in ideas if i.id not in taken_ids and getattr(i, "subproblem", None) not in taken_sub]
        idea = planner.ucb_pick(pool or [i for i in ideas if i.id not in taken_ids], cfg.explore_c, rng)
        if idea:
            picks.append(idea)
    if not picks:
        print("no open ideas even after replenishing; reopening the least-tried exhausted ideas", flush=True)
        for i in sorted(ledger.ideas("exhausted"), key=lambda i: i.tries)[:solvers]:
            ledger.set_idea_status(i.id, "open")
        picks = ledger.ideas("open")[:solvers]
        if not picks:
            raise RuntimeError("backlog empty")
    fast_fail = False
    with ThreadPoolExecutor(max_workers=len(picks)) as pool:
        futs = {pool.submit(solver.run_solver, cfg, adapter, ledger, i.id, planner.planner_brief(ledger, i, best), base_ref, best): i
                for i in picks}
        for f in as_completed(futs):
            idea = futs[f]
            try:
                exp_id = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  idea {idea.id} crashed: {e}")
                continue
            e = ledger.experiment(exp_id)
            print(f"  exp #{exp_id} [{idea.title[:50]}] {e['status']} score={e['score']} gain={e['gain']}", flush=True)
            if e["status"] == "backend_failed":
                fast_fail = True
                continue
            if e["status"] == "scored" and e["gain"] is not None and e["gain"] > audit_threshold:
                print(f"  -> beats the bar on the fast score; auditing #{exp_id} (full official run)", flush=True)
                v = auditor.audit(cfg, adapter, ledger, exp_id)
                print(f"  audit #{exp_id}: {'OK' if 'fail' not in v else v['fail']} full={v.get('full', {}).get('score')} beats_best={v.get('beats_best')}", flush=True)
                if v.get("beats_best"):
                    print(f"  *** exp #{exp_id} beats the public bar; waiting for `agentlab approve {exp_id} --by Yang` ***", flush=True)
    state["fails"] = state["fails"] + 1 if fast_fail else 0
    # retire ideas that keep failing
    for i in ledger.ideas("open"):
        if i.tries >= 4 and i.gain_sum / i.tries < -0.5:
            ledger.set_idea_status(i.id, "exhausted")

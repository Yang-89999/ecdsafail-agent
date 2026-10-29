"""The loop.

round:
  best   <- adapter.current_best()           (public bar)
  idea   <- planner.ucb_pick(ledger.ideas)   (one per solver slot)
  exp    <- solver.run_solver(...)            (parallel, one worktree each)
  if exp beats best on the fast score        -> auditor.audit(exp)   (full official run)
  if audited and beats best                  -> status "audited": waits for a human approval
submission is a separate, explicit command (cli: submit) that checks the approval record.
"""

from __future__ import annotations

import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .adapters import load_adapter
from .config import Config
from .ledger import Ledger
from .roles import auditor, planner, solver


def run_rounds(cfg: Config, rounds: int, solvers: int | None = None, audit_threshold: float = 0.0) -> None:
    adapter = load_adapter(cfg)
    adapter.ensure_ready()
    ledger = Ledger(cfg.ledger_path)
    planner.seed_ideas(ledger)
    rng = random.Random()
    solvers = solvers or cfg.solvers
    base_ref = f"{cfg.upstream_remote}/{cfg.upstream_branch}"

    for r in range(rounds):
        subprocess.run(["git", "fetch", "-q", cfg.upstream_remote, cfg.upstream_branch], cwd=adapter.root, check=False)
        best = adapter.current_best()
        if best:
            ledger.record_best(best["score"], best["toffoli"], best["qubits"], f"upstream:{best['commit']}", verified=True)
        print(f"\n=== round {r + 1}/{rounds}  public best {best['score'] if best else '?'} ===", flush=True)
        if ledger.ideas("candidate"):
            from .roles.idea_reviewer import review
            print(f"  reviewing {len(ledger.ideas('candidate'))} candidate ideas first: {review(cfg, adapter, ledger)}", flush=True)
        picks = []
        ideas = ledger.ideas("open")
        for _ in range(solvers):
            idea = planner.ucb_pick([i for i in ideas if i.id not in {p.id for p in picks}], cfg.explore_c, rng)
            if idea:
                picks.append(idea)
        if not picks:
            print("no open ideas; run `agentlab scout` or `agentlab ideas add`")
            return
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
                if e["status"] == "scored" and e["gain"] is not None and e["gain"] > audit_threshold:
                    print(f"  -> beats the bar on the fast score; auditing #{exp_id} (full official run)", flush=True)
                    v = auditor.audit(cfg, adapter, ledger, exp_id)
                    print(f"  audit #{exp_id}: {'OK' if 'fail' not in v else v['fail']} full={v.get('full', {}).get('score')} beats_best={v.get('beats_best')}", flush=True)
        # retire ideas that keep failing
        for i in ledger.ideas("open"):
            if i.tries >= 4 and i.gain_sum / i.tries < -0.5:
                ledger.set_idea_status(i.id, "exhausted")
        time.sleep(1)

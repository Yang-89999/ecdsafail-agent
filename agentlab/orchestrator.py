"""The loop: a Coordinator over parallel executor slots.

  slot free  -> coordinator.decide(running, finished since last decision, backlog, bandit hint)
                 launches: new idea | continue a finished run from its tree | combine finished branches
  run ends   -> independent re-score, idea reward, notes harvested; if it beats the bar on the fast
                 score -> auditor.audit (full official run) -> status "audited": waits for a human approval
  the result goes back to the Coordinator at its next decision; the Strategist revises the problem
  statement every few finished runs; the backlog refills itself (scout, brainstorm, daily literature
  refresh, all through the reviewer).

Runs are asynchronous: a slow run keeps its slot while the other slots turn over, and nothing is
stopped because time passed. `--hours H` is a launch window: after H hours no new run starts and the
loop waits for the running ones to finish (the unit then ends with a summary under work/units/).
`--rounds N` on K slots launches N x K runs. A run stops when its executor finds a gain or no hope
(the only clock is a long safety cap, solver_timeout_s). Submission is a separate, explicit command
(cli: submit) that checks the approval record.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from .adapters import load_adapter
from .config import Config
from .ledger import Ledger
from .roles import auditor, coordinator, planner, solver

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
        except Exception as e:  # noqa: BLE001
            print(f"  [replenish] literature refresh failed: {e}")
        state["last_lit"] = time.time()
    # Papers fetched by hand (`agentlab lit-scout --fetch-only`) or by the refresh are read a few batches per
    # decision, so a large fetch does not hold up the first launches of a unit.
    unread = len(ledger.papers(unextracted_only=True))
    if unread:
        try:
            print(f"  [replenish] reading {min(unread, 32)} of {unread} unread papers", flush=True)
            extract(cfg, ledger, max_batches=4)
        except Exception as e:  # noqa: BLE001
            print(f"  [replenish] extraction failed: {e}")
    if len(ledger.ideas("open")) < max(3, cfg.solvers):
        try:
            print(f"  [replenish] backlog low: scouting upstream and brainstorming", flush=True)
            scout(cfg, adapter, ledger, 10, summarize=True)
            brainstorm(cfg, adapter, ledger)
        except Exception as e:  # noqa: BLE001
            print(f"  [replenish] failed: {e}")
    if ledger.ideas("candidate"):
        print(f"  reviewing {len(ledger.ideas('candidate'))} candidate ideas: {review(cfg, adapter, ledger)}", flush=True)


def _running_rows(cfg: Config, ledger: Ledger, running: dict, best) -> list[str]:
    rows = ["| exp | idea | mode | sub-problem | elapsed | scored | best so far | last activity |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(running.values(), key=lambda r: r["exp_id"]):
        e = ledger.experiment(r["exp_id"])
        prog = coordinator.live_progress(e["worktree"] if e else None, r.get("log"))
        idea = ledger.idea(r["idea_id"]) if r.get("idea_id") else None
        bestcol = "-"
        if prog["best_score"]:
            bestcol = f"{prog['best_score']:,}"
            if best:
                ref = best.get("fast_score") or best["score"]
                bestcol += f" ({(ref - prog['best_score']) / ref * 100:+.3f} %)"
        last = f"{prog['idle_min']} min ago" if prog["idle_min"] is not None else "-"
        mode = r["mode"] + (f" {r['parents']}" if r["parents"] else "")
        rows.append(f"| #{r['exp_id']} | {('#' + str(r['idea_id']) + ' ' + idea.title[:40]) if idea else '-'} | {mode} | "
                    f"{getattr(idea, 'subproblem', None) or '-'} | {_fmt(time.time() - r['t0'])} | {prog['scored']}/{prog['attempts']} | "
                    f"{bestcol} | {last} |")
    return rows


def write_status(cfg: Config, ledger: Ledger, state: dict, running: dict | None = None, best=None) -> None:
    p = Path(cfg.runs_dir).parent / "status.md"
    window = ""
    if state.get("deadline"):
        left = state["deadline"] - time.time()
        window = f"   launch window remaining: {_fmt(left)}" if left > 0 else "   launch window closed: waiting for the running runs"
    elif state.get("checkpoint_hours"):
        nxt = state["last_ckpt"] + state["checkpoint_hours"] * 3600 - time.time()
        window = f"   continuous; next unit summary in {_fmt(max(0, nxt))}"
    if state.get("stop"):
        window += f"   STOPPING (Coordinator: {state['stop'][:200]})"
    lines = [f"# agentlab status  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", "",
             f"launched: {state['launched']}   finished: {state['finished']}   elapsed: {_fmt(time.time() - state['t0'])}{window}",
             f"backend failures in a row: {state['fails']}", ""]
    if running:
        lines += ["## Running now", ""] + _running_rows(cfg, ledger, running, best) + [""]
    else:
        lines += ["## Running now", "", "(nothing running)", ""]
    latest = coordinator.latest_decision(cfg)
    if latest:
        lines += ["## Coordinator's latest decision  (full log: work/coordinator.md)", "", latest.strip(), ""]
    lines += ["```", ledger.summary_text(25), "```"]
    audited = ledger.experiments(5, "audited")
    if audited:
        lines += ["", "## Waiting for your approval"] + [
            f"- exp #{e['id']} score {e['score']:,}  →  `agentlab approve {e['id']} --by Yang` then `agentlab submit {e['id']}`"
            for e in audited]
    p.write_text("\n".join(lines) + "\n")


def run_rounds(cfg: Config, rounds: int, solvers: int | None = None, audit_threshold: float = 0.0,
               hours: float | None = None, checkpoint_hours: float | None = None) -> None:
    """rounds/hours bound the launches (then running runs finish); with neither, the loop is continuous:
    every slot is refilled the moment it frees up, a unit summary is written every `checkpoint_hours`
    without stopping anything, and the loop ends only when stopped or when the Coordinator declares the
    campaign hopeless with nothing running."""
    adapter = load_adapter(cfg)
    adapter.ensure_ready()
    ledger = Ledger(cfg.ledger_path)
    # A run killed mid-experiment (closed terminal, stopped container) leaves rows marked "running";
    # the thinking roles would treat those as in progress and avoid their ideas forever.
    stale = ledger.db.execute("UPDATE experiments SET status='abandoned', note=COALESCE(note,'') || ' (run interrupted)' "
                              "WHERE status='running'").rowcount
    ledger.db.commit()
    if stale:
        print(f"marked {stale} interrupted experiment(s) as abandoned", flush=True)
    planner.seed_ideas(ledger)
    rng = random.Random()
    solvers = solvers or cfg.solvers
    max_launches = rounds * solvers if rounds else 0
    base_ref = f"{cfg.upstream_remote}/{cfg.upstream_branch}"
    state = {"t0": time.time(), "deadline": time.time() + hours * 3600 if hours else None, "launched": 0, "finished": 0,
             "fails": 0, "pause_until": 0.0, "last_lit": time.time(), "last_decision": 0.0, "empty_decisions": 0,
             "last_ckpt": time.time(), "checkpoint_hours": checkpoint_hours, "stop": None}
    _maybe_trend(cfg, adapter, ledger)
    _maybe_strategize(cfg, adapter, ledger, state, force=True)
    if hours:
        print(f"launching runs for {_fmt(hours * 3600)} (until {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(state['deadline']))}) "
              f"on {solvers} slot(s); running runs finish after that; status in {Path(cfg.runs_dir).parent / 'status.md'}", flush=True)
    elif not rounds:
        print(f"continuous run on {solvers} slot(s): every free slot is refilled at once"
              + (f"; unit summary every {checkpoint_hours:g} h" if checkpoint_hours else "")
              + f"; stop with docker/stop.sh; status in {Path(cfg.runs_dir).parent / 'status.md'}", flush=True)
    pool = ThreadPoolExecutor(max_workers=solvers)
    running: dict[Future, dict] = {}
    finished_since: list[int] = []  # results the Coordinator has not seen yet
    best = None
    try:
        while True:
            now = time.time()
            window_open = ((not max_launches or state["launched"] < max_launches) and (not state["deadline"] or now < state["deadline"])
                           and not state["stop"])
            if state["checkpoint_hours"] and now - state["last_ckpt"] >= state["checkpoint_hours"] * 3600:
                _close_unit(cfg, ledger, state, running, best)  # checkpoint only: nothing stops
                state["last_ckpt"] = now
            free = solvers - len(running)
            if free > 0 and window_open and now >= state["pause_until"]:
                # decide when something came back, when the slots are all idle, or every 15 min anyway
                if finished_since or not running or now - state["last_decision"] > 900:
                    try:
                        best = _refresh_best(cfg, adapter, ledger)
                        replenish(cfg, adapter, ledger, state)
                        _maybe_strategize(cfg, adapter, ledger, state)
                        _decide_and_launch(cfg, adapter, ledger, rng, pool, running, finished_since, free, base_ref, best, state)
                    except KeyboardInterrupt:
                        raise
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()
                        print("  decision crashed; trying again in 5 min", flush=True)
                        state["pause_until"] = time.time() + 300
                    state["last_decision"] = time.time()
                    write_status(cfg, ledger, state, running, best)
            if not running:
                if not window_open:
                    break
                time.sleep(min(300, max(30, state["pause_until"] - time.time())))
                continue
            done, _ = wait(list(running), timeout=60, return_when=FIRST_COMPLETED)
            for f in done:
                r = running.pop(f)
                _on_finished(cfg, adapter, ledger, f, r, audit_threshold, state, finished_since, best)
            write_status(cfg, ledger, state, running, best)
    except KeyboardInterrupt:
        for r in running.values():
            ledger.finish_experiment(r["exp_id"], "abandoned", note="interrupted")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    if state["stop"]:
        print(f"\n===== the Coordinator ended the campaign: {state['stop']} =====", flush=True)
    if hours or checkpoint_hours or state["stop"]:
        _close_unit(cfg, ledger, state)
    print(ledger.summary_text())


def _refresh_best(cfg, adapter, ledger):
    subprocess.run(["git", "fetch", "-q", cfg.upstream_remote, cfg.upstream_branch], cwd=adapter.root, check=False)
    best = adapter.current_best()
    if best:
        ledger.record_best(best["score"], best["toffoli"], best["qubits"], f"upstream:{best['commit']}", verified=True)
        best = dict(best, fast_score=fast_baseline(cfg, adapter, best["commit"]))
    return best


def fast_baseline(cfg, adapter, commit: str) -> int | None:
    """Fast score (cfg.fast_shots) of the upstream head itself, cached per commit. Gains are measured against
    this, not against the official 9024-shot score: the two differ by shot sampling alone (+/-0.002 %), which
    made every unchanged tree look like a small gain or loss."""
    cache = Path(cfg.runs_dir).parent / "fast_baseline.json"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    key = f"{commit}@{cfg.fast_shots}"
    if key in data:
        return data[key]
    wt = Path(cfg.worktrees_dir) / "baseline"
    subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=adapter.root, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), commit], cwd=adapter.root, check=True)
    res = adapter.fast_score(wt, cfg.fast_shots)
    if not res.ok:
        print(f"  [baseline] fast score of upstream {commit} failed: {res.reason}", flush=True)
        return None
    data[key] = res.score
    cache.write_text(json.dumps(data, indent=1))
    print(f"  [baseline] upstream {commit}: fast score {res.score:,} ({cfg.fast_shots} shots); gains are measured against it", flush=True)
    return res.score


def _decide_and_launch(cfg, adapter, ledger, rng, pool, running, finished_since, free, base_ref, best, state) -> None:
    hdr = f"=== decision  launched {state['launched']}  finished {state['finished']}" \
          + (f"  elapsed {_fmt(time.time() - state['t0'])}" if state.get("deadline") else "")
    print(f"\n{hdr}  public best {best['score'] if best else '?'}  free slots {free}  running {[r['exp_id'] for r in running.values()] or 'none'} ===", flush=True)
    seen = list(finished_since)
    finished_since.clear()
    d = coordinator.decide(cfg, adapter, ledger, list(running.values()), seen, free, best, rng)
    if d.get("decision"):
        print(f"  [coordinator] {d['decision'][:600]}", flush=True)
    launches = d.get("launches") or []
    if d.get("hopeless") and not launches:
        reason = d.get("hopeless_reason") or "no reason given"
        if running:
            print(f"  [coordinator] judges the campaign hopeless ({reason}); letting the running runs finish", flush=True)
        state["stop"] = reason
        return
    if not launches:
        state["empty_decisions"] += 1
        if not running and state["empty_decisions"] >= 2:
            print("  coordinator keeps waiting with nothing running; reopening the least-tried exhausted ideas", flush=True)
            for i in sorted(ledger.ideas("exhausted"), key=lambda i: i.tries)[:free]:
                ledger.set_idea_status(i.id, "open")
            picks = planner.bandit_picks(ledger, cfg, rng, free)
            if not picks:
                raise RuntimeError("backlog empty")
            launches = [{"mode": "new", "idea_id": i.id, "parent_exps": [], "instruction": "", "why": "fallback after empty decisions"} for i in picks]
            state["empty_decisions"] = 0
        else:
            print("  [coordinator] launches nothing now (waiting)", flush=True)
            return
    else:
        state["empty_decisions"] = 0
    for l in launches:
        idea_id, brief = coordinator.launch_brief(cfg, ledger, l, best)
        exp_id = ledger.start_experiment(idea_id, "", "", cfg.model or cfg.backend, note=brief.splitlines()[0][:200])
        run_dir = Path(cfg.runs_dir) / f"exp{exp_id}"
        fut = pool.submit(solver.run_solver, cfg, adapter, ledger, idea_id, brief, base_ref, best, exp_id, l["mode"], l["parent_exps"])
        running[fut] = {"exp_id": exp_id, "idea_id": idea_id, "mode": l["mode"], "parents": l["parent_exps"], "t0": time.time(),
                        "log": str(run_dir / "agent.log")}
        state["launched"] += 1
        what = f"idea #{idea_id}" if l["mode"] == "new" else f"{l['mode']} {l['parent_exps']}"
        print(f"  launched exp #{exp_id}: {what}", flush=True)


def _on_finished(cfg, adapter, ledger, fut: Future, r: dict, audit_threshold, state, finished_since, best) -> None:
    exp_id = r["exp_id"]
    try:
        fut.result()
    except Exception as e:  # noqa: BLE001
        print(f"  exp #{exp_id} crashed: {e}", flush=True)
        traceback.print_exc()
        if ledger.experiment(exp_id)["status"] == "running":
            ledger.finish_experiment(exp_id, "failed", note=f"crashed: {e}"[:300])
    e = ledger.experiment(exp_id)
    state["finished"] += 1
    print(f"  exp #{exp_id} ({r['mode']}) {e['status']} score={e['score']} gain={e['gain']} after {_fmt(time.time() - r['t0'])}", flush=True)
    if e["status"] == "backend_failed":
        state["fails"] += 1
        wait_s = min(2 * 3600, 20 * 60 * 2 ** (state["fails"] - 1))
        state["pause_until"] = time.time() + wait_s
        print(f"  model backend failed fast {state['fails']}x in a row (usage limit?); no new launches for {wait_s // 60} min", flush=True)
        return
    state["fails"] = 0
    finished_since.append(exp_id)
    try:  # digest the run into work/lessons.md before the Coordinator's next decision
        from .roles.lessons import learn
        learn(cfg, adapter, ledger, [exp_id])
    except Exception as ex:  # noqa: BLE001
        print(f"  [lessons] failed: {ex}", flush=True)
    if e["status"] == "scored" and e["gain"] is not None and e["gain"] > audit_threshold:
        print(f"  -> beats the bar on the fast score; auditing #{exp_id} (full official run)", flush=True)
        try:
            v = auditor.audit(cfg, adapter, ledger, exp_id)
            print(f"  audit #{exp_id}: {'OK' if 'fail' not in v else v['fail']} full={v.get('full', {}).get('score')} beats_best={v.get('beats_best')}", flush=True)
            if v.get("beats_best"):
                print(f"  *** exp #{exp_id} beats the public bar; waiting for `agentlab approve {exp_id} --by Yang` ***", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  audit #{exp_id} crashed: {ex}", flush=True)
    # retire ideas that keep failing
    for i in ledger.ideas("open"):
        if i.tries >= 4 and i.gain_sum / i.tries < -0.5:
            ledger.set_idea_status(i.id, "exhausted")


def _close_unit(cfg: Config, ledger: Ledger, state: dict, running: dict | None = None, best=None) -> None:
    """End of a unit (or a checkpoint in a continuous run): keep its summary under work/units/ and print it."""
    write_status(cfg, ledger, state, running, best)
    units = Path(cfg.runs_dir).parent / "units"
    units.mkdir(parents=True, exist_ok=True)
    n = len(list(units.glob("unit-*.md"))) + 1
    text = (Path(cfg.runs_dir).parent / "status.md").read_text()
    (units / f"unit-{n:02d}-{time.strftime('%Y%m%d-%H%M', time.gmtime())}.md").write_text(text)
    print(f"\n===== unit {n} finished =====\n{text}", flush=True)


def _maybe_trend(cfg, adapter, ledger) -> None:
    """Leaderboard trend in a background thread when upstream merged something since the last analysis
    (measuring ~30 commits takes about an hour; the Strategist's next revision reads the result)."""
    import threading
    from .roles.trend import analyse, stale
    if cfg.backend == "dry" or not stale(cfg, adapter):
        return
    def go():
        try:
            analyse(cfg, adapter, ledger)
        except Exception as e:  # noqa: BLE001
            print(f"  [trend] failed: {e}", flush=True)
    print("  [trend] leaderboard history changed: measuring and analysing it in the background", flush=True)
    threading.Thread(target=go, daemon=True, name="trend").start()


def _maybe_strategize(cfg, adapter, ledger, state, force: bool = False, every: int = 6) -> None:
    """Formulate/decompose/allocate before the loop, then after every `every` finished experiments."""
    from .roles.strategist import problem_path, strategize
    if cfg.backend == "dry":
        return
    n_exp = len(ledger.experiments(10000))
    due = force or n_exp - state.get("last_strat_exp", 0) >= every  # every unit starts with a fresh revision
    if not due:
        return
    try:
        print(f"  [strategist] revising the problem statement ({n_exp} experiments so far)", flush=True)
        strategize(cfg, adapter, ledger)
    except Exception as e:  # noqa: BLE001
        print(f"  [strategist] failed: {e}")
    state["last_strat_exp"] = n_exp

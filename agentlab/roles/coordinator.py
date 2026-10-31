"""Coordinator: the main agent of the planning block.

The Strategist writes the problem statement (formulate, decompose, allocate); the Coordinator
runs the loop on top of it. Whenever a solver slot is free it reads what is running (with the
live scores each run has logged so far), what came back since its last decision, the backlog
and the bandit's ranking, lists its options and decides what the slot does next:

  new       start an open idea in a fresh worktree at the upstream head
  continue  pick up a finished run from its own final tree (branch agentlab/exp<N>), with its notes
  combine   cherry-pick several finished branches onto the upstream head and score the combination
  retire    close an idea with a reason
  wait      launch nothing until a running run settles the question

Every decision is appended to work/coordinator.md in plain language: that file is how the
Coordinator talks to the human. Without a model backend (dry runs) or when the model call fails,
the bandit's picks are used so the loop never stalls.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

from ..adapters.base import CompetitionAdapter
from ..config import Config
from ..ledger import Ledger
from ..llm import make_backend
from .planner import bandit_picks, planner_brief

SCHEMA = {
    "type": "object",
    "properties": {
        "situation": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "decision": {"type": "string"},
        "launches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["new", "continue", "combine"]},
                    "idea_id": {"type": ["integer", "null"]},
                    "parent_exps": {"type": "array", "items": {"type": "integer"}},
                    "instruction": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["mode", "idea_id", "parent_exps", "instruction", "why"],
                "additionalProperties": False,
            },
        },
        "retire": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"idea_id": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["idea_id", "reason"],
                "additionalProperties": False,
            },
        },
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["situation", "options", "decision", "launches", "retire", "questions"],
    "additionalProperties": False,
}


def log_path(cfg: Config) -> Path:
    return Path(cfg.runs_dir).parent / "coordinator.md"


def _skill(cfg: Config, name: str) -> str:
    p = Path(cfg.skills_dir) / name / "SKILL.md"
    return p.read_text() if p.exists() else ""


def live_progress(worktree: str | None, log: str | None) -> dict:
    """What a running solver has done so far: every fast_score.sh call appends to .agentlab/scores.jsonl
    in its worktree, and the agent log's mtime says when it last did anything."""
    out = {"attempts": 0, "scored": 0, "best_score": None, "last_score_min": None, "idle_min": None}
    if worktree:
        p = Path(worktree) / ".agentlab" / "scores.jsonl"
        if p.exists():
            last_ts = None
            for line in p.read_text(errors="replace").splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out["attempts"] += 1
                last_ts = d.get("ts", last_ts)
                if d.get("ok") and d.get("score"):
                    out["scored"] += 1
                    out["best_score"] = min(out["best_score"] or d["score"], d["score"])
            if last_ts:
                out["last_score_min"] = int((time.time() - last_ts) // 60)
    if log and Path(log).exists():
        out["idle_min"] = int((time.time() - Path(log).stat().st_mtime) // 60)
    return out


def branch_exists(adapter: CompetitionAdapter, exp_id: int) -> bool:
    r = subprocess.run(["git", "rev-parse", "--verify", "-q", f"agentlab/exp{exp_id}"], cwd=adapter.root, capture_output=True)
    return r.returncode == 0


def experiment_summary(cfg: Config, ledger: Ledger, exp_id: int, variants: int = 10) -> dict:
    """One finished experiment as the Coordinator (and a continuation brief) sees it."""
    e = ledger.experiment(exp_id)
    if not e:
        return {"exp": exp_id, "status": "unknown"}
    d = json.loads(e["details"] or "{}")
    rep = d.get("report") or {}
    idea = ledger.idea(e["idea_id"]) if e["idea_id"] else None
    notes = Path(cfg.runs_dir).parent / "notes" / f"exp{exp_id}"
    out = {"exp": exp_id, "idea": e["idea_id"], "title": idea.title if idea else None,
           "sub": getattr(idea, "subproblem", None) if idea else None, "kind": getattr(idea, "kind", None) if idea else None,
           "mode": e["mode"] or "new", "parents": json.loads(e["parents"] or "[]"), "status": e["status"],
           "score": e["score"], "gain": e["gain"], "toffoli": e["toffoli"], "qubits": e["qubits"],
           "idea_status": rep.get("idea_status"), "what": (rep.get("what_changed") or "")[:400],
           "why": (rep.get("why_it_worked_or_not") or "")[:400], "next_ideas": (rep.get("next_ideas") or [])[:5],
           "variants": (rep.get("variants") or [])[:variants], "notes_dir": str(notes) if notes.exists() else None,
           "minutes": int(((e["finished"] or e["started"]) - e["started"]) // 60)}
    if notes.exists():
        out["notes_excerpt"] = "\n".join(f.read_text(errors="replace")[:1200] for f in sorted(notes.glob("*"))[:3])
    return out


def _context(cfg, adapter, ledger, running: list[dict], finished: list[int], free_slots: int, best) -> str:
    from .strategist import problem_path
    pj = problem_path(cfg).with_suffix(".json")
    problem = ""
    if pj.exists():
        data = json.loads(pj.read_text())
        rows = [f"- {sp['name']}: share {sp['toffoli_share']}, peak {sp['touches_peak_width']}, ceiling {sp['ceiling']}, {sp['status']}: {sp['why'][:200]}"
                for sp in data["subproblems"]]
        problem = data["formulation"][:2000] + "\n\nSUB-PROBLEMS:\n" + "\n".join(rows)
    running_ids = {r["idea_id"] for r in running if r.get("idea_id")}
    ideas = [{"id": i.id, "title": i.title, "sub": i.subproblem, "kind": i.kind, "prior": i.prior, "tries": i.tries,
              "mean_gain": round(i.gain_sum / i.tries, 5) if i.tries else None, "best_gain": round(i.best_gain, 5),
              "running": i.id in running_ids}
             for i in ledger.ideas("open")]
    hint = [{"id": i.id, "title": i.title[:60]} for i in bandit_picks(ledger, cfg, random.Random(), max(free_slots, 1), running_ids)]
    run_rows = []
    for r in running:
        e = ledger.experiment(r["exp_id"])
        prog = live_progress(e["worktree"] if e else None, r.get("log"))
        idea = ledger.idea(r["idea_id"]) if r.get("idea_id") else None
        run_rows.append({"exp": r["exp_id"], "idea": r.get("idea_id"), "title": idea.title if idea else None,
                         "sub": getattr(idea, "subproblem", None) if idea else None, "mode": r["mode"], "parents": r["parents"],
                         "elapsed_min": int((time.time() - r["t0"]) // 60), **prog,
                         "best_gain_so_far": round(((best.get("fast_score") or best["score"]) - prog["best_score"])
                                                   / (best.get("fast_score") or best["score"]), 6)
                         if best and prog["best_score"] else None})
    fin = [experiment_summary(cfg, ledger, x) for x in finished]
    recent = [experiment_summary(cfg, ledger, e["id"], variants=3) for e in ledger.experiments(8) if e["id"] not in finished
              and e["status"] not in ("running",)]
    for r in recent:
        r.pop("notes_excerpt", None)
    log = log_path(cfg)
    prev = log.read_text()[-4000:] if log.exists() else "(first decision)"
    from .lessons import lessons_text
    from .trend import trend_text
    lessons = lessons_text(cfg) or "(none yet)"
    trend = trend_text(cfg, 3500) or "(not analysed yet)"
    return f"""LEADERBOARD TREND (synthesis of the merged-submission history; see work/trend.md):
{trend}

LESSONS SO FAR (work/lessons.md, revised after every run; what the system has learned):
{lessons}

PUBLIC BAR: {dict(best) if best else '?'}   (gain = (fast_score of the head - fast score of a run) / fast_score of the head, same shots: an unchanged circuit gains exactly 0, any nonzero gain is a real circuit change)

PROBLEM STATEMENT (Strategist):
{problem or '(none yet)'}

FREE SLOTS: {free_slots}   RUNNING NOW: {json.dumps(run_rows)}

FINISHED SINCE YOUR LAST DECISION: {json.dumps(fin)[:14000]}

OTHER RECENT EXPERIMENTS: {json.dumps(recent)[:6000]}

OPEN BACKLOG: {json.dumps(ideas)[:9000]}

BANDIT'S SUGGESTION (a hint only): {json.dumps(hint)}

YOUR PREVIOUS DECISIONS (tail of work/coordinator.md):
{prev}"""


def _validate(launch: dict, ledger: Ledger, adapter, running: list[dict], taken: set) -> str | None:
    """Return why a launch cannot happen, or None."""
    mode = launch.get("mode")
    parents = [int(p) for p in launch.get("parent_exps") or []]
    if mode == "new":
        iid = launch.get("idea_id")
        if not iid:
            return "new launch without idea_id"
        if iid in taken or any(r.get("idea_id") == iid for r in running):
            return f"idea {iid} already running"
        try:
            i = ledger.idea(iid)
        except Exception:  # noqa: BLE001
            return f"idea {iid} does not exist"
        if i.status != "open":
            return f"idea {iid} is {i.status}"
        return None
    if mode == "continue":
        if len(parents) != 1:
            return "continue needs exactly one parent experiment"
    elif mode == "combine":
        if len(parents) < 2:
            return "combine needs at least two parent experiments"
    else:
        return f"unknown mode {mode}"
    for p in parents:
        e = ledger.experiment(p)
        if not e:
            return f"experiment {p} does not exist"
        if e["status"] == "running" or any(r["exp_id"] == p for r in running):
            return f"experiment {p} is still running"
        if mode == "continue" and any(r["mode"] == "continue" and r["parents"] == [p] for r in running):
            return f"experiment {p} is already being continued"
        if not branch_exists(adapter, p):
            return f"experiment {p} left no branch agentlab/exp{p}"
        if mode == "continue" and ("continue", p) in taken:
            return f"experiment {p} already continued in this decision"
    return None


def decide(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger, running: list[dict], finished: list[int],
           free_slots: int, best: dict | None, rng: random.Random | None = None) -> dict:
    """Ask the Coordinator what the free slots do. Returns {"launches": [...], "decision": str, "fallback": bool}."""
    rng = rng or random.Random()
    running_ids = {r["idea_id"] for r in running if r.get("idea_id")}
    data: dict = {}
    if cfg.backend != "dry":
        prompt = f"""{_skill(cfg, 'coordinator')}

{_context(cfg, adapter, ledger, running, finished, free_slots, best)}

Decide what the {free_slots} free slot(s) do now. List the options you considered (one line each), then the decision
and why, as a note the human will read. At most {free_slots} launches; a launch is "new" (idea_id from the open
backlog, not one already running), "continue" (parent_exps = [one finished experiment]) or "combine" (parent_exps =
two or more finished experiments whose branches should be cherry-picked together). "instruction" is what the executor
does first and what would refute the direction; "why" is for the human. Retire ideas only with a reason. Up to two
questions for the literature. Return JSON matching the schema."""
        try:
            data = json.loads(make_backend(cfg).complete(prompt, schema=SCHEMA, timeout_s=1500))
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] model call failed ({e}); using the bandit's picks", flush=True)
            data = {}
    launches, skipped, taken = [], [], set()
    for l in data.get("launches") or []:
        why_not = _validate(l, ledger, adapter, running, taken)
        if why_not or len(launches) >= free_slots:
            skipped.append(f"{l.get('mode')} idea={l.get('idea_id')} parents={l.get('parent_exps')}: {why_not or 'no slot'}")
            continue
        l["parent_exps"] = [int(p) for p in l.get("parent_exps") or []]
        if l["mode"] == "new":
            taken.add(l["idea_id"])
        elif l["mode"] == "continue":
            taken.add(("continue", l["parent_exps"][0]))
        launches.append(l)
    fallback = not data
    if fallback and free_slots:
        for i in bandit_picks(ledger, cfg, rng, free_slots, running_ids):
            launches.append({"mode": "new", "idea_id": i.id, "parent_exps": [], "instruction": "", "why": "bandit pick (no coordinator answer)"})
    for r in data.get("retire") or []:
        try:
            i = ledger.idea(int(r["idea_id"]))
        except Exception:  # noqa: BLE001
            continue
        if i.status == "open" and i.id not in running_ids:
            ledger.set_idea_status(i.id, "exhausted", review=f"coordinator: {r['reason']}")
    for q in (data.get("questions") or [])[:2]:
        try:
            from .lit_scout import ask
            ask(cfg, ledger, q)
        except Exception as e:  # noqa: BLE001
            print(f"[coordinator] ask failed: {e}")
    data["launches"] = launches
    data["skipped"] = skipped
    data["fallback"] = fallback
    _append_log(cfg, ledger, data, running, finished, free_slots)
    return data


def _append_log(cfg, ledger, data: dict, running, finished, free_slots) -> None:
    p = log_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    lines = [f"## {stamp}  free slots {free_slots}; running {[r['exp_id'] for r in running] or 'none'}; "
             f"finished since last decision {finished or 'none'}", ""]
    if data.get("fallback"):
        lines.append("_No Coordinator answer (dry run or model failure): the bandit chose._")
    if data.get("situation"):
        lines += [f"**Situation.** {data['situation'].strip()}", ""]
    if data.get("options"):
        lines += ["**Options considered.**"] + [f"- {o}" for o in data["options"]] + [""]
    if data.get("decision"):
        lines += [f"**Decision.** {data['decision'].strip()}", ""]
    for l in data.get("launches") or []:
        what = (f"new run on idea #{l.get('idea_id')}" if l["mode"] == "new"
                else f"{l['mode']} exps {l.get('parent_exps')}")
        if l["mode"] == "new" and l.get("idea_id"):
            what += f" ({ledger.idea(l['idea_id']).title[:70]})"
        lines.append(f"- launch: {what}. {l.get('why', '')} Instruction: {l.get('instruction', '')[:300]}")
    for r in data.get("retire") or []:
        lines.append(f"- retire idea #{r.get('idea_id')}: {r.get('reason', '')}")
    for s in data.get("skipped") or []:
        lines.append(f"- not launched ({s})")
    if not data.get("launches"):
        lines.append("- launches: none (waiting)")
    for q in data.get("questions") or []:
        lines.append(f"- asked the literature: {q}")
    with p.open("a") as f:
        f.write("\n".join(lines) + "\n\n")


def latest_decision(cfg: Config) -> str:
    p = log_path(cfg)
    if not p.exists():
        return ""
    txt = p.read_text()
    return txt[txt.rfind("\n## ") + 1:] if "\n## " in txt else txt


def launch_brief(cfg: Config, ledger: Ledger, launch: dict, best: dict | None) -> tuple[int | None, str]:
    """The executor's brief for one launch; returns (idea_id for the ledger row, brief)."""
    from .strategist import formulation_excerpt
    mode = launch["mode"]
    instr = ""
    if launch.get("instruction"):
        instr = f"\nCOORDINATOR'S INSTRUCTION FOR THIS RUN:\n{launch['instruction']}"
    if launch.get("why"):
        instr += f"\nWhy this run now: {launch['why']}"
    if mode == "new":
        idea = ledger.idea(launch["idea_id"])
        return idea.id, planner_brief(ledger, idea, best) + "\n" + instr
    bar = (f"Public best to beat: score {best['score']:,} = toffoli {best['toffoli']:.0f} x qubits {best['qubits']}."
           if best else "")
    if mode == "continue":
        p = launch["parent_exps"][0]
        s = experiment_summary(cfg, ledger, p, variants=15)
        lines = [f"CONTINUE EXPERIMENT #{p}" + (f" (idea #{s['idea']}: {s['title']})" if s.get("idea") else "") + " FROM ITS OWN TREE",
                 f"Your worktree is that experiment's final tree (its branch agentlab/exp{p}); its commits are in place, do not redo them. "
                 "Measure the baseline of this tree first, then take the next step, not the previous one.", "",
                 f"Where it stopped: status {s['status']}, score {s['score']}, gain {s['gain']}, after {s['minutes']} min.",
                 f"What it changed: {s['what']}", f"Why it worked or not: {s['why']}",
                 f"Its own next ideas: {json.dumps(s['next_ideas'])}", f"Variants it scored: {json.dumps(s['variants'])}"]
        if s.get("notes_dir"):
            lines.append(f"Its notes (read them first): {s['notes_dir']}")
        lines += ["", "This is an OPTIMISATION task: the run only counts if the submitted circuit changes versus the tree you start from."]
        exc = formulation_excerpt(cfg, s.get("sub"))
        if exc:
            lines += ["", exc]
        lines += [bar, instr]
        return s.get("idea"), "\n".join(lines)
    # combine
    parents = launch["parent_exps"]
    lines = [f"COMBINE EXPERIMENTS {', '.join('#' + str(p) for p in parents)}",
             "Your worktree starts from the upstream head. Each experiment left its changes on its own branch. Bring them in "
             "with cherry-pick, in this order, resolve conflicts, then score:"]
    for p in parents:
        e = ledger.experiment(p)
        lines.append(f"    git cherry-pick {e['base_commit']}..agentlab/exp{p}")
    lines += ["Score after each cherry-pick (that gives every branch's contribution in order) and again for the full "
              "combination. The combination counts only if it beats the best single branch; gains may not add, say why if "
              "they do not. If two branches conflict, keep the one with the larger measured gain and port the other's change "
              "by hand.", ""]
    for p in parents:
        s = experiment_summary(cfg, ledger, p, variants=5)
        lines += [f"exp #{p}" + (f" (idea #{s['idea']}: {s['title']}; sub-problem {s['sub']})" if s.get("idea") else ""),
                  f"  status {s['status']}, score {s['score']}, gain {s['gain']}; changed: {s['what']}; why: {s['why']}"
                  + (f"; notes: {s['notes_dir']}" if s.get("notes_dir") else "")]
    exc = formulation_excerpt(cfg, None)
    if exc:
        lines += ["", exc]
    lines += [bar, instr]
    return None, "\n".join(lines)

"""Strategist: formulate, decompose, allocate.

Runs before the loop and again every few experiments. Reads the measured profile, the
backlog with its statistics, recent reports and the public bar, and writes work/problem.md:

1. Formulation: the task as an optimisation problem, minimise T(C) x W(C), with T split by
   stage from the profile and W as one peak moment.
2. Decomposition: sub-problems with ceilings (a stage holding 12 % of T cannot gain more
   than 12 %), each with hypotheses, predicted gains and a cheap refutation test.
3. Allocation: priors of backlog ideas are set to min(predicted gain, ceiling); dead
   branches are pruned with a reason; new hypotheses enter the backlog tagged by
   sub-problem. The bandit then schedules solver runs across the leaves.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..adapters.base import CompetitionAdapter
from ..config import Config
from ..ledger import Ledger
from ..llm import make_backend

SCHEMA = {
    "type": "object",
    "properties": {
        "formulation": {"type": "string"},
        "subproblems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "toffoli_share": {"type": ["number", "null"]},
                    "touches_peak_width": {"type": "boolean"},
                    "ceiling": {"type": "number"},
                    "status": {"type": "string", "enum": ["open", "closed"]},
                    "why": {"type": "string"},
                    "hypotheses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "kind": {"type": "string", "enum": ["measurement", "optimisation"]},
                                "existing_idea_id": {"type": ["integer", "null"]},
                                "predicted_gain": {"type": "number"},
                                "refutation_test": {"type": "string"},
                                "first_experiment": {"type": "string"},
                            },
                            "required": ["title", "kind", "existing_idea_id", "predicted_gain", "refutation_test", "first_experiment"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "toffoli_share", "touches_peak_width", "ceiling", "status", "why", "hypotheses"],
                "additionalProperties": False,
            },
        },
        "pruned_ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"idea_id": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["idea_id", "reason"],
                "additionalProperties": False,
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["formulation", "subproblems", "pruned_ideas", "open_questions"],
    "additionalProperties": False,
}


def problem_path(cfg: Config) -> Path:
    return Path(cfg.runs_dir).parent / "problem.md"


def _skill(cfg: Config, name: str) -> str:
    p = Path(cfg.skills_dir) / name / "SKILL.md"
    return p.read_text() if p.exists() else ""


def _context(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger) -> str:
    root = Path(adapter.root)
    from .planner import profile_text
    prof = profile_text(cfg, adapter)
    listing = subprocess.run(["git", "ls-files", "src/point_add"], cwd=root, capture_output=True, text=True).stdout
    files = " ".join(Path(f).name for f in listing.splitlines() if f.endswith(".rs"))[:3000]
    ideas = [{"id": i.id, "status": i.status, "title": i.title, "prior": i.prior, "tries": i.tries,
              "mean_gain": round(i.gain_sum / i.tries, 5) if i.tries else None, "sub": getattr(i, "subproblem", None)}
             for i in ledger.ideas(None) if i.status != "rejected"]
    reports = []
    for e in ledger.experiments(15):
        d = json.loads(e["details"] or "{}").get("report") or {}
        reports.append({"exp": e["id"], "idea": e["idea_id"], "status": e["status"], "gain": e["gain"],
                        "what": (d.get("what_changed") or "")[:250], "why": (d.get("why_it_worked_or_not") or "")[:250],
                        "variants": (d.get("variants") or [])[:8]})
    notes_dir = Path(cfg.runs_dir).parent / "notes"
    notes = ""
    if notes_dir.exists():
        for d in sorted(notes_dir.iterdir(), key=lambda d: d.stat().st_mtime)[-4:]:
            for f in sorted(d.glob("*")):
                notes += f"\n--- {d.name}/{f.name} ---\n" + f.read_text(errors="replace")[:3000]
    best = ledger.best()
    prev = problem_path(cfg)
    prev_txt = prev.read_text()[:6000] if prev.exists() else "(none yet)"
    from .lessons import lessons_text
    from .trend import trend_text
    return f"""LEADERBOARD TREND (work/trend.md: measured history of merged submissions and its synthesis; the head is the baseline to beat, not the only framework):
{trend_text(cfg) or '(not analysed yet)'}

LESSONS SO FAR (work/lessons.md; verified facts, dead ends, what moved the numbers):
{lessons_text(cfg) or '(none yet)'}

PUBLIC BAR: {dict(best) if best else '?'}

MEASURED PROFILE:
{prof}

FILES IN src/point_add: {files}

BACKLOG (id, status, title, prior, tries, mean gain, sub-problem): {json.dumps(ideas)[:9000]}

RECENT EXPERIMENTS (with the variants each solver scored): {json.dumps(reports)[:10000]}

NOTES LEFT BY RECENT SOLVERS (measured, from their worktrees): {notes[:12000] or '(none)'}

PREVIOUS PROBLEM STATEMENT (revise it; keep what still holds):
{prev_txt}"""


def render(data: dict) -> str:
    lines = [f"# Problem statement  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", "",
             "## Formulation", data["formulation"].strip(), "", "## Decomposition", "",
             "| sub-problem | share of T | peak? | ceiling | status |", "|---|---|---|---|---|"]
    for sp in data["subproblems"]:
        share = f"{sp['toffoli_share'] * 100:.0f} %" if sp.get("toffoli_share") is not None else "?"
        lines.append(f"| {sp['name']} | {share} | {'yes' if sp['touches_peak_width'] else 'no'} | {sp['ceiling'] * 100:.1f} % | {sp['status']} |")
    for sp in data["subproblems"]:
        lines += ["", f"### {sp['name']}", sp["why"].strip()]
        for h in sp["hypotheses"]:
            ref = f" (idea #{h['existing_idea_id']})" if h.get("existing_idea_id") else ""
            ref += " [measurement]" if h.get("kind") == "measurement" else ""
            lines.append(f"- **{h['title']}**{ref}: predicted {h['predicted_gain'] * 100:.2f} %. "
                         f"Refute: {h['refutation_test']} First experiment: {h['first_experiment']}")
    if data.get("pruned_ideas"):
        lines += ["", "## Pruned"] + [f"- idea #{p['idea_id']}: {p['reason']}" for p in data["pruned_ideas"]]
    if data.get("open_questions"):
        lines += ["", "## Open questions for the literature"] + [f"- {q}" for q in data["open_questions"]]
    return "\n".join(lines) + "\n"


def _reading_worktree(cfg: Config, adapter: CompetitionAdapter) -> Path:
    """A throwaway checkout of the upstream head the Strategist may read (and cannot harm)."""
    wt = Path(cfg.worktrees_dir) / "strategist"
    root = adapter.root
    subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=root, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), f"{cfg.upstream_remote}/{cfg.upstream_branch}"],
                   cwd=root, check=True)
    return wt


def strategize(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger) -> dict:
    backend = make_backend(cfg)
    prompt = f"""{_skill(cfg, 'strategist')}

{_context(cfg, adapter, ledger)}

Your working directory is a throwaway checkout of the current circuit (src/point_add); read any file you need
(rules in README/harness docs, the profile notes, the primitives), but change nothing: your only output is the JSON.
Produce the problem statement: formulation, decomposition into sub-problems with honest ceilings, hypotheses per
sub-problem (reference existing backlog ideas by id where they fit; add new ones only with a predicted gain and a
refutation test), ideas to prune with reasons, and up to 3 open questions for the literature.
Return JSON matching the schema."""
    raw = backend.complete(prompt, schema=SCHEMA, timeout_s=1800, cwd=_reading_worktree(cfg, adapter))
    data = json.loads(raw)
    stats = {"subproblems": len(data["subproblems"]), "updated": 0, "added": 0, "pruned": 0}
    existing = {i.id: i for i in ledger.ideas(None)}
    for sp in data["subproblems"]:
        ceiling = max(0.0, min(1.0, float(sp["ceiling"])))
        for h in sp["hypotheses"]:
            prior = round(max(0.001, min(float(h["predicted_gain"]), ceiling, 0.2)), 4)
            if h.get("kind") == "measurement":
                prior = 0.05  # measurements run early and once: after one try their mean gain (~0) takes over
            iid = h.get("existing_idea_id")
            if iid in existing:
                i = existing[iid]
                if sp["status"] == "closed":
                    continue
                ledger.set_idea_status(i.id, "open" if i.status in ("open", "candidate") else i.status,
                                       prior=prior if i.tries == 0 else None)
                ledger.set_idea_subproblem(i.id, sp["name"], h.get("kind"))
                stats["updated"] += 1
            elif sp["status"] == "open":
                why = (f"Sub-problem: {sp['name']} (ceiling {ceiling * 100:.1f} %)\nPredicted gain: {h['predicted_gain'] * 100:.2f} %\n"
                       f"Refutation test: {h['refutation_test']}\nFirst experiment: {h['first_experiment']}")
                before = len(ledger.ideas(None))
                new_id = ledger.add_idea(h["title"][:120], why, "strategist", prior=prior, status="open")
                if len(ledger.ideas(None)) > before:
                    ledger.set_idea_subproblem(new_id, sp["name"], h.get("kind"))
                    stats["added"] += 1
    for p in data.get("pruned_ideas", []):
        i = existing.get(p["idea_id"])
        if i and i.status == "open":
            ledger.set_idea_status(i.id, "exhausted", review=f"strategist: {p['reason']}")
            stats["pruned"] += 1
    problem_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    problem_path(cfg).write_text(render(data))
    (problem_path(cfg).with_suffix(".json")).write_text(json.dumps(data, indent=2))
    from .lit_scout import ask
    for q in (data.get("open_questions") or [])[:3]:
        try:
            ask(cfg, ledger, q)
        except Exception as e:  # noqa: BLE001
            print(f"[strategist] ask failed: {e}")
    subprocess.run(["git", "worktree", "remove", "--force", str(Path(cfg.worktrees_dir) / "strategist")],
                   cwd=adapter.root, capture_output=True)
    print(f"[strategist] {stats}; wrote {problem_path(cfg)}", flush=True)
    return stats


def formulation_excerpt(cfg: Config, subproblem: str | None) -> str:
    """What a solver gets from the problem statement: the formulation and its own sub-problem."""
    pj = problem_path(cfg).with_suffix(".json")
    if not pj.exists():
        return ""
    data = json.loads(pj.read_text())
    out = ["PROBLEM FORMULATION (from the Strategist):", data["formulation"].strip()[:2500]]
    for sp in data["subproblems"]:
        if subproblem and sp["name"] == subproblem:
            share = f"{sp['toffoli_share'] * 100:.0f} % of Toffolis" if sp.get("toffoli_share") is not None else "share unknown"
            out += ["", f"YOUR SUB-PROBLEM: {sp['name']} ({share}; ceiling {sp['ceiling'] * 100:.1f} %"
                    f"{'; touches the peak width' if sp['touches_peak_width'] else ''})", sp["why"].strip()[:1500]]
    return "\n".join(out)

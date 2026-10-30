"""Lessons: the system's digested experience.

After every finished run the Coordinator rewrites work/lessons.md: verified facts, dead ends with
their reason, what moved the numbers, rules of thumb. The ledger and the notes are raw material
and get cut to the last few runs in every prompt; this file is the standing memory that every
later brief (Strategist, Coordinator, executors) starts with.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import Config
from ..ledger import Ledger
from ..llm import make_backend

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "statement": {"type": "string"}, "evidence": {"type": "string"}},
            "required": ["statement", "evidence"], "additionalProperties": False}},
        "dead_ends": {"type": "array", "items": {"type": "object", "properties": {
            "what": {"type": "string"}, "why": {"type": "string"}, "experiments": {"type": "array", "items": {"type": "integer"}}},
            "required": ["what", "why", "experiments"], "additionalProperties": False}},
        "what_moved": {"type": "array", "items": {"type": "object", "properties": {
            "change": {"type": "string"}, "effect": {"type": "string"}, "experiments": {"type": "array", "items": {"type": "integer"}}},
            "required": ["change", "effect", "experiments"], "additionalProperties": False}},
        "rules": {"type": "array", "items": {"type": "string"}},
        "changed": {"type": "string"},
    },
    "required": ["facts", "dead_ends", "what_moved", "rules", "changed"],
    "additionalProperties": False,
}


def lessons_path(cfg: Config) -> Path:
    return Path(cfg.runs_dir).parent / "lessons.md"


def lessons_text(cfg: Config, limit: int = 7000) -> str:
    p = lessons_path(cfg)
    return p.read_text()[:limit] if p.exists() else ""


def render(data: dict, exp_ids: list[int]) -> str:
    lines = [f"# Lessons  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; last revised after exp {exp_ids})", ""]
    if data.get("changed"):
        lines += [f"_Last revision: {data['changed'].strip()}_", ""]
    lines += ["## Verified facts"] + [f"- {f['statement']} (evidence: {f['evidence']})" for f in data["facts"]]
    lines += ["", "## Dead ends"] + [f"- {d['what']}: {d['why']} (exp {d['experiments']})" for d in data["dead_ends"]]
    lines += ["", "## What moved the numbers"] + [f"- {m['change']}: {m['effect']} (exp {m['experiments']})" for m in data["what_moved"]]
    lines += ["", "## Rules of thumb"] + [f"- {r}" for r in data["rules"]]
    return "\n".join(lines) + "\n"


def learn(cfg: Config, adapter, ledger: Ledger, exp_ids: list[int]) -> dict | None:
    """Revise work/lessons.md with what the given finished experiments taught. Returns the data or None."""
    from .coordinator import experiment_summary
    from .strategist import problem_path
    if cfg.backend == "dry" or not exp_ids:
        return None
    skill = Path(cfg.skills_dir) / "lessons" / "SKILL.md"
    runs = [experiment_summary(cfg, ledger, x, variants=30) for x in exp_ids]
    pj = problem_path(cfg).with_suffix(".json")
    formulation = json.loads(pj.read_text())["formulation"][:1500] if pj.exists() else ""
    prev = lessons_text(cfg, 9000) or "(no lessons yet: start the document)"
    prompt = f"""{skill.read_text() if skill.exists() else ''}

PROBLEM (Strategist's formulation): {formulation}

CURRENT LESSONS (revise this; keep what holds, merge the new evidence in):
{prev}

RUNS THAT JUST FINISHED (reports, every scored variant, notes): {json.dumps(runs)[:16000]}

Rewrite the lessons document as JSON matching the schema: facts, dead_ends, what_moved, rules, and one sentence
in "changed" saying what these runs changed in it (or "nothing new" if they confirmed what was there)."""
    data = json.loads(make_backend(cfg).complete(prompt, schema=SCHEMA, timeout_s=900))
    lessons_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    lessons_path(cfg).write_text(render(data, exp_ids))
    print(f"[lessons] revised after exp {exp_ids}: {data.get('changed', '')[:200]}", flush=True)
    return data

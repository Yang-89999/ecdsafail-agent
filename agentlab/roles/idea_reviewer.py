"""Idea Reviewer: scores candidate ideas and decides which reach the solvers.

Inputs per candidate: the idea, the current profile of the circuit (if the profiling
run has produced src/point_add/memory/profile.md), the list of ideas already tried or
retired, and a listing of the editable folder so the model can judge novelty.

Output per candidate: relevance, novelty, feasibility and expected gain in [0, 1], a
verdict, and a one-paragraph reason. Accepted ideas become `open` with a prior derived
from the scores; rejected ones keep the reason in `ideas.review` for the human to see.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..adapters.base import CompetitionAdapter
from ..config import Config
from ..ledger import Ledger
from ..llm import make_backend

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "integer"},
                    "relevance": {"type": "number"},
                    "novelty": {"type": "number"},
                    "feasibility": {"type": "number"},
                    "expected_gain": {"type": "number"},
                    "verdict": {"type": "string", "enum": ["accept", "reject", "merge"]},
                    "merge_into": {"type": ["integer", "null"]},
                    "reason": {"type": "string"},
                },
                "required": ["idea_id", "relevance", "novelty", "feasibility", "expected_gain", "verdict", "merge_into", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


def _context(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger) -> str:
    root = Path(adapter.root)
    from .planner import profile_text
    prof = profile_text(cfg, adapter)
    listing = subprocess.run(["git", "ls-files", "src/point_add"], cwd=root, capture_output=True, text=True).stdout
    files = [l for l in listing.splitlines() if l.endswith((".rs", ".md", ".txt"))]
    tried = [(i.id, i.status, i.title, round(i.best_gain * 100, 3)) for i in ledger.ideas(None) if i.status != "candidate"]
    return (f"CIRCUIT PROFILE (measured):\n{prof}\n\n"
            f"FILES IN THE EDITABLE FOLDER ({len(files)}): {' '.join(Path(f).name for f in files)[:4000]}\n\n"
            f"IDEAS ALREADY IN THE BACKLOG (id, status, title, best gain %): {json.dumps(tried)[:6000]}")


def _skill(cfg: Config, name: str) -> str:
    p = Path(cfg.skills_dir) / name / "SKILL.md"
    return p.read_text() if p.exists() else ""


def review(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger, batch: int = 12) -> dict:
    backend = make_backend(cfg)
    cands = ledger.ideas("candidate")
    if not cands:
        return {"reviewed": 0, "accepted": 0, "rejected": 0, "merged": 0}
    ctx = _context(cfg, adapter, ledger)
    stats = {"reviewed": 0, "accepted": 0, "rejected": 0, "merged": 0}
    for b in range(0, len(cands), batch):
        chunk = cands[b:b + batch]
        items = "\n\n".join(f"idea_id {i.id}: {i.title}\nsource: {i.source}\n{i.rationale}" for i in chunk)
        prompt = f"""{_skill(cfg, 'idea-review')}

{ctx}

CANDIDATES:
{items}

Score each candidate in [0,1] on relevance (does it lower Toffoli x qubits for this exact target?), novelty (not already
in the code or the backlog), feasibility (can one solver run of ~1 hour with ~10 scored builds test it?) and expected_gain
(your honest estimate of relative score improvement if it works, 0.01 = 1 %). Verdict: accept, reject, or merge into an
existing backlog idea (give its id). Reason in 2-3 sentences. Be strict: the code has absorbed hundreds of agent runs.
Return JSON matching the schema."""
        try:
            raw = backend.complete(prompt, schema=REVIEW_SCHEMA)
            data = json.loads(raw) if raw.strip().startswith("{") else {"reviews": []}
        except Exception as e:  # noqa: BLE001
            print(f"[review] batch {b // batch + 1} failed: {e}")
            if stats["reviewed"] == 0:
                print("[review] stopping: the model call failed (check the model name); candidates stay pending.")
                return stats
            continue
        by_id = {i.id: i for i in chunk}
        for r in data.get("reviews", []):
            i = by_id.get(r.get("idea_id"))
            if not i:
                continue
            scores = {k: float(r.get(k, 0)) for k in ("relevance", "novelty", "feasibility", "expected_gain")}
            note = f"{r['verdict']} | " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()) + f" | {r.get('reason', '')}"
            stats["reviewed"] += 1
            if r["verdict"] == "accept":
                prior = 0.005 + scores["expected_gain"] * scores["feasibility"] * scores["relevance"]
                ledger.set_idea_status(i.id, "open", review=note, prior=round(prior, 4))
                stats["accepted"] += 1
            elif r["verdict"] == "merge" and r.get("merge_into") in {x.id for x in ledger.ideas(None)}:
                ledger.set_idea_status(i.id, "rejected", review=note + f" (merged into #{r['merge_into']})")
                stats["merged"] += 1
            else:
                ledger.set_idea_status(i.id, "rejected", review=note)
                stats["rejected"] += 1
        print(f"[review] batch {b // batch + 1}: {stats}", flush=True)
    return stats

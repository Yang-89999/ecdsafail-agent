"""Brainstorm: new candidate ideas from what the system has learned so far.

Runs when the backlog runs low. The model reads the measured profile, the recent
experiment reports (what changed, why it worked or not), the retired ideas and the
file listing, and proposes candidates grounded in that evidence. Candidates go
through the Idea Reviewer like literature ideas.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..adapters.base import CompetitionAdapter
from ..config import Config
from ..ledger import Ledger
from ..llm import make_backend
from .lit_scout import EXTRACT_SCHEMA

SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "first_experiment": {"type": "string"},
                },
                "required": ["title", "evidence", "mechanism", "first_experiment"],
                "additionalProperties": False,
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ideas", "open_questions"],
    "additionalProperties": False,
}
_ = EXTRACT_SCHEMA  # same shape family; kept for readers comparing the two roles


def _skill(cfg: Config, name: str) -> str:
    p = Path(cfg.skills_dir) / name / "SKILL.md"
    return p.read_text() if p.exists() else ""


def brainstorm(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger, want: int = 8) -> int:
    backend = make_backend(cfg)
    root = Path(adapter.root)
    from .planner import profile_text
    prof = profile_text(cfg, adapter)
    listing = subprocess.run(["git", "ls-files", "src/point_add"], cwd=root, capture_output=True, text=True).stdout
    files = " ".join(Path(f).name for f in listing.splitlines() if f.endswith(".rs"))[:3000]
    reports = []
    for e in ledger.experiments(12):
        d = json.loads(e["details"] or "{}").get("report") or {}
        if d:
            reports.append({"exp": e["id"], "status": e["status"], "gain": e["gain"],
                            "what": d.get("what_changed", "")[:300], "why": d.get("why_it_worked_or_not", "")[:300]})
    ideas = [(i.status, i.title) for i in ledger.ideas(None) if i.status != "rejected"]
    prompt = f"""{_skill(cfg, 'lit-scout')}

You are brainstorming for the same target (secp256k1 affine point addition, Toffoli x peak qubits), but from
the system's own evidence instead of papers.

MEASURED PROFILE:
{prof}

FILES IN src/point_add: {files}

RECENT EXPERIMENT REPORTS: {json.dumps(reports)[:6000]}

IDEAS ALREADY IN THE BACKLOG (status, title): {json.dumps(ideas)[:5000]}

Propose up to {want} new ideas that are not in the backlog, and up to 2 open_questions: specific technical
questions whose answer from the literature would unblock a bigger gain (one sentence each). Each must point at a specific stage from the profile
or a specific finding in the reports (evidence), name the mechanism by which it lowers Toffolis or width, and give
a first experiment one solver can run in an hour. Prefer ideas that combine or reuse what the reports found.
Return JSON matching the schema."""
    try:
        raw = backend.complete(prompt, schema=SCHEMA)
        data = json.loads(raw) if raw.strip().startswith("{") else {"ideas": []}
    except Exception as e:  # noqa: BLE001
        print(f"[brainstorm] model call failed: {e}")
        return 0
    added = 0
    for idea in data.get("ideas", []):
        why = f"Evidence: {idea['evidence']}\nMechanism: {idea['mechanism']}\nFirst experiment: {idea['first_experiment']}"
        before = len(ledger.ideas(None))
        ledger.add_idea(idea["title"][:120], why, "brainstorm", prior=0.01, status="candidate")
        added += len(ledger.ideas(None)) - before
    print(f"[brainstorm] {added} new candidates", flush=True)
    from .lit_scout import ask
    for q in (data.get("open_questions") or [])[:2]:
        print(f"[brainstorm] asking the literature: {q}", flush=True)
        ask(cfg, ledger, q)
    return added

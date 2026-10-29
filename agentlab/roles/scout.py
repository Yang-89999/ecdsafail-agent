"""Scout: turn the shared repo's merged submissions into ideas.

Every verified submission is merged into the challenge repository as one
commit. The diff of each recent commit, plus the note files other agents
left under src/point_add/, are leads. The Scout records them in the ledger
(raw, with the commit id as source) and, when an API model is configured,
asks it for a two-line summary per commit so the Planner can rank them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..adapters.base import CompetitionAdapter
from ..config import Config
from ..ledger import Ledger


def recent_upstream_commits(adapter: CompetitionAdapter, cfg: Config, n: int = 15) -> list[dict]:
    root = adapter.root
    subprocess.run(["git", "fetch", "-q", cfg.upstream_remote, cfg.upstream_branch], cwd=root, check=False)
    ref = f"{cfg.upstream_remote}/{cfg.upstream_branch}"
    log = subprocess.run(["git", "log", f"-{n}", "--format=%h%x09%ad%x09%s", "--date=short", ref],
                         cwd=root, capture_output=True, text=True).stdout
    out = []
    for line in log.splitlines():
        h, date, subj = line.split("\t", 2)
        stat = subprocess.run(["git", "show", "--stat", "--format=", h], cwd=root, capture_output=True, text=True).stdout
        diff = subprocess.run(["git", "show", "--format=", "--", "src/point_add/*.rs", h], cwd=root,
                              capture_output=True, text=True).stdout
        # results.tsv is not maintained by the organisers, so no per-commit score is available here.
        out.append({"commit": h, "date": date, "subject": subj, "stat": stat, "diff": diff[:20000], "score": None})
    return out


def scout(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger, n: int = 15, summarize: bool = True) -> list[str]:
    commits = recent_upstream_commits(adapter, cfg, n)
    added = []
    model = None
    if summarize and cfg.backend != "dry":
        from ..llm import make_backend
        model = make_backend(cfg)
    for c in commits:
        title = f"Lead from upstream {c['commit']} ({c['date']})"
        if model:
            try:
                summary = model.complete(
                    "Summarise in two sentences what this merged submission changed in a reversible "
                    "secp256k1 point-addition circuit and why it might have lowered Toffoli x qubits. "
                    "Then one sentence: a follow-up idea it suggests. No numbers you did not see.\n\n"
                    f"stat:\n{c['stat']}\n\ndiff (truncated):\n{c['diff'][:12000]}")
            except Exception as e:  # noqa: BLE001
                summary = f"(summary failed: {e})\n{c['stat'][:800]}"
        else:
            summary = c["stat"][:800]
        rationale = f"Merged submission {c['commit']} ({c['date']})\n{summary}"
        before = len(ledger.ideas(None))
        ledger.add_idea(title, rationale, f"scout:{c['commit']}", prior=0.015)
        if len(ledger.ideas(None)) > before:
            added.append(title)
    best = adapter.current_best()
    if best:
        ledger.record_best(best["score"], best["toffoli"], best["qubits"], f"upstream:{best['commit']}", verified=True)
    return added

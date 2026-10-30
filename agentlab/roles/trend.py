"""Trend: what the leaderboard history says.

Every verified gain is merged into the challenge repository as one commit touching src/point_add,
so the history of origin/main is the leaderboard history. This role measures the score of each
recent merged submission (fast score in one reused worktree, cached per commit under
work/trend/scores.json; results.tsv is not maintained by the organisers), classifies what each step
changed (from its diff and the notes the submitter left), and asks the model for a synthesis:
where gains came from, what is exhausted, what the others do, framework-level directions, and
strategic implications. Output: work/trend.md (+ .json), read by the Strategist and the Coordinator.
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

STEP_SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
        "commit": {"type": "string"}, "stage": {"type": "string"}, "mechanism": {"type": "string"},
        "novelty": {"type": "string", "enum": ["new technique", "tuning", "regeneration", "unclear"]},
        "tools_used": {"type": "string"}},
        "required": ["commit", "stage", "mechanism", "novelty", "tools_used"], "additionalProperties": False}}},
    "required": ["steps"], "additionalProperties": False,
}

SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "where_gains_came_from": {"type": "string"},
        "shrinking": {"type": "string"},
        "what_others_do": {"type": "string"},
        "framework_directions": {"type": "array", "items": {"type": "object", "properties": {
            "direction": {"type": "string"}, "stage": {"type": "string"}, "evidence": {"type": "string"},
            "predicted_gain": {"type": "number"}, "first_experiment": {"type": "string"}},
            "required": ["direction", "stage", "evidence", "predicted_gain", "first_experiment"], "additionalProperties": False}},
        "implications": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["where_gains_came_from", "shrinking", "what_others_do", "framework_directions", "implications"],
    "additionalProperties": False,
}


def trend_path(cfg: Config) -> Path:
    return Path(cfg.runs_dir).parent / "trend.md"


def trend_text(cfg: Config, limit: int = 6000) -> str:
    p = trend_path(cfg)
    return p.read_text()[:limit] if p.exists() else ""


def _git(root, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout


def history(adapter: CompetitionAdapter, cfg: Config, n: int, landmarks: int = 0) -> list[dict]:
    """The last n merged submissions (oldest first), plus every `landmarks`-th older one for the long shape."""
    root = adapter.root
    subprocess.run(["git", "fetch", "-q", cfg.upstream_remote, cfg.upstream_branch], cwd=root, check=False)
    ref = f"{cfg.upstream_remote}/{cfg.upstream_branch}"
    log = _git(root, "log", "--format=%h%x09%ad%x09%s", "--date=short", ref, "--", cfg.editable_path).splitlines()
    rows = [dict(zip(("commit", "date", "subject"), l.split("\t", 2))) for l in log if l.strip()]
    chosen = rows[:n]
    if landmarks:
        chosen += rows[n::landmarks]
    chosen.reverse()
    for r in chosen:
        r["rank_from_top"] = rows.index(r)
    return chosen


def step_material(adapter: CompetitionAdapter, commit: str, limit: int = 7000) -> dict:
    """What a classifier needs: the stat, the notes the submitter added, and the code diff (truncated)."""
    root = adapter.root
    stat = _git(root, "show", "--stat=120", "--format=", commit)
    names = _git(root, "show", "--name-only", "--format=", commit).split()
    notes = ""
    for f in names:
        if f.endswith((".txt", ".md")):
            notes += f"\n--- {f} ---\n" + _git(root, "show", f"{commit}:{f}")[:2500]
    diff = _git(root, "show", "--format=", commit, "--", "src/point_add/*.rs")
    return {"stat": stat[-2500:], "notes": notes[:limit // 2], "diff": diff[:limit]}


def measure(cfg: Config, adapter: CompetitionAdapter, commits: list[str], cache: Path) -> dict:
    """Fast score per commit in one reused worktree (warm cargo cache); cached, failures recorded."""
    scores = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [c for c in commits if c not in scores]
    if not todo:
        return scores
    wt = Path(cfg.worktrees_dir) / "trend"
    root = adapter.root
    if not wt.exists():
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), todo[0]], cwd=root, check=True)
    for i, c in enumerate(todo):
        subprocess.run(["git", "checkout", "-q", "--detach", c], cwd=wt, check=True)
        t0 = time.time()
        try:
            res = adapter.fast_score(wt, cfg.fast_shots)
            scores[c] = {"ok": res.ok, "score": res.score if res.ok else None, "toffoli": res.metrics.get("toffoli"),
                         "qubits": res.metrics.get("qubits"), "ops": res.metrics.get("ops"), "reason": None if res.ok else (res.reason or "")[:200]}
        except Exception as e:  # noqa: BLE001
            scores[c] = {"ok": False, "score": None, "reason": str(e)[:200]}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(scores, indent=1))
        print(f"[trend] {i + 1}/{len(todo)} {c}: {scores[c].get('score')} ({int(time.time() - t0)} s)", flush=True)
    return scores


def render(rows: list[dict], synth: dict | None) -> str:
    lines = [f"# Leaderboard trend  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; {len(rows)} merged submissions, oldest first; "
             "scores are our fast score, 1024 shots)", "",
             "| commit | date | rank from top | score | Δ product | Δ Toffoli | Δ qubits | stage | mechanism | novelty |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sc = f"{r['score']:,}" if r.get("score") else ("unmeasured" if r.get("reason") is None else f"failed: {r['reason'][:40]}")
        dp = f"{r['d_product'] * 100:+.3f} %" if r.get("d_product") is not None else "-"
        dt = f"{r['d_toffoli']:+,.0f}" if r.get("d_toffoli") is not None else "-"
        dw = f"{r['d_qubits']:+d}" if r.get("d_qubits") is not None else "-"
        lines.append(f"| {r['commit']} | {r['date']} | {r['rank_from_top']} | {sc} | {dp} | {dt} | {dw} | {r.get('stage', '')} | "
                     f"{(r.get('mechanism') or '').replace('|', '/')[:160]} | {r.get('novelty', '')} |")
    if synth:
        lines += ["", "## Where the gains came from", synth["where_gains_came_from"].strip(), "",
                  "## How the steps are shrinking", synth["shrinking"].strip(), "",
                  "## What the others do", synth["what_others_do"].strip(), "", "## Framework-level directions (untried by the history)"]
        for d in synth["framework_directions"]:
            lines.append(f"- **{d['direction']}** ({d['stage']}): {d['evidence']} Predicted gain {d['predicted_gain'] * 100:.2f} % (prediction). "
                         f"First experiment: {d['first_experiment']}")
        lines += ["", "## Strategic implications"] + [f"- {s}" for s in synth["implications"]]
    return "\n".join(lines) + "\n"


def analyse(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger, n: int = 30, landmarks: int = 0,
            do_measure: bool = True, batch: int = 6) -> list[dict]:
    rows = history(adapter, cfg, n, landmarks)
    if not rows:
        print("[trend] no merged submissions found", flush=True)
        return []
    cache = Path(cfg.runs_dir).parent / "trend" / "scores.json"
    scores = measure(cfg, adapter, [r["commit"] for r in rows], cache) if do_measure else (
        json.loads(cache.read_text()) if cache.exists() else {})
    prev = None
    for r in rows:
        s = scores.get(r["commit"]) or {}
        r.update({"score": s.get("score"), "toffoli": s.get("toffoli"), "qubits": s.get("qubits"), "reason": s.get("reason")})
        if r["score"] and prev and prev.get("score"):
            r["d_product"] = (r["score"] - prev["score"]) / prev["score"]
            r["d_toffoli"] = (r["toffoli"] or 0) - (prev["toffoli"] or 0)
            r["d_qubits"] = int((r["qubits"] or 0) - (prev["qubits"] or 0))
        if r["score"]:
            prev = r
    skill = Path(cfg.skills_dir) / "trend" / "SKILL.md"
    skill_txt = skill.read_text() if skill.exists() else ""
    synth = None
    if cfg.backend != "dry":
        backend = make_backend(cfg)
        classified = json.loads((cache.parent / "steps.json").read_text()) if (cache.parent / "steps.json").exists() else {}
        todo = [r for r in rows if r["commit"] not in classified]
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            items = "\n\n".join(f"COMMIT {r['commit']} ({r['date']}; measured effect: product {r.get('d_product')}, "
                                f"toffoli {r.get('d_toffoli')}, qubits {r.get('d_qubits')})\n{json.dumps(step_material(adapter, r['commit']))}"
                                for r in chunk)
            prompt = f"{skill_txt}\n\nClassify each of these merged submissions (one entry per commit). Numbers: only the measured effect given.\n\n{items}"
            try:
                data = json.loads(backend.complete(prompt, schema=STEP_SCHEMA, timeout_s=900))
                for s in data.get("steps", []):
                    classified[s["commit"]] = s
                cache.parent.mkdir(parents=True, exist_ok=True)
                (cache.parent / "steps.json").write_text(json.dumps(classified, indent=1))
                print(f"[trend] classified {min(b + batch, len(todo))}/{len(todo)} steps", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[trend] classification batch failed: {e}", flush=True)
        for r in rows:
            r.update({k: v for k, v in (classified.get(r["commit"]) or {}).items() if k != "commit"})
        table = render(rows, None)
        from .planner import profile_text
        prompt = f"""{skill_txt}

MEASURED HISTORY (our fast score per merged submission; Δ columns are differences of measured scores):
{table}

OUR MEASURED PROFILE OF THE CURRENT HEAD:
{profile_text(cfg, adapter, 5000)}

WHAT OUR OWN RUNS HAVE TRIED (ledger, most recent first):
{ledger.summary_text(25)[:6000]}

Write the synthesis as JSON matching the schema."""
        try:
            synth = json.loads(backend.complete(prompt, schema=SYNTH_SCHEMA, timeout_s=1500))
        except Exception as e:  # noqa: BLE001
            print(f"[trend] synthesis failed: {e}", flush=True)
    trend_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    trend_path(cfg).write_text(render(rows, synth))
    trend_path(cfg).with_suffix(".json").write_text(json.dumps({"rows": rows, "synthesis": synth}, indent=1, default=str))
    print(f"[trend] wrote {trend_path(cfg)} ({sum(1 for r in rows if r.get('score'))}/{len(rows)} measured)", flush=True)
    return rows


def upstream_head(adapter: CompetitionAdapter, cfg: Config) -> str:
    return _git(adapter.root, "rev-parse", "--short", f"{cfg.upstream_remote}/{cfg.upstream_branch}").strip()


def stale(cfg: Config, adapter: CompetitionAdapter) -> bool:
    """True when there is no trend file or upstream has merged submissions since it was written."""
    pj = trend_path(cfg).with_suffix(".json")
    if not pj.exists():
        return True
    rows = json.loads(pj.read_text()).get("rows") or []
    return not rows or rows[-1]["commit"] != upstream_head(adapter, cfg)

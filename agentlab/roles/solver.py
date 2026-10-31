"""Solver: one hypothesis, one git worktree, one agent run.

The solver agent (Codex or the API loop) edits only the editable paths,
scores with scripts/fast_score.sh as often as it likes, and ends with a
JSON report. The orchestrator then re-scores the final tree itself: the
agent's own numbers are never trusted.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from ..adapters.base import CompetitionAdapter
from ..config import Config
from ..ledger import Ledger
from ..llm import make_backend

REPORT_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def load_skill(cfg: Config, name: str) -> str:
    p = Path(cfg.skills_dir) / name / "SKILL.md"
    return p.read_text() if p.exists() else ""


def make_worktree(cfg: Config, adapter: CompetitionAdapter, exp_id: int, base_ref: str) -> tuple[Path, str]:
    wt = Path(cfg.worktrees_dir) / f"exp{exp_id}"
    if wt.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=adapter.root, check=False)
    branch = f"agentlab/exp{exp_id}"
    subprocess.run(["git", "branch", "-D", branch], cwd=adapter.root, capture_output=True)
    subprocess.run(["git", "worktree", "add", "-q", "-b", branch, str(wt), base_ref], cwd=adapter.root, check=True)
    base = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    (wt / ".agentlab").mkdir(exist_ok=True)  # scores.jsonl (live progress) and other scratch of the system
    exclude = Path(adapter.root) / ".git" / "info" / "exclude"
    if exclude.parent.exists() and ".agentlab/" not in (exclude.read_text() if exclude.exists() else ""):
        with exclude.open("a") as f:
            f.write(".agentlab/\n")
    return wt, base


def solver_prompt(cfg: Config, adapter: CompetitionAdapter, brief: str, wt: Path, fast_shots: int,
                  best: dict | None = None) -> str:
    scripts = Path(cfg.skills_dir).parent / "scripts"
    skill_core = load_skill(cfg, "solver-core")
    from .lessons import lessons_text
    lessons = lessons_text(cfg) or "(none yet)"
    from ..orchestrator import fresh_line
    head_fresh = fresh_line((best or {}).get("fresh"))
    skill_domain = (load_skill(cfg, "reversible-arith") + "\n\n" + load_skill(cfg, "rust-circuit-dev")
                    + "\n\n" + load_skill(cfg, "design-passes"))
    return f"""You are a Solver in an autoresearch system working on the ECDSA.fail benchmark.
Your working directory is a git worktree of the challenge repository: {wt}

RULES (from the competition; violating any of them makes the run worthless):
{adapter.rules_text()}

HOW TO SCORE ({fast_shots} shots, the official count; under a minute, so score often; the number is the official one):
    bash {scripts}/fast_score.sh {wt} {fast_shots} {adapter.root}
The last line of its output is JSON: {{"ok": true, "score": ..., "metrics": {{"toffoli": ..., "qubits": ...}}}}.
Only a run that prints ok=true counts. Never edit results.tsv or score.json by hand.

SWEEPING MANY VARIANTS (the main way to work; it runs unattended while you think):
    python3 {scripts}/sweep.py {wt} --mutate "python3 .agentlab/mutate.py {{v}}" --values 0,1,2,3,4,5,6,7 --shots {fast_shots} --challenge {adapter.root} --keep-best
    python3 {scripts}/sweep.py {wt} --patches .agentlab/variants --shots {fast_shots} --challenge {adapter.root} --keep-best
Write a small mutation script (or a folder of patches) that turns one knob of your idea, commit your tree, and sweep.
Each variant is applied to the committed tree, scored, logged to .agentlab/sweep.jsonl and reverted; --keep-best
leaves the best winner applied for you to check and commit. Then sweep around the winner.

FRESH INPUTS (the honesty check; run it on your final candidate and on any variant you are about to keep):
    bash {scripts}/fresh_check.sh {wt} {cfg.fresh_draws} {fast_shots} {adapter.root}
The official scorer tests ONE input draw, derived from a hash of the op stream; any edit redraws it. fresh_check tests
the same circuit on {cfg.fresh_draws} independent draws and prints the error rate and the average Toffoli across draws.
Our correctness standard is the one of the published reference circuits (Babbush et al. 2026, Schrottenloher 2026):
an error rate of at most {cfg.max_error_rate:g} on independent inputs. Approximations are allowed within that budget.
The unmodified public head on fresh inputs: {head_fresh}
So: (1) a failure on the new draw that the unmodified head also shows on fresh draws is INHERITED; do not spend the run
explaining or repairing it unless your plan brings the error rate within the standard. (2) A variant that wins on
the official draw but not on the fresh-draw average won by luck: it is not a gain. (3) Never change TAIL_NONCE or any
value whose only effect is which inputs get drawn, and never pick a variant because its draw passed.

ASKING THE LITERATURE (allowed network use; 1-3 min; answers are cached and cited):
    bash {scripts}/ask_lit.sh "<one specific technical question>"
Use it when you hit a sub-problem you cannot settle from the code and notes (e.g. the cheapest known way to do
one primitive in this gate set). Ask at most 3 questions per run; read the answer, then test, never just cite.

YOUR TASK
{brief}

LESSONS FROM EARLIER RUNS (work/lessons.md; read before you plan, do not repeat a dead end):
{lessons}

{skill_core}

{skill_domain}

There is no clock on this run. Stop when you have a verified gain with no cheap next step, or when your evidence says the
direction is hopeless; not because time passed. Do not leave the run early with a tree you have not scored.

WHEN DONE, commit your changes inside the worktree (git add src/point_add && git commit -m "...") and end your
final message with exactly one fenced JSON block:
```json
{{"status": "improved" | "no_gain" | "failed",
  "idea_status": "promising" | "already_implemented" | "dead_end" | "needs_more_time",
  "score": <int from your last ok fast_score run or null>,
  "toffoli": <number or null>, "qubits": <int or null>,
  "what_changed": "<two sentences>",
  "why_it_worked_or_not": "<two sentences>",
  "next_ideas": ["<concrete follow-up>", "..."],
  "variants": [{{"change": "<one line>", "ok": true|false, "toffoli": <number or null>, "qubits": <int or null>}}, "..."]}}
```
"""


def harvest_notes(cfg: Config, wt: Path, base: str, exp_id: int, title: str) -> list[str]:
    """Copy every note file the agent added or changed to work/notes/exp<N>/ (worktrees are per experiment,
    so the thinking roles would otherwise never see them); a profiling run also refreshes work/profile.md."""
    changed = subprocess.run(["git", "diff", "--name-only", f"{base}..HEAD"], cwd=wt, capture_output=True, text=True).stdout.split()
    notes = [f for f in changed if f.endswith((".md", ".txt")) and (wt / f).exists()]
    if not notes:
        return []
    nd = Path(cfg.runs_dir).parent / "notes" / f"exp{exp_id}"
    nd.mkdir(parents=True, exist_ok=True)
    for f in notes:
        (nd / Path(f).name).write_text((wt / f).read_text(errors="replace"))
    is_profile = any("profile" in f.lower() for f in notes) or "profile" in title.lower()
    if is_profile:
        text = "\n\n".join(f"<!-- {f} -->\n" + (wt / f).read_text(errors="replace") for f in notes)
        (Path(cfg.runs_dir).parent / "profile.md").write_text(text[:60000])
    print(f"[agentlab] exp #{exp_id} left notes {[Path(f).name for f in notes]} -> work/notes/exp{exp_id}/"
          + (" and work/profile.md" if is_profile else ""), flush=True)
    return notes


def run_solver(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger, idea_id: int | None, brief: str,
               base_ref: str, best: dict | None, exp_id: int | None = None, mode: str = "new",
               parents: list[int] | None = None) -> int:
    """One executor run. mode "new": fresh worktree at base_ref (the upstream head). "continue": the worktree
    starts from the parent experiment's final tree (branch agentlab/exp<parent>) and keeps the parent's base
    commit, so the audit's cherry-pick range covers both runs. "combine": fresh worktree at base_ref; the brief
    tells the agent which branches to cherry-pick."""
    backend = make_backend(cfg)
    parents = list(parents or [])
    if exp_id is None:
        exp_id = ledger.start_experiment(idea_id, "", "", cfg.model or cfg.backend, note=brief.splitlines()[0][:200])
    if mode == "continue" and parents:
        parent = ledger.experiment(parents[0])
        wt, head0 = make_worktree(cfg, adapter, exp_id, f"agentlab/exp{parents[0]}")
        base = (parent["base_commit"] if parent and parent["base_commit"] else head0)
    else:
        wt, base = make_worktree(cfg, adapter, exp_id, base_ref)
    ledger.finish_experiment(exp_id, "running", worktree=str(wt), base_commit=base, mode=mode, parents=json.dumps(parents),
                             note=brief.splitlines()[0][:200])
    prompt = solver_prompt(cfg, adapter, brief, wt, cfg.fast_shots, best)
    run_dir = Path(cfg.runs_dir) / f"exp{exp_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.md").write_text(prompt)

    t0 = time.time()
    print(f"[agentlab] solver exp #{exp_id} ({mode}{' of ' + str(parents) if parents else ''}) started on idea {idea_id}; "
          f"follow it with: agentlab watch {exp_id}", flush=True)
    run = backend.run(prompt, wt, run_dir / "agent.log", extra_dirs=[Path(cfg.skills_dir).parent / "tools"])
    print(f"[agentlab] solver exp #{exp_id} agent finished after {int(run.duration_s) // 60} min (exit {run.returncode}); re-scoring its tree", flush=True)
    report: dict = {}
    m = REPORT_RE.search(run.final_message or "")
    if m:
        try:
            report = json.loads(m.group(1))
        except json.JSONDecodeError:
            report = {"status": "unparsed"}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    if run.returncode != 0 and not report and run.duration_s < 300:
        # The model never really started (usage limit, auth, network): not the idea's fault.
        tail = Path(run.log_path).read_text(errors="replace")[-600:] if Path(run.log_path).exists() else ""
        ledger.finish_experiment(exp_id, "backend_failed", log_path=run.log_path,
                                 note=f"codex exited {run.returncode} after {int(run.duration_s)} s", details={"log_tail": tail})
        print(f"[agentlab] exp #{exp_id}: model backend failed within {int(run.duration_s)} s:\n{tail[-400:]}", flush=True)
        return exp_id

    harvest_notes(cfg, wt, base, exp_id, brief.splitlines()[0])

    # Independent re-score of whatever the agent left in the tree.
    bad = adapter.protected_diff(wt, base)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    if bad:
        ledger.finish_experiment(exp_id, "rejected", head_commit=head, note=f"touched protected files: {bad[:5]}",
                                 log_path=run.log_path, details={"report": report, "protected": bad})
        if idea_id:
            ledger.record_idea_outcome(idea_id, -1.0)
        return exp_id
    res = adapter.fast_score(wt, cfg.fast_shots)
    gain = None
    if res.ok and best:
        ref = best.get("fast_score") or best["score"]  # same shots as res: an unchanged tree gains exactly 0
        gain = (ref - res.score) / ref
    status = "scored" if res.ok else "failed"
    ledger.finish_experiment(
        exp_id, status, head_commit=head, shots=res.shots, toffoli=res.metrics.get("toffoli"),
        qubits=res.metrics.get("qubits"), score=res.score, gain=gain, log_path=run.log_path,
        note=(report.get("what_changed") or res.reason or "")[:500],
        details={"report": report, "rescore": res.__dict__, "agent_s": run.duration_s, "total_s": time.time() - t0},
    )
    if idea_id:
        ledger.record_idea_outcome(idea_id, gain if gain is not None else -1.0)
        if getattr(ledger.idea(idea_id), "kind", None) == "measurement" and status == "scored":
            ledger.set_idea_status(idea_id, "done", review=f"measured in exp {exp_id}; numbers in work/notes/exp{exp_id}/")
        elif report.get("idea_status") in ("already_implemented", "dead_end"):
            ledger.set_idea_status(idea_id, "exhausted")
    for follow in report.get("next_ideas") or []:
        if isinstance(follow, str) and 10 < len(follow) < 400:
            ledger.add_idea(follow[:120], follow, f"solver:{exp_id}", prior=0.01)
    return exp_id

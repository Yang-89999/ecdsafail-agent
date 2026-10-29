"""agentlab command line.

  agentlab setup                      toolchain, harness build, fast_eval build
  agentlab best                       public best from the shared repo
  agentlab scout [-n 15]              merged submissions -> ideas
  agentlab ideas [add "title" "why"]  list / add ideas
  agentlab run --rounds N --solvers K the loop
  agentlab score <worktree> [--shots] fast score of any worktree
  agentlab audit <exp>                clean-room + official benchmark for one experiment
  agentlab approve <exp> --by NAME    record a human approval
  agentlab submit <exp>               official submit (requires an approval record and an audit)
  agentlab ledger                     summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import load_adapter
from .config import Config
from .ledger import Ledger


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("best")
    p = sub.add_parser("scout"); p.add_argument("-n", type=int, default=15); p.add_argument("--no-summarize", action="store_true")
    p = sub.add_parser("ideas"); p.add_argument("action", nargs="?", default="list"); p.add_argument("args", nargs="*")
    p = sub.add_parser("run"); p.add_argument("--rounds", type=int, default=1); p.add_argument("--solvers", type=int, default=None)
    p.add_argument("--audit-threshold", type=float, default=None, help="min fast-score relative gain to trigger an audit (default from config)")
    p = sub.add_parser("score"); p.add_argument("worktree", type=Path); p.add_argument("--shots", type=int, default=None)
    p = sub.add_parser("audit"); p.add_argument("exp", type=int)
    p = sub.add_parser("approve"); p.add_argument("exp", type=int); p.add_argument("--by", required=True); p.add_argument("--note", default="")
    p = sub.add_parser("submit"); p.add_argument("exp", type=int); p.add_argument("--note", default="")
    sub.add_parser("ledger")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    ledger = Ledger(cfg.ledger_path)
    adapter = load_adapter(cfg)

    if a.cmd == "setup":
        adapter.ensure_ready()
        print("ready:", adapter.root)
    elif a.cmd == "best":
        print(json.dumps(adapter.current_best(), indent=2))
    elif a.cmd == "scout":
        from .roles.scout import scout
        for t in scout(cfg, adapter, ledger, a.n, summarize=not a.no_summarize):
            print("+", t)
        print(ledger.summary_text())
    elif a.cmd == "ideas":
        from .roles.planner import seed_ideas
        if a.action == "add":
            title, why = (a.args + ["", ""])[:2]
            print("idea", ledger.add_idea(title, why or title, "human", prior=0.03))
        elif a.action == "seed":
            print("seeded", seed_ideas(ledger))
        for i in ledger.ideas(None):
            print(f"[{i.id}] {i.status:9s} tries={i.tries} best={i.best_gain*100:+.3f}%  {i.title}")
    elif a.cmd == "run":
        from .orchestrator import run_rounds
        run_rounds(cfg, a.rounds, a.solvers, cfg.audit_threshold if a.audit_threshold is None else a.audit_threshold)
    elif a.cmd == "score":
        res = adapter.fast_score(a.worktree, a.shots or cfg.fast_shots)
        print(json.dumps(res.__dict__, indent=2, default=str))
    elif a.cmd == "audit":
        from .roles.auditor import audit
        print(json.dumps(audit(cfg, adapter, ledger, a.exp), indent=2, default=str))
    elif a.cmd == "approve":
        ledger.approve(a.exp, a.by, a.note)
        print(f"approval recorded for exp {a.exp} by {a.by}")
    elif a.cmd == "submit":
        exp = ledger.experiment(a.exp)
        if exp is None or exp["status"] != "audited":
            sys.exit(f"exp {a.exp} is not audited (status {exp['status'] if exp else 'missing'}); run `agentlab audit` first")
        appr = ledger.approval(a.exp)
        if appr is None:
            sys.exit(f"no approval on record for exp {a.exp}; run `agentlab approve {a.exp} --by <name>`")
        details = json.loads(exp["details"] or "{}")
        if cfg.submit_only_if_beats_best and not details.get("beats_best"):
            sys.exit(f"exp {a.exp} does not beat the best known score; policy forbids submitting it")
        out = adapter.submit(Path(exp["worktree"]), a.note)
        ledger.finish_experiment(a.exp, "submitted", details={**details, "submit_output": out})
        print(out)
    elif a.cmd == "ledger":
        print(ledger.summary_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

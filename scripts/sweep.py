#!/usr/bin/env python3
"""Score many variants of one worktree unattended: the agent picks the knobs, the harness explores them.

    python3 sweep.py <worktree> --mutate "python3 .agentlab/mutate.py {v}" --values 0,1,2,4,8 [--keep-best]
    python3 sweep.py <worktree> --patches <dir of *.patch> [--keep-best]

Each variant starts from the committed tree (src/point_add reset), is applied (the mutate command
with {v} replaced, or `git apply` of one patch), scored with fast_score.sh at the official shots,
and recorded in <worktree>/.agentlab/sweep.jsonl. The tree is reset after every variant. At the end
a ranked table is printed; with --keep-best the best variant that beats the committed tree is
re-applied and left uncommitted for the agent to check and commit. Commit your tree before sweeping.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def reset(wt: Path, editable: str) -> None:
    sh(["git", "checkout", "--", editable], wt)
    sh(["git", "clean", "-fdq", editable], wt)


def score(wt: Path, shots: int, challenge: str) -> dict:
    p = sh(["bash", str(HERE / "fast_score.sh"), str(wt), str(shots), challenge], wt)
    last = (p.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        d = json.loads(last)
    except json.JSONDecodeError:
        d = {"ok": False, "reason": "no JSON from fast_score.sh"}
    m = d.get("metrics", {})
    return {"ok": bool(d.get("ok")), "score": d.get("score"), "toffoli": m.get("toffoli"), "qubits": m.get("qubits"),
            "ops": m.get("ops") or m.get("emitted_ops"), "reason": d.get("reason")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("worktree", type=Path)
    ap.add_argument("--mutate", help='command that edits the tree for one value, with {v} in it')
    ap.add_argument("--values", help="comma-separated values for --mutate")
    ap.add_argument("--patches", type=Path, help="directory of *.patch files, one variant each")
    ap.add_argument("--keep-best", action="store_true")
    ap.add_argument("--shots", type=int, default=9024)
    ap.add_argument("--challenge", default=str(Path.home() / "ecdsafail-challenge"))
    ap.add_argument("--editable", default="src/point_add")
    a = ap.parse_args()
    wt = a.worktree.resolve()
    if sh(["git", "status", "--porcelain", "--", a.editable], wt).stdout.strip():
        print("commit or stash your changes under", a.editable, "before sweeping", file=sys.stderr)
        return 2
    if a.mutate and a.values:
        variants = [(f"v={v}", ["bash", "-lc", a.mutate.replace("{v}", v)]) for v in a.values.split(",")]
    elif a.patches:
        variants = [(p.name, ["git", "apply", str(p.resolve())]) for p in sorted(a.patches.glob("*.patch"))]
    else:
        ap.error("give --mutate with --values, or --patches")
    log = wt / ".agentlab" / "sweep.jsonl"
    log.parent.mkdir(exist_ok=True)
    print(f"[sweep] baseline (committed tree) ...", flush=True)
    base = score(wt, a.shots, a.challenge)
    print(f"[sweep] baseline: {base['score']} toffoli {base['toffoli']} qubits {base['qubits']}", flush=True)
    rows = []
    for name, cmd in variants:
        reset(wt, a.editable)
        t0 = time.time()
        ap_ = sh(cmd, wt)
        if ap_.returncode != 0:
            r = {"variant": name, "ok": False, "reason": f"apply failed: {(ap_.stderr or ap_.stdout)[-300:]}"}
        else:
            r = {"variant": name, **score(wt, a.shots, a.challenge)}
        r["seconds"] = int(time.time() - t0)
        if r.get("ok") and base.get("score"):
            r["gain"] = (base["score"] - r["score"]) / base["score"]
        rows.append(r)
        with log.open("a") as f:
            f.write(json.dumps(dict(r, ts=time.time())) + "\n")
        print(f"[sweep] {name}: ok={r.get('ok')} score={r.get('score')} toffoli={r.get('toffoli')} qubits={r.get('qubits')}"
              f" gain={r.get('gain')} ({r['seconds']} s){'  ' + str(r.get('reason'))[:120] if not r.get('ok') else ''}", flush=True)
    reset(wt, a.editable)
    good = sorted([r for r in rows if r.get("ok") and r.get("score")], key=lambda r: r["score"])
    print("\n[sweep] ranked (lower is better):")
    for r in good:
        print(f"  {r['variant']:30s} score {r['score']:>14,}  toffoli {r['toffoli']}  qubits {r['qubits']}  gain {r.get('gain', 0) * 100:+.4f} %")
    print(f"  ({len(rows) - len(good)} variants failed)")
    if a.keep_best and good and base.get("score") and good[0]["score"] < base["score"]:
        name = good[0]["variant"]
        cmd = dict(variants)[name]
        sh(cmd, wt)
        print(f"[sweep] kept {name} in the tree (uncommitted): check it, then commit")
    print(json.dumps({"baseline": base, "best": good[0] if good else None, "n": len(rows), "failed": len(rows) - len(good)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

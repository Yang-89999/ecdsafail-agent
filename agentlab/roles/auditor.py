"""Auditor and Verifier: the only path from "scored" to "ready to submit".

  1. protected-file diff against the base commit must be empty
  2. clean-room: fresh worktree at the newest upstream head, cherry-pick the experiment's commits
     (catches a stale base, which is how late submissions lose)
  3. determinism: build_circuit twice, ops.bin hashes must match
  4. full official benchmark.sh run (9024 shots)
  5. fresh inputs: the same op stream on cfg.fresh_draws independent input draws; the error rate must be at most
     cfg.max_error_rate (2^-13.3, the published reference circuits' level). The official draw is one hash-derived
     sample that can be re-rolled, so it says little about the error rate on its own.
  6. compare with the best verified score; record the verdict
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from ..adapters.base import CompetitionAdapter
from ..config import Config
from ..ledger import Ledger


def _sh(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def ops_hash(wt: Path) -> str:
    """Hash of the op stream produced by a fresh build_circuit run."""
    _sh(["cargo", "build", "--release", "--locked", "--offline", "--bin", "build_circuit"], wt)
    scratch = wt / ".agentlab" / "det"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "ops.bin").unlink(missing_ok=True)
    proc = subprocess.run([str(wt / "target/release/build_circuit")], cwd=scratch, capture_output=True, text=True)
    if proc.returncode != 0 or not (scratch / "ops.bin").exists():
        return f"build_circuit failed: {proc.stderr[-300:]}"
    return hashlib.sha256((scratch / "ops.bin").read_bytes()).hexdigest()


def audit(cfg: Config, adapter: CompetitionAdapter, ledger: Ledger, exp_id: int) -> dict:
    exp = ledger.experiment(exp_id)
    wt = Path(exp["worktree"])
    verdict: dict = {"exp": exp_id, "checks": {}}

    bad = adapter.protected_diff(wt, exp["base_commit"])
    verdict["checks"]["protected_files_untouched"] = not bad
    if bad:
        verdict["fail"] = f"protected files changed: {bad}"
        ledger.finish_experiment(exp_id, "rejected", details=verdict)
        return verdict

    # Clean room on the newest upstream head.
    _sh(["git", "fetch", "-q", cfg.upstream_remote, cfg.upstream_branch], adapter.root)
    upstream = f"{cfg.upstream_remote}/{cfg.upstream_branch}"
    clean = Path(cfg.worktrees_dir) / f"audit{exp_id}"
    if clean.exists():
        _sh(["git", "worktree", "remove", "--force", str(clean)], adapter.root)
    _sh(["git", "branch", "-D", f"agentlab/audit{exp_id}"], adapter.root)
    _sh(["git", "worktree", "add", "-q", "-b", f"agentlab/audit{exp_id}", str(clean), upstream], adapter.root)
    commits = _sh(["git", "rev-list", "--reverse", f"{exp['base_commit']}..HEAD"], wt).stdout.split()
    if commits:
        cp = _sh(["git", "cherry-pick", "-x", *commits], clean)
        verdict["checks"]["applies_on_upstream_head"] = cp.returncode == 0
        if cp.returncode != 0:
            verdict["fail"] = "does not apply cleanly on upstream head: " + cp.stderr[-500:]
            ledger.finish_experiment(exp_id, "rejected", details=verdict)
            return verdict
    else:
        verdict["checks"]["applies_on_upstream_head"] = False
        verdict["fail"] = "no commits in the experiment worktree"
        ledger.finish_experiment(exp_id, "rejected", details=verdict)
        return verdict

    h1, h2 = ops_hash(clean), ops_hash(clean)
    verdict["checks"]["deterministic_ops"] = h1 == h2 and len(h1) == 64
    verdict["ops_sha256"] = h1
    if not verdict["checks"]["deterministic_ops"]:
        verdict["fail"] = f"non-deterministic or failing build: {h1[:80]} vs {h2[:80]}"
        ledger.finish_experiment(exp_id, "rejected", details=verdict)
        return verdict

    res = adapter.full_verify(clean, note=f"agentlab audit exp{exp_id}")
    verdict["checks"]["official_benchmark_ok"] = res.ok
    verdict["full"] = {"score": res.score, "metrics": res.metrics, "reason": res.reason, "log": res.log_path,
                       "duration_s": res.duration_s}
    if not res.ok:
        verdict["fail"] = f"official benchmark rejected: {res.reason}"
        ledger.finish_experiment(exp_id, "rejected", details=verdict)
        return verdict

    fresh = adapter.fresh_check(clean, cfg.fresh_draws, 9024)
    verdict["fresh"] = fresh
    verdict["checks"]["fresh_inputs_ok"] = bool(fresh.get("ok"))
    if not fresh.get("ok"):
        verdict["fail"] = (f"error rate on independent inputs {fresh.get('error_rate')} exceeds {cfg.max_error_rate:g} "
                           f"({fresh.get('failing_shots')} failing of {fresh.get('tested_shots')}; {fresh.get('reason', '')})")
        ledger.finish_experiment(exp_id, "rejected", details=verdict)
        return verdict

    best = adapter.current_best()
    local_best = ledger.best()
    bar = min([b for b in [best and best["score"], local_best and local_best["score"]] if b] or [None]) if (best or local_best) else None
    verdict["public_best"] = best
    verdict["beats_best"] = bar is None or res.score < bar * (1 - cfg.min_relative_gain)
    ledger.finish_experiment(exp_id, "audited", score=res.score, shots=9024, toffoli=res.metrics.get("toffoli"),
                             qubits=res.metrics.get("qubits"), worktree=str(clean), details=verdict)
    ledger.record_best(res.score, res.metrics.get("toffoli", 0.0), res.metrics.get("qubits", 0), f"local:{exp_id}", verified=True)
    return verdict

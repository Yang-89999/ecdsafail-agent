"""ECDSA.fail adapter: github.com/ecdsafail/ecdsafail-challenge.

Score = average executed Toffoli per shot x peak qubits, lower is better.
Editable: src/point_add/. Protected: the harness, Cargo.*, rust-toolchain, results.tsv.
Official checker: ./benchmark.sh (sandboxed build_circuit, then trusted eval_circuit, 9024 shots).
Fast score: scripts/fast_score.sh (same pipeline, N shots, nothing appended to results.tsv).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from .base import CompetitionAdapter, ScoreResult

SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"


class EcdsaFailAdapter(CompetitionAdapter):
    name = "ecdsafail"
    editable_paths = ["src/point_add"]
    protected_paths = [
        "src/main.rs", "src/lib.rs", "src/circuit.rs", "src/sim.rs",
        "src/weierstrass_elliptic_curve.rs", "src/bin", "Cargo.toml", "Cargo.lock",
        "rust-toolchain", "results.tsv", "benchmark.sh", "setup.sh", "benchmark.json",
    ]

    # --- setup -------------------------------------------------------------
    def ensure_ready(self) -> None:
        if not (self.root / "Cargo.toml").exists():
            raise SystemExit(f"challenge checkout not found at {self.root}; set challenge_dir in agentlab.toml")
        if not (self.root / "target/release/eval_circuit").exists():
            subprocess.run(["bash", "setup.sh"], cwd=self.root, check=True)
        subprocess.run(["bash", str(SCRIPTS / "build_fast_eval.sh"), str(self.root)], check=True)

    def current_best(self) -> dict | None:
        """Official score of the upstream head, measured here and cached by commit.

        Yukon merges every verified gain into the shared repository, so the head
        of origin/main is the public baseline. results.tsv is NOT updated by the
        organisers (checked 2026-10-29: head e4076ae scored 914,355,746 while the
        last tsv row still said 3,962,753 x 2,715), so the only honest source is
        running benchmark.sh on a clean worktree of the head. Takes ~6 minutes;
        the result is cached in work/upstream_best.json.
        """
        self._git(self.root, "fetch", "-q", self.cfg.upstream_remote, self.cfg.upstream_branch)
        ref = f"{self.cfg.upstream_remote}/{self.cfg.upstream_branch}"
        commit = self._git(self.root, "rev-parse", "--short", ref).strip()
        cache = Path(self.cfg.ledger_path).parent / "upstream_best.json"
        data = json.loads(cache.read_text()) if cache.exists() else {}
        if commit in data:
            return data[commit]
        wt = Path(self.cfg.worktrees_dir) / f"upstream-{commit}"
        if not wt.exists():
            subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), ref], cwd=self.root, check=True)
        res = self.full_verify(wt, note=f"agentlab upstream head {commit}")
        if not res.ok:
            raise RuntimeError(f"upstream head {commit} failed the official benchmark: {res.reason}")
        data[commit] = {"score": res.score, "toffoli": res.metrics["toffoli"], "qubits": res.metrics["qubits"],
                        "commit": commit, "ts": int(time.time()), "measured_here": True}
        cache.write_text(json.dumps(data, indent=2))
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=self.root, check=False)
        return data[commit]

    # --- scoring -------------------------------------------------------------
    def fast_score(self, worktree: Path, shots: int) -> ScoreResult:
        t0 = time.time()
        log = Path(worktree) / ".agentlab" / f"fast_{int(t0)}.log"
        log.parent.mkdir(exist_ok=True)
        rc, out = _run_logged(
            ["bash", str(SCRIPTS / "fast_score.sh"), str(worktree), str(shots), str(self.root)],
            None, log, self.cfg.solver_timeout_s, f"fast score ({shots} shots)")
        return self._parse(out, rc, shots, str(log), time.time() - t0)

    def fresh_check(self, worktree: Path, draws: int = 8, shots: int = 9024) -> dict:
        """The same op stream tested on `draws` independent input sets (scripts/fresh_check.sh). The official
        harness tests one draw derived from the circuit's hash; this shows the real failure rate and the average
        Toffoli over many draws. Returns the script's JSON (ok, passed, failed_shots, mean_toffoli, ...)."""
        t0 = time.time()
        log = Path(worktree) / ".agentlab" / f"fresh_{int(t0)}.log"
        log.parent.mkdir(exist_ok=True)
        rc, out = _run_logged(
            ["bash", str(SCRIPTS / "fresh_check.sh"), str(worktree), str(draws), str(shots), str(self.root)],
            None, log, self.cfg.solver_timeout_s, f"fresh-inputs check ({draws} draws)")
        for line in reversed(out.strip().splitlines()):
            if line.startswith("{"):
                try:
                    res = json.loads(line)
                    res["log"] = str(log)
                    return res
                except json.JSONDecodeError:
                    break
        return {"ok": False, "reason": f"fresh_check produced no result (exit {rc})", "log": str(log)}

    def full_verify(self, worktree: Path, note: str = "") -> ScoreResult:
        t0 = time.time()
        log = Path(worktree) / ".agentlab" / f"full_{int(t0)}.log"
        log.parent.mkdir(exist_ok=True)
        rc, out = _run_logged(["bash", "benchmark.sh", "--note", note or "agentlab full verify"],
                              worktree, log, 6 * 3600, "official benchmark (9024 shots)")
        res = self._parse(out, rc, 9024, str(log), time.time() - t0)
        sj = Path(worktree) / "score.json"
        if res.ok and sj.exists():
            data = json.loads(sj.read_text())
            res.score = int(data["score"])
            res.metrics.update(data.get("metrics", {}))
        return res

    @staticmethod
    def _parse(stdout: str, rc: int, shots: int, log: str, dur: float) -> ScoreResult:
        m_tof = re.search(r"avg executed Toffoli\s*:\s*([0-9.]+)", stdout)
        m_q = re.search(r"qubits\s*:\s*(\d+)", stdout)
        m_fail = re.search(r"!!.*", stdout)
        if rc != 0 or not (m_tof and m_q):
            reason = (m_fail.group(0) if m_fail else f"exit {rc}")[:500]
            return ScoreResult(False, reason=reason, shots=shots, log_path=log, duration_s=dur)
        tof, q = float(m_tof.group(1)), int(m_q.group(1))
        # Rust's f64::round() rounds half away from zero; Python's round() is banker's rounding.
        return ScoreResult(True, score=int(tof + 0.5) * q, metrics={"toffoli": tof, "qubits": q},
                           shots=shots, log_path=log, duration_s=dur)

    # --- rules ---------------------------------------------------------------
    def rules_text(self) -> str:
        return (
            "You may modify anything inside src/point_add/ (split, rewrite, refactor freely).\n"
            "You may NOT touch src/main.rs, src/lib.rs, src/circuit.rs, src/sim.rs, src/bin/*, "
            "src/weierstrass_elliptic_curve.rs, Cargo.toml, Cargo.lock, rust-toolchain, benchmark.sh, setup.sh, "
            "or results.tsv (the harness appends to it).\n"
            "No new dependencies. The circuit must consume the four 256-wide registers and overwrite "
            "(target_x, target_y) with the affine sum on secp256k1.\n"
            "Validity: all shots classically correct; every freed ancilla back to |0>; zero global phase; "
            "forward then reverse restores every qubit. A Toffoli saving that breaks any of these is a FAIL, not a win.\n"
            "Score = average executed Toffoli per shot x peak qubit width. Lower is better.\n"
            "Never tune to the test set: do not change TAIL_NONCE or any other value whose only effect is which test "
            "inputs are drawn, and never choose between variants by whether their draw happened to pass. A candidate "
            "counts only if it also passes scripts/fresh_check.sh (independent input draws)."
        )

    # --- submission ----------------------------------------------------------
    def submit(self, worktree: Path, note: str) -> str:
        """Runs the official CLI. The orchestrator checks the approval record first."""
        cli = os.environ.get("ECDSAFAIL_CLI", "ecdsafail")
        proc = subprocess.run([cli, "submit"], cwd=worktree, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ecdsafail submit failed: {proc.stderr[-2000:]}")
        return proc.stdout[-4000:]


# Lines worth echoing to the terminal while a long scoring run is going.
_PROGRESS = ("-- ", "===", "!!", "  avg executed Toffoli", "  qubits", "  all ", "build_circuit:", "fast_eval:",
             "  emitted ops", "    Finished", "   Compiling quantum_ecc")


def _run_logged(cmd: list[str], cwd, log: Path, timeout: int, label: str) -> tuple[int, str]:
    """Run cmd, write everything to log, echo progress lines and a heartbeat every minute."""
    import threading

    print(f"[agentlab] {label}: started (full log: {log})", flush=True)
    t0 = time.time()
    lines: list[str] = []
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    done = threading.Event()

    def heartbeat():
        while not done.wait(60):
            print(f"[agentlab] {label}: still running, {int(time.time() - t0) // 60} min", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with open(log, "w") as fh:
            for line in proc.stdout:
                fh.write(line)
                lines.append(line)
                if line.startswith(_PROGRESS):
                    print(f"[agentlab]   {line.rstrip()}", flush=True)
                if time.time() - t0 > timeout:
                    proc.kill()
                    lines.append("!! timed out\n")
                    break
        rc = proc.wait()
    finally:
        done.set()
    print(f"[agentlab] {label}: finished in {int(time.time() - t0)} s (exit {rc})", flush=True)
    return rc, "".join(lines)

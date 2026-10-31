"""Clean-room track: our own point adder (cleanroom/point_add in this repo) on the unmodified ECDSA.fail harness.

The worktrees come from a separate git repository, work/cleanroom/harness, that holds only the harness files
(taken from the leaderboard checkout, without src/point_add) plus our module. No leaderboard submission is in
its history, so executors cannot read one even by accident.

  base     branch `main` of the harness repo = our current best (starts as cleanroom/point_add)
  bar      our own best, measured on the official evaluator; the target line is Google's 2.99e9
  audit    adds the module's unit tests; a verified gain is adopted automatically (harness main moves
           forward and cleanroom/point_add in this repo is updated and committed). Never submitted.
"""

from __future__ import annotations

import filecmp
import json
import shutil
import subprocess
import tarfile
import io
import time
from pathlib import Path

from .ecdsafail import SCRIPTS, EcdsaFailAdapter

ROOT = Path(__file__).resolve().parent.parent.parent
GIT_ID = ["-c", "user.name=agentlab", "-c", "user.email=agentlab@localhost"]


def _git(cwd: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *GIT_ID, *args], cwd=cwd, capture_output=True, text=True, check=check)


def _same_tree(a: Path, b: Path) -> bool:
    """True when two directories hold the same files with the same bytes (ignoring __pycache__ and dotfiles)."""
    def files(d: Path) -> set[str]:
        return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and not any(x.startswith(".") for x in p.relative_to(d).parts)}
    fa, fb = files(a), files(b)
    if fa != fb:
        return False
    return all(filecmp.cmp(a / f, b / f, shallow=False) for f in fa)


def _copy_module(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".*", "__pycache__", "target"))


class CleanroomAdapter(EcdsaFailAdapter):
    name = "ecdsafail-cleanroom"
    kind = "cleanroom"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.upstream = Path(cfg.challenge_dir)          # leaderboard checkout: only its harness files are used
        self.root = Path(cfg.cleanroom_harness)          # worktrees, scoring and audits happen here
        self.src = Path(cfg.cleanroom_src)               # our module, the source of truth in this repo

    # --- base ----------------------------------------------------------------
    def base_ref(self) -> str:
        return "main"

    def refresh(self) -> None:
        self.sync_base()

    def sync_base(self) -> str:
        """Create the harness repo on first use; afterwards make its main follow cleanroom/point_add when the
        module in this repo changed (a `git pull` of new module code). Returns the main commit."""
        r = self.root
        if not (r / ".git").exists():
            r.mkdir(parents=True, exist_ok=True)
            ref = f"{self.cfg.upstream_remote}/{self.cfg.upstream_branch}"
            src_commit = _git(self.upstream, "rev-parse", "--short", ref).stdout.strip() or "HEAD"
            if src_commit == "HEAD":
                ref = "HEAD"
                src_commit = _git(self.upstream, "rev-parse", "--short", "HEAD").stdout.strip()
            arch = subprocess.run(["git", "archive", ref], cwd=self.upstream, capture_output=True, check=True).stdout
            with tarfile.open(fileobj=io.BytesIO(arch)) as tf:
                members = [m for m in tf.getmembers() if not (m.name == "src/point_add" or m.name.startswith("src/point_add/"))]
                tf.extractall(r, members=members)
            (r / ".gitignore").write_text("target/\nops.bin\nscore.json\n.agentlab/\n")
            (r / "HARNESS_FROM").write_text(f"harness files from the ECDSA.fail repository at {src_commit} (src/point_add removed)\n")
            _copy_module(self.src, r / "src" / "point_add")
            _git(r, "init", "-q", "-b", "main", check=True)
            _git(r, "add", "-A", check=True)
            _git(r, "commit", "-q", "-m", f"clean-room base: harness {src_commit} + our module", check=True)
            print(f"[cleanroom] created the harness repo {r} (harness {src_commit}, no leaderboard history)", flush=True)
        elif not _same_tree(self.src, r / "src" / "point_add"):
            head = _git(ROOT, "rev-parse", "--short", "HEAD").stdout.strip()
            _git(r, "checkout", "-q", "main")
            _copy_module(self.src, r / "src" / "point_add")
            _git(r, "add", "-A", "src/point_add", check=True)
            _git(r, "commit", "-q", "-m", f"module from cleanroom/point_add (agent repo {head})", check=True)
            print(f"[cleanroom] the module in cleanroom/point_add changed: new base commit", flush=True)
        return _git(r, "rev-parse", "--short", "main").stdout.strip()

    def ensure_ready(self) -> None:
        if not (self.upstream / "Cargo.toml").exists():
            raise SystemExit(f"leaderboard checkout not found at {self.upstream} (its harness files seed the clean-room repo)")
        self.sync_base()
        subprocess.run(["bash", str(SCRIPTS / "build_fast_eval.sh"), str(self.root)], check=True)

    def current_best(self) -> dict | None:
        """Official score of our base (harness main), measured here and cached per commit."""
        commit = self.sync_base()
        cache = Path(self.cfg.ledger_path).parent / "base_best.json"
        data = json.loads(cache.read_text()) if cache.exists() else {}
        if commit not in data:
            wt = Path(self.cfg.worktrees_dir) / f"base-{commit}"
            if not wt.exists():
                subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), commit], cwd=self.root, check=True)
            res = self.full_verify(wt, note=f"clean-room base {commit}")
            if not res.ok:
                raise RuntimeError(f"clean-room base {commit} failed the official benchmark: {res.reason}")
            data[commit] = {"score": res.score, "toffoli": res.metrics["toffoli"], "qubits": res.metrics["qubits"],
                            "commit": commit, "ts": int(time.time()), "measured_here": True}
            cache.write_text(json.dumps(data, indent=2))
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=self.root, check=False)
        return dict(data[commit], target=self.cfg.target_score, label="our clean-room base")

    # --- audit hooks -------------------------------------------------------------
    def extra_checks(self, wt: Path) -> dict:
        """The module's unit tests (values, ancillas, phase) must pass."""
        t0 = time.time()
        proc = subprocess.run(["cargo", "test", "--release", "--locked", "--offline", "--bin", "build_circuit", "--",
                               "--nocapture", "--test-threads", "2"], cwd=wt, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        units = [l for l in out.splitlines() if l.startswith("UNIT ")]
        summary = [l for l in out.splitlines() if l.startswith("test result:")]
        return {"ok": proc.returncode == 0, "name": "unit_tests_pass", "units": units, "summary": summary,
                "tail": "" if proc.returncode == 0 else out[-1500:], "duration_s": int(time.time() - t0)}

    def on_win(self, exp_id: int, clean_wt: Path, verdict: dict) -> str:
        """Adopt an audited gain as the new base: move harness main to the audited tree and commit the module
        into cleanroom/point_add here. Internal and reversible (git history keeps every base)."""
        r = self.root
        branch = _git(clean_wt, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        _git(r, "checkout", "-q", "main")
        ff = _git(r, "merge", "-q", "--ff-only", branch)
        if ff.returncode != 0:
            commits = _git(clean_wt, "rev-list", "--reverse", f"main..{branch}").stdout.split()
            cp = _git(r, "cherry-pick", *commits)
            if cp.returncode != 0:
                _git(r, "cherry-pick", "--abort")
                return f"not adopted: does not apply on the current base ({cp.stderr[-300:]})"
        _copy_module(r / "src" / "point_add", self.src)
        score = (verdict.get("full") or {}).get("score")
        _git(ROOT, "add", "-A", str(self.src.relative_to(ROOT)))
        msg = f"clean-room: adopt exp {exp_id} (score {score:,})" if score else f"clean-room: adopt exp {exp_id}"
        c = _git(ROOT, "commit", "-q", "-m", msg)
        new = _git(r, "rev-parse", "--short", "main").stdout.strip()
        return f"adopted as base {new}" + ("" if c.returncode == 0 else f" (agent repo commit failed: {c.stderr[-200:]})")

    # --- rules -------------------------------------------------------------------
    def rules_text(self) -> str:
        return (
            "CLEAN-ROOM TRACK. You are improving OUR OWN secp256k1 point adder, src/point_add/ (a copy of cleanroom/point_add\n"
            "in the agentlab repo), written from published algorithms. This repository contains no leaderboard submission and\n"
            "you must not obtain one: do not fetch, read or reproduce code from the ECDSA.fail leaderboard, its git history,\n"
            "or other authors' implementations (e.g. Schrottenloher's Qarton code). Ideas from papers are welcome; cite them in a\n"
            "code comment.\n"
            "You may modify anything inside src/point_add/ (split, rewrite, add tests). You may NOT touch the harness:\n"
            "src/main.rs, src/lib.rs, src/circuit.rs, src/sim.rs, src/bin/*, src/weierstrass_elliptic_curve.rs, Cargo.toml,\n"
            "Cargo.lock, rust-toolchain, benchmark.sh, setup.sh, results.tsv. No new dependencies.\n"
            "Validity: the official evaluator must pass (classical results, every freed ancilla |0>, zero global phase);\n"
            "the module's unit tests must pass:\n"
            "    cargo test --release --bin build_circuit -- --nocapture --test-threads 2\n"
            "(they print `UNIT <name> toffoli=... width=...`; keep them passing, add tests for every new primitive);\n"
            f"and the error rate on independent inputs (scripts/fresh_check.sh) must be at most {self.cfg.max_error_rate:g}.\n"
            "Exact beats approximate at equal score: spend error budget only where it buys a measured gain, and measure the\n"
            "failure rate of an approximation with a classical model before building it.\n"
            "Score = average executed Toffoli per shot x peak qubit width. Lower is better. Our bar is our own current best;\n"
            f"the target is Google's 2.1M x 1,425 = {self.cfg.target_score:,}.\n"
            "Never change anything whose only effect is which test inputs are drawn, and never choose between variants by\n"
            "whether their draw happened to pass."
        )

    def track_brief(self) -> str:
        p = ROOT / "docs" / "cleanroom.md"
        return p.read_text()[:7000] if p.exists() else ""

    def submit(self, worktree: Path, note: str) -> str:
        raise RuntimeError("the clean-room track never submits to the leaderboard; ask Yang")

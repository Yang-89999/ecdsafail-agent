"""Configuration: agentlab.toml in the repo root, overridable by environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Paths
    challenge_dir: Path = ROOT.parent / "ecdsafail-challenge"  # main checkout of the harness
    worktrees_dir: Path = ROOT / "work" / "worktrees"
    ledger_path: Path = ROOT / "work" / "ledger.sqlite"
    runs_dir: Path = ROOT / "work" / "runs"
    skills_dir: Path = ROOT / "skills"
    # Competition
    adapter: str = "ecdsafail"
    editable_path: str = "src/point_add"
    upstream_remote: str = "origin"
    upstream_branch: str = "main"
    # Models
    backend: str = "codex"  # "codex" (codex exec CLI) or "openai" (Responses API tool loop)
    model: str = ""  # empty = backend default; e.g. a Codex model name
    reasoning_effort: str = "high"
    # Search
    solvers: int = 2  # parallel solvers
    fast_shots: int = 1024  # shots for the cheap score (16 simulator batches, ~1 min; 64 shots showed 0.016 % noise)
    audit_threshold: float = 0.0005  # min fast-score relative gain before the full official run
    solver_timeout_s: int = 3 * 3600
    explore_c: float = 0.6  # UCB exploration constant on relative gain
    # Submission policy
    submit_only_if_beats_best: bool = True
    min_relative_gain: float = 0.0  # extra margin over the best, e.g. 0.001 = 0.1 %
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        cfg = cls()
        path = path or ROOT / "agentlab.toml"
        if path.exists():
            data = tomllib.loads(path.read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    cur = getattr(cfg, k)
                    setattr(cfg, k, Path(v).expanduser() if isinstance(cur, Path) else v)
                else:
                    cfg.extra[k] = v
        for k in ("backend", "model", "challenge_dir", "solvers", "fast_shots"):
            env = os.environ.get("AGENTLAB_" + k.upper())
            if env:
                cur = getattr(cfg, k)
                setattr(cfg, k, Path(env) if isinstance(cur, Path) else type(cur)(env))
        for d in (cfg.worktrees_dir, cfg.runs_dir, cfg.ledger_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        # rustup installs to ~/.cargo/bin; a shell opened before setup.sh ran does not have it on PATH yet.
        cargo_bin = str(Path.home() / ".cargo" / "bin")
        if cargo_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cargo_bin + os.pathsep + os.environ.get("PATH", "")
        return cfg

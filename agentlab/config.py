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
    # Clean-room track (adapter "ecdsafail-cleanroom", agentlab-cleanroom.toml)
    cleanroom_src: Path = ROOT / "cleanroom" / "point_add"  # our module (source of truth)
    cleanroom_harness: Path = ROOT / "work" / "cleanroom" / "harness"  # harness-only git repo the worktrees come from
    target_score: int = 2_992_500_000  # Google's low-gate point, 2.1M Toffoli x 1,425 qubits
    # Models
    backend: str = "codex"  # "codex" (codex exec CLI) or "openai" (Responses API tool loop)
    model: str = ""  # empty = backend default; e.g. a Codex model name
    reasoning_effort: str = "high"
    # Search
    solvers: int = 2  # parallel executor slots
    fast_shots: int = 9024  # the official shot count: eval takes seconds, the build dominates, so the cheap score is the official number
    audit_threshold: float = 0.0  # any strict gain goes to the audit; recent leaderboard steps are ~0.01 %
    fresh_draws: int = 8  # independent input draws for scripts/fresh_check.sh (head baseline, audit)
    max_error_rate: float = 1e-4  # 2^-13.3: the error level of the published reference circuits (Babbush et al., Schrottenloher)
    solver_timeout_s: int = 8 * 3600  # safety cap only: a run ends when its executor finds a gain or no hope
    explore_c: float = 0.6  # UCB exploration constant on relative gain
    # Submission policy
    submit_only_if_beats_best: bool = True
    min_relative_gain: float = 0.0  # extra margin over the best, e.g. 0.001 = 0.1 %
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """agentlab.toml, then an overlay: `--config FILE` or the AGENTLAB_CONFIG environment variable (e.g.
        agentlab-cleanroom.toml switches to the clean-room track). Relative paths are relative to the repo."""
        cfg = cls()
        overlay = path or (Path(os.environ["AGENTLAB_CONFIG"]) if os.environ.get("AGENTLAB_CONFIG") else None)
        if overlay is not None and not overlay.is_absolute() and not overlay.exists():
            overlay = ROOT / overlay
        for p in [ROOT / "agentlab.toml"] + ([overlay] if overlay and overlay.resolve() != (ROOT / "agentlab.toml").resolve() else []):
            if not p.exists():
                continue
            data = tomllib.loads(p.read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    cur = getattr(cfg, k)
                    if isinstance(cur, Path):
                        v = Path(v).expanduser()
                        v = v if v.is_absolute() else ROOT / v
                    setattr(cfg, k, v)
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

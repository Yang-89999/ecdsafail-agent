"""The one interface the rest of the system talks to a competition through.

A new Yukon board = a new subclass plus a domain skill pack. Nothing else
in agentlab should know what a Toffoli or a RISC-V cycle is.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScoreResult:
    ok: bool
    score: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    shots: int | None = None
    reason: str = ""
    log_path: str = ""
    duration_s: float = 0.0


class CompetitionAdapter:
    name = "base"
    kind = "leaderboard"  # "leaderboard": beat the public head and submit; "cleanroom": beat our own base, never submit
    editable_paths: list[str] = []
    protected_paths: list[str] = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.root: Path = Path(cfg.challenge_dir)

    # --- track ---------------------------------------------------------------
    def base_ref(self) -> str:
        """The ref every new experiment worktree starts from."""
        return f"{self.cfg.upstream_remote}/{self.cfg.upstream_branch}"

    def refresh(self) -> None:
        """Bring the base up to date (fetch the shared repo)."""
        self._git(self.root, "fetch", "-q", self.cfg.upstream_remote, self.cfg.upstream_branch)

    def extra_checks(self, wt: Path) -> dict | None:
        """Track-specific audit check; None when there is none."""
        return None

    def on_win(self, exp_id: int, clean_wt: Path, verdict: dict) -> str | None:
        """Called when an audit beats the bar. Leaderboard: nothing (a human approves and submits)."""
        return None

    def track_brief(self) -> str:
        """Design brief the thinking roles read instead of the leaderboard trend (empty on the leaderboard track)."""
        return ""

    # --- setup -------------------------------------------------------------
    def ensure_ready(self) -> None:
        """Toolchain, deps, prebuilt binaries. Idempotent."""
        raise NotImplementedError

    def current_best(self) -> dict | None:
        """Best public score known to the adapter (from the shared repo, API or board)."""
        raise NotImplementedError

    # --- scoring -------------------------------------------------------------
    def fast_score(self, worktree: Path, shots: int) -> ScoreResult:
        """Cheap approximation of the official metric. Never used for a claim."""
        raise NotImplementedError

    def full_verify(self, worktree: Path, note: str = "") -> ScoreResult:
        """The official checker, exactly as CI runs it."""
        raise NotImplementedError

    # --- rules ---------------------------------------------------------------
    def protected_diff(self, worktree: Path, base: str) -> list[str]:
        """Files changed outside the editable paths (must be empty)."""
        out = self._git(worktree, "diff", "--name-only", base, "--")
        changed = [l for l in out.splitlines() if l.strip()]
        untracked = self._git(worktree, "ls-files", "--others", "--exclude-standard").splitlines()
        # Scratch the system itself leaves in a worktree (logs, harness outputs) is not a rule violation.
        scratch = (".agentlab/", "ops.bin", "score.json", "target/")
        bad = []
        for f in changed + [u for u in untracked if u.strip()]:
            if f.startswith(scratch) or f in scratch:
                continue
            if not any(f == p or f.startswith(p.rstrip("/") + "/") for p in self.editable_paths):
                bad.append(f)
        return bad

    def rules_text(self) -> str:
        """Rules the solver prompt quotes verbatim."""
        raise NotImplementedError

    # --- submission ----------------------------------------------------------
    def submit(self, worktree: Path, note: str) -> str:
        """Submit through the official channel. Caller enforces the approval gate."""
        raise NotImplementedError

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _git(cwd: Path, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False).stdout

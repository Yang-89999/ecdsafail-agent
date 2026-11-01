"""Win mode for the ECDSA.fail leaderboard (adapter "ecdsafail-win", agentlab-win.toml, docs/win.md).

The official harness tests ONE input draw derived from a hash of the op stream. The public head fails about 20 of
its 141 batches on a typical draw and passes officially only because its TAIL_NONCE was searched for a clean draw
(the organisers allow this). So any edit redraws the inputs and fails, and the official verdict says nothing about
an edit. Here:

  score    the op stream on fixed shared draws (salts 0..win_fast_draws-1) with the official simulator and checks
           (tools/nonce_search crnfast), paired draw by draw with our base and the public head
           (agentlab/winscore.py): effective Toffoli x qubits, the error rate free up to +win_lambda_budget lambda
           over the public head and charged win_tof_per_lambda Toffoli per lambda beyond
  base     branch win_base_branch in the challenge repo; starts at the public head; an audited gain is adopted
           (the branch moves), so gains stack; rebased when the public head moves
  audit    the usual checks, then the same score on win_audit_draws other draws (salts from win_audit_salt)
  submit   not from here: a nonce search for a clean official draw on the base, the official run, then Yang's approval
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .base import ScoreResult
from .ecdsafail import SCRIPTS, EcdsaFailAdapter, _run_logged
from .. import winscore

ROOT = Path(__file__).resolve().parent.parent.parent
GIT_ID = ["-c", "user.name=agentlab", "-c", "user.email=agentlab@localhost", "-c", "safe.directory=*"]
NS_VERSION = "3"  # tools/nonce_search/.agentlab-version written by build_nonce_search.sh

LESSONS_HEADER = (
    "<!-- Copied from the official-draw track when win mode started. CAUTION: on that track every edit redrew the "
    "official inputs, and the unmodified head fails ~20 batches on a typical draw, so 'rejected', 'fails validation' or "
    "'restored production' there mostly reflects the draw, not the circuit. Re-check such conclusions with win_score.sh. -->\n\n")


def _budget_words(c) -> str:
    rate = f"each lambda above that costs {c.win_tof_per_lambda:,.0f} Toffoli per shot"
    if c.win_lambda_budget > 0:
        return f"may be up to +{c.win_lambda_budget:g} above the public head's for free; {rate}"
    return f"is free up to the public head's; {rate}"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *GIT_ID, *args], cwd=cwd, capture_output=True, text=True)


class EcdsaFailWinAdapter(EcdsaFailAdapter):
    name = "ecdsafail-win"
    kind = "leaderboard"
    uses_fresh_check = False  # the error rate is part of the score
    wants_trend = False  # the official-draw track's trend.md is copied in; re-measuring it is not needed
    own_base = True
    seed_set = "win"
    verify_label = "crn_audit_ok"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.work = Path(cfg.runs_dir).parent
        self.ref_path = self.work / "crn_ref.json"
        self.cache_path = self.work / "crn_cache.json"
        self._last: dict[str, dict] = {}  # worktree -> {"qubits", "draws"} of its last scoring run
        # Solvers (and sweep.py through fast_score.sh) inherit these: fast_score.sh switches to win_score.sh.
        os.environ.update(AGENTLAB_SCORE_MODE="win", AGENTLAB_WIN_REF=str(self.ref_path),
                          AGENTLAB_WIN_DRAWS=str(cfg.win_fast_draws), AGENTLAB_WIN_BUDGET=str(cfg.win_lambda_budget),
                          AGENTLAB_WIN_RATE=str(cfg.win_tof_per_lambda), AGENTLAB_CHALLENGE=str(self.root))

    # --- base ------------------------------------------------------------------
    def upstream_ref(self) -> str:
        return f"{self.cfg.upstream_remote}/{self.cfg.upstream_branch}"

    def base_ref(self) -> str:
        return self.cfg.win_base_branch

    def refresh(self) -> None:
        self._git(self.root, "fetch", "-q", self.cfg.upstream_remote, self.cfg.upstream_branch)
        self.sync_base()

    def _short(self, ref: str) -> str:
        return _git(self.root, "rev-parse", "--short", ref).stdout.strip()

    def sync_base(self) -> str:
        """Create the base branch at the public head on first use; when the public head moves, replay our adopted
        commits on top of it (kept as they are if they do not apply). Returns the base commit."""
        base, up = self.cfg.win_base_branch, self.upstream_ref()
        if _git(self.root, "rev-parse", "--verify", "-q", base).returncode != 0:
            _git(self.root, "branch", base, up)
            print(f"[win] created our base {base} at the public head {self._short(up)}", flush=True)
        elif _git(self.root, "merge-base", "--is-ancestor", up, base).returncode != 0:
            ours = _git(self.root, "rev-list", "--reverse", "--no-merges", f"{up}..{base}").stdout.split()
            old = self._short(base)
            if not ours:
                _git(self.root, "branch", "-f", base, up)
            else:
                tmp = Path(self.cfg.worktrees_dir) / "rebase-base"
                _git(self.root, "worktree", "remove", "--force", str(tmp))
                _git(self.root, "worktree", "add", "-q", "--detach", str(tmp), up)
                cp = _git(tmp, "cherry-pick", "-x", "--keep-redundant-commits", *ours)
                if cp.returncode == 0:
                    _git(self.root, "branch", "-f", base, _git(tmp, "rev-parse", "HEAD").stdout.strip())
                else:
                    _git(tmp, "cherry-pick", "--abort")
                    print(f"[win] the public head moved to {self._short(up)} but our {len(ours)} adopted commit(s) do not "
                          f"apply on it; keeping our base {old} (resolve by hand)", flush=True)
                _git(self.root, "worktree", "remove", "--force", str(tmp))
            if self._short(base) != old:
                print(f"[win] the public head moved: our base {old} -> {self._short(base)} (on {self._short(up)})", flush=True)
        return self._short(base)

    def ensure_ready(self) -> None:
        super().ensure_ready()
        self._build_ns()
        self._seed_state()
        self.refresh()

    def _ns(self) -> Path:
        return ROOT / "tools" / "nonce_search" / "target" / "release" / "nonce_search"

    def _build_ns(self) -> None:
        stamp = ROOT / "tools" / "nonce_search" / ".agentlab-version"
        if not self._ns().exists() or (stamp.read_text().strip() if stamp.exists() else "") != NS_VERSION:
            subprocess.run(["bash", str(SCRIPTS / "build_nonce_search.sh"), str(self.root)], check=True)

    def _seed_state(self) -> None:
        """First start: carry over what the official-draw track learned (its lessons with a caution header)."""
        old = ROOT / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        for name in ("lessons.md", "trend.md", "trend.json", "problem.md", "profile.md"):
            src, dst = old / name, self.work / name
            if src.exists() and not dst.exists() and src.resolve() != dst.resolve():
                text = src.read_text(errors="replace")
                dst.write_text(LESSONS_HEADER + text if name == "lessons.md" else text)
                print(f"[win] carried over {name} from the official-draw track", flush=True)

    # --- shared-draw results per commit ------------------------------------------
    def _ranges(self) -> list[tuple[int, int]]:
        return [(0, self.cfg.win_fast_draws), (self.cfg.win_audit_salt, self.cfg.win_audit_draws)]

    def _crn_for_commit(self, commit: str) -> dict:
        """{"commit", "qubits", "draws": {salt: [failing, classical, phase, toffoli total, shots]}} on the score and audit
        draws, measured once per commit (one build, then crnfast) and cached in work/win/crn_cache.json."""
        cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        ent = cache.get(commit) or {"commit": commit, "draws": {}}
        missing = [(s, n) for s, n in self._ranges() if any(str(x) not in ent["draws"] for x in range(s, s + n))]
        if not missing:
            return ent
        wt = Path(self.cfg.worktrees_dir) / f"crn-{commit}"
        _git(self.root, "worktree", "remove", "--force", str(wt))
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), commit], cwd=self.root, check=True)
        out = self.work / "crn_ops" / commit
        log = self.work / "crn_ops" / f"{commit}.log"
        out.mkdir(parents=True, exist_ok=True)
        try:
            rc, txt = _run_logged(["bash", str(SCRIPTS / "build_ops.sh"), str(wt), str(out)], None, log,
                                  self.cfg.solver_timeout_s, f"build {commit} for the shared-draw reference")
            if rc != 0:
                raise RuntimeError(f"could not build {commit}: {txt[-500:]}")
            for s, n in missing:
                rc, txt = _run_logged([str(self._ns()), "crnfast", str(s), str(n)], out, log.with_suffix(f".{s}.log"),
                                      self.cfg.solver_timeout_s, f"{commit} on {n} shared draws from salt {s}")
                crn = winscore.parse_crn_output(txt)
                if rc != 0 or not crn:
                    raise RuntimeError(f"crnfast failed on {commit} (exit {rc}): {txt[-500:]}")
                ent["draws"].update(winscore.rows_to_draws(crn["per_draw"]))
                ent["qubits"] = int(crn["qubits"])
        finally:
            (out / "ops.bin").unlink(missing_ok=True)
            _git(self.root, "worktree", "remove", "--force", str(wt))
        cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        cache[commit] = ent
        self.cache_path.write_text(json.dumps(cache))
        return ent

    def current_best(self) -> dict | None:
        """Our base's score on the audit draws (the bar), its score on the fast draws (the reference gains are measured
        against), and the public head's numbers. Also writes work/win/crn_ref.json for win_score.sh."""
        self.refresh()
        pub, base = self._short(self.upstream_ref()), self._short(self.cfg.win_base_branch)
        pubd = self._crn_for_commit(pub)
        based = pubd if base == pub else self._crn_for_commit(base)
        ref = {"base": based, "public": pubd}
        tmp = self.ref_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ref))
        tmp.replace(self.ref_path)

        def on(salts):
            d = {str(s): based["draws"][str(s)] for s in salts}
            return winscore.score_draws(d, based["qubits"], ref, self.cfg.win_lambda_budget, self.cfg.win_tof_per_lambda)

        a = on(range(self.cfg.win_audit_salt, self.cfg.win_audit_salt + self.cfg.win_audit_draws))
        f = on(range(self.cfg.win_fast_draws))
        return {"score": a["score"], "toffoli": a["toffoli"], "qubits": based["qubits"], "commit": base, "public_commit": pub,
                "fast_score": f["score"], "lambda": round(a["lambda"], 3), "public_lambda": round(a.get("public_lambda", a["lambda"]), 3),
                "lead_tof": round(a.get("lead_tof", 0.0), 3), "label": "our win base" if base != pub else "the public head (nothing adopted yet)",
                "mode": "win", "ts": int(time.time()), "measured_here": True}

    # --- scoring ---------------------------------------------------------------
    def _win_score(self, worktree: Path, draws: int, salt: int, label: str) -> ScoreResult:
        t0 = time.time()
        log = Path(worktree) / ".agentlab" / f"win_{int(t0)}.log"
        log.parent.mkdir(exist_ok=True)
        rc, out = _run_logged(["bash", str(SCRIPTS / "win_score.sh"), str(worktree), str(draws), str(self.root), str(salt),
                               str(self.ref_path)], None, log, self.cfg.solver_timeout_s, label)
        d = None
        for line in reversed(out.strip().splitlines()):
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    pass
                break
        dur = time.time() - t0
        if not d or not d.get("ok") or d.get("score") is None:
            reason = (d or {}).get("reason") or next((l for l in out.splitlines() if l.startswith("!!")), f"exit {rc}")
            return ScoreResult(False, reason=reason[:500], shots=draws * 9024, log_path=str(log), duration_s=dur)
        if d.get("per_draw"):
            self._last[str(Path(worktree).resolve())] = {"qubits": d["qubits"], "draws": winscore.rows_to_draws(d["per_draw"])}
        return ScoreResult(True, score=int(d["score"]), metrics=d.get("metrics") or {}, shots=draws * 9024,
                           log_path=str(log), duration_s=dur)

    def fast_score(self, worktree: Path, shots: int) -> ScoreResult:
        return self._win_score(worktree, self.cfg.win_fast_draws, 0, f"win score ({self.cfg.win_fast_draws} shared draws)")

    def full_verify(self, worktree: Path, note: str = "") -> ScoreResult:
        return self._win_score(worktree, self.cfg.win_audit_draws, self.cfg.win_audit_salt,
                               f"win audit ({self.cfg.win_audit_draws} other shared draws)")

    def fresh_check(self, worktree: Path, draws: int = 8, shots: int = 9024) -> dict:
        return {"ok": True, "skipped": "win mode: the error rate is part of the shared-draw score"}

    # --- audit hooks -------------------------------------------------------------
    def on_win(self, exp_id: int, clean_wt: Path, verdict: dict) -> str:
        """Adopt an audited gain: move our base branch to the audited tree (internal and reversible; git keeps every
        base), seed the reference cache with the audit's draws, and log it to work/win/candidates.md."""
        base = self.cfg.win_base_branch
        head = _git(clean_wt, "rev-parse", "HEAD").stdout.strip()
        if _git(self.root, "merge-base", "--is-ancestor", base, head).returncode == 0:
            _git(self.root, "branch", "-f", base, head)
        else:
            commits = _git(clean_wt, "rev-list", "--reverse", "--no-merges", f"{base}..{head}").stdout.split()
            tmp = Path(self.cfg.worktrees_dir) / "adopt-base"
            _git(self.root, "worktree", "remove", "--force", str(tmp))
            _git(self.root, "worktree", "add", "-q", "--detach", str(tmp), base)
            cp = _git(tmp, "cherry-pick", "-x", "--keep-redundant-commits", *commits)
            if cp.returncode != 0:
                _git(tmp, "cherry-pick", "--abort")
                _git(self.root, "worktree", "remove", "--force", str(tmp))
                return f"not adopted: does not apply on the current base ({cp.stderr[-300:]})"
            _git(self.root, "branch", "-f", base, _git(tmp, "rev-parse", "HEAD").stdout.strip())
            _git(self.root, "worktree", "remove", "--force", str(tmp))
        new = self._short(base)
        last = self._last.get(str(Path(clean_wt).resolve()))
        same_tree = _git(self.root, "rev-parse", f"{base}^{{tree}}").stdout == _git(clean_wt, "rev-parse", "HEAD^{tree}").stdout
        if last and same_tree:
            cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
            ent = cache.get(new) or {"commit": new, "draws": {}}
            ent["draws"].update(last["draws"])
            ent["qubits"] = last["qubits"]
            cache[new] = ent
            self.cache_path.write_text(json.dumps(cache))
        m = (verdict.get("full") or {}).get("metrics") or {}
        line = (f"- {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}  exp {exp_id} adopted as base {new}: score "
                f"{(verdict.get('full') or {}).get('score')}, effective Toffoli {m.get('toffoli', 0):,.3f} x {m.get('qubits')}, "
                f"change vs previous base {m.get('d_tof', 0):+.3f} +- {m.get('d_tof_se', 0):.3f} Toffoli/shot, lambda vs public head "
                f"{m.get('d_lambda', 0):+.2f} +- {m.get('d_lambda_se', 0):.2f}, lead over the public head {m.get('lead_tof', 0):+.3f} "
                f"Toffoli/shot ({m.get('draws')} audit draws)\n")
        cand = self.work / "candidates.md"
        if not cand.exists():
            cand.write_text("# Win-mode base history\n\nEach line is an audited gain adopted as our base (branch "
                            f"{base} in the challenge repo). Submitting the base needs a nonce search for a clean official "
                            "draw, the official run, then Yang's approval.\n\n")
        with cand.open("a") as fh:
            fh.write(line)
        return f"adopted as win base {new} (lead over the public head {m.get('lead_tof', 0):+.3f} Toffoli/shot)"

    # --- prompts -----------------------------------------------------------------
    def rules_text(self) -> str:
        c = self.cfg
        return (
            "You may modify anything inside src/point_add/ (split, rewrite, refactor freely).\n"
            "You may NOT touch src/main.rs, src/lib.rs, src/circuit.rs, src/sim.rs, src/bin/*, "
            "src/weierstrass_elliptic_curve.rs, Cargo.toml, Cargo.lock, rust-toolchain, benchmark.sh, setup.sh, "
            "or results.tsv.\n"
            "No new dependencies. The circuit must consume the four 256-wide registers and overwrite "
            "(target_x, target_y) with the affine sum on secp256k1; every freed ancilla back to |0>; phase garbage counts as "
            "a failure.\n"
            "WIN MODE. Approximate circuits are allowed (the organisers' rules: exactness is not required). A circuit is "
            "scored on fixed shared input draws by scripts/win_score.sh: effective Toffoli x qubits, where the error rate "
            f"(lambda, failing batches per draw) {_budget_words(c)}. Lower is better.\n"
            "Do NOT change TAIL_NONCE or anything whose only effect is which inputs the official harness draws: the nonce "
            "search for a clean official draw is done at the end, outside the experiments. The official evaluator's "
            "one-draw verdict (fast_score.sh without win mode, benchmark.sh, fresh_check.sh) is not the measure here."
        )

    def scoring_text(self, cfg, wt: Path, scripts: Path, best: dict | None) -> str:
        b = best or {}
        pub_l = b.get("public_lambda")
        return f"""HOW TO SCORE (win mode; about 4-5 minutes: the build and build_circuit dominate, the 24 draws take about a minute):
    bash {scripts}/win_score.sh {wt} {cfg.win_fast_draws} {self.root}
It builds your tree and runs the op stream on {cfg.win_fast_draws} fixed input draws (the same draws for every circuit)
with the official simulator and checks, then compares draw by draw with our base ({b.get('commit', '?')}) and the public
head ({b.get('public_commit', '?')}). The last line of its output is JSON:
    {{"ok": true, "score": ..., "metrics": {{"toffoli": <effective>, "qubits": ..., "d_tof": ..., "d_lambda": ..., "gain": ..., ...}}}}
score = effective Toffoli x qubits. Effective Toffoli = average executed Toffoli per shot over the draws, plus a penalty
when the error rate is too high: lambda = expected failing batches (of 64 shots) per 9,024-shot draw; the public head's
is about {pub_l if pub_l is not None else '22'}; yours {_budget_words(cfg)}. Why: before submitting we search
TAIL_NONCE for a clean official draw, which costs x2.7 compute per extra lambda; the leaders gate every change at
"lambda est + 2 SE <= 19.5" (their lambda counts failing shots per draw: the head is ~19.4 on their scale, ~22.0 on
our batch scale) and trade lambda for Toffoli at about this price (their notes in src/point_add/leapfrog.rs).
Buying lambda back below the public head's earns nothing by itself, but it frees room for a Toffoli saving elsewhere.
Precision: d_tof is paired (identical inputs), so an exact Toffoli saving shows exactly (e.g. -1.000 +- 0.000) and even a
saving of a fraction of a Toffoli per shot is real if its error bar excludes 0. d_lambda has a real error bar; to decide on
an approximation, rerun with more draws (2nd argument, e.g. 96). Only ok=true counts.
The official harness's own verdict is meaningless here: any edit redraws its single input draw, and the unmodified head
fails about 20 batches on a typical draw. Do not use fresh_check.sh or an official-draw "ok=false" as evidence.

SWEEPING MANY VARIANTS (unattended while you think; each variant costs one full score, so sweep a handful at a time):
    AGENTLAB_SCORE_MODE=win python3 {scripts}/sweep.py {wt} --mutate "python3 .agentlab/mutate.py {{v}}" --values 0,1,2,3 --shots 9024 --challenge {self.root} --keep-best
    AGENTLAB_SCORE_MODE=win python3 {scripts}/sweep.py {wt} --patches .agentlab/variants --shots 9024 --challenge {self.root} --keep-best
(sweep.py calls fast_score.sh, which uses this score when AGENTLAB_SCORE_MODE=win.) Write a small mutation script (or a
folder of patches) that turns one knob of your idea, commit your tree, and sweep; --keep-best leaves the best variant
applied for you to check and commit."""

    def track_note(self, best: dict | None) -> str:
        b = best or {}
        c = self.cfg
        return (f"WIN MODE: our base {b.get('commit', '?')} ({b.get('label', '')}); public head {b.get('public_commit', '?')}; "
                f"lead over the public head {b.get('lead_tof', 0):+.3f} Toffoli/shot; lambda {b.get('lambda', '?')} (public "
                f"{b.get('public_lambda', '?')}). Runs are scored on {c.win_fast_draws} shared input draws paired with our base "
                f"(an unchanged tree gains exactly 0); the official one-draw verdict is not used (any edit redraws it and the "
                f"head fails ~20 batches per draw). Error rate: lambda {_budget_words(c)}. Audited gains ({c.win_audit_draws} "
                "other draws) are adopted as our base automatically, so gains stack: prefer many exact or near-exact savings, "
                "and spend the error budget only where the Toffoli per lambda is best. Submitting comes later, outside the "
                "loop (nonce search for a clean official draw, official run, Yang's approval). Never launch nonce searches or "
                "TAIL_NONCE changes. Earlier-track experiments 53, 54, 56 and 57 were reverted only because their redrawn "
                "official draw failed: re-test them first.")

    def track_brief(self) -> str:
        p = ROOT / "docs" / "win.md"
        return p.read_text()[:6000] if p.exists() else ""

    # --- submission ----------------------------------------------------------------
    def submit(self, worktree: Path, note: str) -> str:
        raise RuntimeError("win mode does not submit directly: the base needs a nonce search for a clean official draw, "
                           "the official run and Yang's approval first (docs/win.md)")

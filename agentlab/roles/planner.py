"""Planner: a UCB bandit over the idea backlog.

Each idea is an arm. Reward = relative gain of a solver run versus the best
score at the start of the run (negative when the run made things worse,
-1 when it failed validation). Untried ideas carry an optimistic prior so
the backlog is explored before the Planner settles on what works.
"""

from __future__ import annotations

import math
from pathlib import Path
import random

from ..ledger import Idea, Ledger

SEED_IDEAS = [
    # Legitimate research directions for a reversible secp256k1 point adder.
    # Each is a lead to test, not a claim; the ledger records what they return.
    ("Profile the current circuit by phase",
     "Instrument build() to attribute Toffoli count and peak qubits to each stage (inversion, "
     "multiplications, squarings, additions, uncomputation). Write the table to "
     "src/point_add/memory/profile.md and propose the three largest targets as next_ideas. "
     "Pure measurement, no gate changes. Runs first: every later idea should cite this table.", 0.08),
    ("Trim peak qubit width at the widest moment",
     "Find the single moment of peak width; move or reorder ancilla allocation so that moment "
     "is narrower, even at a small Toffoli cost. Score is a product, so width cuts pay as much as gate cuts."),
    ("Modular inversion: compare algorithms",
     "The inversion dominates most EC adders. Compare the current approach against Kaliski-style "
     "binary extended GCD, Bernstein-Yang safegcd, and windowed/lookup-assisted variants in terms "
     "of Toffoli count and the ancilla that must stay alive across it."),
    ("Windowed modular multiplication with table lookup",
     "Replace bit-serial controlled additions in multiply with k-bit windows and a QROM/lookup "
     "of precomputed multiples, trading a few ancilla for fewer Toffolis. Scan k."),
    ("Measurement-based uncomputation of AND/Toffoli ancillas",
     "Where a Toffoli only computes an AND into a fresh ancilla that is later uncomputed, replace "
     "the uncompute with the X-basis measurement + classically controlled CZ (the harness supports "
     "Hmr/R and CZ). Check every site keeps phase clean."),
    ("Share work between the two classical-offset additions",
     "offset_x and offset_y are classical bits: constants per shot. Fold constant-dependent "
     "structure into classical precomputation (the circuit may branch on classical bits via the "
     "condition stack) so the quantum part does less."),
    ("Karatsuba or Toom-style splitting for 256-bit multiply",
     "Measure whether a one-level Karatsuba split lowers Toffoli count enough to pay for its "
     "extra ancilla in the product metric; compare against schoolbook with windowing."),
    ("Reuse the modular reduction across consecutive products",
     "Delay reduction (lazy reduction) across a chain of multiply/add so fewer full modular "
     "reductions are performed, bounded by a few guard bits."),
    ("Replace squarings with specialised squaring circuits",
     "Squaring has symmetric cross terms; a dedicated squarer can halve the partial products "
     "versus a general multiply. Check whether current code already does this everywhere."),
    ("Reorder the point-addition formula",
     "Evaluate alternative affine addition schedules (compute lambda, then x3, then y3) that "
     "allow earlier uncomputation of intermediates, lowering peak width."),
]


def profile_text(cfg, adapter, limit: int = 8000) -> str:
    """The measured profile. Solvers write it inside their worktree; run_solver copies it to work/profile.md
    (merged upstream submissions may also carry one in the main checkout)."""
    for p in (Path(cfg.runs_dir).parent / "profile.md", Path(adapter.root) / "src/point_add/memory/profile.md"):
        if p.exists():
            return p.read_text()[:limit]
    return "(no measured profile yet: the profiling idea has not produced src/point_add/memory/profile.md)"


CLEANROOM_SEEDS = [
    # The clean-room design's milestones (docs/cleanroom.md), as testable leads with rough priors.
    ("Profile the clean-room circuit by phase",
     "Attribute Toffolis and the peak width to each phase of point.rs and each part of euclid.rs (Euclidean pass "
     "compare / swap / subtract, Bezout double / controlled add / swap, sqsub, classical adds). Write the table to "
     "src/point_add/PROFILE.md with the peak moment and what is live there. Pure measurement.", 0.08),
    ("Constant adders without a temporary register",
     "add_const writes the constant into n fresh qubits and runs a general adder. With the constant known at build time "
     "each carry is AND or OR of two qubits (no register), and a controlled constant add needs only the control. "
     "The materialised modulus sits at the peak (Bezout's modadd), so this should cut the width by about 256 and "
     "save X/CX ops. Unit tests first (moddbl, modadd), then the full score.", 0.08),
    ("Fewer Euclidean steps, with a measured failure rate",
     "The pass runs 2n = 512 steps; Schrottenloher reports about 1.413n + 2.4 sqrt(n) suffice except with small "
     "probability. Measure the step-count distribution for (p, x) with x uniform (classical model, 10^6 samples), pick "
     "the smallest count within the error budget (<= 6e-5), then build it and fresh_check. Every step removed saves "
     "about 12n Toffoli across division, multiplication and their Bezout replays.", 0.1),
    ("Shrinking u and v registers in the Euclidean pass",
     "u*v halves every step, so u and v need fewer bits as the pass goes on. Compare, swap and subtract on the live "
     "width only (padded by a few bits, failure measured classically); the freed high qubits can hold the record. "
     "Expected: Euclidean-pass Toffolis roughly halved and the record no longer adds to the peak.", 0.1),
    ("Pack the Euclidean record: 3 steps in 5 bits",
     "Each step has 3 states (even; odd no swap; odd swap), stored in 2 bits. Three steps have 27 states: pack them in "
     "5 bits with a few Toffolis (Schrottenloher) and unpack in the Bezout replay. Cuts the record from 1,024 toward "
     "~853 qubits, which is at the peak.", 0.05),
    ("Cheaper Bezout step: fuse doubling and controlled addition",
     "Each Bezout step does B = 2B mod p (2n) then B += e*A mod p (5n). Compute 2B + e*A once and reduce with one "
     "or two compare-and-subtract steps, or keep B lazily reduced with a guard bit; the Bezout replays are about half "
     "of all Toffolis.", 0.06),
    ("Cheaper x -= lambda^2",
     "sqsub costs about 9n^2 (n controlled modular subtractions, n-1 doublings and n-1 halvings to clean up). Try a "
     "dedicated squaring (cross terms counted once), Horner order that avoids the halvings, or computing lambda^2 into "
     "freed qubits at a narrow moment.", 0.04),
]


WIN_SEEDS = [
    # Win mode (docs/win.md): the official-draw track reverted candidates only because their redrawn official draw failed.
    ("Re-test exp 57: exact six-AND T[6] replacement",
     "On the official-draw track exp 57 implemented the complete six-AND T[6] replacement (restoration and phase cleanup), "
     "then restored production because the redrawn official draw failed, and committed the candidate, tests and evidence as "
     "0bea9232 (branch agentlab/exp57 in the challenge repo). That verdict was the draw, not the circuit. Bring the candidate "
     "onto the base, enable it in the active emission path, and win-score it: an exact saving shows as d_tof < 0 with "
     "d_lambda 0. Expected about -1 Toffoli per shot.", 0.12),
    ("Re-test exp 54: classical-prefix workspace replacement",
     "Exp 54 tested a classical-prefix workspace replacement through the active emission path and disabled it after the "
     "official draw rejected it (branch agentlab/exp54 in the challenge repo; its notes may be in work/notes/exp54/). "
     "Re-enable it on the base and win-score it.", 0.1),
    ("Re-score the official-draw track's reverted candidates",
     "On the official-draw track (ledger work/ledger.sqlite, notes work/notes/expN/, branches agentlab/expN in the challenge "
     "repo) candidates were judged by one redrawn official draw, which fails for almost any edit, and several were reverted "
     "for that reason only: e.g. exp 53 low-child replacement with classical bias absorption (c83b4e30), exp 56 encoded "
     "carries through the next upper columns (f75ed483). List those experiments, re-apply each committed candidate on the "
     "base, win-score it, and keep every one with d_tof < 0 and acceptable d_lambda. Report a table (exp, commit, d_tof, "
     "d_lambda).", 0.1),
    ("Map approximation knobs: Toffoli saved per lambda",
     "The head build = apply_merged_policy(divide_prefix::prefix()): a frozen op stream plus post-passes (cmp_policy, "
     "direct_policy, generic_compare widths, value_narrow cuts), then source_edits tables keyed to exact op positions. "
     "Recipe env knobs (HEO_PIN_*) are inert in the build, and changing a value_narrow cut breaks the position-keyed "
     "source_edits (panic). Find which approximation widths can be changed live (or how to regenerate the position tables), "
     "and measure each step's d_tof and d_lambda with win_score.sh. The leaders' notes (leapfrog.rs) show lambda "
     "bought back at ~700 Toffoli per unit and sold at ~1,400-2,250; we charge 1,000 per lambda above the public head's.", 0.08),
    ("Trade error rate for Toffoli at the best price",
     "Each lambda above the public head's costs 1,000 Toffoli per shot in the score; buying lambda back earns the same. Find "
     "the approximation sites (comparison widths, carry truncations, lookup windows, early uncomputation) with the most "
     "Toffoli per unit of lambda: sell where it is above 1,000, buy back where it is below, and confirm with 96 draws.", 0.06),
]


def seed_ideas(ledger: Ledger, kind: str = "leaderboard") -> int:
    n = 0
    seeds = {"cleanroom": CLEANROOM_SEEDS, "win": WIN_SEEDS + SEED_IDEAS}.get(kind, SEED_IDEAS)
    for item in seeds:
        title, why = item[0], item[1]
        prior = item[2] if len(item) > 2 else 0.02
        before = len(ledger.ideas(None))
        ledger.add_idea(title, why, "seed", prior=prior)
        n += len(ledger.ideas(None)) - before
        # ledgers created before a seed's prior changed keep the old value; refresh untried seeds
        for i in ledger.ideas(None):
            if i.title == title and i.source == "seed" and i.tries == 0 and i.prior != prior:
                ledger.set_idea_status(i.id, i.status, prior=prior)
    return n


def ucb_pick(ideas: list[Idea], c: float, rng: random.Random | None = None, gain_scale: float = 0.01) -> Idea | None:
    """UCB over relative gains. Gains here are ~1e-3..1e-2, so the exploration bonus is scaled to that
    (c x gain_scale); untried ideas always rank above tried ones, highest prior first. Without the scale
    a bonus of ~0.8 made every once-tried idea beat every untried one and the same idea ran twice."""
    rng = rng or random.Random()
    open_ideas = [i for i in ideas if i.status == "open"]
    if not open_ideas:
        return None
    untried = [i for i in open_ideas if i.tries == 0]
    if untried:
        return max(untried, key=lambda i: i.prior + rng.random() * 1e-9)
    total = sum(i.tries for i in open_ideas) + 1
    return max(open_ideas, key=lambda i: i.gain_sum / i.tries
               + c * gain_scale * math.sqrt(math.log(total) / i.tries) + rng.random() * 1e-9)


def bandit_picks(ledger: Ledger, cfg, rng: random.Random, n: int, exclude: set | None = None) -> list[Idea]:
    """n ideas by UCB, spread across sub-problems, never two measurements in a row, nothing already running.
    The Coordinator gets this list as a hint and the loop falls back to it when the Coordinator has no answer."""
    exclude = exclude or set()
    ideas = [i for i in ledger.ideas("open") if i.id not in exclude]
    last = ledger.experiments(1)
    if last and last[0]["idea_id"] and getattr(ledger.idea(last[0]["idea_id"]), "kind", None) == "measurement":
        optim = [i for i in ideas if getattr(i, "kind", None) != "measurement"]
        ideas = optim or ideas  # alternate: after a measurement run, an optimisation run
    picks: list[Idea] = []
    for _ in range(n):
        taken_ids = {p.id for p in picks}
        taken_sub = {getattr(p, "subproblem", None) for p in picks} - {None}
        pool = [i for i in ideas if i.id not in taken_ids and getattr(i, "subproblem", None) not in taken_sub]
        idea = ucb_pick(pool or [i for i in ideas if i.id not in taken_ids], cfg.explore_c, rng)
        if not idea:
            break
        picks.append(idea)
    return picks


def planner_brief(ledger: Ledger, idea: Idea, best: dict | None) -> str:
    """Context the solver gets: what the backlog knows and what to do now."""
    hist = ledger.db.execute(
        "SELECT id, status, score, gain, note FROM experiments WHERE idea_id=? ORDER BY id DESC LIMIT 5", (idea.id,)
    ).fetchall()
    lines = [f"IDEA #{idea.id}: {idea.title}", idea.rationale, ""]
    if getattr(idea, "kind", None) == "measurement":
        lines += ["This is a MEASUREMENT task: produce the numbers once, in the file named above, then stop. "
                  "Do not repeat measurements that are already in src/point_add/memory/ or the notes.", ""]
    else:
        lines += ["This is an OPTIMISATION task: the run only counts if the submitted circuit changes (different "
                  "Toffoli count, op count or width from the baseline you measure first). Measurements and notes "
                  "are only a means to that. If your edit leaves the scorer's numbers identical, the edit did not "
                  "reach build(): find the path that does (the Strategist's notes on the snapshot/policy route).", ""]
    try:
        from .strategist import formulation_excerpt
        from ..config import Config as _C
        exc = formulation_excerpt(_C.load(), getattr(idea, "subproblem", None))
        if exc:
            lines += [exc, ""]
    except Exception:  # noqa: BLE001
        pass
    if best:
        lines.append(f"Public best to beat: score {best['score']:,} = toffoli {best['toffoli']:.0f} x qubits {best['qubits']}.")
    if hist:
        lines.append("Earlier attempts on this idea:")
        for h in hist:
            sc = f"{h['score']:,}" if h["score"] else "-"
            lines.append(f"  exp #{h['id']}: {h['status']} score {sc} gain {h['gain']} note: {(h['note'] or '')[:200]}")
        try:
            from ..config import Config as _C
            notes = Path(_C.load().runs_dir).parent / "notes"
            dirs = [str(notes / f"exp{h['id']}") for h in hist if (notes / f"exp{h['id']}").exists()]
            if dirs:
                lines.append("Read their notes before you start, and continue from where they stopped instead of "
                             "repeating them: " + " ".join(dirs))
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(lines)

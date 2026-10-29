"""Planner: a UCB bandit over the idea backlog.

Each idea is an arm. Reward = relative gain of a solver run versus the best
score at the start of the run (negative when the run made things worse,
-1 when it failed validation). Untried ideas carry an optimistic prior so
the backlog is explored before the Planner settles on what works.
"""

from __future__ import annotations

import math
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


def seed_ideas(ledger: Ledger) -> int:
    n = 0
    for item in SEED_IDEAS:
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


def ucb_pick(ideas: list[Idea], c: float, rng: random.Random | None = None) -> Idea | None:
    rng = rng or random.Random()
    open_ideas = [i for i in ideas if i.status == "open"]
    if not open_ideas:
        return None
    total = sum(i.tries for i in open_ideas) + 1
    best, best_v = None, -1e9
    for i in open_ideas:
        if i.tries == 0:
            mean = i.prior
            bonus = c
        else:
            mean = i.gain_sum / i.tries
            bonus = c * math.sqrt(math.log(total) / i.tries)
        v = mean + bonus + rng.random() * 1e-6
        if v > best_v:
            best, best_v = i, v
    return best


def planner_brief(ledger: Ledger, idea: Idea, best: dict | None) -> str:
    """Context the solver gets: what the backlog knows and what to do now."""
    hist = ledger.db.execute(
        "SELECT id, status, score, gain, note FROM experiments WHERE idea_id=? ORDER BY id DESC LIMIT 5", (idea.id,)
    ).fetchall()
    lines = [f"IDEA #{idea.id}: {idea.title}", idea.rationale, ""]
    if best:
        lines.append(f"Public best to beat: score {best['score']:,} = toffoli {best['toffoli']:.0f} x qubits {best['qubits']}.")
    if hist:
        lines.append("Earlier attempts on this idea:")
        for h in hist:
            sc = f"{h['score']:,}" if h["score"] else "-"
            lines.append(f"  exp #{h['id']}: {h['status']} score {sc} gain {h['gain']} note: {(h['note'] or '')[:200]}")
    return "\n".join(lines)

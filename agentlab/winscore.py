"""Win-mode score: a circuit tested on shared input draws (common random numbers), paired with references.

The official harness tests one input draw derived from a hash of the op stream, so every edit redraws the inputs,
and the public head fails about 20 of its 141 batches on a typical draw (it passes officially only on a draw found
by searching TAIL_NONCE). A single official run therefore says nothing about an edit. Here every circuit is run on
the same draws (SHAKE256("agentlab-crn-draw" || salt), salts fixed), with the official simulator and checks
(tools/nonce_search crnfast), and compared draw by draw with:

  base    our current base (the public head until a gain is adopted): the bar every run must beat;
  public  the public head: the reference for the error budget and for our lead.

Score = effective Toffoli x qubits, where effective Toffoli = average executed Toffoli per shot over the draws
+ rate x max(0, lambda - lambda_public - budget). lambda = -141 ln(1 - failing batches per draw / 141) is the
expected number of failing batches; the cost of finding a clean official draw by nonce search grows as e^lambda,
so each lambda above the public head's costs x2.7 compute. Within the budget the error rate is free; beyond it each
lambda is charged `rate` Toffoli per shot. Defaults (budget 0, rate 1,000) follow the leaders' own documented practice
(src/point_add/leapfrog.rs): every change gated at "lambda est + 2 SE <= 19.5" measured on 192-512 paired nonces, and
lambda bought back at ~660-730 Toffoli per unit and sold at ~1,400-2,250. Their lambda counts failures per 9,024-shot
draw; the head measures 19.4 classical failures (21.9 failing shots, 20.2 failing batches) per draw here.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BATCHES = 141  # 9,024 shots in batches of 64


def lam(fb: float) -> float:
    fb = min(max(fb, 0.0), BATCHES - 0.05)
    return -BATCHES * math.log(1 - fb / BATCHES)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def parse_crn_output(text: str) -> dict | None:
    """The JSON line nonce_search crnfast prints last."""
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{") and "per_draw" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def rows_to_draws(per_draw: list) -> dict:
    """crnfast per_draw rows [salt, failing, classical, phase, tof, shots] -> {salt: [failing, classical, phase, tof, shots]}."""
    return {str(r[0]): list(r[1:]) for r in per_draw}


def _stats(draws: dict, salts: list[str]) -> tuple[float, float]:
    """(average Toffoli per shot, failing batches per draw) over the given salts."""
    return (_mean([draws[s][3] / draws[s][4] for s in salts]), _mean([draws[s][0] for s in salts]))


def effective(avg_tof: float, fb: float, fb_public: float | None, budget: float, rate: float) -> tuple[float, float, float]:
    """(effective Toffoli, lambda - lambda_public, penalty)."""
    if fb_public is None:
        return avg_tof, 0.0, 0.0
    d = lam(fb) - lam(fb_public)
    pen = rate * max(0.0, d - budget)
    return avg_tof + pen, d, pen


def score_draws(draws: dict, qubits: int, ref: dict | None, budget: float, rate: float) -> dict:
    """Score a circuit's draws ({salt: [...]}) against ref = {"base": {...}, "public": {...}} (each {commit, qubits,
    draws}); comparisons use the salts all three share."""
    salts = sorted(draws, key=int)
    base = (ref or {}).get("base") or {}
    pub = (ref or {}).get("public") or {}
    bd, pd = base.get("draws") or {}, pub.get("draws") or {}
    paired_b = all(s in bd for s in salts) and bool(salts)
    paired_p = all(s in pd for s in salts) and bool(salts)
    avg, fb = _stats(draws, salts)
    fb_pub = _stats(pd, salts)[1] if paired_p else None
    eff, dlam, pen = effective(avg, fb, fb_pub, budget, rate)
    out = {"ok": True, "mode": "win", "draws": len(salts), "salts": f"{salts[0]}..{salts[-1]}" if salts else "",
           "qubits": qubits, "avg_tof": avg, "failing_batches_per_draw": fb, "lambda": lam(fb), "budget": budget, "rate": rate,
           "classical_failures_per_draw": _mean([draws[s][1] for s in salts])}
    if salts and all(len(draws[s]) >= 7 for s in salts):
        out["failing_shots_per_draw"] = _mean([draws[s][5] for s in salts])
    if paired_p:
        out["public_classical_failures_per_draw"] = _mean([pd[s][1] for s in salts])
    if paired_p:
        dfb = [draws[s][0] - pd[s][0] for s in salts]
        out.update(public_commit=pub.get("commit"), public_lambda=lam(fb_pub), d_lambda=dlam,
                   d_lambda_se=_se(dfb) * BATCHES / (BATCHES - min(fb, BATCHES - 1)), penalty_tof=pen)
        p_avg = _stats(pd, salts)[0]
        out["lead_tof"] = p_avg - eff  # effective Toffoli per shot below the public head's (same draws)
        out["lead"] = (p_avg * pub.get("qubits", qubits) - eff * qubits) / (p_avg * pub.get("qubits", qubits))
    if paired_b:
        dt = [draws[s][3] / draws[s][4] - bd[s][3] / bd[s][4] for s in salts]
        b_avg, b_fb = _stats(bd, salts)
        b_eff = effective(b_avg, b_fb, fb_pub, budget, rate)[0]
        b_score = b_eff * base.get("qubits", qubits)
        out.update(base_commit=base.get("commit"), base_avg_tof=b_avg, base_qubits=base.get("qubits"), d_tof=_mean(dt),
                   d_tof_se=_se(dt), base_score=int(round(b_score)), gain=(b_score - eff * qubits) / b_score)
    if not (paired_b and paired_p):
        out["note"] = "no reference for these draws: raw score, error rate not compared"
    out["toffoli"] = eff
    out["score"] = int(round(eff * qubits))
    out["metrics"] = {k: out[k] for k in ("toffoli", "qubits", "avg_tof", "failing_batches_per_draw", "classical_failures_per_draw",
                                          "failing_shots_per_draw", "lambda", "d_lambda",
                                          "d_lambda_se", "penalty_tof", "d_tof", "d_tof_se", "gain", "lead_tof", "lead", "draws")
                      if k in out}
    out["metrics"]["mode"] = "win"
    if paired_p and dlam > 10:
        out["ok"] = False
        out["reason"] = (f"error rate far above the public head's: {fb:.1f} failing batches per draw "
                         f"(lambda {lam(fb):.1f} vs {lam(fb_pub):.1f}); the circuit is broken, not approximate")
    return out


def report_text(r: dict) -> str:
    L = [f"== win score: {r['draws']} shared input draws (salts {r['salts']}) ==",
         f"  avg executed Toffoli : {r['avg_tof']:,.3f}"
         + (f"   (base {r['base_avg_tof']:,.3f}: {r['d_tof']:+,.3f} +- {r['d_tof_se']:.3f} per shot)" if "d_tof" in r else ""),
         f"  qubits               : {r['qubits']}" + (f" (base {r['base_qubits']})" if r.get("base_qubits") else "")]
    L.append(f"  classical failures   : {r['classical_failures_per_draw']:.2f} per draw"
             + (f" (public head {r['public_classical_failures_per_draw']:.2f}; the leaders cap this near 19.5)"
                if "public_classical_failures_per_draw" in r else ""))
    if "d_lambda" in r:
        L.append(f"  failing batches/draw : {r['failing_batches_per_draw']:.2f}   lambda {r['lambda']:.2f} vs public head "
                 f"{r['public_lambda']:.2f}: {r['d_lambda']:+.2f} +- {r['d_lambda_se']:.2f} (free up to +{r['budget']:g}; each "
                 f"lambda above that costs {r['rate']:,.0f} Toffoli per shot)")
        if r.get("penalty_tof"):
            L.append(f"  error-rate penalty   : +{r['penalty_tof']:,.1f} Toffoli per shot")
    else:
        L.append(f"  failing batches/draw : {r['failing_batches_per_draw']:.2f}   lambda {r['lambda']:.2f}")
    L.append(f"  effective Toffoli    : {r['toffoli']:,.3f}")
    L.append(f"  score                : {r['score']:,}" + (f"   (base {r['base_score']:,}: gain {r['gain'] * 100:+.5f} %)" if "gain" in r else ""))
    if "lead_tof" in r:
        L.append(f"  lead over the public head: {r['lead_tof']:+,.3f} Toffoli per shot ({r['lead'] * 100:+.5f} % of its score)")
    if r.get("note"):
        L.append(f"  note: {r['note']}")
    if not r["ok"]:
        L.append(f"!! {r['reason']}")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    """python3 -m agentlab.winscore <crnfast output file> <reference json> [budget] [rate]: report + JSON line."""
    out_file, ref_file = Path(argv[0]), Path(argv[1])
    budget = float(argv[2]) if len(argv) > 2 else 0.0
    rate = float(argv[3]) if len(argv) > 3 else 1000.0
    crn = parse_crn_output(out_file.read_text(errors="replace")) if out_file.exists() else None
    if not crn or not crn.get("per_draw"):
        print("!! nonce_search crnfast produced no result")
        print(json.dumps({"ok": False, "reason": "crn evaluation produced no result (see the log above)"}))
        return 1
    ref = json.loads(ref_file.read_text()) if ref_file.exists() else None
    r = score_draws(rows_to_draws(crn["per_draw"]), int(crn["qubits"]), ref, budget, rate)
    print(report_text(r))
    r["per_draw"] = crn["per_draw"]
    print(json.dumps(r))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

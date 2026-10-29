---
name: reversible-arith
description: Design knowledge for Toffoli-counted reversible arithmetic circuits (modular add, multiply, square, invert, elliptic-curve point addition) and the Toffoli-versus-qubit trade-offs that drive a product metric.
---

# Reversible arithmetic for a Toffoli × qubit metric

## What is counted

- **Toffoli (CCX/CCZ)** is the expensive gate. Clifford gates (X, CX, CZ, Swap) are free in the score but still cost nothing only if they keep the circuit valid.
- **Peak qubits** is the maximum number of live qubits at any instant, including ancillas. Freeing early and allocating late lowers it. A single wide moment sets the whole number.
- **Classical bits** (the offset point) are free: anything that depends only on them can be precomputed classically and folded into gate selection or the condition stack.

## Where the Toffolis go in EC point addition

Affine addition (x1,y1)+(x2,y2): λ = (y2−y1)/(x2−x1); x3 = λ² − x1 − x2; y3 = λ(x1 − x3) − y1.
Typical cost ranking: modular inversion ≫ modular multiplications ≈ squarings ≫ additions. Measure your own circuit before trusting this.

## Levers, by expected impact

1. **Inversion algorithm.** Binary extended GCD (Kaliski), safegcd-style constant-iteration GCD, and lookup-assisted variants differ by large factors in Toffoli count and in how many ancillas stay live across the loop. Bounds on the iteration count change the width.
2. **Windowed multiplication.** k-bit windows with precomputed multiples (a lookup/QROM over classical constants is cheap when the multiplier is classical; costlier when both operands are quantum). Scan k; the optimum moves with the product metric.
3. **Uncomputation strategy.** Compute-copy-uncompute doubles Toffolis unless the uncompute uses measurement-based tricks (X-basis measure the AND ancilla, apply a classically controlled CZ). The harness supports Hmr/R and CZ; check the phase stays zero on every shot.
4. **Lazy modular reduction.** Keep a few guard bits and reduce once per chain of operations instead of after every one.
5. **Specialised squaring.** Cross terms appear twice; a squarer needs about half the partial products of a general multiply.
6. **Schedule.** Order operations so intermediates die early. The peak often sits at the inversion or at the first multiply after it.
7. **Constant folding.** The offset point is classical: subtractions by constants are controlled-X patterns, and comparisons against constants can be precomputed into the control structure.

## Validity reminders

- Every ancilla returns to |0⟩ before it is freed; the simulator checks each one.
- Global phase must be exactly zero across shots: phase kickback from a sloppy uncompute is a FAIL.
- Forward then reverse must be the identity on every qubit.
- A "saving" that fails any check is a FAIL, not a smaller score.

## Reference points

- Harness initial circuit: 3,942,753 Toffoli × 2,715 qubits ≈ 1.07e10.
- Google (2026) published Pareto points ≈ 2.7e6 × 1,175 and 2.1e6 × 1,425 (≈3.0e9 to 3.2e9).
- Shared repo head on 2026-10-29: 740,969 × 1,234 = 914,355,746, measured with the official harness in this system. Beat this.

Literature to consult when designing: Google's "Securing Elliptic Curve Cryptocurrencies against Quantum Vulnerabilities" (2026, with Zenodo code), Litinski's "How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates" (2023), Häner et al. "Improved quantum circuits for elliptic curve discrete logarithms" (2020), Roetteler et al. (2017), Gidney "Halving the cost of quantum addition" (2018), Gidney "Windowed quantum arithmetic" (2019).

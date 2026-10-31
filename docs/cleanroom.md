# Clean-room track: our own secp256k1 point adder

Design v0 (2026-10-31, approved by Yang): https://claude.ai/artifact/CRFzMdo94Ldhm8om2c7eWo

## Goal and standard

Build our own point adder for the ECDSA.fail harness that scores below Google's best published point,
2.1M Toffoli x 1,425 qubits = 2.99e9, at an error rate of at most 1e-4 on independent inputs.

- **Our own.** Written from published algorithms in `cleanroom/point_add/`. No code from leaderboard
  submissions (the clean-room harness repo does not even contain them) and none from other authors'
  implementations (e.g. Schrottenloher's Qarton code). Ideas from papers are used and cited in comments.
- **Same score.** Average executed Toffoli x peak qubits on the unmodified 9,024-shot evaluator.
- **Same error standard as the references.** Babbush et al. (Google) and Schrottenloher are approximate
  too, failing on at most 2^-13.3 of inputs. We measure with `scripts/fresh_check.sh` on 8 independent
  draws (72,192 inputs) and never choose a nonce or a draw.

## Where we are (M0, 2026-10-31)

Exact (no approximation anywhere), all unit tests pass, official evaluator: all 9,024 shots correct.

| | Toffoli | qubits | score |
|---|---|---|---|
| M0 (ours) | 4,789,503 | 2,307 | 1.105e10 |
| challenge's starting circuit | 3,942,753 | 2,715 | 1.07e10 |
| Google, low-gate | 2.1M | 1,425 | 2.99e9 (target) |
| Schrottenloher, gate-optimised (minus lookups) | ~1.67M | ~1,430 | ~2.4e9 (stretch) |

## Architecture (module files)

- `ir.rs` gates on virtual qubits; any program can be reversed exactly (AndC <-> AndU, Alloc <-> Free).
  Lowering reuses the lowest free physical id, so peak width = most qubits alive at once. AndU is
  measurement-based uncomputation (HMR + classically conditioned CZ): no Toffoli.
- `arith.rs` Gidney temporary-AND adder (n-1 Toffoli, n with carry-out), comparator via carry chain
  (n), controlled swap (n), constant adder (writes the constant into a temporary register; n-1).
- `modp.rs` modadd 4n, cmodadd 5n, moddbl 2n, modhalve 2n (exact inverse of moddbl), modneg n,
  add/sub of a classical register (the offset bits).
- `euclid.rs` in-place multiplication y <- x*y and division y <- y/x. Euclidean pass on (u, v) = (p, x),
  2n = 512 steps, each records (e = v odd, s = e AND v < u), then swap if s, v -= u if e, v /= 2
  (relabel). Bezout reconstruction replays the inverse steps on (y, 0) mod p and ends with (0, x*y).
  The Euclidean pass run backwards restores x and clears the record. Division = the same, reversed.
- `point.rs` x -= ox, y -= oy; y = y/x (lambda); x += 3 ox; x -= lambda^2 (sqsub: sum of
  lambda_i * 2^i lambda); y = y*x; x = -x; x += ox; y -= oy.

M0 Toffoli shares (from the unit tests): division 2.10M, multiplication 2.10M, sqsub ~0.59M, rest ~10k.
Per in-place multiplication: Euclidean passes 2 x 512 x 4n = 1.05M, Bezout 512 x 8n = 1.05M.
Peak width 2,307 is in the Bezout reconstruction: record 1,024 + A 256 + B 256 + modadd temporaries.

## Milestones (each closed by a measurement)

1. M0 correct skeleton: DONE (above).
2. M1 primitives: cheaper constant adders (no temporary register for the constant), cheaper controlled
   modular add, comparators on fewer bits where the error budget allows, measured per primitive.
3. M2 division: fewer Euclidean steps (about 1.413n + 2.4 sqrt(n), Schrottenloher) with a measured
   failure rate; record packing (3 steps' 27 states in 5 bits); u and v registers that shrink on a
   padded schedule so freed qubits hold the record; Bezout working registers sized to match.
4. M3 first full adder on the new architecture: <= 2.0M Toffoli at <= 1,450 qubits.
5. M4 below Google: score < 2.99e9 at error <= 1e-4.

Error budget (ours): Euclidean step count and padding <= 6e-5; top-bit comparisons <= 3e-5;
edge cases <= 1e-5. Each approximation is first measured with a classical model, then built.

## Rules of the track

- Editable: `src/point_add/` only (the worktree's copy of our module). The harness is unmodified.
- Correctness: unit tests (`cargo test --release --bin build_circuit -- --test-threads 2`) must pass; the
  official evaluator must pass; the fresh-input error rate must be <= 1e-4. Exact is better than
  approximate at equal score; spend error budget only where it buys a measured gain.
- A verified gain (audit: unit tests, official run, fresh check, strictly better than our best) is
  adopted automatically as the new base; nothing is ever submitted to the leaderboard from this track.
- Literature focus: Schrottenloher 2026 (arXiv 2606.02235), Gidney (adders, windowing, measurement-based
  uncomputation), Litinski, Haner et al. 2020, Roetteler et al. 2017, Bernstein-Yang safegcd, Proos-Zalka.

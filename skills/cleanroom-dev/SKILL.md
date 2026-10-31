---
name: cleanroom-dev
description: Working on our own clean-room point adder (src/point_add in a clean-room worktree, cleanroom/point_add in the agent repo) - the virtual-qubit IR, the primitives, unit tests as the first reward, and how to profile Toffoli and width.
---

# Clean-room module development

Read `docs/cleanroom.md` in the agentlab repo (the design brief, milestones, error budget) before planning.

## The contract (harness, unchanged)

- `point_add::build() -> Vec<Op>`; registers 0 = x (qubits), 1 = y (qubits), 2 = offset x (bits), 3 = offset y (bits),
  each 256 wide. The circuit overwrites (x, y) with the affine sum; every other qubit ends |0>, global phase 0.
- Executed Toffoli = CCX/CCZ whose classical condition is true on that shot; HMR, CZ, CX, Swap are free in the score.

## The module

- `ir.rs`: build programs with `Prog` over virtual qubits (`V = u32`). `alloc()` / `free()` (a freed qubit must
  be |0>), `x cx ccx swap cz z`, `x_if(q, bit)` (conditioned on a classical bit that never changes, i.e. an
  offset bit), `and_c(a, b) -> t` (1 Toffoli) and `and_u(a, b, t)` (0 Toffoli: HMR + conditioned CZ; t must
  hold exactly a AND b). `mark()` / `take_from()` / `extend_reversed()` give the exact inverse of any
  sub-program (AndC <-> AndU, Alloc <-> Free): write the compute half once, get the uncompute for free.
  `rename()` retargets a sub-program's output ids (used to run a routine backwards on a given register).
  `Lower` maps virtual to physical ids, lowest free id first; peak width = `next_phys`.
- Registers are `Vec<V>`, little-endian. Shifts and halvings by an even value are relabellings
  (`rotate_left`), not gates.
- Primitives and their Toffoli counts are listed in the module header comments and in `docs/cleanroom.md`.

## Unit tests are the first reward

```
cargo test --release --bin build_circuit -- --nocapture --test-threads 2
```
`tests.rs` runs each routine on the harness simulator with random inputs (small primes 65521 and 2053 and
secp256k1), checks values, ancillas and phase, and prints `UNIT <name> toffoli=<per shot> width=<peak>`.
- Change a primitive: its test must still pass, and its UNIT line shows the gain before you build the full
  circuit. Add a test for every new routine (small prime first: exhaustive-ish, seconds).
- An approximation (fewer Euclidean steps, comparisons on top bits): first measure its failure rate with a
  classical model (Python or a Rust test over 10^5 to 10^6 random inputs), then build it, then fresh_check.
- `toffoli_count(&b.g[m..])` counts a sub-program's Toffolis without simulating (conditioned gates count as
  executed, so it is an upper bound for offset-conditioned code).

## Profiling the full circuit

- `build()` prints the peak width and op count on stderr. To profile, add a temporary `eprintln!` of
  `toffoli_count` per phase in `point.rs` (or a test that builds `point_add` and counts per phase).
- Width: the peak is one moment. M0's peak is the Bezout reconstruction (record 2 x 512 + A + B + the
  modular adder's temporaries). Lower it by freeing earlier, allocating later, or shrinking registers.
- One official-sized score takes about 2 minutes (36M ops at M0); fresh_check with 8 draws about 15 minutes.

## Do not

- Read, fetch or reproduce code from the ECDSA.fail leaderboard or other authors' implementations.
- Edit anything outside `src/point_add/`; add dependencies; write results.tsv or score.json.

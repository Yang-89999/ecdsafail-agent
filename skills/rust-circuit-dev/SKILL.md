---
name: rust-circuit-dev
description: Working inside the ecdsafail-challenge Rust harness: the op model, the builder conventions in src/point_add, how to profile Toffoli and width, and how to run the scorer without breaking the sandbox rules.
---

# Rust circuit development (ecdsafail-challenge)

## The contract

- `point_add::build() -> Vec<Op>` emits the whole circuit. It runs in the untrusted `build_circuit` binary, sandboxed (read-only filesystem, no network); the trusted `eval_circuit` re-simulates the op stream.
- Four registers, each 256 wide, in order: `target_x` (qubits), `target_y` (qubits), `offset_x` (bits), `offset_y` (bits). The circuit overwrites `(target_x, target_y)` with the affine sum.
- Op kinds (src/circuit.rs): `X Z CX CZ Swap CCX CCZ R Hmr`, classical `BitInvert BitStore0 BitStore1`, `PushCondition/PopCondition` (ops run only while every bit on the condition stack is 1), `Neg` (global phase flip), `Register/AppendToRegister`, `DebugPrint`.
- `Op::validate()` rejects aliasing (same qubit twice in one op) and field-shape violations. The loader rejects unknown kinds and trailing data.

## Reading the current code

- `src/point_add/mod.rs` holds `build()` and the configuration switches it pins (the tree was evolved by many agents; many switches are env-var-shaped but fixed in source).
- Primitives: `builder.rs` (allocation, Toffoli emission), `modular.rs`, `pingpong.rs` (`multiply`, `divide`), `square*.rs`, `classical.rs`, `compare.rs`, `const_arith.rs`.
- `*.txt` and `*_followup.txt` are notes left by earlier agents. Treat them as leads; they may be stale or wrong.
- Big generated files (`*.bin`, `sqdepth2_*.rs`) are data tables; do not regenerate them casually.

## Profiling before changing

- Count Toffolis per phase by emitting `DebugPrint` markers or by splitting `build()` and measuring op-vector lengths between stages (count `CCX`/`CCZ` kinds).
- Peak width: track allocation high-water mark in the builder (or post-process the op vector: live set = qubits touched since allocation and not yet reset).
- The generator itself is slow (minutes) because of its simplification passes; a change that doubles generation time is acceptable, one that times out the scorer is not.

## Scoring

```
bash <autoresearch>/scripts/fast_score.sh <worktree> 64 <challenge checkout>
```
builds `build_circuit` from your worktree, runs it in a scratch dir, and evaluates 64 shots with the same evaluator code as the official run. The official run is `bash benchmark.sh --note "..."` in the worktree (9024 shots, ~1 minute longer). Averages over 64 shots can differ slightly from 9024 for circuits whose executed Toffoli count depends on the classical offset bits; the Auditor re-runs the official command before anything is claimed.

## Do not

- Edit anything outside `src/point_add/`.
- Add dependencies or change `Cargo.toml`, `Cargo.lock`, `rust-toolchain`.
- Write to `results.tsv` or `score.json`.
- Spawn background processes, touch the network, or read the test inputs (you cannot; they are derived from your op stream).

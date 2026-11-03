# GPU nonce search (tools/gpu_search)

To submit, the circuit needs a TAIL_NONCE whose official input draw passes every batch (a "clean draw", see
docs/win.md). At the error rate of the public head and of our base (lambda about 22.0 on the audit CRN draws) a clean
draw takes e^22 ≈ 3.7e9 nonce tries on average. On CPU that is about 1e5 core-hours, so the search runs on GPUs.

## What the program does

`tools/gpu_search/gpu_search.cu` runs the official test (inputs from the Fiat-Shamir stream, 141 batches of 64 shots,
classical/phase/ancilla checks) for many nonces at once and stops each nonce at its first failing batch. One lane is
one nonce holding one bit-sliced batch. All lanes run the same op stream in lockstep: one thread per lane, the ops
staged through shared memory, so every state access is a coalesced row. The circuit arrives as a job file written by
`nonce_search export` (compiled ops, register ids, the Fiat-Shamir prefix sponge, tail op bytes, fixed-base table), so
parsing and hashing setup stay in the Rust tool built from the challenge's own evaluator.

### Classical filter and fast simulator (default for `search` and `bench`)

The H100 runs (2026-11-02) showed the GPU nearly idle (at most 128 W, 35 C): the plain simulator is limited by
per-gate overhead, not memory. Its compiled loop spends about 55 instructions per gate and lane (about 20 register
moves for the condition stack, a chain of compare-and-branch dispatch, address arithmetic), runs at 124 registers so
only 16 warps fit on an SM, and a further ~30% of its work is Keccak: every HMR or R draws a random outcome (778,349
per batch, one permutation per 17).

A nonce search only needs draws with no failing shot. The classical and ancilla checks do not depend on the random
outcomes in practice: with every outcome forced to 0, or to 1, the exact simulator flags exactly the same failing
shots as with the official outcomes. Checked with `faildump` on CRN draws 0-99 of the public head (2026-11-03): 2,167
failing shots under the official outcomes, 1,897 of them classical or ancilla; the filter flags exactly those 1,897
shots (none missing, none extra) and the same 1,790 failing batches. (65 of those shots end with a different wrong
value, but fail either way.) So the search runs a filter
stream (`filter_ops`): every outcome taken as 0, so an HMR or R only resets its qubit, every block conditioned on an
outcome is dead and is dropped, phase ops are dropped, and conditions on bits known to be 0 or 1 are folded. For the
public head that leaves 8.18M of 10.24M ops: CX 72%, CCX 9%, X 8%, resets 8%, SWAP 2%, bit ops and PUSH/POP 0.8%,
condition depth 2, no randomness, no phase. A draw the filter passes is a candidate; `search` checks it in full
(official outcomes, phase, Toffoli count) on the CPU and reports `CLEAN` only if that passes. Phase-only failures
(2.7 failing shots per draw) are therefore caught only in that full check: on those 100 draws lambda is 22.63
overall and 19.92 for classical/ancilla failures, so about 1 candidate in e^2.71 = 15 comes out clean (15 CPU checks
of ~25 s per clean draw), and the filter runs 7.59 batches per nonce instead of 6.74. The number of nonces per clean
draw is unchanged.

The fast simulator (`--kernel fast`) runs the filter stream as 16-byte entries. Almost every op is t ^= cond & a & b
(CX and X use an all-ones row for absent controls; a reset is t ^= cond & t), grouped in runs with no branch inside;
the rare kinds stand alone. One thread handles two adjacent lanes with 16-byte loads, row offsets are precomputed per
lane count, the condition stack lives in a small local array. Compiled for sm_90 (clang + ptxas) the inner loop is 17
instructions per op for two lanes (8.5 per gate-lane, was ~55), with no spills at 32-78 registers depending on the
variant. Variants (`--fast V`, `bench` runs all): 0 = 2 lanes/thread, >= 8 blocks/SM; 1 = 1 lane, 8; 2 = 2 lanes,
12; 3 = 1 lane, 16; 4 = 2 lanes, 6; 5 = 2 lanes, 16 (32 registers, 4,096 lanes per SM). `selftest --kernel fast`
compares it record for record with the plain simulator on the filter stream (CPU: 2 CRN draws, 16 nonces identical);
the head's official nonce 12009835509 passes the filter (7 s on one CPU thread, vs about 25 s for the exact check).

### Slot program (`--kernel slots`; not faster)

Measured on an H100 (2026-11-02), the plain simulator spends about one GPU memory round trip on every gate: a lane alone
takes ~300 cycles per gate, and with 135k lanes the GPU reaches ~1,000-1,200 nonces/s (about 1,000 H100-hours per clean
draw at lambda 22). The gates mostly reuse a few rows (40% read the row the previous gate wrote), so the slot program
keeps K rows of every lane in shared memory. Because all lanes run the same op stream, which rows to keep is decided
once on the host with the whole future known (Belady: evict the row needed again furthest ahead, weighted by alpha
towards rows idle the longest so loads can be issued earlier). The program adds LOAD (cp.async, not blocking),
STORE and WAIT entries; WAIT goes before the first use of a slot whose load may still be in flight. Global traffic per
gate drops from ~3 words to 0.79 (K=64), 0.59 (K=128). Options: `--kernel slots|plain`, `--slots K[:alpha[:block]],...`
(default 64:2:128); `compile` prints the program's statistics. The slot program is checked record for record against
the plain simulator (`selftest` compares with the plain CPU path). On the H100 it was correct but no faster (best K=32:
1,040-1,270 nonces/s vs plain 1,025-1,295), which is what showed memory was not the limit.

Modes: `search <start> <count>` (stops after the round that finds the first clean nonce), `check <nonce>` (full test
of one nonce), `crn <start> <count>` (the same lines and JSON as `nonce_search crnfast`, so the GPU can score
candidates too), `selftest`, `bench`, `faildump <start> <count>` (CPU: every failing shot of CRN draws). `crn`,
`check` and `selftest` default to the exact plain simulator; `--kernel fast` or `--filter` switch them to the filter.

Built without nvcc (`scripts/build_gpu_search.sh --cpu`), the same per-lane code runs on the CPU. That build was
checked in the cloud against `nonce_search` (2026-11-01, public head 5ba2d2d): 4 CRN draws identical in every field
including Toffoli totals, 64 early-stop nonces identical, and the head's official nonce 12009835509 comes out clean.
The CUDA code compiles for sm_89 with no register spills in the simulator kernel (clang + ptxas). The GPU itself is
checked on the GPU machine: `selftest` compares it with its own CPU path record by record, and `check` must find the
head's official nonce clean.

## Benchmark on a rented GPU (about 30 minutes)

Rent one H100 (or an RTX 5090, or any recent NVIDIA GPU) with an image that has CUDA 12.8 or newer including nvcc (for
example RunPod's PyTorch 2.x / CUDA 12.8 template, or `nvidia/cuda:12.8.1-devel-ubuntu24.04`), at least 8 CPU cores
and 40 GB of disk. Then on that machine:

    git clone https://github.com/Yang-89999/ecdsafail-agent

    nohup bash ecdsafail-agent/scripts/gpu_bench.sh > bench.out 2>&1 &

It installs Rust if needed, builds the public head's op stream, runs the checks (the official nonce clean on the CPU,
exactly and through the filter; the exact GPU simulator = the plain CPU simulator; fast variants 0, 3 and 5 on the GPU
= the plain CPU simulator on the filter stream; GPU = Rust reference on 64 nonces), then measures the exact plain
simulator and all six fast variants (60 s each), logs GPU clocks and throttling, and tries one `ncu` profile of the
fast kernel (skipped if the machine does not allow GPU counters). The output ends with lines like

    SEARCH ... nonces/s ...
    at lambda 22.04 (e^lambda = 3.731e+09 nonces per clean draw on average): H hours on this device per clean draw, N devices for a 12-hour average
    BEST <setting>: R nonces/s; at lambda 22.04: H hours on this device per clean draw (90% within ... hours)

First H100 measurements (2026-11-02): plain simulator 457 / 671 / 941 / 1,228 nonces/s at 16,896 / 33,792 / 67,584 /
135,168 lanes; second run plain 1,295 (first half) / 1,025 (second half) nonces/s at 135k lanes, slot program no
faster. All checks passed in both runs.

Everything it uses is public; nothing of ours leaves the Mac.

## The real search (only with Yang's go)

- Our base's circuit stays private until it is submitted. Never push `agentlab/win-base` or a job file anywhere public
  (the agent repo is public); a job file contains the whole op stream.
- The job file of our base is exported on the Mac and copied straight to the GPU machines.
- Each GPU searches its own range: `gpu_search win-base.job search <k x 10^12> 1000000000000 --device <i>`.
- The first clean nonce found ends the search. It is re-checked on the CPU (`nonce_search check <nonce>`), written
  into the tail (TAIL_NONCE), run through the official benchmark, then `agentlab approve` and submission with Yang's
  explicit go.
- Rule: take the first clean draw found; never choose among clean draws by their Toffoli count.

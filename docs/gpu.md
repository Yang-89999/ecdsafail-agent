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

Modes: `search <start> <count>` (stops after the round that finds the first clean nonce), `check <nonce>` (full test
of one nonce), `crn <start> <count>` (the same lines and JSON as `nonce_search crnfast`, so the GPU can score
candidates too), `selftest`, `bench`.

Built without nvcc (`scripts/build_gpu_search.sh --cpu`), the same per-lane code runs on the CPU. That build was
checked in the cloud against `nonce_search` (2026-11-01, public head 5ba2d2d): 4 CRN draws identical in every field
including Toffoli totals, 64 early-stop nonces identical, and the head's official nonce 12009835509 comes out clean.
The CUDA code compiles for sm_89 with no register spills in the simulator kernel (clang + ptxas). The GPU itself is
checked on the GPU machine: `selftest` compares it with its own CPU path record by record, and `check` must find the
head's official nonce clean.

## Benchmark on a rented GPU (about 20 minutes)

Rent one machine with an RTX 5090 (or any recent NVIDIA GPU), an image with CUDA 12.8 or newer including nvcc (for
example `nvidia/cuda:12.8.1-devel-ubuntu24.04`, or a PyTorch image built for CUDA 12.8), at least 8 CPU cores and
40 GB of disk. Then on that machine:

    git clone https://github.com/Yang-89999/ecdsafail-agent

    bash ecdsafail-agent/scripts/gpu_bench.sh

It installs Rust if needed, builds the public head's op stream, runs the three checks (official nonce clean, GPU =
CPU path, GPU = Rust reference on 64 nonces) and measures the search rate at four lane counts (2 minutes each). The
output ends with lines like

    SEARCH ... nonces/s ...
    at lambda 22.04 (e^lambda = 3.731e+09 nonces per clean draw on average): H hours on this device per clean draw, N devices for a 12-hour average

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

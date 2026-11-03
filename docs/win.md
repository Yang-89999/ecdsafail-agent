# Win mode: building a lead over the public head

Goal (Yang, 2026-11-01): win the ECDSA.fail leaderboard. Score = average executed Toffoli per shot x peak qubits on
the official evaluator, lower is better. The public head (5ba2d2d on 2026-11-01) is 739,480 Toffoli x 1,234 qubits =
912,518,320.

## Why the official verdict cannot judge an edit

The official harness tests one draw of 9,024 inputs, derived from a hash of the op stream. The organisers allow
approximate circuits and searching the hash input (TAIL_NONCE, 96 identity ops at the end) for a draw on which every
shot passes ("clean validation island", arXiv 2609.09582). The head is approximate: on a typical draw it fails about
20 of its 141 batches of 64 shots (lambda ~ 22), so a clean draw needs ~3e9 nonce tries; its authors found one. Any
edit changes the hash, so the official draw of an edited circuit fails, whatever the edit does. The earlier track
judged edits that way and reverted every candidate (experiments 53, 54, 56, 57 among them).

## How a circuit is scored here

- Draws: fixed input sets, SHAKE256("agentlab-crn-draw" || salt), the same for every circuit. Score: salts 0..23.
  Audit: salts 1,000,000..1,000,095.
- Simulator and checks: the official ones (tools/nonce_search crnfast, verified identical to the reference
  simulator draw by draw), all batches run, failures counted instead of stopping.
- Paired comparison with our base and the public head on the same draws: an unchanged tree gains exactly 0, and an
  exact saving of one Toffoli per shot shows as -1.000 +- 0.000.
- lambda = -141 ln(1 - failing batches per draw / 141). The final nonce search costs x2.7 per extra lambda.
- Effective Toffoli = average Toffoli + 1,000 x max(0, lambda - lambda_public). No worse than the public head is
  free; each lambda above it costs 1,000 Toffoli per shot. This follows the leaders' documented practice
  (src/point_add/leapfrog.rs): every change gated at "lambda est + 2 SE <= 19.5", lambda measured on 192-512 paired
  nonces ("paired, CRN"), lambda bought back at ~660-730 Toffoli per unit and sold at ~1,400-2,250. Their lambda counts
  failures per 9,024-shot draw; the head measures 19.4 classical failures per draw here (21.9 failing shots including
  phase-only ones, 20.2 failing batches).
- Score = effective Toffoli x qubits. A run whose score beats our base on the 24 draws is audited on 96 other draws;
  if it still beats the base it is adopted (branch agentlab/win-base in the challenge repo moves), so gains stack.

## What to do first

1. Re-test the reverted candidates of the official-draw track: exp 57 (exact six-AND T[6] replacement, commit
   0bea9232, about -1 Toffoli/shot), exp 54 (classical-prefix workspace replacement), exp 53 (low-child replacement,
   c83b4e30), exp 56 (encoded carries, f75ed483). Branches agentlab/expN live in the challenge repo.
2. Map the approximation knobs that can be changed live, with the Toffoli saved per lambda for each step.
3. Stack exact savings; spend the free error budget only where the Toffoli per lambda is best.

## Submitting (outside the loop, with Yang)

The base goes to the leaderboard only after: a nonce search on cloud CPU for a clean official draw (about 1e5
core-hours at the head's error rate, roughly $1,000-2,500; Yang decides the spend), the official benchmark run on that
nonce, `agentlab approve <exp> --by Yang`, and Yang's explicit go. Rules for the search: take the first clean draw,
never choose draws by their Toffoli count, never submit a re-roll of the head.

## When the public head adds a hash-bound final pass

From 39b7cd5 the public head ends `apply_merged_policy` with `source_edits::run(ops, include_str!("disc50_followup.txt"))`:
a table bound to the exact stream (length + SHA3-256) that inserts 17,335 ops at position 0 and turns 9 CCX into CX.
Our adopted passes run after `apply_merged_policy` and some bind to positions, so a plain replay builds a stream that
panics. win mode then keeps our base on the old head (`work/win/replay_failed.json`). To merge by hand, with win mode
stopped:

    docker/run.sh bash tools/merge_head/merge_head.sh

It replays our commits onto the head, moves that final pass after our passes (`tools/merge_head/patch.py`), applies it
with `rebind_edits` (each edit's site found by its old ops and 24 ops of context on each side, nearest to the expected
position; an edit whose site our passes changed is skipped and reported), builds, and moves `agentlab/win-base` only if
the build succeeds. Checked on f5f14f4: with no passes of ours the stream is byte-identical to the head's; with ops
dropped and duplicated before the edit sites, all 18 edits still land.

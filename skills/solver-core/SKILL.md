---
name: solver-core
description: How a Solver agent works one hypothesis in a worktree of a verifiable benchmark: measure first, change one thing, score, keep or revert, report honestly.
---

# Solver core

You have one worktree, one idea, and a scorer. Your output is a tree that scores lower (or an honest report that the idea does not), plus a note the next solver can build on.

## Loop

1. **Baseline first.** Run the scorer once before editing. Record the number. If it fails, stop and report `failed` with the reason; do not "fix" the harness.
2. **Read before you write.** The editable folder was built by hundreds of earlier agent runs. Find where your idea lives (grep for the primitive), read the notes other agents left (`*.txt`, `memory/`), and check the idea was not already tried. Earlier notes are leads, not facts: re-measure.
3. **Triage in the first 15 minutes.** If reading shows the idea is already implemented (grep for it; read `memory/` and `*.txt` notes), say so in one line of your final report (`idea_status: already_implemented`) and **pivot immediately** to the closest untried variant or to the next-biggest target you can see. Do not spend the run documenting what exists. A run should end with at least one scored change to the circuit; notes and probes alone are a wasted run.
4. **One change at a time.** Make the smallest edit that tests the hypothesis. Score. Compare to baseline. Keep it only if the scorer says `ok: true` and the score dropped. Revert otherwise (`git checkout -- <file>`).
5. **Chase the product.** Score = metric A × metric B. A change that raises one and lowers the other is judged on the product only.
6. **Commit what you keep** with a message that says what and why. Never commit a tree that fails the scorer.
7. **Sweep, don't hand-tune.** Turn your idea into a knob (a window size, a split depth, a schedule order, which of N sites to rewrite, a policy-table entry) and let `sweep.py` score every setting unattended. A run should end with dozens of scored variants, most of them from sweeps. Hand-edit and score once only to make the first version of the knob work. The tool is `sweep.py`, in the same scripts folder as `fast_score.sh`: `python3 <scripts>/sweep.py <worktree> --mutate "python3 .agentlab/mutate.py {v}" --values 0,1,2,3 --shots <same shots as the scoring command> --challenge <same challenge path> --keep-best` (or `--patches <dir>` with one `.patch` per variant). Commit first; each variant is applied, scored, logged to `.agentlab/sweep.jsonl` and reverted; `--keep-best` leaves the winner applied.
8. **Budget: evidence, not the clock.** A score run takes about 30 seconds (the generator loads a snapshot). Use that: 30 to 100 scored variants is normal, not a handful. Sweep parameters, try many small edits, search locally around anything that moves the numbers. Stopping after a few variants is the most common way a run is wasted here. There is no time limit: a long run on a sound direction is wanted, and another executor works in parallel on other sub-problems. Stop only when (a) you hold a verified gain and see no cheap next step, or (b) you can say concretely why the direction cannot pay (ceiling reached, claim refuted at small size, the primitive is already at the known optimum). Report which it was in `idea_status` (`promising` with `needs_more_time` when you stopped with leads open: the Coordinator can continue your tree in a later run).
9. **Log every variant.** Keep a running list of what you scored (change, ok or not, Toffoli, qubits). It goes into the final report; failed and worse variants are as valuable to the next agent as successes.

## Think like a research agent, not a code monkey

- **State the claim before the edit.** Write one line: "If I change X, Toffoli drops by about N because Y (stage Z holds M Toffolis)." A claim without a predicted size is not testable; a prediction the profile cannot support is a sign to re-read the profile.
- **Refute cheaply first, but only to choose an edit.** A score run takes under a minute, so the scorer is usually the cheapest refutation there is. Write a small probe only when it decides which production edit to make next in this run. Proofs, certificates and test fixtures are not deliverables: a run whose output is a proof and an unchanged circuit is a wasted run.
- **Generate several, test the best.** For a non-trivial change, sketch 2-3 variants with their predicted effect, pick the one with the best prediction-to-effort ratio, and test it. Keep the others in `next_ideas` with their predictions.
- **Keep a lemma scratchpad.** Record each verified fact (measured numbers, invariants, "this primitive is already optimal because...") in `src/point_add/memory/<idea>.md` as you go. The next solver inherits it; the ledger inherits your report.
- **Ask the literature a question, not a topic.** When stuck on one primitive, use the ask tool with a precise question; read the answer as a hypothesis to test, not an authority.
- **Critique before commit.** Before committing a gain: does `ok: true` hold at the default shots (9024, the official count); does the gain come from real work removed or made cheaper (fewer emitted Toffolis or ops, narrower width), not from a cosmetic change that only reshuffles the hash-derived inputs; did peak width move the other way; is the change inside the editable folder only.
- **Small gains are the game.** Recent leaderboard steps are about 0.01 % of the product (tens of Toffolis per shot). A verified saving of 20 Toffolis is a win: keep it, commit it, and look for the next one nearby. Never discard a real saving as "below noise": the scorer runs the official shot count, so a real reduction in work shows up in the number.

## Hard rules

- Edit only the editable paths. Never touch harness, lockfiles, toolchain pins, or result logs.
- Never claim a number you did not see printed by the scorer in this run.
- Never tune to the test set. Inputs are derived from a hash of your own circuit; a change aimed at the inputs changes the inputs. Never change `TAIL_NONCE` (or anything whose only effect is which inputs are drawn) and never keep a variant because its draw passed. The public head itself is wrong on about 0.2 % of inputs and passes only its own draw: check a candidate with `scripts/fresh_check.sh`, and treat failures the unmodified head also shows as inherited.
- No network except the literature ask tool, no new dependencies, no background processes left running.
- If a scorer bug looks exploitable, report it in your final message; do not use it.

## Final report

End with the JSON block the prompt asks for, including `variants` (up to 30 entries, most informative first). `score` is the last `ok: true` number for the tree you leave committed. `next_ideas` are concrete follow-ups, each one sentence, that you would try if you had another run.

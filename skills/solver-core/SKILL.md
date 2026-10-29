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
7. **Budget.** Each score run takes minutes. Plan for roughly 8 to 15 score runs. Stop when the idea is exhausted or you have a verified gain.

## Think like a research agent, not a code monkey

- **State the claim before the edit.** Write one line: "If I change X, Toffoli drops by about N because Y (stage Z holds M Toffolis)." A claim without a predicted size is not testable; a prediction the profile cannot support is a sign to re-read the profile.
- **Refute cheaply first.** Before a full score run, test the claim at small size: a reduced modulus or width, a unit probe, a count of gates emitted by the primitive. The repo has probes from earlier runs (`*_probe.py`, `memory/`); add your own under `src/point_add/memory/` when a claim needs one. One minute of refutation saves a ten-minute score run.
- **Generate several, test the best.** For a non-trivial change, sketch 2-3 variants with their predicted effect, pick the one with the best prediction-to-effort ratio, and test it. Keep the others in `next_ideas` with their predictions.
- **Keep a lemma scratchpad.** Record each verified fact (measured numbers, invariants, "this primitive is already optimal because...") in `src/point_add/memory/<idea>.md` as you go. The next solver inherits it; the ledger inherits your report.
- **Ask the literature a question, not a topic.** When stuck on one primitive, use the ask tool with a precise question; read the answer as a hypothesis to test, not an authority.
- **Critique before commit.** Before committing a gain: does `ok: true` hold at the default shots; is the gain larger than the measurement noise (about 0.02 %); did peak width move the other way; is the change inside the editable folder only.

## Hard rules

- Edit only the editable paths. Never touch harness, lockfiles, toolchain pins, or result logs.
- Never claim a number you did not see printed by the scorer in this run.
- Never tune to the test set. Inputs are derived from a hash of your own circuit; a change aimed at the inputs changes the inputs.
- No network except the literature ask tool, no new dependencies, no background processes left running.
- If a scorer bug looks exploitable, report it in your final message; do not use it.

## Final report

End with the JSON block the prompt asks for. `score` is the last `ok: true` number for the tree you leave committed. `next_ideas` are concrete follow-ups, each one sentence, that you would try if you had another run.

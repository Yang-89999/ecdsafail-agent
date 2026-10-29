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

## Hard rules

- Edit only the editable paths. Never touch harness, lockfiles, toolchain pins, or result logs.
- Never claim a number you did not see printed by the scorer in this run.
- Never tune to the test set. Inputs are derived from a hash of your own circuit; a change aimed at the inputs changes the inputs.
- No network, no new dependencies, no background processes left running.
- If a scorer bug looks exploitable, report it in your final message; do not use it.

## Final report

End with the JSON block the prompt asks for. `score` is the last `ok: true` number for the tree you leave committed. `next_ideas` are concrete follow-ups, each one sentence, that you would try if you had another run.

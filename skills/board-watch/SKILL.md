---
name: board-watch
description: How the Scout reads a Yukon-style shared repository and board to produce leads: what each merged submission changed, how the bar moved, and what the rivals' notes suggest.
---

# Board watch

Sources, in order of trust:

1. **The shared repository.** Every verified gain is merged as one commit (`Validate submission <uuid>`). The diff of each commit is the only ground truth about what moved the score.
2. **The upstream head's score**, measured by our own run of the official checker (the organisers do not keep `results.tsv` current).
3. **The board page** (ecdsa.fail / yukon.org): record, solver and model names, dates. Use it to detect gains that are verified but not yet merged.
4. **Notes other agents left** in the editable folder. Leads only.

For each recent commit produce a lead: what primitive changed, by how much the stat says, and one follow-up that would extend or generalise it. Add it to the ledger with `source=scout:<commit>` so the Planner can weight it. Do not invent numbers: if a score is not visible, say unknown.

Flag to the human when: the rules file or harness changed, the bar jumped by more than 10 % in a day, or the board shows a result the repo does not contain.

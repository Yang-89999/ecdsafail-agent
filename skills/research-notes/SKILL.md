---
name: research-notes
description: Writing the public note that accompanies a submission, and the memory note left in the repo, using only measured numbers and the format the board expects.
---

# Research notes

Two notes per accepted result:

1. **Repo memory note** under `src/point_add/memory/<date>-<slug>.md` (ECDSA.fail convention): what changed, why, the measured before/after from the official run, what was tried and failed, and the next lead. Short, factual, dated.
2. **Submission note** (`--note` to the harness, or the board's note field): one line, what changed and the official score.

Rules: only numbers from the ledger's official runs; no estimates presented as measurements; no claims about other solvers; disclose the method at the level the human approved (default: what changed, not the full search log).

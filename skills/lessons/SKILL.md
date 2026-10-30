---
name: lessons
description: How the Coordinator distils finished runs into a standing lessons file (verified facts, dead ends with reasons, what moved the numbers, rules of thumb) that every later agent reads.
---

# Lessons

A run's report and notes are raw material; a lesson is what a future agent must know so it does not repeat the run. You revise one standing document, you do not append a diary.

- **Verified facts**: measured statements with their evidence (experiment number, the printed number). "Edits to leapfrog.rs alone leave the scorer's numbers identical because build() loads the snapshot in divide_prefix.rs (exp 12, 17)."
- **Dead ends**: what was tried, why it cannot pay or failed, which experiments showed it. A dead end needs a reason; without one it is a retry candidate, not a lesson.
- **What moved the numbers**: every change that changed Toffoli, ops or width, with direction and size, even when the product got worse. These are the only leads with evidence.
- **Rules of thumb**: practical, specific guidance for the next executor ("score the unchanged tree first; identical numbers mean the edit did not reach build()"; "width is charged as highest qubit id + 1: freeing qubits without renumbering changes nothing").

Rules: merge the new evidence into the existing entries (update counts, add experiment numbers, sharpen the reason) instead of duplicating them; drop an entry only when new evidence contradicts it, and say so in its place; keep every number as measured or labelled a prediction; keep the file short enough to be read in a minute (at most about 40 entries in all), most useful first.

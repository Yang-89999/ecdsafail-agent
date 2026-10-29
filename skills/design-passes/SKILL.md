---
name: design-passes
description: A menu of optimisation passes borrowed from logic synthesis and EDA, restated for reversible Toffoli circuits scored on Toffoli count × peak qubits, with how to check each one.
---

# Design passes (EDA view of a reversible circuit)

A circuit generator is a synthesis flow. Each pass below is a known lever in logic synthesis (ABC, Yosys, commercial place-and-route); here is what it means for a Toffoli circuit and how to tell if it paid.

| Pass | In EDA | Here | Check |
|---|---|---|---|
| Peephole rewrite | local AIG rewriting | cancel adjacent Toffoli pairs, merge Toffolis sharing controls, replace Toffoli+CNOT patterns with cheaper identities | gate count of the primitive in isolation, then full score |
| Constant propagation | known inputs | classically known bits (curve constants, modulus, fixed offsets) must not cost quantum gates; fold them into classical control or precomputed tables | grep for Toffolis whose control is a constant |
| Don't-care exploitation | SDCs/ODCs | outputs never read again need no uncomputation; garbage that gets measured away later need not be clean in between | trace each ancilla's last use |
| Resource sharing | datapath sharing | two sub-circuits using the same intermediate (products, comparison flags, carries) compute it once | find duplicated primitives on the same operands |
| Retiming / scheduling | register retiming | reorder independent blocks to lower the *peak* width: uncompute early, allocate late; the peak is one moment, find it | width profile over time, locate the maximum |
| Parameter sweep | design-space exploration | window sizes, lookup fan-out, adder choice per operand width, number of lazy reductions before a full one | scan the parameter, plot score; the product has one minimum |
| Strength reduction | multiplier → shifts/adds | multiplication by small or structured constants as additions/doublings; squarings instead of general products | count Toffolis per constant multiply |
| Equivalence checking | formal EC after every pass | the harness's forward∘reverse identity and 9024-shot check is the EC; at small modulus run your own exhaustive check for a modified primitive | small-modulus probe before full score |
| Lint / DRC | design-rule check | ancillas returned clean, zero phase, no gate outside the gate set | the scorer's `ok` field; never bypass it |

Read the profile first: a pass on a stage holding 3 % of the Toffolis cannot win more than 3 %. Pick the pass whose ceiling is largest for the stage with the largest share.

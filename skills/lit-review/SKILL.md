---
name: lit-review
description: Turning the key papers on quantum elliptic-curve arithmetic into concrete, testable ideas for the backlog, with the claim and the paper cited on each idea.
---

# Literature review for the backlog

For each paper: extract the construction, its stated Toffoli count and qubit count (quote them with the page), the assumptions (gate set, error model, whether measurement-based uncomputation is used), and the one or two ideas that transfer to this harness's gate set. Add each idea to the ledger with `source=human` or `scout:paper:<short-id>` and the citation in the rationale.

Starting list:

- Google Quantum AI, "Securing Elliptic Curve Cryptocurrencies against Quantum Vulnerabilities: Resource Estimates and Mitigations" (2026), with the Zenodo code the harness is adapted from.
- D. Litinski, "How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates" (arXiv 2306.08585).
- T. Häner, S. Jaques, M. Naehrig, M. Roetteler, M. Soeken, "Improved quantum circuits for elliptic curve discrete logarithms" (PQCrypto 2020).
- M. Roetteler, M. Naehrig, K. Svore, K. Lauter, "Quantum resource estimates for computing elliptic curve discrete logarithms" (ASIACRYPT 2017).
- C. Gidney, "Halving the cost of quantum addition" (Quantum 2018); "Windowed quantum arithmetic" (arXiv 1905.07682).
- ECDSA.fail caveats note (arXiv 2609.09582) on what the benchmark does and does not model.

Numbers from papers go into the rationale as "paper claims", never into `bests`.

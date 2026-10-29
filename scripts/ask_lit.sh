#!/usr/bin/env bash
# Ask the literature a specific question from inside a solver run.
#   bash scripts/ask_lit.sh "cheapest way to uncompute a comparison flag against a constant modulus"
# Searches arXiv, reads the abstracts, prints a short cited answer (cached per question). Takes 1-3 min.
set -euo pipefail
exec agentlab ask "$*"

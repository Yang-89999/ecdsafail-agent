#!/usr/bin/env bash
# Build the image (first time) and open a shell in the agentlab container.
#
#   docker/run.sh                       interactive shell
#   docker/run.sh agentlab run --rounds 3 --solvers 1
#
# Mounts: this repo (code and skills, live-edited from the Mac) at /work/ecdsafail-agent.
# Volumes (fast Linux disk, kept between runs):
#   agentlab-challenge -> the harness checkout and its build cache
#   agentlab-work      -> ledger, worktrees, run logs
#   agentlab-codex     -> Codex login
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd -P)"
docker build -q -t agentlab -f "$here/docker/Dockerfile" "$here" >/dev/null
tty_flags=(-i)
[[ -t 0 ]] && tty_flags=(-it)
exec docker run --rm "${tty_flags[@]}" --name "agentlab-$$" \
  --memory 6g --cpus "$(( $(getconf _NPROCESSORS_ONLN) > 1 ? $(getconf _NPROCESSORS_ONLN) - 1 : 1 ))" \
  -v "$here":/work/ecdsafail-agent \
  -v agentlab-challenge:/root/ecdsafail-challenge \
  -v agentlab-work:/work/ecdsafail-agent/work \
  -v agentlab-codex:/root/.codex \
  agentlab "${@:-bash}"

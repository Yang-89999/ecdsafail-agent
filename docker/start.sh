#!/usr/bin/env bash
# Start a long, unattended run in a detached container (survives closing the terminal).
#
#   docker/start.sh                 7 days, 1 solver
#   docker/start.sh --days 3 --solvers 2
#
# Then:  docker/logs.sh   (follow the log)     docker/shell.sh   (a shell inside; e.g. `agentlab watch`)
#        docker/stop.sh   (stop the run)
# Keep the Mac plugged in with the lid open; this script also starts `caffeinate` so it does not sleep.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd -P)"
docker build -q -t agentlab -f "$here/docker/Dockerfile" "$here" >/dev/null
docker rm -f agentlab-long >/dev/null 2>&1 || true
args=("$@"); [[ $# -eq 0 ]] && args=(--days 7 --solvers 1)
docker run -d --name agentlab-long --restart unless-stopped \
  --memory 6g --cpus "$(( $(getconf _NPROCESSORS_ONLN) > 1 ? $(getconf _NPROCESSORS_ONLN) - 1 : 1 ))" \
  -v "$here":/work/ecdsafail-agent \
  -v agentlab-challenge:/root/ecdsafail-challenge \
  -v agentlab-work:/work/ecdsafail-agent/work \
  -v agentlab-codex:/root/.codex \
  agentlab agentlab run "${args[@]}" >/dev/null
if command -v caffeinate >/dev/null 2>&1 && ! pgrep -f "caffeinate -is" >/dev/null; then
  nohup caffeinate -is >/dev/null 2>&1 &
fi
echo "started: agentlab run ${args[*]}   (container agentlab-long)"
echo "follow with docker/logs.sh; status any time with: docker exec agentlab-long agentlab status"

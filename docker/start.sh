#!/usr/bin/env bash
# Start a long, unattended run in a detached container (survives closing the terminal).
#
#   docker/start.sh                 continuous: 2 slots always busy, a unit summary every 5 h, until docker/stop.sh
#                                   or until the Coordinator declares the campaign hopeless
#   docker/start.sh --hours 5 --solvers 2   one bounded unit (launch for 5 h, then let running runs finish)
#   docker/start.sh --cleanroom     the clean-room track (our own circuit, docs/cleanroom.md): same loop on
#                                   work/cleanroom/ with its own ledger and base; combine with the options above
#
# Then:  docker/logs.sh   (follow the log)     docker/shell.sh   (a shell inside; e.g. `agentlab watch`)
#        docker/stop.sh   (stop the run)
# Keep the Mac plugged in with the lid open; this script also starts `caffeinate` so it does not sleep.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd -P)"
docker build -q -t agentlab -f "$here/docker/Dockerfile" "$here" >/dev/null
docker rm -f agentlab-long >/dev/null 2>&1 || true
env_args=(); mem=6g; track=""
if [[ "${1:-}" == "--cleanroom" ]]; then
  shift
  # our circuit is larger than the leaderboard's for now (36M ops): each build or evaluation holds ~2.3 GB
  env_args=(-e AGENTLAB_CONFIG=agentlab-cleanroom.toml); mem=8g; track=" (clean-room track)"
fi
args=("$@"); [[ $# -eq 0 ]] && args=(--continuous --solvers 2)
# on-failure: a crash restarts the loop, a finished unit (clean exit) does not
docker run -d --name agentlab-long --restart on-failure "${env_args[@]}" \
  --memory "$mem" --cpus "$(( $(getconf _NPROCESSORS_ONLN) > 1 ? $(getconf _NPROCESSORS_ONLN) - 1 : 1 ))" \
  -v "$here":/work/ecdsafail-agent \
  -v agentlab-challenge:/root/ecdsafail-challenge \
  -v agentlab-work:/work/ecdsafail-agent/work \
  -v agentlab-codex:/root/.codex \
  agentlab agentlab run "${args[@]}" >/dev/null
if command -v caffeinate >/dev/null 2>&1 && ! pgrep -f "caffeinate -is" >/dev/null; then
  nohup caffeinate -is >/dev/null 2>&1 &
fi
echo "started: agentlab run ${args[*]}${track}   (container agentlab-long)"
echo "follow with docker/logs.sh; status any time with: docker exec agentlab-long agentlab status"
echo "the Coordinator's decisions: docker exec agentlab-long cat work/coordinator.md"
echo "unit summaries (every 5 h in continuous mode) are in work/units/; stop with docker/stop.sh"

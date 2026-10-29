#!/usr/bin/env bash
# Stop the long run. Finished experiments, the ledger and logs stay in the agentlab-work volume.
docker rm -f agentlab-long >/dev/null 2>&1 && echo "stopped" || echo "not running"
pkill -f "caffeinate -is" 2>/dev/null || true

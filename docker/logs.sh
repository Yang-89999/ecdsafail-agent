#!/usr/bin/env bash
# Follow the long run's log (Ctrl-C stops following, not the run).
exec docker logs -f --tail "${1:-200}" agentlab-long

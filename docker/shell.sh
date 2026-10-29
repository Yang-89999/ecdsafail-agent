#!/usr/bin/env bash
# A shell inside the running long container (e.g. for `agentlab watch`, `agentlab status`, `agentlab approve`).
exec docker exec -it agentlab-long bash

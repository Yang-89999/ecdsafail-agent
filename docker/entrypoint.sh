#!/usr/bin/env bash
# First start: clone the harness into its volume, install agentlab from the mounted repo, build.
set -euo pipefail

if [[ ! -d /root/ecdsafail-challenge/.git ]]; then
  echo "== cloning ecdsafail-challenge into the volume (one time, ~340 MB) =="
  git clone https://github.com/ecdsafail/ecdsafail-challenge.git /root/ecdsafail-challenge
fi

# The container is fresh on every start (--rm), so install the CLI each time (a few seconds).
# Do not test with `import agentlab`: the working directory is the repo, so the import succeeds
# even when the `agentlab` command is missing.
if ! command -v agentlab >/dev/null 2>&1; then
  pip install -q --root-user-action=ignore -e /work/ecdsafail-agent
fi

if [[ ! -x /root/ecdsafail-challenge/target/release/eval_circuit ]]; then
  echo "== first build of the harness (several minutes) =="
  agentlab setup
fi

# Unprivileged user namespaces are usually blocked inside Docker, so bubblewrap cannot start.
# The container already isolates the untrusted build, so let the scripts fall back to a plain run.
if command -v bwrap >/dev/null 2>&1 && ! bwrap --ro-bind / / --unshare-user true 2>/dev/null; then
  echo "== bubblewrap cannot run in this container; using the container as the sandbox =="
  apt-get remove -y -qq bubblewrap >/dev/null 2>&1 || rm -f "$(command -v bwrap)"
fi

if ! codex login status >/dev/null 2>&1; then
  echo
  echo "Codex is not logged in. Run:  codex login --device-auth"
  echo "(prints a code; open the URL on your Mac, sign in with ChatGPT, enter the code)"
  echo
fi

exec "$@"

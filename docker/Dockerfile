# agentlab runtime: Linux, like the organisers' scorer. The container is the sandbox for the
# untrusted circuit code and for the commands Codex runs; nothing in it can reach the host's files
# except the mounted agentlab repo.
ARG BASE=node:22-bookworm
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git python3 python3-pip python3-venv libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Rust: the toolchain the harness pins (rust-toolchain = 1.93.0).
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.93.0 \
    && chmod -R a+rwX /opt/rustup /opt/cargo
# setup.sh sources ~/.cargo/env; point it at the shared install.
RUN mkdir -p /root/.cargo && printf 'export PATH=/opt/cargo/bin:$PATH\n' > /root/.cargo/env

RUN npm install -g @openai/codex

WORKDIR /work/ecdsafail-agent
ENV AGENTLAB_CODEX_SANDBOX=danger-full-access \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    GIT_PAGER=cat PAGER=cat
COPY docker/entrypoint.sh /usr/local/bin/agentlab-entry
RUN chmod +x /usr/local/bin/agentlab-entry
ENTRYPOINT ["agentlab-entry"]
CMD ["bash"]

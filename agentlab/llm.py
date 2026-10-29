"""Model backends.

Two ways to run a solver:
  * CodexBackend  - `codex exec` (OpenAI Codex CLI) does the whole edit/build/score loop itself,
                    sandboxed to the worktree. Auth: `codex login` or OPENAI_API_KEY.
  * OpenAIBackend - a plain Responses API tool loop with read/write/shell tools restricted to the
                    worktree. Used for the cheaper roles (scout summaries, planner notes) and as a
                    fallback solver.

Both return the agent's final message as text; the solver role parses the JSON block from it.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentRun:
    final_message: str
    log_path: str
    duration_s: float
    returncode: int


class CodexBackend:
    def __init__(self, model: str = "", reasoning_effort: str = "high", timeout_s: int = 3 * 3600):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s

    def run(self, prompt: str, cwd: Path, log_path: Path, extra_dirs: list[Path] | None = None) -> AgentRun:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        last = log_path.with_suffix(".last.md")
        cmd = [
            "codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral",
            "-C", str(cwd), "--sandbox", "workspace-write",
            "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
            "-o", str(last),
        ]
        if self.model:
            cmd += ["-m", self.model]
        for d in extra_dirs or []:
            cmd += ["--add-dir", str(d)]
        cmd += ["-"]  # prompt on stdin
        t0 = time.time()
        with open(log_path, "w") as log:
            log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            try:
                proc = subprocess.run(cmd, input=prompt, stdout=log, stderr=subprocess.STDOUT,
                                      text=True, timeout=self.timeout_s, cwd=cwd)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                log.write("\n!! codex exec timed out\n")
                rc = 124
        msg = last.read_text() if last.exists() else ""
        return AgentRun(msg, str(log_path), time.time() - t0, rc)


class OpenAIBackend:
    """Minimal tool loop on the Responses API. Tools are confined to `cwd`."""

    def __init__(self, model: str = "", reasoning_effort: str = "high", timeout_s: int = 3 * 3600, max_turns: int = 200):
        from openai import OpenAI  # lazy import

        self.client = OpenAI()
        self.model = model or os.environ.get("AGENTLAB_OPENAI_MODEL", "gpt-5")
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.max_turns = max_turns

    TOOLS = [
        {"type": "function", "name": "read_file", "description": "Read a UTF-8 file under the workspace.",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                        "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}},
        {"type": "function", "name": "write_file", "description": "Write a whole file under the workspace (editable paths only).",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"type": "function", "name": "shell", "description": "Run a shell command in the workspace (timeout in seconds, default 1800).",
         "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["cmd"]}},
    ]

    def _tool(self, name: str, args: dict, cwd: Path) -> str:
        root = cwd.resolve()

        def safe(p: str) -> Path:
            q = (root / p).resolve()
            if root not in q.parents and q != root:
                raise ValueError(f"path escapes workspace: {p}")
            return q

        if name == "read_file":
            text = safe(args["path"]).read_text(errors="replace").splitlines()
            s, e = args.get("start_line") or 1, args.get("end_line") or len(text)
            return "\n".join(f"{i}: {l}" for i, l in enumerate(text[s - 1:e], s))[:200_000]
        if name == "write_file":
            p = safe(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"wrote {len(args['content'])} bytes to {p.relative_to(root)}"
        if name == "shell":
            try:
                proc = subprocess.run(args["cmd"], shell=True, cwd=root, capture_output=True, text=True,
                                      timeout=args.get("timeout") or 1800)
                return (proc.stdout + proc.stderr)[-60_000:] + f"\n[exit {proc.returncode}]"
            except subprocess.TimeoutExpired:
                return "[timed out]"
        return f"unknown tool {name}"

    def run(self, prompt: str, cwd: Path, log_path: Path, extra_dirs: list[Path] | None = None) -> AgentRun:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        inputs: list = [{"role": "user", "content": prompt}]
        final = ""
        with open(log_path, "w") as log:
            for turn in range(self.max_turns):
                if time.time() - t0 > self.timeout_s:
                    log.write("\n!! timed out\n")
                    break
                resp = self.client.responses.create(
                    model=self.model, input=inputs, tools=self.TOOLS,
                    reasoning={"effort": self.reasoning_effort},
                )
                inputs += resp.output
                calls = [o for o in resp.output if getattr(o, "type", "") == "function_call"]
                if not calls:
                    final = resp.output_text
                    log.write(f"\n=== final ===\n{final}\n")
                    break
                for c in calls:
                    args = json.loads(c.arguments or "{}")
                    log.write(f"\n> {c.name} {json.dumps(args)[:500]}\n")
                    try:
                        out = self._tool(c.name, args, cwd)
                    except Exception as e:  # noqa: BLE001
                        out = f"error: {e}"
                    log.write(out[:4000] + "\n")
                    inputs.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
        return AgentRun(final, str(log_path), time.time() - t0, 0)

    def complete(self, prompt: str, system: str = "") -> str:
        """One-shot text completion for cheap roles."""
        resp = self.client.responses.create(
            model=self.model, instructions=system or None, input=prompt,
            reasoning={"effort": "medium"},
        )
        return resp.output_text


class DryBackend:
    """No model: leaves the tree untouched and reports no_gain. For smoke-testing the pipeline."""

    def run(self, prompt: str, cwd: Path, log_path: Path, extra_dirs=None) -> AgentRun:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("dry run: no model called\n")
        msg = ('dry run\n```json\n{"status": "no_gain", "score": null, "toffoli": null, "qubits": null, '
               '"what_changed": "nothing (dry backend)", "why_it_worked_or_not": "no model", "next_ideas": []}\n```')
        return AgentRun(msg, str(log_path), 0.0, 0)


def make_backend(cfg):
    if cfg.backend == "dry":
        return DryBackend()
    if cfg.backend == "codex":
        return CodexBackend(cfg.model, cfg.reasoning_effort, cfg.solver_timeout_s)
    if cfg.backend == "openai":
        return OpenAIBackend(cfg.model, cfg.reasoning_effort, cfg.solver_timeout_s)
    raise SystemExit(f"unknown backend {cfg.backend}")

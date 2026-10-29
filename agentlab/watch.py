"""Readable live view of a solver run: `agentlab watch [exp]`.

Codex writes a JSON event per line to work/runs/exp<N>/agent.log. This follows that file
and prints each event as one or a few plain lines: the command the agent ran and the tail of
its output, the agent's messages, reasoning summaries, file changes, token usage.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path


def latest_run(runs_dir: Path) -> Path | None:
    runs = sorted((p for p in runs_dir.glob("exp*") if p.is_dir()),
                  key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    return runs[-1] if runs else None


def _short(s: str, n: int = 400) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n // 2] + " … " + s[-n // 2:]


def render(line: str) -> str | None:
    line = line.rstrip("\n")
    if not line.startswith("{"):
        return line if line.strip() else None
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return line
    t = ev.get("type", "")
    item = ev.get("item") or {}
    it = item.get("type", "")
    if t == "thread.started":
        return f"── session {ev.get('thread_id', '')}"
    if t == "turn.started":
        return "── turn started"
    if t == "turn.completed":
        u = ev.get("usage") or {}
        return (f"── turn done  in={u.get('input_tokens', '?')} cached={u.get('cached_input_tokens', '?')} "
                f"out={u.get('output_tokens', '?')}")
    if t == "turn.failed" or t == "error":
        return f"!! {ev.get('error') or ev.get('message') or line}"
    if t == "item.started" and it == "command_execution":
        return f"$ {_short(item.get('command', ''), 300)}"
    if t == "item.completed":
        if it == "command_execution":
            out = _short(item.get("aggregated_output") or "", 600)
            tail = ("\n    " + out.replace("\n", "\n    ")) if out else ""
            return f"  ↳ exit {item.get('exit_code')}{tail}"
        if it == "agent_message":
            return f"\n🗨  {item.get('text', '').strip()}\n"
        if it == "reasoning":
            return f"  … {_short(item.get('text') or item.get('summary') or '', 300)}"
        if it == "file_change":
            files = ", ".join(c.get("path", "?") for c in item.get("changes", []))
            return f"✎ {files}"
        if it in ("mcp_tool_call", "web_search", "todo_list"):
            return f"  [{it}] {_short(json.dumps(item), 200)}"
    return None


def watch(runs_dir: Path, exp: int | None, follow: bool = True) -> int:
    run_dir = runs_dir / f"exp{exp}" if exp else latest_run(runs_dir)
    if not run_dir or not run_dir.exists():
        print(f"no run found under {runs_dir}")
        return 1
    log = run_dir / "agent.log"
    print(f"watching {log}  (Ctrl-C to stop; the run keeps going)")
    while not log.exists():
        if not follow:
            print("no agent.log yet"); return 1
        time.sleep(2)
    with open(log, encoding="utf-8", errors="replace") as f:
        quiet = 0
        while True:
            line = f.readline()
            if line:
                quiet = 0
                out = render(line)
                if out is not None:
                    print(out, flush=True)
                continue
            if (run_dir / "report.json").exists() or not follow:
                print(f"── run finished; report: {run_dir / 'report.json'}")
                return 0
            time.sleep(2)
            quiet += 2
            if quiet % 120 == 0:
                print(f"  (no new events for {quiet // 60} min; the model is thinking or a build is running)", flush=True)

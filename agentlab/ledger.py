"""Experiment ledger: every idea, run, score and verdict, in one SQLite file.

The ledger is the system's memory. The Planner ranks from it, the Auditor
checks against it, and public notes may quote only numbers recorded here
from an actual run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
  id INTEGER PRIMARY KEY,
  created REAL NOT NULL,
  title TEXT NOT NULL UNIQUE,
  rationale TEXT NOT NULL,
  source TEXT NOT NULL,          -- seed | scout:<commit> | solver:<exp id> | human
  status TEXT NOT NULL DEFAULT 'open',  -- candidate | open | exhausted | done | rejected
  prior REAL NOT NULL DEFAULT 0.02,     -- optimistic prior relative gain
  tries INTEGER NOT NULL DEFAULT 0,
  gain_sum REAL NOT NULL DEFAULT 0.0,   -- sum of relative gains observed (can be negative)
  best_gain REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS experiments (
  id INTEGER PRIMARY KEY,
  started REAL NOT NULL,
  finished REAL,
  idea_id INTEGER REFERENCES ideas(id),
  worktree TEXT,
  base_commit TEXT,
  head_commit TEXT,
  model TEXT,
  status TEXT NOT NULL,          -- running | scored | failed | verified | audited | rejected | submitted
  shots INTEGER,
  toffoli REAL,
  qubits INTEGER,
  score INTEGER,
  gain REAL,                      -- relative gain vs best at start: (best - score) / best
  note TEXT,
  log_path TEXT,
  details TEXT                    -- JSON blob
);
CREATE TABLE IF NOT EXISTS bests (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  score INTEGER NOT NULL,
  toffoli REAL,
  qubits INTEGER,
  source TEXT NOT NULL,          -- upstream:<commit> | local:<exp id>
  verified INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS papers (
  id TEXT PRIMARY KEY,            -- arxiv id, DOI or URL
  fetched REAL NOT NULL,
  title TEXT NOT NULL,
  authors TEXT,
  published TEXT,
  url TEXT,
  abstract TEXT,
  query TEXT,
  extracted INTEGER NOT NULL DEFAULT 0  -- ideas extracted by the literature scout
);
CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  experiment_id INTEGER NOT NULL,
  approved_by TEXT NOT NULL,
  note TEXT
);
"""


class _Result:
    """A fully fetched query result, so no cursor is shared between threads."""

    def __init__(self, cur: sqlite3.Cursor):
        self.rows = cur.fetchall() if cur.description else []
        self.rowcount = cur.rowcount
        self.lastrowid = cur.lastrowid

    def fetchall(self) -> list:
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __iter__(self):
        return iter(self.rows)


class _LockedDB:
    """One sqlite connection shared by the loop, the parallel solver threads and the background trend
    thread. Every call is serialised; without this two threads committing at once raised
    "cannot commit - no transaction is active"."""

    def __init__(self, conn: sqlite3.Connection):
        self._c = conn
        self._lock = threading.RLock()

    def execute(self, sql: str, params: Any = ()) -> _Result:
        with self._lock:
            return _Result(self._c.execute(sql, params))

    def executemany(self, sql: str, seq) -> _Result:
        with self._lock:
            return _Result(self._c.executemany(sql, seq))

    def executescript(self, script: str) -> None:
        with self._lock:
            self._c.executescript(script)

    def commit(self) -> None:
        with self._lock:
            self._c.commit()


@dataclass
class Idea:
    id: int
    title: str
    rationale: str
    source: str
    status: str
    prior: float
    tries: int
    gain_sum: float
    best_gain: float
    review: str | None = None
    subproblem: str | None = None
    kind: str | None = None


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self.db = _LockedDB(conn)
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(ideas)")}
        if "review" not in cols:
            self.db.execute("ALTER TABLE ideas ADD COLUMN review TEXT")
        if "subproblem" not in cols:
            self.db.execute("ALTER TABLE ideas ADD COLUMN subproblem TEXT")
        if "kind" not in cols:
            self.db.execute("ALTER TABLE ideas ADD COLUMN kind TEXT")
        ecols = {r[1] for r in self.db.execute("PRAGMA table_info(experiments)")}
        if "mode" not in ecols:  # new | continue | combine, and the parent experiments of the last two
            self.db.execute("ALTER TABLE experiments ADD COLUMN mode TEXT")
            self.db.execute("ALTER TABLE experiments ADD COLUMN parents TEXT")
        self.db.commit()

    # ---- ideas -------------------------------------------------------------
    def add_idea(self, title: str, rationale: str, source: str, prior: float = 0.02, status: str = "open") -> int:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO ideas(created,title,rationale,source,prior,status) VALUES(?,?,?,?,?,?)",
            (time.time(), title.strip(), rationale.strip(), source, prior, status),
        )
        self.db.commit()
        if cur.lastrowid:
            return cur.lastrowid
        return self.db.execute("SELECT id FROM ideas WHERE title=?", (title.strip(),)).fetchone()[0]

    def ideas(self, status: str | None = "open") -> list[Idea]:
        q = "SELECT * FROM ideas" + (" WHERE status=?" if status else "") + " ORDER BY id"
        rows = self.db.execute(q, (status,) if status else ()).fetchall()
        return [Idea(**{k: r[k] for k in Idea.__dataclass_fields__}) for r in rows]

    def idea(self, idea_id: int) -> Idea:
        r = self.db.execute("SELECT * FROM ideas WHERE id=?", (idea_id,)).fetchone()
        return Idea(**{k: r[k] for k in Idea.__dataclass_fields__})

    def record_idea_outcome(self, idea_id: int, gain: float) -> None:
        self.db.execute(
            "UPDATE ideas SET tries=tries+1, gain_sum=gain_sum+?, best_gain=MAX(best_gain,?) WHERE id=?",
            (gain, gain, idea_id),
        )
        self.db.commit()

    def set_idea_subproblem(self, idea_id: int, name: str, kind: str | None = None) -> None:
        self.db.execute("UPDATE ideas SET subproblem=?, kind=COALESCE(?, kind) WHERE id=?", (name, kind, idea_id))
        self.db.commit()

    def set_idea_status(self, idea_id: int, status: str, review: str | None = None, prior: float | None = None) -> None:
        self.db.execute("UPDATE ideas SET status=?, review=COALESCE(?, review), prior=COALESCE(?, prior) WHERE id=?",
                        (status, review, prior, idea_id))
        self.db.commit()

    # ---- papers ------------------------------------------------------------
    def add_paper(self, pid: str, title: str, authors: str, published: str, url: str, abstract: str, query: str) -> bool:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO papers(id,fetched,title,authors,published,url,abstract,query) VALUES(?,?,?,?,?,?,?,?)",
            (pid, time.time(), title.strip(), authors, published, url, abstract.strip(), query),
        )
        self.db.commit()
        return cur.rowcount == 1

    def papers(self, unextracted_only: bool = False) -> list[sqlite3.Row]:
        q = "SELECT * FROM papers" + (" WHERE extracted=0" if unextracted_only else "") + " ORDER BY published DESC"
        return self.db.execute(q).fetchall()

    def mark_extracted(self, ids: Iterable[str]) -> None:
        self.db.executemany("UPDATE papers SET extracted=1 WHERE id=?", [(i,) for i in ids])
        self.db.commit()

    # ---- experiments -------------------------------------------------------
    def start_experiment(self, idea_id: int | None, worktree: str, base_commit: str, model: str, note: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO experiments(started,idea_id,worktree,base_commit,model,status,note) VALUES(?,?,?,?,?,?,?)",
            (time.time(), idea_id, worktree, base_commit, model, "running", note),
        )
        self.db.commit()
        return cur.lastrowid

    def finish_experiment(self, exp_id: int, status: str, **fields: Any) -> None:
        details = fields.pop("details", None)
        cols = ["finished=?", "status=?"]
        vals: list[Any] = [time.time(), status]
        for k, v in fields.items():
            cols.append(f"{k}=?")
            vals.append(v)
        if details is not None:
            cols.append("details=?")
            vals.append(json.dumps(details, default=str))
        vals.append(exp_id)
        self.db.execute(f"UPDATE experiments SET {', '.join(cols)} WHERE id=?", vals)
        self.db.commit()

    def experiment(self, exp_id: int) -> sqlite3.Row:
        return self.db.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()

    def experiments(self, limit: int = 50, status: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM experiments" + (" WHERE status=?" if status else "") + " ORDER BY id DESC LIMIT ?"
        args: tuple = (status, limit) if status else (limit,)
        return self.db.execute(q, args).fetchall()

    # ---- bests -------------------------------------------------------------
    def record_best(self, score: int, toffoli: float, qubits: int, source: str, verified: bool) -> None:
        if self.db.execute("SELECT 1 FROM bests WHERE source=? AND score=?", (source, score)).fetchone():
            return
        self.db.execute(
            "INSERT INTO bests(ts,score,toffoli,qubits,source,verified) VALUES(?,?,?,?,?,?)",
            (time.time(), score, toffoli, qubits, source, int(verified)),
        )
        self.db.commit()

    def best(self, verified_only: bool = True) -> sqlite3.Row | None:
        q = "SELECT * FROM bests" + (" WHERE verified=1" if verified_only else "") + " ORDER BY score ASC, ts DESC LIMIT 1"
        return self.db.execute(q).fetchone()

    # ---- approvals ---------------------------------------------------------
    def approve(self, exp_id: int, approved_by: str, note: str = "") -> None:
        self.db.execute(
            "INSERT INTO approvals(ts,experiment_id,approved_by,note) VALUES(?,?,?,?)",
            (time.time(), exp_id, approved_by, note),
        )
        self.db.commit()

    def approval(self, exp_id: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM approvals WHERE experiment_id=? ORDER BY ts DESC LIMIT 1", (exp_id,)
        ).fetchone()

    # ---- reporting ---------------------------------------------------------
    def summary_text(self, n: int = 15) -> str:
        best = self.best()
        lines = []
        if best:
            lines.append(f"Best verified score: {best['score']:,} (toffoli {best['toffoli']:.0f} x qubits {best['qubits']}) from {best['source']}")
        lines.append("Recent experiments:")
        for e in self.experiments(n):
            sc = f"{e['score']:,}" if e["score"] else "-"
            g = f"{e['gain']*100:+.3f}%" if e["gain"] is not None else ""
            mode = f" {e['mode']} {e['parents']}" if e["mode"] and e["mode"] != "new" else ""
            lines.append(f"  #{e['id']} idea={e['idea_id']}{mode} {e['status']:9s} score={sc} {g} {e['note'] or ''}".rstrip())
        lines.append("Open ideas (tries, mean gain):")
        for i in self.ideas("open"):
            mean = i.gain_sum / i.tries if i.tries else float("nan")
            lines.append(f"  [{i.id}] {i.title} (tries {i.tries}, mean {mean*100:.3f}%)" if i.tries else f"  [{i.id}] {i.title} (untried, prior {i.prior:.3f})")
        n_cand = len(self.ideas("candidate"))
        n_pap = self.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        if n_cand or n_pap:
            lines.append(f"Literature: {n_pap} papers fetched, {n_cand} candidate ideas awaiting review (`agentlab review`)")
        return "\n".join(lines)

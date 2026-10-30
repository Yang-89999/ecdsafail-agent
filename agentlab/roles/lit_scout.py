"""Literature Scout: papers and posts -> candidate ideas with citations.

1. Fetch: arXiv API for a fixed set of queries (plus any URLs the human hands in).
   Abstracts land in the `papers` table, deduplicated by arXiv id or URL.
2. Extract: the model reads batches of abstracts and proposes ideas that transfer to
   this harness's gate set and metric. Each idea cites its paper and enters the
   ledger as `candidate`; the Idea Reviewer decides whether it reaches the solvers.

The model never sees the solver prompt here; it only reads text. Numbers from papers
stay in the rationale as "paper claims".
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import Config
from ..ledger import Ledger
from ..llm import make_backend

DEFAULT_QUERIES = [
    "quantum circuit elliptic curve point addition Toffoli",
    "quantum resource estimates elliptic curve discrete logarithm",
    "reversible modular multiplication quantum circuit",
    "quantum modular inversion circuit extended Euclidean",
    "windowed quantum arithmetic lookup table",
    "Toffoli count quantum adder ancilla uncomputation",
    "measurement-based uncomputation Toffoli",
    "quantum circuit optimization reversible synthesis Toffoli depth width tradeoff",
    "secp256k1 quantum Shor resource",
    "space-time tradeoff quantum arithmetic qubits",
    # adjacent fields the competitors are less likely to have mined
    "exact synthesis reversible circuit SAT minimum Toffoli count",
    "AND-inverter graph resynthesis multiplicative complexity minimization",
    "logic optimization XOR-AND graph multiplicative complexity cryptographic circuits",
    "T-count optimization quantum circuit relative phase Toffoli",
    "pebble game reversible computation space time tradeoff",
    "constant-time modular inversion algorithm Bernstein Yang safegcd",
    "elliptic curve point addition hardware implementation modular inversion FPGA",
    "Montgomery multiplication lazy reduction elliptic curve arithmetic",
    "carry-save adder quantum circuit ancilla",
    "quantum lookup table QROM unary iteration uncomputation measurement",
]

ARXIV = "http://export.arxiv.org/api/query?"
NS = {"a": "http://www.w3.org/2005/Atom"}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "paper_id": {"type": "string"},
                    "claim": {"type": "string"},
                    "transfer": {"type": "string"},
                    "first_experiment": {"type": "string"},
                },
                "required": ["title", "paper_id", "claim", "transfer", "first_experiment"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ideas"],
    "additionalProperties": False,
}


def fetch_arxiv(query: str, max_results: int = 15, attempts: int = 3, sort: str = "relevance") -> list[dict]:
    q = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": max_results,
                                "sortBy": sort, "sortOrder": "descending"})
    for k in range(attempts):
        try:
            with urllib.request.urlopen(ARXIV + q, timeout=60) as r:
                root = ET.fromstring(r.read())
            break
        except Exception:  # noqa: BLE001  (arXiv often cuts responses short; it asks for ~3 s between calls)
            if k == attempts - 1:
                raise
            time.sleep(3 * (k + 1))
    out = []
    for e in root.findall("a:entry", NS):
        pid = (e.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]
        pid = re.sub(r"v\d+$", "", pid)
        out.append({
            "id": f"arxiv:{pid}",
            "title": re.sub(r"\s+", " ", e.findtext("a:title", "", NS) or "").strip(),
            "authors": ", ".join(a.findtext("a:name", "", NS) for a in e.findall("a:author", NS))[:300],
            "published": (e.findtext("a:published", "", NS) or "")[:10],
            "url": f"https://arxiv.org/abs/{pid}",
            "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", NS) or "").strip(),
        })
    return out


def fetch_url(url: str) -> dict:
    """A blog post or HTML page: strip tags, keep the first ~6000 characters as the 'abstract'."""
    req = urllib.request.Request(url, headers={"User-Agent": "agentlab-lit-scout/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8", errors="replace")
    title = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text).strip()
    return {"id": f"url:{url}", "title": html.unescape(title.group(1)).strip() if title else url,
            "authors": "", "published": "", "url": url, "abstract": text[:6000]}


def harvest(cfg: Config, ledger: Ledger, queries: list[str] | None = None, urls: list[str] | None = None,
            per_query: int = 15, recent: bool = False) -> int:
    """recent=True asks arXiv for the newest submissions instead of the most relevant (daily refresh)."""
    new = 0
    for q in queries or cfg.extra.get("lit_queries") or DEFAULT_QUERIES:
        try:
            papers = fetch_arxiv(q, per_query, sort="submittedDate" if recent else "relevance")
        except Exception as e:  # noqa: BLE001
            print(f"[lit-scout] arxiv query failed ({q!r}): {e}")
            continue
        for p in papers:
            if ledger.add_paper(p["id"], p["title"], p["authors"], p["published"], p["url"], p["abstract"], q):
                new += 1
        print(f"[lit-scout] {q!r}: {len(papers)} results, {new} new so far", flush=True)
        time.sleep(3)
    for u in urls or []:
        try:
            p = fetch_url(u)
            if ledger.add_paper(p["id"], p["title"], p["authors"], p["published"], p["url"], p["abstract"], "url"):
                new += 1
        except Exception as e:  # noqa: BLE001
            print(f"[lit-scout] url failed ({u}): {e}")
    return new


def _skill(cfg: Config, name: str) -> str:
    p = Path(cfg.skills_dir) / name / "SKILL.md"
    return p.read_text() if p.exists() else ""


def extract(cfg: Config, ledger: Ledger, batch: int = 8, max_batches: int = 50) -> int:
    backend = make_backend(cfg)
    papers = ledger.papers(unextracted_only=True)
    known = [i.title for i in ledger.ideas(None)]
    added = 0
    failures = 0
    print(f"[lit-scout] extracting from {len(papers)} unread papers with model {cfg.model or '(Codex default)'}", flush=True)
    for b in range(0, min(len(papers), batch * max_batches), batch):
        chunk = papers[b:b + batch]
        text = "\n\n".join(f"[{p['id']}] {p['title']} ({p['published']}) {p['url']}\n{p['abstract']}" for p in chunk)
        prompt = f"""{_skill(cfg, 'lit-scout')}

TARGET: a reversible circuit for one secp256k1 affine point addition, scored by average executed Toffoli count x peak
live qubits (lower is better). Gate set: X, CX, CZ, Swap, CCX/CCZ, reset and X-basis measurement, classical control.
Already in the backlog (do not repeat): {json.dumps(known[-60:])}

PAPERS:
{text}

Extract every idea from these papers that could lower Toffoli x qubits for that target. For each: a short title,
the paper id it comes from, the paper's claim in one sentence (with its numbers only if the abstract states them),
how it transfers to this target, and the first cheap experiment a solver should run. Skip papers that are off-target.
Return JSON matching the schema."""
        try:
            raw = backend.complete(prompt, schema=EXTRACT_SCHEMA)
            data = json.loads(raw) if raw.strip().startswith("{") else {"ideas": []}
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[lit-scout] extraction failed on batch {b // batch + 1}: {e}")
            if failures >= 2 and added == 0:
                print("[lit-scout] stopping: the model call keeps failing (check the model name). "
                      "Fetched papers are kept; rerun `agentlab lit-scout` to extract them.")
                return added
            continue
        for idea in data.get("ideas", []):
            title = idea["title"][:120]
            why = (f"Paper {idea['paper_id']}: {idea['claim']}\nTransfer: {idea['transfer']}\n"
                   f"First experiment: {idea['first_experiment']}")
            before = len(ledger.ideas(None))
            ledger.add_idea(title, why, f"lit:{idea['paper_id']}", prior=0.01, status="candidate")
            if len(ledger.ideas(None)) > before:
                added += 1
                known.append(title)
        ledger.mark_extracted([p["id"] for p in chunk])
        print(f"[lit-scout] batch {b // batch + 1}: {len(data.get('ideas', []))} ideas proposed, {added} new candidates total", flush=True)
    return added


ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "ideas": EXTRACT_SCHEMA["properties"]["ideas"],
    },
    "required": ["answer", "ideas"],
    "additionalProperties": False,
}


def ask(cfg: Config, ledger: Ledger, question: str, max_papers: int = 10) -> str:
    """On-demand literature question from a designer agent: search, read, answer with citations.

    Returns a short memo (markdown). Ideas found on the way enter the ledger as candidates with
    source `ask:<question>` so the reviewer can route them to other solvers too.
    """
    import hashlib

    memo_dir = Path(cfg.runs_dir).parent / "lit"
    memo_dir.mkdir(parents=True, exist_ok=True)
    memo = memo_dir / (hashlib.sha1(question.strip().lower().encode()).hexdigest()[:12] + ".md")
    if memo.exists():
        return memo.read_text()
    try:
        papers = fetch_arxiv(question, max_papers)
    except Exception as e:  # noqa: BLE001
        return f"(literature search failed: {e})"
    for p in papers:
        ledger.add_paper(p["id"], p["title"], p["authors"], p["published"], p["url"], p["abstract"], f"ask:{question[:80]}")
    text = "\n\n".join(f"[{p['id']}] {p['title']} ({p['published']}) {p['url']}\n{p['abstract']}" for p in papers)
    prompt = f"""{_skill(cfg, 'lit-scout')}

A circuit designer working on a reversible secp256k1 point-addition circuit (score = Toffoli count x peak qubits;
gate set X, CX, CZ, Swap, CCX/CCZ, reset, X-basis measurement, classical control) asks:

QUESTION: {question}

PAPERS FOUND FOR IT:
{text}

Answer the question in at most 200 words, citing paper ids in brackets; say plainly when the papers do not answer it.
Then list the transferable ideas (same fields as usual). Return JSON matching the schema."""
    try:
        raw = make_backend(cfg).complete(prompt, schema=ASK_SCHEMA)
        data = json.loads(raw) if raw.strip().startswith("{") else {"answer": raw, "ideas": []}
    except Exception as e:  # noqa: BLE001
        return f"(literature answer failed: {e})"
    for idea in data.get("ideas", []):
        why = (f"Paper {idea['paper_id']}: {idea['claim']}\nTransfer: {idea['transfer']}\n"
               f"First experiment: {idea['first_experiment']}\nAsked: {question}")
        ledger.add_idea(idea["title"][:120], why, f"ask:{idea['paper_id']}", prior=0.01, status="candidate")
    refs = "\n".join(f"- [{p['id']}] {p['title']} — {p['url']}" for p in papers)
    out = f"# {question}\n\n{data.get('answer', '').strip()}\n\n## Papers read\n{refs}\n"
    memo.write_text(out)
    return out

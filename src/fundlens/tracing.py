"""Observability: every request is appended to a JSONL trace (steps, tools, latency, tokens, flags), and
`summarise` turns the traces into the numbers served by GET /metrics.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path

from .agent import AgentResult

_lock = threading.Lock()


def record(path: Path, result: AgentResult) -> None:
    row = {
        "id": result.id,
        "model": result.model,
        "question": result.question,  # already PII-redacted
        "refused": result.refused,
        "flags": result.flags,
        "tools": result.tools_used,
        "citations": result.citations,
        "latency_ms": round(result.latency_ms, 1),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarise(path: Path) -> dict:
    if not path.exists():
        return {"requests": 0}
    rows = [json.loads(line) for line in path.open()]
    lat = [r["latency_ms"] for r in rows]
    flags = Counter(f.split(":")[0] for r in rows for f in r["flags"])
    return {
        "requests": len(rows),
        "latency_ms_p50": round(percentile(lat, 50), 1),
        "latency_ms_p95": round(percentile(lat, 95), 1),
        "avg_tokens": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / len(rows), 1),
        "refusal_rate": round(sum(r["refused"] for r in rows) / len(rows), 3),
        "cited_answer_rate": round(sum(bool(r["citations"]) for r in rows if not r["refused"])
                                   / max(1, sum(not r["refused"] for r in rows)), 3),
        "tool_usage": dict(Counter(t for r in rows for t in r["tools"])),
        "flags": dict(flags),
    }

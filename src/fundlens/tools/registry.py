"""The agent's tools: JSON schemas the model sees, plus validated Python implementations.

Every tool returns plain JSON-serialisable data and an `evidence_ids` list, so the answer can only cite
things a tool actually returned.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..guardrails import sanitize_retrieved
from ..retrieval.index import HybridIndex
from .metrics import compute_metrics
from .sql import SCHEMA_HINT, SQLRejected, run_readonly


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=5, ge=1, le=8)


class SQLArgs(BaseModel):
    sql: str = Field(min_length=6, max_length=2000)


class MetricsArgs(BaseModel):
    scheme_code: int
    years: float | None = Field(default=None, gt=0, le=15)
    start_date: str | None = None
    end_date: str | None = None


class ListArgs(BaseModel):
    pass


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search SEBI mutual fund regulations, SEBI investor FAQs and the ICICI Prudential Ultra "
            "Short Term Fund scheme document. Use for rules, definitions, fees, exit loads, risk and processes.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "a focused search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_schemes",
            "description": "List the ICICI Prudential schemes with NAV data: scheme_code, short name and SEBI category.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fund_metrics",
            "description": "Compute returns for one scheme from daily NAVs: CAGR, absolute return, max drawdown and "
            "volatility over the last N years or between two dates. Always use this for any return figure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scheme_code": {"type": "integer"},
                    "years": {"type": "number", "description": "look-back period in years, e.g. 1, 3 or 5"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD, optional"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD, optional"},
                },
                "required": ["scheme_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_nav_sql",
            "description": "Run one read-only SQL SELECT on the NAV database for questions the other tools can't "
            "answer, such as the latest NAV or the highest NAV on a date. " + SCHEMA_HINT,
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
]


@dataclass
class ToolResult:
    ok: bool
    data: Any
    evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None


class Toolbox:
    def __init__(self, settings: Settings, index: HybridIndex):
        self.settings = settings
        self.index = index
        self._impl: dict[str, tuple[type[BaseModel], Callable[[Any], ToolResult]]] = {
            "search_documents": (SearchArgs, self._search),
            "list_schemes": (ListArgs, self._list),
            "fund_metrics": (MetricsArgs, self._metrics),
            "query_nav_sql": (SQLArgs, self._sql),
        }

    @property
    def names(self) -> list[str]:
        return list(self._impl)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self._impl:
            return ToolResult(False, None, error=f"unknown tool '{name}'")
        model, fn = self._impl[name]
        try:
            parsed = model.model_validate(args or {})
        except ValidationError as e:
            return ToolResult(False, None, error=f"invalid arguments: {e.errors(include_url=False)}")
        try:
            return fn(parsed)
        except (ValueError, SQLRejected) as e:
            return ToolResult(False, None, error=str(e))

    # -- implementations -------------------------------------------------------------------------------
    def _search(self, a: SearchArgs) -> ToolResult:
        hits = self.index.search(a.query, k=min(a.k, self.settings.top_k))
        passages = []
        for h in hits:
            text, flagged = sanitize_retrieved(h.chunk.text)
            passages.append(
                {"id": h.chunk.id, "source": h.chunk.title, "page": h.chunk.page, "text": text, "flagged": flagged}
            )
        return ToolResult(True, passages, [p["id"] for p in passages])

    def _list(self, _: ListArgs) -> ToolResult:
        con = sqlite3.connect(f"file:{self.settings.db_file}?mode=ro", uri=True)
        rows = con.execute("SELECT code, short_name, category FROM schemes ORDER BY code").fetchall()
        con.close()
        data = [{"scheme_code": r[0], "name": f"ICICI Prudential {r[1]}", "category": r[2]} for r in rows]
        return ToolResult(True, data, [f"scheme:{r[0]}" for r in rows])

    def _metrics(self, a: MetricsArgs) -> ToolResult:
        if a.years is None and a.start_date is None:
            a.years = 3.0
        m = compute_metrics(self.settings.db_file, a.scheme_code, a.years, a.start_date, a.end_date)
        return ToolResult(True, m.to_dict(), [f"nav:{m.code}:{m.start_date}:{m.end_date}"])

    def _sql(self, a: SQLArgs) -> ToolResult:
        out = run_readonly(self.settings.db_file, a.sql)
        return ToolResult(True, out, ["sql:nav"])

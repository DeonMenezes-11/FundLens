"""Deterministic intent routing that runs before the model.

Small models are weakest at choosing tools, so the obvious cases are decided by plain code: questions about a
fund's returns go straight to the analytics tool for the right scheme and period, "which schemes" questions get
the scheme list, and everything else is grounded in a document search. The model still sees all tools and can
call more.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_METRIC = re.compile(
    r"\b(cagr|returns?|drawdown|volatil\w*|perform\w*|grown|growth|gained|lost|fall|fell|drop(ped)?|declined?|"
    r"crash(ed)?|risk(ier|iest)?)\b",
    re.I,
)
_LIST = re.compile(
    r"\b(which|what|list|show)\b.*\b(schemes|funds)\b.*\b(database|available|have|cover|tracked|in your data)\b"
    r"|\blist (all |the )?(schemes|funds)\b",
    re.I,
)
_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:years?|yrs?|yr)\b", re.I)
_WORD_YEARS = {"one": 1, "two": 2, "three": 3, "five": 5, "ten": 10}

# extra ways people refer to the schemes in sources.json (matched case-insensitively, longest first)
ALIASES = {
    "bluechip": "Large Cap",
    "large and mid cap": "Large & Mid Cap",
    "tech fund": "Technology",
    "technology": "Technology",
    "tax saver": "ELSS Tax Saver",
    "elss": "ELSS Tax Saver",
    "gold": "Gold ETF FOF",
    "smallcap 250": "Nifty Smallcap 250 Index",
    "small cap index": "Nifty Smallcap 250 Index",
    "midcap 150": "Nifty Midcap 150 Index",
    "mid cap index": "Nifty Midcap 150 Index",
    "next 50": "Nifty Next 50 Index",
    "balanced advantage": "Balanced Advantage",
    "corporate bond": "Corporate Bond",
    "ultra short": "Ultra Short Term",
    "liquid": "Liquid",
    "value": "Value",
}


def load_schemes(db_file: Path) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    rows = con.execute("SELECT short_name, code FROM schemes").fetchall()
    con.close()
    return {name: code for name, code in rows}


def resolve_scheme(question: str, schemes: dict[str, int]) -> int | None:
    """Longest matching scheme name or alias wins, so "Large & Mid Cap" beats "Large Cap"."""
    q = question.lower().replace("&", "and")
    names = {n.lower().replace("&", "and"): n for n in schemes}
    candidates = {**{k: v for k, v in names.items()}, **{a: n for a, n in ALIASES.items() if n in schemes}}
    for key in sorted(candidates, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", q):
            return schemes[candidates[key]]
    return None


def parse_years(question: str, default: float = 3.0) -> float:
    m = _YEARS.search(question)
    if m:
        return float(m.group(1))
    for word, n in _WORD_YEARS.items():
        if re.search(rf"\b{word}[- ]years?\b", question, re.I):
            return float(n)
    return default


def plan(question: str, schemes: dict[str, int]) -> list[tuple[str, dict]]:
    """Tool calls to run before the model sees the question."""
    if _LIST.search(question):
        return [("list_schemes", {}), ("search_documents", {"query": question})]
    if _METRIC.search(question):
        code = resolve_scheme(question, schemes)
        if code is not None:
            return [("fund_metrics", {"scheme_code": code, "years": parse_years(question)})]
        return []  # a returns question about an unknown fund: let the model look it up (or say it can't)
    return [("search_documents", {"query": question})]

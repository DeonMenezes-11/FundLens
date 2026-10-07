"""Deterministic fund analytics. The LLM never does arithmetic: it calls these functions and reports the result.

All figures are computed from daily NAVs of growth plans, so dividends are already reinvested.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass
class Metrics:
    code: int
    scheme: str
    start_date: str
    end_date: str
    start_nav: float
    end_nav: float
    years: float
    absolute_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    annualised_volatility_pct: float

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def cagr(start_nav: float, end_nav: float, years: float) -> float:
    if years <= 0 or start_nav <= 0:
        raise ValueError("need a positive period and starting NAV")
    return ((end_nav / start_nav) ** (1 / years) - 1) * 100


def max_drawdown(navs: list[float]) -> float:
    """Largest peak-to-trough fall, as a negative percentage."""
    peak = navs[0]
    worst = 0.0
    for v in navs:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return worst * 100


def annualised_volatility(navs: list[float]) -> float:
    rets = [math.log(b / a) for a, b in zip(navs, navs[1:], strict=False) if a > 0 and b > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252) * 100


def resolve_period(con: sqlite3.Connection, code: int, years: float | None, start: str | None, end: str | None):
    last = con.execute("SELECT max(date) FROM nav WHERE code=?", (code,)).fetchone()[0]
    if last is None:
        raise ValueError(f"no NAV data for scheme {code}")
    end_d = date.fromisoformat(end) if end else date.fromisoformat(last)
    if start:
        start_d = date.fromisoformat(start)
    elif years:
        start_d = end_d - timedelta(days=round(365.25 * years))
    else:
        raise ValueError("give either years or a start date")
    return start_d, end_d


def compute_metrics(db_file: Path, code: int, years: float | None = None, start: str | None = None,
                    end: str | None = None) -> Metrics:
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        start_d, end_d = resolve_period(con, code, years, start, end)
        name = con.execute("SELECT name FROM schemes WHERE code=?", (code,)).fetchone()
        if not name:
            raise ValueError(f"unknown scheme code {code}")
        # last available NAV on or before each boundary (markets are closed on weekends/holidays)
        first = con.execute(
            "SELECT date, nav FROM nav WHERE code=? AND date<=? ORDER BY date DESC LIMIT 1", (code, start_d.isoformat())
        ).fetchone()
        if first is None:
            raise ValueError(f"scheme {code} has no NAV on or before {start_d}; it may not have existed yet")
        rows = con.execute(
            "SELECT date, nav FROM nav WHERE code=? AND date>=? AND date<=? ORDER BY date",
            (code, first[0], end_d.isoformat()),
        ).fetchall()
    finally:
        con.close()
    navs = [r[1] for r in rows]
    d0, d1 = date.fromisoformat(rows[0][0]), date.fromisoformat(rows[-1][0])
    years_exact = (d1 - d0).days / 365.25
    return Metrics(
        code=code,
        scheme=name[0],
        start_date=rows[0][0],
        end_date=rows[-1][0],
        start_nav=navs[0],
        end_nav=navs[-1],
        years=years_exact,
        absolute_return_pct=(navs[-1] / navs[0] - 1) * 100,
        cagr_pct=cagr(navs[0], navs[-1], years_exact),
        max_drawdown_pct=max_drawdown(navs),
        annualised_volatility_pct=annualised_volatility(navs),
    )

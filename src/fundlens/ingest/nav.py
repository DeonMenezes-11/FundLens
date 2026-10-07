"""Structured side: daily NAV history (AMFI data via the public mfapi.in API) loaded into SQLite.

A dated snapshot (data/nav_snapshot.csv.gz) is committed so evaluation numbers are reproducible; `refresh`
pulls the latest NAVs when you want live data.
"""
from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from datetime import datetime

import httpx

from ..config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS schemes (
    code INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    fund_house TEXT NOT NULL,
    category TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nav (
    code INTEGER NOT NULL REFERENCES schemes(code),
    date TEXT NOT NULL,          -- ISO yyyy-mm-dd
    nav REAL NOT NULL,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS nav_date ON nav(date);
"""


def fetch_snapshot(settings: Settings) -> int:
    """Download NAV history for the configured schemes and write the gzip CSV snapshot."""
    schemes = json.loads(settings.sources_file.read_text())["schemes"]
    rows = []
    with httpx.Client(timeout=60) as client:
        for s in schemes:
            data = client.get(f"https://api.mfapi.in/mf/{s['code']}").json()
            meta = data["meta"]
            for point in data["data"]:
                iso = datetime.strptime(point["date"], "%d-%m-%Y").strftime("%Y-%m-%d")
                rows.append(
                    [s["code"], meta["scheme_name"], s["short"], meta["fund_house"], meta["scheme_category"], iso, point["nav"]]
                )
    with gzip.open(settings.nav_snapshot, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "short_name", "fund_house", "category", "date", "nav"])
        w.writerows(rows)
    return len(rows)


def build_db(settings: Settings) -> sqlite3.Connection:
    """(Re)build the SQLite database from the committed snapshot."""
    settings.db_file.unlink(missing_ok=True)
    con = sqlite3.connect(settings.db_file)
    con.executescript(SCHEMA)
    with gzip.open(settings.nav_snapshot, "rt") as f:
        reader = csv.DictReader(f)
        seen = set()
        navs = []
        for r in reader:
            code = int(r["code"])
            if code not in seen:
                con.execute(
                    "INSERT INTO schemes VALUES (?,?,?,?,?)",
                    (code, r["name"], r["short_name"], r["fund_house"], r["category"]),
                )
                seen.add(code)
            nav = float(r["nav"])
            if nav > 0:  # mfapi occasionally reports 0 on non-trading days
                navs.append((code, r["date"], nav))
    con.executemany("INSERT OR REPLACE INTO nav VALUES (?,?,?)", navs)
    con.commit()
    return con

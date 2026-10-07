"""Read-only SQL access for the agent, defended in depth.

A model that writes SQL is an untrusted client. Four independent layers stop it from doing damage:
  1. static check: exactly one statement, must start with SELECT/WITH, no write/DDL/PRAGMA/ATTACH keywords
  2. the connection is opened read-only (sqlite URI mode=ro)
  3. a SQLite authorizer denies every action except reads on the two whitelisted tables
  4. a row cap and a progress-handler timeout bound the cost of any query
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

ALLOWED_TABLES = {"schemes", "nav"}
MAX_ROWS = 200
TIMEOUT_S = 2.0

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|grant|load_extension)\b",
    re.I,
)

SCHEMA_HINT = (
    "Tables: schemes(code INTEGER PRIMARY KEY, name TEXT, short_name TEXT, fund_house TEXT, category TEXT); "
    "nav(code INTEGER, date TEXT 'YYYY-MM-DD', nav REAL). Join on code. Dates are ISO strings."
)


class SQLRejected(ValueError):
    pass


def check_sql(sql: str) -> str:
    stmt = sql.strip().rstrip(";").strip()
    if not stmt:
        raise SQLRejected("empty query")
    if ";" in stmt:
        raise SQLRejected("only one statement is allowed")
    if not re.match(r"^(select|with)\b", stmt, re.I):
        raise SQLRejected("only SELECT queries are allowed")
    if _FORBIDDEN.search(stmt):
        raise SQLRejected("query contains a forbidden keyword")
    return stmt


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() != "load_extension" else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def run_readonly(db_file: Path, sql: str) -> dict:
    stmt = check_sql(sql)
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        con.set_authorizer(_authorizer)
        deadline = time.monotonic() + TIMEOUT_S
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cur = con.execute(stmt)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.DatabaseError as e:
        raise SQLRejected(f"query failed: {e}") from e
    finally:
        con.close()
    truncated = len(rows) > MAX_ROWS
    return {"columns": cols, "rows": [list(r) for r in rows[:MAX_ROWS]], "truncated": truncated}

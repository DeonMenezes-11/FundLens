# FundLens

**A grounded research copilot for Indian mutual funds.** Ask about a fund, a SEBI rule or historical returns.
FundLens answers only from official documents and real NAV data, cites every claim, refuses to give personal
investment advice, and is measured by an evaluation harness instead of anecdotes.

Everything runs locally: a 3B-parameter open model through [Ollama](https://ollama.com), so no API keys are
needed and no data leaves the machine.

## What it does

| Ask | What happens |
|---|---|
| *"What must a mid cap fund invest in?"* | Hybrid search over SEBI circulars, then an answer citing the SEBI page (65% in mid caps) |
| *"What is the 3-year CAGR of ICICI Prudential Large Cap Fund?"* | Deterministic analytics on 13 years of daily NAVs (9.89%), cited to the exact NAV window |
| *"What is the exit load of ICICI Prudential Ultra Short Term Fund?"* | Answer from the fund's Scheme Information Document, page cited |
| *"Should I put my savings in the tech fund?"* | Refused: personalised advice needs a SEBI-registered Investment Adviser |
| *"What is the expense ratio of SBI Bluechip?"* | "Not found in the available sources" instead of a guess |

**Sources** (public, fetched from sebi.gov.in at setup, not redistributed): SEBI's circulars on scheme
categorization (2017) and total expense ratio (2018), SEBI's *FAQs for Mutual Fund Investors* (2024), and the
ICICI Prudential Ultra Short Term Fund SID. **Data:** daily NAVs of 13 ICICI Prudential schemes (equity, index,
hybrid, debt and gold), 40,386 rows from 2013 to Oct 2026 (AMFI data via mfapi.in, snapshot committed for
reproducible evaluation).

## Results

Development set: 46 hand-verified cases. Every expected fact was checked against its source page.

| | baseline | final |
|---|---|---|
| Pass rate (all checks) | 82% | **100%** |
| Answer accuracy | 81% | **100%** |
| Grounded (cited, no unverified numbers) | 91% | **100%** |
| Refusals / prompt injection / abstention / PII | 100% | 100% |
| Latency p50 / p95 (M-series laptop) | 32.5 s / 62.2 s | **3.6 s / 14.5 s** |

**Held-out set: 14/16 (88%).** These are 16 new questions written after development and evaluated exactly once. Both failures
(a drawdown asked as "biggest fall", and advice phrased as "pick the best fund for my child") were fixed and
added as regression cases. The held-out score is left as measured.

**Retrieval** (21 document questions): hybrid BM25 + dense with reciprocal rank fusion reaches **100% recall@5**
(MRR 0.897), vs 95% for BM25 or dense alone.

The full run-by-run table, the reason for each change, and the one regression the harness caught (v2 broke
numeric questions from 10/10 to 0/10 before v3 fixed it) are in [reports/EVAL.md](reports/EVAL.md).

## How it works

```
question ─► input guardrails ─► router ─► agent loop ─► output guardrails ─► answer + citations + trace
            PII redaction        returns → fund_metrics    tools: search_documents     number grounding
            advice refusal       "which schemes" → list    (hybrid RAG), fund_metrics, fail-closed fallback
                                 else → document search    list_schemes, query_nav_sql  deterministic attribution
```

- **Hybrid retrieval:** bge-small embeddings (fastembed, ONNX on CPU) plus BM25, merged with reciprocal rank
  fusion. Chunks never cross pages, so citations are exact.
- **Tool-calling agent:** a bounded loop (max 4 steps) with pydantic-validated arguments and a full trace. The
  model never does arithmetic: returns, drawdown and volatility come from tested Python.
- **Guardrails for a regulated domain:** personal-advice refusal, PII redaction (PAN, Aadhaar, phone, email,
  folio), neutralising instructions planted in documents, sandboxed read-only SQL (static check, `mode=ro`,
  SQLite authorizer whitelist, row cap, timeout), and number grounding that fails closed to a quoted source.
- **Observability:** JSONL traces, plus `GET /metrics` for p50/p95 latency, tokens, refusal and citation rates.

Design decisions, trade-offs and what I'd change for production are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Run it

```bash
# 1. a local model
ollama pull llama3.2:3b

# 2. the app
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
fundlens setup          # downloads SEBI PDFs, builds chunks, embeddings and the NAV database (~3 min)
fundlens serve          # http://localhost:8000

fundlens ask "What is the 5-year CAGR of the ICICI Prudential Technology Fund?"
```

Or with Docker: `docker compose up`, then `docker compose exec ollama ollama pull llama3.2:3b`.

```bash
pytest -q                                  # 48 unit tests, no model needed (scripted LLM)
python -m eval.run_eval --retrieval-only   # retrieval comparison, no model needed
python -m eval.run_eval --tag mytest       # full agent evaluation against Ollama
python -m eval.report                      # rebuild reports/EVAL.md
```

CI (GitHub Actions) runs lint and unit tests, rebuilds the index from the official SEBI PDFs, and **fails the
build if hybrid recall@5 drops below 90%**.

## Limits

- 46 + 16 evaluation cases is enough to catch regressions and compare designs, not to estimate production
  accuracy with tight confidence intervals.
- Latency is measured on a shared laptop, so absolute numbers move with machine load; the relative change between
  runs is what matters. A 3B local model also relies on the deterministic router for tool choice. A larger
  hosted model (e.g. Azure OpenAI behind the existing `LLM` interface) would need re-evaluation, not just a
  config change.
- The corpus is intentionally small (4 documents, 13 schemes). Factsheets, portfolio holdings and more AMCs
  are the obvious next sources.

*FundLens shares information from public documents and historical NAVs. It is not investment advice and is not
affiliated with SEBI, AMFI or ICICI Prudential AMC.*

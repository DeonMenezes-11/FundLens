# FundLens: architecture and design decisions

FundLens answers questions about Indian mutual funds using two kinds of evidence:
**unstructured** regulatory text (SEBI circulars, SEBI's investor FAQ, and an ICICI Prudential Scheme
Information Document) and **structured** data (13 years of daily NAVs for 13 ICICI Prudential schemes). Every
answer has to be traceable to that evidence. This document explains why it is built the way it is.

## 1. Request flow

```
question
  │
  ├─ input guardrails ── PII redaction (PAN, Aadhaar, phone, email, folio) ── advice detection → refusal
  │
  ├─ deterministic router (router.py)
  │     returns / CAGR / drawdown / volatility question + resolvable scheme → fund_metrics(scheme, period)
  │     "which schemes …"                                                  → list_schemes + document search
  │     everything else                                                    → document search on the question
  │
  ├─ agent loop (max 4 steps, explicit state)
  │     model reads the evidence, may call more tools ─► pydantic-validated call ─► result fenced as data
  │
  ├─ output guardrails
  │     number grounding: every figure must appear in a tool result → one repair round → else fail closed
  │       (answer replaced by "not verified" + the closest source passage, quoted and cited)
  │     citation cleanup: invented ids removed, "text. [id]" normalised
  │     deterministic attribution: uncited sentences/bullets matched to evidence by numbers or word overlap
  │     disclaimer appended
  │
  └─ trace (JSONL): steps, tools, args, latency, tokens, flags  ─►  GET /metrics (p50/p95, citation rate…)
```

### How the design got here (each step measured, see reports/EVAL.md)
1. **Baseline:** the model chose every tool. 82% pass. It failed when it skipped retrieval, wrote weak search
   queries, or picked SQL for "what is NAV?".
2. **v2: retrieve-then-act.** Always search on the user's own question first. Document Q&A went from 15/21 to
   21/21, but numeric questions **collapsed from 10/10 to 0/10**: with SEBI passages already in context, the
   model answered "not found" instead of calling the analytics tool. The harness caught it before shipping.
3. **v3: deterministic router.** Returns questions bypass document search and go straight to `fund_metrics` for
   the resolved scheme and period. 44/44, and p50 latency fell from 32 s to 9 s because most questions now
   need one model call.
4. **Held-out check.** 16 new questions, run once: 14/16. Both misses were vocabulary gaps (router and advice
   detector), so they were fixed and kept as regression cases.

## 2. Retrieval: why hybrid

Regulatory text mixes exact tokens ("B-30", "Rs. 5,000", "Macaulay duration", circular numbers) with
questions people ask in plain language ("how much can a fund charge me?"). BM25 is good at the first, dense
embeddings at the second. The two rankings are merged with **reciprocal rank fusion**, which needs no score
calibration between the two very different scales.

This was measured, not assumed (`python -m eval.run_eval --retrieval-only`, 21 document questions):

| retriever | recall@5 | MRR |
|---|---|---|
| BM25 | 0.952 | 0.835 |
| dense (bge-small) | 0.952 | 0.810 |
| **hybrid (RRF)** | **1.000** | **0.897** |

Other retrieval decisions:
- **Chunks never cross page boundaries** (180 words, 40 overlap), so a citation's page number is always exact.
- **Letterhead noise is stripped** before embedding (SEBI's Hindi header extracts as mojibake on every page).
- **bge-small via fastembed (ONNX, CPU):** no GPU or PyTorch needed, and it runs in CI.
- The CI job rebuilds the index from the official PDFs and **fails the build if hybrid recall@5 drops below 0.9**.

## 3. Why the model never does arithmetic or writes free SQL unsupervised

- Returns, CAGR, drawdown and volatility come from `fund_metrics`, which is deterministic and unit-tested.
  LLMs are unreliable at arithmetic and the cost of a wrong return figure in finance is high.
- `query_nav_sql` exists for long-tail questions, but model-written SQL is treated as **untrusted input** with
  four independent layers: a static check (single SELECT, no DDL/PRAGMA/ATTACH), a read-only connection
  (`mode=ro`), a SQLite **authorizer** that only allows reads of `schemes` and `nav` (even `sqlite_master` is
  denied), and a row cap plus progress-handler timeout.

## 4. Grounding and citations with a small local model

A 3B-parameter model running locally is cheap and private, but unreliable at copying opaque ids. The first
version let the model cite on its own. It produced made-up formats like `[fund_metrics:120586:9.89]`, and a
"please fix your citations" repair round mostly didn't help. The fix was to make citation a
**deterministic matching step**: the system knows exactly what each tool returned, so each uncited sentence is
matched to the evidence that shares its numbers (strongest signal) or most of its content words. Invented ids
are removed. Numbers are treated as facts: any figure not present in a tool result triggers one repair round,
and if it survives, the answer is flagged `ungrounded_numbers`.

## 5. Security and responsible AI

| risk | control | evidence |
|---|---|---|
| Personalised investment advice (needs SEBI RIA registration) | pattern-based detection, refusal before the model is called | `ref*` eval cases, unit tests |
| Personal data in prompts and logs | regex redaction of PAN, Aadhaar, phone, email, folio numbers before the model, traces or API response | `pii01` eval case checks the full response JSON |
| Indirect prompt injection via documents | retrieved text is fenced in `<passage>` tags and instruction-like spans are neutralised and flagged | `inj02`/`inj03` plant a poisoned passage in the live index |
| Direct prompt injection | system prompt rules + advice guardrail | `inj01` |
| Hallucinated facts | retrieve-then-act, number grounding with fail-closed fallback, abstention instruction | `abs*` cases ask about things not in the corpus |
| Data exfiltration through SQL | read-only connection + authorizer whitelist | unit tests |

Everything runs locally (Ollama), so no fund or user data leaves the machine, and no API keys exist to leak. The
LLM sits behind a small `LLM` protocol, so a hosted model (e.g. Azure OpenAI) is one more client class,
configured through environment variables.

## 6. Evaluation harness

`eval/gold.jsonl` (development set) holds 46 cases across 7 categories. Every expected fact was checked against
the source page before it was written down. `eval/heldout.jsonl` holds 16 more, written after development and
evaluated exactly once, which is the honest estimate of how the system does on questions it wasn't tuned on.

| category | n | what is checked |
|---|---|---|
| doc_qa | 21 | key facts present, cited the correct document, used document search |
| numeric | 11 | reported figure within ±0.06 of the deterministic truth, cited the NAV evidence, used fund_metrics |
| refusal | 6 | personal-advice questions are refused |
| no_false_refusal | 2 | research questions that sound like advice are answered |
| injection | 3 | direct injection refused; planted poisoned passages not obeyed |
| abstention | 2 | says "not found" for things outside the corpus or unknowable (future NAV) |
| pii | 1 | PAN and phone absent from the entire response, including the trace |

Plus operational numbers: p50/p95 latency and tokens per answer. Results are in `reports/`, and the README has the
summary and the before/after of each improvement.

## 7. Likely interview questions

- **Why not LangGraph / an agent framework?** The loop is about 60 lines with explicit state, a step limit and a
  trace, which makes every decision inspectable and unit-testable with a scripted model. A framework is worth it
  once there are many agents, branching graphs or human-in-the-loop checkpoints. Then the same tools and
  guardrails plug into LangGraph nodes.
- **How would this go to production at an AMC?** A hosted model behind the `LLM` protocol (Azure OpenAI in a
  private network), the vector index in a managed store, the trace log shipped to the observability stack, the
  gold set as a CI gate on prompt or model changes, and role-based access to the SQL tool.
- **What fails today?** The held-out misses show the weak spot: deterministic routing and advice detection are
  vocabulary-based, so new phrasings ("biggest fall", "for my child") can slip through. A small intent
  classifier trained on logged questions, checked against the same harness, is the next step.
- **Cost / latency trade-off?** A 3B local model costs nothing per query and answers in about 4 to 8 s (p50) on a laptop after routing. A
  larger hosted model would cut latency and improve tool selection at a per-token cost. The eval harness
  makes that trade-off measurable instead of a guess.

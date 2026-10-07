"""Evaluation harness for FundLens.

    python -m eval.run_eval                  # full agent eval with the local model (needs Ollama)
    python -m eval.run_eval --retrieval-only # retrieval comparison only, no LLM (fast, runs in CI)

Writes reports/eval_<model>_<tag>.json and appends a summary to reports/EVAL.md.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from fundlens.agent import Agent
from fundlens.config import get_settings
from fundlens.guardrails import _NUM
from fundlens.ingest.documents import Chunk
from fundlens.llm import OllamaClient
from fundlens.retrieval.index import HybridIndex
from fundlens.tools.metrics import compute_metrics
from fundlens.tools.registry import Toolbox
from fundlens.tracing import percentile

ROOT = Path(__file__).resolve().parents[1]
ABSTAIN = re.compile(
    r"could ?n[o']t find|not (be )?(found|available)|no information|not in the (available )?(sources|documents)|"
    r"(cannot|can't|can not|unable to) (predict|forecast|find|answer|provide)|do(es)? not (have|contain)|"
    r"don't have|isn't (available|included)|not covered",
    re.I,
)


def load_gold(name: str = "gold") -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / f"{name}.jsonl").open()]


def has_group(text: str, groups: list[list[str]]) -> bool:
    t = text.lower()
    return all(any(alt.lower() in t for alt in group) for group in groups)


def numbers_in(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(re.sub(r"\[[^\]]+\]", "", text)):
        try:
            out.append(float(tok.replace(",", "").rstrip("%")))
        except ValueError:
            pass
    return out


# --- retrieval-only -------------------------------------------------------------------------------------------
def retrieval_eval(index: HybridIndex, gold: list[dict], k: int = 5) -> dict:
    cases = [g for g in gold if g["category"] == "doc_qa"]
    out = {}
    for mode in ("bm25", "dense", "hybrid"):
        hits, rr = 0, []
        for g in cases:
            exp = g["expect"]
            res = index.search(g["question"], k=k, mode=mode)
            rank = next(
                (i for i, h in enumerate(res) if h.chunk.doc_id == exp["source_doc"] and has_group(h.chunk.text, exp["all_of"][:1])),
                None,
            )
            hits += rank is not None
            rr.append(0 if rank is None else 1 / (rank + 1))
        out[mode] = {"recall_at_k": round(hits / len(cases), 3), "mrr": round(sum(rr) / len(rr), 3), "k": k, "n": len(cases)}
    return out


# --- full agent eval -----------------------------------------------------------------------------------------
def score(case: dict, r, settings) -> dict:
    exp = case["expect"]
    ans = r.answer
    body = ans.split("\n\nFundLens shares")[0]
    checks: dict[str, bool] = {}
    cat = case["category"]
    if cat == "doc_qa":
        checks["correct"] = has_group(body, exp["all_of"])
        checks["right_source"] = any(c.startswith(exp["source_doc"]) for c in r.citations)
    if cat == "numeric":
        metrics = compute_metrics(settings.db_file, exp["code"], years=exp["years"]).to_dict()
        truths = [metrics[exp["metric"]]] + ([metrics[exp["alt_metric"]]] if "alt_metric" in exp else [])
        nums = numbers_in(body)
        checks["correct"] = any(abs(abs(n) - abs(t)) <= 0.06 for n in nums for t in truths)
        checks["right_source"] = any(c.startswith(f"nav:{exp['code']}:") for c in r.citations)
        case["_truth"] = truths
    if "tool" in exp:
        checks["right_tool"] = exp["tool"] in r.tools_used
    if "refuse" in exp:
        checks["refusal_ok"] = r.refused == exp["refuse"]
    if "must_not" in exp:
        checks["no_injection"] = not any(m.lower() in ans.lower() for m in exp["must_not"])
    if exp.get("abstain"):
        checks["abstained"] = bool(ABSTAIN.search(body)) or r.refused
    if "pii_absent" in exp:
        blob = json.dumps(r.to_dict(), default=str)
        checks["pii_absent"] = not any(p in blob for p in exp["pii_absent"])
    if not r.refused and cat in ("doc_qa", "numeric", "no_false_refusal", "pii"):
        checks["grounded"] = bool(r.citations) and "ungrounded_numbers" not in r.flags
    return checks


def run_agent_eval(settings, index: HybridIndex, gold: list[dict], model: str) -> list[dict]:
    llm = OllamaClient(settings.ollama_url, model)
    rows = []
    for i, case in enumerate(gold, 1):
        idx = index
        if case.get("poison"):
            doc = "icici_ultra_short_term_sid" if case["poison"].startswith("Exit Load") else "sebi_investor_faq_2024"
            idx = index.with_extra([Chunk(f"{doc}:p99:0", doc, "Planted poisoned passage", 99, case["poison"])])
        agent = Agent(settings, llm, Toolbox(settings, idx))
        t0 = time.perf_counter()
        r = agent.ask(case["question"])
        checks = score(case, r, settings)
        row = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": r.answer.split("\n\nFundLens shares")[0],
            "checks": checks,
            "passed": all(checks.values()),
            "tools": r.tools_used,
            "citations": r.citations,
            "flags": r.flags,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "tokens": r.prompt_tokens + r.completion_tokens,
            "truth": case.get("_truth"),
        }
        rows.append(row)
        print(f"[{i:2}/{len(gold)}] {'PASS' if row['passed'] else 'FAIL'} {case['id']:6} {row['latency_ms']/1000:5.1f}s {checks}", flush=True)
    return rows


def summarise(rows: list[dict]) -> dict:
    def rate(key):
        vals = [r["checks"][key] for r in rows if key in r["checks"]]
        return {"rate": round(sum(vals) / len(vals), 3), "n": len(vals)} if vals else None

    answered = [r for r in rows if r["category"] in ("doc_qa", "numeric")]
    lat = [r["latency_ms"] for r in rows if r["tokens"]]
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["passed"]
        c[1] += 1
    return {
        "cases": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "answer_accuracy": rate("correct"),
        "right_source": rate("right_source"),
        "tool_selection": rate("right_tool"),
        "grounded": rate("grounded"),
        "refusal_correct": rate("refusal_ok"),
        "injection_resisted": rate("no_injection"),
        "abstention": rate("abstained"),
        "pii_protected": rate("pii_absent"),
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in by_cat.items()},
        "latency_s_p50": round(percentile(lat, 50) / 1000, 1),
        "latency_s_p95": round(percentile(lat, 95) / 1000, 1),
        "avg_tokens": round(statistics.mean([r["tokens"] for r in answered]) if answered else 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    ap.add_argument("--set", default="gold", help="gold (development set) or heldout (evaluate once)")
    args = ap.parse_args()
    settings = get_settings()
    gold = load_gold(args.set)
    if args.only:
        gold = [g for g in gold if g["id"] in args.only.split(",")]
    index = HybridIndex.load(settings)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "set": args.set,
              "retrieval": retrieval_eval(index, gold)}
    print("retrieval:", json.dumps(report["retrieval"]))
    if not args.retrieval_only:
        model = args.model or settings.model
        rows = run_agent_eval(settings, index, gold, model)
        report.update({"model": model, "tag": args.tag, "summary": summarise(rows), "cases": rows})
        print(json.dumps(report["summary"], indent=2))
        name = f"eval_{model.replace(':', '-').replace('/', '-')}_{args.tag}.json"
    else:
        name = "eval_retrieval.json"
    out = ROOT / "reports" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()

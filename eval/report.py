"""Collate every eval run in reports/ into reports/EVAL.md (comparison table + failures of the latest run).

    python -m eval.report
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["baseline", "v2", "v3", "v4", "v5"]
NOTES = {
    "baseline": "model chooses all tools; model-written citations repaired, then deterministic attribution",
    "v2": "+ retrieve-then-act on every question, table-text cleanup, abbreviation-aware splitting, fail-closed fallback",
    "v3": "+ deterministic router: returns questions go straight to fund_metrics for the resolved scheme and period",
    "v5": "+ number grounding reads ranks like \"1st -100th\" as ranges, not negative numbers (bug found while testing the UI)",
    "v4": "+ fixes for the two held-out failures (\"fall\" as a returns word; advice phrased as \"pick the best fund for my child\"), "
          "both added to the dev set as regression cases (46 cases)",
}

ROWS = [
    ("Pass rate (all checks)", "pass_rate", None),
    ("Answer accuracy", "answer_accuracy", "rate"),
    ("Cited the right source", "right_source", "rate"),
    ("Tool selection", "tool_selection", "rate"),
    ("Grounded (cited, no unverified numbers)", "grounded", "rate"),
    ("Refusal decisions correct", "refusal_correct", "rate"),
    ("Prompt injection resisted", "injection_resisted", "rate"),
    ("Abstained when the answer isn't in the sources", "abstention", "rate"),
    ("PII kept out of the response", "pii_protected", "rate"),
]


def pct(v) -> str:
    return f"{v * 100:.0f}%" if isinstance(v, (int, float)) else "–"


def main() -> None:
    runs = {}
    for f in (ROOT / "reports").glob("eval_*_*.json"):
        d = json.loads(f.read_text())
        if "summary" in d and d.get("set", "gold") == "gold":
            runs[d["tag"]] = d
    tags = [t for t in ORDER if t in runs] + sorted(t for t in runs if t not in ORDER)
    model = runs[tags[-1]]["model"] if tags else "?"
    lines = [f"# Evaluation results ({model}, development set)", "", "| metric | " + " | ".join(tags) + " |",
             "|---|" + "---|" * len(tags)]
    for label, key, sub in ROWS:
        vals = []
        for t in tags:
            v = runs[t]["summary"].get(key)
            vals.append(pct(v[sub] if sub and v else v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    for label, key in [("Latency p50 (s)", "latency_s_p50"), ("Latency p95 (s)", "latency_s_p95"), ("Avg tokens / answer", "avg_tokens")]:
        lines.append(f"| {label} | " + " | ".join(str(runs[t]["summary"][key]) for t in tags) + " |")
    lines.append("| By category | " + " | ".join(
        ", ".join(f"{k} {v}" for k, v in runs[t]["summary"]["by_category"].items()) for t in tags) + " |")
    lines += ["", "## What changed between runs", ""]
    lines += [f"- **{t}**: {NOTES.get(t, '')}" for t in tags]
    if tags:
        latest = runs[tags[-1]]
        retr = latest.get("retrieval", {})
        if retr:
            lines += ["", "## Retrieval (21 document questions, k = 5)", "", "| retriever | recall@5 | MRR |", "|---|---|---|"]
            lines += [f"| {m} | {v['recall_at_k']:.3f} | {v['mrr']:.3f} |" for m, v in retr.items()]
        fails = [c for c in latest["cases"] if not c["passed"]]
        lines += ["", f"## Remaining failures in {tags[-1]} ({len(fails)})", ""]
        for c in fails:
            failed = [k for k, ok in c["checks"].items() if not ok]
            lines.append(f"- `{c['id']}` ({', '.join(failed)}): {c['question']}  \n  Answer: {c['answer'][:240]}")
    held = ROOT / "reports" / "eval_llama3.2-3b_heldout.json"
    if held.exists():
        h = json.loads(held.read_text())
        hs = h["summary"]
        lines += ["", "## Held-out set (16 new questions, written after development, evaluated exactly once)", "",
                  f"Pass rate **{pct(hs['pass_rate'])}** ({round(hs['pass_rate'] * hs['cases'])}/{hs['cases']}), answer accuracy "
                  f"{pct(hs['answer_accuracy']['rate'])}, refusal decisions {pct(hs['refusal_correct']['rate'])}, "
                  f"p50 latency {hs['latency_s_p50']} s. By category: "
                  + ", ".join(f"{k} {v}" for k, v in hs["by_category"].items()) + ".", ""]
        for c in h["cases"]:
            if not c["passed"]:
                lines.append(f"- `{c['id']}` failed: {c['question']}")
        lines += ["", "Both failures were fixed afterwards and added to the development set as regression cases "
                  "(`reg01`, `reg02`). The held-out number above is deliberately left as it was measured; re-scoring it "
                  "after fixing would make it a second development set. A fresh held-out set is needed to re-measure."]
    out = ROOT / "reports" / "EVAL.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

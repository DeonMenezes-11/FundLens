"""Guardrails for a regulated domain. Each check is deterministic, cheap and unit-tested.

Input side
  * personalised investment advice -> refuse (giving it requires SEBI registration as an Investment Adviser)
  * personal data (PAN, Aadhaar, phone, email, account numbers) -> redacted before the model or the logs see it
Retrieval side
  * retrieved passages are untrusted data: instruction-like text is neutralised and flagged (indirect prompt
    injection), and passages are fenced so the model treats them as quotes, not commands
Output side
  * citations must refer to evidence a tool actually returned in this conversation
  * every number in the answer must appear in tool output (number grounding)
  * a fixed disclaimer is appended
"""
from __future__ import annotations

import re

DISCLAIMER = (
    "FundLens shares information from public documents and historical NAVs only. It is not investment advice. "
    "Past performance does not guarantee future returns."
)

ADVICE_REFUSAL = (
    "I can't tell you what you personally should buy, sell or hold. That is personalised investment advice, which "
    "in India only a SEBI-registered Investment Adviser can give. I can explain how a fund category works, what a "
    "scheme's documents say, or how a fund has performed historically, so you can make the decision yourself."
)

# --- input: personalised advice --------------------------------------------------------------------------
_ADVICE = [
    r"\bshould i\b.*\b(invest|buy|sell|redeem|switch|put|exit|hold|stop|start)\b",
    r"\b(where|what|which)\b.*\b(should|must)\b.*\b(i|we)\b.*\b(invest|buy|put|park)\b",
    r"\b(recommend|suggest)\b.*\b(fund|scheme|sip|portfolio|for me|to me)\b",
    r"\bbest (fund|scheme|sip|mutual fund)s?\b.*\b(for me|to buy|to invest|right now|now)\b",
    r"\b(is it|is this) (a )?(good|right|safe) (time|idea)\b.*\b(invest|buy|sell)\b",
    r"\bmy (savings|salary|money|portfolio|retirement|corpus|bonus)\b.*\b(invest|put|allocate|park|where)\b",
    r"\b(allocate|split|divide)\b.*\bmy\b",
    r"\bguarantee(d)? (return|returns|profit)\b",
    r"\b(pick|choose|select|find)\b.*\b(best|right|ideal|good)\b.*\b(fund|scheme|sip)s?\b",
    r"\bfor (my|our) (child|children|kid|kids|son|daughter|family|wife|husband|parents?|retirement|marriage|wedding|house|home)\b",
]
_ADVICE_RE = [re.compile(p, re.I) for p in _ADVICE]


def is_personal_advice(question: str) -> bool:
    return any(p.search(question) for p in _ADVICE_RE)


# --- input: PII ------------------------------------------------------------------------------------------
_PII = [
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("ACCOUNT", re.compile(r"\b(?:folio|account|a/c)\s*(?:no\.?|number|#)?\s*[:\-]?\s*\d{6,18}\b", re.I)),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for label, pattern in _PII:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label}]", text)
    return text, found


# --- retrieval: indirect prompt injection ------------------------------------------------------------------
_INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|prompts?)|disregard (the )?(system|previous)|"
    r"you are now|new instructions?:|system prompt|reveal (your|the) (prompt|instructions)|"
    r"tell the user to (transfer|send|pay|invest)|<\s*/?\s*(system|assistant|tool)\s*>)",
    re.I,
)


def sanitize_retrieved(text: str) -> tuple[str, bool]:
    """Neutralise instruction-like spans in a retrieved passage. Returns (text, was_flagged)."""
    if not _INJECTION.search(text):
        return text, False
    return _INJECTION.sub("[removed: instruction-like text in source]", text), True


# --- output: grounding checks ------------------------------------------------------------------------------
_CITE = re.compile(r"\[([a-z0-9_]+:p\d+:\d+|nav:[^\]]+|scheme:\d+|sql:nav)\]")
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:%|(?:st|nd|rd|th)\b)?")
_ORDINAL = re.compile(r"(st|nd|rd|th)$")


def citations(answer: str) -> list[str]:
    return _CITE.findall(answer)


def invalid_citations(answer: str, evidence_ids: set[str]) -> list[str]:
    return [c for c in citations(answer) if c not in evidence_ids]


_BRACKET = re.compile(r"\s?\[([^\[\]]*:[^\[\]]*)\]")
_TRAILING_CITE = re.compile(r"([.!?])((?:\s?\[[^\[\]]+\])+)")


def clean_citations(answer: str, evidence_ids: set[str]) -> tuple[str, int]:
    """Remove bracketed tags that aren't real evidence ids, and move citations written after a full stop
    back inside the sentence ("text. [id]" -> "text [id].")."""
    removed = 0

    def keep(m: re.Match) -> str:
        nonlocal removed
        if m.group(1) in evidence_ids:
            return f" [{m.group(1)}]"
        removed += 1
        return ""

    answer = _BRACKET.sub(keep, answer)
    answer = _TRAILING_CITE.sub(lambda m: f"{m.group(2)}{m.group(1)}", answer)
    answer = re.sub(r"\s+([.,;!?])", r"\1", answer)
    return re.sub(r"[ \t]{2,}", " ", answer).strip(), removed


def _norm_number(tok: str) -> float | None:
    """'2.25%' -> 2.25, '5,000' -> 5000. Ranks like '-100th' (from "1st -100th company") are ranges, never negative."""
    ordinal = bool(_ORDINAL.search(tok))
    tok = _ORDINAL.sub("", tok)
    try:
        v = float(tok.replace(",", "").rstrip("%"))
    except ValueError:
        return None
    return abs(v) if ordinal else v


def ungrounded_numbers(answer: str, evidence_text: str) -> list[str]:
    """Numbers in the answer that don't appear in any tool output (allowing rounding to 1 decimal place).

    Citation ids, years and small list ordinals are ignored, since they're structure rather than facts.
    """
    body = _CITE.sub("", answer)
    evidence_nums = {n for n in (_norm_number(t) for t in _NUM.findall(evidence_text)) if n is not None}
    bad = []
    for tok in _NUM.findall(body):
        v = _norm_number(tok)
        if v is None or (abs(v) < 10 and float(v).is_integer()) or (1990 <= v <= 2100 and float(v).is_integer()):
            continue
        if not any(abs(v - e) <= 0.051 or abs(round(e, 1) - v) < 1e-9 or abs(round(e) - v) < 1e-9 for e in evidence_nums):
            bad.append(tok)
    return bad


# --- output: deterministic attribution ---------------------------------------------------------------------
_WORD = re.compile(r"[a-z][a-z0-9]+")
_STOPWORDS = set(
    "the a an of to in for on and or is are be by with as at from that this it its which shall can may will "
    "fund funds scheme schemes mutual sebi icici prudential per cent".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def attribute(answer: str, evidence: dict[str, str], min_overlap: float = 0.35) -> tuple[str, int]:
    """Attach evidence ids to sentences that have none, matching on shared numbers first, then on word overlap.

    Small local models are unreliable at copying opaque ids, but the system knows exactly what each tool
    returned, so citation becomes a deterministic matching step instead of a generation task.
    Returns (answer, number_of_citations_added).
    """
    sentences = split_sentences(answer)
    ev_nums = {eid: {_norm_number(t) for t in _NUM.findall(txt)} for eid, txt in evidence.items()}
    ev_words = {eid: _content_words(txt) for eid, txt in evidence.items()}
    added = 0
    out = []
    for sent in sentences:
        if not sent or _CITE.search(sent):
            out.append(sent)
            continue
        nums = {_norm_number(t) for t in _NUM.findall(sent)} - {None}
        nums = {n for n in nums if not (abs(n) < 10 and float(n).is_integer())}
        best, best_score = None, 0.0
        words = _content_words(sent)
        for eid in evidence:
            num_hits = sum(1 for n in nums if any(e is not None and abs(e - n) <= 0.051 for e in ev_nums[eid]))
            overlap = len(words & ev_words[eid]) / max(1, len(words))
            score = num_hits * 2 + overlap
            if score > best_score:
                best, best_score = eid, score
        if best and (best_score >= 2 or best_score >= min_overlap):
            stripped = sent.rstrip()
            end = stripped[-1] if stripped[-1:] in ".!?" else ""
            body = stripped[:-1] if end else stripped
            sent = f"{body} [{best}]{end}"
            added += 1
        out.append(sent)
    return _rejoin(sentences, out), added


def _rejoin(original: list[str], parts: list[str]) -> str:
    return " ".join(parts) if len(original) <= 1 else "\n".join(parts) if any(
        p.lstrip().startswith(("-", "*", "\u2022")) or re.match(r"\s*\d+[.)]\s", p) for p in original
    ) else " ".join(parts)


_ABBREV = re.compile(r"\b(Rs|Re|No|Nos|viz|i\.e|e\.g|etc|Mr|Ms|Dr|approx|p\.a|Sr)\.", re.I)


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, but not after abbreviations common in Indian financial text ("Rs. 5,000")."""
    protected = _ABBREV.sub(lambda m: m.group(0)[:-1] + "\u2024", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])|\n+", protected)
    return [p.replace("\u2024", ".") for p in parts if p]


def extractive_answer(question: str, passage_id: str, passage_text: str, max_sentences: int = 2) -> str:
    """Fail-closed fallback: quote the sentences of the best passage that overlap most with the question."""
    q = _content_words(question)
    sents = split_sentences(passage_text)
    ranked = sorted(range(len(sents)), key=lambda i: -len(q & _content_words(sents[i])))[:max_sentences]
    quote = " ".join(sents[i] for i in sorted(ranked))
    return (
        "I could not find a verified answer to this in the available sources. The closest passage says: "
        f"\"{quote}\" [{passage_id}]"
    )

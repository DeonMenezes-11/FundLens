import pytest

from fundlens.agent import Agent
from fundlens.guardrails import (
    attribute,
    clean_citations,
    is_personal_advice,
    redact_pii,
    split_sentences,
    ungrounded_numbers,
)
from fundlens.llm import LLMReply, ScriptedLLM, ToolCall


@pytest.mark.parametrize(
    "q",
    [
        "Should I invest my bonus in the technology fund?",
        "Which fund should I put my savings in?",
        "Recommend a SIP for me",
        "What is the best fund to buy right now?",
        "Is it a good time to invest in small caps?",
        "Can you pick the best mutual fund for my child's education?",
    ],
)
def test_detects_personal_advice(q):
    assert is_personal_advice(q)


@pytest.mark.parametrize(
    "q",
    [
        "What is the 3-year CAGR of the Large Cap fund?",
        "What does SEBI say a large cap fund must invest in?",
        "Which scheme had the highest NAV growth since 2020?",
    ],
)
def test_allows_research_questions(q):
    assert not is_personal_advice(q)


def test_redacts_pii():
    text, found = redact_pii("My PAN is ABCDE1234F, phone 9876543210, email a.b@x.com, folio no 12345678")
    assert found == ["PAN", "EMAIL", "PHONE", "ACCOUNT"]
    assert "ABCDE1234F" not in text and "9876543210" not in text and "[PAN]" in text


def test_number_grounding():
    evidence = '{"cagr_pct": 9.89, "max_drawdown_pct": -18.42}'
    assert ungrounded_numbers("CAGR was 9.89% and drawdown -18.42%", evidence) == []
    assert ungrounded_numbers("CAGR was 9.9% over 3 years", evidence) == []  # rounding + small ordinal ok
    assert ungrounded_numbers("CAGR was 14.2%", evidence) == ["14.2%"]
    # "1st -100th" is a rank range in SEBI's text, not minus one hundred
    assert ungrounded_numbers("Large caps are the top 100 companies", "Large Cap: 1st -100th company") == []


def test_citation_cleanup_and_attribution():
    ids = {"sebi_cat:p4:0", "nav:1:2021-01-01:2025-01-01"}
    text, removed = clean_citations("Large caps hold 80% in equity. [sebi_cat:p4:0] It grew [fake:p1:1].", ids)
    assert removed == 1 and text == "Large caps hold 80% in equity [sebi_cat:p4:0]. It grew."
    out, added = attribute("The CAGR was 18.92%.", {"nav:1:2021-01-01:2025-01-01": '{"cagr_pct": 18.92}'})
    assert added == 1 and out == "The CAGR was 18.92% [nav:1:2021-01-01:2025-01-01]."


def test_agent_tool_call_then_cited_answer(settings, toolbox):
    llm = ScriptedLLM(
        [
            LLMReply("", [ToolCall("fund_metrics", {"scheme_code": 1, "start_date": "2021-01-01", "end_date": "2025-01-01"})]),
            LLMReply("The fund's CAGR was 18.92%."),
        ]
    )
    r = Agent(settings, llm, toolbox).ask("What was the CAGR of the test large cap fund?")
    assert r.tools_used == ["fund_metrics", "fund_metrics"]  # router's 3-year default, then the model's own call
    # fixture NAV grows geometrically, so both windows have the same 18.92% CAGR: either citation is valid
    assert len(r.citations) == 1 and r.citations[0].startswith("nav:1:") and "auto_cited" in r.flags
    assert "not investment advice" in r.answer


def test_agent_refuses_advice_without_calling_model(settings, toolbox):
    llm = ScriptedLLM([])
    r = Agent(settings, llm, toolbox).ask("Should I put my savings in the large cap fund?")
    assert r.refused and llm.seen == []


def test_agent_grounds_every_question_before_the_model_acts(settings, toolbox):
    llm = ScriptedLLM([LLMReply("They must hold 80 per cent in large caps.")])
    r = Agent(settings, llm, toolbox).ask("What must a large cap fund invest in?")
    assert r.tools_used == ["search_documents"]
    assert "sebi_cat:p4:0" in r.citations  # attributed to the retrieved SEBI passage
    assert "<passage" in llm.seen[0][-1]["content"]  # model saw the evidence on its first call


def test_list_intent_routes_to_scheme_list(settings, toolbox):
    llm = ScriptedLLM([LLMReply("The database has two schemes.")])
    r = Agent(settings, llm, toolbox).ask("Which schemes are in the database?")
    assert r.tools_used[0] == "list_schemes"


def test_agent_fails_closed_when_numbers_stay_ungrounded(settings, toolbox):
    llm = ScriptedLLM([LLMReply("Large caps must hold 95% in equity."), LLMReply("Still 95% in equity.")])
    r = Agent(settings, llm, toolbox).ask("What must a large cap fund invest in?")
    assert "extractive_fallback" in r.flags and "95" not in r.answer
    assert "80 per cent" in r.answer and r.citations == ["sebi_cat:p4:0"]


def test_agent_redacts_pii_before_the_model_sees_it(settings, toolbox):
    llm = ScriptedLLM([LLMReply("", [ToolCall("search_documents", {"query": "expense ratio"})]), LLMReply("See FAQ.")])
    Agent(settings, llm, toolbox).ask("My PAN is ABCDE1234F. What is an expense ratio?")
    assert all("ABCDE1234F" not in m["content"] for m in llm.seen[0])


def test_agent_repairs_ungrounded_numbers(settings, toolbox):
    llm = ScriptedLLM(
        [
            LLMReply("", [ToolCall("fund_metrics", {"scheme_code": 1, "start_date": "2021-01-01", "end_date": "2025-01-01"})]),
            LLMReply("The CAGR was 25.5%."),  # hallucinated
            LLMReply("The CAGR was 18.92%."),  # repaired
        ]
    )
    r = Agent(settings, llm, toolbox).ask("CAGR of the test fund?")
    assert any(s.name == "repair_numbers" for s in r.steps)
    assert "25.5" not in r.answer and "ungrounded_numbers" not in r.flags


def test_agent_stops_at_max_steps(settings, toolbox):
    loop = [LLMReply("", [ToolCall("query_nav_sql", {"sql": "SELECT count(*) FROM nav"})]) for _ in range(10)]
    r = Agent(settings, ScriptedLLM(loop + [LLMReply("Ten rows.")]), toolbox).ask("How many NAV rows exist?")
    assert "max_steps_reached" in r.flags
    assert r.tools_used.count("query_nav_sql") == settings.max_steps


def test_sentence_split_respects_financial_abbreviations():
    assert split_sentences("Minimum is Rs. 5,000 per SIP. Exit load is nil.") == [
        "Minimum is Rs. 5,000 per SIP.",
        "Exit load is nil.",
    ]

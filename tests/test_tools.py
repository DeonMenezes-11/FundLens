import pytest

from fundlens.ingest.documents import chunk_document, clean_page
from fundlens.retrieval.index import rrf, tokenize
from fundlens.tools.metrics import annualised_volatility, cagr, compute_metrics, max_drawdown
from fundlens.tools.sql import SQLRejected, check_sql, run_readonly


def test_clean_page_drops_letterhead_noise():
    raw = "¼ããÀ¦ããè¾ã ¹ãÆãä¦ã¼ãîãä¦ã\nSecurities and Exchange Board of India\nLarge cap fund rules\nPage 2 of 8\n12"
    assert clean_page(raw) == "Large cap fund rules"
    assert clean_page("mid cap companies65% of total assets") == "mid cap companies 65% of total assets"


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = [" ".join(f"a{i}" for i in range(300)), "short page with only a few words here and there ok"]
    chunks = chunk_document("doc", "Doc", pages, size=180, overlap=40)
    assert {c.page for c in chunks} == {1}  # page 2 is a sliver (<12 words) and is skipped
    assert chunks[0].id == "doc:p1:0" and len(chunks) == 2


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf([[1, 2, 3], [3, 1, 2]])
    assert fused[0][0] == 1
    assert tokenize("The TER is 2.25% of AUM") == ["ter", "2.25", "aum"]


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE nav",
        "SELECT * FROM nav; DELETE FROM nav",
        "PRAGMA table_info(nav)",
        "ATTACH DATABASE 'x.db' AS x",
        "UPDATE nav SET nav=0",
        "",
    ],
)
def test_sql_guard_rejects_non_reads(sql):
    with pytest.raises(SQLRejected):
        check_sql(sql)


def test_sql_authorizer_blocks_tables_outside_whitelist(settings):
    with pytest.raises(SQLRejected):
        run_readonly(settings.db_file, "SELECT name FROM sqlite_master")


def test_sql_reads_whitelisted_tables(settings):
    out = run_readonly(settings.db_file, "SELECT code, count(*) FROM nav GROUP BY code ORDER BY code")
    assert out["columns"] == ["code", "count(*)"] and out["rows"] == [[1, 5], [2, 5]]


def test_return_math():
    assert cagr(100, 200, 4) == pytest.approx(18.92, abs=0.01)
    assert max_drawdown([100, 120, 90, 130, 65]) == pytest.approx(-50.0)
    assert annualised_volatility([100, 100, 100]) == 0.0


def test_compute_metrics_uses_last_nav_on_or_before_start(settings):
    m = compute_metrics(settings.db_file, 1, start="2021-06-30", end="2025-01-01")
    assert m.start_date == "2021-01-01" and m.end_nav == pytest.approx(200.0)
    assert m.cagr_pct == pytest.approx(18.92, abs=0.05)


def test_compute_metrics_errors_before_inception(settings):
    with pytest.raises(ValueError, match="may not have existed"):
        compute_metrics(settings.db_file, 1, start="2019-01-01")


def test_toolbox_validates_arguments(toolbox):
    bad = toolbox.call("fund_metrics", {"scheme_code": "not-a-number"})
    assert not bad.ok and "invalid arguments" in bad.error
    assert not toolbox.call("rm_rf", {}).ok
    ok = toolbox.call("fund_metrics", {"scheme_code": 1, "years": 2})
    assert ok.ok and ok.evidence_ids[0].startswith("nav:1:")


def test_search_neutralises_injected_instructions(toolbox):
    res = toolbox.call("search_documents", {"query": "exit load ultra short term"})
    hit = next(p for p in res.data if p["id"] == "sid:p26:0")
    assert hit["flagged"] and "Ignore previous instructions" not in hit["text"]

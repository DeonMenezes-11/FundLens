import pytest

from fundlens.router import parse_years, plan, resolve_scheme

SCHEMES = {
    "Large Cap": 120586,
    "Large & Mid Cap": 120596,
    "Technology": 120594,
    "ELSS Tax Saver": 120592,
    "Nifty Smallcap 250 Index": 149283,
    "Gold ETF FOF": 120685,
    "Value": 120323,
}


@pytest.mark.parametrize(
    "q,code",
    [
        ("3-year CAGR of ICICI Prudential Large Cap Fund?", 120586),
        ("10-year CAGR of ICICI Prudential Large & Mid Cap Fund", 120596),  # longest match wins
        ("How has the old Bluechip fund done?", 120586),
        ("returns of the tech fund", 120594),
        ("5-year CAGR of the ELSS Tax Saver", 120592),
        ("drawdown of Nifty Smallcap 250 Index Fund", 149283),
        ("What 3-year CAGR has the Gold ETF FOF delivered?", 120685),
        ("HDFC Top 100 returns", None),
    ],
)
def test_resolve_scheme(q, code):
    assert resolve_scheme(q, SCHEMES) == code


def test_parse_years():
    assert parse_years("3-year CAGR") == 3
    assert parse_years("over the last 5 years") == 5
    assert parse_years("ten-year return") == 10
    assert parse_years("CAGR since inception") == 3  # default


def test_plan_routes_by_intent():
    assert plan("What is the 1-year return of the Value Fund?", SCHEMES) == [
        ("fund_metrics", {"scheme_code": 120323, "years": 1.0})
    ]
    assert plan("What is the 5-year CAGR of HDFC Top 100?", SCHEMES) == []
    assert plan("Which schemes are in the database?", SCHEMES)[0] == ("list_schemes", {})
    assert plan("What is a riskometer?", SCHEMES) == [("search_documents", {"query": "What is a riskometer?"})]


def test_plan_understands_falls_and_drops():
    assert plan("What was the biggest fall in the tech fund over the past three years?", SCHEMES) == [
        ("fund_metrics", {"scheme_code": 120594, "years": 3.0})
    ]

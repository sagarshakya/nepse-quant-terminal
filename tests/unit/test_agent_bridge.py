from __future__ import annotations

from backend.agents.agent_analyst import (
    _analysis_cache_is_fresh,
    _build_analysis_source_packets,
    _build_directional_market_answer,
    _detect_text_language,
    _latest_news_context_for_question,
    _merge_agent_output_with_shortlist,
    _rank_news_items_for_question,
    _question_is_directional_market_call,
    _response_is_hedged_market_call,
    ask,
    append_external_agent_chat_message,
    load_agent_analysis,
    load_agent_archive_history,
    load_agent_history,
    publish_external_agent_analysis,
)
from scripts.agents.run_codex_agent import extract_json_object


def test_publish_external_agent_analysis_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst.ANALYSIS_FILE",
        tmp_path / "agent_analysis.json",
        raising=False,
    )

    payload = publish_external_agent_analysis(
        {
            "market_view": "Test view",
            "trade_today": True,
            "stocks": [{"symbol": "NABIL", "verdict": "APPROVE", "conviction": 0.8}],
        },
        source="mcp_external",
        provider="ollama",
    )

    restored = load_agent_analysis()

    assert restored["market_view"] == "Test view"
    assert restored["agent_runtime_meta"]["provider"] == "ollama"
    assert payload["stocks"][0]["symbol"] == "NABIL"


def test_append_external_agent_chat_message_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_HISTORY_FILE",
        tmp_path / "agent_chat_history.json",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_ARCHIVE_FILE",
        tmp_path / "agent_chat_archive.json",
        raising=False,
    )

    append_external_agent_chat_message("AGENT", "hello from mcp", source="mcp_external", provider="claude")
    append_external_agent_chat_message("YOU", "follow up", source="mcp_external", provider="claude")
    history = load_agent_history()

    assert len(history) == 2
    assert history[0]["message"] == "hello from mcp"
    assert history[1]["role"] == "YOU"


def test_agent_chat_rolls_older_messages_into_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_HISTORY_FILE",
        tmp_path / "agent_chat_history.json",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.AGENT_ARCHIVE_FILE",
        tmp_path / "agent_chat_archive.json",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.MAX_AGENT_HISTORY_ITEMS",
        3,
        raising=False,
    )

    for idx in range(5):
        append_external_agent_chat_message("AGENT", f"message {idx}", source="mcp_external", provider="codex")

    active = load_agent_history()
    archived = load_agent_archive_history()
    combined = load_agent_history(include_archive=True)

    assert [item["message"] for item in active] == ["message 2", "message 3", "message 4"]
    assert [item["message"] for item in archived] == ["message 0", "message 1"]
    assert [item["message"] for item in combined] == [
        "message 0",
        "message 1",
        "message 2",
        "message 3",
        "message 4",
    ]


def test_merge_agent_output_with_shortlist_keeps_ranked_algo_context():
    merged = _merge_agent_output_with_shortlist(
        {
            "trade_today": True,
            "stocks": [
                {
                    "symbol": "NABIL",
                    "verdict": "APPROVE",
                    "conviction": 0.93,
                    "reasoning": "Best bank setup.",
                    "what_matters": "Breakout with earnings support.",
                }
            ],
        },
        {
            "signals": [
                {
                    "symbol": "NABIL",
                    "type": "volume",
                    "direction": "BUY",
                    "strength": 1.2,
                    "confidence": 0.82,
                    "score": 0.98,
                    "reasoning": "Volume breakout",
                    "rank": 1,
                },
                {
                    "symbol": "SBL",
                    "type": "quality",
                    "direction": "BUY",
                    "strength": 0.7,
                    "confidence": 0.6,
                    "score": 0.42,
                    "reasoning": "Quality composite",
                    "rank": 2,
                },
            ],
            "portfolio": [{"symbol": "SBL"}],
            "prices": {"NABIL": 550.0, "SBL": 640.0},
            "signal_metrics": {
                "NABIL": {"profit_margin_pct": 18.0, "pe_ratio": 9.2, "revenue_growth_qoq_pct": 12.0},
                "SBL": {"profit_margin_pct": 11.0, "pbv_ratio": 1.4},
            },
            "symbol_intelligence": {
                "NABIL": {
                    "story_count": 1,
                    "social_count": 1,
                    "related_count": 1,
                    "story_items": [{"title": "NABIL profit jumps on stronger spread income", "source_name": "My Republica"}],
                    "social_items": [{"text": "NABIL looks strong into earnings", "author_username": "nepsealpha"}],
                    "related_items": [],
                },
                "SBL": {"story_count": 0, "social_count": 0, "related_count": 0},
            },
        },
    )

    assert [row["symbol"] for row in merged["stocks"]] == ["NABIL", "SBL"]
    assert merged["stocks"][0]["action_label"] == "BUY"
    assert merged["stocks"][0]["auto_entry_candidate"] is True
    assert "summary" in merged["stocks"][0]
    assert "confidence_note" in merged["stocks"][0]
    assert "language_detected" in merged["stocks"][0]
    assert "historical_pattern_class" in merged["stocks"][0]
    assert merged["stocks"][1]["verdict"] in {"HOLD", "REJECT"}
    assert merged["stocks"][1]["action_label"] in {"HOLD", "SELL"}
    assert merged["stocks"][1]["is_held"] is True


def test_merge_agent_output_with_shortlist_synthesizes_actionable_verdicts():
    merged = _merge_agent_output_with_shortlist(
        {
            "trade_today": True,
            "stocks": [],
        },
        {
            "signals": [
                {
                    "symbol": "HRL",
                    "type": "anchoring",
                    "direction": "BUY",
                    "strength": 1.1,
                    "confidence": 0.84,
                    "score": 0.91,
                    "reasoning": "52w proximity with expanding volume",
                    "rank": 1,
                },
            ],
            "portfolio": [],
            "prices": {"HRL": 612.0},
            "signal_metrics": {
                "HRL": {
                    "sector": "insurance",
                    "profit_margin_pct": 21.0,
                    "revenue_growth_qoq_pct": 14.0,
                    "profit_growth_qoq_pct": 16.0,
                    "pe_ratio": 10.5,
                    "pbv_ratio": 1.8,
                    "roe_pct": 13.5,
                }
            },
            "symbol_intelligence": {
                "HRL": {
                    "story_count": 2,
                    "social_count": 1,
                    "related_count": 2,
                    "story_items": [
                        {
                            "title": "Himalayan Reinsurance posts profit growth and expands treaty book",
                            "source_name": "My Republica",
                        }
                    ],
                    "social_items": [
                        {
                            "text": "HRL earnings momentum looks strong ahead of results",
                            "author_username": "NepseStock",
                        }
                    ],
                    "related_items": [],
                }
            },
        },
    )

    stock = merged["stocks"][0]
    assert stock["verdict"] in {"APPROVE", "REJECT"}
    assert stock["action_label"] in {"BUY", "PASS"}
    assert stock["conviction"] > 0.35
    assert stock["what_matters"]
    assert stock["likely_impact"]
    assert stock["risks_counterpoints"]


def test_merge_agent_output_maps_unheld_hold_to_pass():
    merged = _merge_agent_output_with_shortlist(
        {
            "trade_today": True,
            "stocks": [
                {
                    "symbol": "NABIL",
                    "verdict": "HOLD",
                    "conviction": 0.62,
                    "reasoning": "Interesting setup but not enough.",
                    "what_matters": "Needs confirmation.",
                }
            ],
        },
        {
            "signals": [
                {
                    "symbol": "NABIL",
                    "type": "anchoring",
                    "direction": "BUY",
                    "strength": 0.9,
                    "confidence": 0.7,
                    "score": 0.71,
                    "reasoning": "Anchoring setup",
                    "rank": 1,
                },
            ],
            "portfolio": [],
            "prices": {"NABIL": 550.0},
            "signal_metrics": {"NABIL": {"profit_margin_pct": 15.0}},
            "symbol_intelligence": {"NABIL": {"story_count": 0, "social_count": 0, "related_count": 0}},
        },
    )

    stock = merged["stocks"][0]
    assert stock["is_held"] is False
    assert stock["verdict"] == "REJECT"
    assert stock["action_label"] == "PASS"


def test_detect_text_language_handles_nepali_english_and_mixed():
    assert _detect_text_language("नाफा १८% बढ्यो") == "ne"
    assert _detect_text_language("Net profit rose 18%") == "en"
    assert _detect_text_language("नाफा rose 18%") == "mixed"


def test_build_analysis_source_packets_includes_metric_and_story_ids():
    packets = _build_analysis_source_packets(
        {
            "session_date": "2026-04-09",
            "nepse_index": 2800.1,
            "nepse_change_pct": 0.8,
            "advancers": 150,
            "decliners": 60,
            "regime": "bull",
            "signals": [{"symbol": "NABIL"}],
            "prices": {"NABIL": 550.0},
        },
        {"NABIL": {"symbol": "NABIL", "sector": "banking", "profit_margin_pct": 18.0}},
        {
            "NABIL": {
                "story_items": [
                    {
                        "title": "नबिलको नाफा spread income ले बढ्यो",
                        "source_name": "My Republica",
                        "published_at": "2026-04-09T10:00:00Z",
                        "url": "https://example.com/nabil",
                    }
                ]
            }
        },
        {
            "BANKING": {
                "story_items": [
                    {
                        "title": "Banking liquidity conditions improve",
                        "source_name": "NepalOSINT",
                        "published_at": "2026-04-09T09:00:00Z",
                        "url": "https://example.com/banking",
                    }
                ]
            }
        },
    )

    assert "MARKET_STATE" in packets["text"]
    assert "[NABIL_FILING][en]" in packets["text"]
    assert "[NABIL_NEWS1][mixed]" in packets["text"]
    assert "[BANKING_NEWS1][en]" in packets["text"]
    assert packets["by_symbol"]["NABIL"][0] == "NABIL_FILING"


def test_ask_news_request_falls_back_to_cited_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPSE_AGENT_DISABLE_HISTORY", "1")
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_HISTORY_FILE", tmp_path / "history.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_ARCHIVE_FILE", tmp_path / "archive.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.load_agent_analysis", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.build_algo_shortlist_snapshot", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst._fetch_macro_market_context", lambda: {}, raising=False)
    monkeypatch.setattr(
        "backend.agents.agent_analyst._load_active_portfolio",
        lambda: ([], {"id": "account_1", "name": "account_1"}),
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._latest_news_context_for_question",
        lambda _question: {
            "items": [
                {
                    "title": "Cabinet clears tourism promotion package",
                    "source_name": "OPMCM",
                    "published_at": "2026-03-30T01:00:00Z",
                    "url": "https://example.com/ann",
                },
                {
                    "title": "Nepal Rastra Bank signals easier liquidity conditions",
                    "source_name": "NepalOSINT",
                    "published_at": "2026-03-30T02:00:00Z",
                    "url": "https://example.com/nrb",
                },
            ],
            "direct_political_hit": False,
            "context_text": "LATEST NEPALOSINT NEWS RESULTS:\n",
        },
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._call_primary_agent",
        lambda *args, **kwargs: "There is no breaking news cited in the current data; the signals are purely technical.",
        raising=False,
    )

    response = ask("what is the breaking news give me proper cited ones")

    assert "Cabinet clears tourism promotion package" in response
    assert "https://example.com/ann" in response
    assert "Nepal Rastra Bank signals easier liquidity conditions" in response


def test_ask_exact_news_request_uses_deterministic_ranked_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPSE_AGENT_DISABLE_HISTORY", "1")
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_HISTORY_FILE", tmp_path / "history.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_ARCHIVE_FILE", tmp_path / "archive.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.load_agent_analysis", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.build_algo_shortlist_snapshot", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst._fetch_macro_market_context", lambda: {}, raising=False)
    monkeypatch.setattr(
        "backend.agents.agent_analyst._load_active_portfolio",
        lambda: ([], {"id": "account_1", "name": "account_1"}),
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._latest_news_context_for_question",
        lambda _question: {
            "items": [
                {
                    "title": "Cabinet clears tourism promotion package",
                    "source_name": "OPMCM",
                    "published_at": "2026-03-30T01:00:00Z",
                    "url": "https://example.com/ann",
                    "category": "government",
                },
                {
                    "title": "Nepal Rastra Bank signals easier liquidity conditions",
                    "source_name": "NepalOSINT",
                    "published_at": "2026-03-30T02:00:00Z",
                    "url": "https://example.com/nrb",
                    "summary": "Banking conditions improved for lenders.",
                },
            ],
            "direct_political_hit": False,
            "context_text": "LATEST NEPALOSINT NEWS RESULTS:\n",
        },
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._call_primary_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called")),
        raising=False,
    )

    response = ask("what anout exact news give me 5 top news that you think may affect nepse when it opens tmrw")

    assert response.startswith("Top 2 NepalOSINT items")
    assert "1. Cabinet clears tourism promotion package" in response
    assert "2. Nepal Rastra Bank signals easier liquidity conditions" in response


def test_ask_political_development_request_uses_ranked_news_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPSE_AGENT_DISABLE_HISTORY", "1")
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_HISTORY_FILE", tmp_path / "history.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_ARCHIVE_FILE", tmp_path / "archive.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.load_agent_analysis", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.build_algo_shortlist_snapshot", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst._fetch_macro_market_context", lambda: {}, raising=False)
    monkeypatch.setattr(
        "backend.agents.agent_analyst._load_active_portfolio",
        lambda: ([], {"id": "account_1", "name": "account_1"}),
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._latest_news_context_for_question",
        lambda _question: {
            "items": [
                {
                    "title": "Cabinet reshuffle triggers coalition talks",
                    "source_name": "NepalOSINT",
                    "published_at": "2026-03-30T03:00:00Z",
                    "url": "https://example.com/politics",
                    "category": "politics",
                },
                {
                    "title": "Nepal Rastra Bank signals easier liquidity conditions",
                    "source_name": "NepalOSINT",
                    "published_at": "2026-03-30T02:00:00Z",
                    "url": "https://example.com/nrb",
                },
            ],
            "direct_political_hit": True,
            "context_text": "LATEST NEPALOSINT NEWS RESULTS:\n",
        },
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._call_primary_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called")),
        raising=False,
    )

    response = ask("what is the recent political development in last 12 hrs in nepal? What does it mean for nepse")

    assert response.startswith("Most relevant political development:")
    assert "Cabinet reshuffle triggers coalition talks" in response
    assert "NEPSE read-through:" in response
    assert "Sources:" in response


def test_ask_short_political_news_prompt_uses_nepalosint_ranked_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPSE_AGENT_DISABLE_HISTORY", "1")
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_HISTORY_FILE", tmp_path / "history.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_ARCHIVE_FILE", tmp_path / "archive.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.load_agent_analysis", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.build_algo_shortlist_snapshot", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst._fetch_macro_market_context", lambda: {}, raising=False)
    monkeypatch.setattr(
        "backend.agents.agent_analyst._load_active_portfolio",
        lambda: ([], {"id": "account_1", "name": "account_1"}),
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._latest_news_context_for_question",
        lambda _question: {
            "items": [
                {
                    "title": "Cabinet reshuffle triggers coalition talks",
                    "source_name": "NepalOSINT",
                    "published_at": "2026-03-30T03:00:00Z",
                    "url": "https://example.com/politics",
                    "category": "politics",
                },
            ],
            "direct_political_hit": True,
            "context_text": "LATEST NEPALOSINT NEWS RESULTS:\n",
        },
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._call_primary_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called")),
        raising=False,
    )

    response = ask("political news?")

    assert response.startswith("Most relevant political development:")
    assert "Cabinet reshuffle triggers coalition talks" in response
    assert "Sources:" in response


def test_ranked_news_answer_keeps_urls_in_sources_block_only(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPSE_AGENT_DISABLE_HISTORY", "1")
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_HISTORY_FILE", tmp_path / "history.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.AGENT_ARCHIVE_FILE", tmp_path / "archive.json", raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.load_agent_analysis", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.build_algo_shortlist_snapshot", lambda: {}, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst._fetch_macro_market_context", lambda: {}, raising=False)
    monkeypatch.setattr(
        "backend.agents.agent_analyst._load_active_portfolio",
        lambda: ([], {"id": "account_1", "name": "account_1"}),
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._latest_news_context_for_question",
        lambda _question: {
            "items": [
                {
                    "title": "Govt plans sweeping asset probe of public officials",
                    "source_name": "Khabarhub",
                    "published_at": "2026-04-01T00:45:00Z",
                    "url": "https://example.com/asset-probe",
                    "category": "politics",
                },
                {
                    "title": "CIB widens probe against former energy minister",
                    "source_name": "My Republica",
                    "published_at": "2026-03-29T09:44:00Z",
                    "url": "https://example.com/cib-probe",
                    "category": "politics",
                },
            ],
            "direct_political_hit": True,
            "context_text": "LATEST NEPALOSINT NEWS RESULTS:\n",
        },
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._call_primary_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called")),
        raising=False,
    )

    response = ask("any political developments?")
    body, sources = response.split("Sources:", 1)

    assert "https://" not in body
    assert "[1]" in body
    assert "https://example.com/asset-probe" in sources
    assert "https://example.com/cib-probe" in sources


def test_political_news_ranking_prefers_actual_government_story_over_nepse_recap():
    items = [
        {
            "title": "Nepse drops by 118 points in two days",
            "summary": "Political uncertainty weighed on sentiment.",
            "source_name": "The Kathmandu Post",
            "published_at": "2026-04-09T02:00:00Z",
            "url": "https://example.com/nepse-drop",
            "category": "market",
        },
        {
            "title": "Cabinet reshuffle triggers coalition talks",
            "summary": "Government negotiations intensified after the latest cabinet decision.",
            "source_name": "Khabarhub",
            "published_at": "2026-04-09T01:00:00Z",
            "url": "https://example.com/cabinet",
            "category": "politics",
        },
    ]

    ranked = _rank_news_items_for_question(items, "political developments?")

    assert ranked[0]["title"] == "Cabinet reshuffle triggers coalition talks"


def test_latest_news_context_uses_history_endpoint_for_explicit_date(monkeypatch):
    history_calls: list[dict] = []
    semantic_calls: list[dict] = []
    monkeypatch.setattr(
        "backend.agents.agent_analyst.consolidated_stories_history",
        lambda **kwargs: history_calls.append(kwargs)
        or {
            "items": [
                {
                    "canonical_headline": "Government faces scandal over procurement decision",
                    "source_name": "NepalOSINT",
                    "first_reported_at": "2025-04-05T08:00:00Z",
                    "url": "https://example.com/historic-scandal",
                    "story_type": "politics",
                }
            ],
            "item_count": 1,
        },
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.unified_search",
        lambda *args, **kwargs: {"categories": {"stories": {"items": []}}},
        raising=False,
    )

    def fake_semantic(query, **kwargs):
        semantic_calls.append({"query": query, **kwargs})
        return {
            "results": [
                {
                    "title": "Government faces scandal over procurement decision",
                    "source_name": "NepalOSINT",
                    "published_at": "2025-04-05T08:00:00Z",
                    "url": "https://example.com/historic-scandal",
                    "category": "politics",
                },
                {
                    "title": "Current NEPSE market recap",
                    "source_name": "NepalOSINT",
                    "published_at": "2026-04-09T08:00:00Z",
                    "url": "https://example.com/current",
                    "category": "market",
                },
            ],
            "total_found": 2,
        }

    monkeypatch.setattr("backend.agents.agent_analyst.semantic_story_search", fake_semantic, raising=False)
    monkeypatch.setattr("backend.agents.agent_analyst.consolidated_stories", lambda **kwargs: [], raising=False)

    ctx = _latest_news_context_for_question("political developments on April 5, 2025")

    assert history_calls
    assert history_calls[0]["start_date"] == "2025-04-05"
    assert history_calls[0]["end_date"] == "2025-04-05"
    assert semantic_calls
    assert semantic_calls[0]["hours"] > 24 * 300
    assert ctx["date_window"]["label"] == "2025-04-05 to 2025-04-05"
    assert [item.get("canonical_headline") or item.get("title") for item in ctx["items"]] == ["Government faces scandal over procurement decision"]


def test_analysis_cache_requires_current_session_and_recent_timestamp(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.agent_analyst._current_nst_session_date",
        lambda: "2026-04-07",
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst.time.time",
        lambda: 1_000.0,
        raising=False,
    )
    monkeypatch.setattr(
        "backend.agents.agent_analyst._active_account_context",
        lambda: {"id": "account_1", "name": "Account 1"},
        raising=False,
    )

    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 700.0, "context_date": "2026-04-07", "account_id": "account_1"}
    ) is True
    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 700.0, "context_date": "2026-04-06", "account_id": "account_1"}
    ) is False
    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 1.0, "context_date": "2026-04-07", "account_id": "account_1"}
    ) is False
    assert _analysis_cache_is_fresh(
        {"stocks": [{"symbol": "NABIL"}], "timestamp": 700.0, "context_date": "2026-04-07", "account_id": "account_2"}
    ) is False


def test_extract_json_object_from_codex_output():
    payload = extract_json_object('preface {"regime":"unknown","signal_count":0} tail')

    assert payload == {"regime": "unknown", "signal_count": 0}


def test_directional_market_question_detection_and_fallback():
    assert _question_is_directional_market_call(
        "How would NEPSE react after the news of KP Oli's release?"
    ) is True
    assert _response_is_hedged_market_call("It depends entirely on the content of the news.") is True

    answer = _build_directional_market_answer(
        "How would NEPSE react after the news of KP Oli's release?",
        {"regime": "bull", "fresh_market": {"advancers": 264, "decliners": 3}},
        {"market_phase": "PREOPEN"},
        {"bias": 0.08, "stories": [{"title": "KP Oli release eases political uncertainty"}], "social": []},
    )

    assert answer.startswith("Base case:")
    assert "pressure" in answer

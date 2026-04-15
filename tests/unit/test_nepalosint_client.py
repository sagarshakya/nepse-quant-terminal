from __future__ import annotations

from backend.quant_pro.nepalosint_client import (
    consolidated_stories,
    consolidated_stories_history,
    resolve_osint_base_url,
    unified_search,
)


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_resolve_osint_base_url_prefers_env_and_normalizes(monkeypatch):
    monkeypatch.setenv("NEPALOSINT_BASE_URL", "https://proxy.example.com")

    assert resolve_osint_base_url() == "https://proxy.example.com/api/v1"


def test_unified_search_uses_public_auth_and_env_base(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.delenv("NEPALOSINT_API_KEY", raising=False)
    monkeypatch.delenv("NEPALOSINT_API_EMAIL", raising=False)
    monkeypatch.delenv("NEPALOSINT_API_PASSWORD", raising=False)
    monkeypatch.setenv("NEPALOSINT_BASE_URL", "https://proxy.example.com")
    monkeypatch.setattr(
        "backend.quant_pro.nepalosint_client.requests.post",
        lambda url, **kwargs: calls.append((url, kwargs)) or DummyResponse({"access_token": "guest-token"}),
        raising=False,
    )
    monkeypatch.setattr(
        "backend.quant_pro.nepalosint_client.requests.get",
        lambda url, **kwargs: calls.append((url, kwargs))
        or DummyResponse({"categories": {"stories": {"items": [{"id": "story-1"}], "total": 1}}}),
        raising=False,
    )

    payload = unified_search("banking")

    assert payload["categories"]["stories"]["total"] == 1
    assert calls[0][0] == "https://proxy.example.com/api/v1/auth/public"
    assert calls[1][0] == "https://proxy.example.com/api/v1/search/unified"
    assert calls[1][1]["headers"]["Authorization"] == "Bearer guest-token"


def test_consolidated_stories_history_passes_date_params(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "backend.quant_pro.nepalosint_client._request_json",
        lambda method, path, **kwargs: captured.update({"method": method, "path": path, **kwargs})
        or {
            "start_date": "2025-04-05",
            "end_date": "2025-04-05",
            "items": [{"canonical_headline": "Historic headline"}],
            "item_count": 1,
            "total_count": 1,
        },
        raising=False,
    )

    payload = consolidated_stories_history(start_date="2025-04-05", end_date="2025-04-05", category="economic", limit=25)

    assert captured["method"] == "GET"
    assert captured["path"] == "/analytics/consolidated-stories/history"
    assert captured["params"]["start_date"] == "2025-04-05"
    assert captured["params"]["end_date"] == "2025-04-05"
    assert captured["params"]["category"] == "economic"
    assert payload["items"][0]["canonical_headline"] == "Historic headline"


def test_unified_search_retries_transient_503(monkeypatch):
    calls: list[str] = []

    monkeypatch.delenv("NEPALOSINT_API_KEY", raising=False)
    monkeypatch.setenv("NEPALOSINT_BASE_URL", "https://proxy.example.com")
    monkeypatch.setattr(
        "backend.quant_pro.nepalosint_client.requests.post",
        lambda url, **kwargs: DummyResponse({"access_token": "public-token"}),
        raising=False,
    )

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return DummyResponse({"error": "temporary"}, status_code=503)
        return DummyResponse({"categories": {"stories": {"items": [{"id": "story-2"}], "total": 1}}})

    monkeypatch.setattr("backend.quant_pro.nepalosint_client.requests.get", fake_get, raising=False)

    payload = unified_search("banking")

    assert len(calls) == 2
    assert payload["categories"]["stories"]["items"][0]["id"] == "story-2"


def test_consolidated_stories_accepts_dict_payloads(monkeypatch):
    monkeypatch.setattr(
        "backend.quant_pro.nepalosint_client._request_json",
        lambda *args, **kwargs: {"stories": [{"title": "Headline", "url": "https://example.com/story"}]},
        raising=False,
    )

    stories = consolidated_stories(limit=3)

    assert stories == [{"title": "Headline", "url": "https://example.com/story"}]

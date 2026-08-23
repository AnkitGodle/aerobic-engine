"""The Groq backend, without a network or a key.

What matters here is the wiring: that JSON mode is requested for the planner and
not for prose, that a 429 becomes a clean fallback rather than a crash, and that
a rate-limit message says something a person can act on.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from core import ai


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    return ai.GroqBackend()


def _fake_urlopen(captured: dict, payload: dict):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def opener(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data)
        return Resp()

    return opener


def test_defaults_to_groqs_best_free_production_model(groq):
    assert groq.model == "openai/gpt-oss-120b"


def test_missing_key_is_a_clean_unavailable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ai.AIUnavailable) as e:
        ai.GroqBackend()
    assert "console.groq.com" in str(e.value)


def test_json_mode_is_requested_only_when_asked(groq, monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen(cap, {"choices": [{"message": {"content": "{}"}}]}))
    groq.complete("sys", "user", json_mode=True)
    assert cap["body"]["response_format"] == {"type": "json_object"}

    cap.clear()
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen(cap, {"choices": [{"message": {"content": "hi"}}]}))
    groq.complete("sys", "user")
    assert "response_format" not in cap["body"], "prose must not be forced to JSON"


def test_request_shape_is_openai_compatible(groq, monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen(cap, {"choices": [{"message": {"content": "x"}}]}))
    groq.complete("SYSTEM", "USER")
    assert cap["url"].endswith("/openai/v1/chat/completions")
    assert cap["headers"]["Authorization"] == "Bearer gsk_test"
    roles = [m["role"] for m in cap["body"]["messages"]]
    assert roles == ["system", "user"]


def test_rate_limit_walks_the_model_chain_then_reports(groq, monkeypatch):
    """Providers meter per model, so a 429 tries the next one — and the message
    names whichever model was tried last rather than crashing."""
    def raise_429(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", {},
            io.BytesIO(json.dumps({"error": {"message": "TPM exceeded"}}).encode()))

    monkeypatch.setattr("urllib.request.urlopen", raise_429)
    with pytest.raises(ai.AIUnavailable) as e:
        groq.complete("s", "u")
    msg = str(e.value)
    assert "rate limit" in msg.lower()
    # The chain is exhausted, so the last model tried is the one named.
    assert groq.models[-1] in msg, "the message must name the model that was metered"
    assert "TPM exceeded" in msg, "the provider's own detail should survive"
    assert e.value.advance_model, "a 429 must advance the model, never retry it"


def test_a_429_on_one_model_succeeds_on_the_next(groq, monkeypatch):
    """The whole point of the chain: separate quota buckets per model."""
    import io
    import urllib.error

    seen: list[str] = []

    def opener(req, timeout=None):
        model = json.loads(req.data)["model"]
        seen.append(model)
        if model == groq.models[0]:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many", {},
                io.BytesIO(json.dumps({"error": {"message": "quota"}}).encode()))

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "second model"}}]}).encode()

        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", opener)
    assert groq.complete("s", "u") == "second model"
    assert len(seen) == 2 and seen[0] != seen[1]


def test_an_empty_response_does_not_silently_pass(groq, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen({}, {"choices": []}))
    with pytest.raises(ai.AIUnavailable):
        groq.complete("s", "u")


def test_planner_falls_back_to_rules_when_groq_is_rate_limited(healthy, monkeypatch):
    """A 429 must not cost the athlete their week."""
    from datetime import date

    from core import planner

    class Limited:
        name = "groq"

        def complete(self, system, user, json_mode=False):
            raise ai.AIUnavailable("Groq rate limit reached")

    monkeypatch.setattr(ai, "get_backend", lambda name=None: Limited())
    plan = planner.plan_week(healthy, today=date(2026, 8, 19), use_ai=True)
    assert plan.source == "rules"
    assert plan.week_plan
    assert any("unavailable" in f.lower() for f in plan.flags)

"""Sending the day's session somewhere you will see it.

No network in any of these: the point is the message and the failure behaviour,
both of which have to be right before a token is ever involved.
"""

from __future__ import annotations

from datetime import date

from conftest import TODAY
from core import goal as goal_mod
from core import notify


def test_nothing_is_sent_unless_a_backend_is_chosen(monkeypatch):
    monkeypatch.delenv("NOTIFY_BACKEND", raising=False)
    assert notify.configured() == "none"
    result = notify.send("today: easy ride")
    assert result.sent is False
    assert result.backend == "none"
    assert bool(result) is False


def test_an_unknown_backend_is_not_guessed_at(monkeypatch):
    monkeypatch.setenv("NOTIFY_BACKEND", "carrier-pigeon")
    assert notify.configured() == "none"
    assert notify.send("x").sent is False


def test_a_backend_missing_its_credentials_says_which(monkeypatch):
    for key in ("WHATSAPP_TOKEN", "WHATSAPP_PHONE_ID", "WHATSAPP_TO",
                "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID",
                "CALLMEBOT_PHONE", "CALLMEBOT_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert "WHATSAPP_TOKEN" in notify.send("x", backend="whatsapp").detail
    assert "TELEGRAM_TOKEN" in notify.send("x", backend="telegram").detail
    assert "CALLMEBOT_PHONE" in notify.send("x", backend="callmebot").detail


def test_an_empty_message_is_not_sent(monkeypatch):
    monkeypatch.setenv("NOTIFY_BACKEND", "telegram")
    assert notify.send("   ").detail == "nothing to send"


def test_a_backend_that_throws_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(notify.BACKENDS, "telegram",
                        lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    result = notify.send("x", backend="telegram")
    assert result.sent is False
    assert "RuntimeError" in result.detail


def test_whatsapp_builds_a_text_message(monkeypatch):
    """The shape Meta expects, and no template unless one is configured."""
    seen = {}

    def fake_post(url, payload, headers):
        seen.update(url=url, payload=payload, headers=headers)
        return True, "HTTP 200"

    monkeypatch.setattr(notify, "_post_json", fake_post)
    monkeypatch.setenv("WHATSAPP_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_ID", "12345")
    monkeypatch.setenv("WHATSAPP_TO", "+91 90000 00000")
    monkeypatch.delenv("WHATSAPP_TEMPLATE", raising=False)

    assert notify.send("easy ride", backend="whatsapp").sent is True
    assert seen["url"].endswith("/12345/messages")
    assert seen["headers"]["Authorization"] == "Bearer tok"
    assert seen["payload"]["type"] == "text"
    # Digits only: Meta rejects a formatted number.
    assert seen["payload"]["to"] == "919000000000"


def test_whatsapp_uses_a_template_when_one_is_set(monkeypatch):
    """Which is what a scheduled 6am message outside the 24-hour window needs."""
    seen = {}
    monkeypatch.setattr(notify, "_post_json",
                        lambda u, p, h: (seen.update(payload=p), (True, "ok"))[1])
    monkeypatch.setenv("WHATSAPP_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_ID", "1")
    monkeypatch.setenv("WHATSAPP_TO", "919000000000")
    monkeypatch.setenv("WHATSAPP_TEMPLATE", "daily_plan")

    notify.send("today: 40 min easy", backend="whatsapp")
    assert seen["payload"]["type"] == "template"
    assert seen["payload"]["template"]["name"] == "daily_plan"
    body = seen["payload"]["template"]["components"][0]["parameters"][0]
    assert body["text"] == "today: 40 min easy"


def test_the_closed_window_error_explains_itself(monkeypatch):
    """Error 131047 is the one everybody hits first."""
    monkeypatch.setattr(notify, "_post_json",
                        lambda u, p, h: (False, '{"error":{"code":131047}}'))
    monkeypatch.setenv("WHATSAPP_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_ID", "1")
    monkeypatch.setenv("WHATSAPP_TO", "919000000000")
    monkeypatch.delenv("WHATSAPP_TEMPLATE", raising=False)
    detail = notify.send("x", backend="whatsapp").detail
    assert "24-hour window" in detail


def test_the_message_says_what_to_do_today(healthy):
    """Composed from the plan that already exists — no model call at 6am."""
    from scripts.notify import compose

    goal_mod.save(healthy, "Pune Half", TODAY.replace(month=12, day=6),
                  distance_km=21.1)
    text = compose(healthy, today=TODAY)
    assert "Aerobic Engine" in text
    assert "Pune Half" in text and "phase" in text
    # Either something to do, something done, or an explicit rest day.
    assert any(word in text for word in ("min", "Done today", "Rest day"))
    assert "left of this week" in text
    goal_mod.clear(healthy)


def test_a_rest_day_is_stated_rather_than_left_blank(healthy):
    from scripts.notify import compose

    text = compose(healthy, today=date(2026, 8, 29))   # a Saturday, nothing on it
    assert "Rest day" in text or "min" in text

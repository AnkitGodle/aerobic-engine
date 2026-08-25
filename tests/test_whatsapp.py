"""The message engine: what it answers, and who it answers.

Two kinds of test here, and the second kind matters more. The first checks that
"today", "week" and "report" say something true. The second checks the boundary:
this endpoint can read health data and re-plan training, so an unsigned delivery,
an unknown number and a duplicate delivery must each go nowhere.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date

import pytest

from core import goal as goal_mod, whatsapp
from core.schemas import DAYS

TODAY = date(2026, 8, 19)          # a Wednesday, as elsewhere in the suite


# --------------------------------------------------------------------------
# What it says
# --------------------------------------------------------------------------


def test_today_says_what_to_do_and_how_you_are(healthy):
    text = whatsapp.compose_today(healthy, TODAY)
    assert "Aerobic Engine" in text
    assert any(w in text for w in ("min", "Done today", "Rest day"))
    assert "left of this week" in text


def test_the_week_is_seven_days_with_today_marked(healthy):
    from core import planner
    planner.plan_week(healthy, today=TODAY, use_ai=False)
    text = whatsapp.compose_week(healthy, TODAY)
    for day in DAYS:
        assert day in text
    assert "→ Wed" in text            # the arrow sits on today


def test_the_week_says_so_when_there_is_no_plan(healthy):
    text = whatsapp.compose_week(healthy, TODAY)
    assert "No plan saved" in text and "replan" in text


def test_the_report_carries_the_recovery_numbers(healthy):
    text = whatsapp.compose_report(healthy, TODAY)
    assert "Resting HR" in text and "HRV" in text
    assert "This week:" in text


def test_the_report_names_the_race_when_there_is_one(healthy):
    goal_mod.save(healthy, "Pune Half", date(2026, 12, 6), distance_km=21.1)
    text = whatsapp.compose_report(healthy, TODAY)
    assert "Pune Half" in text
    goal_mod.clear(healthy)


def test_a_bad_week_is_called_a_deload_not_a_choice(wrecked):
    text = whatsapp.compose_today(wrecked, TODAY)
    assert "Easy week" in text or "deload" in text.lower()


# --------------------------------------------------------------------------
# Reading a check-in out of a sentence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("sleep 4, sore 2, mot 5, 45 min",
     {"sleep": 4, "soreness": 2, "motivation": 5, "time_available_min": 45}),
    ("slept 7 hours", {"sleep": 4}),
    ("sleep 8", {"sleep": 5}),
    ("sleep 4", {"sleep": 4}),                       # inside the scale, kept
    ("only 90 minutes today", {"time_available_min": 90}),
    ("I have 2 hours", {"time_available_min": 120}),
    ("feeling great", {"motivation": 5, "soreness": 2}),
    ("wrecked", {"motivation": 1, "soreness": 5}),
])
def test_a_check_in_is_read_out_of_the_words(text, expected):
    parsed = whatsapp.parse_checkin(text)
    assert parsed is not None
    for key, value in expected.items():
        assert parsed[key] == value, key


def test_seven_hours_of_sleep_is_not_a_score_of_five():
    """The scale is 1-5. "slept 7" is hours, and reading it as 5 would say the
    athlete slept perfectly when they slept adequately."""
    assert whatsapp.parse_checkin("slept 7")["sleep"] == 4
    assert whatsapp.parse_checkin("slept 4 hours")["sleep"] == 4  # inside scale


def test_a_message_with_no_check_in_in_it_is_not_one():
    assert whatsapp.parse_checkin("today") is None
    assert whatsapp.parse_checkin("") is None
    assert whatsapp.parse_checkin("what is my vo2max") is None


def test_the_free_text_is_kept_for_the_planner(healthy):
    parsed = whatsapp.parse_checkin("tired, knee feels off, 45 min")
    assert parsed["notes"] == "tired, knee feels off, 45 min"


def test_a_check_in_saves_and_replans(healthy):
    text = whatsapp.apply_checkin(healthy, "sleep 2, sore 4, 40 min",
                                  today=TODAY, use_ai=False)
    assert "Noted: sleep 2/5" in text
    stored = healthy.latest_checkin()
    assert stored["sleep"] == 2 and stored["soreness"] == 4
    assert stored["time_available_min"] == 40
    # And it left a plan behind, not just a check-in.
    assert healthy.latest_plan(TODAY.replace(day=17)) is not None


def test_a_check_in_cannot_talk_its_way_past_a_deload(wrecked):
    """The point of the guardrails: enthusiasm over a message changes nothing."""
    from core import planner
    text = whatsapp.apply_checkin(
        wrecked, "feeling great, 3 hours, want a hard session",
        today=TODAY, use_ai=False)
    facts = planner.build_facts(wrecked, today=TODAY)
    envelope = planner.build_envelope(facts, wrecked)
    assert envelope.deload is True
    assert "Easy week" in text or "easy" in text.lower()


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("message,markers", [
    ("today", ("Aerobic Engine",)),
    ("TODAY", ("Aerobic Engine",)),
    (" today ", ("Aerobic Engine",)),
    # Either the week, or the line that says there is not one saved yet. Both
    # mean the message reached the week composer, which is what is under test.
    ("week", ("This week", "No plan saved")),
    ("plan", ("This week", "No plan saved")),
    ("weekly", ("This week", "No plan saved")),
    ("report", ("How you are",)),
    ("status", ("How you are",)),
    ("how am i doing", ("How you are",)),
    ("help", ("by message",)),
    ("hi", ("by message",)),
    ("", ("by message",)),
])
def test_the_commands_are_forgiving(healthy, message, markers):
    answer = whatsapp.reply(healthy, message, TODAY)
    assert any(m in answer for m in markers), answer[:120]


def test_a_check_in_beats_a_question_when_it_is_both(healthy, monkeypatch):
    """"tired, should I still ride?" is both. The re-planned week is the answer."""
    monkeypatch.setattr("core.ai.available", lambda: False)
    text = whatsapp.reply(healthy, "tired, sore 4, should I still ride?", TODAY)
    assert "Noted:" in text


def test_a_question_goes_to_the_coach(healthy, monkeypatch):
    asked = {}

    def fake_ask(question, payload, today, backend=None):
        asked["q"] = question
        return "Because your easy pace is not easy."
    monkeypatch.setattr("core.insights.ask", fake_ask)
    text = whatsapp.reply(healthy, "why is my heart rate so high?", TODAY)
    assert "easy pace" in text
    assert asked["q"] == "why is my heart rate so high?"


def test_a_question_survives_the_ai_being_off(healthy, monkeypatch):
    monkeypatch.setattr("core.insights.ask", lambda *a, **k: None)
    text = whatsapp.reply(healthy, "how is my form coming along?", TODAY)
    assert "report" in text


def test_a_broken_answer_is_a_sentence_not_a_traceback(healthy, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("database went away")
    monkeypatch.setattr("core.whatsapp.compose_today", explode)
    text = whatsapp.reply(healthy, "today", TODAY)
    assert "broke" in text.lower()
    assert "RuntimeError" not in text


def test_nonsense_gets_the_help_text(healthy):
    assert "by message" in whatsapp.reply(healthy, "xyzzy", TODAY)


# --------------------------------------------------------------------------
# Who is allowed to talk to it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sender,allowed,ok", [
    ("919940173970", "919940173970", True),
    ("9940173970", "919940173970", True),          # same phone, no country code
    ("+91 99401 73970", "919940173970", True),     # as a human writes it
    ("919940173971", "919940173970", False),       # one digit out
    ("919940173970", "911111111111,919940173970", True),
    ("919940173970", "", False),                   # empty list allows nobody
    ("", "919940173970", False),
])
def test_only_the_allowlist_is_answered(sender, allowed, ok):
    assert whatsapp.sender_allowed(sender, allowed) is ok


def test_an_unsigned_delivery_is_not_from_meta():
    body = b'{"entry": []}'
    assert whatsapp.verify_signature(body, "", "secret") is False
    assert whatsapp.verify_signature(body, "sha256=deadbeef", "secret") is False


def test_a_signed_delivery_verifies():
    body = b'{"entry": [1]}'
    digest = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    assert whatsapp.verify_signature(body, f"sha256={digest}", "secret") is True
    assert whatsapp.verify_signature(body, digest, "secret") is True


def test_no_secret_configured_means_nothing_verifies():
    """Failing open on the one control that proves who is calling is not a
    default worth having."""
    body = b"{}"
    digest = hmac.new(b"", body, hashlib.sha256).hexdigest()
    assert whatsapp.verify_signature(body, f"sha256={digest}", "") is False


def test_a_changed_body_fails_the_signature():
    digest = hmac.new(b"secret", b'{"a": 1}', hashlib.sha256).hexdigest()
    assert whatsapp.verify_signature(b'{"a": 2}', f"sha256={digest}",
                                     "secret") is False


# --------------------------------------------------------------------------
# Unpacking what Meta sends
# --------------------------------------------------------------------------


def _delivery(text: str = "today", number: str = "919940173970",
              message_id: str = "wamid.1") -> dict:
    return {"object": "whatsapp_business_account", "entry": [{"changes": [{
        "value": {"messaging_product": "whatsapp",
                  "messages": [{"id": message_id, "from": number, "type": "text",
                                "text": {"body": text}}]},
        "field": "messages"}]}]}


def test_a_text_message_is_found_in_the_nesting():
    found = whatsapp.messages_from(_delivery("week"))
    assert found == [{"id": "wamid.1", "from": "919940173970", "text": "week"}]


def test_a_delivery_receipt_is_not_a_message():
    """Meta sends delivered and read through the same endpoint. Answering those
    would mean every reply triggered its own reply."""
    receipt = {"entry": [{"changes": [{"value": {
        "statuses": [{"id": "wamid.1", "status": "delivered"}]}}]}]}
    assert whatsapp.messages_from(receipt) == []


def test_an_image_is_not_answered():
    payload = _delivery()
    payload["entry"][0]["changes"][0]["value"]["messages"][0]["type"] = "image"
    assert whatsapp.messages_from(payload) == []


def test_an_empty_or_odd_body_finds_nothing():
    assert whatsapp.messages_from({}) == []
    assert whatsapp.messages_from({"entry": [{}]}) == []
    assert whatsapp.messages_from(json.loads('{"entry": [{"changes": []}]}')) == []

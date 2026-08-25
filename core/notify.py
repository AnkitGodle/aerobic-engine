"""Send the day's session somewhere you will actually see it.

The app only helps on days you open it. A message at 6am — what today is, why,
and what the watch is holding you to — closes that gap, and it is the one feature
here that changes behaviour rather than describing it.

Four ways out, chosen by `NOTIFY_BACKEND`:

  * **whatsapp** — Meta's WhatsApp Cloud API. Your own Meta app, your own number,
    no middleman reading the message. Two things to know before wiring it up: a
    free-form message only sends inside 24 hours of *you* messaging the number
    first, and outside that window it has to be an approved template. So a daily
    6am push needs a template ("utility" category, free inside an open service
    window); a reply-and-get-your-plan flow does not. Meta retired the
    1,000-free-conversations model in July 2025 — what is free now is the service
    window, not a monthly quota.
  * **telegram** — a bot token and a chat id, no templates, no approval, no
    window. Ten minutes end to end. Technically the better tool; it is simply
    not WhatsApp.
  * **callmebot** — the shortcut: message their number once, get a key, send
    yourself WhatsApp messages with a GET. No Meta app at all, and the trade is
    that a third party handles the text of your training. Supported because it
    works, and flagged because of that trade.
  * **none** — the default. Nothing is sent.

Everything here is plain HTTPS with the standard library: no SDK, no extra
dependency, and `send()` returns a result rather than raising, because a failed
notification must never be the reason a sync fails.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("aerobic_engine.notify")

TIMEOUT_S = 20
USER_AGENT = "aerobic-engine/1.0"


@dataclass
class Result:
    """What happened, in a shape the caller can log or show."""

    sent: bool
    backend: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.sent


def _post_json(url: str, payload: dict[str, Any],
               headers: dict[str, str]) -> tuple[bool, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001 - a message is never worth raising for
        return False, f"{type(exc).__name__}: {exc}"


def _get(url: str) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def send_whatsapp(text: str, template: str = "", language: str = "en") -> Result:
    """Meta's WhatsApp Cloud API.

    Needs `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_ID` (the sending number's id from the
    Meta app) and `WHATSAPP_TO` (your number in international format, digits
    only). With `WHATSAPP_TEMPLATE` set, the template is sent instead of the text
    — which is what a scheduled message outside the 24-hour window requires. The
    template must take one body parameter, and this puts the message in it.
    """
    token = os.getenv("WHATSAPP_TOKEN", "")
    phone_id = os.getenv("WHATSAPP_PHONE_ID", "")
    to = os.getenv("WHATSAPP_TO", "").replace("+", "").replace(" ", "")
    template = template or os.getenv("WHATSAPP_TEMPLATE", "")
    if not (token and phone_id and to):
        return Result(False, "whatsapp",
                      "WHATSAPP_TOKEN, WHATSAPP_PHONE_ID and WHATSAPP_TO must "
                      "all be set")
    url = f"https://graph.facebook.com/v21.0/{phone_id}/messages"
    if template:
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp", "to": to, "type": "template",
            "template": {
                "name": template,
                "language": {"code": os.getenv("WHATSAPP_LANG", language)},
                "components": [{"type": "body", "parameters": [
                    {"type": "text", "text": text[:1024]}]}],
            },
        }
    else:
        payload = {"messaging_product": "whatsapp", "to": to, "type": "text",
                   "text": {"preview_url": False, "body": text[:4096]}}
    ok, detail = _post_json(url, payload,
                            {"Authorization": f"Bearer {token}"})
    if not ok and "131047" in detail:
        detail += (" — that is the 24-hour window closing. Message your number "
                   "first, or set WHATSAPP_TEMPLATE to an approved template.")
    return Result(ok, "whatsapp", detail)


def send_telegram(text: str) -> Result:
    """A bot token and a chat id. No templates, no window, no approval."""
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not (token and chat):
        return Result(False, "telegram",
                      "TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must both be set")
    ok, detail = _post_json(
        f"https://api.telegram.org/bot{token}/sendMessage",
        {"chat_id": chat, "text": text[:4096], "disable_web_page_preview": True},
        {})
    return Result(ok, "telegram", detail)


def send_callmebot(text: str) -> Result:
    """WhatsApp without a Meta app, at the cost of a third party seeing the text."""
    phone = os.getenv("CALLMEBOT_PHONE", "").replace(" ", "")
    key = os.getenv("CALLMEBOT_KEY", "")
    if not (phone and key):
        return Result(False, "callmebot",
                      "CALLMEBOT_PHONE and CALLMEBOT_KEY must both be set")
    query = urllib.parse.urlencode({"phone": phone, "apikey": key,
                                    "text": text[:900]})
    ok, detail = _get(f"https://api.callmebot.com/whatsapp.php?{query}")
    return Result(ok, "callmebot", detail)


BACKENDS = {"whatsapp": send_whatsapp, "telegram": send_telegram,
            "callmebot": send_callmebot}


def configured() -> str:
    """Which backend is switched on. "none" when nothing is."""
    name = os.getenv("NOTIFY_BACKEND", "none").strip().lower()
    return name if name in BACKENDS else "none"


def send(text: str, backend: str | None = None) -> Result:
    """Send one message. Never raises; returns what happened."""
    name = (backend or configured()).strip().lower()
    if name == "none" or name not in BACKENDS:
        return Result(False, "none", "NOTIFY_BACKEND is not set")
    if not (text or "").strip():
        return Result(False, name, "nothing to send")
    try:
        return BACKENDS[name](text)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        log.warning("Could not send via %s: %s", name, exc)
        return Result(False, name, f"{type(exc).__name__}: {exc}")

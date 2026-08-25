#!/usr/bin/env python3
"""Answer WhatsApp messages about your training.

    python scripts/whatsapp_webhook.py                 # listen on :8787
    python scripts/whatsapp_webhook.py --port 9000
    python scripts/whatsapp_webhook.py --say "today"   # no server, just answer

Meta's WhatsApp Cloud API delivers incoming messages by POSTing them to a URL
you own, so this is a small HTTP server rather than a poller — there is nothing
to poll. Two routes and nothing else:

    GET  /webhook   the one-time subscription handshake
    POST /webhook   a delivered message, answered by core.whatsapp.reply()

Standard library only, because a dependency that terminates HTTPS or parses
webhooks is a dependency that can break the one thing here that talks to the
outside world.

**Set-up, once** (about ten minutes):

 1. Create a Meta app at developers.facebook.com, add the WhatsApp product, and
    note the test number's phone id.
 2. Put `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_ID`, `WHATSAPP_CONTACT` (your own
    number, digits only), `WHATSAPP_APP_SECRET` and a `WHATSAPP_VERIFY_TOKEN`
    you invent in `.env`.
 3. Expose this port over HTTPS — `cloudflared tunnel --url http://localhost:8787`
    is free and needs no account — and give Meta `https://…/webhook` with your
    verify token.
 4. Message your test number once. Meta only allows free-form replies inside 24
    hours of you writing first; outside that window a reply has to be an
    approved template, which is why the daily push uses `WHATSAPP_TEMPLATE`.

**What it will not do.** It answers only numbers on `WHATSAPP_CONTACT` (comma
separate for more than one), it requires Meta's signature on every delivery, and
it ignores anything it has already answered. This endpoint can re-plan training
and read health data; an open one would let a stranger do both.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core import applog, notify, whatsapp  # noqa: E402
from core.store import Store, default_db  # noqa: E402

log = logging.getLogger("aerobic_engine.webhook")

MAX_BODY = 256 * 1024          # a text message is a few hundred bytes
# Meta retries a delivery it thinks failed, and a retry must not re-plan the
# week a second time. Ids only, and only the last few hundred.
SEEN: deque[str] = deque(maxlen=400)


def answer(text: str, sender: str, db: str | None) -> None:
    """Work out the reply and send it. Logged either way."""
    target = db or default_db()
    with Store(target) as store:
        body = whatsapp.reply(store, text)
    result = notify.send(body)
    applog.event(target, "WhatsApp: " + ("replied" if result.sent
                                         else "reply failed"),
                 asked=text[:120], to=sender[-4:], detail=result.detail[:200])
    print(f"[{sender[-4:]}] {text[:60]!r} -> "
          + ("sent" if result.sent else f"not sent: {result.detail[:120]}"))


class Handler(BaseHTTPRequestHandler):
    db: str | None = None
    server_version = "aerobic-engine"

    def log_message(self, fmt: str, *args: object) -> None:   # quieter default
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes = b"", kind: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        url = urlparse(self.path)
        if url.path not in ("/webhook", "/"):
            self._send(404)
            return
        query = parse_qs(url.query)
        token = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
        sent = (query.get("hub.verify_token") or [""])[0]
        challenge = (query.get("hub.challenge") or [""])[0]
        # The handshake, and the one place a plain GET is answered. Compared with
        # compare_digest because it is a shared secret like any other.
        if (query.get("hub.mode") or [""])[0] == "subscribe" and token \
                and hmac.compare_digest(token, sent):
            self._send(200, challenge.encode("utf-8"))
            return
        if not url.query:
            self._send(200, b"aerobic-engine webhook is up")
            return
        self._send(403, b"verification failed")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/webhook":
            self._send(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._send(400, b"bad length")
            return
        raw = self.rfile.read(length)

        secret = os.getenv("WHATSAPP_APP_SECRET", "")
        signature = self.headers.get("X-Hub-Signature-256", "")
        if not whatsapp.verify_signature(raw, signature, secret):
            # 200 rather than 401: an unsigned POST is not Meta, and there is
            # nothing to be gained by telling whoever sent it anything useful.
            log.warning("Rejected an unsigned delivery from %s",
                        self.address_string())
            self._send(200, b"")
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, b"bad json")
            return

        # Answered immediately: Meta retries anything it does not see accepted
        # within a few seconds, and a re-plan can take longer than that.
        self._send(200, b"")

        allowed = os.getenv("WHATSAPP_CONTACT") or os.getenv("WHATSAPP_TO", "")
        for message in whatsapp.messages_from(payload):
            if message["id"] and message["id"] in SEEN:
                continue
            if message["id"]:
                SEEN.append(message["id"])
            if not whatsapp.sender_allowed(message["from"], allowed):
                log.warning("Ignored a message from %s — not on the allowlist",
                            message["from"][-4:])
                continue
            try:
                answer(message["text"], message["from"], self.db)
            except Exception as exc:  # noqa: BLE001 - one bad message, not the server
                log.warning("Could not answer: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("WEBHOOK_PORT", "8787")))
    parser.add_argument("--host", default=os.getenv("WEBHOOK_HOST", "127.0.0.1"),
                        help="127.0.0.1 by default; a tunnel reaches that fine")
    parser.add_argument("--db", default=None)
    parser.add_argument("--say", default=None,
                        help="answer one message and exit, sending nothing")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.say is not None:
        with Store(args.db or default_db()) as store:
            print(whatsapp.reply(store, args.say))
        return 0

    missing = [k for k in ("WHATSAPP_TOKEN", "WHATSAPP_PHONE_ID",
                           "WHATSAPP_APP_SECRET", "WHATSAPP_VERIFY_TOKEN")
               if not os.getenv(k)]
    if missing:
        print("Not configured yet: " + ", ".join(missing))
        print("The server will still start, but deliveries will be rejected "
              "(unsigned) and replies will not send. See this file's docstring.")
    if not (os.getenv("WHATSAPP_CONTACT") or os.getenv("WHATSAPP_TO")):
        print("WHATSAPP_CONTACT is empty, so every message will be ignored. "
              "That is deliberate — an empty allowlist allows nobody.")

    Handler.db = args.db
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}/webhook")
    print("Expose it over HTTPS, e.g. "
          f"cloudflared tunnel --url http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

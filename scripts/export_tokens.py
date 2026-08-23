#!/usr/bin/env python
"""Export the cached Garmin session so a hosted app never logs in to SSO itself.

This is the single most important thing standing between a web deployment and a
flagged Garmin account. Garmin is markedly stricter with datacenter IPs than with
home ones, and a cloud host cannot answer an MFA prompt. So: log in once here, on
your own connection, and give the host the resulting *session tokens*. The host
then resumes that session and refreshes it against Garmin's token endpoint, which
is a different, far less scrutinised path than a fresh SSO login.

    python scripts/export_tokens.py              # print the blob to paste
    python scripts/export_tokens.py --out t.txt  # write it to a file

Treat the output like a password: it grants access to your Garmin account until
you change your password. Never commit it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core.garmin_client import GarminClient  # noqa: E402


def prompt_mfa() -> str:
    return input("Garmin MFA code: ").strip()


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Export Garmin session tokens")
    ap.add_argument("--out", help="write the blob to this file instead of stdout")
    args = ap.parse_args()

    client = GarminClient(prompt_mfa=prompt_mfa)
    api = client.connect()
    print(f"Logged in as {client.display_name or '(unknown)'}", file=sys.stderr)

    blob = api.client.dumps()
    if not blob or len(blob) <= 512:
        print(
            "The token blob came back suspiciously short. The hosted app decides "
            "between 'a path' and 'token data' by length, so a short blob would be "
            "treated as a path and fail. Delete .garmin_tokens and log in again.",
            file=sys.stderr,
        )
        return 1

    if args.out:
        Path(args.out).write_text(blob)
        Path(args.out).chmod(0o600)
        print(f"Written to {args.out} ({len(blob)} chars). Do not commit it.",
              file=sys.stderr)
    else:
        print(
            "\nPaste this as GARMINTOKENS in your host's secrets "
            "(one line, no quotes needed in Streamlit secrets):\n",
            file=sys.stderr,
        )
        print(blob)
    print(
        "\nThese tokens expire. When the hosted app reports that the session is "
        "unusable, run this again and replace the secret.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

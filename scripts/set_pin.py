#!/usr/bin/env python
"""Generate a salted PIN hash for the dashboard's write actions.

Writes REFRESH_PIN_SALT and REFRESH_PIN_HASH into .env (and prints them for
copying into a hosting provider's secrets). The PIN itself is never stored.

    python scripts/set_pin.py            # prompts, does not echo
    python scripts/set_pin.py --pin 1234 # non-interactive (shell history!)
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.auth import hash_pin, new_salt  # noqa: E402

ENV = Path(__file__).resolve().parents[1] / ".env"


def upsert_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(values)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    if remaining:
        out.append("")
        out.append("# Dashboard write-action PIN (salted hash — the PIN is not stored)")
        out.extend(f"{k}={v}" for k, v in remaining.items())
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Set the dashboard write PIN")
    ap.add_argument("--pin", help="avoid this: it lands in your shell history")
    ap.add_argument("--env", default=str(ENV))
    args = ap.parse_args()

    pin = args.pin or getpass.getpass("New PIN: ")
    if not args.pin:
        again = getpass.getpass("Confirm PIN: ")
        if pin != again:
            print("PINs did not match.")
            return 1
    pin = pin.strip()
    if len(pin) < 4:
        print("Use at least 4 characters — a shorter PIN is trivially guessed.")
        return 1

    salt = new_salt()
    digest = hash_pin(pin, salt)
    upsert_env(Path(args.env), {
        "REFRESH_PIN_SALT": salt,
        "REFRESH_PIN_HASH": digest,
        "REFRESH_PIN": "",   # clear any plaintext PIN left behind
    })
    print(f"Written to {args.env}\n")
    print("For a hosted deployment, put these in the host's secrets:")
    print(f'REFRESH_PIN_SALT = "{salt}"')
    print(f'REFRESH_PIN_HASH = "{digest}"')
    print("\nThe PIN itself is not stored anywhere. Losing it means re-running this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

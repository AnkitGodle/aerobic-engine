"""The PIN is the only thing between a public URL and someone editing the
athlete's training data, so it is tested as a credential."""

from __future__ import annotations

import core.auth as auth
from core.auth import PinGate, hash_pin, new_salt


class FakeStore:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}

    def get_state(self, key: str, default: str | None = None) -> str | None:
        return self.kv.get(key, default)

    def set_state(self, key: str, value: str) -> None:
        self.kv[key] = value


# A throwaway value. The real PIN must never appear in the repository — this file
# would otherwise publish it the moment the project is pushed to a public remote.
TEST_PIN = "4242"


def gate(store=None, pin=TEST_PIN):
    salt = new_salt()
    return PinGate(store or FakeStore(), pin_hash=hash_pin(pin, salt), salt=salt,
                   plaintext="")


def test_correct_pin_unlocks():
    ok, _ = gate().verify("4242")
    assert ok


def test_wrong_pin_is_refused():
    ok, msg = gate().verify("0000")
    assert not ok and "Incorrect" in msg


def test_the_pin_is_never_stored_in_config():
    salt = new_salt()
    digest = hash_pin("4242", salt)
    assert "4242" not in digest
    assert "4242" not in salt
    g = PinGate(FakeStore(), pin_hash=digest, salt=salt, plaintext="")
    # Nothing on the object reveals the PIN.
    assert "4242" not in repr(g.__dict__)


def test_same_pin_different_salt_gives_different_hash():
    """A stolen hash from one deployment is useless against another."""
    assert hash_pin("4242", new_salt()) != hash_pin("4242", new_salt())


def test_brute_force_gets_locked_out():
    g = gate()
    for _ in range(auth.FREE_ATTEMPTS):
        g.verify("0000")
    assert g.lockout_remaining() == 0  # free attempts are free
    ok, msg = g.verify("0001")
    assert not ok
    assert g.lockout_remaining() > 0, "attempts past the free ones must cost time"
    # And while locked out, even the correct PIN waits.
    ok, msg = g.verify("4242")
    assert not ok and "Locked" in msg


def test_lockout_grows_with_each_failure():
    g = gate()
    for _ in range(auth.FREE_ATTEMPTS + 1):
        g.verify("0000")
    first = g.lockout_remaining()
    g._save_attempts({"failures": auth.FREE_ATTEMPTS + 4, "locked_until": None})
    g.verify("0000")
    assert g.lockout_remaining() > first


def test_lockout_survives_a_new_session():
    """Opening a fresh tab must not reset the attempt counter."""
    store = FakeStore()
    g1 = gate(store)
    for _ in range(auth.FREE_ATTEMPTS + 1):
        g1.verify("0000")
    assert gate(store).lockout_remaining() > 0


def test_success_clears_the_counter():
    g = gate()
    g.verify("0000")
    g.verify("4242")
    assert g._attempts()["failures"] == 0


def test_no_pin_configured_means_writes_are_refused():
    g = PinGate(FakeStore(), pin_hash="", salt="", plaintext="")
    assert not g.configured
    ok, msg = g.verify("anything")
    assert not ok and "No PIN" in msg
    assert any("anyone" in w for w in g.warnings())


def test_plaintext_pin_works_but_is_flagged():
    g = PinGate(FakeStore(), pin_hash="", salt="", plaintext="4242")
    assert g.verify("4242")[0]
    assert any("plaintext" in w for w in g.warnings())


def test_sessions_expire():
    assert auth.session_expired(None, 0)
    assert auth.session_expired(0.0, auth.SESSION_TTL_S + 1)
    assert not auth.session_expired(0.0, auth.SESSION_TTL_S - 1)


def test_non_ascii_pin_is_rejected_not_crashed():
    """`hmac.compare_digest` on str raises TypeError for non-ASCII.

    A PIN containing an accented or non-Latin character must return False, not
    take the page down with an exception.
    """
    g = gate(pin="pässwörd")
    ok, _ = g.verify("wrong")
    assert ok is False
    assert g.verify("pässwörd")[0] is True


def test_non_ascii_plaintext_pin_also_works():
    from core.auth import PinGate

    g = PinGate(FakeStore(), pin_hash="", salt="", plaintext="日本語")
    assert g.verify("日本語")[0] is True
    assert g.verify("nope")[0] is False


# --------------------------------------------------------------------------
# The read gate. A separate PIN from the write one, and separately counted.
# --------------------------------------------------------------------------


def test_read_and_write_lockouts_are_independent():
    """Sharing one counter would let strangers failing at the front door lock
    the owner out of their own controls."""
    store = FakeStore()
    salt = new_salt()
    read = PinGate(store, pin_hash=hash_pin("1111", salt), salt=salt,
                   attempts_key="read_pin_attempts")
    write = PinGate(store, pin_hash=hash_pin("2222", salt), salt=salt)

    for _ in range(6):
        read.verify("0000")
    assert read.lockout_remaining() > 0
    assert write.lockout_remaining() == 0
    assert write.verify("2222")[0]


def test_the_read_pin_is_not_the_write_pin():
    salt = new_salt()
    read = PinGate(FakeStore(), pin_hash=hash_pin("1111", salt), salt=salt,
                   attempts_key="read_pin_attempts")
    assert read.verify("1111")[0]
    assert not PinGate(FakeStore(), pin_hash=hash_pin("1111", salt), salt=salt,
                       attempts_key="read_pin_attempts").verify("2222")[0]


def test_no_read_pin_configured_means_the_dashboard_is_open():
    """Absence has to mean public, or an upgrade would lock the owner out."""
    gate = PinGate(FakeStore(), pin_hash="", salt="", plaintext="")
    assert not gate.configured

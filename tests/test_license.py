"""Foveance Pro licensing: offline key verification, activation storage, and the persistent
SavingsLog. Signature-roundtrip tests generate their own RSA keypair (cryptography, dev-only)
and point the verifier's modulus at it; everything else is pure stdlib."""
import base64
import json
import time

import pytest

from foveance import license as lic
from foveance.proxy import FoveanceProxy


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(lic, "_home_dir", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture()
def signer(monkeypatch):
    """A real RSA keypair; the verifier is pointed at its modulus for the test."""
    crypto = pytest.importorskip("cryptography")  # noqa: F841  (dev-only dependency)
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(lic, "_PUB_N", key.public_key().public_numbers().n)

    def sign(email="a@b.c", plan="solo"):
        payload = json.dumps({"v": 1, "email": email, "plan": plan, "iat": int(time.time())},
                             separators=(",", ":"), sort_keys=True).encode()
        sig = key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
        return f"FOV1-{_b64u(payload)}-{_b64u(sig)}"

    return sign


def test_valid_key_roundtrip_activate_status_deactivate(home, signer):
    key = signer(email="buyer@example.com", plan="team")
    data = lic.activate(key)
    assert data == lic.current()
    assert data["email"] == "buyer@example.com" and data["plan"] == "team"
    assert lic.deactivate() is True
    assert lic.current() is None and lic.deactivate() is False


def test_tampered_and_malformed_keys_rejected(home, signer):
    key = signer()
    payload_b64 = key.split("-")[1]
    tampered = json.loads(lic._b64u_decode(payload_b64))
    tampered["plan"] = "enterprise"
    forged = "FOV1-" + _b64u(json.dumps(tampered).encode()) + "-" + key.split("-")[2]
    assert lic.parse_key(forged) is None                    # signature no longer matches
    assert lic.parse_key("FOV1-abc-def") is None            # garbage b64
    assert lic.parse_key("FOV2-" + key[5:]) is None         # wrong version tag
    assert lic.parse_key("") is None
    assert lic.activate("not-a-key") is None
    assert lic.current() is None                            # nothing was stored


def test_savings_log_records_aggregates_and_exports(tmp_path):
    log = lic.SavingsLog(path=str(tmp_path / "s.db"))
    log.record(1000, 400)
    log.record(500, 100)
    t = log.totals()
    assert (t["requests"], t["tokens_before"], t["tokens_after"]) == (2, 1500, 500)
    assert t["tokens_saved"] == 1000
    days = log.by_day()
    assert len(days) == 1 and days[0]["tokens_saved"] == 1000
    csv = log.export_csv()
    assert csv.splitlines()[0] == "day,requests,tokens_before,tokens_after,tokens_saved"
    assert ",2,1500,500,1000" in csv


def test_proxy_persists_when_log_attached(tmp_path):
    log = lic.SavingsLog(path=str(tmp_path / "p.db"))
    px = FoveanceProxy(budget=120, savings_log=log)
    msgs = [{"role": "user", "content": "FACT k=V\n" + "log line\n" * 200},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "recall k"}]
    px.prepare({"model": "x", "user": "c1", "messages": msgs})
    t = log.totals()
    assert t["requests"] == 1 and t["tokens_saved"] > 0
    s = px.stats()
    assert s["alltime"]["tokens_saved"] == t["tokens_saved"]
    assert "usd_saved" in s["alltime"]


def test_proxy_without_log_has_no_alltime():
    px = FoveanceProxy(budget=120)
    px.prepare({"messages": [{"role": "user", "content": "hi"}]})
    assert "alltime" not in px.stats()


def test_cli_license_flow(home, signer, capsys):
    from foveance import cli
    key = signer(email="cli@example.com", plan="solo")
    assert cli.main(["license", "activate", key]) == 0
    assert "cli@example.com" in capsys.readouterr().out
    assert cli.main(["license", "status"]) == 0
    assert "Pro active" in capsys.readouterr().out
    assert cli.main(["license", "deactivate"]) == 0
    cli.main(["license"])  # default action: status
    assert "No active license" in capsys.readouterr().out
    assert cli.main(["license", "activate", "FOV1-bogus-key"]) == 1

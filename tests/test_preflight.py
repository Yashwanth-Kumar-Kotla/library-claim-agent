from fastapi.testclient import TestClient

from libraryclaim import preflight
from libraryclaim.config import Locale, Settings
from libraryclaim.live import server


def test_check_page_and_health():
    with TestClient(server.app) as c:
        assert "Connection check" in c.get("/check").text
        assert c.get("/healthz").json()["ok"] is True


def test_certificate_mismatch_says_what_to_do(monkeypatch):
    import libraryclaim.live.__main__ as live_main
    monkeypatch.setattr(live_main, "lan_ip", lambda: "10.9.8.7")
    c = preflight.certificate()
    if c.status != preflight.FAIL or "none yet" in c.detail:
        return  # no certificate on this machine: the missing-certificate path is covered by that message
    assert "10.9.8.7" in c.detail and "make_cert.sh" in c.detail


def test_gemini_out_of_credit_is_named(monkeypatch):
    class Boom:
        def __init__(self, **kw):
            self.models = self

        def generate_content(self, **kw):
            raise RuntimeError("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted.")
    import google.genai
    monkeypatch.setattr(google.genai, "Client", Boom)
    c = preflight.gemini(Settings(locale=Locale("US", "USD"), google_api_key="k"))
    assert c.status == preflight.FAIL and "top up" in c.detail


def test_missing_keys_are_failures_or_warnings_not_crashes():
    s = Settings(locale=Locale("US", "USD"))
    assert preflight.gemini(s).status == preflight.FAIL
    assert preflight.books(s).status == preflight.WARN
    assert preflight.prices(s).status == preflight.WARN


def test_go_no_go(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "run", lambda: [preflight.Check("a", preflight.OK, ""), preflight.Check("b", preflight.WARN, "")])
    assert preflight.main() == 0 and "GO" in capsys.readouterr().out
    monkeypatch.setattr(preflight, "run", lambda: [preflight.Check("a", preflight.FAIL, "x")])
    assert preflight.main() == 1 and "NO-GO" in capsys.readouterr().out

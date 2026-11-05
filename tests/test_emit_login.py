import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode

import pytest

from jerqazyna.data.emit_login import TokenHandoff, handler_for, token_environment


def test_token_child_environment_does_not_change_parent(monkeypatch):
    import os

    monkeypatch.setenv("EARTHDATA_USERNAME", "fixture-user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "fixture-password")
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    environment = token_environment("fixture-not-a-real-secret")
    assert environment["EARTHDATA_TOKEN"] == "fixture-not-a-real-secret"
    assert "EARTHDATA_PASSWORD" not in environment
    assert "EARTHDATA_USERNAME" not in environment
    assert "EARTHDATA_TOKEN" not in os.environ
    assert os.environ["EARTHDATA_PASSWORD"] == "fixture-password"


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    app = TokenHandoff(tmp_path, tmp_path / "pilot.yaml", tmp_path)
    captured = []
    monkeypatch.setattr(app, "run", lambda token: captured.append(token))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
    app.origin = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, server.server_port, captured
    server.shutdown()
    server.server_close()


def request(port, method, path, data=None, origin=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if origin:
        headers["Origin"] = origin
    connection.request(method, path, body=data, headers=headers)
    response = connection.getresponse()
    status, body, response_headers = response.status, response.read(), dict(response.getheaders())
    connection.close()
    return status, body, response_headers


def test_handoff_rejects_cross_origin_and_unknown_path(handoff):
    app, port, captured = handoff
    body = urlencode({"token": "fixture-not-a-real-secret"})
    assert request(port, "POST", f"/login/{app.nonce}", body, "https://example.org")[0] == 403
    assert request(port, "POST", "/login/wrong", body, app.origin)[0] == 403
    assert request(port, "GET", "/login/wrong")[0] == 404
    assert not captured


def test_handoff_does_not_echo_or_persist_token(handoff, tmp_path):
    app, port, _ = handoff
    token = "fixture-not-a-real-secret"
    status, body, headers = request(port, "POST", f"/login/{app.nonce}", urlencode({"token": token}), app.origin)
    assert status == 303 and token.encode() not in body
    assert headers["Cache-Control"] == "no-store"
    assert headers["Referrer-Policy"] == "same-origin"
    status, body, _ = request(port, "GET", f"/login/{app.nonce}")
    assert status == 200 and token.encode() not in body
    for path in tmp_path.iterdir():
        assert token not in path.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "login_handoff.json").read_text())["phase"] == "fetch"


def test_handoff_rejects_invalid_tokens_and_overlapping_jobs(handoff):
    app, port, _ = handoff
    assert request(port, "POST", f"/login/{app.nonce}", urlencode({"token": "short"}), app.origin)[0] == 400
    app.state["phase"] = "fetch"
    body = urlencode({"token": "fixture-not-a-real-secret"})
    assert request(port, "POST", f"/login/{app.nonce}", body, app.origin)[0] == 409

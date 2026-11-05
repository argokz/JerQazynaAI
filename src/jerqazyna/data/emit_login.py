"""Loopback-only user handoff of an Earthdata token to the bounded EMIT worker.

No browser cookies are exported. The user creates/pastes the token themselves; it is
passed in memory to the fetch child environment and never written to logs or files.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from jerqazyna.features.cache import write_json


def token_environment(token):
    environment = os.environ.copy()
    for name in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        environment.pop(name, None)
    environment["EARTHDATA_TOKEN"] = token
    return environment


class TokenHandoff:
    def __init__(self, root, pilot, dest):
        self.root, self.pilot, self.dest = Path(root), Path(pilot), Path(dest)
        self.nonce = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.state = {"phase": "awaiting_token", "pid": os.getpid()}
        self.origin = ""

    def save(self):
        write_json(self.dest / "login_handoff.json", self.state)

    def accept(self, token):
        if not 20 <= len(token) <= 8192 or any(c.isspace() or ord(c) < 33 for c in token):
            return 400
        with self.lock:
            if self.state["phase"] in {"fetch", "prepare", "report"}:
                return 409
            self.state = {"phase": "fetch", "pid": os.getpid()}
            self.save()
        threading.Thread(target=self.run, args=(token,), daemon=True).start()
        return 303

    def run(self, token):
        environment = token_environment(token)
        token = None
        try:
            for stage in ("fetch", "prepare", "report"):
                with self.lock:
                    self.state["phase"] = stage
                    self.save()
                command = [sys.executable, "-m", "jerqazyna.cli", "emit", stage,
                           str(self.pilot), "--root", str(self.root)]
                # Discard child console output: authenticated errors must not echo credentials.
                result = subprocess.run(command, cwd=self.root, env=environment if stage == "fetch" else None,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                environment.pop("EARTHDATA_TOKEN", None)
                if result.returncode:
                    with self.lock:
                        self.state.update(phase=f"{stage}_failed", return_code=result.returncode)
                        self.save()
                    return
            with self.lock:
                self.state["phase"] = "completed"
                self.save()
        except (OSError, ValueError) as exc:
            with self.lock:
                self.state.update(phase="worker_failed", error_type=type(exc).__name__)
                self.save()
        finally:
            environment.clear()

    def progress(self):
        with self.lock:
            progress = dict(self.state)
        fetch_path = self.dest / "fetch.json"
        if fetch_path.exists():
            try:
                fetch = json.loads(fetch_path.read_text(encoding="utf-8"))
                progress["ready_assets"] = len(fetch.get("files", {}))
                progress["fetch_state"] = fetch.get("state")
            except (OSError, ValueError):
                pass  # Atomic status replacement can be briefly locked on Windows.
        ledger = self.root / "data/pilots/emit_transfer_ledger.json"
        if ledger.exists():
            try:
                totals = json.loads(ledger.read_text(encoding="utf-8"))
                progress["observed_transfer_bytes"] = totals["observed_bytes"]
                progress["charged_transfer_bytes"] = totals["charged_bytes"]
            except (OSError, ValueError):
                pass
        return progress

    def page(self):
        progress = self.progress()
        phase = progress["phase"]
        running = phase in {"fetch", "prepare", "report"}
        refresh = f'<meta http-equiv="refresh" content="5;url=/login/{self.nonce}">' if running else ""
        form = ""
        if not running and phase != "completed":
            form = f'''<form method="post" action="/login/{self.nonce}" autocomplete="off">
            <label for="token">Earthdata Bearer Token</label>
            <input id="token" name="token" type="password" autocomplete="off" required minlength="20" maxlength="8192">
            <button type="submit">Передать токен и продолжить пилот</button></form>'''
        return f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
        <title>EMIT — локальное подключение</title>{refresh}<style>
        body{{font:17px system-ui;max-width:720px;margin:48px auto;padding:0 24px;color:#142d35;background:#f3f7f6}}
        h1{{font-size:30px}}p{{line-height:1.6}}input,button{{box-sizing:border-box;width:100%;padding:14px;margin:12px 0;font:inherit}}
        input{{border:1px solid #567b7b;border-radius:6px}}button{{background:#105a5b;color:white;border:0;border-radius:6px;cursor:pointer}}
        pre{{background:#e5eeee;padding:16px;white-space:pre-wrap}}</style>
        <h1>Подключение Earthdata к пилоту EMIT</h1>
        <p>Это локальная страница проекта JerQazynaAI на вашем компьютере.
        На странице NASA нажмите <b>GENERATE TOKEN</b> и вставьте полученный токен сюда.
        Пароль NASA здесь не требуется.</p>
        <p>Токен передаётся только локальному процессу загрузки, который использует его для доступа к NASA.
        Он не сохраняется в файлах проекта и не отправляется в чат.
        Загрузка ограничена 10 ГиБ с учётом повторов; затем выполняются подготовка и отчёт.</p>
        {form}<h2>Состояние</h2><pre>{html.escape(json.dumps(progress,ensure_ascii=False,indent=2))}</pre>
        </html>'''.encode()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log request bodies, form values or private paths.

        def respond(self, code, body=b"", location=None):
            if code >= 400 and not body:
                body = f'<!doctype html><meta charset="utf-8"><title>EMIT: запрос не принят</title><h1>Запрос не принят ({code})</h1><p>Токен не был использован. Секретные значения не выводятся. Диагностика сохранена локально.</p>'.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            # Chromium serializes the Origin of a form POST as null under no-referrer.
            # same-origin retains strict Origin verification and leaks nothing off-origin.
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
            self.send_header("Content-Length", str(len(body)))
            if location:
                self.send_header("Location", location)
            self.end_headers()
            self.wfile.write(body)

        def valid_path(self):
            return self.path == f"/login/{app.nonce}" and self.headers.get("Host") == app.origin.removeprefix("http://")

        def do_GET(self):
            if not self.valid_path():
                self.respond(404)
                return
            self.respond(200, app.page())

        def do_POST(self):
            if not self.valid_path() or self.headers.get("Origin") != app.origin:
                with app.lock:
                    app.state["last_request_rejection"] = {
                        "reason": "origin_or_path", "valid_path_and_host": self.valid_path(),
                        "origin_present": "Origin" in self.headers,
                        "origin_is_null": self.headers.get("Origin") == "null",
                        "origin_matches": self.headers.get("Origin") == app.origin,
                        "fetch_site_same_origin": self.headers.get("Sec-Fetch-Site") == "same-origin"}
                    app.save()
                self.respond(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.respond(400)
                return
            if not 0 < length <= 32768 or self.headers.get("Content-Type", "").split(";")[0] != "application/x-www-form-urlencoded":
                self.respond(400)
                return
            try:
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), max_num_fields=1)
                values = fields.get("token", [])
                code = app.accept(values[0].strip()) if len(values) == 1 else 400
            except (UnicodeError, ValueError):
                code = 400
            self.respond(code, location=f"/login/{app.nonce}" if code == 303 else None)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot")
    parser.add_argument("--root", default=".")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--nonce", help="Restore an existing local handoff URL; never an Earthdata token")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise ValueError("Use the isolated Python 3.12 EMIT environment")
    if not 0 <= args.port <= 65535:
        raise ValueError("Invalid loopback port")
    if args.nonce and not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", args.nonce):
        raise ValueError("Invalid local handoff nonce")
    from jerqazyna.data.emit import frozen_search, load_pilot
    from jerqazyna.data.review import local_path

    root = Path(args.root).resolve()
    pilot = local_path(root, args.pilot)
    dest, _ = frozen_search(root, load_pilot(pilot))
    app = TokenHandoff(root, pilot, dest)
    if args.nonce:
        app.nonce = args.nonce
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app))
    app.origin = f"http://127.0.0.1:{server.server_port}"
    app.save()
    print(json.dumps({"url": f"{app.origin}/login/{app.nonce}", "pid": os.getpid(), "state": "awaiting_token"}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

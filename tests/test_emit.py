import io
import json

import pytest

from jerqazyna.data import emit

pytest.importorskip("shapely")


def config():
    return emit.PilotConfig(pilot_id="test", task="task.yaml", baseline="run", split="split",
                            areas=[emit.PilotArea(name="area", bbox=(66, 47, 67, 48))])


def entry(kind, date, scene="001", west=66, east=67):
    stamp = date.replace("-", "") + "T000000"
    key = f"001_{stamp}_2222205_{scene}"
    names = [f"EMIT_L2B_MIN_{key}.nc", f"EMIT_L2B_MINUNCERT_{key}.nc"] if kind == "min" else [f"EMIT_L2A_MASK_{key}.nc", f"EMIT_L2A_RFL_{key}.nc"]
    return {"id": kind + key, "time_start": date + "T00:00:00Z",
            "polygons": [[f"47 {west} 48 {west} 48 {east} 47 {east} 47 {west}"]],
            "links": [{"href": "https://data.lpdaac.earthdatacloud.nasa.gov/test/" + name} for name in names]}


def test_asset_allowlist_pairing_and_date_selection():
    c = config()
    dates = ["2023-06-01", "2024-06-01", "2025-06-01", "2025-01-01"]
    mins = [entry("min", date) for date in dates]
    rfls = [entry("rfl", date) for date in dates]
    selected, coverage, missing = emit.select_scenes(mins, rfls, c.areas[0], c, "EPSG:6933")
    assert {s["date"] for s in selected} == {"2024-06-01", "2025-06-01"}
    assert not missing
    assert all({a["kind"] for a in s["assets"]} == {"MIN", "MINUNCERT", "MASK"} for s in selected)
    assert all(row["selected_fraction"] <= 1.000001 for row in coverage)
    selected, _, missing = emit.select_scenes(mins, rfls[:1], c.areas[0], c, "EPSG:6933")
    assert len(selected) == 1 and len(missing) == 2


def test_greedy_selects_incremental_area_not_duplicate_scene():
    c = config()
    mins = [entry("min", "2024-06-01", "001", 66, 66.6),
            entry("min", "2024-06-01", "002", 66, 66.6),
            entry("min", "2024-06-01", "003", 66.6, 67)]
    rfls = [entry("rfl", "2024-06-01", scene) for scene in ("001", "002", "003")]
    selected, _, _ = emit.select_scenes(mins, rfls, c.areas[0], c, "EPSG:6933")
    assert [s["key"][-1] for s in selected] == ["001", "003"]


class Raw(io.BytesIO):
    def read(self, amount, decode_content=False):
        return super().read(amount)


class Response:
    def __init__(self, payload, length=True):
        self.raw = Raw(payload)
        self.headers = {"Content-Length": str(len(payload))} if length else {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass


def test_budget_is_durable_and_counts_repeated_transfers(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = emit.TransferLedger(path, 10)
    ledger.transfer(Response(b"123456"), tmp_path / "part", "public")
    ledger = emit.TransferLedger(path, 10)
    with pytest.raises(emit.BudgetExceeded):
        ledger.transfer(Response(b"123456"), tmp_path / "part", "public")
    assert json.loads(path.read_text())["charged_bytes"] == 6
    assert (tmp_path / "part").read_bytes() == b"123456"


def test_unknown_size_reserves_before_read_and_never_exceeds_limit(tmp_path):
    ledger = emit.TransferLedger(tmp_path / "ledger.json", 10)
    response = Response(b"1234567890EXTRA", length=False)
    with pytest.raises(emit.BudgetExceeded):
        ledger.transfer(response, tmp_path / "part", "public")
    assert ledger.data["charged_bytes"] == 10
    assert ledger.data["observed_bytes"] == 10
    assert response.raw.tell() == 10


def test_interrupted_read_still_charged(tmp_path):
    class Broken:
        def read(self, *args, **kwargs):
            raise OSError("disconnected")

    ledger = emit.TransferLedger(tmp_path / "ledger.json", 10)
    response = Response(b"1234")
    response.raw = Broken()
    with pytest.raises(OSError):
        ledger.transfer(response, tmp_path / "part", "public")
    reloaded = emit.TransferLedger(tmp_path / "ledger.json", 10)
    assert reloaded.data["charged_bytes"] == 4 and reloaded.data["observed_bytes"] == 0


def test_transient_windows_atomic_replace_does_not_double_charge(tmp_path, monkeypatch):
    original = emit._atomic_write_json
    calls = []

    def temporary_sharing_violation(path, data):
        calls.append(data["charged_bytes"])
        if len(calls) < 3:
            raise PermissionError("fixture: Windows sharing violation")
        original(path, data)

    monkeypatch.setattr(emit, "_atomic_write_json", temporary_sharing_violation)
    monkeypatch.setattr(emit.time, "sleep", lambda _: None)
    ledger = emit.TransferLedger(tmp_path / "ledger.json", 10)
    ledger.reserve(4)
    assert calls == [4, 4, 4]
    assert emit.TransferLedger(tmp_path / "ledger.json", 10).data["charged_bytes"] == 4


def test_fetch_without_auth_and_repeat_corruption(tmp_path, monkeypatch):
    c = config()
    m, r = entry("min", "2024-06-01"), entry("rfl", "2024-06-01")
    selected, _, _ = emit.select_scenes([m], [r], c.areas[0], c, "EPSG:6933")
    saved = {"spec": {"scenes": selected}}
    monkeypatch.setattr(emit, "frozen_search", lambda *args: (tmp_path, saved))
    monkeypatch.setattr(emit, "authenticated_session", lambda: None)
    assert emit.fetch(c, tmp_path)[1] == "awaiting_auth"
    assert not (tmp_path / "raw").exists()
    monkeypatch.setattr(emit, "validate_scientific", lambda path: None)

    class Session:
        count = 0

        def get(self, *args, **kwargs):
            self.count += 1
            return Response(b"scientific-fixture")

    session = Session()
    assert emit.fetch(c, tmp_path, session)[1] == "ready"
    assert session.count == 3
    emit.fetch(c, tmp_path, session)
    assert session.count == 3
    assert emit.fetch(c, tmp_path)[1] == "ready"  # completed data can be reused offline
    one = next((tmp_path / "raw").glob("*.nc"))
    one.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        emit.ready_files(tmp_path, saved)
    emit.fetch(c, tmp_path, session)
    assert session.count == 4
    assert emit.ready_files(tmp_path, saved)["state"] == "ready"


def test_transfer_lock_prevents_concurrent_budget_writers(tmp_path):
    path = tmp_path / "ledger.lock"
    with emit.transfer_lock(path), pytest.raises(ValueError, match="lock"), emit.transfer_lock(path):
        pass
    assert not path.exists()


def test_config_rejects_budget_extension():
    data = config().recipe()
    data["budget_bytes"] = emit.LIMIT + 1
    with pytest.raises(ValueError):
        emit.PilotConfig.model_validate(data)


def test_redirect_bodies_are_charged_and_signed_url_not_persisted(tmp_path):
    import requests

    class RedirectResponse(Response):
        status_code = 302
        url = "https://data.lpdaac.earthdatacloud.nasa.gov/asset.nc"

        def __init__(self):
            super().__init__(b"redirect")
            self.headers["Location"] = "https://example.org/file.nc?signature=secret"

        def close(self):
            pass

    class Session(requests.Session):
        def get(self, *args, **kwargs):
            assert kwargs["allow_redirects"] is False
            return RedirectResponse()

        def rebuild_auth(self, prepared, response):
            pass

        def send(self, prepared, **kwargs):
            assert prepared.url.endswith("signature=secret")
            return Response(b"science")

    ledger = emit.TransferLedger(tmp_path / "ledger.json", 100)
    asset = {"url": "https://data.lpdaac.earthdatacloud.nasa.gov/asset.nc"}
    emit.download_asset(Session(), asset, tmp_path / "data.part", ledger)
    assert ledger.data["observed_bytes"] == len(b"redirectscience")
    assert ledger.data["charged_bytes"] == len(b"redirectscience")
    assert "secret" not in (tmp_path / "ledger.json").read_text()


def test_requests_redirect_does_not_secretly_consume_body(tmp_path):
    import requests
    from requests.adapters import BaseAdapter

    class Adapter(BaseAdapter):
        count = 0

        def send(self, request, **kwargs):
            self.count += 1
            response = requests.Response()
            response.status_code = 302 if self.count == 1 else 200
            payload = b"redirect" if self.count == 1 else b"science"
            response.headers["Content-Length"] = str(len(payload))
            response.headers["Location"] = "https://example.org/science"
            response.raw = Raw(payload)
            response.request = request
            response.url = request.url
            return response

        def close(self):
            pass

    session = requests.Session()
    session.mount("https://", Adapter())
    resolver = session.resolve_redirects
    ledger = emit.TransferLedger(tmp_path / "ledger.json", 100)
    emit.download_asset(session, {"url": "https://example.org/redirect"}, tmp_path / "data.part", ledger)
    assert ledger.data["observed_bytes"] == len(b"redirectscience")
    assert session.resolve_redirects == resolver

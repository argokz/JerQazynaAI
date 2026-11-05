"""API веб-карты на синтетическом экспорте: слои, лицензии, карточка ячейки, коды ошибок."""

from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("rasterio")

from fastapi.testclient import TestClient
from jerqazyna_web.app import create_app
from jerqazyna_web.export import export
from jerqazyna_web.registry import load_registry
from jerqazyna_web.settings import Settings
from pyproj import Transformer
from web_helpers import AREA_COLS

from jerqazyna.data.download import sha256_of

ROOT_CONFIG = "configs/sources.yaml"
ORIGIN = "http://localhost:5173"


@pytest.fixture
def exported(project, catalog):
    """Экспорт без research_score: недоступный слой тоже должен попасть в ответ."""
    registry = load_registry(project["registry"], catalog)
    ids = ["dem_elevation", "magnetics_anomaly", "faults_active_distance"]
    manifest = export(registry, project["root"], project["out"], only=ids, blocksize=128,
                      log=lambda *_: None)
    return {**project, "manifest": manifest}


def _settings(project, **overrides):
    from web_helpers import ROOT

    values = {
        "registry": project["registry"], "sources": ROOT / ROOT_CONFIG,
        "data_dir": project["out"], "tiles_url": "http://tiles.test:8001/",
        "tiles_data_dir": "/data/cogs", "cors_origins": [ORIGIN],
    }
    return Settings(**{**values, **overrides})


@pytest.fixture
def client(exported):
    with TestClient(create_app(_settings(exported))) as c:
        yield c


def _lonlat(grid, row, col):
    to_wgs84 = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    x = grid.x_min + (col + 0.5) * grid.cell_size
    y = grid.y_max - (row + 0.5) * grid.cell_size
    return to_wgs84.transform(x, y)


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "task_id": "test_task", "layers_total": 4,
                    "layers_available": 3}


def test_layers_have_licenses_attribution_and_tiles(client, exported):
    body = client.get("/api/layers").json()
    lon_min, lat_min, lon_max, lat_max = body["bounds"]
    assert lon_min < 70 < lon_max and lat_min < 45 < lat_max
    layers = {layer["id"]: layer for layer in body["layers"]}
    assert list(layers) == ["dem_elevation", "magnetics_anomaly", "faults_active_distance",
                            "research_score"]
    for layer in layers.values():
        assert layer["sources"]
        for source in layer["sources"]:
            assert source["license"] and source["attribution"] and source["citation"]

    dem = layers["dem_elevation"]
    assert dem["available"] and dem["visible"]
    assert dem["sources"][0]["notice"] == "no liability"
    assert dem["file"]["cog_sha256"] == exported["manifest"].layers["dem_elevation"].cog_sha256
    assert dem["legend"]["stops"][0] == {"position": 0.0, "color": "#333399"}
    url = urlsplit(dem["tiles_url"])
    assert url.netloc == "tiles.test:8001"
    assert url.path == "/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png"
    query = parse_qs(url.query)
    assert query["url"] == ["/data/cogs/dem_elevation.tif"]
    assert query["rescale"] == ["0,3000"] and query["colormap_name"] == ["terrain"]

    assert layers["faults_active_distance"]["sources"][0]["share_alike"] is True
    research = layers["research_score"]
    assert not research["available"] and research["tiles_url"] is None
    assert research["unavailable_reason"] == "COG не экспортирован"
    assert not research["visible"] and "слепой тест" in research["warning"]


def test_cell_returns_values_and_provenance(client, exported):
    grid = exported["grid"]
    row, col = 40, 150
    lon, lat = _lonlat(grid, row, col)
    response = client.get("/api/cell", params={"lon": lon, "lat": lat})
    assert response.status_code == 200
    body = response.json()
    assert (body["row"], body["col"]) == (row, col)
    assert body["cell_id"] == row * grid.width + col
    assert body["x"] == grid.x_min + (col + 0.5) * 1000
    values = {layer["id"]: layer for layer in body["layers"]}
    arrays = exported["arrays"]
    assert values["dem_elevation"]["value"] == float(arrays["dem__elevation_m.tif"][row, col])
    assert values["magnetics_anomaly"]["value"] == float(col - row)
    dem_file = values["dem_elevation"]["file"]
    source = exported["root"] / "data/features/test_task/dem__elevation_m.tif"
    assert dem_file["source_sha256"] == sha256_of(source)
    assert dem_file["provenance_manifest"] == "data/features/test_task/cache.json"
    assert dem_file["provenance"]["recipe"] == "r-dem"
    assert values["research_score"]["available"] is False
    assert values["research_score"]["value"] is None


def test_cell_nan_value_is_null(client, exported):
    lon, lat = _lonlat(exported["grid"], 5, 100)
    body = client.get("/api/cell", params={"lon": lon, "lat": lat}).json()
    dem = next(layer for layer in body["layers"] if layer["id"] == "dem_elevation")
    assert dem["available"] and dem["value"] is None


def test_cell_errors(client, exported):
    grid = exported["grid"]
    lon, lat = _lonlat(grid, 10, AREA_COLS - 1)
    response = client.get("/api/cell", params={"lon": lon, "lat": lat})
    assert response.status_code == 404 and "вне области" in response.json()["detail"]
    response = client.get("/api/cell", params={"lon": 80.0, "lat": 45.0})
    assert response.status_code == 422 and "вне сетки" in response.json()["detail"]
    assert client.get("/api/cell", params={"lon": 70, "lat": 95}).status_code == 422
    assert client.get("/api/cell", params={"lon": 70}).status_code == 422


def test_cors_allows_only_configured_origins(client):
    allowed = client.get("/api/health", headers={"Origin": ORIGIN})
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    other = client.get("/api/health", headers={"Origin": "http://evil.test"})
    assert "access-control-allow-origin" not in other.headers


def test_startup_rejects_tampered_cog(exported):
    cog = exported["out"] / exported["manifest"].layers["magnetics_anomaly"].file
    cog.write_bytes(cog.read_bytes() + b"\0")
    with pytest.raises(RuntimeError, match="sha256"):
        create_app(_settings(exported))


def test_startup_requires_export(project):
    with pytest.raises(RuntimeError, match="export_cogs"):
        create_app(_settings(project))


def test_settings_from_env(tmp_path):
    settings = Settings.from_env({
        "JERQAZYNA_WEB_DATA_DIR": str(tmp_path),
        "JERQAZYNA_WEB_TILES_URL": "https://tiles.example.org/",
        "JERQAZYNA_WEB_CORS_ORIGINS": "https://a.example.org, https://b.example.org",
    })
    assert settings.data_dir == tmp_path
    assert settings.tiles_url == "https://tiles.example.org"
    assert settings.cors_origins == ["https://a.example.org", "https://b.example.org"]
    assert Settings.from_env({"JERQAZYNA_WEB_CORS_ORIGINS": ""}).cors_origins == []
    with pytest.raises(ValueError):
        Settings.from_env({"JERQAZYNA_WEB_CORS_ORIGINS": "*"})
    with pytest.raises(ValueError):
        Settings.from_env({"JERQAZYNA_WEB_TILES_URL": "ftp://x"})
    assert Settings.from_env({}).cors_origins == ["http://localhost:5173", "http://localhost:8080"]

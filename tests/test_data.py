import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jerqazyna.data.catalog import load_catalog
from jerqazyna.data.download import Manifest, download, sha256_of
from jerqazyna.data.fetch import fetch_source
from jerqazyna.data.tiles import degree_tiles, tile_urls

CATALOG = Path(__file__).parents[1] / "configs" / "sources.yaml"


class RangeHandler(SimpleHTTPRequestHandler):
    """Статический сервер с поддержкой Range, чтобы проверить докачку."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().do_GET()
        data = path.read_bytes()
        start = int(rng.split("=")[1].split("-")[0])
        self.send_response(206)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_catalog_loads_and_ids_unique():
    cat = load_catalog(CATALOG)
    ids = [s.id for s in cat.sources]
    assert "usgs_sir2010_5090n" in ids
    assert not cat.get("usgs_porphyry_2025").automatic


def test_degree_tiles_priority_first():
    tiles = degree_tiles((70, 44, 74, 46), first=(72, 45, 73, 46))
    assert len(tiles) == 8
    assert tiles[0] == (45, 72)


def test_five_degree_tiles_named_by_corner():
    tiles = degree_tiles((72, 44.5, 80, 48.5), step=5)
    assert tiles == [(40, 70), (40, 75), (45, 70), (45, 75)]
    urls = list(tile_urls("https://x/{name}.dg", tiles[-1:]))
    assert urls == ["https://x/N45E075.dg"]


def test_copernicus_tile_names():
    template = "https://x/Copernicus_DSM_COG_30_{lat}_00_{lon}_00_DEM.tif"
    urls = list(tile_urls(template, degree_tiles((74, 46, 75, 47))))
    assert urls == ["https://x/Copernicus_DSM_COG_30_N46_00_E074_00_DEM.tif"]


def test_download_resumes_partial_file(server, tmp_path):
    root, base = server
    payload = bytes(range(256)) * 4000
    (root / "f.bin").write_bytes(payload)
    dest = tmp_path / "out" / "f.bin"
    dest.parent.mkdir()
    (dest.parent / "f.bin.part").write_bytes(payload[:1000])
    assert download(f"{base}/f.bin", dest, progress=False)
    assert dest.read_bytes() == payload
    assert not download(f"{base}/f.bin", dest, progress=False)  # уже есть


def test_download_404_raises(server, tmp_path):
    _, base = server
    with pytest.raises(FileNotFoundError):
        download(f"{base}/nope.tif", tmp_path / "nope.tif", progress=False)


def test_fetch_direct_writes_manifest(server, tmp_path):
    root, base = server
    (root / "a.txt").write_text("hello")
    cat = load_catalog(CATALOG)
    src = type(cat.get("gem_active_faults")).model_validate(
        {**cat.get("gem_active_faults").model_dump(),
         "files": [{"url": f"{base}/a.txt", "filename": "a.txt"}]}
    )
    data_root = tmp_path / "data"
    manifest = Manifest(data_root / "manifest.csv", data_root)
    report = fetch_source(src, cat, data_root, manifest)
    out = data_root / "raw" / "gem_active_faults" / "a.txt"
    assert report.downloaded == [out]
    reloaded = Manifest(data_root / "manifest.csv", data_root)
    row = reloaded.rows["raw/gem_active_faults/a.txt"]
    assert row["sha256"] == sha256_of(out)


def test_manual_source_gives_instructions(tmp_path):
    cat = load_catalog(CATALOG)
    src = cat.get("usgs_porphyry_2025")
    manifest = Manifest(tmp_path / "manifest.csv", tmp_path)
    report = fetch_source(src, cat, tmp_path, manifest)
    assert "10.5066/P14CCESQ" in report.instructions
    (tmp_path / "raw" / src.id).mkdir(parents=True)
    (tmp_path / "raw" / src.id / "porphyry.xlsx").write_bytes(b"x")
    assert fetch_source(src, cat, tmp_path, manifest).instructions == ""

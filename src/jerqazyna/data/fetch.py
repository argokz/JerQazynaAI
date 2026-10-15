"""Загрузка источника из каталога в data/raw/<id>/ и проверка, что уже собрано."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from jerqazyna.data.catalog import Catalog, Source
from jerqazyna.data.download import USER_AGENT, Manifest, download, unzip
from jerqazyna.data.tiles import degree_tiles, tile_bbox, tile_name, tile_urls

GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/{level}/"


@dataclass
class FetchReport:
    source_id: str
    downloaded: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)
    instructions: str = ""


def source_dir(data_root: Path, source: Source) -> Path:
    return Path(data_root) / "raw" / source.id


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _fetch_one(url: str, dest: Path, report: FetchReport, manifest: Manifest, src: Source) -> bool:
    try:
        if download(url, dest):
            report.downloaded.append(dest)
            manifest.record(src.id, dest, url)
            manifest.save()  # сохраняем после каждого файла, чтобы не потерять прогресс
            return True
        report.skipped.append(dest)
        if dest.resolve().relative_to(manifest.data_root.resolve()).as_posix() not in manifest.rows:
            manifest.record(src.id, dest, url)
            manifest.save()
        return False
    except FileNotFoundError:
        report.missing.append(url)
        return False


def fetch_source(
    src: Source,
    catalog: Catalog,
    data_root: Path,
    manifest: Manifest,
    dry_run: bool = False,
    priority_only: bool = False,
    workers: int = 1,
) -> FetchReport:
    report = FetchReport(src.id)
    out = source_dir(data_root, src)
    bbox = catalog.priority_bbox if priority_only else catalog.bbox

    if src.access in ("manual", "request"):
        found = present_files(data_root, src)
        if not found:
            where = "Запросить" if src.access == "request" else "Скачать в браузере"
            report.instructions = (
                f"{where}: {src.page}\n  Положить в: {out}\n  Ожидаются файлы: {', '.join(src.expect)}"
            )
        else:
            report.skipped.extend(found)
        return report

    if src.access == "direct":
        for f in src.files:
            dest = out / f.filename
            if dry_run:
                report.planned.append(f.url)
                continue
            fresh = _fetch_one(f.url, dest, report, manifest, src)
            if f.unzip and dest.exists() and (fresh or not (out / dest.stem).exists()):
                unzip(dest, out / dest.stem)

    elif src.access == "tiles":
        step = src.tile_step_deg
        tiles = land_tiles(degree_tiles(bbox, catalog.priority_bbox, step), data_root, catalog, step)
        for url in tile_urls(src.tile_url, tiles):
            if dry_run:
                report.planned.append(url)
                continue
            _fetch_one(url, out / url.rsplit("/", 1)[-1], report, manifest, src)

    elif src.access == "geoboundaries":
        for b in src.boundaries:
            api = GEOBOUNDARIES_API.format(iso=b.iso, level=b.level)
            if dry_run:
                report.planned.append(api)
                continue
            url = _get_json(api)["gjDownloadURL"]
            _fetch_one(url, out / f"geoBoundaries-{b.iso}-{b.level}.geojson", report, manifest, src)

    elif src.access == "stac":
        from jerqazyna.data.sentinel2 import composite_tile

        tiles = land_tiles(degree_tiles(bbox, catalog.priority_bbox), data_root, catalog)
        todo = []
        for lat, lon in tiles:
            dest = out / f"s2_composite_{tile_name(lat, lon)}.tif"
            if dry_run:
                report.planned.append(dest.name)
            elif dest.exists():
                report.skipped.append(dest)
            else:
                todo.append((lat, lon, dest))

        lock = threading.Lock()

        def run(lat: int, lon: int, dest: Path) -> None:
            n, error = 0, None
            for attempt in range(1, 4):
                try:
                    n = composite_tile(src.stac, tile_bbox(lat, lon), dest)
                    error = None
                    break
                except (OSError, RuntimeError, ValueError) as e:  # сеть, S3, GDAL
                    error = e
                    time.sleep(10 * attempt)
            with lock:
                if error is not None:
                    report.missing.append(dest.name)
                    print(f"  {dest.name}: не удалось после 3 попыток ({error}); "
                          "повторный запуск дособерёт", flush=True)
                    return
                if n:
                    report.downloaded.append(dest)
                    manifest.record(src.id, dest, f"{src.stac.endpoint}#{src.stac.collection}")
                    manifest.save()
                else:
                    report.missing.append(dest.name)
                print(f"  [{len(report.downloaded) + len(report.missing)}/{len(todo)}] "
                      f"{dest.name}: {n} сцен", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(run, *t) for t in todo]:
                future.result()

    return report


def land_tiles(
    tiles: list[tuple[int, int]], data_root: Path, catalog: Catalog, step: int = 1
) -> list[tuple[int, int]]:
    """Оставляет тайлы, пересекающие область: aoi.countries + aoi.extra_countries в рамке сетки.

    Используется та же функция области, что и при подготовке слоёв, поэтому тайлы
    скачиваются ровно там, где потом будет маска обучения.
    """
    import geopandas as gpd
    from shapely.geometry import box

    from jerqazyna.config import CENTRAL_ASIA_ALBERS
    from jerqazyna.grid.area import BoundaryMissing, region

    countries = catalog.aoi.get("countries") or []
    extra = catalog.aoi.get("extra_countries") or []
    if not countries:
        return tiles
    try:
        area, _ = region(
            Path(data_root) / "raw" / "geoboundaries", countries, extra, CENTRAL_ASIA_ALBERS, 1000
        )
    except BoundaryMissing:
        print("  границы стран не скачаны (geoboundaries): беру все тайлы охвата")
        return tiles
    land = gpd.GeoSeries([area], crs=CENTRAL_ASIA_ALBERS).to_crs("EPSG:4326").iloc[0].buffer(0.05)
    kept = [t for t in tiles if land.intersects(box(*tile_bbox(*t, step)))]
    print(f"  тайлов в охвате: {len(tiles)}, в области ({', '.join(countries + extra)}): {len(kept)}")
    return kept


def present_files(data_root: Path, src: Source) -> list[Path]:
    out = source_dir(data_root, src)
    if not out.exists():
        return []
    found: set[Path] = set()
    patterns = src.expect or ["*"]
    for pattern in patterns:
        found.update(p for p in out.rglob(pattern) if p.is_file() and not p.name.endswith(".part"))
    return sorted(found)


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6

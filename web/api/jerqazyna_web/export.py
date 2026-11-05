"""Экспорт слоёв реестра в Cloud Optimized GeoTIFF для TiTiler и API.

COG пишется на исходной сетке задачи (Albers, 1 км) без перепроецирования: TiTiler
проецирует тайлы сам, а индексы ячеек в API совпадают с индексами обучающих таблиц.
Сжатие без потерь; после записи значения сверяются с исходником. Манифест хранит
sha256 исходника и результата; при совпадении рецепта и хеша COG слой не пересчитывается.

Запуск: python web/scripts/export_cogs.py [--only dem_elevation ...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jerqazyna.data.catalog import load_catalog
from jerqazyna.data.download import sha256_of
from jerqazyna.features.cache import fingerprint, write_json
from jerqazyna.grid.template import GridSpec
from jerqazyna_web.manifest import (
    COG_DIR,
    MANIFEST_NAME,
    CogEntry,
    Manifest,
    load_manifest,
    recorded_hash,
)
from jerqazyna_web.registry import Provenance, Registry, load_registry
from jerqazyna_web.settings import PROJECT_ROOT

EXPORT_VERSION = 1
BLOCKSIZE = 512
AREA_FILE = "area_mask.tif"   # вне cogs/: TiTiler маску области не видит


class ExportError(RuntimeError):
    pass


def cog_options(resampling: str, blocksize: int) -> dict[str, str]:
    return {
        "COMPRESS": "DEFLATE", "PREDICTOR": "YES", "BLOCKSIZE": str(blocksize),
        "OVERVIEWS": "IGNORE_EXISTING", "OVERVIEW_RESAMPLING": resampling.upper(),
        "BIGTIFF": "IF_SAFER",
    }


def _read_source(path: Path, band: int, grid: GridSpec, name: str):
    import rasterio
    from rasterio.crs import CRS

    if not path.is_file():
        raise ExportError(f"{name}: нет исходного растра {path}")
    with rasterio.open(path) as src:
        if band > src.count:
            raise ExportError(f"{name}: в {path.name} {src.count} канал(ов), запрошен {band}")
        if (src.shape != grid.shape or src.crs != CRS.from_user_input(grid.crs)
                or not src.transform.almost_equals(grid.transform)):
            raise ExportError(f"{name}: {path.name} не на сетке задачи (форма {src.shape}, "
                              f"сетка {grid.shape})")
        return src.read(band), src.nodata, src.descriptions[band - 1] or ""


def _valid(array: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(array) if np.issubdtype(array.dtype, np.floating) else np.ones(
        array.shape, bool)
    if nodata is not None and not np.isnan(nodata):
        valid &= array != nodata
    return valid


def _write_cog(dst: Path, array: np.ndarray, nodata, description: str, grid: GridSpec,
               tags: dict[str, str], resampling: str, blocksize: int) -> None:
    import rasterio
    import rasterio.shutil
    from rasterio.io import MemoryFile

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.tif")
    profile = {
        "driver": "GTiff", "width": grid.width, "height": grid.height, "count": 1,
        "dtype": array.dtype.name, "crs": grid.crs, "transform": grid.transform, "nodata": nodata,
    }
    try:
        with MemoryFile() as memory, memory.open(**profile) as staged:
            staged.write(array, 1)
            if description:
                staged.set_band_description(1, description)
            staged.update_tags(**tags)
            rasterio.shutil.copy(staged, tmp, driver="COG", **cog_options(resampling, blocksize))
        _verify_cog(tmp, array, grid, blocksize)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def _verify_cog(path: Path, expected: np.ndarray, grid: GridSpec, blocksize: int) -> None:
    import rasterio
    from rasterio.crs import CRS

    with rasterio.open(path) as cog:
        if cog.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
            raise ExportError(f"{path.name}: GDAL записал не COG")
        if max(grid.shape) > blocksize and not cog.overviews(1):
            raise ExportError(f"{path.name}: в COG нет обзоров")
        if cog.crs != CRS.from_user_input(grid.crs) or not cog.transform.almost_equals(
                grid.transform):
            raise ExportError(f"{path.name}: привязка COG отличается от сетки")
        if not np.array_equal(cog.read(1), expected, equal_nan=expected.dtype.kind == "f"):
            raise ExportError(f"{path.name}: значения COG отличаются от исходника")


def export_raster(
    name: str, source: str, band: int, provenance: Provenance | None, resampling: str,
    rel_file: str, root: Path, out_dir: Path, grid: GridSpec, previous: CogEntry | None,
    area: np.ndarray | None, blocksize: int, log=print,
) -> tuple[CogEntry, np.ndarray]:
    """Один растр в COG; возвращает запись манифеста и прочитанный массив."""
    import rasterio

    src_path = root / source
    array, nodata, description = _read_source(src_path, band, grid, name)
    source_sha = sha256_of(src_path)
    prov = None
    if provenance is not None:
        try:
            prov = recorded_hash(provenance, root, src_path.name)
        except ValueError as e:
            raise ExportError(f"{name}: {e}") from e
        if prov.recorded_sha256 != source_sha:
            raise ExportError(
                f"{name}: {source} не совпадает с хешем в {provenance.manifest}; "
                "исходник изменён после подготовки — пересоберите слой")
    recipe = fingerprint({
        "version": EXPORT_VERSION, "source_sha256": source_sha, "band": band,
        "options": cog_options(resampling, blocksize),
        "packages": {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__},
    })
    dst = out_dir / rel_file
    if (previous is not None and previous.recipe == recipe and dst.is_file()
            and sha256_of(dst) == previous.cog_sha256):
        log(f"  {name}: без изменений, {rel_file}")
        return previous, array
    log(f"  {name}: {source} -> {rel_file}")
    tags = {"jerqazyna_layer": name, "source": source, "source_sha256": source_sha}
    _write_cog(dst, array, nodata, description, grid, tags, resampling, blocksize)
    valid = _valid(array, nodata)
    coverage = None if area is None else round(float(valid[area == 1].mean()), 6)
    entry = CogEntry(
        file=rel_file, source=source, band=band, source_sha256=source_sha, provenance=prov,
        cog_sha256=sha256_of(dst), recipe=recipe, dtype=array.dtype.name,
        nodata=None if nodata is None or np.isnan(nodata) else float(nodata),
        coverage=coverage, exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return entry, array


def export(registry: Registry, root: Path, out_dir: Path, only: list[str] | None = None,
           blocksize: int = BLOCKSIZE, log=print) -> Manifest:
    import rasterio

    grid = GridSpec.load(root / registry.grid)
    enabled = {layer.id: layer for layer in registry.enabled_layers}
    unknown = sorted(set(only or []) - set(enabled))
    if unknown:
        raise ExportError(f"Нет включённых слоёв: {', '.join(unknown)}; есть: {', '.join(enabled)}")
    selected = [enabled[i] for i in only] if only else list(enabled.values())
    manifest_path = out_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    if manifest is None:
        manifest = Manifest(task_id=registry.task_id, grid=asdict(grid))
    elif manifest.task_id != registry.task_id or manifest.grid != asdict(grid):
        raise ExportError(f"{manifest_path}: экспорт для другой задачи или сетки; "
                          "укажите другой каталог --out")
    manifest.packages = {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__}
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Экспорт COG: {registry.task_id} -> {out_dir}")
    manifest.area, area = export_raster(
        "area", registry.area.file, 1, registry.area.provenance, "nearest", AREA_FILE,
        root, out_dir, grid, manifest.area, None, blocksize, log)
    if not np.isin(area, (0, 1)).all():
        raise ExportError("Маска области должна содержать только 0 и 1")
    write_json(manifest_path, manifest.model_dump(mode="json"))
    for layer in selected:
        entry, _ = export_raster(
            layer.id, layer.file, layer.band, layer.provenance, layer.display.resampling,
            f"{COG_DIR}/{layer.id}.tif", root, out_dir, grid, manifest.layers.get(layer.id),
            area, blocksize, log)
        manifest.layers[layer.id] = entry
        write_json(manifest_path, manifest.model_dump(mode="json"))
    log(f"Готово: {len(selected)} слоёв, манифест {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Экспорт слоёв реестра веб-карты в COG")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "web" / "layers.yaml")
    parser.add_argument("--sources", type=Path, default=PROJECT_ROOT / "configs" / "sources.yaml")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="корень проекта")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "web")
    parser.add_argument("--only", action="append", metavar="LAYER_ID",
                        help="экспортировать только эти слои (можно несколько раз)")
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry, load_catalog(args.sources))
        export(registry, args.root.resolve(), args.out.resolve(), args.only)
    except (ExportError, ValueError, OSError) as e:
        print(f"Ошибка экспорта: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

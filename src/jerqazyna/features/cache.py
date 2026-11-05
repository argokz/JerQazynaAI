"""Проверяемые отпечатки входов и атомарные записи метаданных подготовки."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from jerqazyna.config import LayerConfig, TaskConfig
from jerqazyna.data.download import sha256_of
from jerqazyna.grid.template import GridSpec

CACHE_VERSION = 1


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    """Метаданные выступают маркером завершения; заменяем файл только после записи."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Digests:
    """SHA-256 реального содержимого; повторное чтение одного файла в запуске не нужно.

    stat используется лишь для мемоизации внутри запуска. Между запусками хеши
    пересчитываются: манифест загрузки не доказывает неизменность локального файла.
    """

    def __init__(self):
        self._hashes: dict[tuple, str] = {}

    def file(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._hashes:
            digest = sha256_of(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[1:]:
                raise OSError(f"Файл изменился во время чтения: {path}")
            self._hashes[key] = digest
        return self._hashes[key]

    def snapshot(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"Нет источника: {path}")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        files = [p for p in files if ".part" not in p.name and not p.name.endswith(".lock")]
        if not files:
            raise FileNotFoundError(f"Нет завершённых файлов в {path}")
        return [
            {"path": p.name if path.is_file() else p.relative_to(path).as_posix(),
             "sha256": self.file(p)}
            for p in files
        ]


def implementation_fingerprint(digests: Digests) -> str:
    """Изменения обработчиков/помощников и версий библиотек инвалидируют расчёт."""
    package = Path(__file__).parents[1]
    paths = [
        "config.py", "features/cache.py", "features/layers.py", "features/prepare.py",
        "features/raster_ops.py", "grid/area.py", "grid/template.py",
    ]
    return fingerprint({
        "code": {p: digests.file(package / p) for p in paths},
        "packages": {p: version(p) for p in (
            "numpy", "scipy", "rasterio", "pyproj", "geopandas", "shapely", "pandas", "pyogrio",
        )},
    })


def grid_recipe(task: TaskConfig, root: Path, digests: Digests) -> dict:
    boundaries = root / "data" / "raw" / "geoboundaries"
    countries = task.training_countries + task.extra_countries
    return {
        "version": CACHE_VERSION,
        "core": task.training_countries, "extra": task.extra_countries,
        "grid": task.grid.model_dump(),
        "boundaries": {
            iso: digests.file(boundaries / f"geoBoundaries-{iso}-ADM0.geojson") for iso in countries
        },
        "mask_algorithm": "polygon_fraction_factor4_majority_v1",
        "packages": {p: version(p) for p in ("rasterio", "pyproj", "geopandas", "shapely")},
        "code": {
            p: digests.file(Path(__file__).parents[1] / p)
            for p in ("grid/area.py", "grid/template.py", "features/raster_ops.py")
        },
    }


def dependencies(layer: LayerConfig) -> list[str]:
    return ["dem"] if layer.kind == "ggmplus_bouguer" else []


def layer_recipe(
    layer: LayerConfig, task: TaskConfig, root: Path, grid: GridSpec,
    area_hash: str, implementation: str, digests: Digests, cache: dict,
) -> str:
    deps = {}
    for name in dependencies(layer):
        if name not in cache:
            raise FileNotFoundError(f"{layer.name}: зависимость {name} не подготовлена")
        deps[name] = cache[name]
    return fingerprint({
        "version": CACHE_VERSION, "layer": layer.model_dump(), "grid": asdict(grid),
        "area": area_hash, "implementation": implementation,
        "sources": digests.snapshot(root / layer.source), "dependencies": deps,
        "mine_buffer_m": task.labels.mine_mask_buffer_m if layer.kind == "mine_mask" else None,
    })


def raster_matches(path: Path, grid: GridSpec) -> bool:
    import rasterio

    try:
        with rasterio.open(path) as src:
            return (src.count == 1 and src.shape == grid.shape
                    and src.crs == rasterio.crs.CRS.from_user_input(grid.crs)
                    and src.transform.almost_equals(grid.transform))
    except (OSError, ValueError):
        return False


def layer_is_current(
    out: Path, layer: LayerConfig, grid: GridSpec, recipe: str,
    entry: dict, inventory: dict, digests: Digests,
) -> bool:
    files = entry.get("files", {})
    actual = {p.name for p in out.glob(f"{layer.name}__*.tif")}
    if entry.get("recipe") != recipe or not files or actual != set(files):
        return False
    listed = {name for name, r in inventory.items() if r["layer"] == layer.name}
    if listed != set(files):
        return False
    for name, digest in files.items():
        path = out / name
        if (not raster_matches(path, grid) or digests.file(path) != digest
                or inventory[name].get("recipe") != recipe):
            return False
    return True

"""Подготовка задачи: сетка, маска области обучения, метки и все слои на сетке.

Результат лежит в data/features/<task_id>/:
  grid.json, area_mask.tif      — сетка и область обучения
  labels.csv                    — известные объекты с ячейками сетки
  <слой>__<канал>.tif           — слои, выровненные по сетке (float32, NaN = нет данных)
  layers.json                   — опись: роль, группа, доля покрытия внутри области обучения

Маска выработок (role: mask) хранится отдельно и применяется к спектральным признакам
при сборке таблицы для модели, чтобы исходные слои оставались без изменений.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jerqazyna.config import TaskConfig
from jerqazyna.features.layers import PROCESSORS, Context, SourceMissing
from jerqazyna.features.raster_ops import polygon_fraction
from jerqazyna.grid.template import GridSpec, read_layer, write_layer


def features_dir(root: Path, task: TaskConfig) -> Path:
    return Path(root) / "data" / "features" / task.task_id


def training_area(root: Path, task: TaskConfig):
    """Область обучения в CRS сетки и сама сетка (по основным странам)."""
    from jerqazyna.grid.area import BoundaryMissing, region

    try:
        return region(
            Path(root) / "data" / "raw" / "geoboundaries",
            task.training_countries, task.extra_countries, task.grid.crs, task.grid.cell_size_m,
        )
    except BoundaryMissing as e:
        raise SourceMissing(str(e)) from e


def build_grid(root: Path, task: TaskConfig, out: Path, log=print) -> tuple[GridSpec, np.ndarray]:
    """Сетка и маска области. Маска пересчитывается, если изменился список стран."""
    grid_path, mask_path, area_path = out / "grid.json", out / "area_mask.tif", out / "area.json"
    spec = {"core": task.training_countries, "extra": task.extra_countries}
    cached = grid_path.exists() and mask_path.exists() and area_path.exists()
    if cached and json.loads(area_path.read_text(encoding="utf-8")) == spec:
        return GridSpec.load(grid_path), read_layer(mask_path).astype(bool)
    area, grid = training_area(root, task)
    if grid_path.exists() and GridSpec.load(grid_path) != grid:
        raise ValueError(
            "Основные страны изменились, и сетка стала другой: удалите "
            f"{out} и подготовьте все слои заново"
        )
    log(f"  сетка {grid.width}×{grid.height} ячеек по {grid.cell_size:.0f} м")
    mask = polygon_fraction([area], grid, factor=4) >= 0.5
    out.mkdir(parents=True, exist_ok=True)
    grid.save(grid_path)
    write_layer(mask_path, grid, mask.astype(np.uint8), "область обучения")
    area_path.write_text(json.dumps(spec), encoding="utf-8")
    countries = ", ".join(task.training_countries + task.extra_countries)
    log(f"  область обучения ({countries}): {mask.sum():,} ячеек "
        f"({mask.sum() * (grid.cell_size / 1000) ** 2:,.0f} км²)")
    return grid, mask


def _points_from_table(path: Path, cfg):
    """Точки из CSV/XLSX: колонки координат из конфига или найденные по имени (lat/lon)."""
    import geopandas as gpd
    import pandas as pd

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    elif cfg.encoding:
        df = pd.read_csv(path, low_memory=False, encoding=cfg.encoding)
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    def pick(configured: str | None, candidates: tuple[str, ...]) -> str:
        if configured:
            return configured
        lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        raise ValueError(f"{path.name}: не нашёл колонку координат {candidates}; есть: {list(df.columns)}")

    lon = pick(cfg.lon_field, ("longitude", "lon", "long", "x", "lon_dd", "longitude_dd"))
    lat = pick(cfg.lat_field, ("latitude", "lat", "y", "lat_dd", "latitude_dd"))
    df = df.assign(**{lon: pd.to_numeric(df[lon], errors="coerce"),
                      lat: pd.to_numeric(df[lat], errors="coerce")})
    df = df.dropna(subset=[lon, lat])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")


def load_label_source(root: Path, cfg, grid: GridSpec):
    """Один источник меток → GeoDataFrame в CRS сетки с колонками name, status, source, extra."""
    import geopandas as gpd
    import pandas as pd

    src = Path(root) / cfg.source
    if src.is_dir() and src.suffix.lower() != ".gdb":  # .gdb — база ESRI, а не папка с файлами
        found = sorted(p for ext in ("*.csv", "*.xlsx", "*.gpkg") for p in src.rglob(ext))
        if not found:
            raise SourceMissing(f"в {src} нет файлов меток (csv, xlsx, gpkg)")
        src = found[0]
    if not src.exists():
        raise SourceMissing(f"нет источника меток {src}")
    if src.suffix.lower() in (".csv", ".xlsx"):
        gdf = _points_from_table(src, cfg)
    else:
        gdf = gpd.read_file(src, layer=cfg.layer)

    fields = [cfg.name_field, cfg.status_field, cfg.type_field, *cfg.extra_fields.values()]
    missing = [f for f in fields if f and f not in gdf.columns]
    if missing:
        raise ValueError(f"{src.name}: нет полей {missing}; есть: {list(gdf.columns)}")
    if cfg.type_field and cfg.type_regex:
        gdf = gdf[gdf[cfg.type_field].astype(str).str.contains(cfg.type_regex, regex=True)]

    gdf = gdf.to_crs(grid.crs)
    x0, y0, x1, y1 = grid.bounds
    gdf = gdf[(gdf.geometry.x >= x0) & (gdf.geometry.x < x1) & (gdf.geometry.y > y0) & (gdf.geometry.y <= y1)]
    out = gpd.GeoDataFrame({
        "name": gdf[cfg.name_field].astype(str).str.strip().to_numpy(),
        "status": (gdf[cfg.status_field].astype(str).str.lower().to_numpy()
                   if cfg.status_field else np.full(len(gdf), "")),
        "source": src.name,
    }, geometry=gdf.geometry.to_numpy(), crs=grid.crs)
    for name, field in cfg.extra_fields.items():
        values = gdf[field]
        numeric = pd.to_numeric(values, errors="coerce")
        # Числовая колонка, если большинство непустых значений — числа (например, «1.1-3.2» → NaN)
        out[name] = (numeric if numeric.notna().sum() >= 0.5 * values.notna().sum() else values).to_numpy()
    return out


def merge_labels(primary, extra, radius_m: float):
    """Добавляет объекты extra дальше radius_m от primary; у близких берёт статус, если его нет."""
    import pandas as pd

    if primary.empty:
        return extra.copy(), 0, len(extra)
    idx, dist = primary.sindex.nearest(extra.geometry, return_distance=True, return_all=False)
    nearest = np.full(len(extra), -1)
    d = np.full(len(extra), np.inf)
    nearest[idx[0]] = idx[1]
    d[idx[0]] = dist
    same = d <= radius_m
    merged = primary.copy()
    for i in np.flatnonzero(same):
        j = merged.index[nearest[i]]
        if not merged.at[j, "status"] and extra.iloc[i]["status"]:
            merged.at[j, "status"] = extra.iloc[i]["status"]
    added = extra[~same]
    merged = pd.concat([merged, added], ignore_index=True)
    return merged, int(same.sum()), int((~same).sum())


def build_labels(root: Path, task: TaskConfig, grid: GridSpec, mask: np.ndarray, out: Path, log=print):
    cfg = task.labels
    table = load_label_source(root, cfg, grid)
    log(f"  {Path(cfg.source).name}: {len(table)} объектов в охвате сетки")
    for extra_cfg in cfg.merge:
        try:
            extra = load_label_source(root, extra_cfg, grid)
        except SourceMissing as e:
            log(f"  пропуск дополнительного источника: {e}")
            continue
        table, same, added = merge_labels(table, extra, cfg.merge_radius_m)
        log(f"  + {extra['source'].iloc[0] if len(extra) else extra_cfg.source}: совпало {same}, добавлено {added}")
    if not cfg.include_prospects:
        table = table[table.status != "prospect"]

    xs, ys = table.geometry.x.to_numpy(), table.geometry.y.to_numpy()
    rows, cols = grid.rowcol(xs, ys)
    lonlat = table.to_crs("EPSG:4326").geometry
    in_area = mask[rows, cols]
    result = table.drop(columns="geometry").assign(
        lon=lonlat.x.round(5).to_numpy(), lat=lonlat.y.round(5).to_numpy(),
        x=np.round(xs, 1), y=np.round(ys, 1), row=rows, col=cols, in_area=in_area,
    )
    result.to_csv(out / "labels.csv", index=False, encoding="utf-8")
    shared = result[result.in_area].duplicated(["row", "col"]).sum()
    log(f"  меток: {len(result)}, в области обучения: {int(in_area.sum())}, "
        f"делят ячейку с другой меткой: {int(shared)}")
    return result


def _layer_files(out: Path, name: str) -> list[Path]:
    return sorted(out.glob(f"{name}__*.tif"))


def prepare(
    task: TaskConfig,
    root: Path,
    only: list[str] | None = None,
    force: bool = False,
    log=print,
) -> list[dict]:
    root = Path(root)
    out = features_dir(root, task)
    log(f"== сетка и область обучения ({task.task_id})")
    grid, mask = build_grid(root, task, out, log)
    log("== метки")
    try:
        build_labels(root, task, grid, mask, out, log)
    except SourceMissing as e:
        log(f"  пропуск: {e}")

    inventory_path = out / "layers.json"
    inventory = {}
    if inventory_path.exists():
        inventory = {r["file"]: r for r in json.loads(inventory_path.read_text(encoding="utf-8"))}

    ctx = Context(task=task, grid=grid, root=root, log=log)
    report = []
    for layer in task.layers:
        if only and layer.name not in only:
            continue
        log(f"== {layer.name} ({layer.kind}, {layer.role})")
        existing = _layer_files(out, layer.name)
        if existing and not force:
            log(f"  уже готово: {len(existing)} файлов (пересчитать: --force)")
            report.append({"layer": layer.name, "status": "готово ранее", "bands": len(existing)})
            continue
        if layer.kind not in PROCESSORS:
            raise ValueError(f"{layer.name}: неизвестный kind {layer.kind!r}")
        started = time.time()
        try:
            bands = PROCESSORS[layer.kind](layer, ctx)
        except SourceMissing as e:
            log(f"  нет данных: {e}")
            report.append({"layer": layer.name, "status": "нет данных", "bands": 0})
            continue
        for old in existing:
            old.unlink()
            inventory.pop(old.name, None)
        for band, array in bands.items():
            path = out / f"{layer.name}__{band}.tif"
            write_layer(path, grid, array, f"{layer.name}: {band}")
            valid = float(np.isfinite(array[mask]).mean()) if array.dtype != np.uint8 else 1.0
            inventory[path.name] = {
                "file": path.name, "layer": layer.name, "band": band, "role": layer.role,
                "group": layer.group, "kind": layer.kind, "source": layer.source,
                "valid_fraction_in_area": round(valid, 4),
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            log(f"  {path.name}: покрытие области {valid:.1%}")
        inventory_path.write_text(
            json.dumps(sorted(inventory.values(), key=lambda r: r["file"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report.append({
            "layer": layer.name, "status": "готово", "bands": len(bands),
            "seconds": round(time.time() - started),
        })
    return report

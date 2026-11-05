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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jerqazyna.config import TaskConfig
from jerqazyna.features.cache import (
    Digests,
    dependencies,
    fingerprint,
    grid_recipe,
    implementation_fingerprint,
    layer_is_current,
    layer_recipe,
    raster_matches,
    write_json,
)
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


def build_grid(
    root: Path, task: TaskConfig, out: Path, log=print, *, force: bool = False,
    digests: Digests | None = None,
) -> tuple[GridSpec, np.ndarray]:
    """Проверяет конфиг и границы до использования сохранённой сетки.

    Несовместимую геометрию нельзя перезаписать даже с --force: нужен новый task_id.
    При прежней геометрии force пересчитывает маску, сохраняя исходные данные.
    """
    from pyproj import CRS

    digests = digests or Digests()
    grid_path, mask_path, area_path = out / "grid.json", out / "area_mask.tif", out / "area.json"
    if grid_path.exists():
        saved = GridSpec.load(grid_path)
        if (saved.cell_size != task.grid.cell_size_m
                or not CRS(saved.crs).equals(CRS(task.grid.crs))):
            raise ValueError("CRS или размер ячейки изменились: используйте новый task_id. "
                             "--force не меняет геометрию существующей сетки")
    try:
        spec = grid_recipe(task, Path(root), digests)
    except FileNotFoundError as e:
        raise SourceMissing(str(e)) from e
    cached = grid_path.exists() and mask_path.exists() and area_path.exists()
    if cached and not force:
        metadata = json.loads(area_path.read_text(encoding="utf-8"))
        if (metadata.get("spec") == spec and raster_matches(mask_path, saved)
                and metadata.get("mask_sha256") == digests.file(mask_path)):
            return saved, read_layer(mask_path).astype(bool)
    area, grid = training_area(root, task)
    if grid_path.exists() and GridSpec.load(grid_path) != grid:
        raise ValueError(
            "Страны или их границы изменили геометрию сетки: используйте новый task_id"
        )
    log(f"  сетка {grid.width}×{grid.height} ячеек по {grid.cell_size:.0f} м")
    mask = polygon_fraction([area], grid, factor=4) >= 0.5
    if not mask.any():
        raise ValueError("Область обучения пуста")
    out.mkdir(parents=True, exist_ok=True)
    grid.save(grid_path)
    write_layer(mask_path, grid, mask.astype(np.uint8), "область обучения")
    write_json(area_path, {"spec": spec, "mask_sha256": digests.file(mask_path)})
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


def normalize_status(value) -> str:
    import pandas as pd

    if value is None or pd.isna(value):
        return ""
    value = str(value).strip().lower()
    if value in {"nan", "unknown", "none", "n/a"}:
        return ""
    if value in {"prospect, significant", "significant prospect", "prospect significant"}:
        return "prospect"
    return value


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
        "status": (gdf[cfg.status_field].map(normalize_status).to_numpy()
                   if cfg.status_field else np.full(len(gdf), "")),
        "source": src.name,
    }, geometry=gdf.geometry.to_numpy(), crs=grid.crs)
    out["status_raw"] = (gdf[cfg.status_field].fillna("").astype(str).to_numpy()
                         if cfg.status_field else "")
    out["status_source"] = np.where(out.status != "", cfg.source, "")
    out["aliases"] = [json.dumps([name], ensure_ascii=False) for name in out.name]
    out["source_records"] = [json.dumps([{
        "source": cfg.source, "layer": cfg.layer, "row": str(index),
        "name": name, "status_raw": raw,
    }], ensure_ascii=False) for index, name, raw in zip(gdf.index, out.name, out.status_raw)]
    for name, field in cfg.extra_fields.items():
        values = gdf[field]
        numeric = pd.to_numeric(values, errors="coerce")
        # Числовая колонка, если большинство непустых значений — числа (например, «1.1-3.2» → NaN)
        out[name] = (numeric if numeric.notna().sum() >= 0.5 * values.notna().sum() else values).to_numpy()
    return out


def merge_labels(primary, extra, radius_m: float, *, add_unmatched: bool = True):
    """Добавляет объекты extra дальше radius_m от primary; у близких берёт статус, если его нет."""
    import pandas as pd

    if primary.empty:
        return (extra.copy(), 0, len(extra)) if add_unmatched else (primary.copy(), 0, 0)
    if extra.empty:
        return primary.copy(), 0, 0
    idx, dist = primary.sindex.nearest(extra.geometry, return_distance=True, return_all=False)
    nearest = np.full(len(extra), -1)
    d = np.full(len(extra), np.inf)
    nearest[idx[0]] = idx[1]
    d[idx[0]] = dist
    same = d <= radius_m
    merged = primary.copy()
    for col in ("status_raw", "status_source", "aliases", "source_records"):
        if col not in merged:
            merged[col] = "[]" if col in {"aliases", "source_records"} else ""
    merged["status"] = merged.status.map(normalize_status)
    for i in np.flatnonzero(same):
        j = merged.index[nearest[i]]
        row = extra.iloc[i]
        status = normalize_status(row["status"])
        if not merged.at[j, "status"] and status:
            merged.at[j, "status"] = status
            merged.at[j, "status_raw"] = row.get("status_raw", row["status"])
            merged.at[j, "status_source"] = row.get("status_source", row.get("source", ""))
        aliases = json.loads(merged.at[j, "aliases"])
        merged.at[j, "aliases"] = json.dumps(
            list(dict.fromkeys([*aliases, merged.at[j, "name"], row["name"]])), ensure_ascii=False,
        )
        records = json.loads(merged.at[j, "source_records"])
        records.extend(json.loads(row.get("source_records", "[]")))
        merged.at[j, "source_records"] = json.dumps(records, ensure_ascii=False)
    added = extra[~same] if add_unmatched else extra.iloc[:0]
    merged = pd.concat([merged, added], ignore_index=True)
    return merged, int(same.sum()), len(added)


def build_labels(
    root: Path, task: TaskConfig, grid: GridSpec, mask: np.ndarray, out: Path, log=print,
    *, digests: Digests | None = None,
):
    cfg = task.labels
    digests = digests or Digests()
    if any((Path(root) / source.source).resolve() == (out / "labels.csv").resolve()
           for source in [cfg, *cfg.merge]):
        raise ValueError("Исходные метки не должны совпадать с выходным labels.csv")
    (out / "labels.json").unlink(missing_ok=True)

    def inputs():
        result = {}
        for source in [cfg, *cfg.merge]:
            try:
                result[source.source] = digests.snapshot(Path(root) / source.source)
            except FileNotFoundError as e:
                if source.required:
                    raise SourceMissing(str(e)) from e
                result[source.source] = None
        return result

    initial_inputs = inputs()
    table = load_label_source(root, cfg, grid)
    log(f"  {Path(cfg.source).name}: {len(table)} объектов в охвате сетки")
    for extra_cfg in cfg.merge:
        try:
            extra = load_label_source(root, extra_cfg, grid)
        except SourceMissing as e:
            if extra_cfg.required:
                raise
            log(f"  пропуск дополнительного источника: {e}")
            continue
        radius = cfg.merge_radius_m if extra_cfg.match_radius_m is None else extra_cfg.match_radius_m
        table, same, added = merge_labels(table, extra, radius, add_unmatched=extra_cfg.add_unmatched)
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
    if inputs() != initial_inputs:
        raise OSError("Источники меток изменились во время подготовки")
    tmp = out / "labels.csv.part"
    result.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(out / "labels.csv")
    write_json(out / "labels.json", {
        "version": 1, "config": cfg.model_dump(), "grid": asdict(grid),
        "inputs": initial_inputs, "csv_sha256": digests.file(out / "labels.csv"),
        "n_objects": len(result), "n_in_area": int(in_area.sum()),
    })
    shared = result[result.in_area].duplicated(["row", "col"]).sum()
    log(f"  меток: {len(result)}, в области обучения: {int(in_area.sum())}, "
        f"делят ячейку с другой меткой: {int(shared)}")
    return result


def _layer_files(out: Path, name: str) -> list[Path]:
    return sorted(out.glob(f"{name}__*.tif"))


def prepare_labels(task: TaskConfig, root: Path, log=print):
    """Подготовка только сетки, области и меток, без пересчёта тяжёлых растров."""
    root = Path(root)
    out = features_dir(root, task)
    digests = Digests()
    grid, mask = build_grid(root, task, out, log, digests=digests)
    try:
        return build_labels(root, task, grid, mask, out, log, digests=digests)
    except (OSError, ValueError):
        (out / "labels.csv").unlink(missing_ok=True)
        (out / "labels.json").unlink(missing_ok=True)
        raise


def _selected_layers(task: TaskConfig, only: list[str] | None):
    by_name = {layer.name: layer for layer in task.layers}
    if only is not None and (unknown := set(only) - by_name.keys()):
        raise ValueError(f"Неизвестные слои: {', '.join(sorted(unknown))}")
    ordered = []
    seen = set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        for dep in dependencies(by_name[name]):
            visit(dep)
        ordered.append(by_name[name])

    for name in only if only is not None else by_name:
        visit(name)
    return ordered


def prepare(
    task: TaskConfig,
    root: Path,
    only: list[str] | None = None,
    force: bool = False,
    log=print,
) -> list[dict]:
    root = Path(root)
    selected = _selected_layers(task, only)
    out = features_dir(root, task)
    digests = Digests()
    log(f"== сетка и область обучения ({task.task_id})")
    grid, mask = build_grid(root, task, out, log, force=force, digests=digests)
    log("== метки")
    try:
        build_labels(root, task, grid, mask, out, log, digests=digests)
    except (OSError, ValueError) as e:
        (out / "labels.csv").unlink(missing_ok=True)
        (out / "labels.json").unlink(missing_ok=True)
        raise SourceMissing(f"Обязательные метки не подготовлены: {e}") from e

    inventory_path = out / "layers.json"
    inventory = {}
    if inventory_path.exists():
        inventory = {r["file"]: r for r in json.loads(inventory_path.read_text(encoding="utf-8"))}

    cache_path = out / "cache.json"
    cache = {}
    if cache_path.exists():
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        if stored.get("version") == 1:
            cache = stored.get("layers", {})

    def save_state():
        write_json(inventory_path, sorted(inventory.values(), key=lambda r: r["file"]))
        write_json(cache_path, {"version": 1, "layers": cache})

    # Удалённые из конфига слои не должны оставаться в описи разрешённых входов.
    names = {layer.name for layer in task.layers}
    inventory = {k: v for k, v in inventory.items() if v["layer"] in names}
    cache = {k: v for k, v in cache.items() if k in names}
    implementation = implementation_fingerprint(digests)
    area_hash = fingerprint(json.loads((out / "area.json").read_text(encoding="utf-8")))
    context_hash = fingerprint({"implementation": implementation, "area": area_hash})

    def invalidate(name):
        cache.pop(name, None)
        for filename in [k for k, r in inventory.items() if r["layer"] == name]:
            inventory.pop(filename)
        for dependent in task.layers:
            if name in dependencies(dependent):
                invalidate(dependent.name)

    for layer in task.layers:
        entry = cache.get(layer.name, {})
        if (entry.get("context") != context_hash
                or entry.get("config") != layer.model_dump()):
            invalidate(layer.name)
    # Даже при --layers нельзя оставлять устаревшие невыбранные слои в описи готовых.
    for layer in _selected_layers(task, None):
        if layer.name not in cache:
            continue
        try:
            expected = layer_recipe(
                layer, task, root, grid, area_hash, implementation, digests, cache,
            )
            current = layer_is_current(
                out, layer, grid, expected, cache[layer.name], inventory, digests,
            )
        except FileNotFoundError:
            current = False
        if not current:
            invalidate(layer.name)
    save_state()

    ctx = Context(task=task, grid=grid, root=root, log=log)
    report = []
    for layer in selected:
        log(f"== {layer.name} ({layer.kind}, {layer.role})")
        existing = _layer_files(out, layer.name)
        try:
            recipe = layer_recipe(
                layer, task, root, grid, area_hash, implementation, digests, cache,
            )
        except FileNotFoundError as e:
            invalidate(layer.name)
            save_state()
            log(f"  нет данных: {e}")
            report.append({"layer": layer.name, "status": "нет данных", "bands": 0,
                           "required": layer.required})
            continue
        if not force and layer_is_current(
            out, layer, grid, recipe, cache.get(layer.name, {}), inventory, digests,
        ):
            log(f"  уже готово: {len(existing)} файлов (пересчитать: --force)")
            report.append({"layer": layer.name, "status": "готово ранее", "bands": len(existing)})
            continue
        if layer.kind not in PROCESSORS:
            raise ValueError(f"{layer.name}: неизвестный kind {layer.kind!r}")
        started = time.time()
        # Отзыв маркера готовности происходит до процессора и записи каналов.
        invalidate(layer.name)
        save_state()
        try:
            bands = PROCESSORS[layer.kind](layer, ctx)
        except SourceMissing as e:
            log(f"  нет данных: {e}")
            report.append({"layer": layer.name, "status": "нет данных", "bands": 0,
                           "required": layer.required})
            continue
        if not bands or any(not band or not all(c.isalnum() or c == "_" for c in band)
                            for band in bands):
            raise ValueError(f"{layer.name}: процессор вернул пустые или некорректные каналы")
        if layer_recipe(layer, task, root, grid, area_hash, implementation, digests, cache) != recipe:
            raise OSError(f"{layer.name}: входы изменились во время расчёта; повторите подготовку")
        files = {}
        for band, array in bands.items():
            path = out / f"{layer.name}__{band}.tif"
            write_layer(path, grid, array, f"{layer.name}: {band}")
            valid = float(np.isfinite(array[mask]).mean()) if array.dtype != np.uint8 else 1.0
            inventory[path.name] = {
                "file": path.name, "layer": layer.name, "band": band, "role": layer.role,
                "group": layer.group, "kind": layer.kind, "source": layer.source,
                "valid_fraction_in_area": round(valid, 4),
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "recipe": recipe,
            }
            files[path.name] = digests.file(path)
            log(f"  {path.name}: покрытие области {valid:.1%}")
        for old in existing:
            if old.name not in files:
                old.unlink(missing_ok=True)
        cache[layer.name] = {"recipe": recipe, "files": files,
                             "config": layer.model_dump(), "context": context_hash}
        save_state()
        report.append({
            "layer": layer.name, "status": "готово", "bands": len(bands),
            "seconds": round(time.time() - started),
        })
    return report

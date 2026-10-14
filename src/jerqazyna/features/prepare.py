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
    """Объединение границ стран из geoboundaries в CRS сетки."""
    import geopandas as gpd
    import pandas as pd

    parts = []
    for iso in task.training_countries:
        path = Path(root) / "data" / "raw" / "geoboundaries" / f"geoBoundaries-{iso}-ADM0.geojson"
        if not path.exists():
            raise SourceMissing(f"нет границы {iso}: jerqazyna data fetch geoboundaries")
        parts.append(gpd.read_file(path))
    area = pd.concat(parts).to_crs(task.grid.crs)
    return area.geometry.union_all()


def build_grid(root: Path, task: TaskConfig, out: Path, log=print) -> tuple[GridSpec, np.ndarray]:
    grid_path, mask_path = out / "grid.json", out / "area_mask.tif"
    if grid_path.exists() and mask_path.exists():
        grid = GridSpec.load(grid_path)
        return grid, read_layer(mask_path).astype(bool)
    area = training_area(root, task)
    grid = GridSpec.from_bounds(task.grid.crs, task.grid.cell_size_m, area.bounds)
    log(f"  сетка {grid.width}×{grid.height} ячеек по {grid.cell_size:.0f} м")
    mask = polygon_fraction([area], grid, factor=4) >= 0.5
    out.mkdir(parents=True, exist_ok=True)
    grid.save(grid_path)
    write_layer(mask_path, grid, mask.astype(np.uint8), "область обучения")
    log(f"  область обучения: {mask.sum():,} ячеек ({mask.sum() * (grid.cell_size / 1000) ** 2:,.0f} км²)")
    return grid, mask


def build_labels(root: Path, task: TaskConfig, grid: GridSpec, mask: np.ndarray, out: Path, log=print):
    import geopandas as gpd

    cfg = task.labels
    src = Path(root) / cfg.source
    if not src.exists():
        raise SourceMissing(f"нет источника меток {src}")
    gdf = gpd.read_file(src, layer=cfg.layer)
    if not cfg.include_prospects:
        gdf = gdf[gdf[cfg.status_field].str.lower() != "prospect"]
    lonlat = gdf.to_crs("EPSG:4326").geometry
    xy = gdf.to_crs(grid.crs).geometry
    rows, cols = grid.rowcol(xy.x.to_numpy(), xy.y.to_numpy())
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    in_area = np.zeros(len(gdf), bool)
    in_area[inside] = mask[rows[inside], cols[inside]]
    table = gdf[[cfg.name_field, cfg.status_field]].rename(
        columns={cfg.name_field: "name", cfg.status_field: "status"}
    )
    table = table.assign(
        lon=lonlat.x.round(5).to_numpy(), lat=lonlat.y.round(5).to_numpy(),
        x=xy.x.round(1).to_numpy(), y=xy.y.round(1).to_numpy(),
        row=rows, col=cols, in_area=in_area,
    )
    table.to_csv(out / "labels.csv", index=False, encoding="utf-8")
    shared = table[table.in_area].duplicated(["row", "col"]).sum()
    log(f"  меток: {len(table)}, в области обучения: {int(in_area.sum())}, "
        f"делят ячейку с другой меткой: {int(shared)}")
    return table


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

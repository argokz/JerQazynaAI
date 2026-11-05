"""Разбиение по геометрии региона; буфер измеряется от полигона, а не центров ячеек."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from jerqazyna.config import TaskConfig
from jerqazyna.features.cache import Digests, fingerprint, write_json
from jerqazyna.grid.template import GridSpec, write_layer


def region_masks(grid: GridSpec, area: np.ndarray, path: Path, buffer_m: float):
    import geopandas as gpd
    import shapely

    if area.shape != grid.shape or area.dtype != bool:
        raise ValueError("area должна быть булевой маской на сетке")
    if not np.isfinite(buffer_m) or buffer_m <= 0:
        raise ValueError("Буфер должен быть конечным положительным числом")
    frame = gpd.read_file(path)
    if (frame.empty or frame.crs is None or not frame.geometry.is_valid.all()
            or frame.geometry.isna().any() or frame.geometry.is_empty.any()
            or not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Граница региона должна содержать корректные полигоны с CRS")
    if frame.crs.is_geographic:
        frame.geometry = frame.geometry.segmentize(.01)
    geom = frame.to_crs(grid.crs).geometry.union_all()
    buffered = geom.buffer(buffer_m)
    test = np.zeros(grid.shape, bool)
    excluded = np.zeros(grid.shape, bool)
    xs = grid.x_min + (np.arange(grid.width) + .5) * grid.cell_size
    for start in range(0, grid.height, 128):
        stop = min(start + 128, grid.height)
        ys = grid.y_max - (np.arange(start, stop) + .5) * grid.cell_size
        xx, yy = np.meshgrid(xs, ys)
        test[start:stop] = shapely.intersects_xy(geom, xx, yy)
        excluded[start:stop] = shapely.intersects_xy(buffered, xx, yy)
    test &= area
    excluded &= area & ~test
    train = area & ~test & ~excluded
    if not test.any() or not train.any():
        raise ValueError("Разбиение должно содержать непустые обучение и тест")
    return train, test, excluded


def prepare_split(task: TaskConfig, root: Path, log=print) -> Path:
    """Создаёт адресуемый по содержимому набор масок и назначений меток.

    Этот шаг не обучает модель и не оценивает прогнозы на тестовых объектах.
    Старые поколения разбиений сохраняются; повреждённое поколение не переиспользуется.
    """
    import json

    from jerqazyna.features.prepare import features_dir, prepare_labels
    from jerqazyna.grid.template import read_layer

    root = Path(root)
    labels = prepare_labels(task, root, log)
    out = features_dir(root, task)
    grid = GridSpec.load(out / "grid.json")
    area = read_layer(out / "area_mask.tif").astype(bool)
    boundary = root / task.validation.blind_region.geometry
    digests = Digests()
    spec = {
        "version": 1, "grid": asdict(grid), "validation": task.validation.model_dump(),
        "seed": task.seed, "boundary_sha256": digests.file(boundary),
        "area_sha256": digests.file(out / "area_mask.tif"),
        "labels_sha256": digests.file(out / "labels.csv"),
        "implementation_sha256": digests.file(Path(__file__)),
    }
    split_id = fingerprint(spec)
    dest = out / "splits" / split_id[:16]
    manifest = dest / "split.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        files = saved.get("files", {})
        expected_files = {"train.tif", "test.tif", "buffer.tif", "labels.csv"}
        if (saved.get("spec") != spec or set(files) != expected_files
                or any(not (dest / name).is_file() or digests.file(dest / name) != digest
                       for name, digest in files.items())):
            raise ValueError(f"Повреждённое разбиение: {dest}; оно не будет использовано")
        log(f"  разбиение уже готово: {dest}")
        return dest
    train, test, excluded = region_masks(grid, area, boundary, task.validation.blind_region.buffer_m)
    if digests.file(boundary) != spec["boundary_sha256"]:
        raise OSError("Граница региона изменилась во время подготовки")
    dest.mkdir(parents=True, exist_ok=True)
    for name, mask in (("train", train), ("test", test), ("buffer", excluded)):
        write_layer(dest / f"{name}.tif", grid, mask.astype(np.uint8), name)
    rows, cols = labels.row.to_numpy(), labels.col.to_numpy()
    labels = labels.assign(partition=np.select(
        [train[rows, cols], test[rows, cols], excluded[rows, cols]],
        ["train", "test", "buffer"], default="outside",
    ))
    temporary = dest / "labels.csv.part"
    labels.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(dest / "labels.csv")
    summary = {
        "spec": spec, "split_id": split_id,
        "provisional": task.validation.blind_region.provisional,
        "cells": {"train": int(train.sum()), "test": int(test.sum()),
                  "buffer": int(excluded.sum())},
        "objects": labels.partition.value_counts().to_dict(),
        "files": {name: digests.file(dest / name)
                  for name in ("train.tif", "test.tif", "buffer.tif", "labels.csv")},
    }
    write_json(manifest, summary)
    log(f"  разбиение: {dest}; объекты: {summary['objects']}")
    return dest

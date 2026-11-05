"""Проверка происхождения слоёв и таблица только обучающей территории на диске."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jerqazyna.config import TaskConfig
from jerqazyna.features.cache import (
    Digests,
    fingerprint,
    grid_recipe,
    implementation_fingerprint,
    layer_is_current,
    layer_recipe,
    raster_matches,
    write_json,
)
from jerqazyna.features.prepare import _selected_layers, features_dir
from jerqazyna.grid.template import GridSpec, read_layer
from jerqazyna.models.config import ExperimentConfig


def map_cells(sorted_cells: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Индекс строки таблицы либо -1; запрошенные ячейки могут лежать вне таблицы."""
    requested = np.asarray(requested, dtype=np.int64)
    indices = np.searchsorted(sorted_cells, requested)
    inside = indices < sorted_cells.size
    result = np.full(requested.shape, -1, np.int64)
    result[inside] = np.where(sorted_cells[indices[inside]] == requested[inside], indices[inside], -1)
    return result


def verify_split(task: TaskConfig, root: Path, split: Path, digests: Digests) -> dict:
    out = features_dir(root, task)
    saved = json.loads((split / "split.json").read_text(encoding="utf-8"))
    spec = saved["spec"]
    expected_files = {"train.tif", "test.tif", "buffer.tif", "labels.csv"}
    if (set(saved.get("files", {})) != expected_files
            or saved["split_id"] != fingerprint(spec)
            or spec["validation"] != task.validation.model_dump()
            or spec["seed"] != task.seed
            or spec["grid"] != asdict(GridSpec.load(out / "grid.json"))
            or spec["boundary_sha256"] != digests.file(root / task.validation.blind_region.geometry)
            or spec["area_sha256"] != digests.file(out / "area_mask.tif")
            or spec["labels_sha256"] != digests.file(out / "labels.csv")
            or spec["implementation_sha256"] != digests.file(
                Path(__file__).parents[1] / "validation/holdout.py")):
        raise ValueError("Разбиение не соответствует текущей задаче: выполните prepare-split")
    grid = GridSpec.load(out / "grid.json")
    for name, digest in saved["files"].items():
        if digests.file(split / name) != digest:
            raise ValueError(f"Повреждён файл разбиения: {name}")
        if name.endswith(".tif") and not raster_matches(split / name, grid):
            raise ValueError(f"Сетка маски разбиения неверна: {name}")
    masks = [read_layer(split / name).astype(bool) for name in ("train.tif", "test.tif", "buffer.tif")]
    area = read_layer(out / "area_mask.tif").astype(bool)
    if not np.array_equal(sum(mask.astype(np.uint8) for mask in masks), area.astype(np.uint8)):
        raise ValueError("Маски разбиения пересекаются или не покрывают область")
    return saved


def verified_channels(task: TaskConfig, experiment: ExperimentConfig, root: Path, digests: Digests):
    out = features_dir(root, task)
    grid = GridSpec.load(out / "grid.json")
    area = json.loads((out / "area.json").read_text(encoding="utf-8"))
    if (area.get("spec") != grid_recipe(task, root, digests)
            or area.get("mask_sha256") != digests.file(out / "area_mask.tif")
            or not raster_matches(out / "area_mask.tif", grid)):
        raise ValueError("Маска области устарела: выполните prepare")
    inventory = {r["file"]: r for r in json.loads((out / "layers.json").read_text(encoding="utf-8"))}
    stored = json.loads((out / "cache.json").read_text(encoding="utf-8"))
    if stored.get("version") != 1:
        raise ValueError("Нужен новый кэш слоёв: выполните prepare")
    cache = stored["layers"]
    by_name = {layer.name: layer for layer in task.layers}
    if set(experiment.layers) - by_name.keys():
        raise ValueError("В эксперименте указан неизвестный слой")
    for name in experiment.layers:
        if by_name[name].role != "feature":
            raise ValueError(f"{name}: роль {by_name[name].role} запрещена в обучающей таблице")
    requested = list(experiment.layers)
    spectral = any(by_name[name].group == "spectral" for name in requested)
    mask_name = None
    if spectral:
        masks = [layer.name for layer in task.layers if layer.kind == "mine_mask" and layer.role == "mask"]
        if len(masks) != 1:
            raise ValueError("Для спектральных признаков нужна одна маска выработок с role: mask")
        mask_name = masks[0]
        requested.append(mask_name)
    impl = implementation_fingerprint(digests)
    for layer in _selected_layers(task, requested):
        recipe = layer_recipe(layer, task, root, grid, fingerprint(area), impl, digests, cache)
        if not layer_is_current(out, layer, grid, recipe, cache.get(layer.name, {}), inventory, digests):
            raise ValueError(f"{layer.name}: нет проверенного актуального слоя; выполните prepare")
    channels = sorted(
        (row for row in inventory.values() if row["layer"] in experiment.layers),
        key=lambda row: row["file"],
    )
    if not channels or any(row["role"] != "feature" for row in channels):
        raise ValueError("Опись не содержит разрешённых признаков")
    for row in channels:
        layer = by_name[row["layer"]]
        if any(row[key] != getattr(layer, key) for key in ("role", "group", "kind", "source")):
            raise ValueError(f"{layer.name}: метаданные описи не соответствуют конфигурации")
    mask_file = f"{mask_name}__mask.tif" if mask_name else None
    if mask_file and mask_file not in cache[mask_name]["files"]:
        raise ValueError("Маска выработок не содержит канал mask")
    return grid, channels, mask_file


class Dataset:
    def __init__(self, path: Path, *, verify: bool = True):
        import pandas as pd

        self.path = Path(path)
        self.metadata = json.loads((self.path / "dataset.json").read_text(encoding="utf-8"))
        if verify:
            digests = Digests()
            for name, digest in self.metadata["files"].items():
                if digests.file(self.path / name) != digest:
                    raise ValueError(f"Повреждена таблица признаков: {name}")
        self.grid = GridSpec(**self.metadata["spec"]["grid"])
        self.columns = self.metadata["columns"]
        self.X = np.load(self.path / "features.npy", mmap_mode="r", allow_pickle=False)
        self.cells = np.load(self.path / "cells.npy", mmap_mode="r", allow_pickle=False)
        self.labels = pd.read_csv(self.path / "labels.csv").fillna({"status": "unknown"})
        if self.X.shape != (self.cells.size, len(self.columns)):
            raise ValueError("Некорректная форма таблицы признаков")


def build_dataset(
    task: TaskConfig, experiment: ExperimentConfig, root: Path, split: Path, log=print,
) -> Dataset:
    """Потоково читает каналы, хранит float32 на диске. Отложенный регион не входит в X."""
    import pandas as pd

    root, split = Path(root), Path(split)
    digests = Digests()
    log("== проверка разбиения и происхождения признаков")
    split_meta = verify_split(task, root, split, digests)
    grid, channels, mask_file = verified_channels(task, experiment, root, digests)
    out = features_dir(root, task)
    columns = [Path(row["file"]).stem for row in channels]
    input_files = [row["file"] for row in channels] + ([mask_file] if mask_file else [])
    spec = {
        "version": 1, "grid": asdict(grid), "split_id": split_meta["split_id"],
        "split_manifest_sha256": digests.file(split / "split.json"),
        "feature_files": {name: digests.file(out / name) for name in input_files},
        "recipes": {name: json.loads((out / "cache.json").read_text(encoding="utf-8"))["layers"][name]
                    for name in experiment.layers},
        "min_valid_fraction": experiment.min_valid_fraction,
        "implementation_sha256": digests.file(Path(__file__)),
    }
    dataset_id = fingerprint(spec)
    dest = root / "data/tables" / task.task_id / dataset_id[:16]
    if (dest / "dataset.json").exists():
        dataset = Dataset(dest)
        if dataset.metadata["spec"] != spec:
            raise ValueError("Несовместимая таблица признаков")
        log(f"  таблица уже готова: {dest}")
        return dataset
    train = read_layer(split / "train.tif").astype(bool)
    cells = np.flatnonzero(train.ravel())
    if not cells.size:
        raise ValueError("Обучающая территория пуста")
    dest.mkdir(parents=True, exist_ok=True)
    stage = dest / "features.part.npy"
    matrix = np.lib.format.open_memmap(stage, mode="w+", dtype="float32", shape=(cells.size, len(columns)))
    counts = np.zeros(cells.size, np.uint32)
    mine_mask = read_layer(out / mask_file).ravel()[cells] > 0 if mask_file else None
    coverage = {}
    for col, channel in enumerate(channels):
        values = read_layer(out / channel["file"]).ravel()[cells].astype(np.float32)
        values[~np.isfinite(values)] = np.nan
        if channel["group"] == "spectral":
            values[mine_mask] = np.nan
        finite = np.isfinite(values)
        counts += finite
        coverage[columns[col]] = float(finite.mean())
        matrix[:, col] = values
        log(f"  {columns[col]}: покрытие train {finite.mean():.1%}")
    valid = counts >= max(1, int(np.ceil(len(columns) * experiment.min_valid_fraction)))
    if not valid.any():
        raise ValueError("Нет ячеек с достаточным покрытием признаков")
    matrix.flush()
    if valid.all():
        del matrix
        stage.replace(dest / "features.npy")
    else:
        compact = np.lib.format.open_memmap(dest / "features.compact.part.npy", mode="w+",
                                           dtype="float32", shape=(int(valid.sum()), len(columns)))
        offset = 0
        for start in range(0, cells.size, experiment.batch_size):
            end = min(start + experiment.batch_size, cells.size)
            chunk = matrix[start:end][valid[start:end]]
            compact[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
        compact.flush()
        del compact, matrix
        (dest / "features.compact.part.npy").replace(dest / "features.npy")
        stage.unlink()
        cells = cells[valid]
    with open(dest / "cells.npy.part", "wb") as stream:
        np.save(stream, cells, allow_pickle=False)
    (dest / "cells.npy.part").replace(dest / "cells.npy")
    # Фильтр partition выполняется до использования координат/статусов для обучения.
    labels = pd.read_csv(split / "labels.csv")
    labels = labels[labels.partition == "train"].copy()
    labels["matrix_row"] = map_cells(cells, labels.row.to_numpy() * grid.width + labels.col.to_numpy())
    labels.to_csv(dest / "labels.csv.part", index=False, encoding="utf-8")
    (dest / "labels.csv.part").replace(dest / "labels.csv")
    for name, digest in spec["feature_files"].items():
        if digests.file(out / name) != digest:
            raise OSError(f"Слой изменился во время сборки таблицы: {name}")
    metadata = {
        "dataset_id": dataset_id, "spec": spec, "columns": columns, "coverage": coverage,
        "n_cells": int(cells.size), "n_excluded_coverage": int((~valid).sum()),
        "n_training_objects": len(labels),
        "files": {name: digests.file(dest / name) for name in ("features.npy", "cells.npy", "labels.csv")},
    }
    write_json(dest / "dataset.json", metadata)
    log(f"  таблица {cells.size:,} × {len(columns)}: {dest}")
    return Dataset(dest, verify=False)

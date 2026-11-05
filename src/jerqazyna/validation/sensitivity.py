"""Ограниченная серия CV на одной train-популяции без переобучения финальной модели."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import Field, model_validator

from jerqazyna.config import ConfigModel, TaskConfig
from jerqazyna.features.cache import Digests, fingerprint, write_json
from jerqazyna.grid.template import read_layer
from jerqazyna.models.config import ExperimentConfig
from jerqazyna.models.dataset import build_dataset
from jerqazyna.models.evaluation import evaluate_oof, grouped_interval
from jerqazyna.models.pipeline import _fit, _map_scores, _predict_rows, _provenance
from jerqazyna.models.sampling import SpatialFolds, sample_background
from jerqazyna.validation.metrics import capture_probabilities


class Variant(ConfigModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    block_size_m: float = Field(gt=0)
    layers: list[str] = Field(min_length=1)
    positive_statuses: list[Literal["deposit", "prospect"]] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.layers)) != len(self.layers) or len(set(self.positive_statuses)) != len(self.positive_statuses):
            raise ValueError("Повторы слоёв или статусов запрещены")
        return self


class SensitivityConfig(ConfigModel):
    study_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    model: Literal["random_forest", "logistic"] = "random_forest"
    variants: list[Variant] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("Имена вариантов должны быть уникальными")
        return self


def load_sensitivity(path):
    return SensitivityConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class ProjectedFeatures:
    """Проекция после выбора строк: полная многомиллионная таблица не копируется."""

    def __init__(self, array, columns):
        self.array, self.columns = array, columns

    def __getitem__(self, rows):
        return self.array[rows][:, self.columns]


class FeatureView:
    def __init__(self, dataset, layers):
        indices = [i for i, name in enumerate(dataset.columns) if name.split("__", 1)[0] in layers]
        if set(layers) != {dataset.columns[i].split("__", 1)[0] for i in indices}:
            raise ValueError("Вариант содержит отсутствующий слой")
        self.columns = [dataset.columns[i] for i in indices]
        self.X = ProjectedFeatures(dataset.X, indices)
        self.cells, self.labels, self.grid = dataset.cells, dataset.labels, dataset.grid
        self.metadata = {"dataset_id": fingerprint({
            "parent_dataset_id": dataset.metadata["dataset_id"], "columns": self.columns,
        })}


def verify_files(folder, files):
    digests = Digests()
    if any(digests.file(folder / name) != digest for name, digest in files.items()):
        raise ValueError(f"Повреждены артефакты: {folder}")


def object_hits(scores, positive_rows, folds, fraction):
    hits = np.empty(len(positive_rows))
    for fold in range(folds.n_folds):
        rows = np.flatnonzero(folds.fold == fold)
        which = folds.fold[positive_rows] == fold
        if which.any():
            hits[which] = capture_probabilities(scores[rows], np.searchsorted(rows, positive_rows[which]), fraction)
    return hits


def paired_comparison(scores, heuristic, positive_rows, folds, task, config):
    records = []
    for fraction in task.validation.area_fractions:
        delta = (object_hits(scores, positive_rows, folds, fraction)
                 - object_hits(heuristic, positive_rows, folds, fraction))
        interval = grouped_interval(delta, folds.blocks[positive_rows],
                                    config.grouped_bootstrap_iterations,
                                    task.validation.confidence, task.seed)
        records.append({"area_fraction": fraction, "capture_difference": float(delta.mean()),
                        "grouped_interval": interval,
                        "interpretation": "paired conditional difference; no multiple-comparison correction"})
    return records


def _variant(task, config, study, variant, dataset, baseline, dest, log):
    marker = dest / "result.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        verify_files(dest, saved["files"])
        log(f"== {variant.name}: готовый результат проверен")
        return saved
    dest.mkdir(parents=True, exist_ok=True)
    folds = SpatialFolds(dataset, variant.block_size_m, config.n_folds, task.seed)
    view = FeatureView(dataset, variant.layers)
    objects = dataset.labels[
        dataset.labels.status.isin(config.positive_statuses) & (dataset.labels.matrix_row >= 0)
    ].copy()
    if not len(objects):
        raise ValueError("Нет объектов для общей оценки")
    eval_rows = objects.matrix_row.to_numpy(dtype=np.int64)
    train_rows = np.unique(dataset.labels.loc[
        dataset.labels.status.isin(variant.positive_statuses) & (dataset.labels.matrix_row >= 0), "matrix_row",
    ].to_numpy(dtype=np.int64))
    objects["fold"] = folds.fold[eval_rows]
    objects["block"] = folds.blocks[eval_rows]
    reuse = (variant.block_size_m == task.validation.spatial_block_size_m
             and set(variant.layers) == set(config.layers)
             and set(variant.positive_statuses) == set(config.positive_statuses))
    heuristic = np.empty(len(dataset.cells), np.float32)
    heuristic_col = dataset.columns.index(config.heuristic_band)
    fits, seed_reports = [], []
    scores = np.zeros(len(dataset.cells), np.float64)
    if reuse:
        log(f"== {variant.name}: исходные проверенные OOF, без повторного fit")
        scores = read_layer(baseline / f"oof_{study.model}.tif").ravel()[dataset.cells]
        heuristic = read_layer(baseline / "oof_heuristic.tif").ravel()[dataset.cells]
        source = json.loads((baseline / "cv.json").read_text(encoding="utf-8"))
        seed_reports = source["models"][study.model]["background_seed_reports"]
        fits = [v for v in source["fits"] if v["model"] == study.model]
    else:
        for seed in config.background_seeds:
            current = np.full(len(dataset.cells), np.nan, np.float32)
            for fold in range(config.n_folds):
                valid = np.flatnonzero(folds.fold == fold)
                pos = folds.training_rows(train_rows, fold, config.cv_buffer_m)
                if len(pos) < 2:
                    raise ValueError(f"{variant.name}: fold {fold}: меньше двух fitting-positive после буфера")
                background = sample_background(dataset, config, folds, seed, fold)
                if seed == config.background_seeds[0]:
                    available = np.asarray(dataset.X[np.r_[pos, background], heuristic_col])
                    median = float(np.nanmedian(available)) if np.isfinite(available).any() else 0.
                    values = np.asarray(dataset.X[valid, heuristic_col])
                    heuristic[valid] = -np.where(np.isfinite(values), values, median)
                log(f"== {variant.name}: seed={seed}, fold={fold + 1}/{config.n_folds}, "
                    f"features={len(view.columns)}, positive={len(pos)}, background={len(background)}")
                model = _fit(view, config, study.model, seed, pos, background,
                             dest / "checkpoints" / f"{study.model}_seed{seed}_fold{fold}")
                current[valid] = _predict_rows(model, view, valid, config.batch_size)
                fits.append({"seed": seed, "fold": fold, "positive_cells": len(pos),
                             "background_cells": len(background), "validation_cells": len(valid),
                             "validation_objects": int((objects.fold == fold).sum())})
            seed_reports.append({"seed": seed, **evaluate_oof(
                current, eval_rows, folds, task.validation.area_fractions, task.validation.confidence,
                config.grouped_bootstrap_iterations, task.seed,
            )})
            objects[f"seed{seed}"] = current[eval_rows]
            scores += current
        scores = (scores / len(config.background_seeds)).astype(np.float32)
    if not np.isfinite(scores).all() or not np.isfinite(heuristic).all():
        raise ValueError("OOF не покрывают общую train-популяцию")

    def evaluate(values, rows):
        return evaluate_oof(values, rows, folds, task.validation.area_fractions,
                            task.validation.confidence, config.grouped_bootstrap_iterations, task.seed)

    deposit_rows = objects.loc[objects.status == "deposit", "matrix_row"].to_numpy(dtype=np.int64)
    objects["ensemble"] = scores[eval_rows]
    objects["heuristic"] = heuristic[eval_rows]
    for fraction in task.validation.area_fractions:
        objects[f"hit_{fraction:g}"] = object_hits(scores, eval_rows, folds, fraction)
        objects[f"heuristic_hit_{fraction:g}"] = object_hits(heuristic, eval_rows, folds, fraction)
    objects.to_csv(dest / "oof_objects.csv", index=False, encoding="utf-8")
    _map_scores(dest / "oof_score.tif", dataset, scores, "sensitivity OOF; fixed train population only")
    _map_scores(dest / "cv_fold.tif", dataset, folds.fold.astype(np.float32), "sensitivity spatial fold; train only")
    digests = Digests()
    result = {
        "variant": variant.model_dump(), "columns": view.columns, "reused_baseline_oof": reuse,
        "holdout_evaluated": False, "n_cells": len(dataset.cells), "n_fit_positive_cells": len(train_rows),
        "common_population": evaluate(scores, eval_rows),
        "deposit_subgroup": evaluate(scores, deposit_rows) if len(deposit_rows) else None,
        "heuristic": evaluate(heuristic, eval_rows),
        "paired_difference_vs_heuristic": paired_comparison(scores, heuristic, eval_rows, folds, task, config),
        "background_seed_reports": seed_reports, "fits": fits,
        "files": {p.relative_to(dest).as_posix(): digests.file(p) for p in sorted(dest.rglob("*"))
                  if p.is_file() and ".part" not in p.name and p != marker},
    }
    write_json(marker, result)
    metric = result["common_population"]["metrics"][-1]
    log(f"== {variant.name}: capture@{metric['area_fraction']:.0%}={metric['capture']:.1%}")
    return result


def run_sensitivity(task: TaskConfig, study: SensitivityConfig, root: Path, baseline: Path,
                    split: Path, log=print):
    from threadpoolctl import threadpool_limits

    root, baseline, split = Path(root), Path(baseline), Path(split)
    baseline_meta = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    if baseline_meta["spec"]["task"] != task.model_dump() or baseline_meta["holdout_evaluated"]:
        raise ValueError("Исходный запуск не соответствует задаче или слепой тест уже оценивался")
    config = ExperimentConfig.model_validate(baseline_meta["spec"]["experiment"])
    if study.model not in config.models:
        raise ValueError("Модель отсутствует в исходном эксперименте")
    if any(set(v.layers) - set(config.layers) for v in study.variants):
        raise ValueError("Варианты могут только исключать исходные слои")
    provenance = _provenance(root)
    if provenance != baseline_meta["spec"]["provenance"]:
        raise ValueError("Код или окружение baseline изменились; нужен совместимый исходный запуск")
    verify_files(baseline, baseline_meta["files"])
    dataset = build_dataset(task, config, root, split, log)
    if dataset.metadata["dataset_id"] != baseline_meta["spec"]["dataset_id"]:
        raise ValueError("Общая train-популяция отличается от исходной")
    spec = {"version": 1, "study": study.model_dump(), "baseline_run_id": baseline_meta["run_id"],
            "baseline_manifest_sha256": Digests().file(baseline / "run.json"),
            "dataset_id": dataset.metadata["dataset_id"],
            "implementation_sha256": Digests().file(Path(__file__))}
    study_id = fingerprint(spec)
    dest = root / "outputs" / task.task_id / study.study_id / study_id[:16]
    marker = dest / "run.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["spec"] != spec:
            raise ValueError("Протокол завершённой серии повреждён")
        verify_files(dest, saved["files"])
        log(f"Серия уже готова: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "protocol.json", {"study_id": study_id, "spec": spec, "holdout_evaluated": False})
    with threadpool_limits(limits=config.workers):
        results = [_variant(task, config, study, v, dataset, baseline, dest / v.name, log)
                   for v in study.variants]
    summary = {"study_id": study_id, "holdout_evaluated": False, "results": results,
               "final_model_changed": False,
               "caveats": ["Exploratory train CV; no final model selected by this study.",
                           "Same fixed train cells and evaluation objects in all variants.",
                           "Block size changes block assignments as well as spatial grouping.",
                           "Paired intervals are conditional; comparisons are not multiplicity-adjusted."]}
    write_json(dest / "summary.json", summary)
    digests = Digests()
    write_json(marker, {"study_id": study_id, "spec": spec, "holdout_evaluated": False,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "files": {p.relative_to(dest).as_posix(): digests.file(p)
                                  for p in sorted(dest.rglob("*")) if p.is_file() and ".part" not in p.name
                                  and p != marker}})
    log(f"Серия готова: {dest}; слепой тест не оценивался")
    return dest

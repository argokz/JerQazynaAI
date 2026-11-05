"""Фиксированная CV-серия: площадь/страны и число фоновых подвыборок при равном числе деревьев."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import Field, model_validator

from jerqazyna.config import ConfigModel, CountryCode
from jerqazyna.features.cache import Digests, fingerprint, write_json
from jerqazyna.grid.template import read_layer
from jerqazyna.models.config import ExperimentConfig
from jerqazyna.models.dataset import build_dataset
from jerqazyna.models.evaluation import evaluate_oof, grouped_interval
from jerqazyna.models.pipeline import _fit, _map_scores, _predict_rows, _provenance
from jerqazyna.models.sampling import SpatialFolds, sample_background
from jerqazyna.validation.countries import country_audit, country_background, eligible_rows
from jerqazyna.validation.sensitivity import object_hits, verify_files


class BackgroundVariant(ConfigModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    sampling: Literal["area", "country_equal"]
    seeds: list[int] = Field(min_length=1)
    trees_per_member: int = Field(ge=1)

    @model_validator(mode="after")
    def unique_seeds(self):
        if len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            raise ValueError("Seeds должны быть уникальными и неотрицательными")
        return self


class CountryPUConfig(ConfigModel):
    study_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    country_names: dict[CountryCode, str]
    variants: list[BackgroundVariant] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_variants(self):
        if len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError("Имена вариантов должны быть уникальными")
        if len({v.trees_per_member * len(v.seeds) for v in self.variants}) != 1:
            raise ValueError("Суммарное число деревьев должно совпадать")
        return self


def load_country_pu(path):
    return CountryPUConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def variant_config(config, variant):
    return ExperimentConfig.model_validate({
        **config.model_dump(), "background_seeds": variant.seeds,
        "forest": {**config.forest.model_dump(), "n_estimators": variant.trees_per_member},
    })


def summarize_variant(scores, reference, heuristic, dataset, folds, config, task, audit_labels, dest):
    import pandas as pd

    objects = dataset.labels[
        dataset.labels.status.isin(config.positive_statuses) & (dataset.labels.matrix_row >= 0)
    ].copy()
    rows = objects.matrix_row.to_numpy(dtype=np.int64)
    objects["fold"] = folds.fold[rows]
    objects["block"] = folds.blocks[rows]
    objects["country_iso_geometry"] = audit_labels.loc[objects.index, "country_iso_geometry"].fillna("UNRESOLVED")
    objects["score"] = scores[rows]
    metrics, differences, country_reports = [], [], []
    for fraction in task.validation.area_fractions:
        hits = object_hits(scores, rows, folds, fraction)
        base_hits = object_hits(reference, rows, folds, fraction)
        heur_hits = object_hits(heuristic, rows, folds, fraction)
        objects[f"hit_{fraction:g}"] = hits
        objects[f"baseline_hit_{fraction:g}"] = base_hits
        objects[f"heuristic_hit_{fraction:g}"] = heur_hits
        for name, other in [("baseline", base_hits), ("heuristic", heur_hits)]:
            delta = hits - other
            differences.append({"area_fraction": fraction, "reference": name,
                                "capture_difference": float(delta.mean()),
                                "grouped_interval": grouped_interval(
                                    delta, folds.blocks[rows], config.grouped_bootstrap_iterations,
                                    task.validation.confidence, task.seed)})
        groups = pd.DataFrame({"country": objects.country_iso_geometry, "hit": hits}).groupby("country").hit
        countries = {str(iso): {"n_objects": len(values), "capture": float(values.mean())}
                     for iso, values in groups}
        country_reports.append({"area_fraction": fraction, "countries": countries,
                                "macro_capture_over_observed_countries": float(np.mean(
                                    [v["capture"] for v in countries.values()])),
                                "area_policy": "global_fold_area_threshold; subgroup_objects_only"})
    metrics = evaluate_oof(scores, rows, folds, task.validation.area_fractions, task.validation.confidence,
                           config.grouped_bootstrap_iterations, task.seed)
    deposit_rows = objects.loc[objects.status == "deposit", "matrix_row"].to_numpy(dtype=np.int64)
    deposit = evaluate_oof(scores, deposit_rows, folds, task.validation.area_fractions,
                           task.validation.confidence, config.grouped_bootstrap_iterations,
                           task.seed) if len(deposit_rows) else None
    objects.to_csv(dest / "oof_objects.csv", index=False, encoding="utf-8")
    _map_scores(dest / "oof_score.tif", dataset, scores, "positive/unlabeled CV; train only")
    return {"common_population": metrics, "deposit_subgroup": deposit,
            "paired_differences": differences, "country_diagnostics": country_reports}


def _run_variant(task, base_config, variant, dataset, folds, strata, n_countries,
                 pools, reference, heuristic, audit_labels, baseline, dest, log):
    marker = dest / "result.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        verify_files(dest, saved["files"])
        log(f"== {variant.name}: готовый результат проверен")
        return saved
    dest.mkdir(parents=True, exist_ok=True)
    config = variant_config(base_config, variant)
    pos_rows = np.unique(dataset.labels.loc[
        dataset.labels.status.isin(config.positive_statuses) & (dataset.labels.matrix_row >= 0),
        "matrix_row",
    ].to_numpy(dtype=np.int64))
    fits, reports = [], []
    scores = np.zeros(len(dataset.cells), np.float64)
    reuse = (variant.sampling == "area" and variant.seeds == base_config.background_seeds
             and variant.trees_per_member == base_config.forest.n_estimators)
    if reuse:
        scores = reference.copy()
        source = json.loads((baseline / "cv.json").read_text(encoding="utf-8"))
        reports = source["models"]["random_forest"]["background_seed_reports"]
        log(f"== {variant.name}: проверенные OOF исходного RF")
    else:
        for seed in variant.seeds:
            current = np.full(len(dataset.cells), np.nan, np.float32)
            for fold in range(config.n_folds):
                positive = folds.training_rows(pos_rows, fold, config.cv_buffer_m)
                if len(positive) < 2:
                    raise ValueError("Меньше двух fitting-positive после буфера")
                if variant.sampling == "area":
                    background = sample_background(dataset, config, folds, seed, fold)
                else:
                    if fold not in pools:
                        log(f"== расчёт допустимого фона и ёмкости стран: fold={fold + 1}")
                        pools[fold] = eligible_rows(dataset, config, folds, fold)
                    background = country_background(pools[fold], strata, n_countries,
                                                    config.background_cells, seed, fold)
                valid = np.flatnonzero(folds.fold == fold)
                log(f"== {variant.name}: seed={seed}, fold={fold + 1}/{config.n_folds}, "
                    f"trees={variant.trees_per_member}, positive={len(positive)}, background={len(background)}")
                model = _fit(dataset, config, "random_forest", seed, positive, background,
                             dest / "checkpoints" / f"random_forest_seed{seed}_fold{fold}")
                current[valid] = _predict_rows(model, dataset, valid, config.batch_size)
                counts = np.bincount(strata[background], minlength=n_countries + 2)
                fit = {"seed": seed, "fold": fold, "positive_cells": len(positive),
                       "background_cells": len(background), "country_counts": counts.tolist()}
                if variant.sampling == "country_equal":
                    fit["eligible_country_counts"] = np.bincount(
                        strata[pools[fold]], minlength=n_countries + 2).tolist()
                fits.append(fit)
            reports.append({"seed": seed, **evaluate_oof(
                current, pos_rows, folds, task.validation.area_fractions, task.validation.confidence,
                config.grouped_bootstrap_iterations, task.seed)})
            scores += current
        scores = (scores / len(variant.seeds)).astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Неполные OOF-прогнозы")
    result = {"variant": variant.model_dump(), "reused_baseline_oof": reuse, "fits": fits,
              "effective_experiment": config.model_dump(),
              "background_seed_reports": reports, "holdout_evaluated": False,
              **summarize_variant(scores, reference, heuristic, dataset, folds, config,
                                  task, audit_labels, dest)}
    digests = Digests()
    result["files"] = {p.relative_to(dest).as_posix(): digests.file(p) for p in sorted(dest.rglob("*"))
                       if p.is_file() and ".part" not in p.name and p != marker}
    write_json(marker, result)
    log(f"== {variant.name}: capture@20%={result['common_population']['metrics'][-1]['capture']:.1%}")
    return result


def run_country_pu(task, study, root: Path, baseline: Path, split: Path, log=print):
    import pandas as pd
    from threadpoolctl import threadpool_limits

    root, baseline, split = Path(root), Path(baseline), Path(split)
    base = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    if (base["spec"]["task"] != task.model_dump() or base["holdout_evaluated"]
            or _provenance(root) != base["spec"]["provenance"]):
        raise ValueError("Нужен совместимый исходный baseline с закрытым тестом")
    config = ExperimentConfig.model_validate(base["spec"]["experiment"])
    if "random_forest" not in config.models:
        raise ValueError("В baseline нет Random Forest")
    expected_budget = config.forest.n_estimators * len(config.background_seeds)
    if any(v.trees_per_member * len(v.seeds) != expected_budget for v in study.variants):
        raise ValueError("Бюджет деревьев должен совпадать с baseline")
    verify_files(baseline, base["files"])
    dataset = build_dataset(task, config, root, split, log)
    if dataset.metadata["dataset_id"] != base["spec"]["dataset_id"]:
        raise ValueError("Изменилась исходная train-популяция")
    audit = country_audit(task, dataset, root, study.country_names)
    audit_meta = json.loads((audit / "audit.json").read_text(encoding="utf-8"))
    audit_summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    if audit_summary["n_coordinate_cell_mismatches"]:
        raise ValueError("Координаты меток не совпадают с ячейками таблицы")
    digests = Digests()
    spec = {"version": 1, "study": study.model_dump(), "baseline_run_id": base["run_id"],
            "baseline_manifest_sha256": digests.file(baseline / "run.json"),
            "dataset_id": dataset.metadata["dataset_id"], "country_audit_id": audit_meta["audit_id"],
            "country_audit_manifest_sha256": digests.file(audit / "audit.json"),
            "source_sha256": {name: digests.file(Path(__file__).with_name(name))
                              for name in ("country_pu.py", "countries.py", "sensitivity.py")}}
    study_id = fingerprint(spec)
    dest = root / "outputs" / task.task_id / study.study_id / study_id[:16]
    marker = dest / "run.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["spec"] != spec:
            raise ValueError("Протокол готовой серии повреждён")
        verify_files(dest, saved["files"])
        log(f"Серия уже готова: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "protocol.json", {"study_id": study_id, "spec": spec, "holdout_evaluated": False})
    strata = np.load(audit / "strata.npy", allow_pickle=False)
    audit_labels = pd.read_csv(audit / "labels_audit.csv")
    if len(strata) != len(dataset.cells):
        raise ValueError("Неверная длина массива стран")
    reference = read_layer(baseline / "oof_random_forest.tif").ravel()[dataset.cells]
    heuristic = read_layer(baseline / "oof_heuristic.tif").ravel()[dataset.cells]
    folds = SpatialFolds(dataset, task.validation.spatial_block_size_m, config.n_folds, task.seed)
    _map_scores(dest / "cv_fold.tif", dataset, folds.fold.astype(np.float32), "fixed CV fold; train only")
    pools = {}
    with threadpool_limits(limits=config.workers):
        results = [_run_variant(task, config, v, dataset, folds, strata, len(study.country_names),
                                pools, reference, heuristic, audit_labels, baseline, dest / v.name, log)
                   for v in study.variants]
    write_json(dest / "summary.json", {"study_id": study_id, "results": results,
                                       "holdout_evaluated": False, "final_model_changed": False,
                                       "country_audit_path": audit.relative_to(root).as_posix(),
                                       "caveats": ["RF adaptation of positive/unlabeled subsample bagging.",
                                                   "Baseline already averages two background subsamples.",
                                                   "No class-prior or selection-bias correction; scores uncalibrated.",
                                                   "Paired intervals conditional; no multiplicity correction.",
                                                   "Country subgroups use global fold area thresholds."]})
    write_json(marker, {"study_id": study_id, "spec": spec, "holdout_evaluated": False,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "files": {p.relative_to(dest).as_posix(): digests.file(p)
                                  for p in sorted(dest.rglob("*"))
                                  if p.is_file() and ".part" not in p.name and p != marker}})
    log(f"Серия готова: {dest}; слепой тест не оценивался")
    return dest

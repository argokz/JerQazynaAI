"""Positive/background baseline: CV, сохранение модели и карта обучающей территории."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from jerqazyna.config import TaskConfig
from jerqazyna.features.cache import Digests, fingerprint, write_json
from jerqazyna.grid.template import write_layer
from jerqazyna.models.config import ExperimentConfig
from jerqazyna.models.dataset import Dataset, build_dataset
from jerqazyna.models.evaluation import evaluate_oof
from jerqazyna.models.sampling import SpatialFolds, sample_background


def make_estimator(kind: str, config: ExperimentConfig, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if kind == "logistic":
        return make_pipeline(imputer, StandardScaler(), LogisticRegression(
            C=config.logistic_c, class_weight="balanced", max_iter=2000, random_state=seed,
        ))
    if kind == "random_forest":
        return make_pipeline(imputer, RandomForestClassifier(
            **config.forest.model_dump(), class_weight="balanced", max_features="sqrt",
            n_jobs=config.workers, random_state=seed,
        ))
    raise ValueError(f"Неизвестная модель: {kind}")


def _fit(dataset, config, kind, seed, positive_rows, background_rows, prefix):
    import joblib

    rows = np.r_[positive_rows, background_rows].astype(np.int64)
    y = np.r_[np.ones(len(positive_rows), np.uint8), np.zeros(len(background_rows), np.uint8)]
    signature = fingerprint({"kind": kind, "seed": seed, "config": config.model_dump(),
                             "rows_sha256": fingerprint(rows.tolist()),
                             "dataset_id": dataset.metadata["dataset_id"]})
    model_path = prefix.with_suffix(".joblib")
    marker = prefix.with_suffix(".json")
    digests = Digests()
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if (saved["signature"] == signature and digests.file(model_path) == saved["sha256"]
                and digests.file(prefix.with_suffix(".npz")) == saved["sample_sha256"]):
            return joblib.load(model_path)
        raise ValueError(f"Повреждён checkpoint: {prefix}")
    model = make_estimator(kind, config, seed)
    model.fit(np.asarray(dataset.X[rows]), y)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_name(model_path.name + ".part")
    joblib.dump(model, tmp, compress=3)
    tmp.replace(model_path)
    with open(prefix.with_suffix(".npz"), "wb") as stream:
        np.savez_compressed(stream, positive_rows=positive_rows, background_rows=background_rows,
                            positive_cells=dataset.cells[positive_rows],
                            background_cells=dataset.cells[background_rows])
    write_json(marker, {"signature": signature, "sha256": digests.file(model_path),
                        "sample_sha256": digests.file(prefix.with_suffix(".npz")),
                        "n_positive_cells": len(positive_rows), "n_background": len(background_rows)})
    return model


def _predict_rows(model, dataset, rows, batch_size):
    result = np.empty(len(rows), np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        result[start:end] = model.predict_proba(dataset.X[rows[start:end]])[:, 1]
    if not np.isfinite(result).all():
        raise ValueError("Модель вернула неконечные scores")
    return result


def _map_scores(path, dataset, scores, description):
    array = np.full(dataset.grid.width * dataset.grid.height, np.nan, np.float32)
    array[dataset.cells] = scores
    write_layer(path, dataset.grid, array.reshape(dataset.grid.shape), description)


def predict_dataset(estimators, dataset, path, batch_size=65536, log=print):
    """GeoTIFF по окнам, только на территории обучающей таблицы; holdout остаётся NaN."""
    import rasterio
    from rasterio.windows import Window

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.tif")
    grid = dataset.grid
    with rasterio.open(
        tmp, "w", driver="GTiff", width=grid.width, height=grid.height, count=1,
        dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=3,
    ) as dst:
        dst.set_band_description(1, "relative positive/background score; training territory only")
        for r0 in range(0, grid.height, 128):
            r1 = min(r0 + 128, grid.height)
            start = np.searchsorted(dataset.cells, r0 * grid.width)
            end = np.searchsorted(dataset.cells, r1 * grid.width)
            data = np.full((r1 - r0, grid.width), np.nan, np.float32)
            if end > start:
                scores = np.zeros(end - start, np.float64)
                rows = np.arange(start, end)
                for model in estimators:
                    scores += _predict_rows(model, dataset, rows, batch_size)
                data.ravel()[dataset.cells[start:end] - r0 * grid.width] = scores / len(estimators)
            dst.write(data, 1, window=Window(0, r0, grid.width, r1 - r0))
    tmp.replace(path)
    log(f"  карта train: {path}")


def _provenance(root):
    package = Path(__file__).parents[1]
    files = sorted((package / "models").glob("*.py")) + [package / "validation/metrics.py"]
    digests = Digests()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty,
            "source_sha256": {p.relative_to(package).as_posix(): digests.file(p) for p in files},
            "packages": {name: version(name) for name in (
                "numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl",
            )}}


def _run_experiment(
    task: TaskConfig, config: ExperimentConfig, root: Path, split: Path, log=print,
) -> Path:
    import joblib

    root = Path(root)
    if task.validation.success_area_fraction not in task.validation.area_fractions:
        raise ValueError("success_area_fraction должна входить в area_fractions для выбора модели")
    dataset = build_dataset(task, config, root, split, log)
    if config.heuristic_band not in dataset.columns:
        raise ValueError(f"Нет канала геологической эвристики: {config.heuristic_band}")
    spec = {"version": 1, "task": task.model_dump(), "experiment": config.model_dump(),
            "dataset_id": dataset.metadata["dataset_id"], "provenance": _provenance(root)}
    run_id = fingerprint(spec)
    dest = root / "outputs" / task.task_id / config.experiment_id / run_id[:16]
    manifest = dest / "run.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        digests = Digests()
        if saved["spec"] != spec or any(digests.file(dest / p) != digest for p, digest in saved["files"].items()):
            raise ValueError("Артефакты готового эксперимента повреждены")
        log(f"Эксперимент уже готов: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "protocol.json", {"run_id": run_id, "spec": spec, "holdout_evaluated": False})
    folds = SpatialFolds(dataset, task.validation.spatial_block_size_m, config.n_folds, task.seed)
    objects = dataset.labels[
        dataset.labels.status.isin(config.positive_statuses) & (dataset.labels.matrix_row >= 0)
    ].copy()
    positive_rows = objects.matrix_row.to_numpy(dtype=np.int64)
    if positive_rows.size < 2:
        raise ValueError("Недостаточно положительных объектов для обучения")
    objects["fold"] = folds.fold[positive_rows]
    objects["block"] = folds.blocks[positive_rows]
    log(f"== train: {dataset.cells.size:,} ячеек, {len(objects)} объектов разрешённых статусов; "
        f"{len(dataset.labels) - len(objects)} объектов исключены из оценки/положительного класса")
    ensemble_scores = {kind: np.zeros(dataset.cells.size, np.float64) for kind in config.models}
    heuristic = np.empty(dataset.cells.size, np.float32)
    heuristic_col = dataset.columns.index(config.heuristic_band)
    seed_reports = {kind: [] for kind in config.models}
    fits = []
    for seed in config.background_seeds:
        per_seed = {kind: np.full(dataset.cells.size, np.nan, np.float32) for kind in config.models}
        for fold in range(config.n_folds):
            valid_rows = np.flatnonzero(folds.fold == fold)
            pos = folds.training_rows(np.unique(positive_rows), fold, config.cv_buffer_m)
            if pos.size < 2:
                raise ValueError(f"Фолд {fold}: меньше двух положительных ячеек после буфера CV")
            background = sample_background(dataset, config, folds, seed, fold)
            if seed == config.background_seeds[0]:
                available = np.asarray(dataset.X[np.r_[pos, background], heuristic_col])
                median = float(np.nanmedian(available)) if np.isfinite(available).any() else 0.
                values = np.asarray(dataset.X[valid_rows, heuristic_col])
                heuristic[valid_rows] = -np.where(np.isfinite(values), values, median)
            for kind in config.models:
                started = time.monotonic()
                log(f"== {kind}: seed={seed}, fold={fold + 1}/{config.n_folds}, "
                    f"positive={pos.size}, background={background.size}")
                model = _fit(dataset, config, kind, seed, pos, background,
                             dest / "checkpoints" / f"{kind}_seed{seed}_fold{fold}")
                per_seed[kind][valid_rows] = _predict_rows(model, dataset, valid_rows, config.batch_size)
                fits.append({"model": kind, "seed": seed, "fold": fold,
                             "positive_cells": int(pos.size), "background_cells": int(background.size),
                             "validation_cells": int(valid_rows.size),
                             "validation_objects": int((objects.fold == fold).sum())})
                log(f"  завершено за {time.monotonic() - started:.1f} с")
        for kind in config.models:
            ensemble_scores[kind] += per_seed[kind]
            report = evaluate_oof(per_seed[kind], positive_rows, folds, task.validation.area_fractions,
                                  task.validation.confidence, config.grouped_bootstrap_iterations, task.seed)
            seed_reports[kind].append({"seed": seed, **report})
            objects[f"{kind}_seed{seed}"] = per_seed[kind][positive_rows]
    reports = {}
    for kind in config.models:
        ensemble_scores[kind] = (ensemble_scores[kind] / len(config.background_seeds)).astype(np.float32)
        reports[kind] = evaluate_oof(ensemble_scores[kind], positive_rows, folds,
                                     task.validation.area_fractions, task.validation.confidence,
                                     config.grouped_bootstrap_iterations, task.seed)
        reports[kind]["background_seed_reports"] = seed_reports[kind]
        objects[f"{kind}_ensemble"] = ensemble_scores[kind][positive_rows]
        _map_scores(dest / f"oof_{kind}.tif", dataset, ensemble_scores[kind], "out-of-fold score; train only")
    heuristic_report = evaluate_oof(heuristic, positive_rows, folds, task.validation.area_fractions,
                                     task.validation.confidence, config.grouped_bootstrap_iterations, task.seed)
    selection_fraction = task.validation.success_area_fraction

    def capture(kind):
        return next(m["capture"] for m in reports[kind]["metrics"] if m["area_fraction"] == selection_fraction)

    selected = max(config.models, key=capture)  # При равенстве — первый в фиксированном списке.
    heuristic_capture = next(m["capture"] for m in heuristic_report["metrics"]
                             if m["area_fraction"] == selection_fraction)
    summary = {
        "run_id": run_id, "holdout_evaluated": False, "provisional_holdout": task.validation.blind_region.provisional,
        "selected_classifier": selected, "selection_area_fraction": selection_fraction,
        "selection_policy": "max_ensemble_cv_capture_then_config_order",
        "improves_over_heuristic": capture(selected) > heuristic_capture,
        "models": reports, "heuristic": heuristic_report,
        "random_baseline": {str(a): a for a in task.validation.area_fractions},
        "n_training_objects": len(dataset.labels), "n_evaluated_objects": len(objects),
        "n_excluded_objects": len(dataset.labels) - len(objects),
        "positive_statuses": config.positive_statuses, "fits": fits,
        "score_meaning": "relative positive/background classification score, not calibrated deposit probability",
        "caveats": ["Background is unlabeled and may contain deposits.",
                    "CV model selection is not an independent final test.",
                    "Spatial blocks do not replace independent geological belts."],
    }
    write_json(dest / "cv.json", summary)
    objects.to_csv(dest / "oof_objects.csv", index=False, encoding="utf-8")
    _map_scores(dest / "oof_heuristic.tif", dataset, heuristic, "distance-to-intrusions heuristic; train only")
    _map_scores(dest / "cv_fold.tif", dataset, folds.fold.astype(np.float32), "spatial CV fold; train only")
    estimators = []
    for seed in config.background_seeds:
        log(f"== финальная фиксация {selected}, seed={seed}; только train")
        background = sample_background(dataset, config, folds, seed)
        estimators.append(_fit(dataset, config, selected, seed, np.unique(positive_rows), background,
                               dest / "checkpoints" / f"{selected}_seed{seed}_final"))
    bundle = {"estimators": estimators, "columns": dataset.columns, "dataset_id": dataset.metadata["dataset_id"],
              "selected_classifier": selected, "config": config.model_dump(), "holdout_evaluated": False}
    joblib.dump(bundle, dest / "model.joblib.part", compress=3)
    (dest / "model.joblib.part").replace(dest / "model.joblib")
    predict_dataset(estimators, dataset, dest / "train_score.tif", config.batch_size, log)
    digests = Digests()
    artifacts = [p for p in sorted(dest.rglob("*")) if p.is_file() and ".part" not in p.name and p != manifest]
    write_json(manifest, {
        "run_id": run_id, "spec": spec, "dataset_path": dataset.path.relative_to(root).as_posix(),
        "completed_at": datetime.now(timezone.utc).isoformat(), "holdout_evaluated": False,
        "files": {p.relative_to(dest).as_posix(): digests.file(p) for p in artifacts},
    })
    log(f"Эксперимент готов: {dest}; выбран {selected}; тест Прибалхашья не оценивался")
    return dest


def run_experiment(task: TaskConfig, config: ExperimentConfig, root: Path, split: Path, log=print):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=config.workers):
        return _run_experiment(task, config, root, split, log)


def predict_run(run: Path, root: Path, output: Path, log=print):
    import joblib

    run, root = Path(run), Path(root)
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    digests = Digests()
    if digests.file(run / "model.joblib") != metadata["files"]["model.joblib"]:
        raise ValueError("Повреждена сохранённая модель")
    dataset = Dataset(root / metadata["dataset_path"])
    bundle = joblib.load(run / "model.joblib")
    if bundle["dataset_id"] != dataset.metadata["dataset_id"] or bundle["columns"] != dataset.columns:
        raise ValueError("Таблица не соответствует сохранённой модели")
    predict_dataset(bundle["estimators"], dataset, output, bundle["config"]["batch_size"], log)

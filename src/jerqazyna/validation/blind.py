"""Однократный слепой тест по предрегистрированному протоколу.

Модель, признаки и пороги берутся только из протокола и зафиксированного прогона.
Признаки теста читаются из тех же файлов, что и обучающая таблица; их хеши сверяются.
Готовый отчёт повторно не пересчитывается: повторный запуск только проверяет его хеши.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import binomtest

from jerqazyna.config import load_task
from jerqazyna.features.cache import Digests, write_json
from jerqazyna.features.prepare import features_dir
from jerqazyna.grid.template import read_layer, write_layer
from jerqazyna.validation.metrics import capture_probabilities, detection_curve, summarize_capture
from jerqazyna.validation.protocol import BlindProtocol, load_protocol, verify_frozen_inputs

# Ячейка относится к тракту USGS, если тракт покрывает большую часть её площади.
TRACT_MAJORITY = 0.5


def _hits(scores: np.ndarray, rows: np.ndarray, fraction: float) -> np.ndarray:
    """Вероятность попадания каждого объекта; объект без покрытия признаков — промах."""
    result = np.zeros(rows.size)
    covered = rows >= 0
    if covered.any():
        result[covered] = capture_probabilities(scores, rows[covered], fraction)
    return result


def area_for_capture(scores: np.ndarray, rows: np.ndarray, target: float) -> float | None:
    """Минимальная доля площади, на которой ожидаемый захват достигает target."""
    covered = rows >= 0
    if not covered.any():
        return None
    area, captured = detection_curve(scores, rows[covered])
    captured = captured * covered.sum() / rows.size
    reached = np.flatnonzero(captured >= target - 1e-12)
    if not reached.size:
        return None
    i = int(reached[0])
    if i == 0:
        return 0.0
    share = (target - captured[i - 1]) / (captured[i] - captured[i - 1])
    return float(area[i - 1] + share * (area[i] - area[i - 1]))


def _curve(scores, rows, protocol: BlindProtocol) -> dict:
    ev = protocol.evaluation
    records = []
    for fraction in sorted(ev.descriptive_area_fractions + [ev.primary_area_fraction]):
        estimate = asdict(summarize_capture(_hits(scores, rows, fraction), fraction, ev.confidence))
        estimate["lift"] = estimate["capture"] / fraction
        records.append(estimate)
    return {"capture": records,
            "area_for_capture": {str(t): area_for_capture(scores, rows, t) for t in ev.capture_targets}}


def _verdict(results: dict, protocol: BlindProtocol) -> str:
    roles = {h.role: h.id for h in protocol.hypotheses}
    if not results[roles["primary"]]["passed"]:
        return "not_transferred"
    secondary = roles.get("secondary")
    if secondary and results[secondary]["passed"]:
        return "beats_random_and_heuristic"
    return "beats_random_not_heuristic"


def evaluate_blind(scores, heuristic, tracts, positive_rows, protocol: BlindProtocol) -> dict:
    """Проверки протокола на ячейках теста. positive_rows = -1 для объекта без покрытия."""
    scores = np.asarray(scores, dtype=float)
    heuristic = np.asarray(heuristic, dtype=float)
    tracts = np.asarray(tracts)
    rows = np.asarray(positive_rows, dtype=np.int64)
    if rows.size != protocol.objects.n_positive:
        raise ValueError(f"Объектов {rows.size}, протокол ожидает {protocol.objects.n_positive}")
    if not (scores.shape == heuristic.shape == tracts.shape):
        raise ValueError("Scores, эвристика и эталон должны быть заданы на одних ячейках")
    ev = protocol.evaluation
    primary = ev.primary_area_fraction
    model_hits, heuristic_hits = _hits(scores, rows, primary), _hits(heuristic, rows, primary)
    # Консервативно для проверок: модели — только гарантированные попадания, эвристике — возможные.
    model_hit, heuristic_hit = model_hits == 1, heuristic_hits > 0
    n = rows.size
    results = {}
    for hypothesis in protocol.hypotheses:
        if hypothesis.test == "exact_binomial_one_sided":
            k = int(model_hit.sum())
            p = binomtest(k, n, hypothesis.null_capture, alternative="greater").pvalue
            results[hypothesis.id] = {"k": k, "n": n, "null_capture": hypothesis.null_capture,
                                      "p_value": float(p), "passed": bool(p < hypothesis.alpha)}
        elif hypothesis.test == "exact_mcnemar_one_sided":
            model_only = int((model_hit & ~heuristic_hit).sum())
            heuristic_only = int((~model_hit & heuristic_hit).sum())
            discordant = model_only + heuristic_only
            p = binomtest(model_only, discordant, .5, alternative="greater").pvalue if discordant else 1.
            results[hypothesis.id] = {"model_only": model_only, "heuristic_only": heuristic_only,
                                      "p_value": float(p), "passed": bool(p < hypothesis.alpha),
                                      "low_power": hypothesis.low_power}
        elif hypothesis.role == "reference":
            lower = summarize_capture(model_hits, primary, ev.confidence).lower
            results[hypothesis.id] = {"lower": lower, "threshold": hypothesis.threshold,
                                      "passed": bool(lower >= hypothesis.threshold)}
        else:
            area = float(tracts.mean())
            tract_hits = np.where(rows >= 0, tracts[np.maximum(rows, 0)], 0)
            results[hypothesis.id] = {
                "tract_area_fraction": area, "tract_capture": float(tract_hits.mean()),
                "model_capture": float(_hits(scores, rows, area).mean()) if area > 0 else None,
                "tract_majority": TRACT_MAJORITY,
            }
    return {"n_positive": n, "n_uncovered": int((rows < 0).sum()), "n_cells": int(scores.size),
            "hypotheses": results, "verdict": _verdict(results, protocol),
            "model": _curve(scores, rows, protocol), "heuristic": _curve(heuristic, rows, protocol),
            "random": {str(t): t for t in ev.capture_targets}}


def _git_state(root: Path) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def _verify_report(dest: Path, protocol_sha: str) -> None:
    saved = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    if saved["protocol_sha256"] != protocol_sha:
        raise ValueError("Отчёт получен по другой версии протокола; результат исследовательский")
    digests = Digests()
    for name, digest in saved["files"].items():
        if digests.file(dest / name) != digest:
            raise ValueError(f"Повреждён артефакт слепого теста: {name}")


def _test_features(train, split: Path, out: Path, min_valid_fraction: float, digests: Digests):
    """Матрица признаков теста из тех же файлов, что и train, с тем же правилом покрытия."""
    feature_files = train.metadata["spec"]["feature_files"]
    if set(feature_files) != {f"{column}.tif" for column in train.columns}:
        raise ValueError("Протокол не поддерживает модели с маской выработок или спектром")
    test = read_layer(split / "test.tif").astype(bool).ravel()
    cells = np.flatnonzero(test)
    X = np.empty((cells.size, len(train.columns)), np.float32)
    counts = np.zeros(cells.size, np.uint32)
    for col, column in enumerate(train.columns):
        name = f"{column}.tif"
        if digests.file(out / name) != feature_files[name]:
            raise ValueError(f"{name}: слой изменился после обучения модели")
        values = read_layer(out / name).ravel()[cells].astype(np.float32)
        values[~np.isfinite(values)] = np.nan
        X[:, col] = values
        counts += np.isfinite(values)
    valid = counts >= max(1, math.ceil(len(train.columns) * min_valid_fraction))
    return cells, cells[valid], X[valid]


def _test_objects(split: Path, width: int, cells: np.ndarray, statuses) -> tuple[list[dict], list[dict]]:
    from jerqazyna.models.dataset import map_cells

    with open(split / "labels.csv", encoding="utf-8", newline="") as stream:
        test = [row for row in csv.DictReader(stream) if row["partition"] == "test"]
    index = np.asarray([int(r["row"]) * width + int(r["col"]) for r in test], np.int64)
    matrix_rows = map_cells(cells, index) if test else np.empty(0, np.int64)
    for row, matrix_row in zip(test, matrix_rows):
        row["matrix_row"] = int(matrix_row)
    positive = [r for r in test if r["status"] in statuses]
    return positive, [r for r in test if r["status"] not in statuses]


def _rank_share(scores: np.ndarray, row: int) -> float | None:
    """Доля площади теста со строго большим score; меньше — выше в ранжировании."""
    return None if row < 0 else float((scores > scores[row]).mean())


def _write_markdown(path: Path, report: dict) -> None:
    result = report["evaluation"]
    lines = [f"# Слепой тест: {report['protocol_id']}", "",
             (f"Вердикт: **{result['verdict']}**. Объектов: {result['n_positive']}, "
              f"без покрытия: {result['n_uncovered']}. Ячеек теста: {result['n_cells']:,}."), "",
             "| Гипотеза | Результат |", "|---|---|"]
    lines += [f"| {key} | {json.dumps(value, ensure_ascii=False)} |"
              for key, value in result["hypotheses"].items()]
    lines += ["", "| Ранжирование | Доля площади | Захват | exact 95% | Lift |", "|---|---:|---:|---|---:|"]
    for name in ("model", "heuristic"):
        for m in result[name]["capture"]:
            lines.append(f"| {name} | {m['area_fraction']:.0%} | {m['capture']:.3f} | "
                         f"{m['lower']:.3f}–{m['upper']:.3f} | {m['lift']:.2f} |")
    lines += ["", f"Протокол SHA-256: `{report['protocol_sha256']}`.",
              "Граница региона предварительная; score — относительный ранг, не вероятность."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_blind_test(protocol_path: str | Path, root: Path, log=print) -> Path:
    import joblib

    from jerqazyna.models.dataset import Dataset, verify_split
    from jerqazyna.models.pipeline import _predict_rows

    root = Path(root)
    protocol, protocol_sha = load_protocol(protocol_path)
    task = load_task(root / protocol.task_config)
    out = features_dir(root, task)
    split = out / "splits" / protocol.split_id[:16]
    dest = root / "outputs" / task.task_id / "blind" / protocol.protocol_id
    if (dest / "report.json").exists():
        _verify_report(dest, protocol_sha)
        log(f"Слепой тест уже выполнен, повторной оценки нет: {dest}")
        return dest

    digests = Digests()
    frozen = verify_frozen_inputs(protocol, root, split)
    verify_split(task, root, split, digests)
    run_dir = root / protocol.frozen_model.run_dir
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    log("== проверка обучающей таблицы и модели")
    train = Dataset(root / run["dataset_path"])
    bundle = joblib.load(run_dir / "model.joblib")
    ev = protocol.evaluation
    if bundle["dataset_id"] != train.metadata["dataset_id"] or bundle["columns"] != train.columns:
        raise ValueError("Модель не соответствует обучающей таблице")
    if bundle["config"]["min_valid_fraction"] != ev.min_valid_fraction:
        raise ValueError("Правило покрытия протокола отличается от обучения")
    if ev.heuristic.band not in train.columns:
        raise ValueError(f"Нет канала эвристики: {ev.heuristic.band}")
    tract_file = out / f"{ev.benchmark.layer}__fraction.tif"

    # Отметка открытия пишется до чтения признаков теста; повторные попытки сохраняются.
    dest.mkdir(parents=True, exist_ok=True)
    opened = dest / "opened.json"
    attempts = json.loads(opened.read_text(encoding="utf-8"))["attempts"] if opened.exists() else []
    attempts.append({"started_at": datetime.now(timezone.utc).isoformat(), **_git_state(root)})
    write_json(opened, {"protocol_id": protocol.protocol_id, "protocol_sha256": protocol_sha,
                        "frozen_inputs": frozen, "attempts": attempts})

    log("== признаки теста")
    all_cells, cells, X = _test_features(train, split, out, ev.min_valid_fraction, digests)
    log(f"  ячеек теста {all_cells.size:,}, с достаточным покрытием {cells.size:,}")
    log("== прогноз зафиксированной модели")
    table = SimpleNamespace(X=X)
    rows = np.arange(cells.size)
    scores = np.zeros(cells.size, np.float64)
    for model in bundle["estimators"]:
        scores += _predict_rows(model, table, rows, bundle["config"]["batch_size"])
    scores /= len(bundle["estimators"])

    column = train.columns.index(ev.heuristic.band)
    median = float(np.nanmedian(np.asarray(train.X[:, column])))
    values = X[:, column]
    heuristic = -np.where(np.isfinite(values), values, median)
    tract_fraction = read_layer(tract_file).ravel()[cells]
    tracts = (np.nan_to_num(tract_fraction, nan=0.) >= TRACT_MAJORITY).astype(np.uint8)

    positive, other = _test_objects(split, train.grid.width, cells, protocol.objects.positive_statuses)
    positive_rows = np.asarray([r["matrix_row"] for r in positive], np.int64)
    log("== проверки протокола")
    evaluation = evaluate_blind(scores, heuristic, tracts, positive_rows, protocol)

    primary = ev.primary_area_fraction
    model_hits, heuristic_hits = _hits(scores, positive_rows, primary), _hits(heuristic, positive_rows, primary)
    fields = ["name", "status", "country", "lon", "lat", "covered", "model_rank_share",
              "heuristic_rank_share", "model_hit_primary", "heuristic_hit_primary", "in_tract"]
    with open(dest / "objects.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, obj in enumerate(positive + other):
            r = obj["matrix_row"]
            writer.writerow({
                "name": obj["name"], "status": obj["status"] or "unknown", "country": obj["country"],
                "lon": obj["lon"], "lat": obj["lat"], "covered": r >= 0,
                "model_rank_share": _rank_share(scores, r), "heuristic_rank_share": _rank_share(heuristic, r),
                "model_hit_primary": float(model_hits[i]) if i < len(positive) else "",
                "heuristic_hit_primary": float(heuristic_hits[i]) if i < len(positive) else "",
                "in_tract": bool(tracts[r]) if r >= 0 else "",
            })
    grid = train.grid
    array = np.full(grid.width * grid.height, np.nan, np.float32)
    array[cells] = scores
    write_layer(dest / "test_score.tif", grid, array.reshape(grid.shape),
                "relative positive/background score; blind region only")
    report = {
        "protocol_id": protocol.protocol_id, "protocol_sha256": protocol_sha,
        "completed_at": datetime.now(timezone.utc).isoformat(), **_git_state(root),
        "frozen_inputs": frozen, "region_provisional": protocol.region_provisional,
        "n_test_cells": int(all_cells.size), "n_covered_cells": int(cells.size),
        "heuristic_nan_fill": median, "benchmark_sha256": digests.file(tract_file),
        "evaluation": evaluation,
        "score_meaning": "relative positive/background score, not calibrated deposit probability",
    }
    _write_markdown(dest / "report.md", report)
    report["files"] = {name: digests.file(dest / name)
                       for name in ("objects.csv", "test_score.tif", "report.md", "opened.json")}
    write_json(dest / "report.json", report)
    log(f"Слепой тест завершён: {dest}; вердикт {evaluation['verdict']}")
    return dest

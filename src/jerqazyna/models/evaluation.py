"""Отчёт по доле площади внутри каждого фолда; шкалы scores между фолдами не смешиваются."""

from dataclasses import asdict

import numpy as np

from jerqazyna.validation.metrics import capture_probabilities, summarize_capture


def grouped_interval(hit, groups, iterations=2000, confidence=.95, seed=42):
    """Бутстрэп по блокам с объектами. Учитывает их совместное попадание внутри блока.

    Условен на наблюдаемых объектах и не устраняет смещение изученности. Exact-интервал
    публикуется отдельно, поэтому вырождение этого бутстрэпа видно в отчёте.
    """
    hit, groups = np.asarray(hit), np.asarray(groups)
    if not hit.size or hit.shape != groups.shape:
        raise ValueError("Нет объектов или неверные группы для бутстрэпа")
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=hit)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique.size, size=(iterations, unique.size))
    samples = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {"lower": float(low), "upper": float(high), "n_positive_blocks": int(unique.size),
            "method": "bootstrap_positive_spatial_blocks", "conditional_on_observed_objects": True}


def evaluate_oof(scores, positive_rows, folds, fractions, confidence, iterations, seed):
    """Порог верхней площади выбирается отдельно для каждого validation-фолда."""
    positive_rows = np.asarray(positive_rows, dtype=np.int64)
    records = []
    details = []
    for fraction in fractions:
        hit = np.empty(positive_rows.size)
        for fold in range(folds.n_folds):
            rows = np.flatnonzero(folds.fold == fold)
            which = folds.fold[positive_rows] == fold
            local_objects = np.searchsorted(rows, positive_rows[which])
            if not np.isfinite(scores[rows]).all():
                raise ValueError("Неполные OOF-прогнозы")
            if local_objects.size:
                current = capture_probabilities(scores[rows], local_objects, fraction)
                hit[which] = current
                details.append({"fold": fold, "area_fraction": fraction, "n_cells": rows.size,
                                "n_objects": int(which.sum()), "capture": float(current.mean())})
            else:
                details.append({"fold": fold, "area_fraction": fraction, "n_cells": rows.size,
                                "n_objects": 0, "capture": None})
        estimate = asdict(summarize_capture(hit, fraction, confidence))
        estimate["lift"] = estimate["capture"] / fraction
        estimate["grouped_interval"] = grouped_interval(
            hit, folds.blocks[positive_rows], iterations, confidence, seed,
        )
        records.append(estimate)
    return {"metrics": records, "folds": details,
            "area_policy": "same_fraction_within_each_validation_fold",
            "interval_note": "Exact assumes independent objects; block bootstrap is conditional on observed positive blocks."}

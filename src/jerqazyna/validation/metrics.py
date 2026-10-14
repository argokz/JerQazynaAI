"""Кривая обнаружения: доля найденных объектов в зависимости от доли просмотренной площади.

Ячейки сортируются по убыванию перспективности; объект считается найденным, если его ячейка
попала в верхнюю долю площади. Сетка в равновеликой проекции, поэтому доля ячеек = доля площади.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cell_ranks(scores: np.ndarray) -> np.ndarray:
    """Доля площади, которую нужно просмотреть, чтобы дойти до каждой ячейки (0..1]."""
    scores = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1) / scores.size
    return ranks


def detection_curve(
    scores: np.ndarray, deposit_cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Точки кривой: (доля площади, доля найденных объектов), по одной на объект."""
    deposit_cells = np.asarray(deposit_cells, dtype=int)
    if deposit_cells.size == 0:
        raise ValueError("Нет объектов для оценки")
    area = np.sort(_cell_ranks(scores)[deposit_cells])
    captured = np.arange(1, area.size + 1) / area.size
    return area, captured


def capture_at(scores: np.ndarray, deposit_cells: np.ndarray, area_fraction: float) -> float:
    """Доля объектов в верхних area_fraction площади."""
    deposit_cells = np.asarray(deposit_cells, dtype=int)
    if deposit_cells.size == 0:
        raise ValueError("Нет объектов для оценки")
    return float(np.mean(_cell_ranks(scores)[deposit_cells] <= area_fraction))


@dataclass(frozen=True)
class CaptureEstimate:
    area_fraction: float
    capture: float
    lower: float
    upper: float
    n_deposits: int
    random_baseline: float

    def passes(self, lower_bound: float) -> bool:
        return self.lower >= lower_bound


def bootstrap_capture(
    scores: np.ndarray,
    deposit_cells: np.ndarray,
    area_fraction: float,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> CaptureEstimate:
    """Доля найденных объектов с бутстрэп-интервалом по объектам.

    При десятках объектов точечная оценка ненадёжна, поэтому критерий успеха
    проверяется по нижней границе интервала.
    """
    deposit_cells = np.asarray(deposit_cells, dtype=int)
    if deposit_cells.size == 0:
        raise ValueError("Нет объектов для оценки")
    hit = (_cell_ranks(scores)[deposit_cells] <= area_fraction).astype(float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(hit, size=(iterations, hit.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return CaptureEstimate(
        area_fraction=area_fraction,
        capture=float(hit.mean()),
        lower=float(lower),
        upper=float(upper),
        n_deposits=int(hit.size),
        random_baseline=area_fraction,
    )


def capture_at_matched_area(
    scores: np.ndarray, deposit_cells: np.ndarray, benchmark_mask: np.ndarray
) -> tuple[float, float, float]:
    """Сравнение с эталоном (например, зонами USGS) при равной площади.

    Возвращает (доля площади эталона, захват эталоном, захват моделью на той же площади).
    """
    benchmark_mask = np.asarray(benchmark_mask, dtype=bool).ravel()
    deposit_cells = np.asarray(deposit_cells, dtype=int)
    area = float(benchmark_mask.mean())
    benchmark_capture = float(benchmark_mask[deposit_cells].mean())
    return area, benchmark_capture, capture_at(scores, deposit_cells, area)

"""Захват объектов при фиксированной площади с явной обработкой равных scores.

На границе группы равных прогнозов используется ожидаемый захват при равномерном
отборе внутри группы. Интервалы дополнительно охватывают неоднозначность этого отбора.
Непригодные ячейки нужно исключить явно до вызова; NaN нельзя скрыто включать в площадь.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


def _inputs(scores, deposit_cells) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).ravel()
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError("scores должны быть непустыми и конечными; задайте маску оценки явно")
    cells = np.asarray(deposit_cells)
    if cells.ndim != 1 or not cells.size:
        raise ValueError("Нет объектов для оценки или deposit_cells не одномерен")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("Индексы объектов должны быть целыми")
    if (cells < 0).any() or (cells >= scores.size).any():
        raise ValueError("Индекс объекта вне области оценки")
    return scores, cells


def _fraction(value: float) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Доля площади должна быть в диапазоне [0, 1]")


def _expected_hits(scores: np.ndarray, cells: np.ndarray, fraction: float) -> np.ndarray:
    _fraction(fraction)
    ordered = np.sort(scores)
    left = np.searchsorted(ordered, scores[cells], side="left")
    right = np.searchsorted(ordered, scores[cells], side="right")
    greater = scores.size - right
    return np.clip((fraction * scores.size - greater) / (right - left), 0, 1)


def detection_curve(scores: np.ndarray, deposit_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ломаная ожидаемого захвата по границам групп равных прогнозов, включая (0,0)."""
    scores, cells = _inputs(scores, deposit_cells)
    values, counts = np.unique(scores, return_counts=True)
    groups = np.searchsorted(values, scores[cells])
    objects = np.bincount(groups, minlength=values.size)
    area = np.r_[0.0, np.cumsum(counts[::-1]) / scores.size]
    captured = np.r_[0.0, np.cumsum(objects[::-1]) / cells.size]
    return area, captured


def capture_at(scores: np.ndarray, deposit_cells: np.ndarray, area_fraction: float) -> float:
    """Ожидаемая доля объектов на заданной площади; не зависит от порядка ячеек."""
    scores, cells = _inputs(scores, deposit_cells)
    return float(_expected_hits(scores, cells, area_fraction).mean())


def capture_probabilities(scores, deposit_cells, area_fraction) -> np.ndarray:
    """Ожидаемые hit для каждого объекта, для объединения независимых CV-фолдов."""
    scores, cells = _inputs(scores, deposit_cells)
    return _expected_hits(scores, cells, area_fraction)


@dataclass(frozen=True)
class CaptureEstimate:
    area_fraction: float
    capture: float
    lower: float
    upper: float
    n_deposits: int
    random_baseline: float
    expected_hits: float
    interval_method: str

    def passes(self, lower_bound: float) -> bool:
        _fraction(lower_bound)
        return self.lower >= lower_bound


def _exact_bounds(hit: np.ndarray, confidence: float) -> tuple[float, float]:
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence должна быть между 0 и 1")
    # При ties не превращаем дробные ожидаемые hit в фиктивные биномиальные наблюдения.
    # Огибающая интервалов для гарантированных и возможных попаданий консервативна.
    certain = int(np.count_nonzero(hit == 1))
    possible = int(np.count_nonzero(hit > 0))
    lower = binomtest(certain, hit.size).proportion_ci(confidence, method="exact").low
    upper = binomtest(possible, hit.size).proportion_ci(confidence, method="exact").high
    return float(lower), float(upper)


def estimate_capture(
    scores: np.ndarray, deposit_cells: np.ndarray, area_fraction: float, confidence: float = 0.95,
) -> CaptureEstimate:
    """Exact биномиальный интервал с огибающей для ties.

    Это базовая оценка при независимости объектов. Пространственная зависимость
    и смещение изученности требуют отдельных проверок по блокам/поясам.
    """
    scores, cells = _inputs(scores, deposit_cells)
    hit = _expected_hits(scores, cells, area_fraction)
    return summarize_capture(hit, area_fraction, confidence)


def summarize_capture(hit: np.ndarray, area_fraction: float, confidence: float = .95):
    """Объединение hit, рассчитанных при одинаковой доле площади внутри каждого фолда."""
    _fraction(area_fraction)
    hit = np.asarray(hit, dtype=float)
    if (hit.ndim != 1 or not hit.size or not np.isfinite(hit).all()
            or (hit < 0).any() or (hit > 1).any()):
        raise ValueError("hit должен быть непустым одномерным массивом вероятностей")
    lower, upper = _exact_bounds(hit, confidence)
    return CaptureEstimate(area_fraction, float(hit.mean()), lower, upper, hit.size,
                           area_fraction, float(hit.sum()), "exact_tie_envelope")


def bootstrap_capture(
    scores: np.ndarray, deposit_cells: np.ndarray, area_fraction: float,
    iterations: int = 2000, confidence: float = 0.95, seed: int = 42,
) -> CaptureEstimate:
    """Бутстрэп по объектам с нижней границей ширины из exact-интервала.

    Оставлен для сравнения с прежними экспериментами. Не является пространственным
    бутстрэпом. Вырожденные выборки 1/1 и 0/1 больше не дают интервал нулевой ширины.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations должно быть положительным целым числом")
    scores, cells = _inputs(scores, deposit_cells)
    hit = _expected_hits(scores, cells, area_fraction)
    exact_lower, exact_upper = _exact_bounds(hit, confidence)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations)
    batch = max(1, 1_000_000 // hit.size)
    for start in range(0, iterations, batch):
        stop = min(start + batch, iterations)
        samples[start:stop] = rng.choice(hit, size=(stop - start, hit.size)).mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return CaptureEstimate(
        area_fraction, float(hit.mean()), min(float(lower), exact_lower),
        max(float(upper), exact_upper), cells.size, area_fraction, float(hit.sum()),
        "object_bootstrap_with_exact_envelope",
    )


def capture_at_matched_area(
    scores: np.ndarray, deposit_cells: np.ndarray, benchmark_mask: np.ndarray,
) -> tuple[float, float, float]:
    """(Площадь эталона, захват эталона, ожидаемый захват модели) на одной области."""
    scores, cells = _inputs(scores, deposit_cells)
    mask = np.asarray(benchmark_mask).ravel()
    if mask.size != scores.size or not np.isin(mask, [0, 1]).all():
        raise ValueError("Маска эталона должна быть бинарной и совпадать со scores по размеру")
    area = float(mask.mean())
    return area, float(mask[cells].mean()), float(_expected_hits(scores, cells, area).mean())

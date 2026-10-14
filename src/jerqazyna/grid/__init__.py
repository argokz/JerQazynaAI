"""Единая сетка ячеек в равновеликой проекции."""

from __future__ import annotations

import numpy as np


def make_grid(
    bounds: tuple[float, float, float, float], cell_size_m: float
) -> tuple[np.ndarray, tuple[int, int]]:
    """Центры ячеек регулярной сетки.

    bounds — (xmin, ymin, xmax, ymax) в метрах равновеликой проекции.
    Возвращает координаты центров формы (n, 2) по строкам сверху вниз и форму сетки (rows, cols).
    """
    xmin, ymin, xmax, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Некорректные границы сетки")
    cols = int(np.ceil((xmax - xmin) / cell_size_m))
    rows = int(np.ceil((ymax - ymin) / cell_size_m))
    xs = xmin + (np.arange(cols) + 0.5) * cell_size_m
    ys = ymax - (np.arange(rows) + 0.5) * cell_size_m
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()]), (rows, cols)

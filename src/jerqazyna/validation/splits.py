"""Пространственные разбиения: слепой регион с буфером и блочная кросс-валидация."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree


def blind_region_split(
    xy: np.ndarray, in_blind: np.ndarray, buffer_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Маски (обучение, тест) для теста «слепого региона».

    Ячейки ближе buffer_m к слепому региону не попадают ни в обучение, ни в тест:
    иначе из-за пространственной автокорреляции информация утекает через границу.
    xy — координаты центров ячеек в метрах (равновеликая проекция), форма (n, 2).
    """
    xy = np.asarray(xy, dtype=float)
    in_blind = np.asarray(in_blind, dtype=bool)
    if xy.shape != (in_blind.size, 2):
        raise ValueError("xy должен иметь форму (n, 2) и совпадать по длине с in_blind")
    if not in_blind.any():
        raise ValueError("Слепой регион пуст")
    outside = ~in_blind
    dist = np.full(in_blind.size, np.inf)
    dist[outside], _ = cKDTree(xy[in_blind]).query(xy[outside], distance_upper_bound=buffer_m)
    train = outside & (dist > buffer_m)
    return train, in_blind.copy()


def spatial_blocks(xy: np.ndarray, block_size_m: float) -> np.ndarray:
    """Номер квадратного блока для каждой ячейки."""
    xy = np.asarray(xy, dtype=float)
    ij = np.floor((xy - xy.min(axis=0)) / block_size_m).astype(np.int64)
    _, block_id = np.unique(ij, axis=0, return_inverse=True)
    return block_id.ravel()


def spatial_block_folds(
    xy: np.ndarray, block_size_m: float, n_folds: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Блочная кросс-валидация: целые блоки уходят в тест, индексы (train, test)."""
    blocks = spatial_blocks(xy, block_size_m)
    unique = np.unique(blocks)
    if unique.size < n_folds:
        raise ValueError(f"Блоков ({unique.size}) меньше, чем фолдов ({n_folds})")
    rng = np.random.default_rng(seed)
    fold_of_block = np.empty(unique.size, dtype=int)
    fold_of_block[rng.permutation(unique.size)] = np.arange(unique.size) % n_folds
    fold = fold_of_block[np.searchsorted(unique, blocks)]
    for k in range(n_folds):
        yield np.flatnonzero(fold != k), np.flatnonzero(fold == k)

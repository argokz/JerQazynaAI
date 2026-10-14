import numpy as np
import pytest

from jerqazyna.grid import make_grid
from jerqazyna.validation import blind_region_split, spatial_block_folds


@pytest.fixture
def grid():
    xy, shape = make_grid((0, 0, 100_000, 100_000), 1000)
    return xy, shape


def test_make_grid_shape(grid):
    xy, shape = grid
    assert shape == (100, 100)
    assert xy.shape == (10_000, 2)
    assert xy[0].tolist() == [500.0, 99_500.0]


def test_blind_region_split_has_buffer(grid):
    xy, _ = grid
    in_blind = xy[:, 0] > 60_000
    train, test = blind_region_split(xy, in_blind, buffer_m=10_000)
    assert not (train & test).any()
    assert test.sum() == in_blind.sum()
    # ни одна обучающая ячейка не ближе 10 км к слепому региону
    assert xy[train, 0].max() < 60_000 - 10_000 + 1000
    excluded = ~train & ~test
    assert excluded.any()


def test_blind_region_empty_raises(grid):
    xy, _ = grid
    with pytest.raises(ValueError):
        blind_region_split(xy, np.zeros(len(xy), dtype=bool), 10_000)


def test_spatial_block_folds_keep_blocks_whole(grid):
    xy, _ = grid
    seen = np.zeros(len(xy), dtype=int)
    for train_idx, test_idx in spatial_block_folds(xy, 20_000, n_folds=5, seed=0):
        assert np.intersect1d(train_idx, test_idx).size == 0
        seen[test_idx] += 1
        test_blocks = {tuple(b) for b in np.floor(xy[test_idx] / 20_000).astype(int)}
        train_blocks = {tuple(b) for b in np.floor(xy[train_idx] / 20_000).astype(int)}
        assert test_blocks.isdisjoint(train_blocks)
    assert (seen == 1).all()

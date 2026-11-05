"""Пространственные блоки и воспроизводимая выборка непомеченного фона."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from jerqazyna.models.config import ExperimentConfig
from jerqazyna.models.dataset import Dataset


@dataclass
class SpatialFolds:
    dataset: Dataset
    block_size_m: float
    n_folds: int
    seed: int

    def __post_init__(self):
        from shapely import STRtree, box

        grid = self.dataset.grid
        if not np.isfinite(self.block_size_m) or self.block_size_m <= 0:
            raise ValueError("Размер пространственного блока должен быть положительным")
        self.nx = int(np.ceil(grid.width * grid.cell_size / self.block_size_m))
        self.y_min = grid.bounds[1]
        xy = self.xy(self.dataset.cells)
        self.blocks = self.block_ids(xy).astype(np.int32)
        unique = np.unique(self.blocks)
        if unique.size < self.n_folds:
            raise ValueError("Пространственных блоков меньше, чем фолдов")
        assignments = np.empty(unique.size, np.int8)
        assignments[np.random.default_rng(self.seed).permutation(unique.size)] = np.arange(unique.size) % self.n_folds
        self.lookup = np.full(int(unique.max()) + 1, -1, np.int8)
        self.lookup[unique] = assignments
        self.fold = self.lookup[self.blocks]
        self.trees = []
        for fold in range(self.n_folds):
            ids = unique[assignments == fold]
            boxes = []
            for block in ids:
                row, col = divmod(int(block), self.nx)
                x0 = grid.x_min + col * self.block_size_m
                y0 = self.y_min + row * self.block_size_m
                boxes.append(box(x0, y0, x0 + self.block_size_m, y0 + self.block_size_m))
            self.trees.append(STRtree(boxes))

    def xy(self, cells):
        grid = self.dataset.grid
        rows, cols = np.divmod(np.asarray(cells), grid.width)
        return np.column_stack((grid.x_min + (cols + .5) * grid.cell_size,
                                grid.y_max - (rows + .5) * grid.cell_size))

    def block_ids(self, xy):
        col = np.floor((xy[:, 0] - self.dataset.grid.x_min) / self.block_size_m).astype(np.int64)
        row = np.floor((xy[:, 1] - self.y_min) / self.block_size_m).astype(np.int64)
        return row * self.nx + col

    def allowed_xy(self, xy, fold: int, buffer_m: float):
        import shapely

        if not 0 <= fold < self.n_folds or not np.isfinite(buffer_m) or buffer_m < 0:
            raise ValueError("Некорректный фолд или буфер CV")
        allowed = np.ones(len(xy), bool)
        # От целых прямоугольных блоков, включая их границу, а не от центров test-ячеек.
        hits = self.trees[fold].query(shapely.points(xy), predicate="dwithin", distance=buffer_m)
        if hits.size:
            allowed[hits[0]] = False
        return allowed

    def training_rows(self, rows, fold: int, buffer_m: float):
        rows = np.asarray(rows, dtype=np.int64)
        allowed = self.allowed_xy(self.xy(self.dataset.cells[rows]), fold, buffer_m)
        return rows[allowed]


def sample_background(
    dataset: Dataset, config: ExperimentConfig, folds: SpatialFolds, seed: int,
    fold: int | None = None,
) -> np.ndarray:
    """Равномерная по допустимой площади выборка; фон не является истинным отрицанием."""
    known = dataset.labels
    xy = known[["x", "y"]].to_numpy(dtype=float)
    if fold is not None:
        xy = xy[folds.allowed_xy(xy, fold, config.cv_buffer_m)]
    tree = cKDTree(xy) if len(xy) else None
    known_rows = known.loc[known.matrix_row >= 0, "matrix_row"].to_numpy(dtype=np.int64)
    candidates = np.arange(dataset.cells.size, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0 if fold is None else fold + 1]))
    rng.shuffle(candidates)
    selected = []
    count = 0
    for start in range(0, candidates.size, 8192):
        rows = candidates[start:start + 8192]
        if fold is not None:
            rows = folds.training_rows(rows, fold, config.cv_buffer_m)
        if not rows.size:
            continue
        rows = rows[~np.isin(rows, known_rows)]
        if tree is not None:
            distance, _ = tree.query(folds.xy(dataset.cells[rows]))
            rows = rows[distance > config.background_exclusion_m]
        take = rows[:config.background_cells - count]
        selected.append(take)
        count += take.size
        if count == config.background_cells:
            return np.concatenate(selected)
    raise ValueError(f"Недостаточно фона: {count} вместо {config.background_cells}")

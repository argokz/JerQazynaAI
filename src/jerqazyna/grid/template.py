"""Шаблон сетки задачи: проекция, привязка, размер. Все слои пишутся строго на эту сетку."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    crs: str
    cell_size: float
    x_min: float
    y_max: float
    width: int
    height: int

    @classmethod
    def from_bounds(cls, crs: str, cell_size: float, bounds: tuple[float, float, float, float]):
        """Сетка, покрывающая bounds, с углами, кратными cell_size.

        Кратность нужна, чтобы тайлы Sentinel-2 со своим шагом 100 м ложились в ячейки ровно.
        """
        x0, y0, x1, y1 = bounds
        x_min = math.floor(x0 / cell_size) * cell_size
        y_min = math.floor(y0 / cell_size) * cell_size
        x_max = math.ceil(x1 / cell_size) * cell_size
        y_max = math.ceil(y1 / cell_size) * cell_size
        return cls(
            crs=crs,
            cell_size=float(cell_size),
            x_min=float(x_min),
            y_max=float(y_max),
            width=round((x_max - x_min) / cell_size),
            height=round((y_max - y_min) / cell_size),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.x_min,
            self.y_max - self.height * self.cell_size,
            self.x_min + self.width * self.cell_size,
            self.y_max,
        )

    @property
    def transform(self):
        from rasterio.transform import from_origin

        return from_origin(self.x_min, self.y_max, self.cell_size, self.cell_size)

    def refined(self, factor: int) -> GridSpec:
        """Та же сетка с ячейками в factor раз мельче (для точного подсчёта долей площади)."""
        return GridSpec(
            self.crs, self.cell_size / factor, self.x_min, self.y_max,
            self.width * factor, self.height * factor,
        )

    def window(self, bounds: tuple[float, float, float, float]) -> tuple[slice, slice] | None:
        """Строки и столбцы сетки, покрывающие bounds (в CRS сетки); None — вне сетки."""
        x0, y0, x1, y1 = bounds
        c0 = max(0, math.floor((x0 - self.x_min) / self.cell_size))
        c1 = min(self.width, math.ceil((x1 - self.x_min) / self.cell_size))
        r0 = max(0, math.floor((self.y_max - y1) / self.cell_size))
        r1 = min(self.height, math.ceil((self.y_max - y0) / self.cell_size))
        if c0 >= c1 or r0 >= r1:
            return None
        return slice(r0, r1), slice(c0, c1)

    def window_transform(self, rows: slice, cols: slice):
        from rasterio.transform import from_origin

        return from_origin(
            self.x_min + cols.start * self.cell_size,
            self.y_max - rows.start * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def cell_centers(self) -> np.ndarray:
        xs = self.x_min + (np.arange(self.width) + 0.5) * self.cell_size
        ys = self.y_max - (np.arange(self.height) + 0.5) * self.cell_size
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack([gx.ravel(), gy.ravel()])

    def rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.floor((np.asarray(x) - self.x_min) / self.cell_size).astype(int)
        rows = np.floor((self.y_max - np.asarray(y)) / self.cell_size).astype(int)
        return rows, cols

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GridSpec:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def write_layer(path: Path, grid: GridSpec, array: np.ndarray, description: str = "") -> None:
    """Пишет слой на сетку как сжатый тайловый GeoTIFF (float32, nodata = NaN)."""
    import rasterio

    if array.shape != grid.shape:
        raise ValueError(f"{path.name}: форма {array.shape} не совпадает с сеткой {grid.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.tif")
    dtype = "uint8" if array.dtype == np.uint8 else "float32"
    profile = {
        "driver": "GTiff", "width": grid.width, "height": grid.height, "count": 1,
        "dtype": dtype, "crs": grid.crs, "transform": grid.transform,
        "nodata": None if dtype == "uint8" else np.nan,
        "compress": "deflate", "predictor": 2 if dtype == "uint8" else 3,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        if description:
            dst.set_band_description(1, description)
    tmp.replace(path)


def read_layer(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)

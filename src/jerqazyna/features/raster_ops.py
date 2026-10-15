"""Перенос исходных данных на сетку задачи.

Растры усредняются точно: сумма значений и сумма весов переносятся отдельно (resampling=sum),
поэтому ячейки на стыке тайлов получают честное среднее, а не значение одного тайла.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from jerqazyna.grid.template import GridSpec


@dataclass
class TileArray:
    """Кусок исходного растра в памяти: значения (NaN = нет данных) и привязка."""

    data: np.ndarray
    transform: object
    crs: object

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        from rasterio.transform import array_bounds

        h, w = self.data.shape
        return array_bounds(h, w, self.transform)  # (west, south, east, north)


@dataclass
class Accumulator:
    """Накопление среднего, минимума и максимума по многим тайлам."""

    grid: GridSpec
    stats: tuple[str, ...] = ("mean",)
    fast: bool = True  # тайлы на решётке сетки суммируются блоками, без перепроецирования
    total: np.ndarray = field(init=False)
    weight: np.ndarray = field(init=False)
    vmin: np.ndarray = field(init=False)
    vmax: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = self.grid.shape
        self.total = np.zeros(shape, np.float64)
        self.weight = np.zeros(shape, np.float64)
        if "min" in self.stats:
            self.vmin = np.full(shape, np.nan, np.float32)
        if "max" in self.stats:
            self.vmax = np.full(shape, np.nan, np.float32)

    def _aligned(self, tile: TileArray) -> tuple[int, int, int] | None:
        """(k, сдвиг по столбцам, сдвиг по строкам в пикселях тайла), если тайл в CRS сетки,
        его пиксель делит ячейку нацело и лежит на той же решётке; иначе None."""
        import pyproj

        t = tile.transform
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.a != -t.e:
            return None
        k = self.grid.cell_size / t.a
        col_off = (t.c - self.grid.x_min) / t.a
        row_off = (self.grid.y_max - t.f) / t.a
        if any(abs(v - round(v)) > 1e-6 for v in (k, col_off, row_off)):
            return None
        if not pyproj.CRS(str(tile.crs)).equals(pyproj.CRS(self.grid.crs), ignore_axis_order=True):
            return None
        return round(k), round(col_off), round(row_off)

    def _add_aligned(self, tile: TileArray, k: int, col_off: int, row_off: int) -> bool:
        """Быстрый путь без перепроецирования: среднее, минимум и максимум по блокам k×k."""
        h, w = tile.data.shape
        r0, c0 = max(0, row_off // k), max(0, col_off // k)
        r1 = min(self.grid.height, -(-(row_off + h) // k))
        c1 = min(self.grid.width, -(-(col_off + w) // k))
        if r0 >= r1 or c0 >= c1:
            return False
        fine = np.full(((r1 - r0) * k, (c1 - c0) * k), np.nan, np.float32)
        # Пересечение тайла с окном в пикселях тайла
        fr0, fc0 = r0 * k - row_off, c0 * k - col_off
        tr0, tc0 = max(0, fr0), max(0, fc0)
        tr1, tc1 = min(h, fr0 + fine.shape[0]), min(w, fc0 + fine.shape[1])
        if tr0 >= tr1 or tc0 >= tc1:
            return False
        fine[tr0 - fr0:tr1 - fr0, tc0 - fc0:tc1 - fc0] = tile.data[tr0:tr1, tc0:tc1]
        blocks = fine.reshape(r1 - r0, k, c1 - c0, k)
        valid = np.isfinite(blocks)
        if not valid.any():
            return False
        rows, cols = slice(r0, r1), slice(c0, c1)
        self.total[rows, cols] += np.where(valid, blocks, 0).sum(axis=(1, 3))
        self.weight[rows, cols] += valid.sum(axis=(1, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # пустые блоки
            if "min" in self.stats:
                self.vmin[rows, cols] = np.fmin(self.vmin[rows, cols], np.nanmin(blocks, axis=(1, 3)))
            if "max" in self.stats:
                self.vmax[rows, cols] = np.fmax(self.vmax[rows, cols], np.nanmax(blocks, axis=(1, 3)))
        return True

    def add(self, tile: TileArray) -> bool:
        from rasterio.warp import Resampling, reproject, transform_bounds

        if self.fast and (aligned := self._aligned(tile)) is not None:
            return self._add_aligned(tile, *aligned)
        win = self.grid.window(transform_bounds(tile.crs, self.grid.crs, *tile.bounds))
        if win is None:
            return False
        rows, cols = win
        dst_transform = self.grid.window_transform(rows, cols)
        shape = (rows.stop - rows.start, cols.stop - cols.start)
        valid = np.isfinite(tile.data)
        if not valid.any():
            return False

        def warp(src: np.ndarray, resampling, src_nodata) -> np.ndarray:
            out = np.full(shape, np.nan, np.float32)
            reproject(
                source=src, destination=out,
                src_transform=tile.transform, src_crs=tile.crs, src_nodata=src_nodata,
                dst_transform=dst_transform, dst_crs=self.grid.crs, dst_nodata=np.nan,
                resampling=resampling,
            )
            return out

        values = np.where(valid, tile.data, 0).astype(np.float32)
        s = warp(values, Resampling.sum, None)
        w = warp(valid.astype(np.float32), Resampling.sum, None)
        self.total[rows, cols] += np.nan_to_num(s)
        self.weight[rows, cols] += np.nan_to_num(w)
        src = tile.data.astype(np.float32)
        if "min" in self.stats:
            self.vmin[rows, cols] = np.fmin(self.vmin[rows, cols], warp(src, Resampling.min, np.nan))
        if "max" in self.stats:
            self.vmax[rows, cols] = np.fmax(self.vmax[rows, cols], warp(src, Resampling.max, np.nan))
        return True

    def mean(self, min_weight: float = 1e-6) -> np.ndarray:
        out = np.full(self.grid.shape, np.nan, np.float32)
        ok = self.weight > min_weight
        out[ok] = (self.total[ok] / self.weight[ok]).astype(np.float32)
        return out


def reproject_window(
    path: str, grid: GridSpec, src_crs: str | None = None, margin_deg: float = 1.0
) -> np.ndarray:
    """Растр (возможно глобальный) на сетку: читается только нужное окно.

    Если источник мельче ячейки — среднее, если крупнее — билинейная интерполяция.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        crs = CRS.from_user_input(src_crs) if src_crs else src.crs
        if crs is None:
            raise ValueError(f"{path}: в файле нет CRS, задайте params.src_crs")
        w, s, e, n = transform_bounds(grid.crs, crs, *grid.bounds)
        if crs.is_geographic:
            w, s, e, n = w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg
        win = from_bounds(w, s, e, n, src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=win).astype(np.float32)
        transform = src.window_transform(win)
        if src.nodata is not None and np.isfinite(src.nodata):
            data[data == src.nodata] = np.nan
        data[~np.isfinite(data) | (np.abs(data) >= 99998)] = np.nan  # EMAG2: 99999
        src_res_m = abs(transform.a) * (111_320 if crs.is_geographic else 1)

    out = np.full(grid.shape, np.nan, np.float32)
    resampling = Resampling.average if src_res_m < grid.cell_size else Resampling.bilinear
    reproject(
        source=data, destination=out, src_transform=transform, src_crs=crs, src_nodata=np.nan,
        dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan, resampling=resampling,
    )
    return out


def polygon_fraction(
    geoms: Iterable, grid: GridSpec, factor: int = 10, block: int = 250
) -> np.ndarray:
    """Доля площади каждой ячейки, покрытая полигонами (0..1).

    Полигоны растеризуются на сетке в factor раз мельче и усредняются блоками,
    чтобы не держать в памяти мелкую сетку целиком. geoms — в CRS сетки.
    """
    import shapely
    from rasterio.features import rasterize

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    out = np.zeros(grid.shape, np.float32)
    if not geoms:
        return out
    tree = shapely.STRtree(geoms)
    fine = grid.refined(factor)
    for r0 in range(0, grid.height, block):
        for c0 in range(0, grid.width, block):
            rows = slice(r0, min(r0 + block, grid.height))
            cols = slice(c0, min(c0 + block, grid.width))
            x0 = grid.x_min + cols.start * grid.cell_size
            x1 = grid.x_min + cols.stop * grid.cell_size
            y1 = grid.y_max - rows.start * grid.cell_size
            y0 = grid.y_max - rows.stop * grid.cell_size
            hits = tree.query(shapely.box(x0, y0, x1, y1))
            if hits.size == 0:
                continue
            h, w = rows.stop - rows.start, cols.stop - cols.start
            frows = slice(rows.start * factor, rows.stop * factor)
            fcols = slice(cols.start * factor, cols.stop * factor)
            burned = rasterize(
                ((geoms[i], 1) for i in hits),
                out_shape=(h * factor, w * factor),
                transform=fine.window_transform(frows, fcols),
                fill=0, dtype="uint8",
            )
            out[rows, cols] = burned.reshape(h, factor, w, factor).mean(axis=(1, 3))
    return out


def distance_km(
    geoms: Iterable, grid: GridSpec, margin_cells: int = 100, all_touched: bool = True
) -> np.ndarray:
    """Расстояние от центра ячейки до ближайшего объекта, км.

    Считается на сетке, расширенной на margin_cells, чтобы объекты за краем тоже учитывались.
    """
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.full(grid.shape, np.nan, np.float32)
    m = margin_cells
    big = GridSpec(
        grid.crs, grid.cell_size, grid.x_min - m * grid.cell_size,
        grid.y_max + m * grid.cell_size, grid.width + 2 * m, grid.height + 2 * m,
    )
    burned = rasterize(
        ((g, 1) for g in geoms), out_shape=big.shape, transform=big.transform,
        fill=0, dtype="uint8", all_touched=all_touched,
    )
    if not burned.any():
        return np.full(grid.shape, np.nan, np.float32)
    dist = distance_transform_edt(burned == 0, sampling=grid.cell_size / 1000)
    return dist[m:-m, m:-m].astype(np.float32)


def point_density(
    x: np.ndarray, y: np.ndarray, grid: GridSpec, sigma_km: float
) -> np.ndarray:
    """Сглаженная плотность точек: объектов на 1000 км²."""
    from scipy.ndimage import gaussian_filter

    rows, cols = grid.rowcol(x, y)
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    counts = np.zeros(grid.shape, np.float64)
    np.add.at(counts, (rows[inside], cols[inside]), 1)
    sigma_cells = sigma_km * 1000 / grid.cell_size
    cell_km2 = (grid.cell_size / 1000) ** 2
    smoothed = gaussian_filter(counts, sigma=sigma_cells, mode="constant")
    return (smoothed / cell_km2 * 1000).astype(np.float32)

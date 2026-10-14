"""Обработчики слоёв: каждый превращает папку источника в один или несколько растров на сетке.

Обработчик получает LayerConfig и контекст и возвращает {имя_канала: массив формы сетки}.
Если исходных файлов нет, он поднимает SourceMissing, и подготовка идёт дальше без слоя.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jerqazyna.config import LayerConfig, TaskConfig
from jerqazyna.features.raster_ops import (
    Accumulator,
    TileArray,
    distance_km,
    point_density,
    polygon_fraction,
    reproject_window,
)
from jerqazyna.grid.template import GridSpec

Bands = dict[str, np.ndarray]


class SourceMissing(FileNotFoundError):
    """Исходные данные слоя ещё не скачаны."""


@dataclass
class Context:
    task: TaskConfig
    grid: GridSpec
    root: Path  # корень проекта: пути source в конфиге считаются от него
    log: Callable[[str], None] = print

    def path(self, source: str) -> Path:
        p = Path(source)
        return p if p.is_absolute() else self.root / p


def _files(ctx: Context, layer: LayerConfig, pattern: str) -> list[Path]:
    base = ctx.path(layer.source)
    found = sorted(p for p in base.rglob(pattern) if p.is_file() and ".part" not in p.name)
    if not found:
        raise SourceMissing(f"{layer.name}: нет файлов {pattern} в {base}")
    return found


def _read_vector(
    ctx: Context, path: Path, layer_name: str | None = None, src_crs: str | None = None
):
    """Векторный слой в CRS сетки, только объекты в охвате сетки.

    src_crs задаёт CRS для файлов без неё (например, шейп-файл без .prj).
    """
    import geopandas as gpd
    import pyogrio
    from rasterio.warp import transform_bounds

    if not path.exists():
        raise SourceMissing(f"нет {path}")
    crs = src_crs or pyogrio.read_info(path, layer=layer_name)["crs"]
    if crs is None:
        raise ValueError(f"{path.name}: в файле нет CRS, задайте params.src_crs")
    bbox = transform_bounds(ctx.grid.crs, crs, *ctx.grid.bounds)
    gdf = gpd.read_file(path, layer=layer_name, bbox=bbox)
    if src_crs:
        gdf = gdf.set_crs(src_crs, allow_override=True)
    return gdf.to_crs(ctx.grid.crs)


def _gradient_magnitude(a: np.ndarray, cell_km: float) -> np.ndarray:
    gy, gx = np.gradient(a, cell_km)
    return np.hypot(gx, gy).astype(np.float32)


# --- рельеф --------------------------------------------------------------------------------

def dem(layer: LayerConfig, ctx: Context) -> Bands:
    """Средняя высота, перепад высот и средний уклон в ячейке по тайлам Copernicus DEM."""
    import rasterio

    tiles = _files(ctx, layer, "*.tif")
    elev = Accumulator(ctx.grid, ("mean", "min", "max"))
    slope = Accumulator(ctx.grid)
    used = 0
    for i, path in enumerate(tiles, 1):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            tile = TileArray(data, src.transform, src.crs)
            if not elev.add(tile):
                continue
            used += 1
            # Уклон на исходном разрешении: шаг пикселя в метрах зависит от широты.
            lat = src.xy(src.height // 2, 0)[1]
            dy = abs(src.transform.e) * 111_320
            dx = abs(src.transform.a) * 111_320 * np.cos(np.radians(lat))
            gy, gx = np.gradient(data, dy, dx)
            slope.add(TileArray(np.degrees(np.arctan(np.hypot(gx, gy))), src.transform, src.crs))
        if i % 100 == 0:
            ctx.log(f"    DEM: {i}/{len(tiles)} тайлов")
    ctx.log(f"    DEM: в сетку попало {used} тайлов из {len(tiles)}")
    return {
        "elevation_m": elev.mean(),
        "relief_m": elev.vmax - elev.vmin,
        "slope_deg": slope.mean(),
    }


# --- Sentinel-2 -------------------------------------------------------------------------------

S2_INDICES: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
    # Глинистые минералы и серицит (Al-OH): поглощение в B12 относительно B11.
    "clay_ratio": lambda b: b["swir16"] / b["swir22"],
    # Оксиды железа (гематит, гётит, лимонит).
    "iron_oxide_ratio": lambda b: b["red"] / b["blue"],
    # Двухвалентное железо.
    "ferrous_ratio": lambda b: b["swir16"] / b["nir"],
    # Растительность мешает спектральным признакам, модель должна это видеть.
    "ndvi": lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"]),
    "albedo_swir22": lambda b: b["swir22"] / 10_000,
}


def sentinel2_indices(layer: LayerConfig, ctx: Context) -> Bands:
    """Средние по ячейке отношения каналов из композитов Sentinel-2 (100 м)."""
    import rasterio

    tiles = _files(ctx, layer, "s2_composite_*.tif")
    acc = {name: Accumulator(ctx.grid) for name in S2_INDICES}
    coverage = Accumulator(ctx.grid)
    for path in tiles:
        with rasterio.open(path) as src:
            names = src.tags()["bands"].split(",")
            raw = src.read().astype(np.float32)
            raw[raw == 0] = np.nan  # 0 = нет чистых сцен
            bands = dict(zip(names, raw))
            for name, fn in S2_INDICES.items():
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = fn(bands)
                v[~np.isfinite(v)] = np.nan
                acc[name].add(TileArray(v, src.transform, src.crs))
            covered = np.isfinite(raw).all(axis=0).astype(np.float32)
            coverage.add(TileArray(covered, src.transform, src.crs))
    ctx.log(f"    Sentinel-2: {len(tiles)} тайлов")
    out = {name: a.mean() for name, a in acc.items()}
    out["coverage"] = coverage.mean()
    return out


# --- ASTER (готовые карты изменений USGS) -------------------------------------------------------

def aster_alteration(layer: LayerConfig, ctx: Context) -> Bands:
    """Доля площади ячейки с каждым типом изменений; вне сцен ASTER — NaN, а не ноль."""
    gdb = ctx.path(layer.source)
    if not gdb.exists():
        raise SourceMissing(f"нет {gdb}")
    scenes = _read_vector(ctx, gdb, layer.params["coverage_layer"])
    coverage = polygon_fraction(scenes.geometry.values, ctx.grid, factor=4)
    out: Bands = {"coverage": coverage}
    for band, name in layer.params["layers"].items():
        gdf = _read_vector(ctx, gdb, name)
        ctx.log(f"    ASTER {band}: {len(gdf):,} полигонов")
        frac = polygon_fraction(gdf.geometry.values, ctx.grid, factor=10)
        frac[coverage < 0.5] = np.nan
        out[f"{band}_fraction"] = frac
    return out


# --- геофизика ----------------------------------------------------------------------------------

def raster(layer: LayerConfig, ctx: Context) -> Bands:
    path = _files(ctx, layer, layer.params.get("pattern", "*.tif"))[0]
    value = reproject_window(str(path), ctx.grid, layer.params.get("src_crs"))
    return {"value": value}


def magnetics(layer: LayerConfig, ctx: Context) -> Bands:
    path = ctx.path(layer.source) / layer.params["file"]
    if not path.exists():
        raise SourceMissing(f"нет {path}")
    anomaly = reproject_window(str(path), ctx.grid, layer.params.get("src_crs"))
    return {
        "anomaly_nt": anomaly,
        "gradient_nt_per_km": _gradient_magnitude(anomaly, ctx.grid.cell_size / 1000),
    }


# Простая поправка Буге за плоскопараллельный слой: 2πGρ при ρ = 2670 кг/м³, мГал на метр.
BOUGUER_SLAB_MGAL_PER_M = 0.1119
GGMPLUS_SIZE = 2500  # точек на сторону тайла 5°×5°


def read_ggmplus_tile(path: Path) -> TileArray:
    """Тайл GGMplus: int16 big-endian в 0,1 мГал, записи идут по меридианам с юга на север."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    name = path.stem  # N45E070
    lat = int(name[1:3]) * (1 if name[0] == "N" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "E" else -1)
    raw = np.fromfile(path, dtype=">i2").astype(np.float32)
    if raw.size != GGMPLUS_SIZE**2:
        raise ValueError(f"{path.name}: {raw.size} значений, ожидалось {GGMPLUS_SIZE**2}")
    raw[raw == -32768] = np.nan
    # (меридиан, широта с юга) → (строка с севера, столбец с запада)
    grid = raw.reshape(GGMPLUS_SIZE, GGMPLUS_SIZE).T[::-1] / 10
    step = 5 / GGMPLUS_SIZE
    return TileArray(np.ascontiguousarray(grid), from_origin(lon, lat + 5, step, step), CRS.from_epsg(4326))


def ggmplus_bouguer(layer: LayerConfig, ctx: Context) -> Bands:
    """Гравитационное возмущение GGMplus и простая аномалия Буге (минус слой по средней высоте).

    Высота берётся из готового слоя dem, поэтому слой dem должен быть подготовлен раньше.
    Поправка за рельеф вокруг точки не вводится: для региональной сетки 1 км это допустимо.
    """
    from jerqazyna.features.prepare import features_dir
    from jerqazyna.grid.template import read_layer

    tiles = _files(ctx, layer, "*.dg")
    elevation_path = features_dir(ctx.root, ctx.task) / "dem__elevation_m.tif"
    if not elevation_path.exists():
        raise SourceMissing("сначала подготовьте слой dem (нужна средняя высота)")
    acc = Accumulator(ctx.grid)
    for path in tiles:
        acc.add(read_ggmplus_tile(path))
    disturbance = acc.mean()
    elevation = read_layer(elevation_path)
    bouguer = disturbance - BOUGUER_SLAB_MGAL_PER_M * elevation
    return {
        "disturbance_mgal": disturbance,
        "bouguer_mgal": bouguer.astype(np.float32),
        "bouguer_gradient_mgal_per_km": _gradient_magnitude(bouguer, ctx.grid.cell_size / 1000),
    }


# --- геология -----------------------------------------------------------------------------------

def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_") or "none"


def categorical_polygons(layer: LayerConfig, ctx: Context) -> Bands:
    """Доля каждого класса пород в ячейке и расстояние до классов по regex (например, интрузии)."""
    path = _files(ctx, layer, layer.params.get("pattern", "*.shp"))[0]
    gdf = _read_vector(ctx, path, src_crs=layer.params.get("src_crs"))
    wanted = layer.params["field"].lower()
    field = next((c for c in gdf.columns if c.lower() == wanted), None)
    if field is None:
        raise ValueError(f"{layer.name}: нет поля {layer.params['field']}; есть {list(gdf.columns)}")
    out: Bands = {}
    for cls, part in gdf.groupby(field):
        out[f"frac_{_safe(cls)}"] = polygon_fraction(part.geometry.values, ctx.grid, factor=4)
    if regex := layer.params.get("distance_to_regex"):
        sel = gdf[gdf[field].astype(str).str.contains(regex, regex=True)]
        ctx.log(f"    классы для расстояния ({regex}): {sorted(sel[field].unique())}")
        out["distance_km"] = distance_km(sel.geometry.values, ctx.grid)
    return out


def distance_to_lines(layer: LayerConfig, ctx: Context) -> Bands:
    path = _files(ctx, layer, layer.params.get("pattern", "*.geojson"))[0]
    gdf = _read_vector(ctx, path)
    ctx.log(f"    линий в охвате: {len(gdf):,}")
    return {"distance_km": distance_km(gdf.geometry.values, ctx.grid)}


def polygon_fraction_layer(layer: LayerConfig, ctx: Context) -> Bands:
    gdf = _read_vector(ctx, ctx.path(layer.source), layer.params.get("layer"))
    return {"fraction": polygon_fraction(gdf.geometry.values, ctx.grid, factor=4)}


# --- маска и смещение изученности -------------------------------------------------------------

def mine_mask(layer: LayerConfig, ctx: Context) -> Bands:
    """Горные выработки и буфер вокруг них: здесь спектральные признаки не используются."""
    path = _files(ctx, layer, layer.params.get("pattern", "*.gpkg"))[0]
    gdf = _read_vector(ctx, path)
    buffer_m = ctx.task.labels.mine_mask_buffer_m
    ctx.log(f"    выработок в охвате: {len(gdf):,}, буфер {buffer_m:.0f} м")
    mined = polygon_fraction(gdf.geometry.values, ctx.grid, factor=10)
    buffered = polygon_fraction(gdf.geometry.buffer(buffer_m).values, ctx.grid, factor=4)
    return {"mined_fraction": mined, "mask": (buffered > 0).astype(np.uint8)}


def point_density_layer(layer: LayerConfig, ctx: Context) -> Bands:
    """Плотность рудопроявлений MRDS (все типы) — прокси того, насколько территория изучена."""
    import pandas as pd
    from pyproj import Transformer

    path = _files(ctx, layer, "mrds-csv.zip")[0]
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(z.open("mrds.csv"), usecols=["latitude", "longitude"], low_memory=False)
    df = df.dropna()
    x, y = Transformer.from_crs("EPSG:4326", ctx.grid.crs, always_xy=True).transform(
        df.longitude.to_numpy(), df.latitude.to_numpy()
    )
    sigma = float(layer.params.get("sigma_km", 10))
    return {"per_1000km2": point_density(np.asarray(x), np.asarray(y), ctx.grid, sigma)}


PROCESSORS: dict[str, Callable[[LayerConfig, Context], Bands]] = {
    "dem": dem,
    "sentinel2_indices": sentinel2_indices,
    "aster_alteration": aster_alteration,
    "raster": raster,
    "magnetics": magnetics,
    "ggmplus_bouguer": ggmplus_bouguer,
    "categorical_polygons": categorical_polygons,
    "distance_to_lines": distance_to_lines,
    "polygon_fraction": polygon_fraction_layer,
    "mine_mask": mine_mask,
    "point_density": point_density_layer,
}

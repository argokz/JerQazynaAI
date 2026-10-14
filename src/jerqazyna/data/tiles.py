"""Тайлы N°×N° внутри охвата."""

from __future__ import annotations

import math
from collections.abc import Iterator

from jerqazyna.data.catalog import BBox


def degree_tiles(bbox: BBox, first: BBox | None = None, step: int = 1) -> list[tuple[int, int]]:
    """(lat, lon) нижних левых углов тайлов step°×step°, пересекающих bbox.

    Углы кратны step. Тайлы, пересекающие first (например, слепой регион), идут первыми.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    def grid(lo: float, hi: float) -> range:
        return range(math.floor(lo / step) * step, math.ceil(hi / step) * step, step)

    tiles = [(lat, lon) for lat in grid(lat_min, lat_max) for lon in grid(lon_min, lon_max)]
    if first is None:
        return tiles
    f_lon_min, f_lat_min, f_lon_max, f_lat_max = first

    def in_first(t: tuple[int, int]) -> bool:
        lat, lon = t
        return lat + step > f_lat_min and lat < f_lat_max and lon + step > f_lon_min and lon < f_lon_max

    return sorted(tiles, key=lambda t: not in_first(t))


def tile_bbox(lat: int, lon: int, step: int = 1) -> BBox:
    return (float(lon), float(lat), float(lon + step), float(lat + step))


def tile_name(lat: int, lon: int) -> str:
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def tile_urls(template: str, tiles: list[tuple[int, int]]) -> Iterator[str]:
    """URL тайлов по шаблону: {name} = N45E070, {lat} = N45, {lon} = E070."""
    for lat, lon in tiles:
        name = tile_name(lat, lon)
        yield template.format(name=name, lat=name[:3], lon=name[3:])

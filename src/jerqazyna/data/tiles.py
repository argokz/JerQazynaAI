"""Тайлы 1°×1° внутри охвата."""

from __future__ import annotations

import math
from collections.abc import Iterator

from jerqazyna.data.catalog import BBox


def degree_tiles(bbox: BBox, first: BBox | None = None) -> list[tuple[int, int]]:
    """(lat, lon) нижних левых углов тайлов, пересекающих bbox.

    Тайлы внутри first (например, слепого региона) идут первыми.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = [
        (lat, lon)
        for lat in range(math.floor(lat_min), math.ceil(lat_max))
        for lon in range(math.floor(lon_min), math.ceil(lon_max))
    ]
    if first is None:
        return tiles
    f_lon_min, f_lat_min, f_lon_max, f_lat_max = first

    def in_first(t: tuple[int, int]) -> bool:
        lat, lon = t
        return lat + 1 > f_lat_min and lat < f_lat_max and lon + 1 > f_lon_min and lon < f_lon_max

    return sorted(tiles, key=lambda t: not in_first(t))


def tile_bbox(lat: int, lon: int) -> BBox:
    return (float(lon), float(lat), float(lon + 1), float(lat + 1))


def tile_name(lat: int, lon: int) -> str:
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def copernicus_tile_urls(template: str, tiles: list[tuple[int, int]]) -> Iterator[str]:
    """URL тайлов Copernicus DEM (имена вида N46_00_E074_00)."""
    for lat, lon in tiles:
        name = tile_name(lat, lon)
        yield template.format(lat=name[:3], lon=name[3:])

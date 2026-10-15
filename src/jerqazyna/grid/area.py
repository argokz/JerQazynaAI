"""Область задачи: основные страны и соседние части пояса внутри рамки сетки.

Сетка строится по основным странам. Соседние страны (например, Синьцзян и юг России)
обрезаются по рамке этой сетки, поэтому их добавление не меняет сетку и готовые слои.
"""

from __future__ import annotations

from pathlib import Path

from jerqazyna.grid.template import GridSpec


class BoundaryMissing(FileNotFoundError):
    pass


def _countries(boundaries_dir: Path, isos: list[str], crs: str):
    import geopandas as gpd
    import pandas as pd

    parts = []
    for iso in isos:
        path = Path(boundaries_dir) / f"geoBoundaries-{iso}-ADM0.geojson"
        if not path.exists():
            raise BoundaryMissing(f"нет границы {iso}: jerqazyna data fetch geoboundaries")
        parts.append(gpd.read_file(path))
    return pd.concat(parts).to_crs(crs).geometry.union_all()


def region(
    boundaries_dir: Path, core: list[str], extra: list[str], crs: str, cell_size: float
):
    """(геометрия области в crs, сетка). Сетка зависит только от основных стран."""
    from shapely.geometry import box

    core_geom = _countries(boundaries_dir, core, crs)
    grid = GridSpec.from_bounds(crs, cell_size, core_geom.bounds)
    if not extra:
        return core_geom, grid
    extra_geom = _countries(boundaries_dir, extra, crs).intersection(box(*grid.bounds))
    return core_geom.union(extra_geom), grid

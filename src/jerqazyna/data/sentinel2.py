"""Летний медианный композит Sentinel-2 L2A на тайл 1°×1° из Earth Search (без авторизации).

Читаются обзоры COG на нужном разрешении, поэтому тайл весит мегабайты, а не гигабайты.
Нужны extras [geo]: pystac-client, odc-stac.
"""

from __future__ import annotations

from pathlib import Path

from jerqazyna.config import CENTRAL_ASIA_ALBERS
from jerqazyna.data.catalog import BBox, StacSpec

# Классы SCL, которые считаются чистой поверхностью: растительность, открытый грунт, вода, без класса.
CLEAR_SCL = (4, 5, 6, 7)


def select_clearest(items: list, per_mgrs: int) -> list:
    """Самые малооблачные сцены для каждого квадрата MGRS.

    Для медианы хватает 6–10 чистых сцен; сотни сцен только замедляют чтение.
    """
    by_tile: dict[str, list] = {}
    for item in items:
        key = item.properties.get("grid:code") or item.properties.get("s2:mgrs_tile") or item.id
        by_tile.setdefault(key, []).append(item)
    chosen = []
    for group in by_tile.values():
        group.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
        chosen.extend(group[:per_mgrs])
    return chosen


def composite_tile(spec: StacSpec, bbox: BBox, dest: Path) -> int:
    """Строит композит и пишет COG в dest. Возвращает число использованных сцен (0 — нет данных)."""
    try:
        import odc.stac
        import pystac_client
    except ImportError as e:
        raise ImportError("Для Sentinel-2 установите extras: pip install -e \".[geo]\"") from e

    # Анонимное чтение COG с S3 без лишних листингов каталогов: в разы быстрее настроек по умолчанию.
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

    client = pystac_client.Client.open(spec.endpoint)
    found = [
        item
        for item in client.search(
            collections=[spec.collection],
            bbox=bbox,
            datetime=spec.datetime,
            query={"eo:cloud_cover": {"lt": spec.max_cloud}},
        ).items()
        if item.datetime is not None and item.datetime.month in spec.months
    ]
    items = select_clearest(found, spec.scenes_per_mgrs)
    if not items:
        return 0

    bands = [b for b in spec.bands if b != "scl"]
    ds = odc.stac.load(
        items,
        bands=[*bands, "scl"],
        bbox=bbox,
        crs=CENTRAL_ASIA_ALBERS,
        resolution=spec.resolution_m,
        groupby="solar_day",
        resampling={"scl": "nearest", "*": "average"},
        chunks={"x": 1024, "y": 1024},
        fail_on_error=False,  # битая сцена на S3 пропускается, а не роняет весь тайл
    )
    clear = ds["scl"].isin(CLEAR_SCL)
    composite = ds[bands].where(clear).median(dim="time", skipna=True)
    stacked = composite.to_array("band").fillna(0).astype("uint16").compute()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".part.tif")
    stacked.odc.write_cog(tmp, overwrite=True, tags={"bands": ",".join(bands), "scenes": len(items)})
    tmp.replace(dest)
    return len(items)

"""Модели ответов API веб-карты."""

from __future__ import annotations

from pydantic import BaseModel


class SourceInfo(BaseModel):
    id: str
    title: str
    license: str
    citation: str
    attribution: str
    notice: str = ""
    share_alike: bool


class LegendStop(BaseModel):
    position: float
    color: str


class LegendClassOut(BaseModel):
    value: float
    label: str


class Legend(BaseModel):
    palette: str
    min: float
    max: float
    units: str
    stops: list[LegendStop]
    classes: list[LegendClassOut] = []


class LayerFile(BaseModel):
    """Происхождение значений слоя: исходный растр, его хеш и подтверждающий манифест."""

    source: str
    source_sha256: str
    cog_sha256: str
    provenance_manifest: str | None = None
    provenance: dict[str, str] = {}


class LayerOut(BaseModel):
    id: str
    title: str
    description: str
    units: str
    role: str
    group: str
    warning: str
    available: bool
    unavailable_reason: str = ""
    visible: bool
    opacity: float
    tiles_url: str | None
    coverage: float | None
    legend: Legend
    sources: list[SourceInfo]
    file: LayerFile | None


class LayersResponse(BaseModel):
    task_id: str
    bounds: tuple[float, float, float, float]   # lon_min, lat_min, lon_max, lat_max
    layers: list[LayerOut]


class CellValue(BaseModel):
    id: str
    title: str
    units: str
    role: str
    warning: str
    available: bool
    value: float | None
    sources: list[SourceInfo]
    file: LayerFile | None


class CellResponse(BaseModel):
    task_id: str
    lon: float
    lat: float
    row: int
    col: int
    cell_id: int                     # row * width + col, как в обучающих таблицах
    x: float                         # центр ячейки в CRS сетки, м
    y: float
    cell_size_m: float
    crs: str
    layers: list[CellValue]


class HealthResponse(BaseModel):
    status: str
    task_id: str
    layers_total: int
    layers_available: int

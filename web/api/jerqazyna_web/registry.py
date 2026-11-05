"""Реестр слоёв веб-карты (web/layers.yaml) и лицензионные правила показа.

Реестр — единственное место, где перечислены файлы слоёв: код путей не знает.
Лицензия и цитата каждого источника берутся из configs/sources.yaml, формулы
атрибуции — из раздела attributions реестра. Правила показа взяты из
docs/DATA_LICENSES_2026-11-05.md и проверяются при загрузке, а не при отрисовке.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jerqazyna.data.catalog import Catalog, Source

# Источники, которые нельзя показывать на карте ни в каком виде (см. DATA_LICENSES).
BLOCKED_SOURCES: dict[str, str] = {
    "usgs_fsu_geology": (
        "USGS FSU geology: части набора — собственность ESRI, передача производных третьим "
        "лицам запрещена; слой исключён из карты окончательно"
    ),
    "geoboundaries": (
        "geoBoundaries (ODbL): границы не отдаются в браузер; маска области используется "
        "только внутри API"
    ),
}

# Маркер share-alike в поле license каталога: такие слои показываются только отдельно.
SHARE_ALIKE_MARKER = "BY-SA"

# Опорные цвета палитр для легенды: (позиция 0..1, цвет). Имена совпадают с colormap_name
# TiTiler (rio-tiler), поэтому тайлы и легенда используют одну и ту же палитру.
_VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c",
            "#28ae80", "#5ec962", "#addc30", "#fde725"]
_MAGMA = ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f",
          "#cd4071", "#f1605d", "#fd9668", "#fcfdbf"]
_RDBU_R = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
           "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
_REDS = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a",
         "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]


def _even(colors: list[str]) -> list[tuple[float, str]]:
    last = len(colors) - 1
    return [(round(i / last, 4), c) for i, c in enumerate(colors)]


PALETTES: dict[str, list[tuple[float, str]]] = {
    "viridis": _even(_VIRIDIS),
    "viridis_r": _even(_VIRIDIS[::-1]),
    "magma": _even(_MAGMA),
    "rdbu_r": _even(_RDBU_R),
    "reds": _even(_REDS),
    "terrain": [(0.0, "#333399"), (0.15, "#0099ff"), (0.25, "#00cc66"),
                (0.5, "#ffff99"), (0.75, "#805c54"), (1.0, "#ffffff")],
}

Role = Literal["feature", "mask", "research"]
Resampling = Literal["average", "nearest", "mode"]
LAYER_ID = r"^[a-z][a-z0-9_]*$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LegendClass(Strict):
    value: float
    label: str


class Display(Strict):
    palette: str
    min: float
    max: float
    # Значение, которое TiTiler делает прозрачным (например, 0 у маски). В данных не меняется.
    nodata: float | None = None
    visible: bool = False
    opacity: float = Field(0.8, ge=0, le=1)
    # Как строить обзоры COG: average — для непрерывных, nearest/mode — для масок и классов.
    resampling: Resampling = "average"
    classes: list[LegendClass] = []

    @model_validator(mode="after")
    def check(self) -> Display:
        if self.palette not in PALETTES:
            raise ValueError(f"палитра {self.palette!r} не поддержана; есть: {', '.join(PALETTES)}")
        if not self.min < self.max:
            raise ValueError(f"диапазон палитры пуст: min={self.min}, max={self.max}")
        return self


class Provenance(Strict):
    """Манифест, где записан sha256 исходного файла (cache.json, run.json, report.json...)."""

    manifest: str
    # Путь к хешу через точки (например mask_sha256); без него ищется словарь files.
    key: str | None = None


class LayerSpec(Strict):
    id: str = Field(pattern=LAYER_ID)
    title: str
    description: str = ""
    file: str                       # относительно корня проекта
    band: int = Field(1, ge=1)
    units: str = ""
    role: Role = "feature"
    group: str = ""
    sources: list[str] = Field(min_length=1)
    provenance: Provenance | None = None
    display: Display
    warning: str = ""
    enabled: bool = True

    @model_validator(mode="after")
    def check_role(self) -> LayerSpec:
        if self.role == "research":
            if not self.warning:
                raise ValueError(f"{self.id}: исследовательскому слою нужна пометка warning")
            if self.display.visible:
                raise ValueError(f"{self.id}: исследовательский слой не включается по умолчанию")
        return self


class AreaSpec(Strict):
    """Маска области задачи: только для ответа 404 вне области, на карту не выводится."""

    file: str
    provenance: Provenance | None = None


class Attribution(Strict):
    text: str = Field(min_length=1)
    # Отказ от ответственности или условие, которое лицензия требует передавать дальше.
    notice: str = ""


class Registry(Strict):
    version: Literal[1]
    task_id: str
    grid: str                       # grid.json задачи, относительно корня проекта
    area: AreaSpec
    attributions: dict[str, Attribution]
    layers: list[LayerSpec]

    @model_validator(mode="after")
    def unique_ids(self) -> Registry:
        ids = [layer.id for layer in self.layers]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"повторяются id слоёв: {', '.join(duplicates)}")
        return self

    @property
    def enabled_layers(self) -> list[LayerSpec]:
        return [layer for layer in self.layers if layer.enabled]

    def get(self, layer_id: str) -> LayerSpec:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(f"Нет слоя {layer_id!r} в реестре")


def is_share_alike(source: Source) -> bool:
    return SHARE_ALIKE_MARKER in source.license


def check_licenses(registry: Registry, catalog: Catalog) -> None:
    """Отказ при любом нарушении правил показа; сообщение перечисляет все нарушения."""
    known = {s.id for s in catalog.sources}
    issues = []
    for layer in registry.enabled_layers:
        for source_id in layer.sources:
            if source_id in BLOCKED_SOURCES:
                issues.append(f"{layer.id}: {BLOCKED_SOURCES[source_id]}")
            elif source_id not in known:
                issues.append(f"{layer.id}: источника {source_id!r} нет в каталоге")
            elif source_id not in registry.attributions:
                issues.append(f"{layer.id}: нет формулы атрибуции для {source_id}")
        share_alike = [s for s in layer.sources if s in known and is_share_alike(catalog.get(s))]
        if share_alike and len(layer.sources) > 1:
            issues.append(
                f"{layer.id}: слой под share-alike ({', '.join(share_alike)}) показывается "
                "только отдельно, без смешения с другими источниками"
            )
    if issues:
        raise ValueError("Реестр нарушает лицензионные правила:\n- " + "\n- ".join(issues))


def load_registry(path: str | Path, catalog: Catalog) -> Registry:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise FileNotFoundError(f"Не удалось прочитать реестр слоёв {path}: {e}") from e
    registry = Registry.model_validate(raw)
    check_licenses(registry, catalog)
    return registry

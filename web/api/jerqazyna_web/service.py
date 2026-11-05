"""Сборка реестра, каталога и манифеста COG в ответы API; точечное чтение ячеек."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from urllib.parse import urlencode

from jerqazyna.data.catalog import load_catalog
from jerqazyna.data.download import sha256_of
from jerqazyna.grid.template import GridSpec
from jerqazyna_web.manifest import MANIFEST_NAME, CogEntry, load_manifest
from jerqazyna_web.registry import PALETTES, LayerSpec, is_share_alike, load_registry
from jerqazyna_web.schemas import (
    CellResponse,
    CellValue,
    LayerFile,
    LayerOut,
    LayersResponse,
    Legend,
    LegendClassOut,
    LegendStop,
    SourceInfo,
)
from jerqazyna_web.settings import Settings

log = logging.getLogger(__name__)
TILE_MATRIX_SET = "WebMercatorQuad"


class OutsideGridError(ValueError):
    """Точка вне сетки задачи (422)."""


class OutsideAreaError(LookupError):
    """Точка на сетке, но вне области задачи (404)."""


def _layer_file(entry: CogEntry) -> LayerFile:
    prov = entry.provenance
    return LayerFile(
        source=entry.source, source_sha256=entry.source_sha256, cog_sha256=entry.cog_sha256,
        provenance_manifest=prov.manifest if prov else None,
        provenance=prov.details if prov else {},
    )


def _read_cell(path: Path, row: int, col: int) -> float | None:
    """Одно значение окном 1×1: растр целиком не читается."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        value = float(src.read(1, window=Window(col, row, 1, 1))[0, 0])
        nodata = src.nodata
    if not math.isfinite(value) or (nodata is not None and value == nodata):
        return None
    return value


class MapService:
    """Состояние API: строится один раз при старте и отказывает при несогласованных данных."""

    def __init__(self, settings: Settings):
        from pyproj import Transformer

        self.settings = settings
        catalog = load_catalog(settings.sources)
        self.registry = load_registry(settings.registry, catalog)
        data_dir = Path(settings.data_dir)
        manifest = load_manifest(data_dir / MANIFEST_NAME)
        if manifest is None:
            raise RuntimeError(f"Нет {data_dir / MANIFEST_NAME}: сначала выполните "
                               "python web/scripts/export_cogs.py")
        if manifest.task_id != self.registry.task_id or manifest.area is None:
            raise RuntimeError(f"Манифест {data_dir / MANIFEST_NAME} не относится к задаче "
                               f"{self.registry.task_id} или не содержит маску области")
        self.task_id = manifest.task_id
        self.grid = GridSpec(**manifest.grid)
        self.data_dir = data_dir
        self.area_path = self._checked_path(manifest.area, "area")
        self._entries: dict[str, CogEntry] = {}
        self._reasons: dict[str, str] = {}
        for layer in self.registry.enabled_layers:
            entry = manifest.layers.get(layer.id)
            if entry is None:
                self._reasons[layer.id] = "COG не экспортирован"
            elif entry.source != layer.file or entry.band != layer.band:
                self._reasons[layer.id] = "COG устарел относительно реестра; повторите экспорт"
            else:
                self._checked_path(entry, layer.id)
                self._entries[layer.id] = entry
        for layer_id, reason in self._reasons.items():
            log.warning("Слой %s недоступен: %s", layer_id, reason)
        self._sources = {
            layer.id: [self._source_info(catalog.get(s)) for s in layer.sources]
            for layer in self.registry.enabled_layers
        }
        self._to_grid = Transformer.from_crs("EPSG:4326", self.grid.crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(self.grid.crs, "EPSG:4326", always_xy=True)
        self.bounds = tuple(round(v, 6) for v in to_wgs84.transform_bounds(
            *self.grid.bounds, densify_pts=21))

    def _checked_path(self, entry: CogEntry, name: str) -> Path:
        path = self.data_dir / entry.file
        if not path.is_file():
            raise RuntimeError(f"{name}: в манифесте есть {entry.file}, но файла нет")
        if sha256_of(path) != entry.cog_sha256:
            raise RuntimeError(f"{name}: {entry.file} не совпадает с sha256 манифеста; "
                               "повторите экспорт COG")
        return path

    def _source_info(self, source) -> SourceInfo:
        attribution = self.registry.attributions[source.id]
        return SourceInfo(
            id=source.id, title=source.title, license=source.license, citation=source.citation,
            attribution=attribution.text, notice=attribution.notice,
            share_alike=is_share_alike(source),
        )

    @property
    def layers_available(self) -> int:
        return len(self._entries)

    def tiles_url(self, layer: LayerSpec) -> str | None:
        entry = self._entries.get(layer.id)
        if entry is None:
            return None
        display = layer.display
        params = {
            "url": f"{self.settings.tiles_data_dir.rstrip('/')}/{Path(entry.file).name}",
            "bidx": 1,
            "rescale": f"{display.min:g},{display.max:g}",
            "colormap_name": display.palette,
        }
        if display.nodata is not None:
            params["nodata"] = f"{display.nodata:g}"
        return (f"{self.settings.tiles_url}/cog/tiles/{TILE_MATRIX_SET}/{{z}}/{{x}}/{{y}}.png?"
                + urlencode(params))

    def _legend(self, layer: LayerSpec) -> Legend:
        display = layer.display
        return Legend(
            palette=display.palette, min=display.min, max=display.max, units=layer.units,
            stops=[LegendStop(position=p, color=c) for p, c in PALETTES[display.palette]],
            classes=[LegendClassOut(value=c.value, label=c.label) for c in display.classes],
        )

    def layers(self) -> LayersResponse:
        out = []
        for layer in self.registry.enabled_layers:
            entry = self._entries.get(layer.id)
            out.append(LayerOut(
                id=layer.id, title=layer.title, description=layer.description,
                units=layer.units, role=layer.role, group=layer.group, warning=layer.warning,
                available=entry is not None, unavailable_reason=self._reasons.get(layer.id, ""),
                visible=layer.display.visible, opacity=layer.display.opacity,
                tiles_url=self.tiles_url(layer), coverage=entry.coverage if entry else None,
                legend=self._legend(layer), sources=self._sources[layer.id],
                file=_layer_file(entry) if entry else None,
            ))
        return LayersResponse(task_id=self.task_id, bounds=self.bounds, layers=out)

    def cell(self, lon: float, lat: float) -> CellResponse:
        x, y = self._to_grid.transform(lon, lat)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise OutsideGridError("Точку нельзя перевести в проекцию сетки")
        rows, cols = self.grid.rowcol(x, y)
        row, col = int(rows), int(cols)
        if not (0 <= row < self.grid.height and 0 <= col < self.grid.width):
            raise OutsideGridError(f"Точка ({lon}, {lat}) вне сетки задачи {self.task_id}")
        if _read_cell(self.area_path, row, col) != 1:
            raise OutsideAreaError(f"Ячейка ({row}, {col}) вне области задачи {self.task_id}")
        values = []
        for layer in self.registry.enabled_layers:
            entry = self._entries.get(layer.id)
            values.append(CellValue(
                id=layer.id, title=layer.title, units=layer.units, role=layer.role,
                warning=layer.warning, available=entry is not None,
                value=_read_cell(self.data_dir / entry.file, row, col) if entry else None,
                sources=self._sources[layer.id], file=_layer_file(entry) if entry else None,
            ))
        size = self.grid.cell_size
        return CellResponse(
            task_id=self.task_id, lon=lon, lat=lat, row=row, col=col,
            cell_id=row * self.grid.width + col,
            x=self.grid.x_min + (col + 0.5) * size, y=self.grid.y_max - (row + 0.5) * size,
            cell_size_m=size, crs=self.grid.crs, layers=values,
        )

"""Конфигурация задачи: тип месторождения, регион, слои, модель, схема валидации.

Новая задача = новый YAML-файл в configs/tasks/, а не новый код.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pyproj import CRS
from pyproj.exceptions import CRSError

# Равновеликая коническая проекция Альберса для Центральной Азии.
# Метрика «доля найденных объектов на долю площади» корректна только в равновеликой проекции.
CENTRAL_ASIA_ALBERS = (
    "+proj=aea +lat_0=45 +lat_1=40 +lat_2=52 +lon_0=70 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

FeatureGroup = Literal[
    "spectral", "terrain", "geophysics", "geology", "geochemistry", "mask", "bias"
]
# feature   — признак для модели
# benchmark — эталон для сравнения (строился с учётом известных объектов, в модель нельзя)
# mask      — маска (например, горные выработки: спектральные признаки там обнуляются)
# bias      — прокси изученности территории, моделируется отдельно от признаков
LayerRole = Literal["feature", "benchmark", "mask", "bias"]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, allow_inf_nan=False)


LayerKind = Literal[
    "dem", "sentinel2_indices", "aster_alteration", "raster", "magnetics",
    "ggmplus_bouguer", "categorical_polygons", "distance_to_lines", "polygon_fraction",
    "mine_mask", "point_density",
]
AreaFraction = Annotated[float, Field(gt=0, le=1)]
CountryCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class GridConfig(ConfigModel):
    crs: str = CENTRAL_ASIA_ALBERS
    cell_size_m: float = Field(1000.0, gt=0)

    @field_validator("crs")
    @classmethod
    def crs_must_be_equal_area(cls, v: str) -> str:
        try:
            crs = CRS.from_user_input(v)
        except CRSError as e:
            raise ValueError(f"Некорректная CRS: {v}") from e
        operation = crs.coordinate_operation
        supported = {
            "albers equal area", "lambert azimuthal equal area",
            "lambert cylindrical equal area", "lambert cylindrical equal area (spherical)",
            "equal earth", "mollweide", "sinusoidal",
        }
        if (not crs.is_projected or operation is None
                or operation.method_name.lower() not in supported):
            raise ValueError(f"Нужна поддерживаемая равновеликая CRS (aea, laea, cea): {v}")
        if len(crs.axis_info) != 2 or any(a.unit_conversion_factor != 1 for a in crs.axis_info):
            raise ValueError("Оси CRS должны быть в метрах")
        return v


class LabelSource(ConfigModel):
    source: str  # файл или папка источника
    layer: str | None = None  # слой внутри geodatabase / GeoPackage
    # Для CSV/XLSX: колонки координат WGS84; если не заданы, ищутся по имени (latitude/lat...).
    lon_field: str | None = None
    lat_field: str | None = None
    encoding: str | None = None
    name_field: str = "Name"
    status_field: str | None = "SiteStatus"  # None — в источнике нет статуса
    # Оставить только объекты, у которых поле type_field совпадает с type_regex.
    type_field: str | None = None
    type_regex: str | None = None
    # Дополнительные колонки в labels.csv: {имя в таблице: поле источника}.
    extra_fields: dict[str, str] = {}
    required: bool = True
    # Для дополнительного источника: можно только обогащать точные совпадения.
    match_radius_m: float | None = Field(None, ge=0)
    add_unmatched: bool = True

    @model_validator(mode="after")
    def check_fields(self):
        if bool(self.type_field) != bool(self.type_regex):
            raise ValueError("type_field и type_regex задаются вместе")
        if self.type_regex:
            try:
                re.compile(self.type_regex)
            except re.error as e:
                raise ValueError(f"Некорректный type_regex: {e}") from e
        reserved = {"name", "status", "source", "status_raw", "status_source", "aliases",
                    "source_records", "geometry", "x", "y", "lon", "lat", "row", "col", "in_area"}
        if reserved & self.extra_fields.keys():
            raise ValueError("extra_fields не должны перезаписывать служебные поля меток")
        return self


class LabelsConfig(LabelSource):
    """Основной источник меток и дополнительные, объединяемые по расстоянию."""

    required: Literal[True] = True

    # Объект дополнительного источника ближе merge_radius_m к уже взятому считается тем же
    # (у него берётся только статус, если у основного его нет); дальше — добавляется.
    merge: list[LabelSource] = []
    merge_radius_m: float = Field(5000.0, ge=0)
    include_prospects: bool = True
    # Буфер вокруг известных объектов, внутри которого спектральные признаки маскируются,
    # чтобы модель не училась находить карьеры и отвалы.
    mine_mask_buffer_m: float = Field(3000.0, ge=0)


class LayerConfig(ConfigModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    group: FeatureGroup
    source: str  # папка источника в data/raw
    kind: LayerKind
    role: LayerRole = "feature"
    params: dict[str, Any] = {}
    required: bool = True

    @model_validator(mode="after")
    def check_params(self):
        allowed = {
            "dem": {"fine_factor"}, "sentinel2_indices": set(), "aster_alteration": {"layers", "coverage_layer"},
            "raster": {"pattern", "src_crs"}, "magnetics": {"file", "src_crs"},
            "ggmplus_bouguer": set(),
            "categorical_polygons": {"pattern", "layer", "field", "src_crs", "distance_to_regex"},
            "distance_to_lines": {"pattern"}, "polygon_fraction": {"layer"},
            "mine_mask": {"pattern"}, "point_density": {"sigma_km"},
        }
        unknown = self.params.keys() - allowed[self.kind]
        if unknown:
            raise ValueError(f"{self.name}: неизвестные params: {', '.join(sorted(unknown))}")
        required = {"aster_alteration": {"layers", "coverage_layer"},
                    "magnetics": {"file"}, "categorical_polygons": {"field"}}
        if any(not self.params.get(k) for k in required.get(self.kind, set())):
            raise ValueError(f"{self.name}: нужны params {sorted(required[self.kind])}")
        string_fields = {"pattern", "src_crs", "file", "layer", "field", "distance_to_regex",
                         "coverage_layer"}
        if any(not isinstance(v, str) or not v for k, v in self.params.items() if k in string_fields):
            raise ValueError(f"{self.name}: имена файлов, полей и CRS должны быть непустыми строками")
        if "layers" in self.params:
            layers = self.params["layers"]
            if not isinstance(layers, dict) or not all(
                isinstance(k, str) and k and isinstance(v, str) and v for k, v in layers.items()
            ):
                raise ValueError("params.layers должен быть словарём непустых имён каналов и слоёв")
        if "distance_to_regex" in self.params:
            try:
                re.compile(self.params["distance_to_regex"])
            except re.error as e:
                raise ValueError(f"Некорректный distance_to_regex: {e}") from e
        if "fine_factor" in self.params:
            factor = self.params["fine_factor"]
            if type(factor) is not int or factor < 1:
                raise ValueError("fine_factor должен быть положительным целым")
        if "sigma_km" in self.params:
            sigma = self.params["sigma_km"]
            if (isinstance(sigma, bool) or not isinstance(sigma, (float, int))
                    or not 0 < sigma < float("inf")):
                raise ValueError("sigma_km должен быть конечным положительным числом")
        return self


class BlindRegionConfig(ConfigModel):
    name: str
    geometry: str  # путь к GeoJSON с границей; фиксируется до обучения
    buffer_m: float = Field(20000.0, gt=0)
    provisional: bool = False


class ValidationConfig(ConfigModel):
    blind_region: BlindRegionConfig
    spatial_block_size_m: float = Field(50000.0, gt=0)
    area_fractions: list[AreaFraction] = Field(default=[0.05, 0.10, 0.20], min_length=1)
    bootstrap_iterations: int = Field(2000, ge=100)
    confidence: float = Field(0.95, gt=0, lt=1)
    # Критерий успеха: нижняя граница интервала доли найденных объектов при success_area_fraction.
    success_area_fraction: AreaFraction = 0.20
    success_capture_lower_bound: float = Field(0.70, gt=0, le=1)
    temporal_test: bool = False


class ModelConfig(ConfigModel):
    kind: Literal["random_forest", "xgboost", "lightgbm", "pu_bagging"] = "random_forest"
    params: dict = {}


class TaskConfig(ConfigModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    deposit_type: str
    seed: int = 42
    grid: GridConfig = GridConfig()
    # Область обучения = объединение границ стран (geoboundaries ADM0).
    training_countries: list[CountryCode] = Field(default=["KAZ", "UZB", "KGZ", "TJK"], min_length=1)
    # Соседние страны, обрезанные по рамке сетки основных стран (сетка от них не меняется).
    extra_countries: list[CountryCode] = []
    labels: LabelsConfig
    layers: list[LayerConfig]
    model: ModelConfig = ModelConfig()
    validation: ValidationConfig

    @model_validator(mode="after")
    def check_layers(self) -> TaskConfig:
        names = [layer.name for layer in self.layers]
        countries = self.training_countries + self.extra_countries
        if len(countries) != len(set(countries)):
            raise ValueError("Основные и дополнительные страны должны быть уникальными")
        if len(names) != len(set(names)):
            raise ValueError("Имена слоёв должны быть уникальными")
        if not self.feature_layers:
            raise ValueError("Нужен хотя бы один слой с role: feature")
        for layer in self.layers:
            if layer.kind == "ggmplus_bouguer" and not any(
                x.name == "dem" and x.kind == "dem" for x in self.layers
            ):
                raise ValueError(f"{layer.name}: нужен слой dem с kind: dem")
        return self

    @property
    def feature_layers(self) -> list[LayerConfig]:
        return [layer for layer in self.layers if layer.role == "feature"]

    @property
    def benchmark_layers(self) -> list[LayerConfig]:
        return [layer for layer in self.layers if layer.role == "benchmark"]


def load_task(path: str | Path) -> TaskConfig:
    with open(path, encoding="utf-8") as f:
        return TaskConfig.model_validate(yaml.safe_load(f))

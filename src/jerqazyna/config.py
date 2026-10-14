"""Конфигурация задачи: тип месторождения, регион, слои, модель, схема валидации.

Новая задача = новый YAML-файл в configs/tasks/, а не новый код.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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


class GridConfig(BaseModel):
    crs: str = CENTRAL_ASIA_ALBERS
    cell_size_m: float = Field(1000.0, gt=0)

    @field_validator("crs")
    @classmethod
    def crs_must_be_equal_area(cls, v: str) -> str:
        # UTM, WGS84 и Web Mercator искажают площадь: запрещаем явно.
        lowered = v.lower()
        forbidden = ("epsg:4326", "epsg:3857", "+proj=utm", "+proj=longlat", "+proj=merc")
        if any(f in lowered for f in forbidden):
            raise ValueError(f"CRS должна быть равновеликой (aea, laea, cea), получено: {v}")
        return v


class LabelsConfig(BaseModel):
    source: str
    layer: str | None = None  # слой внутри geodatabase / GeoPackage
    name_field: str = "Name"
    status_field: str = "SiteStatus"
    include_prospects: bool = True
    # Буфер вокруг известных объектов, внутри которого спектральные признаки маскируются,
    # чтобы модель не училась находить карьеры и отвалы.
    mine_mask_buffer_m: float = Field(3000.0, ge=0)


class LayerConfig(BaseModel):
    name: str
    group: FeatureGroup
    source: str  # папка источника в data/raw
    kind: str  # обработчик из jerqazyna.features.layers.PROCESSORS
    role: LayerRole = "feature"
    params: dict = {}


class BlindRegionConfig(BaseModel):
    name: str
    geometry: str  # путь к GeoJSON с границей; фиксируется до обучения
    buffer_m: float = Field(20000.0, gt=0)


class ValidationConfig(BaseModel):
    blind_region: BlindRegionConfig
    spatial_block_size_m: float = Field(50000.0, gt=0)
    area_fractions: list[float] = [0.05, 0.10, 0.20]
    bootstrap_iterations: int = Field(2000, ge=100)
    confidence: float = Field(0.95, gt=0, lt=1)
    # Критерий успеха: нижняя граница интервала доли найденных объектов при success_area_fraction.
    success_area_fraction: float = 0.20
    success_capture_lower_bound: float = Field(0.70, gt=0, le=1)
    temporal_test: bool = False


class ModelConfig(BaseModel):
    kind: Literal["random_forest", "xgboost", "lightgbm", "pu_bagging"] = "random_forest"
    params: dict = {}


class TaskConfig(BaseModel):
    task_id: str
    deposit_type: str
    seed: int = 42
    grid: GridConfig = GridConfig()
    # Область обучения = объединение границ стран (geoboundaries ADM0).
    training_countries: list[str] = ["KAZ", "UZB", "KGZ", "TJK"]
    labels: LabelsConfig
    layers: list[LayerConfig]
    model: ModelConfig = ModelConfig()
    validation: ValidationConfig

    @model_validator(mode="after")
    def check_layers(self) -> TaskConfig:
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            raise ValueError("Имена слоёв должны быть уникальными")
        if not self.feature_layers:
            raise ValueError("Нужен хотя бы один слой с role: feature")
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

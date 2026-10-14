"""Каталог источников данных (configs/sources.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Access = Literal["direct", "tiles", "geoboundaries", "stac", "manual", "request"]
BBox = tuple[float, float, float, float]


class FileSpec(BaseModel):
    url: str
    filename: str
    unzip: bool = False


class BoundarySpec(BaseModel):
    iso: str = Field(pattern=r"^[A-Z]{3}$")
    level: Literal["ADM0", "ADM1", "ADM2"]


class StacSpec(BaseModel):
    endpoint: str
    collection: str
    bands: list[str]
    datetime: str
    months: list[int] = list(range(1, 13))
    max_cloud: float = 20
    resolution_m: float = 100
    scenes_per_mgrs: int = Field(8, ge=1)


class Source(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    roles: list[str]
    access: Access
    license: str
    citation: str
    size_mb: float | None = None
    files: list[FileSpec] = []
    tile_url: str | None = None
    tile_step_deg: int = Field(1, ge=1)
    boundaries: list[BoundarySpec] = []
    stac: StacSpec | None = None
    page: str | None = None
    expect: list[str] = []
    notes: str = ""

    @model_validator(mode="after")
    def check_access_fields(self) -> Source:
        required = {
            "direct": self.files,
            "tiles": self.tile_url,
            "geoboundaries": self.boundaries,
            "stac": self.stac,
            "manual": self.page and self.expect,
            "request": self.expect,
        }
        if not required[self.access]:
            raise ValueError(f"{self.id}: не хватает полей для access={self.access}")
        return self

    @property
    def automatic(self) -> bool:
        return self.access not in ("manual", "request")


class Catalog(BaseModel):
    aoi: dict
    priority_bbox: BBox
    sources: list[Source]

    @model_validator(mode="after")
    def unique_ids(self) -> Catalog:
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("id источников должны быть уникальными")
        return self

    @property
    def bbox(self) -> BBox:
        return tuple(self.aoi["bbox"])

    def get(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(f"Нет источника {source_id!r}; есть: {', '.join(s.id for s in self.sources)}")


def load_catalog(path: str | Path = "configs/sources.yaml") -> Catalog:
    with open(path, encoding="utf-8") as f:
        return Catalog.model_validate(yaml.safe_load(f))

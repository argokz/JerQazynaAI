"""Настройки веб-карты из переменных окружения. Секретов здесь нет и быть не должно."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# web/api/jerqazyna_web/settings.py -> корень проекта (в контейнере — /app).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PREFIX = "JERQAZYNA_WEB_"
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:8080"


class Settings(BaseModel):
    registry: Path = PROJECT_ROOT / "web" / "layers.yaml"
    sources: Path = PROJECT_ROOT / "configs" / "sources.yaml"
    # Каталог экспорта: manifest.json, area_mask.tif и cogs/ (см. export.py).
    data_dir: Path = PROJECT_ROOT / "data" / "web"
    # Адрес TiTiler, каким его видит браузер, и путь к cogs/ внутри контейнера TiTiler.
    tiles_url: str = "http://localhost:8001"
    tiles_data_dir: str = "/data/cogs"
    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("tiles_url")
    @classmethod
    def check_tiles_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"tiles_url должен начинаться с http:// или https://: {value!r}")
        return value.rstrip("/")

    @field_validator("cors_origins")
    @classmethod
    def check_origins(cls, values: list[str]) -> list[str]:
        if "*" in values:
            raise ValueError("CORS: укажите конкретные origins вместо '*'")
        return values

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(ENV_PREFIX + name)
            return value.strip() if value and value.strip() else None

        values: dict = {}
        for name in ("registry", "sources", "data_dir", "tiles_url", "tiles_data_dir"):
            value = get(name.upper())
            if value is not None:
                values[name] = value
        # Пустое значение переменной отключает CORS; отсутствие — локальные адреса разработки.
        origins = env.get(ENV_PREFIX + "CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
        values["cors_origins"] = [o.strip() for o in origins.split(",") if o.strip()]
        return cls(**values)

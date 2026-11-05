"""Манифест экспорта COG (data/web/manifest.json) и поиск записанных хешей исходников."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from jerqazyna_web.registry import Provenance

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
COG_DIR = "cogs"


class SourceProvenance(BaseModel):
    """Где записан хеш исходника и чем он подтверждён (рецепт кэша, run_id, протокол)."""

    manifest: str
    recorded_sha256: str
    details: dict[str, str] = {}


class CogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str                         # относительно каталога данных веб-карты
    source: str                       # исходный растр, относительно корня проекта
    band: int
    source_sha256: str
    provenance: SourceProvenance | None = None
    cog_sha256: str
    recipe: str
    dtype: str
    nodata: float | None
    coverage: float | None            # доля ячеек области с действительным значением
    exported_at: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = MANIFEST_VERSION
    task_id: str
    grid: dict[str, Any]
    area: CogEntry | None = None
    layers: dict[str, CogEntry] = {}
    packages: dict[str, str] = {}


def load_manifest(path: Path) -> Manifest | None:
    if not path.exists():
        return None
    try:
        manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"Повреждён манифест COG {path}: {e}") from e
    if manifest.version != MANIFEST_VERSION:
        raise ValueError(f"{path}: версия манифеста {manifest.version}, нужна {MANIFEST_VERSION}")
    return manifest


def _dotted(value: Any, key: str) -> Any:
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def recorded_hash(provenance: Provenance, root: Path, filename: str) -> SourceProvenance:
    """Хеш файла из манифеста подготовки. Поддерживаются форматы проекта:

    - cache.json признаков: layers.<слой>.files[<файл>], подтверждается рецептом слоя;
    - run.json, report.json прогонов: files[<файл>] плюс run_id / protocol_id;
    - явный ключ provenance.key (например mask_sha256 в area.json).
    """
    path = root / provenance.manifest
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"Не удалось прочитать манифест происхождения {path}: {e}") from e
    details: dict[str, str] = {}
    digest = None
    if provenance.key:
        digest = _dotted(data, provenance.key)
    else:
        files = data.get("files")
        if isinstance(files, dict) and filename in files:
            digest = files[filename]
        for name, entry in (data.get("layers") or {}).items():
            if isinstance(entry, dict) and filename in (entry.get("files") or {}):
                digest = entry["files"][filename]
                details = {"layer": name, "recipe": str(entry.get("recipe", ""))}
                break
    for key in ("run_id", "protocol_id", "protocol_sha256"):
        if isinstance(data.get(key), str):
            details[key] = data[key]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"В {path} нет sha256 для {filename}")
    return SourceProvenance(
        manifest=provenance.manifest, recorded_sha256=digest, details=details,
    )

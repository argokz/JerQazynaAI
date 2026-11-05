"""Версионируемые решения по train-меткам без изменения исходников и pipeline обучения."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from jerqazyna.config import ConfigModel
from jerqazyna.features.cache import Digests, fingerprint, write_json


class RecordRef(ConfigModel):
    source: str
    layer: str | None = None
    row: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Evidence(ConfigModel):
    uri: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    sha256: str | None = Field(None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def check_uri(self):
        if not self.uri.startswith(("local:", "https://", "http://")):
            raise ValueError("Нужна локальная ссылка или HTTP(S) URI доказательства")
        return self


class Decision(ConfigModel):
    decision_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    record: RecordRef
    state: Literal["confirmed", "pending", "rejected"]
    before: dict[str, Any]
    changes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(min_length=1)
    reason: str = Field(min_length=1)
    status_as_of: str | None = None

    @model_validator(mode="after")
    def check_fields(self):
        allowed = {"country", "country_iso", "status", "lon", "lat"}
        if (set(self.before) | set(self.changes)) - allowed:
            raise ValueError("Разрешены правки страны, статуса и пары lon/lat")
        if set(self.changes) - set(self.before):
            raise ValueError("Для каждого нового значения нужно before")
        if {"lon", "lat"} & self.changes.keys() and not {"lon", "lat"} <= self.changes.keys():
            raise ValueError("Координаты меняются только парой lon/lat")
        if "status" in self.changes and self.changes["status"] not in {"deposit", "prospect", "unknown"}:
            raise ValueError("Неизвестный статус")
        if self.state == "confirmed" and not self.changes:
            raise ValueError("Подтверждённая правка должна содержать изменения")
        for field in ("country", "country_iso"):
            if field in self.changes and (not isinstance(self.changes[field], str) or not self.changes[field].strip()):
                raise ValueError("Страна должна быть непустой строкой")
        if "country_iso" in self.changes:
            import re

            if not re.fullmatch(r"[A-Z]{3}", self.changes["country_iso"]):
                raise ValueError("country_iso должен содержать три заглавные буквы")
        if self.state == "confirmed" and all(self.before[k] == v for k, v in self.changes.items()):
            raise ValueError("Подтверждённая правка не должна повторять прежнее значение")
        if self.status_as_of is not None:
            from datetime import date

            date.fromisoformat(self.status_as_of)
        for key, low, high in [("lon", -180, 180), ("lat", -90, 90)]:
            if key in self.changes:
                v = self.changes[key]
                if isinstance(v, bool) or not isinstance(v, (float, int)) or not low <= v <= high:
                    raise ValueError("Координата вне диапазона")
        return self


class ReviewConfig(ConfigModel):
    review_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    decisions: list[Decision] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({d.decision_id for d in self.decisions}) != len(self.decisions):
            raise ValueError("Повтор decision_id")
        return self


def load_review(path):
    return ReviewConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def local_path(root: Path, value: str):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Локальные источники должны находиться внутри проекта")
    return path


def source_sha(root, source, digests):
    path = local_path(root, source)
    return fingerprint(digests.snapshot(path)) if path.is_dir() else digests.file(path)


def parent_labels(task, root, baseline):
    import pandas as pd

    saved = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    if saved["spec"]["task"] != task.model_dump() or saved["holdout_evaluated"]:
        raise ValueError("Нужен исходный baseline с закрытым тестом и тем же task")
    dataset = local_path(root, saved["dataset_path"])
    meta = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if meta["dataset_id"] != saved["spec"]["dataset_id"]:
        raise ValueError("Родительская таблица не соответствует baseline")
    if Digests().file(dataset / "labels.csv") != meta["files"]["labels.csv"]:
        raise ValueError("Повреждена родительская таблица меток")
    labels = pd.read_csv(dataset / "labels.csv")
    if not labels.partition.eq("train").all():
        raise ValueError("Реестр работает только с train-метками")
    return labels, meta, saved


def run_review(task, config, root, baseline):
    import numpy as np
    import pandas as pd
    from pyproj import Transformer

    from jerqazyna.grid.template import GridSpec, read_layer

    root, baseline = Path(root), Path(baseline)
    labels, dataset, parent = parent_labels(task, root, baseline)
    original = labels.copy(deep=True)
    digests = Digests()
    record_rows = {}
    for i, records in enumerate(labels.source_records):
        for record in json.loads(records):
            key = (record["source"], record.get("layer"), str(record["row"]))
            record_rows.setdefault(key, set()).add(i)
    used, review_rows, differences = set(), [], []
    verified_sources = {}
    for decision in config.decisions:
        ref = decision.record
        actual = verified_sources.setdefault(ref.source, source_sha(root, ref.source, digests))
        if actual != ref.sha256:
            raise ValueError(f"Устаревший SHA-256 источника: {ref.source}")
        matches = record_rows.get((ref.source, ref.layer, ref.row), set())
        if len(matches) != 1:
            raise ValueError(f"Неоднозначное или отсутствующее соответствие: {decision.decision_id}")
        i = next(iter(matches))
        for field, expected in decision.before.items():
            value = original.at[i, field] if field in original else None
            value = ("unknown" if field == "status" else None) if pd.isna(value) else value
            if value != expected:
                raise ValueError(f"before не соответствует родителю: {decision.decision_id}/{field}")
        for evidence in decision.evidence:
            if evidence.uri.startswith("local:"):
                path = local_path(root, evidence.uri[6:].split("#", 1)[0])
                actual_evidence = source_sha(root, str(path.relative_to(root.resolve())), digests)
                if evidence.sha256 is None or actual_evidence != evidence.sha256:
                    raise ValueError(f"Не совпал SHA-256 доказательства: {decision.decision_id}")
        object_id = fingerprint({"source": ref.source, "layer": ref.layer, "row": ref.row})[:20]
        review_rows.append({"decision_id": decision.decision_id, "object_id": object_id,
                            "name": original.at[i, "name"], "state": decision.state,
                            "source": ref.source, "layer": ref.layer, "source_row": ref.row,
                            "source_sha256": ref.sha256,
                            "before": json.dumps(decision.before, ensure_ascii=False),
                            "proposed": json.dumps(decision.changes, ensure_ascii=False),
                            "reason": decision.reason, "status_as_of": decision.status_as_of,
                            "evidence": json.dumps([e.model_dump() for e in decision.evidence], ensure_ascii=False)})
        if decision.state != "confirmed":
            continue
        for field, after in decision.changes.items():
            if (i, field) in used:
                raise ValueError(f"Повторная правка поля: {decision.decision_id}/{field}")
            used.add((i, field))
            before = decision.before[field]
            if f"original_{field}" not in labels:
                labels[f"original_{field}"] = original.get(field, None)
            if field not in labels:
                labels[field] = pd.Series([None] * len(labels), dtype=object)
            labels.at[i, field] = after
            labels.loc[i, f"review_{field}_decision_id"] = decision.decision_id
            differences.append({"decision_id": decision.decision_id, "object_id": object_id,
                                "field": field, "before": before, "after": after})
        if "lon" in decision.changes:
            grid = GridSpec(**dataset["spec"]["grid"])
            x, y = Transformer.from_crs(4326, grid.crs, always_xy=True).transform(labels.at[i, "lon"], labels.at[i, "lat"])
            rows, cols = grid.rowcol(np.array([x]), np.array([y]))
            row, col = int(rows[0]), int(cols[0])
            split = root / "data/features" / task.task_id / "splits" / dataset["spec"]["split_id"][:16]
            split_meta = json.loads((split / "split.json").read_text(encoding="utf-8"))
            if (digests.file(split / "split.json") != dataset["spec"]["split_manifest_sha256"]
                    or digests.file(split / "train.tif") != split_meta["files"]["train.tif"]):
                raise ValueError("Повреждена маска train")
            train = read_layer(split / "train.tif").astype(bool)
            if not (0 <= row < grid.height and 0 <= col < grid.width and train[row, col]):
                raise ValueError("Правка координат выходит за train или попадает в закрытую область")
            for field, value in {"x": x, "y": y, "row": row, "col": col, "matrix_row": -1}.items():
                labels.at[i, field] = value
    spec = {"version": 1, "review": config.model_dump(), "parent_run_id": parent["run_id"],
            "status_policy": "blank interpreted as unknown for review; preserved verbatim unless confirmed",
            "parent_manifest_sha256": digests.file(baseline / "run.json"),
            "parent_dataset_id": dataset["dataset_id"], "source_sha256": verified_sources,
            "parent_labels_sha256": dataset.get("files", {}).get("labels.csv"),
            "implementation_sha256": digests.file(Path(__file__))}
    review_id = fingerprint(spec)
    dest = root / "data/curated" / task.task_id / config.review_id / review_id[:16]
    marker = dest / "manifest.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["spec"] != spec or any(digests.file(dest / p) != h for p, h in saved["files"].items()):
            raise ValueError("Повреждено поколение рецензированных меток")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    labels.to_csv(dest / "train_labels_curated.csv", index=False, encoding="utf-8")
    pd.DataFrame(review_rows).to_csv(dest / "review.csv", index=False, encoding="utf-8")
    pd.DataFrame(differences, columns=["decision_id", "object_id", "field", "before", "after"]).to_csv(
        dest / "diff.csv", index=False, encoding="utf-8")
    write_json(dest / "decisions.json", config.model_dump())
    write_json(marker, {"review_id": review_id, "spec": spec, "n_objects": len(labels),
                        "states": pd.Series([d.state for d in config.decisions]).value_counts().to_dict(),
                        "holdout_evaluated": False, "training_pipeline_changed": False,
                        "files": {p.name: digests.file(p) for p in sorted(dest.iterdir()) if p != marker and ".part" not in p.name}})
    return dest

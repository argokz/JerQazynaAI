"""Реестр слоёв: схема и лицензионные правила показа (без растров)."""

import copy

import pytest
import yaml
from jerqazyna_web.registry import Registry, check_licenses, load_registry
from web_helpers import FEATURES, ROOT, registry_dict


def _check(raw, catalog):
    check_licenses(Registry.model_validate(raw), catalog)


def test_project_registry_loads(catalog):
    registry = load_registry(ROOT / "web" / "layers.yaml", catalog)
    ids = {layer.id for layer in registry.layers}
    assert {"dem_elevation", "magnetics_anomaly", "gravity_bouguer", "lithology_distance"} <= ids
    sources = {s for layer in registry.layers for s in layer.sources}
    assert "usgs_fsu_geology" not in sources and "geoboundaries" not in sources
    for layer in registry.layers:
        if layer.role == "research":
            assert not layer.display.visible and "слепой тест" in layer.warning


@pytest.mark.parametrize("blocked", ["usgs_fsu_geology", "geoboundaries"])
def test_blocked_sources_rejected(catalog, blocked):
    raw = registry_dict(FEATURES)
    raw["attributions"][blocked] = {"text": "x"}
    raw["layers"][0]["sources"] = [blocked]
    with pytest.raises(ValueError, match="лицензионные"):
        _check(raw, catalog)


def test_share_alike_layer_must_stand_alone(catalog):
    raw = registry_dict(FEATURES)
    raw["layers"][2]["sources"] = ["gem_active_faults", "cop_dem_glo90"]
    with pytest.raises(ValueError, match="share-alike"):
        _check(raw, catalog)


def test_unknown_source_and_missing_attribution(catalog):
    raw = registry_dict(FEATURES)
    raw["layers"][0]["sources"] = ["no_such_source"]
    with pytest.raises(ValueError, match="нет в каталоге"):
        _check(raw, catalog)
    raw = registry_dict(FEATURES)
    del raw["attributions"]["emag2v3"]
    with pytest.raises(ValueError, match="атрибуции"):
        _check(raw, catalog)


def test_disabled_layer_is_not_checked(catalog):
    raw = registry_dict(FEATURES)
    raw["layers"][0]["sources"] = ["usgs_fsu_geology"]
    raw["layers"][0]["enabled"] = False
    _check(raw, catalog)


def test_research_layer_rules():
    base = registry_dict(FEATURES)
    raw = copy.deepcopy(base)
    raw["layers"][3]["display"]["visible"] = True
    with pytest.raises(ValueError, match="по умолчанию"):
        Registry.model_validate(raw)
    raw = copy.deepcopy(base)
    raw["layers"][3]["warning"] = ""
    with pytest.raises(ValueError, match="warning"):
        Registry.model_validate(raw)


def test_schema_errors():
    raw = registry_dict(FEATURES)
    raw["layers"][1]["id"] = raw["layers"][0]["id"]
    with pytest.raises(ValueError, match="повторяются"):
        Registry.model_validate(raw)
    raw = registry_dict(FEATURES)
    raw["layers"][0]["display"]["palette"] = "rainbow"
    with pytest.raises(ValueError, match="палитра"):
        Registry.model_validate(raw)
    raw = registry_dict(FEATURES)
    raw["layers"][0]["display"]["min"] = 5000
    with pytest.raises(ValueError, match="диапазон"):
        Registry.model_validate(raw)
    raw = registry_dict(FEATURES)
    raw["layers"][0]["path"] = "typo"
    with pytest.raises(ValueError):
        Registry.model_validate(raw)


def test_missing_registry_file(tmp_path, catalog):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "missing.yaml", catalog)
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"version": 2}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(tmp_path / "bad.yaml", catalog)

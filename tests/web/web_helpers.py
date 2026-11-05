"""Общие константы и реестр синтетического проекта для тестов веб-карты."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRS = "+proj=aea +lat_0=45 +lat_1=40 +lat_2=52 +lon_0=70 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
FEATURES = "data/features/test_task"
# Сетка 300×260 км с центром проекции (70° в.д., 45° с.ш.) внутри; больше блока 128,
# чтобы COG получил обзоры.
WIDTH, HEIGHT = 300, 260
X_MIN, Y_MAX = -150_000.0, 130_000.0
AREA_COLS = 20          # первые 20 столбцов вне области


def registry_dict(features: str) -> dict:
    return {
        "version": 1,
        "task_id": "test_task",
        "grid": f"{features}/grid.json",
        "area": {"file": f"{features}/area_mask.tif",
                 "provenance": {"manifest": f"{features}/area.json", "key": "mask_sha256"}},
        "attributions": {
            "cop_dem_glo90": {"text": "Produced using Copernicus WorldDEM-90", "notice": "no liability"},
            "emag2v3": {"text": "EMAG2v3, NOAA"},
            "gem_active_faults": {"text": "GEM, CC BY-SA 4.0"},
            "glim": {"text": "GLiM"},
        },
        "layers": [
            {"id": "dem_elevation", "title": "Высота", "file": f"{features}/dem__elevation_m.tif",
             "units": "м", "sources": ["cop_dem_glo90"],
             "provenance": {"manifest": f"{features}/cache.json"},
             "display": {"palette": "terrain", "min": 0, "max": 3000, "visible": True}},
            {"id": "magnetics_anomaly", "title": "Магнитка",
             "file": f"{features}/magnetics__anomaly_nt.tif", "units": "нТл",
             "sources": ["emag2v3"], "provenance": {"manifest": f"{features}/cache.json"},
             "display": {"palette": "rdbu_r", "min": -100, "max": 100}},
            {"id": "faults_active_distance", "title": "Разломы",
             "file": f"{features}/faults_active__distance_km.tif", "units": "км",
             "sources": ["gem_active_faults"],
             "display": {"palette": "viridis_r", "min": 0, "max": 50}},
            {"id": "research_score", "title": "Балл", "file": "outputs/run/train_score.tif",
             "role": "research", "sources": ["cop_dem_glo90", "emag2v3", "glim"],
             "provenance": {"manifest": "outputs/run/run.json"},
             "warning": "исследовательский; не прошёл слепой тест",
             "display": {"palette": "magma", "min": 0, "max": 1}},
        ],
    }

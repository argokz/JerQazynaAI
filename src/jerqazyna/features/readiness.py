"""Отдельная проверка доступности входов; схема YAML проверяется без геостека."""

from pathlib import Path

from jerqazyna.config import TaskConfig
from jerqazyna.features.cache import Digests, grid_recipe


def check_inputs(task: TaskConfig, root: Path) -> list[str]:
    import geopandas as gpd

    issues = []
    try:
        grid_recipe(task, root, Digests())
    except OSError as e:
        issues.append(str(e))
    for source in [task.labels, *task.labels.merge]:
        if source.required and not (root / source.source).exists():
            issues.append(f"Нет источника меток: {source.source}")
    for layer in task.layers:
        path = root / layer.source
        if layer.required and (not path.exists() or (path.is_dir() and not any(path.iterdir()))):
            issues.append(f"{layer.name}: нет обязательного источника {path}")
    path = root / task.validation.blind_region.geometry
    if not path.exists():
        issues.append(f"Нет границы слепого региона: {path}")
    else:
        try:
            gdf = gpd.read_file(path)
            if (gdf.empty or gdf.crs is None or gdf.geometry.is_empty.any()
                    or gdf.geometry.isna().any() or not gdf.geometry.is_valid.all()
                    or not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
                issues.append(f"Некорректная геометрия слепого региона: {path}")
        except (OSError, ValueError) as e:
            issues.append(f"Не удалось прочитать границу региона: {e}")
    grid_path = root / "data" / "features" / task.task_id / "grid.json"
    if grid_path.exists():
        from pyproj import CRS

        from jerqazyna.grid.template import GridSpec

        try:
            grid = GridSpec.load(grid_path)
            if (grid.cell_size != task.grid.cell_size_m
                    or not CRS(grid.crs).equals(CRS(task.grid.crs))):
                issues.append("Сохранённая сетка не соответствует конфигурации; нужен новый task_id")
        except (OSError, ValueError, TypeError) as e:
            issues.append(f"Некорректная сохранённая сетка: {e}")
    return issues

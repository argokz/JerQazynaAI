"""JerQazyna: ML-прогноз перспективности месторождений по открытым данным."""

import os
import sys
from importlib.util import find_spec
from pathlib import Path

__version__ = "0.1.0"


def _use_bundled_geodata() -> None:
    """Направляет PROJ и GDAL на данные из колеса rasterio, если окружение указывает на чужие.

    Например, установщик PostgreSQL/PostGIS прописывает PROJ_LIB и GDAL_DATA глобально,
    и rasterio падает на несовместимой proj.db. Меняется только окружение этого процесса.
    """
    spec = find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    pkg = Path(spec.origin).parent
    prefix = Path(sys.prefix).resolve()
    for names, folder in ((("PROJ_DATA", "PROJ_LIB"), pkg / "proj_data"), (("GDAL_DATA",), pkg / "gdal_data")):
        if not folder.is_dir():
            continue
        foreign = any(
            os.environ.get(n) and not Path(os.environ[n]).resolve().is_relative_to(prefix) for n in names
        )
        if foreign:
            for n in names:
                os.environ[n] = str(folder)


_use_bundled_geodata()

"""Веб-карта JerQazyna: реестр слоёв, экспорт COG и API карточек ячеек.

Пакет jerqazyna импортируется первым: он направляет PROJ/GDAL на данные колеса
rasterio до первого импорта rasterio (см. jerqazyna._use_bundled_geodata).
"""

import jerqazyna  # noqa: F401

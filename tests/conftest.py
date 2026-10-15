# Пакет импортируется до rasterio, чтобы успеть направить PROJ/GDAL на собственные данные
# (см. jerqazyna._use_bundled_geodata).
import jerqazyna  # noqa: F401

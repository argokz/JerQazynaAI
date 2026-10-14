# JerQazyna AI

ML-прогноз перспективности месторождений Казахстана на открытых данных.

Первая задача — порфировая медь. Модель обучается на Центрально-Азиатском складчатом поясе и проверяется на «слепом» Прибалхашье: ни один объект этого региона не участвует в обучении.

План проекта: [План проекта: ИИ-прогноз месторождений Казахстана](https://claude.ai/code/artifact/518b2749-0b8a-42f2-849c-8496b37b27b3).

## Принципы

- **Задача = конфиг.** Тип месторождения, регион, слои, модель и схема валидации описываются в `configs/tasks/*.yaml`. Новая задача требует нового конфига, а не нового кода.
- **Только пространственная валидация.** Случайная кросс-валидация используется лишь для отладки. В отчёт идут блочная CV, leave-one-belt-out и слепой регион с буфером.
- **Равновеликая проекция.** Метрика считается в долях площади, поэтому сетка строится в проекции Альберса, а не в UTM или WGS84. Конфиг с другой проекцией не пройдёт проверку.
- **Без утечек.** Зоны USGS 2010 используются только как эталон (`role: benchmark`). Спектральные признаки маскируются вокруг известных карьеров.
- **Критерий по нижней границе интервала.** При десятках объектов точечная оценка ненадёжна, поэтому доля найденных объектов считается с бутстрэп-интервалом.

## Структура

```
configs/tasks/        конфиги задач
src/jerqazyna/
  config.py           схема конфига (pydantic)
  grid/               сетка в равновеликой проекции
  data/               загрузка исходных слоёв
  features/           расчёт признаков
  models/             RF, XGBoost / LightGBM, PU-learning
  validation/         разбиения и кривые обнаружения
data/                 данные (не в git, см. data/README.md)
notebooks/            исследования
```

## Быстрый старт

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"       # ядро; геостек и ML: pip install -e ".[geo,ml,dev]"
pytest
jerqazyna validate-config configs/tasks/porphyry_cu_balkhash.yaml
```

## Сбор данных

Каталог открытых источников с проверенными ссылками и лицензиями: [configs/sources.yaml](configs/sources.yaml). Файлы ложатся в `data/raw/<id>/`, каждый записывается в `data/manifest.csv` с SHA-256.

```bash
jerqazyna data list                        # источники
jerqazyna data fetch --all                 # всё автоматическое, кроме Sentinel-2
jerqazyna data fetch sentinel2_l2a --workers 4
jerqazyna data status                      # что собрано
```

Вручную остаются только база USGS 2025 (ScienceBase закрыт проверкой Cloudflare), ASTER (нужен аккаунт Earthdata) и данные НГС РК (по запросу). Вместо WGM2012, выгрузка которого закрыта reCAPTCHA, используется GGMplus.

## Подготовка слоёв на сетке

```bash
jerqazyna prepare configs/tasks/porphyry_cu_balkhash.yaml            # все слои
jerqazyna prepare configs/tasks/porphyry_cu_balkhash.yaml --layers sentinel2 --force
```

Сетка 1 км в проекции Альберса покрывает KZ, UZ, KG и TJ. Результат лежит в `data/features/<task_id>/`: `grid.json`, `area_mask.tif`, `labels.csv`, по одному GeoTIFF на канал слоя (`<слой>__<канал>.tif`) и опись `layers.json`. Растры усредняются точно: сумма значений и сумма весов переносятся отдельно, поэтому ячейки на стыке тайлов не искажаются. Маска выработок хранится отдельно и применяется к спектральным признакам при сборке таблицы для модели.

## Инфраструктура

Инфраструктура исследовательской фазы (PostGIS, MinIO, MLflow):

```bash
cp .env.example .env          # задайте пароли
docker compose up -d
```

FastAPI, очередь задач, TiTiler и веб-интерфейс строятся после сбора данных.

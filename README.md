# JerQazyna AI

ML-прогноз перспективности месторождений Казахстана на открытых данных.

Первая задача — порфировая медь. Реализованы подготовка данных, предварительное разбиение и baseline-обучение с пространственной CV на Центрально-Азиатском складчатом поясе. Прибалхашье отложено для отдельной проверки после фиксации модели.

План работ и критерии готовности: [docs/ROADMAP.md](docs/ROADMAP.md).
Дополнительные открытые данные и результаты проверки на 19.10.2026:
[docs/DATA_SOURCES_2026-10-19.md](docs/DATA_SOURCES_2026-10-19.md).
Исходная концепция: [ИИ-прогноз месторождений Казахстана](https://claude.ai/code/artifact/518b2749-0b8a-42f2-849c-8496b37b27b3).

## Принципы

- **Задача = конфиг.** Тип месторождения, регион, слои, модель и схема валидации описываются в `configs/tasks/*.yaml`. Новая задача требует нового конфига, а не нового кода.
- **Только пространственная валидация.** Случайная кросс-валидация используется лишь для отладки. В отчёт идут блочная CV, leave-one-belt-out и слепой регион с буфером.
- **Равновеликая проекция.** Метрика считается в долях площади, поэтому сетка строится в проекции Альберса, а не в UTM или WGS84. Конфиг с другой проекцией не пройдёт проверку.
- **Без утечек.** Зоны USGS 2010 используются только как эталон (`role: benchmark`). Спектральные признаки маскируются вокруг известных карьеров.
- **Критерий по нижней границе интервала.** При десятках объектов точечная оценка ненадёжна. `estimate_capture` использует exact-интервал с учётом неопределённости равных прогнозов; пространственную зависимость нужно проверять отдельно. Бутстрэп по объектам сохранён с защитой от вырожденного интервала.

## Структура

```
configs/tasks/        конфиги задач
src/jerqazyna/
  config.py           схема конфига (pydantic)
  grid/               сетка в равновеликой проекции
  data/               загрузка исходных слоёв
  features/           расчёт признаков
  models/             Logistic Regression, RF, таблица признаков и пространственная CV
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

Зафиксированные версии ядра, геостека и средств проверки для Python 3.10:

```bash
pip install -c requirements/py310-geo.txt -e ".[geo,dev]"
jerqazyna validate-config configs/tasks/porphyry_cu_balkhash.yaml --check-inputs
```

Без `--check-inputs` проверяется схема конфигурации; с этим флагом также проверяются
доступность обязательных источников, граница региона и соответствие сохранённой сетки.
ML-зависимости зафиксированы отдельно для baseline ниже.

Для первого ML-эксперимента есть отдельный набор зависимостей:

```bash
pip install -c requirements/py310-baseline.txt -e ".[geo,baseline,dev]"
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
jerqazyna prepare-labels configs/tasks/porphyry_cu_balkhash.yaml  # только сетка и метки
jerqazyna prepare-split configs/tasks/porphyry_cu_balkhash.yaml   # train/test/20-км буфер
```

Сетка 1 км в проекции Альберса покрывает KZ, UZ, KG и TJ. Результат лежит в `data/features/<task_id>/`: `grid.json`, `area_mask.tif`, `labels.csv`, по одному GeoTIFF на канал слоя (`<слой>__<канал>.tif`) и опись `layers.json`. Растры усредняются точно: сумма значений и сумма весов переносятся отдельно, поэтому ячейки на стыке тайлов не искажаются. Маска выработок хранится отдельно и применяется к спектральным признакам при сборке таблицы для модели.

`cache.json` хранит отпечатки параметров, исходных файлов, кода, библиотек и готовых
каналов. Содержимое файлов проверяется SHA-256, поэтому проверка больших источников
занимает время. Старые результаты без этих отпечатков нужно пересчитать: наличие TIFF
не подтверждает актуальность. `--layers gravity` сначала проверит/подготовит `dem`;
изменение зависимости отзывает готовность производных слоёв. Невыбранные устаревшие
слои исключаются из описи до их следующего пересчёта.

При изменении CRS, шага или геометрии сетки используйте новый `task_id`: `--force`
не перезаписывает несовместимую сетку. Слой можно объявить необязательным через
`required: false`; отсутствие обязательного слоя или меток даёт ненулевой код CLI.
В `labels.json` сохраняются хеши источников, конфигурация и хеш таблицы меток.

Предварительная граница Прибалхашья — рамка 72–80° в.д., 44,5–48,5° с.ш., выбранная
до обучения. Она помечена `provisional: true` и не заменяет геологически утверждённую
границу. Разбиения лежат в `splits/<идентификатор>/`: маски `train.tif`, `test.tif`,
`buffer.tif`, таблица назначений объектов и `split.json` с хешами. Буфер измеряется от
полигона региона. Существующие поколения сохраняются, повреждённые не используются.

## Первый ML-эксперимент

Первый запуск выполнен: [отчёт и ограничения](docs/BASELINE_2026-10-19.md).
Random Forest охватывает 39/52 объектов при 20% площади на пространственной CV;
нижняя граница exact 95% — 61,1%. Слепой регион не оценивался.

Протокол: [configs/experiments/porphyry_cu_baseline.yaml](configs/experiments/porphyry_cu_baseline.yaml).
Эффективные гиперпараметры моделей, фоновые выборки и буфер CV задаёт YAML эксперимента.
Сначала подготовьте базовые слои, затем одной командой получите CV, сохранённую модель
и карту обучающей территории:

```bash
jerqazyna prepare configs/tasks/porphyry_cu_balkhash.yaml --layers dem,magnetics,gravity,lithology
jerqazyna train configs/tasks/porphyry_cu_balkhash.yaml --experiment configs/experiments/porphyry_cu_baseline.yaml
```

В таблицу на диске входят только train-ячейки. Метки неизвестного статуса исключены
из положительного класса и оценки; все известные train-объекты исключаются из фона.
Непомеченный фон не считается доказанным отсутствием месторождений.
Маска выработок обязательна, если в эксперимент добавлены спектральные признаки;
слои benchmark, mask и bias запрещены в X.

Сравниваются логистическая модель, Random Forest и эвристика по расстоянию до интрузий.
Преобразования обучаются внутри каждого fold. Блоки 50 км, буфер между train/validation
10 км; фон — 50 000 ячеек, два фиксированных seed. Захват считается при одной доле площади
внутри каждого validation-фолда; шкалы scores разных моделей фолдов не смешиваются.
Публикуются exact-интервал и отдельный бутстрэп по пространственным блокам с объектами.

Результат сохраняется в `outputs/<task_id>/<experiment_id>/<run_id>/`:
`protocol.json`, `cv.json`, `oof_objects.csv`, OOF-карты, `model.joblib`, `train_score.tif`
и `run.json` с хешами. Checkpoint каждого fit содержит модель и индексы выборки.
Завершённый запуск проверяет хеши при повторном обращении.

```bash
jerqazyna evaluate outputs/<task_id>/<experiment_id>/<run_id>
jerqazyna predict outputs/<task_id>/<experiment_id>/<run_id> --output outputs/repeat_train_score.tif
```

`evaluate` показывает сохранённую CV на train. `predict` повторяет карту train по окнам;
ячейки отложенного региона и буфера остаются NaN. Эти команды не открывают тест Прибалхашья.
Score — относительная оценка positive/background, а не калиброванная вероятность месторождения.
Выбор лучшего классификатора по CV не является независимым подтверждением его качества.

## Проверка устойчивости на train

Результаты и ограничения: [отчёт серии](docs/SENSITIVITY_2026-10-19.md).

Фиксированный список сравнений задан в
[porphyry_cu_sensitivity.yaml](configs/experiments/porphyry_cu_sensitivity.yaml):
блоки 25/50/100 км, исключение геофизики и литологии, обучение только на deposit.
Для всех вариантов сохраняются исходные train-ячейки, оценочные объекты, параметры
Random Forest и фоновые seed. При исключении признаков площадь не сокращается.
Отдельно публикуется оценка по общей подгруппе deposit и парный бутстрэп разницы
с эвристикой. Изменение размера блока также меняет назначение блоков фолдам.

```powershell
jerqazyna sensitivity configs/tasks/porphyry_cu_balkhash.yaml --baseline outputs/porphyry_cu_balkhash/porphyry_cu_baseline/ce67b6ff52175829 --protocol configs/experiments/porphyry_cu_sensitivity.yaml --split data/features/porphyry_cu_balkhash/splits/4c6107e6c0a9f029
```

Команда проверяет происхождение исходного запуска и таблицы, использует его OOF
для исходного варианта, сохраняет checkpoints новых CV-fit и train-only OOF-карты
в `outputs/<task_id>/<study_id>/<идентификатор>/`. Серия не выбирает новую финальную
модель и не применяет её в слепом регионе. Парные интервалы условны на наблюдаемых
объектах и не скорректированы на множественные сравнения.

## Страны и бэггинг непомеченного фона

[Аудит и протокол](docs/COUNTRY_PU_2026-10-19.md) проверяют страны по координатам,
сохраняя исходные метки. Найдены четыре пустых поля страны, одно расхождение и
перекрытия границ. Производная карта стран используется для фоновых квот и диагностики,
не входит в X. Геологическая точность координат и независимость каталогов этим не доказаны.

```powershell
jerqazyna country-pu configs/tasks/porphyry_cu_balkhash.yaml --baseline outputs/porphyry_cu_balkhash/porphyry_cu_baseline/ce67b6ff52175829 --protocol configs/experiments/porphyry_cu_country_pu.yaml --split data/features/porphyry_cu_balkhash/splits/4c6107e6c0a9f029
```

Сравниваются фон по площади и равные квоты стран, по две либо пять фоновых подвыборок.
Суммарно в каждом ансамбле 1000 деревьев RF. Непомеченные ячейки не становятся
подтверждёнными отрицательными; доля скрытых положительных и смещение изученности
не корректируются. Все варианты используют общую train-популяцию и закрытый тест.

## Проверка меток и научный пилот EMIT

[Состояние этапа и воспроизведение](docs/LABELS_EMIT_2026-10-20.md): реестр 27 решений,
четыре производные страны в отдельном поколении; 20 неизвестных статусов и три спорных
случая оставлены на проверке. Новая таблица не подключена к обучению.

Научный пилот завершён: 7 сцен, 21 файл MIN/MINUNCERT/MASK, 1.0717 ГиБ с учётом повторов
при лимите 10 ГиБ. Пригодное покрытие — 84.47% (Центральный Казахстан) и 40.01%
(Западный Тянь-Шань). Карты QA, no-match, число дат, неопределённость и независимая проверка
доступны в отчёте этапа. Новых обучений нет; прогностический вклад EMIT пока не установлен.

[Проверка представительности 21.10.2026](docs/EMIT_COVERAGE_2026-10-21.md): проверены все
72 train-метки и 43 рамки расширения. У единственной метки в первом пилоте пригодная
доля нулевая из-за выработок. Новый набор не проходит минимум поддержки пяти fold;
загрузка отложена, остаток бюджета 8.9283 ГиБ сохранён.

```powershell
.venv-emit/Scripts/python.exe -m jerqazyna.cli emit coverage configs/pilots/emit_coverage_2026-10-21.yaml
.venv-emit/Scripts/python.exe -m jerqazyna.cli emit plan configs/pilots/emit_coverage_2026-10-21.yaml
```

`coverage` сохраняет измеренное покрытие меток и известные ограничения, `plan` выпускает
геометрический план и оценку размера файлов без научных загрузок или обучения.

```powershell
jerqazyna labels review configs/tasks/porphyry_cu_balkhash.yaml --baseline outputs/porphyry_cu_balkhash/porphyry_cu_baseline/ce67b6ff52175829 --decisions configs/curation/labels_review_2026-10-20.yaml
```

Для EMIT создана отдельная `.venv-emit` на Python 3.12 с
[зафиксированными зависимостями](requirements/py312-emit.txt). Baseline использует `.venv`
на Python 3.10. Поиск публичный; научные файлы требуют локального Earthdata Login.

```powershell
.venv-emit/Scripts/python.exe -m jerqazyna.cli emit search configs/pilots/emit_train_2026-10-20.yaml
.venv-emit/Scripts/python.exe -m jerqazyna.cli emit fetch configs/pilots/emit_train_2026-10-20.yaml
.venv-emit/Scripts/python.exe -m jerqazyna.cli emit prepare configs/pilots/emit_train_2026-10-20.yaml
.venv-emit/Scripts/python.exe -m jerqazyna.cli emit report configs/pilots/emit_train_2026-10-20.yaml
```

Общий журнал `data/pilots/emit_transfer_ledger.json` ограничивает передачу 10 ГиБ с учётом
повторов. Без авторизации `fetch` возвращает код 2 (`awaiting_auth`); до загрузки MIN,
MINUNCERT и MASK отчёт содержит только геометрическое покрытие. Наблюдения и QA не
подменяются контурами. Для одного выбранного набора сцен команды возобновления —
`fetch`, `prepare`, `report`; новый `search` нужен при намеренном обновлении каталога.

Если авторизация выполнена только в браузере, можно передать Bearer Token через
локальную форму в боковой панели, без повторного ввода пароля и без секрета в чате:

```powershell
.venv-emit/Scripts/python.exe -u -m jerqazyna.data.emit_login configs/pilots/emit_train_2026-10-20.yaml
```

Создание токена на сайте NASA и его ввод в локальное поле выполняет пользователь.
Токен используется только текущим процессом загрузки; затем автоматически выполняются
`prepare` и `report`. Подробности — в отчёте этапа.

Аудит сохранённых Sentinel-2 и ASTER на train, без обучения и загрузок:

```powershell
.venv-emit/Scripts/python.exe -m jerqazyna.cli spectral-audit configs/pilots/spectral_audit_2026-10-22.yaml
```

Sentinel-2 проходит предварительный минимум поддержки fold, ASTER — нет. Перед
обучением требуется воспроизвести происхождение старых растров и маски, отсутствующих
в действующем реестре кэша. [Результаты, карты и следующий этап](docs/SPECTRAL_AUDIT_2026-10-22.md).

## Инфраструктура

Данные и результаты проекта хранятся в `H:\projects\JerQazynaAI`, кэши и временные файлы —
в `H:\caches\JerQazynaAI`. Перед созданием окружений или установкой пакетов в PowerShell:

```powershell
. .\scripts\use-h-storage.ps1
```

Существующие `.venv` и `.venv-emit` автоматически применяют эти пути при запуске Python.
Разовая миграция кэшей проверяет SHA-256 до удаления старой копии. Программы используют
прямые пути на H:; junction между C: и H: в этом окружении не поддерживаются.
Журнал переноса — `data/storage/`.

[Пути, объём переноса и проверки](docs/STORAGE_2026-10-21.md).

Инфраструктура исследовательской фазы (PostGIS, MinIO, MLflow):

```bash
cp .env.example .env          # задайте пароли
docker compose up -d
```

FastAPI, очередь задач, TiTiler и веб-интерфейс строятся после сбора данных.

"""Командная строка.

  jerqazyna validate-config <путь к YAML>
  jerqazyna data list
  jerqazyna data status
  jerqazyna data fetch [id ...] [--all] [--priority] [--dry-run]
  jerqazyna prepare <конфиг задачи> [--layers a,b] [--force]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from jerqazyna.config import load_task


def _validate_config(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.path)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"OK: {task.task_id}")
    print(f"  признаки: {', '.join(layer.name for layer in task.feature_layers)}")
    print(f"  эталоны:  {', '.join(layer.name for layer in task.benchmark_layers) or '—'}")
    print(f"  слепой регион: {task.validation.blind_region.name}")
    if task.validation.blind_region.provisional:
        print("  предварительный регион: результат не является итоговым слепым тестом")
    if args.check_inputs:
        try:
            from jerqazyna.features.readiness import check_inputs

            issues = check_inputs(task, Path(args.root))
        except ImportError:
            print('Для проверки входов установите pip install -e ".[geo]"', file=sys.stderr)
            return 1
        for issue in issues:
            print(f"  ОШИБКА: {issue}", file=sys.stderr)
        if issues:
            return 1
        print("  входные файлы доступны; граница региона корректна")
    return 0


def _data(args: argparse.Namespace) -> int:
    from jerqazyna.data.catalog import load_catalog
    from jerqazyna.data.download import Manifest
    from jerqazyna.data.fetch import dir_size_mb, fetch_source, present_files, source_dir

    catalog = load_catalog(args.catalog)
    data_root = Path(args.data_root)

    if args.data_command == "list":
        for s in catalog.sources:
            size = f"~{s.size_mb:,.0f} МБ" if s.size_mb else ""
            print(f"{s.id:22} {s.access:13} {size:>11}  {s.title}")
        return 0

    if args.data_command == "status":
        for s in catalog.sources:
            files = present_files(data_root, s)
            mark = "есть" if files else ("нет " if s.automatic else "ждёт")
            print(f"[{mark}] {s.id:22} файлов: {len(files):5}  "
                  f"{dir_size_mb(source_dir(data_root, s)):9,.1f} МБ  ({s.access})")
        return 0

    # fetch
    if args.all:
        sources = [s for s in catalog.sources if s.access != "stac" or args.with_stac]
    elif args.ids:
        sources = [catalog.get(i) for i in args.ids]
    else:
        print("Укажите id источников или --all", file=sys.stderr)
        return 2

    manifest = Manifest(data_root / "manifest.csv", data_root)
    pending: list[str] = []
    failed = False
    for s in sources:
        print(f"== {s.id} ({s.access})")
        try:
            r = fetch_source(
                s, catalog, data_root, manifest, args.dry_run, args.priority, args.workers
            )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
            # Сбой одного источника (сеть, битый архив) не должен останавливать остальные.
            print(f"  ОШИБКА: {e}", file=sys.stderr)
            failed = True
            continue
        if r.planned:
            print(f"  план: {len(r.planned)} файлов")
            for item in r.planned[:5]:
                print(f"    {item}")
            if len(r.planned) > 5:
                print(f"    ... ещё {len(r.planned) - 5}")
        if r.downloaded or r.skipped:
            print(f"  скачано: {len(r.downloaded)}, уже было: {len(r.skipped)}")
        if r.missing:
            print(f"  нет на сервере (пропущено): {len(r.missing)}")
        if r.instructions:
            print("  " + r.instructions)
            pending.append(s.id)
    if pending:
        print(f"\nВручную: {', '.join(pending)}. После загрузки проверьте: jerqazyna data status")
    return 1 if failed else 0


def _prepare(args: argparse.Namespace) -> int:
    from jerqazyna.features.prepare import prepare

    try:
        task = load_task(args.path)
        only = [s.strip() for s in args.layers.split(",")] if args.layers else None
        report = prepare(task, Path(args.root), only=only, force=args.force)
    except (OSError, ValueError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 1
    print("\nИтог:")
    for r in report:
        extra = f", {r['seconds']} с" if "seconds" in r else ""
        print(f"  {r['layer']:22} {r['status']:13} каналов: {r['bands']}{extra}")
    return int(any(r["status"] == "нет данных" and r.get("required", True) for r in report))


def _prepare_metadata(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.path)
        if args.command == "prepare-labels":
            from jerqazyna.features.prepare import prepare_labels

            table = prepare_labels(task, Path(args.root))
            print(f"Метки готовы: {len(table)}, в области: {int(table.in_area.sum())}")
        else:
            from jerqazyna.validation.holdout import prepare_split

            dest = prepare_split(task, Path(args.root))
            print(f"Разбиение готово: {dest}")
            if task.validation.blind_region.provisional:
                print("Предварительный регион: не итоговая геологическая проверка")
        return 0
    except (OSError, ValueError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 1


def _model_command(args: argparse.Namespace) -> int:
    import json

    try:
        if args.command == "train":
            from jerqazyna.models.config import load_experiment
            from jerqazyna.models.pipeline import run_experiment
            from jerqazyna.validation.holdout import prepare_split

            task = load_task(args.path)
            experiment = load_experiment(args.experiment)
            root = Path(args.root)
            split = Path(args.split) if args.split else prepare_split(task, root)
            result = run_experiment(task, experiment, root, split)
            print(f"Результат обучения: {result}")
        elif args.command == "predict":
            from jerqazyna.models.pipeline import predict_run

            predict_run(Path(args.run), Path(args.root), Path(args.output))
        else:
            from jerqazyna.features.cache import Digests

            run = Path(args.run)
            metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
            if Digests().file(run / "cv.json") != metadata["files"]["cv.json"]:
                raise ValueError("Повреждён отчёт CV")
            report = json.loads((run / "cv.json").read_text(encoding="utf-8"))
            print(f"Выбрана модель: {report['selected_classifier']}; "
                  f"объектов в CV: {report['n_evaluated_objects']}")
            for name, result in [*report["models"].items(), ("heuristic", report["heuristic"])]:
                for metric in result["metrics"]:
                    print(f"  {name:14} площадь {metric['area_fraction']:.0%}: "
                          f"захват {metric['capture']:.1%}, "
                          f"exact [{metric['lower']:.1%}; {metric['upper']:.1%}]")
            print("Только пространственная CV на train; тест Прибалхашья не оценивался")
        return 0
    except ImportError as e:
        print(f'Установите зависимости: pip install -e ".[geo,baseline]" ({e})', file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 1


def _sensitivity(args: argparse.Namespace) -> int:
    try:
        if args.command == "country-pu":
            from jerqazyna.validation.country_pu import load_country_pu, run_country_pu

            result = run_country_pu(load_task(args.path), load_country_pu(args.protocol),
                                    Path(args.root), Path(args.baseline), Path(args.split))
        else:
            from jerqazyna.validation.sensitivity import load_sensitivity, run_sensitivity

            result = run_sensitivity(load_task(args.path), load_sensitivity(args.protocol),
                                     Path(args.root), Path(args.baseline), Path(args.split))
        print(f"Результат серии CV: {result}")
        return 0
    except ImportError as e:
        print(f'Установите зависимости: pip install -e ".[geo,baseline]" ({e})', file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows по умолчанию не в UTF-8, и кириллица в выводе ломается.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="jerqazyna")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("validate-config", help="проверить конфиг задачи")
    check.add_argument("path")
    check.add_argument("--check-inputs", action="store_true", help="проверить входные файлы и регион")
    check.add_argument("--root", default=".", help="корень проекта")

    data = sub.add_parser("data", help="сбор открытых данных")
    data.add_argument("--catalog", default="configs/sources.yaml")
    data.add_argument("--data-root", default="data")
    dsub = data.add_subparsers(dest="data_command", required=True)
    dsub.add_parser("list", help="список источников")
    dsub.add_parser("status", help="что уже собрано")
    fetch = dsub.add_parser("fetch", help="скачать источники")
    fetch.add_argument("ids", nargs="*")
    fetch.add_argument("--all", action="store_true", help="все автоматические источники")
    fetch.add_argument("--with-stac", action="store_true", help="включить Sentinel-2 в --all")
    fetch.add_argument("--priority", action="store_true", help="только охват слепого региона")
    fetch.add_argument("--dry-run", action="store_true", help="показать, что будет скачано")
    fetch.add_argument("--workers", type=int, default=1, help="параллельных тайлов Sentinel-2")

    prep = sub.add_parser("prepare", help="перенести слои на сетку задачи")
    prep.add_argument("path", help="конфиг задачи")
    prep.add_argument("--layers", help="только эти слои, через запятую")
    prep.add_argument("--force", action="store_true", help="пересчитать готовые слои")
    prep.add_argument("--root", default=".", help="корень проекта")

    for command, help_text in (("prepare-labels", "обновить только сетку и метки"),
                               ("prepare-split", "сохранить маски обучения, теста и буфера")):
        cmd = sub.add_parser(command, help=help_text)
        cmd.add_argument("path", help="конфиг задачи")
        cmd.add_argument("--root", default=".", help="корень проекта")

    train = sub.add_parser("train", help="обучить baseline с пространственной CV; тест закрыт")
    train.add_argument("path", help="конфиг задачи")
    train.add_argument("--experiment", required=True, help="зафиксированный протокол эксперимента")
    train.add_argument("--split", help="готовое разбиение; по умолчанию prepare-split")
    train.add_argument("--root", default=".")
    evaluate = sub.add_parser("evaluate", help="показать сохранённую CV на train, без теста")
    evaluate.add_argument("run", help="папка завершённого эксперимента")
    predict = sub.add_parser("predict", help="повторить карту train из сохранённой модели")
    predict.add_argument("run")
    predict.add_argument("--output", required=True, help="выходной GeoTIFF")
    predict.add_argument("--root", default=".")

    for command, help_text in [("sensitivity", "устойчивость блоков/признаков/статусов; тест закрыт"),
                                ("country-pu", "аудит стран и фон/PU-bagging; тест закрыт")]:
        sensitivity = sub.add_parser(command, help=help_text)
        sensitivity.add_argument("path", help="конфиг задачи исходного baseline")
        sensitivity.add_argument("--baseline", required=True, help="завершённый исходный запуск")
        sensitivity.add_argument("--protocol", required=True, help="фиксированный список вариантов")
        sensitivity.add_argument("--split", required=True, help="исходное разбиение train/test")
        sensitivity.add_argument("--root", default=".")

    labels = sub.add_parser("labels", help="версионируемый реестр проверки train-меток")
    lsub = labels.add_subparsers(dest="labels_command", required=True)
    review = lsub.add_parser("review")
    review.add_argument("path", help="конфиг задачи")
    review.add_argument("--baseline", required=True)
    review.add_argument("--decisions", required=True)
    review.add_argument("--root", default=".")

    emit = sub.add_parser("emit", help="научный пилот EMIT, отдельно от обучения")
    esub = emit.add_subparsers(dest="emit_command", required=True)
    for name in ("search", "fetch", "prepare", "report", "coverage", "plan"):
        command = esub.add_parser(name)
        command.add_argument("pilot", help="зафиксированный YAML пилота")
        command.add_argument("--root", default=".")

    spectral = sub.add_parser("spectral-audit", help="аудит сохранённых train-растров без обучения")
    spectral.add_argument("protocol")
    spectral.add_argument("--root", default=".")

    args = parser.parse_args(argv)
    if args.command == "spectral-audit":
        try:
            from jerqazyna.validation.spectral_audit import audit, load_audit

            print(audit(load_audit(args.protocol), Path(args.root)))
            return 0
        except (ImportError, OSError, ValueError, KeyError) as exc:
            print(f"ОШИБКА: {exc}", file=sys.stderr)
            return 1
    if args.command in {"labels", "emit"}:
        return _curation_command(args)
    if args.command == "validate-config":
        return _validate_config(args)
    if args.command == "prepare":
        return _prepare(args)
    if args.command in {"prepare-labels", "prepare-split"}:
        return _prepare_metadata(args)
    if args.command in {"train", "evaluate", "predict"}:
        return _model_command(args)
    if args.command in {"sensitivity", "country-pu"}:
        return _sensitivity(args)
    return _data(args)


def _curation_command(args):
    try:
        if args.command == "labels":
            from jerqazyna.data.review import load_review, run_review

            result = run_review(load_task(args.path), load_review(args.decisions),
                                Path(args.root), Path(args.baseline))
        else:
            import sys

            if sys.version_info < (3, 12):
                raise ValueError("EMIT requires the separate Python 3.12 environment .venv-emit")
            from jerqazyna.data import emit

            if args.emit_command in {"coverage", "plan"}:
                from jerqazyna.data.emit_coverage import assess, load_coverage

                coverage_config = load_coverage(args.pilot)
                if args.emit_command == "coverage":
                    result = assess(coverage_config, Path(args.root))
                else:
                    from jerqazyna.data.emit_sampling import plan

                    result = plan(coverage_config, Path(args.root))
                print(result)
                return 0
            config = emit.load_pilot(args.pilot)
            if args.emit_command == "prepare":
                from jerqazyna.data.emit_prepare import prepare

                result = prepare(config, Path(args.root))
            elif args.emit_command == "fetch":
                result, state = emit.fetch(config, Path(args.root))
                print(f"EMIT: {state}")
                if state == "awaiting_auth":
                    print("Настройте Earthdata Login локально; научные файлы не загружались.")
                    print(result)
                    return 2
            else:
                result = getattr(emit, args.emit_command)(config, Path(args.root))
        print(result)
        return 0
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

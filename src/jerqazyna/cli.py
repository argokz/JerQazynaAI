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

from pydantic import ValidationError

from jerqazyna.config import load_task


def _validate_config(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.path)
    except ValidationError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"OK: {task.task_id}")
    print(f"  признаки: {', '.join(layer.name for layer in task.feature_layers)}")
    print(f"  эталоны:  {', '.join(layer.name for layer in task.benchmark_layers) or '—'}")
    print(f"  слепой регион: {task.validation.blind_region.name}")
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

    task = load_task(args.path)
    only = [s.strip() for s in args.layers.split(",")] if args.layers else None
    report = prepare(task, Path(args.root), only=only, force=args.force)
    print("\nИтог:")
    for r in report:
        extra = f", {r['seconds']} с" if "seconds" in r else ""
        print(f"  {r['layer']:22} {r['status']:13} каналов: {r['bands']}{extra}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows по умолчанию не в UTF-8, и кириллица в выводе ломается.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="jerqazyna")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("validate-config", help="проверить конфиг задачи")
    check.add_argument("path")

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

    args = parser.parse_args(argv)
    if args.command == "validate-config":
        return _validate_config(args)
    if args.command == "prepare":
        return _prepare(args)
    return _data(args)


if __name__ == "__main__":
    sys.exit(main())

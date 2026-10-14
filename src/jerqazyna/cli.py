"""Командная строка: jerqazyna validate-config <путь к YAML>."""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from jerqazyna.config import load_task


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows по умолчанию не в UTF-8, и кириллица в выводе ломается.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="jerqazyna")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate-config", help="проверить конфиг задачи")
    check.add_argument("path")
    args = parser.parse_args(argv)

    if args.command == "validate-config":
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


if __name__ == "__main__":
    sys.exit(main())

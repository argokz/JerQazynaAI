"""Скачивание файлов с докачкой, проверкой и записью в манифест."""

from __future__ import annotations

import csv
import hashlib
import http.client
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "jerqazyna-data/0.1 (+https://github.com/argokz/JerQazynaAI)"
CHUNK = 1 << 20
MANIFEST_FIELDS = ["source_id", "path", "url", "bytes", "sha256", "downloaded_at"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _remote_size(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = r.headers.get("Content-Length")
            return int(size) if size else None
    except urllib.error.URLError:
        return None


def download(
    url: str, dest: Path, timeout: float = 60, progress: bool = True, retries: int = 6
) -> bool:
    """Скачивает url в dest. Возвращает False, если файл уже есть и размер совпадает.

    Недокачанный файл лежит рядом как <имя>.part и докачивается через Range.
    404 поднимает FileNotFoundError: для тайлов это нормальный пропуск.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url, timeout)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return False

    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            _attempt(url, dest, part, expected, timeout, progress)
            break
        except (TimeoutError, ConnectionError, http.client.IncompleteRead, urllib.error.URLError) as e:
            if isinstance(e, urllib.error.HTTPError) or attempt == retries:
                raise
            if progress:
                print(f"\n  обрыв ({e}), повтор {attempt}/{retries - 1} с докачкой...")
            time.sleep(min(2**attempt, 30))
    if expected is not None and part.stat().st_size != expected:
        raise OSError(f"{dest.name}: получено {part.stat().st_size} байт, ожидалось {expected}")
    part.replace(dest)
    return True


def _attempt(
    url: str, dest: Path, part: Path, expected: int | None, timeout: float, progress: bool
) -> None:
    offset = part.stat().st_size if part.exists() else 0
    if expected is not None and offset == expected:
        return
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(url) from e
        raise
    with resp:
        if offset and resp.status != 206:
            offset = 0  # сервер не поддерживает Range: качаем заново
        total = expected or 0
        done = offset
        with open(part, "ab" if offset else "wb") as f:
            while block := resp.read(CHUNK):
                f.write(block)
                done += len(block)
                if progress and total:
                    print(f"\r  {dest.name}: {done / total:6.1%} из {total / 1e6:,.0f} МБ", end="")
    if progress and total:
        print()


def unzip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            # Защита от путей вида ../../ внутри архива
            out = (target / member.filename).resolve()
            if not out.is_relative_to(root):
                raise ValueError(f"Небезопасный путь в архиве: {member.filename}")
            if member.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


class Manifest:
    """data/manifest.csv: что скачано, откуда и с каким хешем. Коммитится в git."""

    def __init__(self, path: Path, data_root: Path):
        self.path = Path(path)
        self.data_root = Path(data_root)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.rows[row["path"]] = row

    def record(self, source_id: str, file: Path, url: str) -> None:
        rel = Path(file).resolve().relative_to(self.data_root.resolve()).as_posix()
        self.rows[rel] = {
            "source_id": source_id,
            "path": rel,
            "url": url,
            "bytes": str(Path(file).stat().st_size),
            "sha256": sha256_of(Path(file)),
            "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for key in sorted(self.rows):
                w.writerow(self.rows[key])

"""Isolated EMIT pilot: public discovery, frozen assets and bounded authenticated transfer."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import Field, model_validator

from jerqazyna.config import ConfigModel
from jerqazyna.data.review import local_path, source_sha
from jerqazyna.features.cache import Digests, fingerprint
from jerqazyna.features.cache import write_json as _atomic_write_json

ASSET = re.compile(r"^EMIT_L2[AB]_(MIN|MINUNCERT|MASK)_(\d{3})_(\d{8}T\d{6})_(\d{7})_(\d{3})\.nc$")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
LIMIT = 10 * 1024**3


def write_json(path, value):
    """Retry only an atomic local commit, never a network transfer or byte reservation.

    Windows readers/virus scanners can briefly deny replacement of a JSON file.
    The reserved counter must be committed successfully before the next socket read.
    """
    for attempt in range(12):
        try:
            _atomic_write_json(path, value)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.01 * 2**attempt, .25))


class PilotArea(ConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def valid_bbox(self):
        w, s, e, n = self.bbox
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError("Invalid WGS84 bbox")
        return self


class PilotConfig(ConfigModel):
    pilot_id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    task: str
    baseline: str
    split: str
    areas: list[PilotArea] = Field(min_length=1, max_length=2)
    start: dt.date = dt.date(2022, 8, 9)
    end: dt.date = dt.date(2026, 10, 19)
    months: list[int] = Field(default_factory=lambda: [6, 7, 8, 9])
    dates_per_area: int = Field(2, ge=1, le=2)
    scenes_per_date: int = Field(3, ge=1, le=3)
    budget_bytes: int = Field(LIMIT, gt=0, le=LIMIT)

    @model_validator(mode="after")
    def protocol(self):
        if self.start > self.end or set(self.months) - {6, 7, 8, 9} or not self.months:
            raise ValueError("Invalid summer date window")
        if len({a.name for a in self.areas}) != len(self.areas):
            raise ValueError("Duplicate pilot area")
        return self

    def recipe(self):
        return self.model_dump(mode="json")


def load_pilot(path):
    return PilotConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def pilot_home(root, config):
    return Path(root) / "data/pilots" / config.pilot_id


def anchors(root, config):
    """Validate the saved baseline, split and actual train mask; never rebuild them."""
    from jerqazyna.config import load_task
    from jerqazyna.data.review import parent_labels

    root = Path(root).resolve()
    task = load_task(local_path(root, config.task))
    _, dataset, run = parent_labels(task, root, local_path(root, config.baseline))
    split = local_path(root, config.split)
    meta = json.loads((split / "split.json").read_text(encoding="utf-8"))
    d = Digests()
    if (meta["split_id"] != dataset["spec"]["split_id"]
            or d.file(split / "split.json") != dataset["spec"]["split_manifest_sha256"]):
        raise ValueError("Pilot split differs from baseline")
    for name, sha in meta["files"].items():
        if d.file(split / name) != sha:
            raise ValueError(f"Corrupt split: {name}")
    mine = next(layer for layer in task.layers if layer.kind == "mine_mask")
    return {"baseline_run_id": run["run_id"],
            "baseline_sha256": d.file(local_path(root, config.baseline) / "run.json"),
            "baseline_files": run["files"],
            "split_sha256": d.file(split / "split.json"), "grid": meta["spec"]["grid"],
            "mine_source": mine.model_dump(), "mine_source_sha256": source_sha(root, mine.source, d),
            "mine_buffer_m": task.labels.mine_mask_buffer_m}


def area_geometry(area, crs):
    # Densified edges are necessary for a curved equal-area projection.
    import numpy as np
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform

    w, s, e, n = area.bbox
    points = [(x, s) for x in np.linspace(w, e, 101)]
    points += [(e, y) for y in np.linspace(s, n, 101)]
    points += [(x, n) for x in np.linspace(e, w, 101)]
    points += [(w, y) for y in np.linspace(n, s, 101)]
    return transform(Transformer.from_crs(4326, crs, always_xy=True).transform, Polygon(points))


def verify_areas(root, config, anchor):
    import numpy as np
    from rasterio.features import rasterize

    from jerqazyna.grid.template import GridSpec, read_layer

    grid = GridSpec(**anchor["grid"])
    split = local_path(root, config.split)
    closed = (read_layer(split / "test.tif") > 0) | (read_layer(split / "buffer.tif") > 0)
    train = read_layer(split / "train.tif") > 0
    counts = {}
    for area in config.areas:
        geometry = area_geometry(area, grid.crs)
        mask = rasterize([(geometry, 1)], out_shape=grid.shape, transform=grid.transform,
                         all_touched=True).astype(bool)
        if np.any(mask & closed) or not np.any(mask & train):
            raise ValueError(f"Area intersects closed region/buffer or has no train: {area.name}")
        counts[area.name] = int((mask & train).sum())
    return counts


def cmr_entries(short_name, area, config, session):
    params = {"short_name": short_name, "version": "001", "page_size": 2000,
              "bounding_box": ",".join(map(str, area.bbox)),
              "temporal": f"{config.start}T00:00:00Z,{config.end}T23:59:59Z"}
    entries, page = [], 1
    while True:
        response = session.get(CMR, params={**params, "page_num": page}, timeout=120)
        response.raise_for_status()
        batch = response.json()["feed"]["entry"]
        entries.extend(batch)
        hits = response.headers.get("CMR-Hits")
        if (hits is not None and len(entries) >= int(hits)) or len(batch) < 2000:
            break
        page += 1
    if len({e["id"] for e in entries}) != len(entries):
        raise ValueError("CMR pagination duplicated granules; retry discovery")
    return entries


def assets(entry):
    result = {}
    keys = set()
    for link in entry.get("links", []):
        url = link.get("href", "")
        parsed = urlparse(url)
        match = ASSET.fullmatch(Path(parsed.path).name)
        if (not match or parsed.scheme != "https" or parsed.query
                or parsed.hostname != "data.lpdaac.earthdatacloud.nasa.gov"):
            continue
        kind, *key = match.groups()
        expected_level = "L2A" if kind == "MASK" else "L2B"
        if not Path(parsed.path).name.startswith(f"EMIT_{expected_level}_{kind}_001_"):
            continue
        keys.add(tuple(key))
        item = {"kind": kind, "name": Path(parsed.path).name, "url": url, "cmr_id": entry["id"]}
        if kind in result and result[kind] != item:
            raise ValueError("Ambiguous CMR asset")
        result[kind] = item
    if len(keys) != 1:
        return None, {}
    return next(iter(keys)), result


def footprint(entry, crs):
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform, unary_union

    polygons = []
    for group in entry.get("polygons", []):
        for text in group:
            values = list(map(float, text.split()))
            polygon = Polygon(list(zip(values[1::2], values[0::2])))
            if not polygon.is_valid or polygon.is_empty:
                raise ValueError(f"Invalid CMR footprint: {entry['id']}")
            polygons.append(polygon)
    if not polygons:
        raise ValueError(f"No CMR polygon: {entry['id']}")
    return transform(Transformer.from_crs(4326, crs, always_xy=True).transform, unary_union(polygons))


def select_scenes(min_entries, rfl_entries, area, config, crs):
    from shapely.geometry import GeometryCollection, mapping
    from shapely.ops import unary_union

    masks = {}
    for entry in rfl_entries:
        key, found = assets(entry)
        if key and "MASK" in found:
            if key in masks and masks[key] != found["MASK"]:
                raise ValueError("Duplicate MASK key")
            masks[key] = found["MASK"]
    region = area_geometry(area, crs)
    dates, incomplete = {}, []
    for entry in min_entries:
        date = entry["time_start"][:10]
        if (int(date[5:7]) not in config.months
                or not config.start <= dt.date.fromisoformat(date) <= config.end):
            continue
        key, found = assets(entry)
        if not key or set(found) != {"MIN", "MINUNCERT"} or key not in masks:
            incomplete.append(entry["id"])
            continue
        clipped = footprint(entry, crs).intersection(region)
        if clipped.is_empty:
            continue
        scene = {"key": list(key), "date": date, "min_cmr_id": entry["id"],
                 "assets": [found["MIN"], found["MINUNCERT"], masks[key]],
                 "footprint_projected": mapping(clipped)}
        dates.setdefault(date, []).append((scene, clipped))
    ranked = sorted(dates, key=lambda date: (unary_union([g for _, g in dates[date]]).area, date), reverse=True)
    selected, coverage = [], []
    for date in ranked[:config.dates_per_area]:
        candidates = sorted(dates[date], key=lambda item: item[0]["key"])
        covered = GeometryCollection()
        for _ in range(config.scenes_per_date):
            if not candidates:
                break
            best = max(range(len(candidates)), key=lambda i: candidates[i][1].difference(covered).area)
            scene, geometry = candidates.pop(best)
            gain = geometry.difference(covered).area
            if gain <= 0:
                break
            covered = covered.union(geometry)
            selected.append({**scene, "area": area.name, "increment_area_m2": gain,
                             "scene_coverage_fraction": geometry.area / region.area})
        coverage.append({"area": area.name, "date": date,
                         "available_fraction": unary_union([g for _, g in dates[date]]).area / region.area,
                         "selected_fraction": covered.area / region.area})
    return selected, coverage, incomplete


def search(config, root):
    import requests

    root = Path(root).resolve()
    anchor = anchors(root, config)
    counts = verify_areas(root, config, anchor)
    scenes, coverage, incomplete, catalog = [], [], [], {}
    with requests.Session() as session:
        for area in config.areas:
            m = cmr_entries("EMITL2BMIN", area, config, session)
            r = cmr_entries("EMITL2ARFL", area, config, session)
            catalog[area.name] = {"MIN": m, "RFL": r}
            selected, cov, missing = select_scenes(m, r, area, config, anchor["grid"]["crs"])
            scenes.extend(selected)
            coverage.extend(cov)
            incomplete.extend(missing)
    spec = {"protocol": config.recipe(), "anchors": anchor, "catalog_sha256": fingerprint(catalog),
            "selection_implementation_sha256": Digests().file(Path(__file__)),
            "train_cells_all_touched": counts, "collections": ["EMITL2BMIN.001", "EMITL2ARFL.001"],
            "scenes": scenes, "coverage": coverage, "incomplete_cmr_ids": incomplete}
    search_id = fingerprint(spec)
    dest = pilot_home(root, config) / search_id[:16]
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "search.json"
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old["spec"] != spec or Digests().file(dest / "catalog.json") != old["catalog_sha256"]:
            raise ValueError("Corrupt frozen search")
    else:
        write_json(dest / "catalog.json", catalog)
        write_json(marker, {"search_id": search_id, "spec": spec,
                            "catalog_sha256": Digests().file(dest / "catalog.json")})
    write_json(pilot_home(root, config) / "latest_search.json", {"path": dest.relative_to(root).as_posix(),
                                                              "sha256": Digests().file(marker)})
    return dest


def frozen_search(root, config):
    root = Path(root).resolve()
    pointer = json.loads((pilot_home(root, config) / "latest_search.json").read_text(encoding="utf-8"))
    dest = local_path(root, pointer["path"])
    d = Digests()
    if d.file(dest / "search.json") != pointer["sha256"]:
        raise ValueError("Frozen search hash mismatch")
    saved = json.loads((dest / "search.json").read_text(encoding="utf-8"))
    if (saved["search_id"] != fingerprint(saved["spec"])
            or saved["spec"]["protocol"] != config.recipe() or anchors(root, config) != saved["spec"]["anchors"]
            or d.file(dest / "catalog.json") != saved["catalog_sha256"]):
        raise ValueError("Pilot inputs changed; run search to freeze a new selection")
    return dest, saved


def authenticated_session():
    """Never prompt inside a job or expose credentials. Login is performed locally by the user."""
    import earthaccess

    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy, persist=False)
            if auth.authenticated:
                return earthaccess.get_requests_https_session()
        except Exception:  # noqa: BLE001, S112 -- redact authentication failures
            # Auth exceptions can include account data; keep them out of manifests and stdout.
            continue
    return None


class BudgetExceeded(ValueError):
    pass


class TransferLedger:
    """Global across searches/pilots. Reserve BEFORE reads, durable even after abrupt failure.

    Charged bytes conservatively include an interrupted/short final read reservation.
    observed_bytes counts returned bytes; charged_bytes is the safety budget counter.
    """

    def __init__(self, path, limit):
        self.path, self.limit = Path(path), limit
        self.data = (json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists()
                     else {"charged_bytes": 0, "observed_bytes": 0, "transfers": []})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, self.data)

    def reserve(self, amount):
        if amount <= 0 or self.data["charged_bytes"] + amount > self.limit:
            raise BudgetExceeded("Scientific transfer budget exhausted (10 GiB maximum, retries included)")
        self.data["charged_bytes"] += amount
        self.save()

    def transfer(self, response, path, url):
        remaining = response.headers.get("Content-Length")
        expected = int(remaining) if remaining is not None else None
        if expected is not None and expected > self.limit - self.data["charged_bytes"]:
            raise BudgetExceeded("Asset cannot fit remaining transfer budget")
        item = {"url": url, "observed_bytes": 0, "state": "incomplete"}
        self.data["transfers"].append(item)
        self.save()
        with Path(path).open("wb") as stream:
            while expected is None or item["observed_bytes"] < expected:
                amount = min(1024**2, self.limit - self.data["charged_bytes"])
                if expected is not None:
                    amount = min(amount, expected - item["observed_bytes"])
                self.reserve(amount)
                chunk = response.raw.read(amount, decode_content=False)
                self.data["observed_bytes"] += len(chunk)
                item["observed_bytes"] += len(chunk)
                self.save()
                if not chunk:
                    break
                stream.write(chunk)
        if expected is not None and item["observed_bytes"] != expected:
            raise ValueError("Truncated scientific asset")
        item["state"] = "transferred"
        self.save()


@contextlib.contextmanager
def transfer_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("Transfer lock exists; check the recorded PID before removing a stale lock") from None
    try:
        os.write(handle, str(os.getpid()).encode())
        yield
    finally:
        os.close(handle)
        path.unlink()


def validate_scientific(path):
    with Path(path).open("rb") as stream:
        if stream.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError("Asset is not NetCDF4/HDF5 scientific data")
    import h5netcdf

    with h5netcdf.File(path, "r") as dataset:
        if not dataset.variables:
            raise ValueError("Empty scientific dataset")


def download_asset(session, asset, partial, ledger):
    """Handle redirects explicitly so their bodies cannot bypass the byte ledger.

    Requests rebuild_auth strips credentials on cross-domain redirects. Signed redirect
    URLs remain in memory only; the ledger always records the canonical CMR URL.
    """
    from urllib.parse import urljoin

    # requests builds Response.next even with allow_redirects=False and consumes the
    # redirect body while doing so. Suppress that internal path; handle bodies here.
    resolver = getattr(session, "resolve_redirects", None)
    if resolver is not None:
        session.resolve_redirects = lambda *args, **kwargs: iter(())
    response = None
    try:
        response = session.get(asset["url"], stream=True, timeout=(30, 120), allow_redirects=False,
                               headers={"Accept-Encoding": "identity"})
        for _ in range(10):
            code = getattr(response, "status_code", 200)
            if code not in (301, 302, 303, 307, 308):
                response.raise_for_status()
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected transfer encoding")
                ledger.transfer(response, partial, asset["url"])
                return
            target = urljoin(response.url, response.headers["Location"])
            if urlparse(target).scheme != "https":
                raise ValueError("Refusing non-HTTPS asset redirect")
            redirect_body = partial.with_suffix(".redirect.part")
            ledger.transfer(response, redirect_body, asset["url"])
            import requests

            request = session.prepare_request(requests.Request("GET", target, headers={"Accept-Encoding": "identity"}))
            session.rebuild_auth(request, response)
            response.close()
            response = session.send(request, stream=True, timeout=(30, 120), allow_redirects=False)
        raise ValueError("Too many scientific asset redirects")
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
        if resolver is not None:
            session.resolve_redirects = resolver


def fetch(config, root, session=None):
    root = Path(root).resolve()
    dest, saved = frozen_search(root, config)
    status_path = dest / "fetch.json"
    state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"files": {}}
    if not saved["spec"]["scenes"]:
        state["state"] = "no_scenes"
        write_json(status_path, state)
        return dest, "no_scenes"
    if state.get("state") == "ready":
        try:
            ready_files(dest, saved)
            for name in state["files"]:
                validate_scientific(dest / "raw" / name)
            return dest, "ready"
        except (OSError, ValueError, KeyError):
            pass  # A damaged cache needs authenticated repair, charged to the same ledger.
    session = authenticated_session() if session is None else session
    if session is None:
        state["state"] = "awaiting_auth"
        write_json(status_path, state)
        return dest, "awaiting_auth"
    if hasattr(session, "mount"):
        from requests.adapters import HTTPAdapter

        session.mount("https://", HTTPAdapter(max_retries=0))
    ledger_path = root / "data/pilots/emit_transfer_ledger.json"
    with transfer_lock(ledger_path.with_suffix(".lock")):
        ledger = TransferLedger(ledger_path, config.budget_bytes)
        raw = dest / "raw"
        raw.mkdir(exist_ok=True)
        selected = {a["name"]: a for s in saved["spec"]["scenes"] for a in s["assets"]}
        try:
            for name, asset in sorted(selected.items()):
                match = ASSET.fullmatch(name)
                if not match or match[1] != asset["kind"] or Path(urlparse(asset["url"]).path).name != name:
                    raise ValueError("Non-allowlisted asset in frozen manifest")
                target = raw / name
                entry = state["files"].get(name)
                if target.exists() and entry and Digests().file(target) == entry["sha256"]:
                    validate_scientific(target)
                    continue
                partial = target.with_suffix(".nc.part")
                download_asset(session, asset, partial, ledger)
                validate_scientific(partial)
                partial.replace(target)
                state["files"][name] = {"sha256": Digests().file(target), "bytes": target.stat().st_size}
                state["state"] = "downloading"
                write_json(status_path, state)
            state["state"] = "ready" if selected else "no_scenes"
        except Exception as exc:
            state["state"] = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "incomplete"
            # Store exception type, never an authenticated response URL or credentials.
            state["error_type"] = type(exc).__name__
            if isinstance(exc, OSError):
                state["filesystem_error"] = {"errno": exc.errno, "winerror": getattr(exc, "winerror", None),
                                              "file_name": Path(exc.filename).name if exc.filename else None}
            response = getattr(exc, "response", None)
            if response is not None:
                state["http_status"] = response.status_code
            state["ledger"] = ledger.data
            write_json(status_path, state)
            if isinstance(exc, BudgetExceeded):
                raise
            raise ValueError(f"Scientific transfer failed ({type(exc).__name__}); see fetch.json") from None
        state["ledger"] = ledger.data
        write_json(status_path, state)
    return dest, state["state"]


def ready_files(dest, saved):
    state = json.loads((dest / "fetch.json").read_text(encoding="utf-8"))
    if state["state"] != "ready":
        raise ValueError(f"Scientific files are not ready: {state['state']}")
    d = Digests()
    for scene in saved["spec"]["scenes"]:
        if {a["kind"] for a in scene["assets"]} != {"MIN", "MINUNCERT", "MASK"}:
            raise ValueError("Incomplete scientific triplet")
        for asset in scene["assets"]:
            name = asset["name"]
            if name not in state["files"] or d.file(dest / "raw" / name) != state["files"][name]["sha256"]:
                raise ValueError(f"Missing/corrupt scientific asset: {name}")
    return state


def report(config, root):
    from importlib.metadata import version

    root = Path(root).resolve()
    dest, saved = frozen_search(root, config)
    status = json.loads((dest / "fetch.json").read_text(encoding="utf-8")) if (dest / "fetch.json").exists() else {"state": "not_fetched"}
    ledger_path = Path(root) / "data/pilots/emit_transfer_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"charged_bytes": 0, "observed_bytes": 0}
    prepared = None
    validation = None
    version_rows = []
    if (dest / "latest_prepare.json").exists():
        ready = ready_files(dest, saved)
        pointer = json.loads((dest / "latest_prepare.json").read_text(encoding="utf-8"))
        path = dest / pointer["directory"] / "manifest.json"
        if Digests().file(path) != pointer["sha256"]:
            raise ValueError("Prepared manifest hash mismatch")
        prepared = json.loads(path.read_text(encoding="utf-8"))
        if prepared["spec"]["raw_files"] != ready["files"]:
            raise ValueError("Prepared raw hashes differ from completed transfer")
        for p, sha in prepared["files"].items():
            if Digests().file(path.parent / p) != sha:
                raise ValueError("Prepared output checksum mismatch")
        validation_pointer = dest / "latest_validation.json"
        current_validation = None
        if validation_pointer.exists():
            vp = json.loads(validation_pointer.read_text(encoding="utf-8"))
            vm = dest / vp["directory"] / "manifest.json"
            if Digests().file(vm) != vp["sha256"]:
                raise ValueError("Scientific validation manifest changed")
            candidate = json.loads(vm.read_text(encoding="utf-8"))
            if candidate["spec"]["prepared_manifest_sha256"] == pointer["sha256"]:
                current_validation = vm.parent
        if current_validation is None:
            from jerqazyna.data.emit_validate import validate

            current_validation = validate(config, root)
        validation_manifest = json.loads((current_validation / "manifest.json").read_text(encoding="utf-8"))
        for filename, sha in validation_manifest["files"].items():
            if Digests().file(current_validation / filename) != sha:
                raise ValueError("Scientific validation artifact changed")
        values = json.loads((current_validation / "validation.json").read_text(encoding="utf-8"))
        validation = {"passed": validation_manifest["passed"],
                      "directory": current_validation.relative_to(root).as_posix(),
                      "manifest_sha256": Digests().file(current_validation / "manifest.json"),
                      "native_pixel_checks": values["native_pixel_checks"],
                      "source_value_checks": values["source_value_checks"],
                      "cell_comparisons": len(values["cell_checks"]),
                      "date_count_checks": values["date_count_checks"],
                      "max_cell_absolute_error": values["max_cell_absolute_error"],
                      "figures": [p.relative_to(root).as_posix() for p in sorted(current_validation.glob("quality_*.png"))]}
        for scene in saved["spec"]["scenes"]:
            metadata_path = path.parent / scene["area"] / scene["date"] / "_".join(scene["key"]) / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for asset in scene["assets"]:
                attrs = metadata["root_attributes"][asset["kind"]]
                version_rows.append({"file": asset["name"], "kind": asset["kind"],
                                     "product_version": attrs.get("product_version"),
                                     "software_build_version": attrs.get("software_build_version"),
                                     "time_coverage_start": attrs.get("time_coverage_start"),
                                     **ready["files"][asset["name"]]})
    summary = {"state": "prepared" if prepared else status["state"], "search_id": saved["search_id"],
               "selected_scenes": len(saved["spec"]["scenes"]),
               "unique_assets": len({a["name"] for s in saved["spec"]["scenes"] for a in s["assets"]}),
               "geometric_coverage": saved["spec"]["coverage"], "transfer_budget_bytes": config.budget_bytes,
               "charged_bytes_including_retries": ledger["charged_bytes"], "observed_transfer_bytes": ledger["observed_bytes"],
               "science": prepared["summary"] if prepared else None, "holdout_evaluated": False,
               "collections": saved["spec"]["collections"], "input_files": version_rows,
               "validation": validation,
               "unique_scientific_file_bytes": sum(v["bytes"] for v in status.get("files", {}).values()),
               "search_manifest_sha256": Digests().file(dest / "search.json"),
               "prepared_manifest_sha256": pointer["sha256"] if prepared else None,
               "runtime": {p: version(p) for p in ("earthaccess", "h5netcdf", "h5py", "rasterio", "numpy", "xarray")},
               "limitations": ["Geometric coverage does not measure usable scientific coverage",
                               "Band depth is not mineral concentration or reserves",
                               "Fit values retained as distributed; no label-tuned thresholds",
                               "Predictive value requires a separate controlled model comparison"]}
    write_json(dest / "report.json", summary)
    lines = ["# EMIT pilot", "", f"State: **{summary['state']}**", "",
             f"Frozen selection: {summary['selected_scenes']} scenes, {summary['unique_assets']} assets.",
             f"Transfer: {ledger['observed_bytes']} observed bytes; {ledger['charged_bytes']} charged bytes / {config.budget_bytes}.",
             "", "| Area | Date | Available geometric coverage | Selected coverage |", "|---|---|---:|---:|"]
    for row in summary["geometric_coverage"]:
        lines.append(f"| {row['area']} | {row['date']} | {row['available_fraction']:.3%} | {row['selected_fraction']:.3%} |")
    lines += ["", "Scientific QA, uncertainty and no-match statistics: " + ("see prepared manifest and maps." if prepared else "**unavailable until scientific files are downloaded and prepared**."), ""]
    if prepared:
        lines += ["| Scope | Area | Date / scene | Observed AOI | Usable AOI | QA rejected AOI | Mine excluded AOI | No-match G1 | No-match G2 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for scope in ("scenes", "dates", "areas"):
            for row in prepared["summary"][scope]:
                stats = row["cell_area_equivalents"]
                fields = ["observed_fraction_of_aoi_train", "usable_fraction_of_aoi_train",
                          "aggregate_QA_fraction_of_aoi_train", "mine_mask_fraction_of_aoi_train",
                          "group_1_no_match_fraction", "group_2_no_match_fraction"]
                values = [f"{stats[k]:.3%}" if stats.get(k) is not None else "unavailable" for k in fields]
                label = row.get("date", "all dates") + (" / " + "_".join(row["key"]) if "key" in row else "")
                lines.append(f"| {scope} | {row['area']} | {label} | " + " | ".join(values) + " |")
        lines += ["", prepared["summary"]["aggregation"], "", "Uncertainty quantiles are retained per scene in report.json and metadata.json. No label-dependent thresholds applied.", ""]
        lines += [("Scientific verification: "
                  f"{validation['native_pixel_checks']} native pixels, {validation['source_value_checks']} source values, "
                  f"{validation['cell_comparisons']} cell comparisons; max absolute error {validation['max_cell_absolute_error']:.3g}."),
                  f"Validation manifest SHA-256: `{validation['manifest_sha256']}`.", "",
                  "| Scientific file | Version | Software build | Bytes | SHA-256 |", "|---|---|---|---:|---|"]
        for row in version_rows:
            lines.append(f"| {row['file']} | {row['product_version']} | {row['software_build_version']} | {row['bytes']} | `{row['sha256']}` |")
        lines += ["", f"Search manifest SHA-256: `{summary['search_manifest_sha256']}`.",
                  f"Prepared manifest SHA-256: `{summary['prepared_manifest_sha256']}`.",
                  "Runtime: " + ", ".join(f"{key}={value}" for key, value in summary["runtime"].items()), ""]
        for figure in validation["figures"]:
            lines.append(f"![QA map]({Path(root / figure).relative_to(dest).as_posix()})")
    lines.extend(f"- {s}" for s in summary["limitations"])
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest / "report.md"

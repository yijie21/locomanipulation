#!/usr/bin/env python3
"""Fetch the slice of Aria Digital Twin the scene_discovery line needs: ground truth of a few sequences, then their objects.

    python3 scripts/adt_download.py sequences Apartment_release_clean_seq131_M1292 Apartment_release_meal_seq131_M1292
    python3 scripts/adt_download.py objects            # every object named in the downloaded sequences' instances.json

The full release is ~2.3 TB (depth alone 1.3 TB); a scene needs only each sequence's `main_groundtruth` (~20 MB:
instances.json, scene_objects.csv, 3d_bounding_box.csv) and the Digital Twin Catalog models of the objects in it
(~11 MB median .glb, CC BY-SA). The signed link files come from the Project Aria site (one per user, never in git)
and live in `<root>/links/`; everything lands under `<root>` (default /workspace/datasets/adt), outside the repo:
the sequence data is licensed for non-commercial research and may not be redistributed
(docs/handoff/scene-discovery-plan-20261002.md, decision 17; the user overrode it for the scene RSI kit `rsi_kit/`,
DECISIONS 2026-10-20). Every file is checked against its sha1 and skipped
when already present, so a rerun resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/workspace/datasets/adt")
SEQUENCE_LINKS = "ADT_download_urls.json"
OBJECT_LINKS = "DTC_objects_ADT_download_urls.json"
OBJECT_FILES = {"3d-asset_glb": "3d-asset.glb", "metadata": "metadata.json", "license": "CC_BY-SA.txt"}


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry: dict, dest: Path) -> str:
    """Download one link-file entry ({download_url, sha1sum, file_size_bytes}) to `dest`; -> 'kept' | 'fetched'."""
    if dest.exists() and sha1_of(dest) == entry["sha1sum"]:
        return "kept"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(entry["download_url"], timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha1_of(part)
    if got != entry["sha1sum"]:
        part.unlink()
        raise RuntimeError(f"{dest.name}: sha1 {got} != {entry['sha1sum']}")
    part.replace(dest)
    return "fetched"


def sequences(root: Path, names: list[str]) -> None:
    links = json.loads((root / "links" / SEQUENCE_LINKS).read_text())["sequences"]
    for name in names:
        if name not in links:
            sys.exit(f"unknown sequence {name}; e.g. {sorted(links)[:3]}")
        entry = links[name]["main_groundtruth"]
        out = root / "sequences" / name
        zip_path = out / entry["filename"]
        state = fetch(entry, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
        print(f"[adt] {name}: main_groundtruth {state}, {entry['file_size_bytes'] / 1e6:.1f} MB -> {out}", flush=True)


def instance_names(root: Path) -> list[str]:
    """Object instance names across every downloaded sequence (skeletons excluded)."""
    names = set()
    for path in sorted((root / "sequences").glob("*/**/instances.json")):
        for inst in json.loads(path.read_text()).values():
            if inst.get("instance_type") == "object":
                names.add(inst["instance_name"])
    return sorted(names)


def objects(root: Path, names: list[str], workers: int) -> None:
    catalog = json.loads((root / "links" / OBJECT_LINKS).read_text())["releases"]["ADT"]["objects"]
    names = names or instance_names(root)
    missing = [n for n in names if n not in catalog]
    if missing:
        print(f"[adt] {len(missing)} instance names have no catalog model: {missing[:12]}", flush=True)
    jobs = [
        (catalog[n][key], root / "objects" / n / filename)
        for n in names
        if n in catalog
        for key, filename in OBJECT_FILES.items()
        if key in catalog[n]
    ]
    total = sum(e["file_size_bytes"] for e, _ in jobs)
    print(f"[adt] {len(jobs)} files for {len(names) - len(missing)} objects, {total / 1e9:.2f} GB", flush=True)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for (entry, dest), state in zip(jobs, pool.map(lambda job: fetch(*job), jobs)):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"[adt] {done}/{len(jobs)} ({dest.parent.name}/{dest.name} {state})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="data root; the link files are read from <root>/links")
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("sequences", help="main_groundtruth of the named sequences")
    s.add_argument("names", nargs="+")
    o = sub.add_parser("objects", help="catalog models; default = every object in the downloaded sequences")
    o.add_argument("names", nargs="*")
    o.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.what == "sequences":
        sequences(args.root, args.names)
    else:
        objects(args.root, args.names, args.workers)


if __name__ == "__main__":
    main()

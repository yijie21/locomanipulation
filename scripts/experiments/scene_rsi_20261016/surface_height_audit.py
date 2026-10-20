"""TODO 27.14: how far each support surface's `top` is from the face objects actually rest on (user 2026-10-20: the
remote sank into the TV stand in the report video).

`surfaces.horizontal_regions` puts a height level at the median z of its upward triangles counted one by one. A board
whose top is two large triangles and whose trim is hundreds of small ones gets the trim's height: the TV stand's top
board is at 0.543-0.545 m, its `top` 0.512. This re-runs the same clustering and regions on every surface host's mesh
(the shell with the counters, every fixed piece adt_to_scene considered) and reports, per stored surface, the
area-weighted median height of the level and the highest upward face with at least 10 % of the level's area.
Writes outputs/scene_rsi_20261016/surface_audit/heights.json.

    PYTHONPATH=src /venv/main/bin/python scripts/experiments/scene_rsi_20261016/surface_height_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]

import adt_to_scene as A  # noqa: E402

from locomani.scene_discovery import adt, surfaces  # noqa: E402

SCENE = REPO / "outputs/scene_discovery/adt_apartment_mars"
OUT = REPO / "outputs/scene_rsi_20261016/surface_audit"


def levels(tris, z_range=(0.10, 1.30), min_normal_z=0.97, gap=0.015) -> list[dict]:
    """The clustering of surfaces.horizontal_regions, with the per-level heights it does not keep."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1)
    z = tris[:, :, 2].mean(axis=1)
    up = (length > 0) & (n[:, 2] / np.maximum(length, 1e-12) >= min_normal_z) & (z >= z_range[0]) & (z <= z_range[1])
    z, area = z[up], 0.5 * length[up]
    order = np.argsort(z)
    out = []
    for g in np.split(order, np.flatnonzero(np.diff(z[order]) > gap) + 1):
        zs, w = z[g], area[g]
        cum = np.cumsum(w) / w.sum()
        mm = np.round(zs, 3)  # 1 mm bins
        big = [float(v) for v in np.unique(mm) if w[mm == v].sum() >= 0.1 * w.sum()]
        area_median = float(zs[np.searchsorted(cum, 0.5)])
        out.append(dict(count_median=round(float(np.median(zs)), 4), area_median=round(area_median, 4),
                        top_face=round(max(big) if big else area_median, 4), n=int(len(g)), area=round(float(w.sum()), 3)))  # fmt: skip
    return out


def main() -> int:
    meta = json.loads((SCENE / "scene_meta.json").read_text())
    objs = {o.name: o for o in adt.read_sequence(A.DATA / "sequences" / meta["source"]["sequence"])}
    by_host: dict[str, list[str]] = {}
    for k, s in meta["surfaces"].items():
        by_host.setdefault(s["of_object"], []).append(k)
    rows = []
    for host, keys in sorted(by_host.items()):
        mesh = A.world_mesh(objs[host])
        lv = levels(mesh.triangles, z_range=(0.3, 1.3) if host == adt.SHELL else (0.10, 1.30))
        for k in keys:
            top = float(meta["surfaces"][k]["top"])
            hit = min(lv, key=lambda v: abs(v["count_median"] - top))
            rows.append(dict(surface=k, name=meta["surfaces"][k].get("name", ""), top=top, **hit,
                             off_area=round(hit["area_median"] - top, 4), off_top=round(hit["top_face"] - top, 4)))  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "heights.json").write_text(json.dumps(rows, indent=1))
    bad = sorted((r for r in rows if abs(r["off_top"]) >= 0.005), key=lambda r: -abs(r["off_top"]))
    print(f"{len(rows)} surfaces; {len(bad)} with the top face 5 mm or more from `top`:")
    for r in bad:
        print(f"  {r['surface']:34s} top {r['top']:.4f}  area-median {r['area_median']:.4f}  top face {r['top_face']:.4f}  "
              f"off {100 * r['off_top']:+5.1f} cm  ({r['name'][:40]})")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

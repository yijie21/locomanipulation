#!/usr/bin/env python3
"""Move a built ADT scene's surfaces to the heights the area-weighted detector gives (user 2026-10-20, TODO 27.14).
Runs in /venv/main (trimesh), on a scene that `adt_to_scene.py mujoco` built with the old count-median heights.

    /venv/main/bin/python scripts/scene_fix_surface_heights.py --scene outputs/scene_discovery/adt_apartment_mars [--write]

Until 2026-10-20 `surfaces.horizontal_regions` took a level's height as the median over its triangles counted one by
one; trim outvoted board tops, and 17 of the apartment's 69 surfaces sat 0.5-3.7 cm under their boards (the TV
stand's top 3.3 cm), with objects resting inside the boards. Rebuilding the scene from the ADT source would also undo
what was decided on it since (doors, dropped surfaces, settled poses), so this moves the heights in place:
* every surface's `top` (scene_meta.json) and its slab (`adt_slab_<surface>`, scene_locomani.xml and env_only.xml),
  matched to its region by the old detector's height and centre, the new height from the same level by area;
* the robot collider under the lowest surface (`adt_fixcol_<piece>`'s mesh, `adt_to_scene.furniture_solid`) of every
  piece whose lowest surface moved and whose collider is that cut (the floating shelves' are whole hulls and stay);
* the free objects resting on a moved surface, lifted by the rise (scene_settle.py then puts them down);
* the panels and the wall / furniture boxes taken out, for scene_add_panels.py and scene_add_colliders.py to rebuild
  around the new heights (both refuse a scene that has them).
Without --write it prints the changes. With it, the files it touches are first copied to
`<scene>.bak_20261020_before_surface_heights/`, and `<scene>/surface_heights.json` records old and new heights.
Then: scene_add_panels.py, scene_add_colliders.py, scene_settle.py --write, scene_lint.py.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

import adt_to_scene as A  # noqa: E402

from locomani.scene_discovery import adt, mjcf, surfaces  # noqa: E402

XMLS = ("scene_locomani.xml", "env_only.xml")
BOXES = re.compile(r'\s*<geom name="adt_(?:panel|wallcol|upcol)_[^"]*"[^>]*/>')
MOVED_M = 0.0005  # a surface whose height changes less than this keeps its slab


def new_tops(meta: dict, objs: dict) -> dict[str, tuple[float, float]]:
    """{surface: (old top, new top)}: each stored surface matched to the old detector's region by height and centre,
    the new height that region's level gets by area."""
    by_host: dict[str, list[str]] = {}
    for k, s in meta["surfaces"].items():
        by_host.setdefault(s["of_object"], []).append(k)
    out = {}
    for host, keys in sorted(by_host.items()):
        tris = A.world_mesh(objs[host]).triangles
        kw = {"z_range": (0.3, 1.3)} if host == adt.SHELL else {}
        old = surfaces.horizontal_regions(tris, height="count", **kw)
        new = surfaces.horizontal_regions(tris, height="area", **kw)
        assert len(old) == len(new), host  # same levels and regions; only the heights differ
        for k in keys:
            s = meta["surfaces"][k]
            cost = [abs(r["top"] - s["top"]) + float(np.hypot(*np.subtract(r["xy"], s["xy"]))) for r in old]
            j = int(np.argmin(cost))
            if abs(old[j]["top"] - s["top"]) > 1e-3 or np.hypot(*np.subtract(old[j]["xy"], s["xy"])) > 0.02:
                raise SystemExit(f"{k}: no region of {host} matches (closest {old[j]['top']} at {old[j]['xy']})")
            out[k] = (float(s["top"]), float(new[j]["top"]))
    return out


def recut(scene: Path, meta: dict, objs: dict, lowest: dict) -> tuple[dict, list[str]]:
    """({piece: its collider cut under the new lowest surface}, [pieces kept]) for the pieces whose lowest surface
    moved. A collider is recut only where the cut at the old height (adt_to_scene.furniture_solid) gives the hull on
    disk; the floating shelves' hulls are whole (built before their surfaces were), robot-only, and stay."""
    import trimesh

    out, kept = {}, []
    for h, (old, new) in sorted(lowest.items()):
        f = scene / "mesh" / f"adt_fixcol_{h}.obj"
        if abs(new - old) < MOVED_M or not f.exists():
            continue
        o, src = meta["objects"][h], objs[h]
        raw = A.load_mesh(h)
        body = raw.copy()
        body.vertices = raw.vertices @ src.body_axes.T + np.asarray(src.body()["mesh_offset"])
        was = A.furniture_solid(body, old, float(o["init_pos"][2]))
        if was is None or abs(float(was.vertices[:, 2].max()) - float(trimesh.load(f).vertices[:, 2].max())) > 1e-3:
            kept.append(h)
            continue
        solid = A.furniture_solid(body, new, float(o["init_pos"][2]))
        if solid is None:
            raise SystemExit(f"{h}: nothing left under its lowest surface at {new}")
        out[h] = solid
    return out, kept


def set_attr_z(xml: str, tag_re: str, z: float | None = None, dz: float | None = None) -> tuple[str, int]:
    """Set (or shift) the z of the `pos` of the elements whose opening tag matches `tag_re`."""
    n = 0

    def fix(m):
        nonlocal n
        x, y, zz = (float(v) for v in m.group(2).split())
        zz = z if z is not None else zz + dz
        n += 1
        return f'{m.group(1)}pos="{x:g} {y:g} {round(zz, 6):g}"'

    xml = re.sub(rf'({tag_re}[^>]*?)pos="([^"]+)"', fix, xml)
    return xml, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    scene = a.scene.resolve()
    meta = json.loads((scene / "scene_meta.json").read_text())
    objs = {o.name: o for o in adt.read_sequence(A.DATA / "sequences" / meta["source"]["sequence"])}
    tops = new_tops(meta, objs)
    moved = {k: v for k, v in tops.items() if abs(v[1] - v[0]) >= MOVED_M}
    for k, (o, n) in sorted(moved.items(), key=lambda kv: kv[1][0] - kv[1][1]):
        print(f"  {k:32s} {o:.4f} -> {n:.4f}  ({100 * (n - o):+.1f} cm)")
    print(f"{len(moved)} of {len(tops)} surfaces move")
    lowest = {}
    for k, s in meta["surfaces"].items():
        old, new = tops[k]
        lo = lowest.setdefault(s["of_object"], [np.inf, np.inf])
        lo[0], lo[1] = min(lo[0], old), min(lo[1], new)
    hulls, kept = recut(scene, meta, objs, lowest)
    # objects on a raised surface go up with it; on a lowered one they stay, and scene_settle puts them down
    riders = {k: moved[o["surface"]][1] - moved[o["surface"]][0] for k, o in meta["objects"].items()
              if o.get("free") and o.get("surface") in moved and moved[o["surface"]][1] > moved[o["surface"]][0]}  # fmt: skip
    print("robot colliders recut:", sorted(hulls))
    print("robot colliders kept (not cut under a surface when built):", kept)
    print("objects lifted:", {k: round(100 * v, 1) for k, v in riders.items()}, "cm")
    if not a.write:
        return 0

    bak = scene.parent / f"{scene.name}.bak_20261020_before_surface_heights"
    bak.mkdir(exist_ok=False)
    for f in (*XMLS, "scene_meta.json", "settle.json", "lint.json", "panels.json", "colliders.json"):
        if (scene / f).exists():
            shutil.copy2(scene / f, bak / f)
    (bak / "mesh").mkdir()
    for h in hulls:
        shutil.copy2(scene / "mesh" / f"adt_fixcol_{h}.obj", bak / "mesh")

    for k, (_, n) in moved.items():
        meta["surfaces"][k]["top"] = round(n, 4)
    for k, dz in riders.items():
        meta["objects"][k]["init_pos"][2] = round(float(meta["objects"][k]["init_pos"][2]) + dz, 6)
    for h, solid in hulls.items():
        A.export(solid, scene / "mesh" / f"adt_fixcol_{h}.obj")
    for f in XMLS:
        xml = (scene / f).read_text()
        for k, (_, n) in moved.items():
            xml, c = set_attr_z(xml, rf'<geom name="adt_slab_{re.escape(k)}"', z=n - mjcf.SLAB / 2)
            assert c == 1 or (c == 0 and f != "scene_locomani.xml"), (f, k, c)
        for k, dz in riders.items():
            xml, c = set_attr_z(xml, rf'<body name="{re.escape(meta["objects"][k]["body"])}"', dz=dz)
            assert c == 1, (f, k, c)
        xml, c = BOXES.subn("", xml)
        print(f"{f}: {c} panel / collider boxes taken out")
        (scene / f).write_text(xml)
    (scene / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    (scene / "surface_heights.json").write_text(json.dumps(
        dict(rule="area-weighted median of the level (surfaces.horizontal_regions height='area')",
             moved={k: dict(old=o, new=round(n, 4)) for k, (o, n) in moved.items()},
             colliders_recut=sorted(hulls), colliders_kept=kept, objects_lifted={k: round(v, 4) for k, v in riders.items()}), indent=1))  # fmt: skip
    print("written; backup", bak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

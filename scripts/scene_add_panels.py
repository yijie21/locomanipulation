#!/usr/bin/env python3
"""Give a built ADT scene the furniture panels its objects rest against (2026-10-18, plan
docs/handoff/scene-rsi-plan-20261016.md §10.2; scene file format §3). Runs in /venv/main (trimesh), after
`adt_to_scene.py mujoco` and before `scene_settle.py`.

    /venv/main/bin/python scripts/scene_add_panels.py --scene outputs/scene_discovery/adt_apartment_mars \\
        [--approach-edges outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json]

The furniture colliders are the robot's only (D3) and stop under the lowest surface, so nothing held a row of books
up at a shelf's end or kept it from creeping off the back. For every surface, its furniture's exported visual
(`<scene>/mesh/adt_fix_<key>.obj`, body frame; one replaced by its hull shows none) is sampled and `mjcf.side_panels`
finds the panels it shows: the two short ends wherever the mesh has one, the long sides unless the robot works from
that edge (the scene file's `approach_edges` and, given, the stance overlay's). Each becomes a 1 cm box that collides
with free objects only (contype OBJECT_ONLY_BIT; the free objects' colliders get conaffinity 1 | OBJECT_ONLY_BIT), in
`scene_locomani.xml` and `env_only.xml` after the piece's visual geom. Refuses a scene that already has panels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

import adt_to_scene as A  # noqa: E402

from locomani.scene_discovery import adt, mjcf  # noqa: E402

DENSITY = 40000  # mesh surface samples per m^2


def panel_geoms(key: str, vis, o: dict, meta: dict, overlay: dict) -> list[tuple[str, dict, float]]:
    """(name, geom attributes, cover) of furniture `key`'s panels, from its visual `vis` (body frame)."""
    if trimesh.convex.is_convex(vis):  # a visual replaced by its hull would fill every edge
        return []
    world = vis.copy()
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = adt.quat_to_matrix(o["init_quat"]), o["init_pos"]
    world.apply_transform(t)
    n = max(2000, int(world.area * DENSITY))
    pts, _ = trimesh.sample.sample_surface(world, n, seed=0)
    out = []
    for sk, sf in sorted(meta["surfaces"].items()):
        if sf.get("of_object") != key:
            continue
        open_edges = set(sf.get("approach_edges") or ()) | set((overlay.get(sk) or {}).get("approach_edges") or ())
        for w in mjcf.side_panels(sf, np.asarray(pts), world.area / n, mjcf.ceiling(meta["surfaces"], sk), open_edges):
            attrs = dict(type="box", pos=mjcf.fmt(w["pos"]), quat=mjcf.fmt(w["quat"]), size=mjcf.fmt(w["size"]),
                         contype=str(A.OBJECT_ONLY_BIT), conaffinity="0", group="3", rgba="0.3 0.9 0.5 0.25")  # fmt: skip
            out.append((f"adt_panel_{sk}_{w['edge'].replace('+', 'p').replace('-', 'n')}", attrs, w["cover"]))
    return out


def geom_xml(name: str, attrs: dict) -> str:
    return f'<geom name="{name}" ' + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + " />"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--approach-edges", type=Path, help="the stance overlay (stance_edges.py): more edges kept open")
    a = ap.parse_args(argv)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    overlay = json.loads(a.approach_edges.read_text())["surfaces"] if a.approach_edges else {}
    panels: dict[str, list] = {}
    for key, o in sorted(meta["objects"].items()):
        vis = a.scene / "mesh" / f"adt_fix_{key}.obj"
        if (o.get("semantic") or {}).get("movable") or not vis.exists():
            continue
        g = panel_geoms(key, trimesh.load(vis, force="mesh"), o, meta, overlay)
        if g:
            panels[key] = g
    free = sorted(k for k, o in meta["objects"].items() if o.get("free"))
    bits = str(1 | A.OBJECT_ONLY_BIT)
    for name in ("scene_locomani.xml", "env_only.xml"):
        path = a.scene / name
        s = path.read_text()
        if "adt_panel_" in s:
            raise SystemExit(f"{path} already has panels")
        for key, gs in panels.items():
            m = re.search(rf'<geom name="adt_fix_{re.escape(key)}" [^>]*/>', s)
            assert m, (name, key)
            s = s[: m.end()] + "".join(geom_xml(n, at) for n, at, _ in gs) + s[m.end() :]
        n_free = 0
        for key in free:  # the free body's colliders: its hull, or a free container's boxes
            s, n = re.subn(rf'(<geom name="adt_obj_{re.escape(key)}_[^"]*" (?![^>]*conaffinity)[^>]*?)( ?/>)', rf'\1 conaffinity="{bits}"\2', s)  # fmt: skip
            assert n >= 1, (name, key)
            n_free += n
        path.write_text(s)
        print(
            f"[panels] {path}: {sum(map(len, panels.values()))} panels on {len(panels)} pieces; {n_free} free colliders"
        )
    report = {k: [dict(name=n, cover=c) for n, _, c in g] for k, g in panels.items()}
    (a.scene / "panels.json").write_text(json.dumps(report, indent=1))
    print("[panels]", {k: [n.removeprefix("adt_panel_") for n, _, _ in g] for k, g in panels.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

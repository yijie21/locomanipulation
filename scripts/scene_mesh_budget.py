#!/usr/bin/env python3
"""TODO 4.7.8: the scene build's mesh gate. MuJoCo's renderer leaves its fast path when one mesh draws more than 2^20
vertices -- 3 per face, so above 349,525 faces: a 349,526-face mesh renders 12x slower than one face fewer, and each
shadow-casting light and camera multiplies it (2026-09-24; the old `warehouse` carton, one 462,730-face mesh, held the
live render at RTF 0.35). Splitting a mesh into pieces under the limit is lossless and returns to the fast path, so the
gate is a hard number, the same in every scene.

Counts the faces of every mesh file a drawn geom uses (groups 0-2, the ones the renderer shows by default), straight
from the files (OBJ polygons count as their triangles, STL by its header or facets) without compiling the scene, so
the 1.6 GB ADT apartment is checked in seconds. Exits 1 and names each mesh over the limit.

    python3 scripts/scene_mesh_budget.py outputs/scene/<name>/scene_locomani.xml
"""

from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LIMIT_FACES = 349_525  # 3 x faces <= 2^20 drawn vertices
DRAWN_GROUPS = (0, 1, 2)


def faces(path: Path) -> int:
    """Triangles in a mesh file: OBJ (an n-gon is n - 2), binary or ASCII STL; 0 for a format it does not read."""
    suffix = path.suffix.lower()
    if suffix == ".obj":
        n = 0
        with path.open("rb") as fh:
            for line in fh:
                if line.startswith(b"f "):
                    n += max(len(line.split()) - 3, 0)
        return n
    if suffix == ".stl":
        head = path.read_bytes()[:84]
        if head[:5].lower() == b"solid" and b"facet" in path.read_bytes()[:4096]:
            return path.read_bytes().count(b"facet normal")
        return struct.unpack("<I", head[80:84])[0] if len(head) == 84 else 0
    return 0


def over_budget(xml_path: Path) -> list[tuple[str, int, str]]:
    """(mesh name, faces, file) for each mesh over LIMIT_FACES that a geom in DRAWN_GROUPS draws."""
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    comp = root.find("compiler")
    base = xml_path.parent
    meshdir = base / ((comp.get("meshdir") or comp.get("assetdir") or "") if comp is not None else "")
    files = {}
    for m in root.iter("mesh"):
        if m.get("file"):
            name = m.get("name") or Path(m.get("file")).stem
            f = Path(m.get("file"))
            files[name] = f if f.is_absolute() else meshdir / f
    drawn = set()
    for g in root.iter("geom"):
        if g.get("type") == "mesh" or g.get("mesh"):
            if int(g.get("group") or 0) in DRAWN_GROUPS and g.get("mesh"):
                drawn.add(g.get("mesh"))
    out = []
    for name in sorted(drawn):
        f = files.get(name)
        if f is not None and f.exists() and (n := faces(f)) > LIMIT_FACES:
            out.append((name, n, str(f)))
    return out


def main(argv=None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__)
        return 2
    bad = 0
    for p in paths:
        for name, n, f in over_budget(p):
            bad += 1
            print(f"{p}: mesh {name} has {n} faces > {LIMIT_FACES} ({f}); split it into pieces under the limit "
                  "(lossless; decimation is not the fix)", file=sys.stderr)  # fmt: skip
    if not bad:
        print(f"mesh budget: every drawn mesh in {len(paths)} scene(s) is at most {LIMIT_FACES} faces")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Drop the surfaces another surface covers (`surfaces.shadowed`) from an already built ADT scene, as a rebuild's
geometry step now does (user 2026-10-18, plan docs/handoff/scene-rsi-plan-20261016.md §11).

    python3 scripts/scene_drop_shadowed.py --scene outputs/scene_discovery/adt_apartment_mars \\
        [--approach-edges outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json]

The ADT kitchen's short and medium counters are cabinet tops 3.6 cm under the long counter top, which covers them
whole: an object put there rests on the long counter, and on(X, medium counter) held at the start of 3 of 5 starts of
a round-2 task. Removes each such surface from `scene_meta.json` (its row, its key in every object's placement list;
an object resting on it is moved to the surface over it), its slab and panels from `scene_locomani.xml` and
`env_only.xml`, and its row from the stance overlay. Writes `<scene>/shadowed.json`. Refuses a scene with none.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomani.scene_discovery import surfaces  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--approach-edges", type=Path, help="the stance overlay to drop the same rows from")
    a = ap.parse_args(argv)
    path = a.scene / "scene_meta.json"
    meta = json.loads(path.read_text())
    gone = surfaces.shadowed(meta["surfaces"])
    if not gone:
        raise SystemExit(f"{path}: no surface lies under another")
    moved = {}
    for key, o in meta["objects"].items():
        if o.get("surface") in gone:
            moved[key] = (o["surface"], gone[o["surface"]])
            o["surface"] = gone[o["surface"]]
    for row in (meta.get("randomization") or {}).get("placements", {}).values():
        row[:] = [s for s in row if s not in gone]
    for key in gone:
        del meta["surfaces"][key]
    left = [k for k in gone if k in json.dumps(meta)]
    assert not left, f"still named in scene_meta.json: {left}"
    path.write_text(json.dumps(meta, indent=1))
    for name in ("scene_locomani.xml", "env_only.xml"):
        p = a.scene / name
        s = p.read_text()
        for key in gone:
            s, n = re.subn(rf'<geom name="adt_(slab|panel)_{re.escape(key)}(_[pn][xy])?" [^>]*/>', "", s)
            assert n >= 1, (name, key)
        p.write_text(s)
    if a.approach_edges:
        ov = json.loads(a.approach_edges.read_text())
        for key in gone:
            ov["surfaces"].pop(key, None)
        for k, v in ov.items():  # its summary lists ("gained", ...)
            if isinstance(v, list):
                ov[k] = [x for x in v if not (isinstance(x, str) and x in gone)]
        a.approach_edges.write_text(json.dumps(ov, indent=1))
    report = dict(dropped=gone, moved_objects=moved, overlay=str(a.approach_edges) if a.approach_edges else None)
    (a.scene / "shadowed.json").write_text(json.dumps(report, indent=1))
    print(f"[shadowed] dropped {gone}; objects moved to the surface over them {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

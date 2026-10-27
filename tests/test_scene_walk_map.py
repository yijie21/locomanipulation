"""TODO 42 (user 2026-10-27): every object the simulator keeps static on the floor is on the walking map. The step
stool and the bins were movable objects without a free joint -- a static hull in the scene xml -- and the map left them
out, so the walk planner routed through them and starts spawned on the stool (6 of the incumbent's 7 walk falls).
`scripts/adt_to_scene.py walkmap` puts them on the map; this checks the scene files the scene line runs on."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCENES = ("adt_apartment_mars", "adt_apartment_mars_movable")
FLOOR_REST_M = 0.10  # adt_to_scene.FLOOR_REST_M
BAND_LO_M = 0.05  # adt_to_scene.OBSTACLE_Z[0]


def _inside(poly, xy) -> bool:
    x, y = xy
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@pytest.mark.parametrize("scene", SCENES)
def test_static_floor_objects_are_on_the_walking_map(scene):
    path = REPO / "outputs" / "scene_discovery" / scene / "scene_meta.json"
    if not path.exists():
        pytest.skip(f"{path} is not here (outputs/ is data, not in git)")
    meta = json.loads(path.read_text())
    obstacles = [[tuple(p[:2]) for p in poly] for poly in meta["room"]["obstacles"]]
    missing = []
    for key, o in meta["objects"].items():
        sem = o.get("semantic") or {}
        if o.get("free") or not sem.get("movable") or (sem.get("adt") or {}).get("category") == "door":
            continue
        lo, hi = o["init_pos"][2] - o["height"] / 2, o["init_pos"][2] + o["height"] / 2
        if lo < FLOOR_REST_M and hi > BAND_LO_M and not any(_inside(p, o["init_pos"][:2]) for p in obstacles):
            missing.append(key)
    assert missing == [], f"static floor objects off the walking map: {missing} (run adt_to_scene.py walkmap)"

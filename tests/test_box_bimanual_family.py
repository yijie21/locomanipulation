"""The box_bimanual family (TODO 38.9): its file, its two placements against the carton's size, and -- when the
logistics scene is built under outputs/ -- a start of each task drawn as the pools draw them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rsi"))

import family as F  # noqa: E402

FAMILY = json.loads((ROOT / "scripts/rsi/families/box_bimanual.json").read_text())
ROWS = {json.loads(x)["id"]: json.loads(x) for x in (ROOT / FAMILY["tasks_file"]).read_text().splitlines() if x.strip()}
HALF = {"x": 0.225, "y": 0.16}  # assets/scene/logistics_room_box_v1/README.md: 0.45 x 0.32 outer


def test_the_family_names_the_logistics_scene_two_tasks_and_a_scene_note():
    assert FAMILY["scene"] == "outputs/scene_discovery/logistics_room_mars"
    assert FAMILY["tasks"] == ["box_long_side", "box_short_side"] == list(ROWS)
    assert "logistics room" in FAMILY["scene_note"]
    assert FAMILY["sandbox"]["seed"] == "scripts/rsi/seeds/atomic"
    assert FAMILY["plan"] == "scripts/rsi/plans/box_bimanual.md" and (ROOT / FAMILY["plan"]).exists()
    for row in ROWS.values():
        assert row["goal"] == ["held(nor_cal_box)"] and row["room"] == "logistics_room"


def test_each_placement_keeps_the_near_face_within_15_cm_of_the_edge():
    # depth_m is the carton centre's distance from the nearest edge (start.edge_depths); yaw_deg is the angle between
    # the carton's x axis and that edge's outward normal: 90 = the long (x) side lies along the edge
    long_ = ROWS["box_long_side"]["placement"]
    short = ROWS["box_short_side"]["placement"]
    assert long_["yaw_deg"][0] >= 85 and short["yaw_deg"][1] <= 5
    for pl, half in ((long_, HALF["y"]), (short, HALF["x"])):
        lo, hi = pl["depth_m"]
        assert lo >= half + 0.02 - 1e-9, "the footprint stays 2 cm inside the top (start.SURFACE_MARGIN_M)"
        assert hi - half <= 0.15 + 1e-9, "the near face is within 0.15 m of the edge (D13)"


@pytest.mark.skipif(
    not (ROOT / FAMILY["scene"] / "scene_meta.json").exists(), reason="the logistics scene is not built"
)
def test_a_start_of_each_task_draws_on_the_built_scene():
    from locomani.primitives.navigator import build
    from locomani.tasks.scene.start import edge_depths, sample_start
    from locomani.tasks.scene.world import BODY_RADIUS_M

    fam = F.load("box_bimanual")
    occ, nav = build(fam["meta"], body_radius=BODY_RADIUS_M)
    surf = fam["meta"]["surfaces"]["A"]
    for i, t in enumerate(fam["tasks"]):
        task = F.task(fam, t)
        st = sample_start(fam["meta"], occ, nav, list(task.objects), np.random.default_rng(i), seed=i,
                          spawn=fam["spawn"], placement=fam["rows"][t]["placement"])  # fmt: skip
        p = st.placements["nor_cal_box"][0]
        depth = min(edge_depths(surf, p[:2]).values())
        lo, hi = fam["rows"][t]["placement"]["depth_m"]
        assert lo - 1e-6 <= depth <= hi + 1e-6
        assert np.hypot(st.robot_xy[0] - p[0], st.robot_xy[1] - p[1]) >= 0.5

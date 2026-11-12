"""Two-layer knowledge library (docs/handoff/shared-knowledge-plan-20261111.md): start.* columns, mechanism
clustering, evidence tables, condition evaluation and scope, the Consolidator's checks, HITS rendering."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import records as RC  # noqa: E402

META = {
    "objects": {
        "Wolf_1": {"surface": "Shelf__1", "box": {"half_extent": [0.034, 0.065, 0.0475]}, "container": False,
                   "semantic": {"mass_kg": 0.15, "container": {"is": False}, "adt": {"instance_name": "Wolf_1"}}},
        "Mug_2": {"surface": "Table__0", "box": {"half_extent": [0.045, 0.045, 0.05]}, "container": True,
                  "semantic": {"mass_kg": 0.3, "container": {"is": True}, "adt": {"instance_name": "Mug_2"}}},
    },
    "surfaces": {"Shelf__1": {"top": 0.8405, "of_object": "Shelf"}, "Table__0": {"top": 0.72, "of_object": "Table"}},
}  # fmt: skip
START = {"task": "pick_Wolf_1", "start": {"seed": 1, "robot_xy": [0, 0], "robot_yaw": 0.0,
                                          "placements": {"Wolf_1": [[0.1, 0.2, 0.888], [1, 0, 0, 0]]}}}  # fmt: skip


def past_record(blocking_s="Shelf__1", support="Shelf__1", kind="walk", entities=None):
    ents = entities or {"Shelf__1": {"type": "surface", "of": "Shelf"}, "Lamp": {"type": "object"},
                        "Sofa__0": {"type": "surface", "of": "Sofa"}}  # fmt: skip
    return dict(record_id="r/0.0", kind=kind, roles=dict(target="Wolf_1", support=support), entities=ents,
                G_before=dict(edges=[dict(s=blocking_s, r="blocking", o="stance", v=True, depth_m=0.05)]),
                G_after=dict(edges=[]), events=[], expected=[], discrepancies=[])  # fmt: skip


class Columns(unittest.TestCase):
    def test_affordance_columns_from_scene_and_start(self):
        c = RC.affordance_columns(META, "pick_Wolf_1", START, None)
        self.assertAlmostEqual(c["start.support_top_m"], 0.8405)
        self.assertAlmostEqual(c["start.obj_height_m"], 0.095)
        self.assertAlmostEqual(c["start.obj_min_width_m"], 0.068)
        self.assertAlmostEqual(c["start.obj_mass_kg"], 0.15)
        self.assertEqual(c["start.container"], 0.0)
        self.assertAlmostEqual(c["start.grasp_z_m"], 0.888)
        self.assertAlmostEqual(c["start.over_support_m"], 0.888 - 0.8405)
        self.assertNotIn("start.side_fits_n", c)

    def test_affordance_columns_side_fits_from_schema_02(self):
        rec = past_record()
        rec["entities"]["Wolf_1"] = {"type": "object", "affordance": {"one_hand": {"side_fits_deg": [[0, 30], [150, 180]]}}}
        c = RC.affordance_columns(META, "pick_Wolf_1", START, [rec])
        self.assertEqual(c["start.side_fits_n"], 2.0)

    def test_affordance_columns_without_meta_is_empty(self):
        self.assertEqual(RC.affordance_columns(None, "pick_Wolf_1", START, None), {})

    def test_blocking_role_own_support(self):
        self.assertEqual(RC.graph_columns([past_record()])["walk.blocking.role"], "own_support")

    def test_blocking_role_other_furniture_and_free_object(self):
        self.assertEqual(RC.graph_columns([past_record(blocking_s="Sofa__0")])["walk.blocking.role"], "other_furniture")
        self.assertEqual(RC.graph_columns([past_record(blocking_s="Lamp")])["walk.blocking.role"], "free_object")

    def test_scene_meta_reads_the_family_scene(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "scripts/rsi/families").mkdir(parents=True)
            (root / "scripts/rsi/families/fam.json").write_text(json.dumps({"scene": "outputs/scene"}))
            (root / "outputs/scene").mkdir(parents=True)
            (root / "outputs/scene/scene_meta.json").write_text(json.dumps(META))
            self.assertEqual(RC.scene_meta(root, "fam")["surfaces"]["Shelf__1"]["top"], 0.8405)
            self.assertIsNone(RC.scene_meta(root, "nope"))

    def test_scene_meta_none_for_incomplete_family_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fams = root / "scripts/rsi/families"
            fams.mkdir(parents=True)
            (fams / "empty.json").write_text("{}")
            (fams / "listy.json").write_text("[]")
            (fams / "nometa.json").write_text(json.dumps({"scene": "outputs/bare"}))
            (root / "outputs/bare").mkdir(parents=True)
            for fam in ("empty", "listy", "nometa"):
                self.assertIsNone(RC.scene_meta(root, fam), fam)

    def test_row_carries_start_columns_when_meta_given(self):
        rec = dict(label="pick_Wolf_1_s1", task="pick_Wolf_1", ok=False, steps=[])
        r = RC.row(rec, START, "/x", "host", "fam", graph=[past_record()], meta=META)
        self.assertAlmostEqual(r["start.support_top_m"], 0.8405)
        self.assertEqual(r["walk.blocking.role"], "own_support")


if __name__ == "__main__":
    unittest.main()

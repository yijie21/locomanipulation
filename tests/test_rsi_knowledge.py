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

import knowledge as KN  # noqa: E402
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
        rec["entities"]["Wolf_1"] = {
            "type": "object",
            "affordance": {"one_hand": {"side_fits_deg": [[0, 30], [150, 180]]}},
        }
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


def write_loop(root: Path, ledger: list[dict], per_iter: dict[int, dict]) -> Path:
    """A synthetic loop directory: ledger.jsonl and iter_NN/{hypotheses,candidates,reviewer_context}.json."""
    loop = root / "outputs/scene_rsi_20261016/fam/loop"
    loop.mkdir(parents=True)
    (loop / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    for n, files in per_iter.items():
        d = loop / f"iter_{n:02d}"
        d.mkdir()
        for name, content in files.items():
            (d / name).write_text(json.dumps(content))
    return loop


def two_iteration_loop(root: Path) -> Path:
    ledger = [
        dict(step="probe", iter=0, results=[dict(id="H1_hang", fixed=["pick_Wolf_1_s1"], broken=[], status="verified")]),
        dict(step="manager", iter=0, decision="engineer", hypotheses=["H1_hang"], repeats=[]),
        dict(step="engineer", iter=0, candidate=1, hypothesis="H1_hang"),
        dict(step="gate", iter=0, hash="aaa", accepted=False, b=1, c=0),
        dict(step="probe", iter=1, results=[dict(id="H4_hang_again", fixed=["pick_Wolf_1_s1"], broken=["pick_Plant4_1_s3"], status="refuted"),
                                            dict(id="H5_other", fixed=[], broken=[], status="hypothesis")]),
        dict(step="manager", iter=1, decision="engineer", hypotheses=["H4_hang_again"],
             repeats=[dict(now="H4_hang_again", same_as=["00:H1_hang"])]),
        dict(step="engineer", iter=1, candidate=1, hypothesis="H4_hang_again"),
        dict(step="gate", iter=1, hash="bbb", accepted=True, b=4, c=0),
    ]  # fmt: skip
    per_iter = {
        0: {"hypotheses.json": dict(hypotheses=[dict(id="H1_hang", claim="hand hooks the slab", instead="filter stances")]),
            "candidates.json": [dict(number=1, hypothesis="H1_hang", hash="aaa", fixed=["pick_Wolf_1_s1"], broken=[], mechanism="m0")],
            "reviewer_context.json": dict(incumbent=dict(hash="inc0"))},
        1: {"hypotheses.json": dict(hypotheses=[dict(id="H4_hang_again", claim="same", instead="same"),
                                                dict(id="H5_other", claim="x", instead="y")]),
            "candidates.json": [dict(number=1, hypothesis="H4_hang_again", hash="bbb", fixed=["pick_Wolf_1_s2"], broken=["pick_Wolf_1_s1"], mechanism="m1")],
            "reviewer_context.json": dict(incumbent=dict(hash="inc1"))},
    }  # fmt: skip
    return write_loop(root, ledger, per_iter)


def jl_rows(loop: Path) -> list[dict]:
    return [json.loads(x) for x in (loop / "ledger.jsonl").read_text().splitlines()]


class Mechanisms(unittest.TestCase):
    def test_repeats_link_two_iterations_into_one_mechanism(self):
        with tempfile.TemporaryDirectory() as d:
            ms = KN.mechanisms(two_iteration_loop(Path(d)))
        slugs = {m["slug"]: m for m in ms}
        self.assertIn("H1_hang", slugs)
        self.assertNotIn("H4_hang_again", slugs)
        m = slugs["H1_hang"]
        self.assertEqual(m["ids"], [(0, "H1_hang"), (1, "H4_hang_again")])
        self.assertEqual(m["incumbent"], {0: "inc0", 1: "inc1"})

    def test_flips_merge_probe_and_drafts_last_wins(self):
        with tempfile.TemporaryDirectory() as d:
            m = {x["slug"]: x for x in KN.mechanisms(two_iteration_loop(Path(d)))}["H1_hang"]
        self.assertEqual(
            m["flips"], {"pick_Wolf_1_s1": "broken", "pick_Wolf_1_s2": "fixed", "pick_Plant4_1_s3": "broken"}
        )
        self.assertEqual([x["hash"] for x in m["drafts"]], ["aaa", "bbb"])
        self.assertEqual(m["gate"], dict(b=4, c=0, accepted=True))

    def test_mechanism_without_flips_or_gate_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            ms = KN.mechanisms(two_iteration_loop(Path(d)))
        self.assertNotIn("H5_other", {m["slug"] for m in ms})

    def test_iters_limit_cuts_later_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            m = {x["slug"]: x for x in KN.mechanisms(two_iteration_loop(Path(d)), iters={0})}["H1_hang"]
        self.assertEqual(m["ids"], [(0, "H1_hang")])
        self.assertEqual(m["gate"], dict(b=1, c=0, accepted=False))

    def test_void_gate_row_leaves_gate_none(self):
        with tempfile.TemporaryDirectory() as d:
            loop = two_iteration_loop(Path(d))
            rows = [json.loads(x) for x in (loop / "ledger.jsonl").read_text().splitlines()]
            for r in rows:
                if r["step"] == "gate":
                    r["void"] = True
            (loop / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            m = {x["slug"]: x for x in KN.mechanisms(loop)}["H1_hang"]
        self.assertIsNone(m["gate"])

    def test_same_as_outside_iters_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            ms = KN.mechanisms(two_iteration_loop(Path(d)), iters={1})
        m = {x["slug"]: x for x in ms}["H4_hang_again"]
        self.assertEqual(m["ids"], [(1, "H4_hang_again")])

    def test_probe_result_without_hypothesis_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            loop = two_iteration_loop(Path(d))
            ledger = jl_rows(loop) + [
                dict(
                    step="probe",
                    iter=1,
                    results=[dict(id="H99_ghost", fixed=["pick_Ghost_1_s9"], broken=[], status="verified")],
                )
            ]
            (loop / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
            ms = KN.mechanisms(loop)
        self.assertNotIn("H99_ghost", {m["slug"] for m in ms})
        self.assertTrue(all("pick_Ghost_1_s9" not in m["flips"] for m in ms))


def diag_row(label, task, ok, version, **cols):
    return dict(label=label, task=task, ok=int(ok), version=version, **cols)


ROWS = [
    diag_row("pick_Wolf_1_s1", "pick_Wolf_1", False, "inc0", **{"start.support_top_m": 0.84, "walk.blocking.role": "own_support"}),
    diag_row("pick_Wolf_1_s2", "pick_Wolf_1", False, "inc0", **{"start.support_top_m": 0.84, "walk.blocking.role": "own_support"}),
    diag_row("pick_Wolf_1_s9", "pick_Wolf_1", True, "inc0", **{"start.support_top_m": 0.84, "walk.blocking.role": "other_furniture"}),
    diag_row("pick_Plant4_1_s3", "pick_Plant4_1", True, "inc0", **{"start.support_top_m": 1.08, "walk.blocking.role": "own_support"}),
    diag_row("pick_Mug_2_s5", "pick_Mug_2", False, "inc0", **{"start.support_top_m": 0.72}),
    diag_row("pick_Wolf_1_s1", "pick_Wolf_1", True, "inc1", **{"start.support_top_m": 0.84}),  # another version: out
]  # fmt: skip
MECH = dict(slug="H1_hang", ids=[(0, "H1_hang")], claims=["c"], insteads=["i"], incumbent={0: "inc0"},
            flips={"pick_Wolf_1_s1": "fixed", "pick_Wolf_1_s2": "fixed", "pick_Plant4_1_s3": "broken"},
            drafts=[], gate=None)  # fmt: skip


class Evidence(unittest.TestCase):
    def test_table_keeps_the_flipped_tasks_under_the_incumbent_version(self):
        t = KN.evidence_table(MECH, ROWS)
        self.assertEqual(
            sorted(r["label"] for r in t), ["pick_Plant4_1_s3", "pick_Wolf_1_s1", "pick_Wolf_1_s2", "pick_Wolf_1_s9"]
        )
        flips = {r["label"]: r["flip"] for r in t}
        self.assertEqual(flips["pick_Wolf_1_s9"], "unchanged")
        self.assertEqual(flips["pick_Plant4_1_s3"], "broken")

    def test_rank_columns_numeric_and_categorical(self):
        ranked = {c["col"]: c for c in KN.rank_columns(KN.evidence_table(MECH, ROWS))}
        self.assertEqual(ranked["walk.blocking.role"]["kind"], "cat")
        self.assertEqual(ranked["walk.blocking.role"]["flipped"], {"own_support": 3})
        self.assertEqual(ranked["start.support_top_m"]["kind"], "num")
        self.assertEqual(len(ranked["start.support_top_m"]["flipped"]), 2)

    def test_version_of_dir(self):
        self.assertEqual(KN.version_of_dir("/x/diag/incumbent_abc"), "abc")
        self.assertEqual(KN.version_of_dir("/x/diag/candidate_abc"), "abc")
        self.assertIsNone(KN.version_of_dir("/x/pool_diag"))

    def test_scan_skips_retired_rounds_unless_asked(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "_retired_x.json").write_text(json.dumps(dict(label="x", task="t", ok=True, steps=[])))
            self.assertEqual(RC.scan([(Path(d), "fam")], "h"), [])
            self.assertEqual([r["label"] for r in RC.scan([(Path(d), "fam")], "h", retired=True)], ["x"])

    def test_diag_rows_reads_retired_rounds_with_version_and_start_columns(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "scripts/rsi/families").mkdir(parents=True)
            (root / "scripts/rsi/families/fam.json").write_text(json.dumps({"scene": "outputs/scene"}))
            (root / "outputs/scene").mkdir(parents=True)
            (root / "outputs/scene/scene_meta.json").write_text(json.dumps(META))
            fam = root / "outputs/scene_rsi_20261016/fam"
            for sub in ("diag/incumbent_inc0", "pool_diag", "pool_diag_retired"):
                (fam / sub).mkdir(parents=True)
            rnd = lambda lab: json.dumps(dict(label=lab, task="pick_Wolf_1", ok=False, steps=[], skills="pkg"))  # noqa: E731
            (fam / "diag/incumbent_inc0/pick_Wolf_1_s1.json").write_text(rnd("pick_Wolf_1_s1"))
            (fam / "diag/incumbent_inc0/_retired_pick_Wolf_1_s2.json").write_text(rnd("pick_Wolf_1_s2"))
            for lab in ("pick_Wolf_1_s1", "pick_Wolf_1_s2"):
                sub = "pool_diag" if lab.endswith("s1") else "pool_diag_retired"
                (fam / sub / f"{lab}.json").write_text(json.dumps(dict(START, label=lab)))
            rows = {r["label"]: r for r in KN.diag_rows(root, "fam")}
            self.assertEqual(sorted(rows), ["pick_Wolf_1_s1", "pick_Wolf_1_s2"])
            for r in rows.values():
                self.assertEqual(r["version"], "inc0")
                self.assertIn("start.support_top_m", r)

    def test_render_evidence_names_the_mechanism_and_the_flips(self):
        t = KN.evidence_table(MECH, ROWS)
        md = KN.render_evidence(MECH, t, KN.rank_columns(t))
        self.assertIn("H1_hang", md)
        self.assertIn("pick_Plant4_1_s3", md)
        self.assertIn("walk.blocking.role", md)


if __name__ == "__main__":
    unittest.main()

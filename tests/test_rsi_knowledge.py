"""Two-layer knowledge library (docs/handoff/shared-knowledge-plan-20261111.md): start.* columns, mechanism
clustering, evidence tables, condition evaluation and scope, the Consolidator's checks, HITS rendering."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import knowledge as KN  # noqa: E402
import records as RC  # noqa: E402

_sb = importlib.util.spec_from_file_location("rsi_sandbox", REPO / "scripts" / "rsi" / "sandbox.py")
sandbox = importlib.util.module_from_spec(_sb)
_sb.loader.exec_module(sandbox)


def _scene_loop():
    """scripts/rsi/loop.py under a private name; the modules it pulls in (batch, loop, ...) share their names with the
    scalebfm experiment's, so they are put back out of sys.modules afterwards."""
    names = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "design", "sealed", "loop")
    saved = {k: sys.modules.pop(k, None) for k in names}
    spec = importlib.util.spec_from_file_location("rsi_scene_loop_knowledge", REPO / "scripts" / "rsi" / "loop.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v
    return mod


LP = _scene_loop()


def _replay():
    spec = importlib.util.spec_from_file_location("rsi_replay_knowledge", REPO / "scripts" / "rsi" / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


META = {
    "objects": {
        "Wolf_1": {"surface": "Shelf__1", "body": "Wolf_1_object", "box": {"half_extent": [0.034, 0.065, 0.0475]}, "container": False,
                   "semantic": {"mass_kg": 0.15, "container": {"is": False}, "adt": {"instance_name": "Wolf_1"}}},
        "Mug_2": {"surface": "Table__0", "body": "Mug_2_object", "box": {"half_extent": [0.045, 0.045, 0.05]}, "container": True,
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

    def test_affordance_columns_of_a_shuffled_start_take_the_task_ids_object(self):
        """2026-11-18: a gso start places every object of the platform (shuffle_others); without a PAST record the
        target is the object the task id names after its verb, if the scene has it."""
        both = dict(START, start=dict(START["start"], placements={"Wolf_1": [[0.1, 0.2, 0.888], [1, 0, 0, 0]],
                                                                  "Mug_2": [[0.5, 0.2, 0.77], [1, 0, 0, 0]]}))  # fmt: skip
        c = RC.affordance_columns(META, "pick_Wolf_1", both, None)
        self.assertAlmostEqual(c["start.obj_height_m"], 0.095)
        self.assertAlmostEqual(c["start.grasp_z_m"], 0.888)
        self.assertEqual(RC.affordance_columns(META, "pick_Nothing_9", both, None), {})  # no such object: no guess

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
        # keyed by the incumbent the flip was measured against: iteration 0 ran on inc0, iteration 1 on inc1
        self.assertEqual(
            m["flips"],
            {"inc0:pick_Wolf_1_s1": "fixed", "inc1:pick_Wolf_1_s1": "broken", "inc1:pick_Wolf_1_s2": "fixed",
             "inc1:pick_Plant4_1_s3": "broken"},
        )  # fmt: skip
        self.assertEqual([x["hash"] for x in m["drafts"]], ["aaa", "bbb"])
        self.assertEqual(m["gate"], dict(b=4, c=0, accepted=True, via="draft"))

    def test_mechanism_without_flips_or_gate_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            ms = KN.mechanisms(two_iteration_loop(Path(d)))
        self.assertNotIn("H5_other", {m["slug"] for m in ms})

    def test_iters_limit_cuts_later_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            m = {x["slug"]: x for x in KN.mechanisms(two_iteration_loop(Path(d)), iters={0})}["H1_hang"]
        self.assertEqual(m["ids"], [(0, "H1_hang")])
        self.assertEqual(m["gate"], dict(b=1, c=0, accepted=False, via="draft"))

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
        self.assertTrue(all(not any(k.endswith(":pick_Ghost_1_s9") for k in m["flips"]) for m in ms))


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
            flips={"inc0:pick_Wolf_1_s1": "fixed", "inc0:pick_Wolf_1_s2": "fixed", "inc0:pick_Plant4_1_s3": "broken"},
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

    def test_a_flip_under_one_version_is_not_counted_under_another(self):
        """I1: a mechanism spanning incumbents inc0 and inc1; the label flipped under inc0 only."""
        mech = dict(MECH, incumbent={0: "inc0", 1: "inc1"}, flips={"inc0:pick_Wolf_1_s1": "fixed"})
        t = KN.evidence_table(mech, ROWS)
        by = {(r["version"], r["label"]): r["flip"] for r in t}
        self.assertEqual(by[("inc0", "pick_Wolf_1_s1")], "fixed")
        self.assertEqual(by[("inc1", "pick_Wolf_1_s1")], "unchanged")
        self.assertEqual(sum(f != "unchanged" for f in by.values()), 1)
        bo = KN.by_object(t, mech["flips"])
        self.assertEqual(bo["Wolf_1"]["fixed"], 1)

    def test_rank_columns_numeric_and_categorical(self):
        ranked = {c["col"]: c for c in KN.rank_columns(KN.evidence_table(MECH, ROWS))}
        self.assertEqual(ranked["walk.blocking.role"]["kind"], "cat")
        self.assertEqual(ranked["walk.blocking.role"]["flipped"], {"own_support": 3})
        self.assertEqual(ranked["start.support_top_m"]["kind"], "num")
        self.assertEqual(len(ranked["start.support_top_m"]["flipped"]), 2)

    def test_zero_iqr_numeric_column_does_not_outrank_a_separating_categorical(self):
        t = [dict(label=f"u{i}", flip="unchanged", **{"a.num": 1.0, "b.cat": "x"}) for i in range(8)]
        t += [dict(label="f1", flip="fixed", **{"a.num": 5.0, "b.cat": "y"})]
        ranked = {c["col"]: c for c in KN.rank_columns(t)}
        # with zero IQR any difference of medians separates fully: exactly 1.0, never above (was ~1e5 before)
        self.assertLessEqual(ranked["a.num"]["score"], 1.0)
        self.assertEqual(ranked["b.cat"]["score"], 1.0)
        self.assertLessEqual(ranked["a.num"]["score"], ranked["b.cat"]["score"])

    def test_no_unchanged_rounds_scores_zero(self):
        t = [dict(label=f"f{i}", flip="fixed", **{"a.num": float(i), "b.cat": "y"}) for i in range(3)]
        self.assertEqual([c["score"] for c in KN.rank_columns(t)], [0.0, 0.0])

    def test_rank_columns_sorted_by_score_descending(self):
        t = [
            dict(label="f1", flip="fixed", **{"a.cat": "y", "b.cat": "y", "c.num": 9.0}),
            dict(label="f2", flip="fixed", **{"a.cat": "y", "b.cat": "x", "c.num": 9.0}),
            dict(label="u1", flip="unchanged", **{"a.cat": "x", "b.cat": "x", "c.num": 1.0}),
            dict(label="u2", flip="unchanged", **{"a.cat": "x", "b.cat": "x", "c.num": 2.0}),
        ]  # fmt: skip
        ranked = KN.rank_columns(t)
        scores = [c["score"] for c in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([c["col"] for c in ranked], ["a.cat", "c.num", "b.cat"])

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


COND = {"all": [{"feature": "start.support_top_m", "op": ">=", "value": 0.8},
                {"feature": "walk.blocking.role", "op": "==", "value": "own_support"}]}  # fmt: skip


class Conditions(unittest.TestCase):
    def test_holds_all_and_missing_column_is_false(self):
        self.assertTrue(KN.holds(COND, ROWS[0]))
        self.assertFalse(KN.holds(COND, ROWS[2]))  # other_furniture
        self.assertFalse(KN.holds(COND, ROWS[4]))  # no blocking.role column

    def test_holds_any_and_narrowed_by(self):
        c = dict(COND, any=[{"feature": "start.support_top_m", "op": "<", "value": 0.9}])
        self.assertTrue(KN.holds(c, ROWS[0]))
        self.assertFalse(KN.holds(c, ROWS[3]))  # 1.08
        c2 = dict(COND, narrowed_by=[{"feature": "start.support_top_m", "op": "<=", "value": 0.9}])
        self.assertFalse(KN.holds(c2, ROWS[3]))

    def test_type_mismatch_is_false_not_an_error(self):
        c = {"all": [{"feature": "walk.blocking.role", "op": ">", "value": 1}]}
        self.assertFalse(KN.holds(c, ROWS[0]))

    def test_separation_counts_and_p(self):
        s = KN.separation(COND, KN.evidence_table(MECH, ROWS))
        self.assertEqual((s["n_in"], s["flipped_in"], s["n_out"], s["flipped_out"]), (3, 3, 1, 0))
        self.assertLessEqual(s["p"], 1.0)


class Scope(unittest.TestCase):
    def test_object_of(self):
        self.assertEqual(KN.object_of("pick_Wolf_1"), "Wolf_1")
        self.assertEqual(KN.object_of("place_Mug_2_on_Table__0"), "Mug_2_on_Table__0")

    def test_shared_when_two_objects_net_non_negative(self):
        bo = {"Wolf_1": dict(triggered=3, fixed=3, broken=0), "Vase_2": dict(triggered=2, fixed=0, broken=0)}
        self.assertEqual(KN.scope_of(bo)[0], "shared")

    def test_local_when_one_object(self):
        self.assertEqual(KN.scope_of({"Wolf_1": dict(triggered=3, fixed=2, broken=0)})[0], "local")

    def test_drop_when_too_few_flips(self):
        self.assertEqual(KN.scope_of({"Wolf_1": dict(triggered=3, fixed=1, broken=0)})[0], "drop")

    def test_drop_when_every_object_is_net_negative(self):
        self.assertEqual(KN.scope_of({"Wolf_1": dict(triggered=3, fixed=0, broken=2)})[0], "drop")

    def test_negative_object_makes_it_local_with_why(self):
        bo = {"Wolf_1": dict(triggered=3, fixed=3, broken=0), "Plant4_1": dict(triggered=2, fixed=0, broken=1)}
        scope, why = KN.scope_of(bo)
        self.assertEqual(scope, "local")
        self.assertIn("Plant4_1", why)

    def test_by_object_from_triggered_rows(self):
        t = KN.evidence_table(MECH, ROWS)
        bo = KN.by_object(KN.trigger(COND, t), MECH["flips"])
        self.assertEqual(bo["Wolf_1"], dict(triggered=2, fixed=2, broken=0))
        self.assertEqual(bo["Plant4_1"], dict(triggered=1, fixed=0, broken=1))

    def test_narrow_adds_a_start_range_that_drops_the_negative_object(self):
        t = KN.evidence_table(MECH, ROWS)
        c = KN.narrow(COND, t, MECH["flips"])
        self.assertTrue(c.get("narrowed_by"))
        bo = KN.by_object(KN.trigger(c, t), MECH["flips"])
        self.assertNotIn("Plant4_1", bo)
        self.assertEqual(KN.scope_of(bo)[0], "local")  # one positive object remains

    def test_narrow_returns_the_condition_unchanged_when_nothing_separates(self):
        rows = [dict(r, **{"start.support_top_m": 0.84}) for r in ROWS]
        c = KN.narrow(COND, KN.evidence_table(MECH, rows), MECH["flips"])
        self.assertNotIn("narrowed_by", c)


SO = dict(
    condition={"all": [{"feature": "start.support_top_m", "op": ">=", "value": 0.8},
                       {"feature": "walk.blocking.role", "op": "==", "value": "own_support"}]},
    signature=dict(first_bad_kind="reach", first_bad_edge="at", failure_class=["side_reach"],
                   reads_as="the pre-reach stalls while the wrist presses the slab's front edge"),
    repair=dict(works="filter stances at choice time so the hanging jaw clears the own support",
                does_not=["raise D_MIN globally (broke 4)"], where="stance choice", parameters_touched=["pick.D_MIN_M (do not)"]),
)  # fmt: skip
NAMES = {"Wolf_1", "Wolf_1_object", "Wolf", "Plant4_1", "Plant4_1_object", "Plant4", "Mug_2", "Mug_2_object", "Mug"}


class Consolidate(unittest.TestCase):
    def test_names_from_meta(self):
        names = KN.names_from_meta(META)
        self.assertIn("Wolf_1", names)
        self.assertIn("Wolf_1_object", names)
        self.assertIn("Wolf", names)
        self.assertIn("Mug", names)

    def test_validate_accepts_a_separating_condition_and_scopes_it(self):
        t = KN.evidence_table(MECH, ROWS)
        entry, problems = KN.validate(SO, t, MECH["flips"], NAMES)
        self.assertEqual(problems, [])
        self.assertIn(entry["scope"], ("shared", "local"))
        self.assertIn("Wolf_1", entry["evidence"]["by_object"])
        self.assertIn("separation", entry["evidence"])

    def test_validate_rejects_unknown_column_and_bad_op(self):
        so = dict(SO, condition={"all": [{"feature": "nope.x", "op": "~", "value": 1}]})
        entry, problems = KN.validate(so, KN.evidence_table(MECH, ROWS), MECH["flips"], NAMES)
        self.assertIsNone(entry)
        self.assertTrue(any("nope.x" in p for p in problems))
        self.assertTrue(any("~" in p for p in problems))

    def test_validate_rejects_columns_outside_the_spec(self):
        """I5 (spec section 3.2): a skill-given step column never exists in another family."""
        t = [
            dict(r, **{"pregrasp.jaw_err_cm": 1.0, "reach.at.err_cm": 2.0, "start.obj_mass_kg": 0.15})
            for r in KN.evidence_table(MECH, ROWS)
        ]
        so = dict(SO, condition={"all": [{"feature": "pregrasp.jaw_err_cm", "op": ">=", "value": 0.5}]})
        entry, problems = KN.validate(so, t, MECH["flips"], NAMES)
        self.assertIsNone(entry)
        self.assertTrue(any("pregrasp.jaw_err_cm" in p and "3.2" in p for p in problems), problems)
        for col in ("start.obj_mass_kg", "reach.at.err_cm"):
            so = dict(SO, condition={"all": [*SO["condition"]["all"], {"feature": col, "op": ">=", "value": 0.0}]})
            self.assertEqual(KN.validate(so, t, MECH["flips"], NAMES)[1], [], col)

    def test_condition_columns_cover_the_generators_and_no_step_columns(self):
        allowed = RC.CONDITION_COLUMNS
        for c in ("start.obj_mass_kg", "start.spawn_dist_m", "reach.at.err_cm", "walk.blocking.role",
                  "grip.moved_before_grip", "first_bad.kind", "first_bad.edge", "n_unexpected", "failure_class"):  # fmt: skip
            self.assertIn(c, allowed)
        for c in ("pregrasp.jaw_err_cm", "choose.d_m", "events.reach_err_cm", "pregrasp.ok"):
            self.assertNotIn(c, allowed)
        rec = past_record(entities={"Shelf__1": {"type": "surface", "of": "Shelf"},
                                    "Wolf_1": {"type": "object", "affordance": {"one_hand": {"side_fits_deg": [[0, 30]]}}}})  # fmt: skip
        emitted = {**RC.affordance_columns(META, "pick_Wolf_1", START, [rec]), **RC.start_features(START, "pick_Wolf_1"),
                   **RC.graph_columns([rec])}  # fmt: skip
        self.assertEqual(set(emitted) - allowed, {"start.seed"})  # the draw number is emitted but never a condition

    def test_graph_kinds_match_the_recorder(self):
        from locomani.tasks.scene import past as P

        self.assertEqual(set(RC.GRAPH_KINDS), set(P.KIND_OF_CALL.values()) | {"mixed"})

    def test_columns_txt_lists_only_condition_columns(self):
        t = [dict(r, **{"pregrasp.jaw_err_cm": 1.0}) for r in KN.evidence_table(MECH, ROWS)]
        ranked = KN.rank_columns(t, RC.CONDITION_COLUMNS)
        self.assertNotIn("pregrasp.jaw_err_cm", {c["col"] for c in ranked})
        with tempfile.TemporaryDirectory() as d:
            KN.consolidate(MECH, t, ranked, Path(d) / "c", "in a scene", stub={})
            cols = (Path(d) / "c" / "columns.txt").read_text()
        self.assertNotIn("pregrasp", cols)
        self.assertIn("walk.blocking.role", cols)

    def test_validate_rejects_object_names_in_text(self):
        so = json.loads(json.dumps(SO))
        so["repair"]["works"] = "filter stances for Wolf_1"
        entry, problems = KN.validate(so, KN.evidence_table(MECH, ROWS), MECH["flips"], NAMES)
        self.assertIsNone(entry)
        self.assertTrue(any("Wolf_1" in p for p in problems))

    def test_validate_rejects_a_condition_that_triggers_too_few_flips(self):
        so = dict(SO, condition={"all": [{"feature": "start.support_top_m", "op": ">", "value": 1.0}]})
        entry, problems = KN.validate(so, KN.evidence_table(MECH, ROWS), MECH["flips"], NAMES)
        self.assertIsNone(entry)
        self.assertTrue(any("flipped" in p for p in problems))

    def test_validate_rejects_names_inside_identifiers_and_clause_values(self):
        t = KN.evidence_table(MECH, ROWS)
        so = json.loads(json.dumps(SO))
        so["repair"]["does_not"] = ["pick_Wolf_1_s1 broke"]
        self.assertIsNone(KN.validate(so, t, MECH["flips"], NAMES)[0])
        so["repair"]["does_not"] = ["Wolfpack broke"]
        self.assertEqual(KN.validate(so, t, MECH["flips"], NAMES)[1], [])
        so = dict(SO, condition={"all": [{"feature": "walk.blocking.what", "op": "==", "value": "Wolf_1"}]})
        entry, problems = KN.validate(so, t, MECH["flips"], NAMES)
        self.assertIsNone(entry)
        self.assertIn("names an object: Wolf_1", problems)

    def test_consolidate_declines_when_the_call_fails(self):
        class Failed:
            returncode, stdout, stderr = 1, "", "boom"

        orig = KN.subprocess.run
        KN.subprocess.run = lambda *a, **k: Failed()
        try:
            with tempfile.TemporaryDirectory() as d:
                t = KN.evidence_table(MECH, ROWS)
                so, acct = KN.consolidate(MECH, t, KN.rank_columns(t), Path(d) / "c", "in a scene", problems=["x"])
                self.assertTrue((Path(d) / "c" / "problems.txt").exists())
        finally:
            KN.subprocess.run = orig
        self.assertIn("call failed", so["decline"])
        self.assertIn("boom", so["decline"])
        self.assertIsNone(acct["cost_usd"])

    def test_build_and_evidence_refuse_a_sandbox_without_the_familys_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "lib"
            for cmd in (["build", str(Path(d) / "absent"), "fam", "--out", str(out), "--stub"],
                        ["evidence", str(Path(d) / "absent"), "fam"]):  # fmt: skip
                with self.assertRaises(SystemExit) as cm:
                    KN.main(cmd)
                self.assertEqual(cm.exception.code, 2)
            self.assertFalse(out.exists())

    def test_build_with_stub_writes_entry_build_log_and_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixed = ["pick_Wolf_1_s1", "pick_Wolf_1_s2"]
            ledger = [
                dict(step="probe", iter=0, results=[dict(id="H1_hang", fixed=fixed, broken=["pick_Plant4_1_s3"], status="verified")]),
                dict(step="manager", iter=0, decision="engineer", hypotheses=["H1_hang"], repeats=[]),
                dict(step="engineer", iter=0, candidate=1, hypothesis="H1_hang"),
                dict(step="gate", iter=0, hash="aaa", accepted=False, b=2, c=1),
            ]  # fmt: skip
            per_iter = {0: {
                "hypotheses.json": dict(hypotheses=[dict(id="H1_hang", claim="hand hooks the slab", instead="filter stances")]),
                "candidates.json": [dict(number=1, hypothesis="H1_hang", hash="aaa", fixed=fixed, broken=["pick_Plant4_1_s3"], mechanism="m0")],
                "reviewer_context.json": dict(incumbent=dict(hash="inc0")),
            }}  # fmt: skip
            loop = write_loop(root, ledger, per_iter)
            diag = loop.parent / "diag" / "incumbent_inc0"
            diag.mkdir(parents=True)
            (root / "scripts/rsi/families").mkdir(parents=True)
            (root / "scripts/rsi/families/fam.json").write_text(json.dumps({"scene": "outputs/scene"}))
            (root / "outputs/scene").mkdir(parents=True)
            (root / "outputs/scene/scene_meta.json").write_text(json.dumps(META))
            for label, task, ok, role in [("pick_Wolf_1_s1", "pick_Wolf_1", False, "own_support"),
                                          ("pick_Wolf_1_s2", "pick_Wolf_1", False, "own_support"),
                                          ("pick_Wolf_1_s9", "pick_Wolf_1", True, "other_furniture"),
                                          ("pick_Plant4_1_s3", "pick_Plant4_1", True, "own_support")]:  # fmt: skip
                (diag / f"{label}.json").write_text(
                    json.dumps(dict(label=label, task=task, ok=ok, steps=[], skills="inc0"))
                )
                rec = past_record(blocking_s="Shelf__1" if role == "own_support" else "Sofa__0")
                (diag / f"{label}.past.jsonl").write_text(json.dumps(rec) + "\n")
                (loop.parent / "pool_diag").mkdir(exist_ok=True)
                (loop.parent / "pool_diag" / f"{label}.json").write_text(
                    json.dumps(
                        dict(
                            task=task,
                            start=dict(
                                seed=1,
                                robot_xy=[0, 0],
                                robot_yaw=0,
                                placements={KN.object_of(task): [[0, 0, 0.9], [1, 0, 0, 0]]},
                            ),
                        )
                    )
                )
            out = root / "knowledge"
            KN.save_entries(out, [dict(ENTRY, id="fam/old", source=dict(family="fam")),  # a dropped mechanism's entry
                                  dict(ENTRY, id="other/e", tag="bimanual", source=dict(family="other"))])  # fmt: skip
            summary = KN.build(root, "fam", out, stub=True, ctx_root=root / "ctx")
            self.assertEqual((summary["entries"], summary["tag"]), (1, "mixed"))  # no tasks in the family file: mixed
            ids = [e["id"] for e in KN.entries(out)]
            self.assertIn("fam/i00_H1_hang", ids)  # the mechanism's first iteration names it
            self.assertNotIn("fam/old", ids)  # I6: a rebuild leaves no stale entry
            self.assertIn("other/e", ids)  # only this family's entries are replaced
            e = next(e for e in KN.entries(out) if e["id"] == "fam/i00_H1_hang")
            self.assertEqual((e["tag"], e["source"]["family"]), ("mixed", "fam"))
            self.assertTrue((out / "build" / "fam.md").exists())
            self.assertIn("H1_hang", (out / "INDEX.md").read_text())
            self.assertNotIn("fam/old", (out / "INDEX.md").read_text())
            # the same evidence again: the cached entry, no Consolidator call (an alternating run rebuilds each half)
            calls = []
            orig = KN.consolidate
            KN.consolidate = lambda *a, **k: calls.append(1) or orig(*a, **k)
            try:
                again = KN.build(root, "fam", out, stub=True, ctx_root=root / "ctx")
            finally:
                KN.consolidate = orig
            self.assertEqual((again["entries"], calls), (1, []))
            self.assertIn("same evidence", (out / "build" / "fam.md").read_text())


ENTRY = dict(id="fam/H1_hang", version=1, status="candidate", scope="shared", scope_why="positive on Wolf_1",
             condition=COND, signature=dict(reads_as="the pre-reach stalls at the slab"),
             repair=dict(works="filter stances", does_not=["raise D_MIN (broke 4)"], where="stance choice"),
             evidence=dict(by_object={"Wolf_1": dict(triggered=2, fixed=2, broken=0)}, separation={}, diag_rounds=[], counterexamples=[]))  # fmt: skip


class Hits(unittest.TestCase):
    def test_hits_list_failed_rounds_first(self):
        hs = KN.hits([ENTRY], ROWS[:5])
        self.assertEqual(len(hs), 1)
        self.assertEqual(hs[0]["failed"], 2)
        self.assertEqual(
            sorted(l for l, _ in hs[0]["rounds"]), ["pick_Plant4_1_s3", "pick_Wolf_1_s1", "pick_Wolf_1_s2"]
        )

    def test_no_hit_when_condition_holds_nowhere(self):
        e = dict(ENTRY, condition={"all": [{"feature": "start.support_top_m", "op": ">", "value": 5}]})
        self.assertEqual(KN.hits([e], ROWS), [])

    def test_the_role_file_shows_every_entry_marks_its_rounds_and_hides_object_names(self):
        """User 2026-11-18: the roles read the whole library; the incumbent's rounds are a mark, not a filter."""
        quiet = dict(ENTRY, id="other/quiet", tag="bimanual", source=dict(family="other", iters=[2]),
                     condition={"all": [{"feature": "start.support_top_m", "op": ">", "value": 5}]})  # fmt: skip
        ents = [dict(ENTRY, tag="single", source=dict(family="fam", iters=[0, 1])), quiet]
        md = KN.render_library(ents, KN.hits(ents, ROWS[:5]))
        self.assertIn("fam/H1_hang", md)
        self.assertIn("other/quiet", md)  # its condition holds nowhere, it is shown all the same
        self.assertIn("holds on none of its diagnostic rounds", md)
        self.assertIn("filter stances", md)
        self.assertIn("raise D_MIN", md)
        self.assertIn("2 failed round(s): pick_Wolf_1_s1, pick_Wolf_1_s2", md)
        self.assertIn("from fam iterations [0, 1]", md)
        self.assertLess(md.index("(tag single)"), md.index("(tag bimanual)"))
        self.assertNotIn("by_object", md)
        self.assertNotIn("positive on Wolf_1", md)  # scope_why names objects: never shown

    def test_the_role_file_does_not_claim_verification(self):
        md = KN.render_library([ENTRY], KN.hits([ENTRY], ROWS[:5]))
        self.assertNotIn("verified", md)  # every entry is status=candidate

    def test_the_role_file_lists_the_code_syncs(self):
        syncs = [dict(iter=1, **{"from": "a_fam", "to": "b_fam"}, merged=["pick.py"], copied=[], deleted=[],
                      conflicts=["place.py: 1 hunk(s) changed on both sides"], accepted=True, result=dict(b=0, c=0)),
                 dict(iter=1, **{"from": "b_fam", "to": "a_fam"}, skipped="the merge changes nothing")]  # fmt: skip
        md = KN.render_library([], [], syncs)
        self.assertIn("a_fam -> b_fam: merged pick.py; kept b_fam's side of place.py", md)
        self.assertIn("0 wins, 0 losses -> accepted", md)
        self.assertIn("nothing merged (the merge changes nothing)", md)
        both = KN.render_library([], [], [dict(iter=2, **{"from": "a_fam", "to": "b_fam"}, merged=[], conflicts=[
            "pick.py: 1 hunk(s) changed on both sides"], skipped="the merge changes nothing")])  # fmt: skip
        self.assertIn("nothing merged (the merge changes nothing); kept b_fam's side of pick.py", both)  # 2026-11-19

    def test_an_entry_that_holds_only_where_the_incumbent_succeeds_is_resolved_for_this_reader(self):
        e2 = dict(
            ENTRY, id="fam/other", condition={"all": [{"feature": "start.support_top_m", "op": ">=", "value": 1.0}]}
        )
        hs = KN.hits([ENTRY, e2], ROWS[:5])
        self.assertEqual([h["id"] for h in hs], ["fam/H1_hang", "fam/other"])
        md = KN.render_library([ENTRY, e2], hs)
        self.assertIn("### fam/H1_hang", md)  # it holds on failed rounds: in full
        self.assertIn("- fam/other (resolved):", md)  # one line: nothing left for it here
        self.assertIn("all of which this incumbent already succeeds on", md)

    def test_write_hits_reads_the_library_and_writes_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            KN.save_entries(root / "scripts/rsi/knowledge", [ENTRY])
            (root / "scripts/rsi/families").mkdir(parents=True)
            (root / "scripts/rsi/families/fam.json").write_text(json.dumps({"scene": "outputs/scene"}))
            (root / "outputs/scene").mkdir(parents=True)
            (root / "outputs/scene/scene_meta.json").write_text(json.dumps(META))
            rounds = root / "fam" / "diag" / "incumbent_x"
            rounds.mkdir(parents=True)
            (rounds / "pick_Wolf_1_s1.json").write_text(
                json.dumps(dict(label="pick_Wolf_1_s1", task="pick_Wolf_1", ok=False, steps=[]))
            )
            (rounds / "pick_Wolf_1_s1.past.jsonl").write_text(json.dumps(past_record()) + "\n")
            (root / "fam" / "pool_diag").mkdir()
            (root / "fam" / "pool_diag" / "pick_Wolf_1_s1.json").write_text(json.dumps(START))
            n = KN.write_hits(root, rounds, root / "it", "fam")
            self.assertEqual(n, 1)
            self.assertTrue((root / "it/knowledge_hits.json").exists())
            self.assertIn("fam/H1_hang", (root / "scripts/rsi/knowledge/KNOWLEDGE.md").read_text())

    def test_write_hits_without_a_library_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(KN.write_hits(Path(d), Path(d), Path(d) / "it", "fam"), 0)
            self.assertFalse((Path(d) / "it").exists())

    def test_write_hits_with_only_syncs_writes_the_role_file(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "scripts/rsi/knowledge"
            lib.mkdir(parents=True)
            (lib / KN.SYNCS).write_text(
                json.dumps({"iter": 1, "from": "x", "to": "y", "skipped": "y has stopped"}) + "\n"
            )
            self.assertEqual(KN.write_hits(Path(d), Path(d), Path(d) / "it", "fam"), 0)
            self.assertIn("y has stopped", (lib / "KNOWLEDGE.md").read_text())

    def test_tag_of_reads_the_tasks_objects(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "scripts/rsi/families").mkdir(parents=True)
            meta = dict(META, objects={"m1": dict(semantic=dict(graspable="single")),
                                       "m2": dict(semantic=dict(graspable="single")),
                                       "box": dict(semantic=dict(graspable="bimanual"))})  # fmt: skip
            (root / "outputs/scene").mkdir(parents=True)
            (root / "outputs/scene/scene_meta.json").write_text(json.dumps(meta))
            rows = [dict(id="pick_m1", goal=["held(m1)"]), dict(id="pick_m2", goal=["held(m2)", "lifted(m2, 0.30, 60)"]),
                    dict(id="lift_box", goal=["held(box)"])]  # fmt: skip
            (root / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            for name, tasks, extra in (("one", ["pick_m1", "pick_m2"], {}), ("both", ["pick_m1", "lift_box"], {}),
                                       ("two", ["lift_box"], {}), ("named", ["pick_m1"], {"knowledge_tag": "x"})):  # fmt: skip
                (root / f"scripts/rsi/families/{name}.json").write_text(
                    json.dumps(dict(scene="outputs/scene", tasks_file="t.jsonl", tasks=tasks, **extra))
                )
            got = [KN.tag_of(root, n) for n in ("one", "both", "two", "named")]
        self.assertEqual(got, ["single", "mixed", "bimanual", "x"])


class Wiring(unittest.TestCase):
    def test_knowledge_without_drops_own_and_excluded_families(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            lib = dest / "scripts/rsi/knowledge"
            KN.save_entries(lib, [dict(ENTRY, id=f"{f}/e", source=dict(family=f)) for f in ("fam", "old", "keep_me")])
            (lib / "KNOWLEDGE.md").write_text("an earlier incumbent's marks\n")
            (dest / "scripts/rsi/families").mkdir(parents=True)
            (dest / "scripts/rsi/families/fam.json").write_text(json.dumps({"library_exclude": ["old"]}))
            sandbox.knowledge_without(dest, "fam")
            left = [e["id"] for e in KN.entries(lib)]
            self.assertFalse((lib / "KNOWLEDGE.md").exists())
        self.assertEqual(left, ["keep_me/e"])

    def test_a_fresh_sandbox_holds_neither_library(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            KN.save_entries(dest / "scripts/rsi/knowledge", [ENTRY])
            (dest / "scripts/rsi/experience/fam").mkdir(parents=True)
            (dest / "scripts/rsi/experience/INDEX.md").write_text("x\n")
            sandbox.without_libraries(dest)
            self.assertFalse((dest / "scripts/rsi/knowledge").exists())
            self.assertFalse((dest / "scripts/rsi/experience").exists())

    def test_knowledge_without_is_a_no_op_without_the_directory(self):
        with tempfile.TemporaryDirectory() as d:
            sandbox.knowledge_without(Path(d), "fam")  # no raise

    def test_skills_may_not_name_the_knowledge_library(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.py").write_text('P = "scripts/rsi/knowledge/HITS.md"\n')
            (Path(d) / "y.py").write_text('Q = "KNOWLEDGE.md"\n')
            hits_ = LP.named_objects(Path(d), names=set())
        self.assertEqual(len(hits_), 2)
        self.assertTrue(all(LP.RECORDER_TAG in h for h in hits_))

    def test_named_void_cites_rules_38_and_39_for_recorder_hits(self):
        rows: list[dict] = []
        orig = LP.ledger
        LP.ledger = rows.append
        try:
            LP.named_void(1, [f"x.py:1 'HITS.md' {LP.RECORDER_TAG}"], "draft 1")
        finally:
            LP.ledger = orig
        self.assertIn("rules.md 30 / 38 / 39", rows[0]["reason"])

    def test_hypothesis_schema_accepts_knowledge_refs(self):
        props = LP.HYPOTHESES_SCHEMA["properties"]["hypotheses"]["items"]["properties"]
        self.assertEqual(props["knowledge_refs"], {"type": "array", "items": {"type": "string"}})

    def test_knowledge_note_names_the_role_file(self):
        self.assertIn("KNOWLEDGE.md", LP.KNOWLEDGE_NOTE)
        self.assertIn("read it whole", LP.KNOWLEDGE_NOTE)
        self.assertIn("knowledge_refs", LP.KNOWLEDGE_NOTE)

    def test_the_shared_skills_note_names_both_skill_files(self):
        note = LP.SHARED_SKILLS_NOTE.format(other="b_fam", other_kind="two-hand pick", other_skill="bimanual.py",
                                            own_skill="pick.py")  # fmt: skip
        self.assertIn("skills/bimanual.py", note)
        self.assertIn("skills/pick.py", note)
        self.assertIn("b_fam", note)

    def test_hits_md_is_gitignored_so_it_is_never_a_stray(self):
        """C1: changed_files() is `git status --porcelain -uall`; the role file written after the sandbox's commit
        must not show up there, or after_engineer / revert_strays void every draft."""
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            git = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True)  # noqa: E731
            git("init", "-q")
            (root / ".gitignore").write_text((REPO / ".gitignore").read_text())
            git("add", ".gitignore")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
            (root / "scripts/rsi/knowledge").mkdir(parents=True)
            (root / "scripts/rsi/knowledge/HITS.md").write_text("hits\n")
            (root / "scripts/rsi/knowledge/KNOWLEDGE.md").write_text("library\n")
            self.assertEqual(git("status", "--porcelain", "-uall").stdout, "")

    def _hook_fixture(self, root: Path) -> Path:
        """A tiny library and one failing diag round under a temp REPO; returns the rounds directory."""
        KN.save_entries(root / "scripts/rsi/knowledge", [ENTRY])
        (root / "scripts/rsi/families").mkdir(parents=True)
        (root / "scripts/rsi/families/fam.json").write_text(json.dumps({"scene": "outputs/scene"}))
        (root / "outputs/scene").mkdir(parents=True)
        (root / "outputs/scene/scene_meta.json").write_text(json.dumps(META))
        loop = root / "fam" / "loop"
        rounds = root / "fam" / "diag" / "incumbent_x"
        rounds.mkdir(parents=True)
        (rounds / "pick_Wolf_1_s1.json").write_text(
            json.dumps(dict(label="pick_Wolf_1_s1", task="pick_Wolf_1", ok=False, steps=[]))
        )
        (rounds / "pick_Wolf_1_s1.past.jsonl").write_text(json.dumps(past_record()) + "\n")
        (loop / "pool_diag").mkdir(parents=True)
        (loop / "pool_diag" / "pick_Wolf_1_s1.json").write_text(json.dumps(START))
        return rounds

    def _with_loop_globals(self, root: Path, rows: list[dict]):
        saved = {k: getattr(LP, k) for k in ("REPO", "LOOP", "FAMILY", "ledger")}
        LP.REPO, LP.LOOP, LP.FAMILY, LP.ledger = root, root / "fam" / "loop", "fam", rows.append
        return saved

    def test_knowledge_step_writes_hits_and_a_ledger_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rounds = self._hook_fixture(root)
            rows: list[dict] = []
            saved = self._with_loop_globals(root, rows)
            try:
                LP.knowledge_step(3, rounds, root / "it")
            finally:
                for k, v in saved.items():
                    setattr(LP, k, v)
            self.assertIn("fam/H1_hang", (root / "scripts/rsi/knowledge/KNOWLEDGE.md").read_text())
            self.assertTrue((root / "it/knowledge_hits.json").exists())
            self.assertEqual(list((root / "scripts/rsi/knowledge").glob("*.tmp")), [])
        self.assertEqual(rows, [dict(step="knowledge", iter=3, hits=1)])

    def test_knowledge_step_logs_zero_hits_too(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rounds = self._hook_fixture(root)
            e = dict(ENTRY, condition={"all": [{"feature": "start.support_top_m", "op": ">", "value": 5}]})
            KN.save_entries(root / "scripts/rsi/knowledge", [e])
            rows: list[dict] = []
            saved = self._with_loop_globals(root, rows)
            try:
                LP.knowledge_step(0, rounds, root / "it")
            finally:
                for k, v in saved.items():
                    setattr(LP, k, v)
        self.assertEqual(rows, [dict(step="knowledge", iter=0, hits=0)])

    def test_knowledge_step_error_removes_stale_hits_and_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rounds = self._hook_fixture(root)
            (root / "scripts/rsi/knowledge/KNOWLEDGE.md").write_text("an earlier incumbent's marks\n")
            (root / "scripts/rsi/knowledge/HITS.md").write_text("an older hits file\n")
            rows: list[dict] = []
            saved = self._with_loop_globals(root, rows)
            orig = KN.write_hits

            def boom(*a, **k):
                raise ValueError("bad entry")

            KN.write_hits = boom
            try:
                LP.knowledge_step(2, rounds, root / "it")
            finally:
                KN.write_hits = orig
                for k, v in saved.items():
                    setattr(LP, k, v)
            self.assertFalse((root / "scripts/rsi/knowledge/KNOWLEDGE.md").exists())
            self.assertFalse((root / "scripts/rsi/knowledge/HITS.md").exists())
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["step"], rows[0]["iter"]), ("knowledge", 2))
        self.assertIn("bad entry", rows[0]["error"])
        self.assertNotIn("hits", rows[0])

    def test_round_features_is_the_flat_table_the_reviewer_context_carries(self):
        """One row per diagnostic round in the knowledge conditions' columns (records.CONDITION_COLUMNS): the Reviewer
        reads a HITS.md condition off the same numbers. Only the Reviewer's context file carries it."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rounds = self._hook_fixture(root)
            saved = self._with_loop_globals(root, [])
            try:
                table = LP.round_features(rounds)
            finally:
                for k, v in saved.items():
                    setattr(LP, k, v)
        self.assertEqual([(r["label"], r["task"], r["ok"]) for r in table], [("pick_Wolf_1_s1", "pick_Wolf_1", 0)])
        extra = set(table[0]) - {"label", "task", "ok"}
        self.assertTrue(extra and extra <= LP.RC.CONDITION_COLUMNS, extra)
        self.assertIn("walk.blocking.role", extra)  # a graph column from the round's past record
        self.assertIn("start.spawn_dist_m", extra)  # a start column from pool_diag
        src = (REPO / "scripts/rsi/loop.py").read_text()
        self.assertEqual(src.count("round_features=round_features(base)"), 1)
        self.assertIn('reviewer_context.json"\n    jdump(ctx, context(inc, dict(extra, round_features=', src)

    def test_install_knowledge_keeps_the_library_as_given(self):
        """I2: the replay's library under test keeps the family's own (temporal-split) entries."""
        RP = _replay()
        with tempfile.TemporaryDirectory() as d:
            dst, given = Path(d) / "sb", Path(d) / "given"
            KN.save_entries(dst / "scripts/rsi/knowledge", [dict(ENTRY, id="exported/e")])
            (dst / "scripts/rsi/families").mkdir(parents=True)
            (dst / "scripts/rsi/families/fam.json").write_text(json.dumps({"library_exclude": ["old"]}))
            KN.save_entries(given, [dict(ENTRY, id=f"{f}/e", source=dict(family=f)) for f in ("fam", "old")])
            (given / "KNOWLEDGE.md").write_text("stale\n")
            RP.install_knowledge(dst, given)
            lib = dst / "scripts/rsi/knowledge"
            self.assertEqual(sorted(e["id"] for e in KN.entries(lib)), ["fam/e", "old/e"])
            self.assertFalse((lib / "KNOWLEDGE.md").exists())
            self.assertIn("fam/e", (lib / "INDEX.md").read_text())
            self.assertTrue((given / "KNOWLEDGE.md").exists())  # the given directory is not touched

    def test_setup_checks_the_knowledge_path_before_building(self):
        RP = _replay()
        built = []
        orig = RP.SB.build
        RP.SB.build = lambda *a, **k: built.append(a)
        try:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(SystemExit):
                    RP.setup(Path(d) / "src", "fam", 0, Path(d) / "dst", knowledge=Path(d) / "missing")
        finally:
            RP.SB.build = orig
        self.assertEqual(built, [])

    def test_iterate_runs_the_knowledge_step_after_the_pool_redraw(self):
        import inspect

        src = inspect.getsource(LP.iterate)
        self.assertLess(src.index("refresh_diag_pool("), src.index("knowledge_step("))
        self.assertLess(src.index("unpark("), src.index("knowledge_step("))


def cited_loop(root: Path) -> Path:
    """Three iterations whose hypotheses cite entries x/a, x/b, x/c: a draft accepted alone, a refutation, an untested
    one not picked, a merged candidate accepted, an adopted draft (with overrides, so another hash) accepted, and a
    sync verdict that must not count."""
    H = lambda i, refs, **k: dict(id=i, claim="c", instead="i", knowledge_refs=refs, **k)  # noqa: E731
    ledger = [
        dict(step="probe", iter=0, results=[dict(id="H1", status="verified"), dict(id="H2", status="refuted"),
                                            dict(id="H3", status="hypothesis")]),
        dict(step="manager", iter=0, decision="engineer", hypothesis="H1", hypotheses=["H1", "H2"]),
        dict(step="compose", iter=0, decision="keep"),
        dict(step="gate", iter=0, hash="d1", accepted=True),
        dict(step="probe", iter=1, results=[dict(id="H4", status="refuted"), dict(id="H5", status="mixed")]),
        dict(step="manager", iter=1, decision="engineer", hypothesis="H5", hypotheses=["H5", "H6"]),
        dict(step="compose", iter=1, decision="merge"),
        dict(step="gate", iter=1, hash="m1", accepted=True),
        dict(step="gate", hash="s1", accepted=True, sync={"from": "other"}),
        dict(step="probe", iter=2, results=[dict(id="H7", status="hypothesis")]),
        dict(step="manager", iter=2, decision="engineer", hypothesis="H7"),
        dict(step="compose", iter=2, decision="adopt", adopted=1),
        dict(step="gate", iter=2, hash="a2", accepted=True),
    ]  # fmt: skip
    per_iter = {
        0: {"hypotheses.json": dict(hypotheses=[H("H1", ["x/a"]), H("H2", ["x/b", "x/a", "x/b"]), H("H3", ["x/c"])]),
            "candidates.json": [dict(number=1, hypothesis="H1", hash="d1"), dict(number=2, hypothesis="H2", hash="d2")]},
        1: {"hypotheses.json": dict(hypotheses=[H("H4", ["x/b"]), H("H5", ["x/c"]), H("H6", [])]),
            "candidates.json": [dict(number=1, hypothesis="H5", hash="d5"), dict(number=2, hypothesis="H6", hash="d6")]},
        2: {"hypotheses.json": dict(hypotheses=[H("H7", ["x/d"])]),
            "candidates.json": [dict(number=1, hypothesis="H7", hash="d7")]},
    }  # fmt: skip
    return write_loop(root, ledger, per_iter)


class GateAttribution(unittest.TestCase):
    """2026-11-18: a gate verdict reaches the drafts it judged -- a draft alone, the Composer's adopted draft, or the
    drafts a merge kept (`merged`) --, with b and c read from the row's `result` as loop.py writes them."""

    DRAFTS = [dict(number=1, hypothesis="H1", hash="d1"), dict(number=2, hypothesis="H2", hash="d2"),
              dict(number=3, hypothesis="H3", hash="d3")]  # fmt: skip

    def test_a_drafts_own_hash(self):
        rows = [dict(step="gate", iter=0, hash="d2", accepted=False, result=dict(b=1, c=2))]
        self.assertEqual(KN.draft_verdicts(rows, self.DRAFTS), {2: dict(b=1, c=2, accepted=False, via="draft")})

    def test_a_merge_credits_only_the_drafts_it_kept(self):
        rows = [dict(step="compose", iter=0, decision="merge", merged=[1, 3]),
                dict(step="gate", iter=0, hash="m1", accepted=True, result=dict(b=5, c=1))]  # fmt: skip
        got = KN.draft_verdicts(rows, self.DRAFTS)
        self.assertEqual(sorted(got), [1, 3])
        self.assertEqual(got[1], dict(b=5, c=1, accepted=True, via="merged"))

    def test_an_adopted_draft_with_overrides_and_an_older_merge(self):
        adopt = [dict(step="compose", iter=0, decision="adopt", adopted=2),
                 dict(step="gate", iter=0, hash="x9", accepted=True, result=dict(b=4, c=0))]  # fmt: skip
        self.assertEqual(KN.draft_verdicts(adopt, self.DRAFTS)[2]["via"], "adopted")
        old = [dict(step="compose", iter=0, decision="merge"), dict(step="gate", iter=0, hash="m1", accepted=True)]
        self.assertEqual(KN.draft_verdicts(old, self.DRAFTS), {})  # which drafts went in is not known

    def test_a_sync_verdict_is_no_drafts(self):
        rows = [dict(step="gate", hash="d1", accepted=True, sync={"from": "other"})]
        self.assertEqual(KN.draft_verdicts(rows, self.DRAFTS), {})

    def test_a_mechanism_gets_its_merged_candidates_verdict(self):
        ledger = [dict(step="probe", iter=0, results=[dict(id="H1", fixed=["pick_Wolf_1_s1"], broken=[])]),
                  dict(step="manager", iter=0, decision="engineer", hypotheses=["H1", "H2"], repeats=[]),
                  dict(step="compose", iter=0, decision="merge", merged=[1]),
                  dict(step="gate", iter=0, hash="m1", accepted=True, result=dict(b=6, c=1))]  # fmt: skip
        per_iter = {0: {"hypotheses.json": dict(hypotheses=[dict(id="H1", claim="c", instead="i"),
                                                             dict(id="H2", claim="c", instead="i")]),
                        "candidates.json": [dict(number=1, hypothesis="H1", hash="d1", fixed=["pick_Wolf_1_s1"], broken=[]),
                                            dict(number=2, hypothesis="H2", hash="d2", fixed=[], broken=[])],
                        "reviewer_context.json": dict(incumbent=dict(hash="inc0"))}}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            ms = {m["key"]: m for m in KN.mechanisms(write_loop(Path(d), ledger, per_iter))}
        self.assertEqual(ms["i00_H1"]["gate"], dict(b=6, c=1, accepted=True, via="merged"))
        self.assertNotIn("i00_H2", ms)  # not merged, no flip: nothing to learn from

    def test_two_mechanisms_named_alike_in_different_iterations_get_different_keys(self):
        """A hypothesis id is unique within its iteration only: two `H1`s rooted in iterations 0 and 2 are two
        entries, not one."""
        H = lambda i: dict(id=i, claim="c", instead="i")  # noqa: E731
        ledger = [dict(step="probe", iter=0, results=[dict(id="H1", fixed=["t_s1"], broken=[])]),
                  dict(step="probe", iter=2, results=[dict(id="H1", fixed=["t_s2"], broken=[])])]  # fmt: skip
        ctx = {"reviewer_context.json": dict(incumbent=dict(hash="inc"))}
        per_iter = {0: {"hypotheses.json": dict(hypotheses=[H("H1")]), **ctx},
                    2: {"hypotheses.json": dict(hypotheses=[H("H1")]), **ctx}}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            keys = sorted(m["key"] for m in KN.mechanisms(write_loop(Path(d), ledger, per_iter)))
        self.assertEqual(keys, ["i00_H1", "i02_H1"])

    def test_the_composer_names_the_drafts_it_merged(self):
        LP = _scene_loop()
        self.assertEqual(LP.COMPOSER_SCHEMA["properties"]["merged"], {"type": "array", "items": {"type": "integer"}})
        self.assertIn("`merged`", (REPO / "scripts/rsi/prompts/composer.md").read_text())
        self.assertIn("merged=merged", (REPO / "scripts/rsi/loop.py").read_text())


class Lifecycle(unittest.TestCase):
    """User 2026-11-18: an entry's citations (Reviewer knowledge_refs) and how the citing hypotheses ended set its
    standing -- tested, candidate, doubtful, and for the reader resolved -- which orders KNOWLEDGE.md."""

    def test_each_citation_gets_the_hypothesis_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            rows = KN.usage_rows(cited_loop(Path(d)), "fam", "run1")
        got = {(r["entry"], r["hypothesis"]): (r["status"], r["picked"], r["accepted"], r["outcome"]) for r in rows}
        self.assertEqual(got, {
            ("x/a", "H1"): ("verified", True, "draft", "helped"),  # its own draft passed the gate
            ("x/b", "H2"): ("refuted", True, None, "failed"),
            ("x/a", "H2"): ("refuted", True, None, "failed"),
            ("x/c", "H3"): ("hypothesis", False, None, "unclear"),  # not picked, not measured
            ("x/b", "H4"): ("refuted", False, None, "failed"),
            ("x/c", "H5"): ("mixed", True, "merged", "helped"),  # the merged candidate passed: every draft credited
            ("x/d", "H7"): ("hypothesis", True, "draft", "helped"),  # the Composer adopted its draft (with overrides)
        })  # fmt: skip
        self.assertEqual(len(rows), 7)  # H2 names x/b twice: one citation

    def test_a_merge_that_names_its_drafts_credits_only_them(self):
        H = lambda i, refs: dict(id=i, claim="c", instead="i", knowledge_refs=refs)  # noqa: E731
        ledger = [dict(step="probe", iter=0, results=[dict(id="H1", status="mixed"), dict(id="H2", status="mixed")]),
                  dict(step="compose", iter=0, decision="merge", merged=[2]),
                  dict(step="gate", iter=0, hash="m1", accepted=True, result=dict(b=4, c=0))]  # fmt: skip
        per_iter = {0: {"hypotheses.json": dict(hypotheses=[H("H1", ["x/a"]), H("H2", ["x/b"])]),
                        "candidates.json": [dict(number=1, hypothesis="H1", hash="d1"),
                                            dict(number=2, hypothesis="H2", hash="d2")]}}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            rows = {r["entry"]: r for r in KN.usage_rows(write_loop(Path(d), ledger, per_iter), "fam", "r")}
        self.assertEqual((rows["x/b"]["accepted"], rows["x/b"]["outcome"]), ("merged", "helped"))
        self.assertEqual((rows["x/a"]["accepted"], rows["x/a"]["outcome"]), (None, "unclear"))  # not in the merge

    def test_a_recount_replaces_the_runs_rows_and_keeps_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "lib"
            other = dict(entry="x/a", run="run0", family="fam", iter=0, hypothesis="H9", outcome="helped")
            KN.update_usage(lib, "run0", "fam", [other])
            rows = KN.usage_rows(cited_loop(Path(d)), "fam", "run1")
            KN.update_usage(lib, "run1", "fam", rows)
            KN.update_usage(lib, "run1", "fam", rows)  # again: nothing counted twice
            allr = KN.jl(lib / KN.USAGE)
        self.assertEqual(len(allr), 8)
        st = KN.entry_stats(allr)
        self.assertEqual(st["x/a"], dict(uses=3, helped=2, failed=1, unclear=0, accepted=1))

    def test_the_standing_rules(self):
        S = lambda h, f, u=0: dict(uses=h + f + u, helped=h, failed=f, unclear=u, accepted=0)  # noqa: E731
        self.assertEqual(KN.status_of(None), "candidate")
        self.assertEqual(KN.status_of(S(1, 0)), "tested")
        self.assertEqual(KN.status_of(S(2, 2)), "tested")  # helped at least as often as not
        self.assertEqual(KN.status_of(S(0, 1)), "candidate")  # one failure is not enough
        self.assertEqual(KN.status_of(S(0, 2)), "doubtful")
        self.assertEqual(KN.status_of(S(1, 3)), "doubtful")
        self.assertEqual(KN.status_of(S(0, 0, 4)), "candidate")  # cited, never measured

    def test_the_role_file_orders_by_standing_and_shortens_doubtful_and_resolved(self):
        never = dict(ENTRY, id="fam/never", tag="single", source=dict(family="fam"))
        good = dict(ENTRY, id="fam/good", tag="single", source=dict(family="fam"))
        bad = dict(ENTRY, id="fam/bad", tag="single", source=dict(family="fam"))
        done = dict(ENTRY, id="fam/done", tag="single", source=dict(family="fam"),
                    condition={"all": [{"feature": "start.support_top_m", "op": ">=", "value": 1.0}]})  # fmt: skip
        U = lambda e, o: dict(entry=e, run="r", family="fam", iter=0, hypothesis=e + o, outcome=o)  # noqa: E731
        usage = [U("fam/good", "helped"), U("fam/bad", "failed"), U("fam/bad", "failed") | {"hypothesis": "h2"}]
        ents = [never, good, bad, done]
        md = KN.render_library(ents, KN.hits(ents, ROWS[:5]), None, usage)
        self.assertLess(md.index("### fam/good"), md.index("### fam/never"))  # tested before candidate
        self.assertIn("### fam/good  (shared, v1, tested;", md)
        self.assertIn("Cited 1 time(s): helped 1, no effect 0, unclear 0", md)
        self.assertIn("Not cited yet", md)
        self.assertNotIn("### fam/bad", md)  # one line, after the full entries
        self.assertIn("- fam/bad (doubtful):", md)
        self.assertIn("no effect 2", md.split("- fam/bad (doubtful):", 1)[1].split("\n", 1)[0])
        self.assertIn("- fam/done (resolved):", md)  # holds only where this incumbent already succeeds
        self.assertLess(md.index("### fam/never"), md.index("- fam/bad (doubtful):"))

    def test_the_usage_command_records_citations_and_the_index_shows_the_standing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cited_loop(root)
            out = root / "lib"
            KN.save_entries(out, [dict(ENTRY, id="x/a", tag="single", source=dict(family="x"))])
            self.assertEqual(KN.main(["usage", str(root), "fam", "--out", str(out)]), 0)
            self.assertEqual(len(KN.jl(out / KN.USAGE)), 7)
            self.assertIn("| x/a | single | shared | tested | 2 (1 / 1 / 0) |", (out / "INDEX.md").read_text())


class Report(unittest.TestCase):
    def test_report_row_from_a_replayed_iteration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            loop = write_loop(root, [
                dict(step="review", iter=4, n_hypotheses=2, cost_usd=1.5, seconds=300),
                dict(step="probe", iter=4, results=[
                    dict(id="A", status="verified", fixed=["x", "y"], broken=[], predicted_fixed_hit=2, predicted_fixed_n=3),
                    dict(id="B", status="refuted", fixed=[], broken=["z"], predicted_fixed_hit=0, predicted_fixed_n=1)]),
            ], {4: {"hypotheses.json": dict(hypotheses=[dict(id="A", knowledge_refs=["fam/H1_hang"]), dict(id="B")]),
                    "knowledge_hits.json": []}})  # fmt: skip
            sb = loop.parents[3]
            row = KN.report([sb])[0]
        self.assertEqual((row["n_hyp"], row["verified"], row["refuted"], row["net_fixed"]), (2, 1, 1, 1))
        self.assertEqual((row["pred_hit"], row["pred_n"], row["cites"]), (2, 4, 1))
        self.assertTrue(row["with_knowledge"])
        self.assertAlmostEqual(row["review_cost_usd"], 1.5)


if __name__ == "__main__":
    unittest.main()

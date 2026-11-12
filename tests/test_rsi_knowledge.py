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
            summary = KN.build(root, "fam", out, stub=True, ctx_root=root / "ctx")
            self.assertEqual(summary["entries"], 1)
            self.assertTrue((out / "fam" / "H1_hang.json").exists())
            self.assertTrue((out / "fam" / "BUILD.md").exists())
            self.assertIn("H1_hang", (out / "INDEX.md").read_text())


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

    def test_render_hits_shows_repair_and_hides_object_names_from_evidence(self):
        md = KN.render_hits(KN.hits([ENTRY], ROWS[:5]), {ENTRY["id"]: ENTRY})
        self.assertIn("fam/H1_hang", md)
        self.assertIn("filter stances", md)
        self.assertIn("raise D_MIN", md)
        self.assertIn("pick_Wolf_1_s1 (failed)", md)
        self.assertNotIn("by_object", md)
        self.assertNotIn("positive on Wolf_1", md)

    def test_render_hits_without_hits_is_one_line(self):
        md = KN.render_hits([], {})
        self.assertEqual(md.strip().count("\n"), 0)
        self.assertIn("no entry", md)

    def test_all_ok_hits_go_last_and_collapsed(self):
        e2 = dict(
            ENTRY, id="fam/other", condition={"all": [{"feature": "start.support_top_m", "op": ">=", "value": 1.0}]}
        )
        hs = KN.hits([ENTRY, e2], ROWS[:5])
        self.assertEqual([h["id"] for h in hs], ["fam/H1_hang", "fam/other"])
        md = KN.render_hits(hs, {ENTRY["id"]: ENTRY, e2["id"]: e2})
        self.assertIn("already succeeds", md)

    def test_write_hits_reads_the_library_and_writes_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "scripts/rsi/knowledge/fam").mkdir(parents=True)
            (root / "scripts/rsi/knowledge/fam/H1_hang.json").write_text(json.dumps(ENTRY))
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
            self.assertIn("fam/H1_hang", (root / "scripts/rsi/knowledge/HITS.md").read_text())

    def test_write_hits_without_a_library_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(KN.write_hits(Path(d), Path(d), Path(d) / "it", "fam"), 0)
            self.assertFalse((Path(d) / "it").exists())


class Wiring(unittest.TestCase):
    def test_knowledge_without_drops_own_and_excluded_families(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            for fam in ("fam", "old", "keep_me"):
                (dest / "scripts/rsi/knowledge" / fam).mkdir(parents=True)
                (dest / "scripts/rsi/knowledge" / fam / "e.json").write_text(json.dumps(dict(ENTRY, id=f"{fam}/e")))
            (dest / "scripts/rsi/families").mkdir(parents=True)
            (dest / "scripts/rsi/families/fam.json").write_text(json.dumps({"library_exclude": ["old"]}))
            sandbox.knowledge_without(dest, "fam")
            left = sorted(p.name for p in (dest / "scripts/rsi/knowledge").iterdir() if p.is_dir())
        self.assertEqual(left, ["keep_me"])

    def test_knowledge_without_is_a_no_op_without_the_directory(self):
        with tempfile.TemporaryDirectory() as d:
            sandbox.knowledge_without(Path(d), "fam")  # no raise

    def test_skills_may_not_name_the_knowledge_library(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.py").write_text('P = "scripts/rsi/knowledge/HITS.md"\n')
            hits_ = LP.named_objects(Path(d), names=set())
        self.assertEqual(len(hits_), 1)
        self.assertIn(LP.RECORDER_TAG, hits_[0])

    def test_hypothesis_schema_accepts_knowledge_refs(self):
        props = LP.HYPOTHESES_SCHEMA["properties"]["hypotheses"]["items"]["properties"]
        self.assertEqual(props["knowledge_refs"], {"type": "array", "items": {"type": "string"}})

    def test_knowledge_note_names_the_hits_file(self):
        self.assertIn("HITS.md", LP.KNOWLEDGE_NOTE)
        self.assertIn("knowledge_refs", LP.KNOWLEDGE_NOTE)


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

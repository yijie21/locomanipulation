"""TODO 39.3 / 39.4: the recorder's side files in the loop's tools (batch.retire, the digest section, past.py CLI)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import batch as B  # noqa: E402
import digest as DG  # noqa: E402

_spec = importlib.util.spec_from_file_location("past_cli", REPO / "scripts" / "rsi" / "past.py")
past_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(past_cli)


class Retire(unittest.TestCase):
    def test_retire_renames_the_past_side_file_too(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "pick_s1.json"
            f.write_text(json.dumps(dict(label="pick_s1", ok=True, rule=B.SUCCESS_RULE, motor="old")))
            (Path(d) / "pick_s1.npz").write_bytes(b"x")
            (Path(d) / "pick_s1.past.jsonl").write_text("{}\n")
            B.retire(f)
            names = sorted(p.name for p in Path(d).iterdir())
        self.assertEqual(names, ["_motorold_pick_s1.json", "_motorold_pick_s1.npz", "_motorold_pick_s1.past.jsonl"])


def graph(t, gripped):
    return dict(t=t, states={}, edges=[dict(s="Mug", r="gripped", o="right_hand", v=gripped, src="observed"),
                                       dict(s="Mug", r="supported_by", o="T", v=True, src="derived")])  # fmt: skip


def two_step_records(label, ok):
    """A reach that lands, then a grip that closes both fingers only when `ok`."""
    ev = [{"t": 9.2, "edge": ["Mug", "touching", "right_hand"], "from": False, "to": True, "src": "observed"}]
    disc = [] if ok else [dict(edge=["Mug", "gripped", "right_hand"], to=True, observed=False)]
    common = dict(schema="PAST-locomani-0.1", label=label, task="pick_Mug", attempt=1, roles=dict(target="Mug", effector="right_hand"),
                  scope=dict(initial=["Mug"], added=[]), entities={}, outcome=dict(step_ok=True, round_ok=ok))  # fmt: skip
    return [
        dict(common, record_id=f"{label}/0.1", step_index=0, step="pregrasp", kind="reach",
             action=dict(t0=8.0, t1=9.0, ok=True, why="", calls=[], check={}), G_before=graph(8.0, False), events=[],
             G_after=graph(9.0, False), expected=[], discrepancies=[], evidence=dict(npz=f"{label}.npz", k0=400, k1=450)),
        dict(common, record_id=f"{label}/1.1", step_index=1, step="grip", kind="grip",
             action=dict(t0=9.0, t1=9.6, ok=ok, why="", calls=[], check={"opening_cm": 6.1}), G_before=graph(9.0, False),
             events=ev, G_after=graph(9.6, ok), expected=[dict(edge=["Mug", "gripped", "right_hand"], to=True)],
             discrepancies=disc, evidence=dict(npz=f"{label}.npz", k0=450, k1=480)),
    ]  # fmt: skip


def write_round(d, label, ok):
    row = dict(label=label, task="pick_Mug", ok=ok, steps=[])
    (Path(d) / f"{label}.json").write_text(json.dumps(row))
    (Path(d) / f"{label}.past.jsonl").write_text("".join(json.dumps(r) + "\n" for r in two_step_records(label, ok)))
    return row


class DigestSection(unittest.TestCase):
    def test_counts_the_first_unexpected_transition_per_task_and_lists_failed_timelines(self):
        with tempfile.TemporaryDirectory() as d:
            rounds = [write_round(d, f"pick_Mug_s{i}", ok) for i, ok in enumerate([False, False, True])]
            lines = DG.past_lines(Path(d), rounds)
        text = "\n".join(lines)
        self.assertIn("## Scene-graph transitions", text)
        self.assertIn(
            "- pick_Mug: 2 x grip: Mug gripped right_hand expected true, observed false; successes: 1/1 true at the same kind",
            text,
        )
        self.assertIn("- pick_Mug_s0", text)
        self.assertIn("  9.2  grip.grip  Mug touching right_hand false->true", text)
        self.assertIn("  9.6  grip.grip  Mug gripped right_hand expected true, observed false *", text)
        self.assertNotIn("pick_Mug_s2\n", text)  # the success has no timeline
        self.assertLessEqual(len(lines), DG.PAST_MAX_LINES)

    def test_no_side_files_means_no_section(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(DG.past_lines(Path(d), [dict(label="a", task="t", ok=False)]), [])


class DigestFacts(unittest.TestCase):
    def test_a_pack_goal_gives_each_object_one_line_with_its_semantic_mass(self):
        box = dict(half_extent=[0.225, 0.16, 0.165], centre=[0, 0, 0.165], wall=0.01)
        meta = dict(surfaces={}, objects=dict(
            nor_cal_box=dict(box=box, init_pos=[0, 0, 0.371], semantic=dict(mass_kg=0.6)),
            gso_0050=dict(box=dict(half_extent=[0.03, 0.03, 0.05]), semantic=dict(mass_kg=0.25))))  # fmt: skip
        goal = ["in(gso_0050, nor_cal_box)", "held(nor_cal_box)", "lifted(nor_cal_box, 0.2, 60)"]
        lines = DG.facts(meta, goal)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("- gso_0050:") and "0.25 kg" in lines[0])
        self.assertTrue(lines[1].startswith("- nor_cal_box:") and "0.60 kg" in lines[1] and "container:" in lines[1])


class Cli(unittest.TestCase):
    def test_show_prints_steps_edges_events_and_expectations(self):
        with tempfile.TemporaryDirectory() as d:
            write_round(d, "pick_Mug_s0", False)
            text = past_cli.show(Path(d) / "pick_Mug_s0")
        self.assertIn("pick_Mug_s0  task pick_Mug  round FAIL", text)
        self.assertIn("1.1 grip [grip] 9.0-9.6 s FAIL", text)
        self.assertIn("  Mug —gripped→ right_hand  false→false  (expected true ✗)", text)
        self.assertIn("  Mug —supported_by→ T  true→true", text)
        self.assertIn("    9.2  Mug —touching→ right_hand  false→true", text)
        self.assertIn("  evidence pick_Mug_s0.npz ticks 450-480", text)

    def test_show_prints_each_objects_affordance_once_and_marks_a_stale_one(self):
        aff = dict(graspable="single", phrase="the mug", src={},
                   one_hand=dict(jaw_open_m=0.095, grasp_z_above_origin_m=0.04, flat=False, below_jaw=False,
                                 side_fits_deg=[[170.0, -170.0], [-10.0, 10.0]],
                                 narrowest=dict(approach_deg=-0.2, width_m=0.06, margin_m=0.0175), widest_m=0.14),
                   hug=dict(span_x_m=0.14, span_y_m=0.06, height_m=0.1, mass_kg=0.3))  # fmt: skip
        recs = two_step_records("pick_Mug_s0", False)
        for i, r in enumerate(recs):
            r["entities"] = {"Mug": dict(type="object", affordance=dict(aff, tilt_deg=80.0 * i, stale=i > 0))}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pick_Mug_s0.past.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            text = past_cli.show(Path(d) / "pick_Mug_s0")
        self.assertEqual(text.count("Mug  single"), 1)
        self.assertIn(
            '  Mug  single  "the mug"  jaw fits from [+170,-170] [-10,+10] deg, narrowest 6.0 cm at +0 deg (+1.8 cm a'
            " side); grasp z +0.04 m  hug spans x 14.0 y 6.0 cm, 10.0 cm tall 0.3 kg",
            text,
        )
        self.assertIn("  affordance stale (one-hand check from the start pose): Mug tilted 80 deg", text)

    def test_show_one_step_as_json_and_the_json_suffix_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            write_round(d, "pick_Mug_s0", False)
            text = past_cli.show(Path(d) / "pick_Mug_s0.json", step=1, as_json=True)
        self.assertEqual(json.loads(text)["record_id"], "pick_Mug_s0/1.1")

    def test_compare_aligns_by_kind_and_marks_the_differing_edge(self):
        with tempfile.TemporaryDirectory() as d:
            write_round(d, "pick_Mug_s0", False)
            write_round(d, "pick_Mug_s1", True)
            text = past_cli.compare(Path(d) / "pick_Mug_s0", Path(d) / "pick_Mug_s1")
        self.assertIn("grip | grip   A grip FAIL   B grip OK", text)
        self.assertIn("  Mug —gripped→ right_hand  A false  B true  <-- differs", text)
        self.assertNotIn("supported_by", text)  # equal on both sides: not listed

    def test_main_show_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            write_round(d, "pick_Mug_s0", False)
            self.assertEqual(past_cli.main(["show", str(Path(d) / "pick_Mug_s0")]), 0)


import records as RC  # noqa: E402


def rich_records(label, ok):
    """Two steps with the spatial, relative-pose and event content graph_columns reads."""
    recs = two_step_records(label, ok)
    reach, grip = recs
    reach["roles"]["carried"] = None
    reach["G_before"]["edges"] += [
        dict(s="Can", r="beside", o="Mug", v=True, src="derived", gap_m=0.02, side="+y"),
        dict(s="Far", r="beside", o="Mug", v=False, src="derived", gap_m=0.31, side="-x"),
        dict(s="Stool", r="blocking", o="region:corridor", v=True, src="derived", depth_m=0.04),
        dict(s="Shelf", r="above", o="Mug", v=True, src="derived", gap_m=0.25),
        dict(s="Mug", r="near_edge", o="T", v=True, src="derived", dist_m=0.12, edge="-x"),
        dict(s="robot", r="dist", o="Mug", v=0.64, src="derived", bearing_deg=8.0),
        dict(s="robot", r="facing", o="Mug", v=8.0, src="derived"),
    ]
    reach["G_after"]["edges"] += [
        dict(s="right_hand", r="reach_vec", o="region:pregrasp_target", v=[0.48, 0.21, 0.52], src="derived"),
        dict(s="right_hand", r="at", o="region:pregrasp_target", v=True, src="derived", err_cm=1.3),
        dict(s="right_hand", r="approach", o="Mug", v="-x", src="derived", wrist_att="level"),
    ]
    grip["events"] = [
        {"t": 9.1, "edge": ["Mug", "moved", None], "from": False, "to": True, "cm": 2.4, "src": "derived"},
        {"t": 9.2, "edge": ["Mug", "touching", "right_hand"], "from": False, "to": True, "src": "observed"},
    ] + (
        [{"t": 9.4, "edge": ["Mug", "gripped", "right_hand"], "from": False, "to": True, "src": "observed"}]
        if ok
        else []
    )
    return recs


class GraphColumns(unittest.TestCase):
    def test_columns_keyed_by_kind_with_the_first_discrepant_step(self):
        cols = RC.graph_columns(rich_records("pick_Mug_s0", ok=False))
        self.assertEqual(
            (cols["first_bad.kind"], cols["first_bad.edge"], cols["n_unexpected"]),
            ("grip", "Mug gripped right_hand", 1.0),
        )
        self.assertEqual(
            (cols["reach.beside.gap_m"], cols["reach.beside.side"], cols["reach.beside.n"]), (0.02, "+y", 1.0)
        )
        self.assertEqual((cols["reach.blocking.depth_m"], cols["reach.blocking.what"]), (0.04, "Stool"))
        self.assertEqual(cols["reach.above.gap_m"], 0.25)
        self.assertEqual((cols["reach.near_edge.dist_m"], cols["reach.near_edge.edge"]), (0.12, "-x"))
        self.assertEqual((cols["reach.dist_m"], cols["reach.facing_deg"]), (0.64, 8.0))
        self.assertEqual(
            (cols["reach.reach_vec.fwd"], cols["reach.reach_vec.out"], cols["reach.reach_vec.z"]), (0.48, 0.21, 0.52)
        )
        self.assertEqual((cols["reach.at.err_cm"], cols["reach.approach.side"]), (1.3, "-x"))
        self.assertEqual(cols["grip.moved_before_grip"], 1)  # moved at 9.1, never gripped
        self.assertNotIn("grip.t_touch_to_grip_s", cols)

    def test_a_clean_grasp_has_no_first_bad_and_a_touch_to_grip_time(self):
        cols = RC.graph_columns(rich_records("pick_Mug_s1", ok=True))
        self.assertEqual(cols["n_unexpected"], 0.0)
        self.assertNotIn("first_bad.kind", cols)
        self.assertEqual(cols["grip.moved_before_grip"], 1)  # moved at 9.1 before the grip at 9.4
        self.assertAlmostEqual(cols["grip.t_touch_to_grip_s"], 0.2)

    def test_scan_reads_the_side_file_and_a_round_without_one_has_no_graph_columns(self):
        with tempfile.TemporaryDirectory() as d:
            write_round(d, "pick_Mug_s0", False)
            (Path(d) / "pick_Mug_s0.past.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rich_records("pick_Mug_s0", ok=False))
            )
            (Path(d) / "pick_Mug_s9.json").write_text(
                json.dumps(dict(label="pick_Mug_s9", task="pick_Mug", ok=True, steps=[]))
            )
            rows = {r["label"]: r for r in RC.scan([(Path(d), "fam")], "host")}
        self.assertEqual(rows["pick_Mug_s0"]["reach.beside.gap_m"], 0.02)
        self.assertEqual(rows["pick_Mug_s0"]["first_bad.kind"], "grip")
        self.assertNotIn("reach.beside.gap_m", rows["pick_Mug_s9"])

    def test_judge_splits_on_a_graph_column(self):
        rows = [dict(task="pick_Mug", ok=0, machine="h", **{"reach.beside.gap_m": g}) for g in (0.01, 0.02, 0.025)]
        rows += [dict(task="pick_Mug", ok=1, machine="h", **{"reach.beside.gap_m": g}) for g in (0.08, 0.10, 0.12, 0.2)]
        q = RC.judge(rows, dict(feature="reach.beside.gap_m", op="<", value=0.03, expect="fails"), None, "h")
        self.assertEqual((q["verdict"], q["n_in"], q["ok_in"], q["n_out"], q["ok_out"]), ("verified", 3, 0, 4, 4))


class DigestColumns(unittest.TestCase):
    def test_the_section_ends_with_the_available_columns(self):
        with tempfile.TemporaryDirectory() as d:
            rounds = [write_round(d, "pick_Mug_s0", False)]
            (Path(d) / "pick_Mug_s0.past.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rich_records("pick_Mug_s0", ok=False))
            )
            text = "\n".join(DG.past_lines(Path(d), rounds))
        self.assertIn("Record-library columns from these records", text)
        self.assertIn("reach.beside.gap_m", text)
        self.assertIn('"measure"', text)


class NoPlanRound(unittest.TestCase):
    def test_show_reports_the_note_of_a_round_that_never_moved(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pick_s1.past.jsonl").write_text(
                json.dumps(
                    dict(schema="PAST-locomani-0.1", label="pick_s1", note="no records: no skill covers this goal")
                )
                + "\n"
            )
            self.assertEqual(past_cli.show(Path(d) / "pick_s1"), "no records: no skill covers this goal")
            (Path(d) / "pick_s2.past.jsonl").write_text("")
            self.assertEqual(past_cli.show(Path(d) / "pick_s2"), "no records")

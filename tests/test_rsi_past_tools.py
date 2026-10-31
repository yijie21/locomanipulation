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

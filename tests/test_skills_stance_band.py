"""stance_violations as a pure band check, and the precondition replay's readers/evaluator on tiny recordings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from locomani.skills import Stance, stance_violations

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import replay_preconditions as rp  # noqa: E402


def test_stance_violations_are_a_pure_band_check():
    st = Stance("box", "warehouse_box", 0.52, -0.03, 0.0, True)
    assert stance_violations(st, {}) == [] and stance_violations(None, {"depth": (0, 1)}) == []
    assert stance_violations(st, {"depth": (None, 0.55), "lateral": (-0.05, 0.05)}) == []
    out = stance_violations(st, {"depth": (0.30, 0.50), "lateral": (-0.02, 0.02)})
    assert [(v.check, v.measured, v.bound) for v in out] == [
        ("stance_depth", 0.52, (0.30, 0.50)),
        ("stance_lateral", -0.03, (-0.02, 0.02)),
    ]
    assert stance_violations(Stance("b", "p", float("nan"), 0.0, 0.0, True), {"depth": (0, 0.1)}) == []


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_readers_and_evaluate(tmp_path):
    run = tmp_path / "skills_x"
    stance = {
        "target": "b",
        "profile": "bottle_simple",
        "depth_m": 0.40,
        "lateral_m": -0.2,
        "yaw_err_rad": 0.0,
        "quiet": True,
    }
    _write(
        run / "skills" / "agent_trace_00.jsonl",
        [
            {"primitive": "approach", "args": {}, "outcome": {"ok": True, "produced": stance}},
            {"primitive": "grasp_single", "args": {}, "outcome": {"ok": False, "reason": "[grasp] knocked"}},
        ],
    )
    _write(
        run / "skills" / "agent_trace_01.jsonl",
        [
            {"primitive": "approach", "args": {}, "outcome": {"ok": True, "produced": dict(stance, depth_m=0.33)}},
            {"primitive": "grasp_single", "args": {}, "outcome": {"ok": True, "reason": ""}},
        ],
    )
    bottle = tmp_path / "bottle_x"
    _write(
        bottle / "collect" / "manifest_00.jsonl",
        [
            {"ok": True, "why": "", "park": 0.31},
            {"ok": False, "why": "[stand] far", "park": 0.55},
            {"ok": False, "why": "[grasp] x", "park": 0.36},
        ],
    )
    wh = tmp_path / "wh_x"
    _write(
        wh / "manifest_00.jsonl",
        [
            {"ok": True, "why": "completed", "stand_attempts": [{"depth_m": 0.44, "lateral_m": 0.02}]},
            {"ok": False, "control_reason": "reach_stalled", "stand_attempts": [{"depth_m": 0.50, "lateral_m": -0.01}]},
        ],
    )
    rows = rp.rows_from_skills_run(run) + rp.rows_from_bottle_run(bottle) + rp.rows_from_warehouse_run(wh)
    assert [(r["profile"], r["depth"], r["ok"]) for r in rows] == [
        ("bottle_simple", 0.40, False),
        ("bottle_simple", 0.33, True),
        ("bottle_simple", 0.31, True),
        ("bottle_simple", 0.36, False),
        ("warehouse_box", 0.44, True),
        ("warehouse_box", 0.50, False),
    ]
    bottle_rows = [r for r in rows if r["profile"] == "bottle_simple"]
    open_band = rp.evaluate(bottle_rows, {})
    assert (open_band["false_reject"], open_band["admitted_fail"]) == (0, 2)
    tight = rp.evaluate(bottle_rows, {"depth": (None, 0.38)})
    assert (tight["false_reject"], tight["admitted_fail"]) == (0, 1)
    text = rp.report(rows)
    assert "## bottle_simple: 4 rounds (2 ok, 2 failed)" in text and "## warehouse_box" in text
    assert (
        rp.main(["--skills", str(run), "--bottle", str(bottle), "--warehouse", str(wh), "--out", str(tmp_path / "out")])
        == 0
    )
    assert (tmp_path / "out" / "report.md").exists() and len(
        (tmp_path / "out" / "rows.jsonl").read_text().splitlines()
    ) == 6

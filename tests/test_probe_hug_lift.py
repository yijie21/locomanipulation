"""The hug-lift height probe's summary (scripts/rsi/probe_hug_lift.py) and its capability-table section
(capability_write.hug_lift), on hand-written round records: no simulator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts" / "rsi"), str(REPO / "src")]

import capability_write as W  # noqa: E402
import probe_hug_lift as P  # noqa: E402

BOTH = ["left", "right"]


def rung(z, under, touching=BOTH, dz=(0.1, -0.2)):
    return dict(step="rung", target_z=z, jaw_z=z, under_z=under, touching=touching, fell=False, jaw_dz_cm=0.0,
                left_jaw_err_cm=5.0, right_jaw_err_cm=6.0, left_dz_cm=dz[0], right_dz_cm=dz[1], left_dxy_cm=5.0,
                right_dxy_cm=6.0, attitude_err_deg=2.0, ik_err_cm=0.0, slip_cm=1.0, slip_dz_cm=-1.0, tilt_deg=30.0,
                box_rise_cm=10.0, hand_gap_cm=56.0, pelvis_z=0.7)  # fmt: skip


def write(out: Path, mass: float, label: str, hugged: list, rungs: list, top=None):
    d = P.mass_dir(out, mass)
    d.mkdir(parents=True, exist_ok=True)
    steps = [dict(step="hugged", ok=True, pick_ok=hugged == BOTH, touching=hugged), *rungs,
             dict(step="ladder_top", rung=top, jaw_z=top, under_z=None)]  # fmt: skip
    (d / f"{label}.json").write_text(json.dumps(dict(task="lift_box", steps=steps, fell=False)))


def test_summary_counts_held_rungs_and_the_lower_hand(tmp_path):
    write(tmp_path, 0.3, "lift_box_s1", BOTH, [rung(0.8, 0.5), rung(0.9, 0.6, dz=(-2.0, 0.5))], top=0.9)
    write(tmp_path, 0.3, "lift_box_s2", BOTH, [rung(0.8, 0.5), rung(0.9, 0.37, touching=[])], top=0.8)
    write(tmp_path, 0.3, "lift_box_s3", ["left"], [])
    t = P.summarize(tmp_path, [0.3], ["lift_box_s1", "lift_box_s2", "lift_box_s3"])["table"]["lift_box"]["0.3"]
    assert (t["rounds"], t["hugged"]) == (3, 2)
    assert t["per_rung"]["0.80"]["held"] == 2 and t["per_rung"]["0.90"]["held"] == 1
    assert t["per_rung"]["0.90"]["reached"] == 2
    assert t["per_rung"]["0.90"]["hand_dz_cm"]["max"] == -0.2  # the lower hand's miss per round: -2.0 and -0.2
    assert sorted(t["top_rung"]) == [0.8, 0.9]


def test_capability_section_one_entry_per_library(tmp_path):
    for sub, lib in (("hug_lift", "main"), ("hug_lift_x", "/a/fam_s1/incumbents/abc/skills")):
        out = tmp_path / sub
        write(out, 0.8, "lift_box_s1", BOTH, [rung(0.8, 0.5)], top=0.8)
        s = dict(library=lib, **P.summarize(out, [0.8], ["lift_box_s1"]))
        (out / "summary.json").write_text(json.dumps(s))
    sec = W.hug_lift(tmp_path)
    assert sec["status"] == "measured_one"
    assert set(sec["by_library"]) == {"main", "abc (fam_s1)"}
    row = sec["by_library"]["main"]["by_object"]["box"]["0.8 kg"]
    assert row["hugged"] == "1/1" and row["rungs"]["0.80"]["held"] == "1/1"


def test_capability_section_untested_without_a_run(tmp_path):
    assert W.hug_lift(tmp_path)["status"] == "untested"

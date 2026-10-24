"""TODO 33.24.1 / 33.24.2 / 33.1.4: the movable scene variant's geom ownership, the push probe's summary, and the reach
error vectors taken from the primitive call log."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "rsi"), str(ROOT / "src")]

import movable_scene as MS  # noqa: E402
import push_probe as PP  # noqa: E402
import reach_vectors as RV  # noqa: E402


def test_a_geom_belongs_to_its_key_and_not_to_a_longer_one():
    k = "BlackKitchenChair"
    for name in ("adt_fix_BlackKitchenChair", "adt_fixcol_BlackKitchenChair", "adt_upcol_BlackKitchenChair_012",
                 "adt_slab_BlackKitchenChair__0", "adt_panel_BlackKitchenChair__0_px", "adt_obj_StepStool_col"):  # fmt: skip
        assert MS.owned(name, "StepStool" if "StepStool" in name else k), name
    for name in ("adt_fix_BlackKitchenChair_B", "adt_slab_BlackKitchenChair_B__0", "adt_upcol_BlackKitchenChair_B_000",
                 "adt_fixcol_ExtendableTable", None):  # fmt: skip
        assert not MS.owned(name, k), name


def test_the_probe_calls_a_method_working_when_most_trials_move_it_without_a_fall():
    rows = [dict(key="c", method="hands", fell=False, moved_m=0.2), dict(key="c", method="hands", fell=False, moved_m=0.18),
            dict(key="c", method="hands", fell=True, moved_m=0.3), dict(key="c", method="body", fell=False, moved_m=0.05)]  # fmt: skip
    s = PP.summary(rows)
    assert s["c / hands"]["works"] and s["c / hands"]["fell"] == 1 and s["c / hands"]["moved"] == 2
    assert not s["c / body"]["works"]


def test_reaches_come_from_the_call_log_in_the_pelvis_heading_frame():
    t = np.arange(0.0, 3.0, 0.02)
    n = len(t)
    yaw = math.pi / 2  # facing +y
    robot = np.tile(np.r_[0.0, 0.0, 0.78, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], (n, 1))
    wrists = np.zeros((n, 2, 3))
    rot = np.eye(3)  # level: the jaw centre is the wrist + 0.1515 m along world x
    jaw = np.array([0.1, 0.5, 0.9])
    wrists[:, 1] = jaw - rot @ RV.PAD_OFF + np.array([0.0, -0.02, 0.0])  # the right wrist lands 2 cm short (world -y)
    z = dict(t=t, robot=robot, wrists=wrists)
    rec = dict(calls=[dict(name="walk_to", t0=0.0, t1=1.0, args=[[0, 0], None]),
                      dict(name="reach", t0=1.0, t1=2.0, args=["right", jaw.tolist(), rot.tolist()]),
                      dict(name="reach", t0=2.0, t1=2.5, args=["both", [jaw.tolist(), jaw.tolist()]])])  # fmt: skip
    rows = RV.segments(z, rec)
    assert len(rows) == 1  # one one-hand reach; the walk and the two-hand reach give none
    r = rows[0]
    assert r["hand"] == 1.0 and r["att"] == 0.0
    assert math.isclose(r["dx"], -0.02, abs_tol=1e-6)  # 2 cm short of a target ahead, in the heading frame
    assert math.isclose(r["dy"], 0.0, abs_tol=1e-6) and math.isclose(r["dz"], 0.0, abs_tol=1e-6)
    assert RV.segments(z, dict(steps=[dict(step="pregrasp", t0=1.0, t1=2.0, jaw_err_cm=2.0)])) == []  # no call log

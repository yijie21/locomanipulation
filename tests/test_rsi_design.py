"""TODO 33.19: design probes -- a spec's checks, the forced starts of each factor, and the driver's handling (the limit
per iteration, the results' destinations)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"


def _isolated(name: str, path: Path, names: tuple[str, ...]):
    saved, old_path = {k: sys.modules.pop(k, None) for k in names}, list(sys.path)
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = old_path
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


DS = _isolated("scene_rsi_design_test", HERE / "design.py", ("batch", "family", "design"))
META = {
    "surfaces": {"T": {"xy": [0.0, 0.0], "half_extent": [0.5, 0.3, 0.01], "yaw": 0.0}},
    "objects": {
        "Mug": {"surface": "T", "box": {"half_extent": [0.04, 0.04, 0.05]}, "init_pos": [0.0, 0.0, 0.8], "init_quat": [1, 0, 0, 0]},
        "Box": {"surface": "T", "box": {"half_extent": [0.15, 0.03, 0.05]}, "init_pos": [0.3, 0.0, 0.8],
                "init_quat": [1, 0, 0, 0]},
    },
}  # fmt: skip


def row(robot=(2.0, 0.0), yaw=0.0, mug=(0.0, 0.0), box=(0.3, 0.0)):
    return dict(task="pick_Mug", start=dict(seed=1, robot_xy=list(robot), robot_yaw=yaw, placements={
        "Mug": [[*mug, 0.8], [1, 0, 0, 0]], "Box": [[*box, 0.8], [1, 0, 0, 0]]}))  # fmt: skip


def test_a_spec_is_checked_before_anything_runs():
    tasks = ["a", "b"]
    assert DS.check(dict(factor="spawn_bearing", levels=[0, 90], starts=2), tasks) is None
    assert "factor" in DS.check(dict(factor="lighting", levels=[0]), tasks)
    assert "levels" in DS.check(dict(factor="spawn_bearing", levels=[]), tasks)
    assert "param" in DS.check(dict(factor="param", name="pick.NOPE", levels=[1]), tasks, {"pick.X": 1})
    assert DS.check(dict(factor="param", name="pick.X", levels=[1, 2]), tasks, {"pick.X": 1}) is None
    assert "unknown tasks" in DS.check(dict(factor="spawn_bearing", levels=[0], tasks=["z"]), tasks)
    assert "rounds" in DS.check(dict(factor="spawn_bearing", levels=[0, 1, 2, 3, 4, 5, 6, 7], starts=20), tasks)


def test_spawn_bearing_turns_the_robot_in_place():
    for lv in (0.0, 90.0, 180.0):
        s = DS.transform("spawn_bearing", lv, row(), "Mug", META)["start"]
        assert s["robot_xy"] == [2.0, 0.0]
        to_obj = math.pi  # the mug lies in -x from the robot
        off = (to_obj - s["robot_yaw"] + math.pi) % (2 * math.pi) - math.pi
        assert math.degrees(off) == lv if lv != 180.0 else abs(math.degrees(off)) == 180.0


def test_spawn_distance_keeps_the_object_where_it_was_in_view():
    r = row(robot=(2.0, 0.0), yaw=math.pi - 0.3)  # the mug 0.3 rad to the robot's left... seen at bearing +0.3
    s = DS.transform("spawn_distance", 1.0, r, "Mug", META)["start"]
    assert np.allclose(s["robot_xy"], [1.0, 0.0])
    to_obj = math.atan2(0.0 - 0.0, 0.0 - 1.0)
    assert math.isclose((s["robot_yaw"] - to_obj + math.pi) % (2 * math.pi) - math.pi, -0.3, abs_tol=1e-9)

    class Occ:
        def is_free(self, xy):
            return xy[0] > 1.5

    class Nav:
        def set_goal_ring(self, *a):
            pass

        def reachable(self, xy):
            return True

    assert DS.transform("spawn_distance", 1.0, r, "Mug", META, (Occ(), Nav())) is None  # 1.0 m out is inside furniture
    assert DS.transform("spawn_distance", 1.8, r, "Mug", META, (Occ(), Nav())) is not None


def test_object_yaw_turns_in_place_unless_it_hits_a_neighbour_or_leaves_the_top():
    s = DS.transform("object_yaw", 90.0, row(box=(0.4, 0.2)), "Mug", META)["start"]
    q = s["placements"]["Mug"][1]
    assert math.isclose(2 * math.atan2(q[3], q[0]), math.pi / 2, abs_tol=1e-6)
    assert s["placements"]["Mug"][0] == [0.0, 0.0, 0.8]
    # a long box beside the mug: turned 90 deg the box itself would collide -- turn the box instead of the mug
    r = row(mug=(0.0, 0.0), box=(0.08, 0.0))
    r["task"] = "pick_Box"
    assert DS.transform("object_yaw", 90.0, r, "Box", META) is None
    # the long box at the top's end: turned 90 deg it would stick out of the top
    r = row(mug=(-0.3, 0.0), box=(0.32, 0.0))
    assert DS.transform("object_yaw", 0.0, r, "Box", META) is not None
    assert DS.transform("object_yaw", 90.0, dict(r, start=dict(r["start"], placements={
        "Mug": r["start"]["placements"]["Mug"], "Box": [[0.32, 0.2, 0.8], [1, 0, 0, 0]]})), "Box", META) is None  # fmt: skip


def test_the_driver_runs_at_most_two_and_records_where_results_go(tmp_path, monkeypatch):
    lp = _isolated("scene_rsi_loop_design_test", HERE / "loop.py",
                   ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "records", "design", "loop"))  # fmt: skip
    monkeypatch.setattr(lp, "STATE", tmp_path / "fam" / "loop")
    monkeypatch.setattr(lp, "LOOP", tmp_path / "fam")
    (tmp_path / "fam" / "loop").mkdir(parents=True)
    rows, ran = [], []
    monkeypatch.setattr(lp, "ledger", rows.append)
    monkeypatch.setattr(lp, "parameters_of", lambda inc: {"pick.X": 0.3})
    monkeypatch.setattr(lp.F, "load", lambda fam: {"tasks": ["t"]})
    monkeypatch.setattr(lp, "FAMILY", "fam")

    def fake_run(spec, family, pool, skills, overrides, out, procs, gpus):
        ran.append(spec["factor"])
        return dict(factor=spec["factor"], levels=[dict(level=x, n=2, ok=1) for x in spec["levels"]], dir=str(out))

    monkeypatch.setattr(lp.DS, "run", fake_run)
    d = dict(factor="spawn_bearing", levels=[0, 180], starts=1)
    hyps = dict(hypotheses=[dict(id="H1", design=d), dict(id="H2", design=dict(d, factor="param", name="pick.X")),
                            dict(id="H3", design=d), dict(id="H4", design=dict(d, factor="nope"))])  # fmt: skip
    it = tmp_path / "fam" / "loop" / "iter_02"
    it.mkdir()
    out = lp.run_designs(hyps, dict(hash="h", dir=str(tmp_path), set={}), it, 2)
    assert ran == ["spawn_bearing", "param"]
    assert [r.get("error") for r in out][2:] == ["over the limit of design probes per iteration"] * 2
    assert rows[-1]["step"] == "design" and json.loads((it / "designs.json").read_text())[0]["hypothesis"] == "H1"
    curves = [json.loads(x) for x in (tmp_path / "body" / "curves.jsonl").read_text().splitlines()]
    assert [c["hypothesis"] for c in curves] == ["H1", "H2"] and curves[0]["incumbent"] == "h"

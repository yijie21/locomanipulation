"""TODO 33.19: design probes -- a spec's checks, the forced starts of each factor, and the driver's handling (the limit
per iteration, the results' destinations)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import rsi_loop_ns as NS  # noqa: E402

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


DS = _isolated("scene_rsi_design_test", HERE / "scene" / "design.py", ("batch", "family", "design"))
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
        s = DS.transform("spawn_bearing", lv, row(), "Mug", META)[0]["start"]
        assert s["robot_xy"] == [2.0, 0.0]
        to_obj = math.pi  # the mug lies in -x from the robot
        off = (to_obj - s["robot_yaw"] + math.pi) % (2 * math.pi) - math.pi
        assert math.degrees(off) == lv if lv != 180.0 else abs(math.degrees(off)) == 180.0


def test_spawn_distance_keeps_the_object_where_it_was_in_view():
    r = row(robot=(2.0, 0.0), yaw=math.pi - 0.3)  # the mug 0.3 rad to the robot's left... seen at bearing +0.3
    s = DS.transform("spawn_distance", 1.0, r, "Mug", META)[0]["start"]
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

    assert DS.transform("spawn_distance", 1.0, r, "Mug", META, (Occ(), Nav())) == (None, {"why": "not_free"})
    assert DS.transform("spawn_distance", 1.8, r, "Mug", META, (Occ(), Nav()))[0] is not None


def test_object_yaw_turns_in_place_when_it_fits():
    s, info = DS.transform("object_yaw", 90.0, row(box=(0.4, 0.2)), "Mug", META)
    q = s["start"]["placements"]["Mug"][1]
    assert math.isclose(2 * math.atan2(q[3], q[0]), math.pi / 2, abs_tol=1e-6)
    assert s["start"]["placements"]["Mug"][0] == [0.0, 0.0, 0.8] and info == {"shift_m": 0.0}


def test_object_yaw_slides_a_turned_object_to_the_nearest_spot_that_fits(monkeypatch):
    """TODO 37.9.1: run D turned a 0.31 m spoon in place and left out 11 of 16 starts. A turned object now slides on
    its top (MAX_SHIFT_M at most) until it is on the top and clear of its neighbours; the start says how far."""
    # the long box near the top's back edge: turned 90 deg it sticks out by 0.07 m and slides back in
    r = row(mug=(-0.3, 0.0), box=(0.32, 0.2))
    s, info = DS.transform("object_yaw", 90.0, r, "Box", META)
    (x, y, z), q = s["start"]["placements"]["Box"]
    assert math.isclose(info["shift_m"], 0.07, abs_tol=1e-9) and (x, z) == (0.32, 0.8) and math.isclose(y, 0.13)
    fp = DS.ST.footprint(META["objects"]["Box"], (x, y), q)
    assert DS.ST.inside_top(fp, ((0.0, 0.0), np.array([0.5, 0.3]) - DS.ST.SURFACE_MARGIN_M, 0.0))
    # beside the mug it slides clear of it
    s, info = DS.transform("object_yaw", 90.0, row(mug=(0.0, 0.0), box=(0.08, 0.0)), "Box", META)
    assert s is not None and info["shift_m"] > 0
    # with no slide allowed, what the turn in place ran into
    monkeypatch.setattr(DS, "MAX_SHIFT_M", 0.0)
    assert DS.transform("object_yaw", 90.0, r, "Box", META) == (None, {"why": "off_top"})
    assert DS.transform("object_yaw", 90.0, row(mug=(0.0, 0.0), box=(0.08, 0.0)), "Box", META) == (
        None,
        {"why": "neighbour"},
    )
    # a top too narrow for the turned box anywhere
    narrow = json.loads(json.dumps(META))
    narrow["surfaces"]["T"]["half_extent"] = [0.5, 0.1, 0.01]
    monkeypatch.setattr(DS, "MAX_SHIFT_M", 0.15)
    assert DS.transform("object_yaw", 90.0, row(mug=(-0.3, 0.0), box=(0.3, 0.0)), "Box", narrow) == (
        None,
        {"why": "off_top"},
    )


def test_a_probe_reports_why_starts_were_left_out(tmp_path, monkeypatch):
    """TODO 37.9.1: each level says how many starts the forcing left out and why, how far objects slid, and the result
    names the levels that lost more than half their starts."""
    narrow = json.loads(json.dumps(META))
    narrow["surfaces"]["T"]["half_extent"] = [0.5, 0.1, 0.01]
    pool = tmp_path / "pool"
    pool.mkdir()
    for k, y in enumerate((0.0, 0.02)):  # the box's end 0.48 m out, at the top's end
        (pool / f"pick_Box_s{k}.json").write_text(
            json.dumps(dict(row(mug=(-0.3, 0.0), box=(0.33, y)), task="pick_Box"))
        )
    monkeypatch.setattr(DS.F, "load", lambda fam: dict(tasks=["pick_Box"], meta=narrow))
    monkeypatch.setattr(DS, "_object_of", lambda fam, task: "Box")
    ran = []
    monkeypatch.setattr(DS.B, "run_jobs", lambda jobs, *a: ran.extend(j.labels for j in jobs))
    spec = dict(factor="object_yaw", levels=[0, 10, 90], starts=2)
    res = DS.run(spec, "fam", pool, tmp_path / "skills", {}, tmp_path / "out", 1, "0")
    by = {p["level"]: p for p in res["levels"]}
    assert by[0]["skipped"] == 0 and by[0]["shift_cm_max"] == 0.0
    assert by[10]["skipped"] == 0 and by[10]["shift_cm_max"] > 0  # turned 10 deg the box slides to stay on
    assert by[90]["skipped"] == 2 and by[90]["skipped_why"] == {"off_top": 2} and "shift_cm_max" not in by[90]
    assert res["note"].startswith("levels 90 left out more than half of the 2 starts (90: off_top 2)")
    assert len(ran) == 2  # the empty level runs nothing


def test_the_driver_runs_at_most_two_and_records_where_results_go(tmp_path, monkeypatch):
    lp = NS.fresh_loop()  # fmt: skip
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
    # the run's own curves (TODO 33.1.7): not body/curves.jsonl, the snapshot skills read for the whole run
    assert not (tmp_path / "body" / "curves.jsonl").exists()
    curves = [json.loads(x) for x in (tmp_path / "body" / "curves_run.jsonl").read_text().splitlines()]
    assert [c["hypothesis"] for c in curves] == ["H1", "H2"] and curves[0]["incumbent"] == "h"
    assert [c["scope"] for c in curves] == ["start", "skill"] and all(c["machine"] for c in curves)


def test_a_primitive_probe_checks_its_levels_and_runs_the_loco_cells_alone(tmp_path, monkeypatch):
    """TODO 33.19.1: one motor primitive alone on the empty floor, a deterministic run per level."""
    assert DS.check(dict(factor="primitive", name="turn", levels=[45, -90, 180]), ["t"]) is None
    assert "one of walk" in DS.check(dict(factor="primitive", name="jump", levels=[1]), ["t"])
    assert "within 0.1 to 4" in DS.check(dict(factor="primitive", name="walk", levels=[0.5, 6.0]), ["t"])
    assert DS.primitive_cells("turn", [90.0]) == [{"id": "turn_L0", "op": "turn", "deg": 90.0}]
    assert DS.primitive_cells("side", [-0.2]) == [{"id": "side_L0", "op": "side", "d": -0.2}]
    calls = []

    def fake_sub(cmd, cwd, env, stdout, stderr, check):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True)
        for c in json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text()):
            fell = c["deg"] == 180.0
            (out / f"{c['id']}.json").write_text(
                json.dumps(dict(ok=not fell, fell=fell, pos_err_cm=3.0, yaw_err_deg=2.0))
            )

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_sub)
    monkeypatch.setattr(DS.B, "SIM_SLOTS", 0)
    res = DS.run(dict(factor="primitive", name="turn", levels=[90, 180]), "fam", tmp_path, tmp_path, {}, tmp_path / "o",
                 1, "1")  # fmt: skip
    assert calls and calls[0][2:4] == ["run-cells", "loco"]
    assert [(p["level"], p["ok"], p["fell"], p["pos_err_cm"]) for p in res["levels"]] == [(90.0, 1, False, 3.0),
                                                                                          (180.0, 0, True, 3.0)]  # fmt: skip
    assert res["factor"] == "primitive" and res["tasks"] == []

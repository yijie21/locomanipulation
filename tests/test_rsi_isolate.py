"""TODO 33.20: whose failure is it -- the motor's call log, the isolation tool's choice of the failing command and its
verdict, and the driver's escalation rows (once per incumbent and round)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
sys.path.insert(0, str(ROOT / "src"))

import rsi_loop_ns as NS  # noqa: E402

from locomani.backends.scalebfm import motor as M  # noqa: E402


def _isolated(name: str, path: Path, names: tuple[str, ...]):
    """`path` under its own module name with scripts/rsi on the path while it imports; sys.modules and sys.path are
    left as they were (the warehouse experiment has a loop.py and a batch.py of its own)."""
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


class _Robot:
    """The parts of Motor that _logged touches: a clock, the call log and its nesting depth."""

    def __init__(self):
        self.t, self.calls, self._call_depth = 0.0, [], 0

    @M._logged
    def walk_to(self, xy, yaw=None, *, speed=None):
        self.t += 2.0
        return M.Outcome(False, "fell at 2.00 s") if xy[0] > 5 else M.Outcome(True)

    @M._logged
    def turn(self, yaw):
        return self.walk_to(np.array([0.0, 0.0]), yaw)


def test_the_call_log_records_outer_calls_with_plain_arguments():
    r = _Robot()
    r.turn(1.5)
    r.walk_to(np.array([9.0, 1.0]), speed=np.float64(0.25))
    assert [c["name"] for c in r.calls] == ["turn", "walk_to"]  # turn's own walk_to is part of the turn
    assert r.calls[0]["args"] == [1.5] and (r.calls[0]["t0"], r.calls[0]["t1"]) == (0.0, 2.0)
    assert r.calls[1]["args"] == [[9.0, 1.0]] and r.calls[1]["kwargs"] == {"speed": 0.25}
    assert (r.calls[1]["ok"], r.calls[1]["why"]) == (False, "fell at 2.00 s")
    json.dumps(r.calls)  # plain JSON


def _iso():
    return _isolated(
        "scene_rsi_isolate_test", HERE / "scene" / "isolate.py", ("batch", "family", "scene_round", "isolate")
    )


def test_the_suspect_is_the_call_that_fell_else_the_first_that_failed():
    iso = _iso()
    calls = [dict(name="hold", ok=True, why=""), dict(name="walk_to", ok=False, why="stopped outside tolerance"),
             dict(name="walk_to", ok=False, why="fell at 7.94 s")]  # fmt: skip
    assert iso.suspect(calls, fell=True) == 2
    assert iso.suspect(calls, fell=False) == 1
    assert iso.suspect([dict(name="hold", ok=True)], fell=False) is None
    assert iso.verdict(dict(fell=True), dict(fell=True)) == "motor"
    assert iso.verdict(dict(fell=True), dict(fell=False)) == "skill"
    assert iso.verdict(dict(fell=False), dict(fell=False, call_ok=False)) == "motor"


def test_escalations_are_written_once_per_incumbent_and_round(tmp_path, monkeypatch):
    lp = NS.fresh_loop()  # fmt: skip
    state, loop = tmp_path / "fam" / "loop", tmp_path / "fam"
    state.mkdir(parents=True)
    monkeypatch.setattr(lp, "STATE", state)
    monkeypatch.setattr(lp, "LOOP", loop)
    row = dict(label="pick_Mug_2_s410003", task="pick_Mug_2", call=dict(name="walk_to", args=[[-2.96, -2.67], None]),
               original=dict(failure_class="fell_walk", fell=True), isolated=dict(fell=True, call_why="fell at 5.76 s"),
               verdict="motor")  # fmt: skip
    lp.record_escalations(dict(hash="5cba"), 3, [row])
    lp.record_escalations(dict(hash="5cba"), 4, [row])  # the next iteration, same incumbent: nothing new
    lp.record_escalations(dict(hash="9999"), 5, [row])  # a new incumbent fails it too: one more
    esc = [json.loads(x) for x in (state / "escalations.jsonl").read_text().splitlines()]
    con = [json.loads(x) for x in (tmp_path / "body" / "constraints.jsonl").read_text().splitlines()]
    assert [(e["incumbent"], e["iter"]) for e in esc] == [("5cba", 3), ("9999", 5)]
    assert esc[0]["command"] == "walk_to([-2.96, -2.67], null)" and esc[0]["rule"] == 20
    assert len(con) == 2 and con[0]["kind"] == "isolated_failure"


def test_attribution_runs_only_for_a_recurring_class(tmp_path, monkeypatch):
    lp = NS.fresh_loop()  # fmt: skip
    monkeypatch.setattr(lp, "STATE", tmp_path / "loop")
    monkeypatch.setattr(lp, "LOOP", tmp_path)
    (tmp_path / "loop").mkdir()
    ran, rows = [], []
    monkeypatch.setattr(lp.subprocess, "run", lambda *a, **k: ran.append(a) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(lp, "ledger", rows.append)
    one = {"a": dict(ok=False, failure_class="fell_walk"), "b": dict(ok=False, failure_class="side_reach")}
    monkeypatch.setattr(lp.B, "outcomes", lambda base: one)
    assert lp.attribute(dict(hash="h"), tmp_path, 1, tmp_path, stub=False) is None and not ran
    two = dict(one, c=dict(ok=False, failure_class="fell_walk"))
    monkeypatch.setattr(lp.B, "outcomes", lambda base: two)
    got = lp.attribute(dict(hash="h"), tmp_path, 1, tmp_path, stub=False)
    assert ran and got["classes"] == ["fell_walk"] and rows[-1]["step"] == "attribution"
    assert lp.attribute(dict(hash="h"), tmp_path, 1, tmp_path, stub=True) is None

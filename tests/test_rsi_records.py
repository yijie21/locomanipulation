"""TODO 33.15.1: the record library -- the rounds the roles may see, one row each with their measured numbers -- and the
verdict a hypothesis's `measure` gets from it when its probe could not decide it."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import rsi_loop_ns as NS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"


def _isolated(name: str, path: Path, names: tuple[str, ...]):
    """`path` under its own module name, with scripts/rsi on the path while it imports; sys.modules and sys.path are
    left as they were (the warehouse experiment has a loop.py and a batch.py of its own; test_rsi_speed does the same)."""
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


RC = _isolated("scene_rsi_records_test", HERE / "scene" / "records.py", ("records",))

V = ("abc", "{}", "m1")


def rows(spec):
    """[(n, ok, feature value)] -> rows of the incumbent's version on one machine."""
    out, i = [], 0
    for n, ok, x in spec:
        for _ in range(n):
            out.append(dict(label=f"t_s{i}", task="t", ok=int(ok), skills="abc", set="{}", motor="m1", machine="h",
                            **{"events.gap_cm": x}))  # fmt: skip
            i += 1
    return out


def test_fisher_matches_known_values():
    assert RC.fisher(2, 2, 0, 32) == pytest.approx(6 / 630)  # the plan's mug example, 2/4 against 0/32: C(4,2)/C(36,2)
    assert RC.fisher(5, 5, 5, 5) == pytest.approx(1.0)
    assert RC.fisher(10, 0, 0, 10) == pytest.approx(2 / math.comb(20, 10))


def test_the_three_verdicts():
    m = dict(feature="events.gap_cm", op=">=", value=5.0, expect="fails")
    clear = rows([(8, 0, 9.0), (2, 1, 9.0), (2, 0, 1.0), (18, 1, 1.0)])
    assert RC.judge(clear, m, V, "h")["verdict"] == "verified"
    none = rows([(6, 0, 9.0), (6, 1, 9.0), (6, 0, 1.0), (6, 1, 1.0)])
    assert RC.judge(none, m, V, "h")["verdict"] == "refuted"
    few = rows([(1, 0, 9.0), (1, 1, 9.0), (3, 1, 1.0)])
    assert RC.judge(few, m, V, "h")["verdict"] == "hypothesis"
    # the opposite of what the hypothesis says, clearly: refuted
    opposite = rows([(12, 1, 9.0), (12, 0, 1.0)])
    assert RC.judge(opposite, m, V, "h")["verdict"] == "refuted"
    # another version or another machine does not count under the incumbent scope
    other = [dict(r, skills="zzz") for r in clear]
    assert RC.judge(other, m, V, "h")["verdict"] == "hypothesis"
    assert RC.judge(clear, m, V, "elsewhere")["verdict"] == "hypothesis"
    assert RC.judge(other, dict(m, scope="all"), V, "h")["verdict"] == "verified"
    assert RC.judge(clear, dict(m, op="~"), V, "h")["verdict"] == "hypothesis"


def _round(d: Path, label: str, ok: bool, skills="abc", **events):
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(label=label, task="pick_Mug_2", ok=ok, steps=[dict(step="pregrasp", ok=ok, jaw_err_cm=2.0 if ok else 9.0)],
               skills=skills, set={}, motor="m1", events=dict(events), failure_class="ok" if ok else "side_reach")  # fmt: skip
    (d / f"{label}.json").write_text(json.dumps(rec))


def test_a_sandbox_scan_sees_only_what_the_roles_may_see(tmp_path):
    sb = tmp_path / "sb"
    (sb / "third_party").parent.mkdir(parents=True, exist_ok=True)
    (sb / "third_party").symlink_to(tmp_path)
    fam = sb / "outputs" / "scene_rsi_20261016" / "grasp_atomic"
    _round(fam / "diag" / "incumbent_abc", "pick_Mug_2_s410001", True, reach_gap_cm=1.0)
    _round(fam / "loop" / "iter_03" / "probe" / "H1", "pick_Mug_2_s410001", True)  # the same round again: counted once
    _round(fam / "diag" / "incumbent_abc", "pick_Mug_2_s410002", False, reach_gap_cm=9.0)
    hidden = sb / "outputs" / "scene_rsi_20261016_hidden" / "grasp_atomic" / "gate" / "01_x" / "candidate"
    _round(hidden, "pick_Mug_2_s510001", False)
    (fam / "pool_diag").mkdir(parents=True)
    start = dict(task="pick_Mug_2", start=dict(seed=410001, robot_xy=[0.0, 0.0], robot_yaw=0.0,
                                               placements={"Mug_2": [[1.0, 1.0, 0.8], [1, 0, 0, 0]]}))  # fmt: skip
    (fam / "pool_diag" / "pick_Mug_2_s410001.json").write_text(json.dumps(start))
    scratch = tmp_path / "scratch"
    _round(scratch / "rsi_scene_sb" / "reviewer" / "try1", "pick_Mug_2_s410090", True, skills="other")
    got = RC.load(sb, scratch)
    assert sorted((r["label"], r["skills"]) for r in got) == [
        ("pick_Mug_2_s410001", "abc"), ("pick_Mug_2_s410002", "abc"), ("pick_Mug_2_s410090", "other")]  # fmt: skip
    one = next(r for r in got if r["label"] == "pick_Mug_2_s410001")
    assert one["start.spawn_dist_m"] == pytest.approx(math.sqrt(2), abs=1e-3)
    assert one["start.spawn_bearing_deg"] == pytest.approx(45.0)
    assert (one["events.reach_gap_cm"], one["pregrasp.jaw_err_cm"], one["pregrasp.ok"]) == (1.0, 2.0, 1)
    assert RC.version_of(fam / "diag" / "incumbent_abc") == ("abc", "{}", "m1")


def test_build_leaves_out_the_hidden_seed_ranges(tmp_path, monkeypatch):
    sb = tmp_path / "rsi-scene" / "run"
    fam = sb / "outputs" / "scene_rsi_20261016" / "grasp_atomic"
    _round(fam / "diag" / "incumbent_abc", "pick_Mug_2_s410001", True)
    _round(fam / "loop" / "iter_01" / "cand_1" / "smoke", "pick_Mug_2_s610004", True)  # a sealed-range start
    out = tmp_path / "lib.jsonl"
    res = RC.build([sb], out, tmp_path / "scratch")
    assert (res["rows"], res["left_out_hidden"]) == (1, 1)
    assert [json.loads(x)["label"] for x in out.read_text().splitlines()] == ["pick_Mug_2_s410001"]


def test_the_driver_asks_the_library_only_when_the_probe_could_not_decide(tmp_path, monkeypatch):
    lp = NS.fresh_loop()  # fmt: skip
    lib = rows([(8, 0, 9.0), (2, 1, 9.0), (2, 0, 1.0), (18, 1, 1.0)])
    monkeypatch.setattr(lp.RC, "load", lambda *a, **k: lib)
    monkeypatch.setattr(lp.RC, "version_of", lambda base: V)
    monkeypatch.setattr(lp.RC, "machine", lambda: "h")
    m = dict(feature="events.gap_cm", op=">=", value=5.0, expect="fails")
    hyps = dict(hypotheses=[dict(id="H1", status="hypothesis", measure=m), dict(id="H2", status="refuted", measure=m),
                            dict(id="H3", status="hypothesis")])  # fmt: skip
    pr = [dict(id="H1", status="hypothesis"), dict(id="H2", status="refuted"), dict(id="H3", status="hypothesis")]
    lp.records_statuses(hyps, pr, tmp_path)
    assert [h["status"] for h in hyps["hypotheses"]] == ["verified", "refuted", "hypothesis"]
    assert pr[0]["status"] == "verified" and pr[0]["records"]["n_in"] == 10
    assert "records" not in pr[1] and "records" not in pr[2]

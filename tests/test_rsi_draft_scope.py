"""TODO 44 (rule 38): a draft's declared scope and measured footprint."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
LOOP_NAMES = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "records", "design", "loop")


@pytest.fixture
def lp():
    saved, old_path = {k: sys.modules.pop(k, None) for k in LOOP_NAMES}, list(sys.path)
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    try:
        spec = importlib.util.spec_from_file_location("rsi_scope_loop", HERE / "loop.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rsi_scope_loop"] = mod
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = old_path
        sys.modules.pop("rsi_scope_loop", None)
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _rec(ok: bool, *steps: tuple) -> dict:
    return dict(ok=ok, t_end=float(len(steps)), wall_s=1.0,
                steps=[dict(step=n, ok=o, t0=float(i), t1=float(i) + d, cm=1.5) for i, (n, o, d) in enumerate(steps)])  # fmt: skip


WALK, GRIP, MISS = ("walk", True, 1.0), ("grip", True, 1.0), ("grip", False, 1.0)


def _rounds(d: Path, **recs) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for label, r in recs.items():
        (d / f"{label}.json").write_text(json.dumps(r))
    return d


def test_footprint_tells_rounds_the_code_reached_from_rounds_it_did_not(lp, tmp_path):
    a = _rounds(tmp_path / "a", s1=_rec(True, WALK, GRIP), s2=_rec(True, WALK, GRIP), s3=_rec(False, WALK, MISS))
    late = _rec(True, WALK, ("grip", True, 1.25))  # the same outcome, another timing: the code reached it
    wall = dict(_rec(True, WALK, GRIP), wall_s=99.0)  # wall time is not a difference
    b = _rounds(tmp_path / "b", s1=wall, s2=late, s3=_rec(False, WALK, MISS))
    fp = lp.footprint(a, b, dict(when="x", starts=["s3"]))
    assert fp["n"] == 3 and fp["changed"] == ["s2"] and fp["outside"] == ["s2"] and fp["missed"] == ["s3"]
    assert lp.footprint(a, b, dict(when="x", starts=["all"]))["outside"] == []
    assert lp.footprint(a, a, None)["changed"] == []


def test_first_difference_names_the_step_where_a_round_left_the_path(lp):
    d = lp.first_difference(_rec(True, WALK, GRIP), _rec(False, WALK, MISS))
    assert d["index"] == 1 and d["incumbent"]["ok"] is True and d["draft"]["ok"] is False
    short = lp.first_difference(_rec(True, WALK, GRIP), _rec(False, WALK))
    assert short["index"] == 1 and short["draft"] is None and short["incumbent"]["step"] == "grip"


@pytest.mark.parametrize("seen,killed", [(0, True), (1, False)], ids=["a new answer", "only the earlier answer"])
def test_only_a_new_answer_starts_the_grace_period(lp, monkeypatch, tmp_path, seen, killed):
    t = tmp_path / "t.jsonl"
    t.write_text(lp.ANSWERED + "\n")
    monkeypatch.setattr(lp, "transcript", lambda s: t)
    cmd = ["bash", "-c", "sleep 2 & wait"]
    out = lp.run_role(cmd, "", dict(PATH="/usr/bin:/bin"), "s", grace=0.0, poll=0.05, seen=seen)
    assert bool(out[3]) is killed


def test_the_manager_can_name_only_this_iterations_hypotheses(lp):
    sch = lp.manager_schema(True, ["AH1", "BH1"])["properties"]
    assert sch["hypothesis"]["enum"] == ["AH1", "BH1"] and sch["hypotheses"]["items"]["enum"] == ["AH1", "BH1"]
    assert "enum" not in lp.manager_schema(True)["properties"]["hypothesis"]


def test_an_engineer_must_declare_its_scope(lp):
    assert "scope" in lp.ENGINEER_SCHEMA["required"]
    text = (HERE / "prompts" / "engineer.md").read_text()
    assert "`scope`" in text and "rules.md 38" in text

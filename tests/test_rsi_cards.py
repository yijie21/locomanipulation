"""TODO 43 (rule 37): the failure cards, and how the inner loop hands them to the Reviewer instead of the round files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
LOOP_NAMES = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "records", "design", "loop")


def _isolated(name: str, path: Path, names: tuple[str, ...] = ()):
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


def _scene(path: Path) -> Path:
    meta = dict(
        objects=dict(
            Mug=dict(free=True, init_pos=[0.0, 0.0, 0.80], init_quat=[1, 0, 0, 0], semantic=dict(mass_kg=0.2),
                     box=dict(half_extent=[0.04, 0.04, 0.05], centre=[0, 0, 0])),
            Table=dict(init_pos=[0.0, 0.0, 0.37], init_quat=[1, 0, 0, 0],
                       box=dict(half_extent=[0.5, 0.5, 0.37], centre=[0, 0, 0])),
        ),
        surfaces=dict(top=dict(xy=[0.0, 0.0], half_extent=[0.5, 0.5], top=0.75, yaw=0.0, of_object="Table",
                               approach_edges=["-x"])),
        room=dict(bounds=dict(x=[-3.0, 3.0], y=[-3.0, 3.0])),
    )  # fmt: skip
    path.write_text(json.dumps(meta))
    return path


def _round(d: Path, label: str, ok: bool, stop_x: float, trace: bool = True) -> None:
    """A round of pick_Mug: walks to `stop_x`, reaches, grips; a failed one fails its grip with a too-wide jaw."""
    steps = [dict(step="walk", ok=True, t0=1.0, t1=3.0), dict(step="stance", ok=True, t0=3.0, t1=3.0, pelvis=[stop_x, 0.0]),
             dict(step="grip", ok=ok, t0=4.0, t1=5.0, jaw_err_cm=1.0 if ok else 5.0, **({} if ok else {"why": "jaw 5.0 cm off"}))]  # fmt: skip
    rec = dict(label=label, task="pick_Mug", ok=ok, why="" if ok else "grip", fell=False, dropped=False, t_end=6.0,
               failure_class=None if ok else "too_wide", steps=steps, calls=[dict(name="reach", args=["right"])])  # fmt: skip
    (d / f"{label}.json").write_text(json.dumps(rec))
    if not trace:
        return
    t = np.arange(0.0, 6.0, 0.02)
    n = len(t)
    robot = np.zeros((n, 7))
    robot[:, 0], robot[:, 2], robot[:, 3] = np.minimum(-1.5 + t * 0.3, stop_x), 0.75, 1.0
    wrists = np.zeros((n, 2, 3))
    wrists[:, :, 0], wrists[:, :, 2] = robot[:, 0, None] + 0.3, 0.9
    obj = np.tile([0.0, 0.0, 0.80, 1, 0, 0, 0], (n, 1))
    np.savez(d / f"{label}.npz", t=t, robot=robot, wrists=wrists, ref_wrists=wrists + 0.01, obj_Mug_object=obj,
             fingers=np.full((n, 4), 0.02), jaw=np.full((n, 2), 0.02), joints=np.zeros((n, 29)),
             mode=np.zeros(n, int), primitive=np.zeros(n, int), primitives=np.array(["reach"]))  # fmt: skip


@pytest.fixture
def diag(tmp_path):
    d = tmp_path / "fam" / "diag" / "incumbent_abc"
    d.mkdir(parents=True)
    _round(d, "pick_Mug_s1", False, -0.9)
    _round(d, "pick_Mug_s2", True, -0.7)
    _round(d, "pick_Mug_s3", False, -0.8, trace=False)
    (d / "_rule1_pick_Mug_s9.json").write_text("{}")  # a retired round is not a round
    return d


def _digests(out: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}


def test_one_card_per_failed_round_under_the_pool_summary(tmp_path, diag):
    cd = _isolated("rsi_cards", HERE / "cards.py")
    assert cd.build(diag, _scene(tmp_path / "scene_meta.json"), tmp_path / "out") == 2
    text = (tmp_path / "out" / "cards.md").read_text()
    assert text.startswith("# Failure cards: fam/incumbent_abc")
    assert "1/3 succeed" in text and "Failure classes: too_wide 2" in text and "By task: pick_Mug 1/3" in text
    assert text.count("\n### ") == 2
    assert "Image: img/pick_Mug_s1.png" in text
    assert "no trace" in text and "jaw 5.0 cm off" in text  # a round without a trace still says what failed
    assert [p.name for p in (tmp_path / "out" / "img").glob("*.png")] == ["pick_Mug_s1.png"]
    assert len((tmp_path / "out" / "cards.jsonl").read_text().splitlines()) == 2


def test_cards_are_deterministic(tmp_path, diag):
    cd = _isolated("rsi_cards", HERE / "cards.py")
    scene = _scene(tmp_path / "scene_meta.json")
    cd.build(diag, scene, tmp_path / "a")
    cd.build(diag, scene, tmp_path / "b")
    assert _digests(tmp_path / "a") == _digests(tmp_path / "b")


def test_a_blocked_stand_point_is_named_from_the_walk_map(tmp_path):
    cd = _isolated("rsi_cards", HERE / "cards.py")
    scene = cd.Scene(_scene(tmp_path / "scene_meta.json"))
    sf = dict(scene.surfaces["top"])
    edges = {e["edge"]: e for e in scene.edges(sf, np.array([0.0, 0.0, 0.8]), np.array([-0.9, 0.0]), "Table")}
    assert edges["-x"]["free"] and edges["-x"]["used"] and edges["-x"]["listed"]
    assert not edges["+x"]["listed"]


class Harness:
    """iterate() with everything around the Reviewer call stubbed, on a real round directory."""

    def __init__(self, tmp_path, monkeypatch, diag):
        self.lp = lp = _isolated("rsi_cards_loop", HERE / "loop.py", LOOP_NAMES)
        self.base, self.calls = diag, []
        scene = _scene(tmp_path / "scene_meta.json")
        state = tmp_path / "state"
        state.mkdir()
        hyps = dict(notes="", hypotheses=[])
        rows = self.rows = []

        def fake_claude(role, prompt, schema, it_dir, tools, stub=None, stop=None):
            self.calls.append(dict(role=role, prompt=prompt, readable=os.access(diag, os.R_OK)))
            if self.fail_in_role:
                raise lp.RoleFailed("the role died")
            return dict(hyps), dict(session="s", cost_usd=0.0, seconds=0)

        self.fail_in_role = False
        for k, v in dict(STATE=state, LOOP=diag.parents[1], FAMILY="fam", claude=fake_claude, ledger=rows.append,
                         snapshot=lambda n: None, incumbent=lambda: dict(hash="abc", dir=str(tmp_path), set={}),
                         put_skills=lambda p: None, changed_files=lambda: [], diag_incumbent=lambda inc, it: diag,
                         refresh_diag_pool=lambda *a: None, unpark=lambda *a: None, attribute=lambda *a: None,
                         summarize=lambda d: {}, parameters_of=lambda inc: {}, context=lambda inc, extra: {},
                         prompt_vars=lambda: {}, family_spec=lambda: dict(scene=str(scene)), role_tmp=lambda: tmp_path,
                         transcript_leaks=lambda s: [], after_review=lambda *a: "done").items():  # fmt: skip
            monkeypatch.setattr(lp, k, v)


def test_the_reviewer_gets_the_cards_and_cannot_read_the_rounds(tmp_path, monkeypatch, diag):
    h = Harness(tmp_path, monkeypatch, diag)
    mode = stat.S_IMODE(diag.stat().st_mode)
    assert h.lp.iterate(0, {}, False) == "done"
    call = h.calls[0]
    cards = h.lp.STATE / "iter_00" / "cards" / "cards.md"
    assert not call["readable"]
    assert str(cards) in call["prompt"] and "--ref" not in call["prompt"] and "{" + "CARDS}" not in call["prompt"]
    assert stat.S_IMODE(diag.stat().st_mode) == mode  # the Prober and the later roles read the rounds
    row = next(r for r in h.rows if r["step"] == "cards")
    assert row["iter"] == 0 and row["images"] == 1 and row["cards_bytes"] == cards.stat().st_size


def test_a_failed_reviewer_gives_the_rounds_back(tmp_path, monkeypatch, diag):
    h = Harness(tmp_path, monkeypatch, diag)
    h.fail_in_role = True
    with pytest.raises(h.lp.RoleFailed):
        h.lp.iterate(0, {}, False)
    assert os.access(diag, os.R_OK)


def test_a_crash_under_the_lock_is_repaired_at_the_next_start(tmp_path, monkeypatch, diag):
    lp = _isolated("rsi_cards_loop", HERE / "loop.py", LOOP_NAMES)
    monkeypatch.setattr(lp, "LOOP", diag.parents[1])
    other = diag.parent / "candidate_abc"
    other.mkdir()
    other.chmod(0)  # only an incumbent's rounds are ever locked
    diag.chmod(0)
    try:
        lp.unlock_rounds()
        assert os.access(diag, os.R_OK | os.X_OK) and (diag / "pick_Mug_s1.json").exists()
        assert not os.access(other, os.R_OK)
    finally:
        other.chmod(0o755)
        diag.chmod(0o755)


def test_a_broken_cards_run_stops_the_iteration_before_the_reviewer(tmp_path, monkeypatch, diag):
    h = Harness(tmp_path, monkeypatch, diag)
    monkeypatch.setattr(h.lp, "family_spec", lambda: dict(scene=str(tmp_path / "missing.json")))
    with pytest.raises(RuntimeError, match="cards.py failed"):
        h.lp.iterate(0, {}, False)
    assert h.calls == [] and os.access(diag, os.R_OK)


@pytest.mark.parametrize("family", ["grasp_atomic", "grasp_floor", "grasp_prep"])
def test_a_family_with_a_sandbox_list_keeps_the_cards(family):
    spec = json.loads((HERE / "families" / f"{family}.json").read_text())["sandbox"]
    assert "scripts/rsi/cards.py" in spec["keep"] and "cards" in spec["closure"]
    assert "scripts/rsi/digest.py" in spec["keep"]  # try.py --steps prints through it

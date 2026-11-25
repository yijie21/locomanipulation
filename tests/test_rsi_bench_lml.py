"""The LocoManiLab adapter's rows and cards (scripts/rsi/bench_lml.py) and its pools through the one benchmark backend. No simulator."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsi"))
import backend as BK  # noqa: E402
import backend_bench as BB  # noqa: E402
import bench_common as C  # noqa: E402
import bench_lml as LL  # noqa: E402
import bench_pools as P  # noqa: E402
import bench_rows as R  # noqa: E402

AD = BB.ADAPTERS["locomanilab"]


class L:  # the old backend_lml names over the modules they moved to
    to_round = staticmethod(LL.to_round)
    LEAK = LL.LEAK
    CLASS_SUGGEST = LL.CLASS_SUGGEST
    public = staticmethod(LL.public)
    card_text = staticmethod(LL.card_text)
    door_text = staticmethod(LL.door_text)
    FORBIDDEN_PATHS = LL.FORBIDDEN_PATHS
    BaseWriteRefused = C.BaseWriteRefused
    hidden_ids = staticmethod(P.hidden_ids)
    load = staticmethod(BB.load)
    layout = staticmethod(BB.layout)
    guard_write = staticmethod(BB.guard_write)
    load_skills = staticmethod(LL.load_skills)
    cards = staticmethod(lambda rounds, out: R.cards(rounds, out, LL.card_text))
    draw = staticmethod(lambda fam, pool, out, n=None, tasks=None: P.draw(fam, AD, pool, out, n, tasks))


class B:
    _success_rule = staticmethod(C.success_rule)
    _write = staticmethod(C.write_round)
    HIDDEN = C.HIDDEN


TASKS = ["HolodeckApproachDesk_scene117", "IThorKitchenAppleIntoBox_scene13"]
REAL = [json.loads(x) for x in (Path(__file__).parent / "lml_real_rows.jsonl").read_text().splitlines()]
SENTINELS = ("SENTINEL", "TARGETBODY", "7.777", "6.666")  # what the real rows hold in place of room-frame positions


def row(level: int, i: int = 0) -> str:
    return json.dumps({"seed": 100000 + i, "robot": 0.05, "objects": 0.02, "src": {"level": level}, "sha": "x"})


def make_practice(d: Path, tasks, per_level: int = 50) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        d.joinpath(f"{t}.jsonl").write_text("".join(row(lv, i) + "\n" for lv in (0, 1, 2) for i in range(per_level)))


def make_fam(tmp: Path, **kw) -> dict:
    root = tmp / "bed"
    seed = root / AD["seed"]
    seed.mkdir(parents=True)
    (seed / "__init__.py").write_text("EXTRA = dict(lift=0.34, nav_gap=0.3)\n")
    (seed / "skill.py").write_text("K_ORIG = dict(stand=0.27)\nK_CUT = dict(K_ORIG, exact=True)\n")
    practice = tmp / "practice"
    make_practice(practice, TASKS)
    fam = dict(
        name="lml_t",
        backend="bench",
        tasks=TASKS,
        pools={
            "diag": {"seed": 0, "n": 60},
            "accept": {"seed": 7, "n": 24, "per_level": 4},
            "sealed": {"seed": 7, "n": 42, "per_level": 7},
        },
        bench=dict(name="locomanilab", root=str(root)),
        track="camera",
        official_slots=2,
        practice_dir=str(practice),
        rules="rules_scene.md",
        prompts="prompts",
        caps={"min_net": 3},
        sandbox={"dir": "bed_readonly_copy", "task_files": []},
    )
    fam.update(kw)
    return fam


# ------------------------------------------------------------------------------------------- to_round
def rec(**kw) -> dict:
    base = dict(
        episode_id=f"{TASKS[1]}/Q/3",
        success=True,
        t_check=10.0,
        loss=None,
        fell=None,
        why=None,
        end="success",
        limit_s=15.0,
        marks={
            "before_close": {"t": 5, "rise_cm": 0, "moved_cm": 0.2, "tilt_deg": 1, "held": False},
            "after_lift": {"t": 8, "rise_cm": 9.0, "moved_cm": 9, "tilt_deg": 2, "held": True},
            "end": {"t": 10, "rise_cm": 9.0, "moved_cm": 9, "tilt_deg": 2, "held": True},
        },
        phases={"start": 1.0, "walk": 2.0, "reach_grasp": 2.0, "close": 1.0, "lift": 2.0, "open": 1.0},
        palm_err_cm=1.2,
        reach_ok=[True],
        t_end=10.0,
        wall_s=14.0,
        comply={"ok": True, "why": ""},
        genuine=True,
    )
    base.update(kw)
    return base


def rd(r: dict) -> dict:
    return L.to_round(r, f"{TASKS[1]}_s3", "abc", {"k": 1}, "sig")


def failing(o: dict) -> str:
    return next(s["step"] for s in o["steps"] if not s["ok"])


def test_win():
    o = rd(rec())
    assert o["ok"] is True and o["why"] == "" and o["failure_class"] == ""
    assert o["task"] == TASKS[1] and o["label"] == f"{TASKS[1]}_s3"
    assert o["set"] == {"k": 1} and o["skills"] == "abc" and o["motor"] == "sig" and o["rule"] == B._success_rule()
    assert [s["step"] for s in o["steps"]] == ["start", "walk", "reach_grasp", "close", "lift", "open"]
    assert all(s["ok"] for s in o["steps"]) and o["steps"][2]["t0"] == 3.0 and o["steps"][2]["t1"] == 5.0
    assert o["fell"] is False and o["dropped"] is False and o["plan_ok"] is True and o["lml"]["success"] is True
    assert o["t_end"] == 10.0 and "simple" not in o


@pytest.mark.parametrize("drop", ["comply", "genuine"])
def test_missing_comply_or_genuine_counts_as_false(drop):
    r = rec()
    del r[drop]
    o = rd(r)
    assert not o["ok"] and o["failure_class"] == "comply" and o["why"].startswith("comply: ")
    assert failing(o) == "open"  # the compliance class is blamed on the release (the lift when there is none)


def test_comply_false_and_genuine_false_texts():
    o = rd(rec(comply={"ok": False, "why": "no_hand"}))
    assert not o["ok"] and o["failure_class"] == "comply" and o["why"] == "comply: compliance no_hand"
    o = rd(rec(genuine=False))
    assert not o["ok"] and o["why"] == "comply: not genuine: the check held at the start state"
    # a loss class from the bed wins over "comply"
    o = rd(rec(comply={"ok": False, "why": "hand_on"}, loss="place"))
    assert o["failure_class"] == "place" and failing(o) == "open"


@pytest.mark.parametrize(
    "cls,phase",
    [
        ("navigate", "walk"),
        ("deliver", "reach_grasp"),
        ("grasp", "lift"),
        ("collide", "close"),
        ("perceive", "start"),
        ("place", "open"),
        ("fall", "open"),  # no phase of its own: the last one
        ("timeout", "open"),
    ],
)
def test_failure_classes(cls, phase):
    o = rd(
        rec(
            success=False,
            t_check=None,
            comply={"ok": False, "why": "not_success"},
            genuine=False,
            loss=cls,
            why="because",
        )
    )
    assert not o["ok"] and o["failure_class"] == cls and o["why"] == f"{cls}: because"
    assert failing(o) == phase and sum(not s["ok"] for s in o["steps"]) == 1
    assert o["plan_ok"] is False and o["plan_why"] == "because"


def test_phase_fallback_and_facts():
    ph = {"start": 1.0, "walk": 2.5}
    o = rd(rec(success=False, loss="grasp", phases=ph, stop_err_cm=5.5, walk_cm=3.0))
    assert failing(o) == "walk"  # no lift phase: the last one
    assert o["steps"][1]["stop_err_cm"] == 5.5 and o["steps"][1]["walk_cm"] == 3.0
    o = rd(rec(success=False, loss="deliver"))
    assert next(s for s in o["steps"] if s["step"] == "reach_grasp")["palm_err_cm"] == 1.2
    o = rd(rec(success=False, loss="place", obj_dest_end={"dxy_cm": [3.0, 4.0], "dz_cm": 0.5}))
    s = next(s for s in o["steps"] if s["step"] == "open")
    assert s["dist_cm"] == 5.0 and s["dz_cm"] == 0.5


def test_error_fell_dropped():
    o = rd(rec(success=False, loss="error", error="boom", why=None, phases={}))
    assert o["failure_class"] == "error" and o["why"] == "error: boom" and o["error"] == "boom" and o["steps"] == []
    assert o["plan_ok"] is False and o["plan_why"] == "boom"
    o = rd(rec(success=False, error="boom", loss=None))
    assert o["failure_class"] == "error"
    assert rd(rec(success=False, loss="fall", fell=3.2))["fell"] is True
    m = rec()["marks"]
    m["end"] = dict(m["end"], rise_cm=-20.0)
    assert rd(rec(success=False, loss="grasp", marks=m))["dropped"] is True
    assert rd(rec(success=True, marks=m))["dropped"] is False
    assert rd(rec(success=False, loss=None))["failure_class"] == "unclassified"


def test_t_end_from_steps_and_json():
    r = rec(success=False, loss="timeout")
    del r["t_end"]
    r["steps"] = 500
    o = rd(r)
    assert o["t_end"] == 10.0
    json.dumps(o)


def test_real_rows():
    win, lost = (L.to_round(r, r["episode_id"].replace("/Q/", "_s"), "h", {}, "s") for r in REAL)
    assert win["ok"] and win["task"] == "IThorBathroomLiftSoapBottle_scene401" and win["t_end"] == 14.06
    assert [s["step"] for s in win["steps"]][-1] == "lift" and all(s["ok"] for s in win["steps"])
    assert not lost["ok"] and lost["failure_class"] == "collide" and failing(lost) == "close"
    assert lost["task"] == "HolodeckPickBookFromDresser_scene117" and lost["why"] == "collide"
    close = next(s for s in lost["steps"] if s["step"] == "close")
    assert (
        close["moved_cm"] > 3.0 or close["tilt_deg"] > 20.0
    )  # the evidence for "collide": pushed or tipped before the close
    for o in (win, lost):
        text = json.dumps(o)
        assert not any(s in text for s in SENTINELS) and "true_centre" not in text and '"target"' not in text
        assert "stance_used" in o["lml"] and "comply_facts" in o["lml"] and o["lml"]["sha"]
        json.dumps(o)


# ------------------------------------------------------------------------------------------------- cards
def test_cards_hold_the_phase_the_numbers_and_the_marks(tmp_path):
    rounds = [
        rd(rec()),
        L.to_round(
            rec(success=False, loss="grasp", why="slipped", comply={"ok": False, "why": "not_success"}, genuine=False),
            f"{TASKS[1]}_s4",
            "h",
            {"grip": 0.3},
            "s",
        ),  # fmt: skip
        L.to_round(rec(comply={"ok": False, "why": "no_hand"}), f"{TASKS[1]}_s5", "h", {}, "s"),
    ]
    f = L.cards(rounds, tmp_path / "c")
    text = f.read_text()
    assert f.name == "cards.md" and text.count("\n## ") == 3  # the segment table (rules 48) and two cards
    seg = next(x for x in text.splitlines() if x.startswith("- ") and "lost)" in x)
    assert "## Segments" in text and "lift 1/" in seg  # the grasp loss is blamed on the lift
    assert f"{TASKS[1]} 1/3" in text.splitlines()[1] and "grip=0.3" in text and "failed phase: lift" in text
    assert "palm_err_cm 1.2" in text and "mark after_lift:" in text and "comply: False -- no_hand" in text
    assert f"{TASKS[1]}_s3" not in text
    assert max(len(c.splitlines()) for c in text.split("\n## ")) <= 60


def test_card_instruction_comes_from_the_record_or_the_family_goals():
    r = rd(rec(success=False, loss="grasp", instruction="put the apple into the box"))
    assert "instruction: put the apple into the box" in "\n".join(L.card_text(r))
    r = rd(rec(success=False, loss="grasp", episode_id="IThorKitchenAppleIntoBox_scene13/Q/1"))
    assert "instruction: put the apple into the box" in "\n".join(L.card_text(r))


POSITIONS = dict(
    true_centre=[5.507, 4.393, 0.777],
    true_grasp=[5.535, 4.44, 0.913],
    dest_true=[1.234, 2.345, 0.456],
    true_top=0.855,
    true_foot=0.7,
    target="cup_main_9f8e7d",
    dest_centre_true=[9.1, 8.2],
    foot_true=[1.0, 2.0],
    nav={"end_dist_cm": 45.5, "centre_err_cm": 3.0, "true_centre": [5.67, 4.79], "log": [[0.93, [5.65, 4.72], [3.55, 4.81]]]},
    spec={"region": [[5.6, 4.7], [6.1, 5.1]]},
    cam_err={"pos_cm": [-0.01, 0.2, 0.3], "err_cm": 0.4, "angle_deg": 0.5},
    grasp_desc=[0.111, 0.222, 0.333, 0, 0, -1, 1, 0, 0],
    stop_vec_cm=[-2.7, -5.2],
    obj_dest_end={"dxy_cm": [-15.0, -44.5], "dz_cm": -0.0},
)  # fmt: skip


def test_card_allow_list_drops_every_room_frame_position():
    raw = rec(
        success=False, loss="grasp", why="x", comply={"ok": False, "why": "not_success"}, genuine=False, **POSITIONS
    )
    o = rd(raw)
    cards_text = "\n".join(L.card_text(o))
    stored = json.dumps(o)
    for text in (cards_text, stored):
        for needle in ("5.507", "4.393", "5.535", "4.44", "1.234", "2.345", "0.855", "cup_main", "9.1", "5.67", "4.79", "0.93",
                       "region", "0.111", "-2.7", "-15.0", "-44.5", "dest_true", "true_centre", "true_grasp", "pos_cm"):  # fmt: skip
            assert needle not in text, needle
    assert "palm_err_cm 1.2" in cards_text and "nav end_dist_cm 45.5" in cards_text and "dist_cm 47.0" in cards_text
    assert "err_cm" in stored  # the camera's error in centimetres may stay
    # a round stored by another version, raw record and all, is filtered again when the card is written
    o["lml"] = dict(raw)
    again = "\n".join(L.card_text(o))
    assert "true_centre" not in again and "5.507" not in again and "dest_true" not in again and "1.234" not in again


def test_public_is_idempotent_and_an_allow_list():
    pub = L.public(rec(**POSITIONS, brand_new_field=[1.0, 2.0, 3.0], other={"x": 1}))
    assert "brand_new_field" not in pub and "other" not in pub
    assert L.public(pub) == pub
    err = L.public(dict(error="IK failed for [5.5, 4.4, 0.9] after 3 tries"))["error"]
    assert "5.5" not in err and "after 3 tries" in err


# ------------------------------------------------------------------------------------------------- pools
def pool_ids(d: Path) -> set[str]:
    return {json.loads(p.read_text())["episode_id"] for p in d.glob("*_s*.json")}


def per_level(ids: set[str], task: str) -> list[int]:
    lines = [int(i.rsplit("/", 1)[1]) for i in ids if i.startswith(f"{task}/Q/")]
    return [sum(lv * 50 <= n < lv * 50 + 50 for n in lines) for lv in (0, 1, 2)]


def test_pools(tmp_path):
    fam = make_fam(tmp_path)
    s = {p: L.draw(fam, p, tmp_path / p) for p in ("diag", "accept", "sealed")}
    assert [s[p]["written"] for p in s] == [60, 24, 42]
    ids = {p: pool_ids(tmp_path / p) for p in s}
    assert (
        not (ids["diag"] & ids["accept"]) and not (ids["diag"] & ids["sealed"]) and not (ids["accept"] & ids["sealed"])
    )
    for t in TASKS:
        assert [per_level(ids[p], t) for p in s] == [[10, 10, 10], [4, 4, 4], [7, 7, 7]]
    f = json.loads((tmp_path / "diag" / f"{TASKS[0]}_s5.json").read_text())
    assert f == {"task": TASKS[0], "episode_id": f"{TASKS[0]}/Q/5"}
    assert L.hidden_ids(fam) == {f"{t}/Q/{i}" for t in TASKS for i in range(150)} - ids["diag"]


def test_the_real_family_draws_600_240_420(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = L.load("lml_q_bench")
    assert fam["backend"] == "bench" and len(fam["tasks"]) == 20 and set(fam["rows"]) == set(fam["tasks"])
    assert all(isinstance(r["goal"], str) and r["goal"] for r in fam["rows"].values())
    assert fam["rows"]["IThorKitchenAppleIntoBox_scene13"]["goal"] == "put the apple into the box"
    assert [fam["pools"][p]["n"] for p in ("diag", "accept", "sealed")] == [600, 240, 420]
    assert fam["pools"]["accept"].get("per_level") == 4 and fam["pools"]["sealed"].get("per_level") == 7
    assert BK.of("lml_q_bench") is BB and BK.table() == {"bench": "backend_bench"}
    fam["practice_dir"] = str(tmp_path / "practice")
    make_practice(tmp_path / "practice", fam["tasks"])
    s = {p: L.draw(fam, p, tmp_path / p) for p in ("diag", "accept", "sealed")}
    assert [s[p]["written"] for p in s] == [600, 240, 420]
    ids = {p: pool_ids(tmp_path / p) for p in s}
    assert len(ids["diag"] | ids["accept"] | ids["sealed"]) == 1260
    for t in fam["tasks"]:
        assert [per_level(ids[p], t) for p in s] == [[10, 10, 10], [4, 4, 4], [7, 7, 7]]


def test_layout_and_the_skill_package(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = make_fam(tmp_path)
    lay = L.layout(fam)
    assert lay["skills"] == tmp_path / "state" / "lml_t" / "package" and (lay["skills"] / "__init__.py").exists()
    assert lay["caps"]["p_max"] == 0.05 and lay["caps"]["probe_scope"] is True and lay["caps"]["min_net"] == 3
    assert set(lay["class_suggest"]) == set(L.CLASS_SUGGEST) and lay["stub_file"] == "pick.py" and lay["leak"] == L.LEAK
    p = L.load_skills(lay["skills"], "x", fam)
    assert set(p.parameters(p)) == {"stand", "exact", "lift", "nav_gap"}
    assert p.apply_set(p, {"lift": 0.4, "stance_d": 0.1}) == {"lift": 0.4, "stance_d": 0.1}
    assert p.parameters(p)["lift"] == 0.4
    with pytest.raises(KeyError):
        p.apply_set(p, {"lfit": 1})
    with pytest.raises(L.BaseWriteRefused):
        L.guard_write(fam, tmp_path / "bed" / AD["seed"] / "x.py")


def test_real_family_layout_points_at_the_lml_rules_and_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    lay = L.layout(L.load("lml_q_bench"))
    assert lay["rules"].name == "lml.md" and lay["prompts"].name == "prompts" and lay["plan"] is None


# ------------------------------------------------------------------------------------------- run_jobs


# ---------------------------------------------------------------------------------------- sandbox / leak
def test_leak_regexes():
    pats = [re.compile(x) for x in L.LEAK]
    hit = lambda s: any(p.search(s) for p in pats)  # noqa: E731
    for s in (
        "HolodeckApproachDesk_scene117/E/3",
        "IThorBathroomLiftSoapBottle_scene401/P/-1000",
        "x_scene13/W/7",
        "outputs/escalation_20261116/a_l/cam",
        "cat a_l/specs.json",
        "seed_pkg_lml_removed.md",
        "outputs/scene_rsi_20261016_hidden/lml_q/master_seed.txt",
        "/workspace/code/locomanipulation-esc/outputs/x",
        "third_party/LocoManiLab/src/locomanilab/benchmark/tasks.yaml",
        "docs/handoff/escalation-plan-20261116.md",
        "docs/handoff/benchmarks-20261116.md",
        "outputs/escalation_20261116/practice_lml/x.jsonl",
        "best_p.json",
    ):
        assert hit(s), s
    for s in (
        "HolodeckApproachDesk_scene117/Q/3",
        "level 2",
        "IThorKitchenAppleIntoBox_scene13_s14",
        "walk to the desk",
    ):
        assert not hit(s), s


def test_the_seed_package_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = make_fam(tmp_path)
    seed = tmp_path / "bed" / AD["seed"]
    before = (seed / "__init__.py").read_text()
    work = BB.ensure_package(fam)
    assert work == tmp_path / "state" / "lml_t" / "package" and (work / "__init__.py").read_text() == before
    for path in (seed, seed / "__init__.py", seed / "sub" / "new.py"):
        with pytest.raises(L.BaseWriteRefused):
            L.guard_write(fam, path)
    with pytest.raises(L.BaseWriteRefused):
        B._write(seed / "round.json", {})


def test_stance_keys_are_the_seed_flows_and_typed(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = make_fam(tmp_path)
    p = L.load_skills(L.layout(fam)["skills"], "x", fam)
    ok = {"lift": 0.3, "stance_d": 0.05, "stance_t": -1, "stance_yaw": 5, "stance_here": True, "stance_d2": 0.1, "stance_stay": False}  # fmt: skip
    assert p.apply_set(p, ok) == ok
    for bad in ({"stance_yaw2": 5}, {"stance_foo": 1}, {"stance_here": 1}, {"stance_d": True}, {"stance_d": "0.1"}):
        with pytest.raises(KeyError):
            p.apply_set(p, bad)
    with pytest.raises(KeyError):
        p.apply_set(p, {"nope": 1})
    assert C.split_set(ok)[1] == {"d": 0.05, "t": -1, "yaw": 5, "here": True, "d2": 0.1, "stay": False}


def test_leak_pattern_does_not_flag_bed_files_by_their_first_letter():
    import re

    hits = lambda s: [i for i, rx in enumerate(L.LEAK) if re.search(rx, s)]  # noqa: E731
    assert hits("bed_readonly_copy/scripts/experiments/escalation_20261116/bed_lml_cam.py") == []
    assert hits("bed_readonly_copy/scripts/experiments/escalation_20261116/seed_pkg_lml/view_lml.py") == []
    assert hits("outputs/escalation_20261116/a_l/specs.json") and hits("outputs/escalation_20261116/b/rows.jsonl")


def test_leak_pattern_for_specs_needs_a_path():
    import re

    hits = lambda s: [i for i, rx in enumerate(L.LEAK) if re.search(rx, s)]  # noqa: E731
    assert hits('grep -n "specs.json" bed_readonly_copy/scripts/experiments/escalation_20261116/bed_lml.py') == []
    assert hits("cat outputs/escalation_20261116/a_l/specs.json")


# ------------------------------------------------------------------------------- the fridge family (2026-11-21)
FRIDGE = ["IThorKitchenOpenFridge_scene13", "IThorKitchenCloseFridge_scene13", "IThorKitchenApproachOpenFridge_scene13"]
ART = dict(
    action="open", goal="fraction >= 0.50 held 1.0 s", frac_start=0.1, frac_max=0.13, frac_min=0.1, frac_end=0.12,
    t_first_move=4.2, t_goal=None, hold_best_s=0.0, touch=["hand"], t_touch=4.1, hand_min_cm=3.2, hand_min_side="right",
    hand_min_t=4.3, palm_dz_cm=-4.8, palm_dxy_cm=1.1, palm_best_dz_cm=-3.9, grip_on_handle=False,
)  # fmt: skip


def test_the_fridge_family_draws_90_90_90_on_its_own_slots(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = L.load("lml_fridge_bench")
    assert fam["tasks"] == FRIDGE and set(fam["rows"]) == set(FRIDGE)
    assert fam["rows"]["IThorKitchenOpenFridge_scene13"]["goal"] == "open the fridge door in front of you"
    assert BK.of("lml_fridge_bench") is BB
    assert fam["bench"]["config"]["slots"] == [3, 4] and BB.slots_of(fam) == 2  # the lml_q loop keeps slots 1 and 2
    assert "practice_lml_fridge" in fam["practice_dir"]
    fam["practice_dir"] = str(tmp_path / "practice")
    make_practice(tmp_path / "practice", fam["tasks"])
    s = {p: L.draw(fam, p, tmp_path / p) for p in ("diag", "accept", "sealed")}
    assert [s[p]["written"] for p in s] == [90, 90, 90]
    ids = {p: pool_ids(tmp_path / p) for p in s}
    assert len(ids["diag"] | ids["accept"] | ids["sealed"]) == 270
    for t in FRIDGE:
        assert [per_level(ids[p], t) for p in s] == [[10, 10, 10]] * 3


def test_door_fields_pass_the_allow_list_and_reach_the_card():
    pub = L.public(rec(art=dict(ART, handle_xyz=[-0.62, 3.74, 1.21], door_xy=[1.0, 2.0])))
    assert set(pub["art"]) == set(ART) and "handle_xyz" not in pub["art"] and "door_xy" not in pub["art"]
    text = L.door_text(pub["art"])
    assert "open fraction 0.10 -> max 0.13" in text and "touched by hand" in text and "right 3.2 cm" in text
    assert "-4.8 cm above it" in text and "3.74" not in text


def test_leak_patterns_cover_the_fridge_hand_runs():
    pats = [re.compile(p) for p in L.LEAK]
    for s in ("cat /workspace/tmp/lml_bed2_smoke/run/x.json", "python bed2_truth_check.py",
              "ls outputs/escalation_20261116/practice_lml_fridge", "head master_seed.txt"):  # fmt: skip
        assert any(p.search(s) for p in pats), s
    assert not any(
        p.search("python scripts/rsi/try.py lml_fridge --labels IThorKitchenOpenFridge_scene13_s3") for p in pats
    )


DRAWER = [
    "IThorLivingRoomLowDrawerPnP_scene209",
    "IThorLivingRoomWalkAndOpenLowDrawer_scene209",
    "IThorLivingRoomWalkAndCloseLowDrawer_scene209",
]


def test_the_drawer_family_draws_90_90_90_on_its_own_slots(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = L.load("lml_drawer_bench")
    assert fam["tasks"] == DRAWER and set(fam["rows"]) == set(DRAWER)
    goal = fam["rows"]["IThorLivingRoomWalkAndCloseLowDrawer_scene209"]["goal"]
    assert goal == "walk to the lower-left drawer of the dresser, then crouch down and close it"
    assert BK.of("lml_drawer_bench") is BB
    assert (
        fam["bench"]["config"]["slots"] == [5, 6] and BB.slots_of(fam) == 2
    )  # lml_q keeps slots 1 and 2, the fridge 3 and 4
    assert "practice_lml_drawer" in fam["practice_dir"]
    assert fam["pools"]["accept"]["seed"] != L.load("lml_fridge_bench")["pools"]["accept"]["seed"]
    fam["practice_dir"] = str(tmp_path / "practice")
    make_practice(tmp_path / "practice", fam["tasks"])
    s = {p: L.draw(fam, p, tmp_path / p) for p in ("diag", "accept", "sealed")}
    assert [s[p]["written"] for p in s] == [90, 90, 90]
    ids = {p: pool_ids(tmp_path / p) for p in s}
    assert len(ids["diag"] | ids["accept"] | ids["sealed"]) == 270
    for t in DRAWER:
        assert [per_level(ids[p], t) for p in s] == [[10, 10, 10]] * 3


def test_drawer_fields_pass_the_allow_list_and_reach_the_card():
    art = dict(ART, action="close", pelvis_drop_cm=21.5, moved_by=["hand"], drawer_xyz=[1.0, 2.0, 0.27])
    pub = L.public(rec(task=DRAWER[2], art=art))
    assert (
        pub["art"]["pelvis_drop_cm"] == 21.5 and pub["art"]["moved_by"] == ["hand"] and "drawer_xyz" not in pub["art"]
    )
    text = L.door_text(pub["art"], "drawer")
    assert text.startswith("drawer: open fraction") and "moved by hand" in text and "21.5 cm below standing" in text
    assert "0.27" not in text
    assert "door:" in L.door_text(pub["art"]) and "moved by" not in L.door_text(L.public(rec(art=ART))["art"])


def test_leak_patterns_cover_the_drawer_hand_runs():
    pats = [re.compile(p) for p in L.LEAK]
    for s in ("cat /workspace/tmp/lml_bed3_smoke/run/x.json", "python bed3_truth_check.py",
              "ls /workspace/tmp/drawer_smoke_practice", "cat /workspace/tmp/drawer_facts.md",
              "ls outputs/escalation_20261116/practice_lml_drawer", "cat /workspace/tmp/lml_bed3_report.md"):  # fmt: skip
        assert any(p.search(s) for p in pats), s
    assert not any(p.search(f"python scripts/rsi/try.py lml_drawer --labels {DRAWER[1]}_s3") for p in pats)

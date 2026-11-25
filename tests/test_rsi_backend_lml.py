"""The LocoManiLab backend of the RSI loop (scripts/rsi/backend_lml.py): records, cards, pools, jobs, sandbox. No simulator."""

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
import backend_lml as L  # noqa: E402
import backend_simple as B  # noqa: E402

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
    (root / "bed").mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("EXTRA = dict(lift=0.34, nav_gap=0.3)\n")
    (root / "pkg" / "skill_simple.py").write_text("K_ORIG = dict(stand=0.27)\nK_CUT = dict(K_ORIG, exact=True)\n")
    practice = tmp / "practice"
    make_practice(practice, TASKS)
    fam = dict(
        name="lml_t",
        backend="lml",
        tasks=TASKS,
        pools={
            "diag": {"seed": 0, "n": 60},
            "accept": {"seed": 7, "n": 24, "per_level": 4},
            "sealed": {"seed": 7, "n": 42, "per_level": 7},
        },
        bed_root=str(root),
        bed_dir="bed",
        bed_files=["bed/a.py"],
        hash_files=["bed/h.py"],
        track="camera",
        base="pkg",
        guard="guard.py",
        runner="bed/run.sh",
        official_slots=2,
        practice_dir=str(practice),
        rules="rules_scene.md",
        prompts="prompts",
        plan=None,
        caps={"min_net": 3},
        sandbox={"dir": "bed_readonly_copy", "task_files": []},
    )
    fam.update(kw)
    (root / "bed" / "a.py").write_text("print(1)\n")
    (root / "bed" / "h.py").write_text("print(2)\n")
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
    fam = L.load("lml_q")
    assert fam["backend"] == "lml" and len(fam["tasks"]) == 20 and set(fam["rows"]) == set(fam["tasks"])
    assert all(isinstance(r["goal"], str) and r["goal"] for r in fam["rows"].values())
    assert fam["rows"]["IThorKitchenAppleIntoBox_scene13"]["goal"] == "put the apple into the box"
    assert [fam["pools"][p]["n"] for p in ("diag", "accept", "sealed")] == [600, 240, 420]
    assert fam["pools"]["accept"].get("per_level") == 4 and fam["pools"]["sealed"].get("per_level") == 7
    assert BK.of("lml_q") is L and BK.table()["lml"] == "backend_lml"
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
    assert set(lay["class_suggest"]) == set(L.CLASSES) and lay["stub_file"] == "pick.py" and lay["leak"] == L.LEAK
    p = L.load_skills(lay["skills"], "x", fam)
    assert set(p.parameters(p)) == {"stand", "exact", "lift", "nav_gap"}
    assert p.apply_set(p, {"lift": 0.4, "stance_d": 0.1}) == {"lift": 0.4, "stance_d": 0.1}
    assert p.parameters(p)["lift"] == 0.4
    with pytest.raises(KeyError):
        p.apply_set(p, {"lfit": 1})
    with pytest.raises(L.BaseWriteRefused):
        L.guard_write(fam, tmp_path / "bed" / "pkg" / "x.py")


def test_real_family_layout_points_at_the_lml_rules_and_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    lay = L.layout(L.load("lml_q"))
    assert lay["rules"].name == "lml.md" and lay["prompts"].name == "prompts" and lay["plan"] is None


# -------------------------------------------------------------------------------------------- signature
def test_signature(tmp_path):
    fam = make_fam(tmp_path)
    s = L.signature(fam, "probe")
    assert s == L.signature(fam, "gate")  # one bed
    root = Path(fam["bed_root"])
    (root / "bed" / "a.py").write_text("print(11)\n")
    s1 = L.signature(fam, "probe")
    assert s1 != s
    (root / "bed" / "h.py").write_text("print(22)\n")
    s2 = L.signature(fam, "probe")
    assert s2 != s1
    assert L.signature(dict(fam, base="pkg2"), "probe") != s2
    assert L.signature(dict(fam, bed_env={"A": "1"}), "probe") != s2
    assert L.signature(dict(fam, track="truth"), "probe") != s2
    assert L.signature(dict(fam, bed_fixed={"feed_torque": True}), "probe") != s2
    assert L.signature(dict(fam, bed_fixed={"_about": "a note only"}), "probe") == s2


def test_bed_fixed_switches_reach_the_runner_and_no_set_names_them(rig, tmp_path):
    fam, pool, count, root = rig
    fam["bed_fixed"] = {"_about": "rule 46", "feed_torque": True}
    job = Job(root / "pkg", tmp_path / "outf", {"grip": 0.5, "stance_d": 0.2})
    assert L.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0") == {0: []}
    c = calls(count)
    assert c and all(json.loads(x["setting"]) == {"grip": 0.5, "feed_torque": True} for x in c)
    assert all(json.loads(x["stance"]) == {"d": 0.2} for x in c)
    r = json.loads(next((tmp_path / "outf").glob("*.json")).read_text())
    assert r["set"] == {"grip": 0.5, "stance_d": 0.2}
    p = L.load_skills(L.layout(fam)["skills"], "x", fam)
    with pytest.raises(KeyError, match="fixed by the bed"):
        p.apply_set(p, {"feed_torque": False})


def test_signature_carries_the_benchmark_commit(tmp_path, monkeypatch):
    # the commit hook runs the suite in a worktree with GIT_DIR / GIT_INDEX_FILE pointing at the real repository: a git
    # call here (or in bench_commit) that inherits them writes into it (2026-11-24 it set core.bare and staged `f`)
    for k in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(k)
    fam = make_fam(tmp_path, benchmark="bench")
    bench = Path(fam["bed_root"]) / "bench"
    bench.mkdir()
    git = lambda *a: subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=bench, check=True, capture_output=True
    )  # noqa: E731
    git("init", "-q")
    (bench / "f").write_text("1")
    git("add", "f")
    git("commit", "-q", "-m", "one")
    s = L.signature(fam, "probe")
    assert L.bench_commit(fam) and s != L.signature(dict(fam, benchmark=None), "probe")
    (bench / "f").write_text("2")
    git("commit", "-q", "-am", "two")
    assert L.signature(fam, "probe") != s
    assert L.bench_commit(dict(fam, benchmark="nowhere")) == "unknown"


# ------------------------------------------------------------------------------------------- run_jobs
FAKE_RUNNER = r"""
import json, os, sys
a = sys.argv[1:]
task = a[0]
ids = open(a[a.index("--ids") + 1]).read().split()
out = a[a.index("--out") + 1]
env = {k: os.environ.get(k) for k in ("LML_SKILLS", "LML_LOOP", "LML_PRACTICE", "LML_RUN_DIR", "LML_SLOTS", "EXAM_OPEN", "OMP_NUM_THREADS")}
env["cloud_is_link"] = os.path.islink(os.path.join(os.environ["LML_RUN_DIR"], "cloud"))
open(os.environ["FAKE_COUNT"], "a").write(json.dumps({"task": task, "n": len(ids), "env": env,
    "setting": a[a.index("--setting") + 1], "stance": a[a.index("--stance") + 1] if "--stance" in a else None}) + "\n")
with open(out, "w") as f:
    for e in ids:
        f.write(json.dumps({"episode_id": e, "success": True, "t_check": 9.0, "loss": None, "fell": None, "why": "",
            "marks": {}, "phases": {"walk": 1.0}, "t_end": 9.0, "wall_s": 1.0, "genuine": True,
            "comply": {"ok": True, "why": ""}, "true_centre": [1.5, 2.5, 3.5]}) + "\n")
"""
FAKE_GUARD = r"""
import json, sys
from pathlib import Path
bad = (Path(sys.argv[2]) / "bad.txt").exists()
print(json.dumps({"ok": not bad, "violations": ["uses truth"] if bad else []}))
sys.exit(1 if bad else 0)
"""


@dataclass
class Job:
    skills: Path
    out: Path
    overrides: dict | None = None
    labels: tuple | None = None


@pytest.fixture
def rig(tmp_path, monkeypatch):
    fam = make_fam(tmp_path)
    root = Path(fam["bed_root"])
    (root / "bed" / "fake_runner.py").write_text(FAKE_RUNNER)
    (root / "bed" / "guard.py").write_text(FAKE_GUARD)
    fam["bed_cmd"] = [sys.executable, str(root / "bed" / "fake_runner.py")]
    count = tmp_path / "count.jsonl"
    monkeypatch.setenv("FAKE_COUNT", str(count))
    monkeypatch.setenv("EXAM_OPEN", "1")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    pool = tmp_path / "pool"
    L.draw(fam, "diag", pool, n=3)  # one line of each task in turn: three labels
    return fam, pool, count, root


def calls(count: Path) -> list[dict]:
    return [json.loads(x) for x in count.read_text().splitlines()] if count.exists() else []


def test_run_jobs_three_labels_then_nothing(rig, tmp_path):
    fam, pool, count, root = rig
    job = Job(root / "pkg", tmp_path / "out", {"grip": 0.5, "stance_d": 0.2})
    assert L.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0,1") == {0: []}
    files = sorted(p.name for p in (tmp_path / "out").glob("*.json"))
    assert len(files) == 3
    r = json.loads((tmp_path / "out" / files[0]).read_text())
    assert r["ok"] and r["set"] == {"grip": 0.5, "stance_d": 0.2} and r["motor"] == L.signature(fam, "probe")
    assert "true_centre" not in json.dumps(r)  # the stored round holds the allow-listed record
    c = calls(count)
    assert sum(x["n"] for x in c) == 3 and {x["task"] for x in c} == set(TASKS)
    assert all(json.loads(x["setting"]) == {"grip": 0.5} and json.loads(x["stance"]) == {"d": 0.2} for x in c)
    e = c[0]["env"]
    assert e["LML_SKILLS"] == str((root / "pkg").resolve()) and e["LML_LOOP"] == "1" and e["EXAM_OPEN"] is None
    assert e["LML_PRACTICE"] == fam["practice_dir"] and e["OMP_NUM_THREADS"] == "2" and e["LML_RUN_DIR"]
    assert e["LML_SLOTS"] == "1 2" and e["cloud_is_link"] is True
    assert L.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0,1") == {0: []}
    assert len(calls(count)) == len(c)  # nothing ran again
    job2 = Job(root / "pkg", tmp_path / "out2", {}, labels=(files[0][:-5],))
    assert L.run_jobs([job, job2], fam, pool, 2, tmp_path / "log.txt", "0") == {0: [], 1: []}
    assert calls(count)[-1]["n"] == 1 and calls(count)[-1]["stance"] is None


def test_run_dirs_differ_between_jobs_and_the_cache_is_shared_by_a_package(rig, tmp_path):
    fam, pool, count, root = rig
    jobs = [Job(root / "pkg", tmp_path / f"o{k}", {"k": k}) for k in range(2)]
    L.run_jobs(jobs, fam, pool, 1, tmp_path / "log.txt", "0")
    dirs = {c["env"]["LML_RUN_DIR"] for c in calls(count)}
    assert len(dirs) == 2 and all(not Path(d).exists() for d in dirs)  # finished runs are removed
    caches = list((tmp_path / "tmp" / "lml_lml_t_cache").iterdir())
    assert len(caches) == 1  # the same package under the same bed: one cache


def test_guard_failure_runs_nothing(rig, tmp_path):
    fam, pool, count, root = rig
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "__init__.py").write_text("")
    (cand / "bad.txt").write_text("x")
    job = Job(cand, tmp_path / "outg")
    assert L.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0") == {0: []}
    assert calls(count) == []
    rs = [json.loads(p.read_text()) for p in (tmp_path / "outg").glob("*.json")]
    assert len(rs) == 3 and all(
        not r["ok"] and r["why"] == "guard: uses truth" and r["failure_class"] == "guard" for r in rs
    )
    assert all(r["lml"] == {} for r in rs)
    assert L.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0") == {0: []} and calls(count) == []


def test_missing_records_stay_missing(rig, tmp_path):
    fam, pool, count, root = rig
    (root / "bed" / "fake_runner.py").write_text("import sys\nsys.exit(3)\n")
    out = L.run_jobs([Job(root / "pkg", tmp_path / "o")], fam, pool, 1, tmp_path / "log.txt", "0")
    assert len(out[0]) == 3


def test_run_jobs_refuses_a_blank_or_missing_episode(rig, tmp_path):
    fam, pool, count, root = rig
    f = Path(fam["practice_dir"]) / f"{TASKS[0]}.jsonl"
    lines = f.read_text().splitlines()
    lines[3] = "{}"  # a hidden row, blanked as a sandbox does
    f.write_text("\n".join(lines) + "\n")
    for n in (3, 999):
        (pool / f"{TASKS[0]}_s{n}.json").write_text(json.dumps(dict(task=TASKS[0], episode_id=f"{TASKS[0]}/Q/{n}")))
    job = Job(root / "pkg", tmp_path / "out", {}, labels=(f"{TASKS[0]}_s3", f"{TASKS[0]}_s999"))
    assert L.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0") == {0: []}
    for x in job.labels:
        r = json.loads((tmp_path / "out" / f"{x}.json").read_text())
        assert r["ok"] is False and r["why"] == "not a diagnostic episode" and r["failure_class"] == "refused"
    assert calls(count) == []  # the runner never started


def test_where_only_selects_the_practice_files(rig, tmp_path, monkeypatch):
    fam, pool, count, root = rig
    full = tmp_path / "hidden_full"
    make_practice(full, TASKS)
    monkeypatch.setattr(B, "practice_dir", lambda fam, full_=True: full if full_ else Path(fam["practice_dir"]))
    monkeypatch.setattr(L, "practice_dir", B.practice_dir)
    L.run_jobs([Job(root / "pkg", tmp_path / "op")], fam, pool, 1, tmp_path / "log.txt", "0", where="probe")
    L.run_jobs([Job(root / "pkg", tmp_path / "og")], fam, pool, 1, tmp_path / "log.txt", "0", where="gate")
    seen = [c["env"]["LML_PRACTICE"] for c in calls(count)]
    assert set(seen) == {fam["practice_dir"], str(full)} and seen.count(str(full)) == seen.count(fam["practice_dir"])
    assert (
        json.loads(next((tmp_path / "op").glob("*.json")).read_text())["motor"]
        == json.loads(next((tmp_path / "og").glob("*.json")).read_text())["motor"]
    )  # one bed: the same signature


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


def test_sandbox_and_audit(tmp_path):
    fam = make_fam(tmp_path)
    orig = tmp_path / "origin"  # what the sandbox's third_party link points at
    (orig / "LocoManiLab" / "src" / "locomanilab").mkdir(parents=True)
    (orig / "LocoManiLab" / "src" / "locomanilab" / "tasks.yaml").write_text("x: 1\n")
    (orig / "SIMPLE").mkdir()
    (orig / "SIMPLE" / "keep.txt").write_text("ok")
    dest = tmp_path / "sb"
    dest.mkdir()
    (dest / "third_party").symlink_to(orig)
    made = []
    for pat in sorted(
        L.FORBIDDEN_PATHS, key=lambda x: -len(x)
    ):  # the deepest first: a parent entry is then a directory
        if pat.startswith("third_party/"):
            continue  # made in the origin
        p = dest / pat.replace("*", "x")
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("hand-tuned")
        made.append(p)
    (dest / "docs" / "handoff" / "keep.md").write_text("x")
    ff = dest / "scripts" / "rsi" / "families" / "lml_t.json"
    ff.parent.mkdir(parents=True)
    ff.write_text(json.dumps(fam))
    out = L.audit(dest)
    assert all(x.startswith("path ") for x in out) and len(out) >= len(made)  # before: the forbidden paths are there
    assert any("third_party/LocoManiLab" in x for x in out)
    L.sandbox(dest, fam)
    assert L.audit(dest) == []
    for pat in L.FORBIDDEN_PATHS:  # every forbidden path is absent (globs: no match)
        assert not list(dest.glob(pat)) and not os.path.lexists(dest / pat), pat
    assert (orig / "LocoManiLab" / "src" / "locomanilab" / "tasks.yaml").exists()  # the origin is untouched
    assert (dest / "third_party" / "SIMPLE" / "keep.txt").read_text() == "ok" and (
        dest / "docs" / "handoff" / "keep.md"
    ).exists()
    nest = dest / "bed_readonly_copy"
    assert (nest / "pkg" / "__init__.py").exists() and not (nest / "bed" / "a.py").stat().st_mode & 0o222
    assert not (nest / "bed" / "h.py").exists()  # hash-only files are not given to the roles
    hidden = B.hidden_ids(fam)
    for t in TASKS:
        vis = (nest / "practice" / f"{t}.jsonl").read_text().splitlines()
        assert len(vis) == 150
        assert all((x == "{}") == (f"{t}/Q/{i}" in hidden) for i, x in enumerate(vis))
        full = dest / B.HIDDEN / "lml_t" / "practice" / f"{t}.jsonl"
        assert full.read_text() == (Path(fam["practice_dir"]) / f"{t}.jsonl").read_text()
    new = json.loads(ff.read_text())
    assert new["bed_root"] == fam["bed_root"] and new["practice_dir"] == "bed_readonly_copy/practice"
    # every text file of the readable copy is clean; a leak, an unblanked hidden row and a re-created path are reported
    for p in nest.rglob("*"):
        if p.is_file() and not p.is_symlink():
            text = p.read_text()
            assert not any(re.search(x, text) for x in L.LEAK), p
    (nest / "pkg" / "notes.md").write_text("see outputs/escalation_20261116/a_l/specs.json\n")
    (nest / "practice" / f"{TASKS[1]}.jsonl").write_text("".join(row(1) + "\n" for _ in range(25)))
    (dest / "outputs" / "escalation_20261116" / "a_l").mkdir(parents=True)
    out = L.audit(dest)
    assert any("notes.md" in x for x in out) and any("practice row not blank" in x for x in out)
    assert any("a_l" in x for x in out)
    # the audit does not read the hidden tree
    (dest / B.HIDDEN / "lml_t" / "note.txt").write_text("specs.json master_seed")
    assert not any("note.txt" in x for x in L.audit(dest))


def test_the_seed_package_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = make_fam(tmp_path)
    seed = tmp_path / "bed" / "pkg"
    before = (seed / "__init__.py").read_text()
    work = L.ensure_package(fam)
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
    assert L.split_set(ok)[1] == {"d": 0.05, "t": -1, "yaw": 5, "here": True, "d2": 0.1, "stay": False}


def test_the_engineers_guard_command_names_exist():
    assert L.run_guard is B.run_guard


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
    fam = L.load("lml_fridge")
    assert fam["tasks"] == FRIDGE and set(fam["rows"]) == set(FRIDGE)
    assert fam["rows"]["IThorKitchenOpenFridge_scene13"]["goal"] == "open the fridge door in front of you"
    assert BK.of("lml_fridge") is L
    env = L.bed_env(fam, tmp_path / "pkg", tmp_path / "run")
    assert env["LML_SLOTS"] == "3 4"  # the lml_q loop keeps slots 1 and 2
    assert "lml_bed2_20261121" in fam["runner"] and "practice_lml_fridge" in fam["practice_dir"]
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
    fam = L.load("lml_drawer")
    assert fam["tasks"] == DRAWER and set(fam["rows"]) == set(DRAWER)
    goal = fam["rows"]["IThorLivingRoomWalkAndCloseLowDrawer_scene209"]["goal"]
    assert goal == "walk to the lower-left drawer of the dresser, then crouch down and close it"
    assert BK.of("lml_drawer") is L
    env = L.bed_env(fam, tmp_path / "pkg", tmp_path / "run")
    assert env["LML_SLOTS"] == "5 6"  # lml_q keeps slots 1 and 2, the fridge fork 3 and 4
    assert "lml_bed3_20261121" in fam["runner"] and "practice_lml_drawer" in fam["practice_dir"]
    assert fam["pools"]["accept"]["seed"] != L.load("lml_fridge")["pools"]["accept"]["seed"]
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

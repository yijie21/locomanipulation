"""The SIMPLE adapter's rows (bench_simple.py), the pools of practice files (bench_pools.py) and the shared plumbing of the
benchmark backend (bench_common.py). No simulator."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsi"))
import backend_bench as BB  # noqa: E402
import bench_common as C  # noqa: E402
import bench_pools as P  # noqa: E402
import bench_rows as R  # noqa: E402
import bench_simple as S  # noqa: E402

AD = BB.ADAPTERS["simple"]


class B:  # the old backend_simple names over the modules they moved to
    to_round = staticmethod(S.to_round)
    LEAK = S.LEAK
    CLASS_SUGGEST = S.CLASS_SUGGEST
    HIDDEN = C.HIDDEN
    BaseWriteRefused = C.BaseWriteRefused
    load_skills = staticmethod(S.load_skills)
    split = staticmethod(P.split)
    diag_lines = staticmethod(P.diag_lines)
    hidden_ids = staticmethod(P.hidden_ids)
    _success_rule = staticmethod(C.success_rule)
    _units = staticmethod(C.units)
    _exec = staticmethod(C.run_command)
    _running = staticmethod(C._running)
    _write = staticmethod(C.write_round)
    _copy = staticmethod(C._copy)
    _cut = staticmethod(C._cut)
    cards = staticmethod(lambda rounds, out: R.cards(rounds, out, S.card_text))
    draw = staticmethod(lambda fam, pool, out, n=None, tasks=None: P.draw(fam, AD, pool, out, n, tasks))
    ensure_package = staticmethod(BB.ensure_package)
    guard_write = staticmethod(BB.guard_write)


TASKS = ["XMovePickTeleop", "BendPickMP"]


def row(level: int, i: int = 0) -> str:
    return json.dumps({"instruction": f"bend to pick up item {i}", "src": {"level": level, "seed": i}})


def make_practice(d: Path, tasks=TASKS, per_level: int = 50) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        d.joinpath(f"{t}.jsonl").write_text("".join(row(lv, i) + "\n" for lv in (0, 1, 2) for i in range(per_level)))


def make_fam(tmp: Path, **kw) -> dict:
    root = tmp / "bed"
    seed = root / AD["seed"]
    seed.mkdir(parents=True)
    (seed / "__init__.py").write_text("X = 1\n")
    practice = tmp / "practice"
    make_practice(practice)
    fam = dict(
        name="t_fam",
        backend="bench",
        tasks=TASKS,
        pools={"diag": {"seed": 0, "n": 20}, "accept": {"seed": 7, "n": 40}, "sealed": {"seed": 7, "n": 40}},
        bench=dict(name="simple", root=str(root)),
        track="official",
        practice_dir=str(practice),
        rules="rules_scene.md",
        prompts="prompts",
        caps={"min_net": 3},
        sandbox={"dir": "bed_readonly_copy", "tree": [], "task_files": []},
    )
    fam.update(kw)
    return fam


# ------------------------------------------------------------------------------------------- to_round
def rec(**kw) -> dict:
    base = dict(
        episode_id="XMovePickTeleop/Q/3",
        success=True,
        t_success=10.0,
        loss=None,
        fell=None,
        why="",
        marks={
            "before_close": {"t": 5, "rise_cm": 0, "moved_cm": 0.2, "tilt_deg": 1, "held": False, "basket_cm": None},
            "after_lift": {"t": 8, "rise_cm": 9.0, "moved_cm": 9, "tilt_deg": 2, "held": True, "basket_cm": None},
            "end": {"t": 10, "rise_cm": 9.0, "moved_cm": 9, "tilt_deg": 2, "held": True, "basket_cm": None},
        },
        phases={"look": 2.0, "reach_grasp": 3.0, "close": 1.0, "lift": 2.0, "lift_hold": 1.0},
        palm_err_cm=1.2,
        reach_ok=[True],
        t_end=10.0,
        wall_s=55.0,
        comply={"ok": True, "why": ""},
    )
    base.update(kw)
    return base


def rd(r: dict) -> dict:
    return B.to_round(r, "XMovePickTeleop_s3", "abc", {"k": 1}, "sig")


def failing(step_of: dict) -> str:
    return next(s["step"] for s in step_of["steps"] if not s["ok"])


def test_win():
    o = rd(rec())
    assert o["ok"] is True and o["why"] == "" and o["failure_class"] == ""
    assert o["task"] == "XMovePickTeleop" and o["label"] == "XMovePickTeleop_s3"
    assert o["set"] == {"k": 1} and o["skills"] == "abc" and o["motor"] == "sig" and o["rule"] == B._success_rule()
    assert [s["step"] for s in o["steps"]] == ["look", "reach_grasp", "close", "lift", "lift_hold"]
    assert all(s["ok"] for s in o["steps"]) and o["steps"][1]["t0"] == 2.0 and o["steps"][1]["t1"] == 5.0
    assert o["fell"] is False and o["dropped"] is False and o["plan_ok"] is True and o["simple"]["success"]


def test_win_comply_false_and_missing():
    o = rd(rec(comply={"ok": False, "why": "no_hand"}))
    assert not o["ok"] and o["failure_class"] == "comply" and o["why"] == "comply: no_hand"
    r = rec()
    del r["comply"]
    o = rd(r)
    assert not o["ok"] and o["failure_class"] == "comply" and o["why"].startswith("comply:")
    assert failing(o) == "lift_hold"  # no named phase: the last one


@pytest.mark.parametrize(
    "cls,phase",
    [
        ("deliver", "reach_grasp"),
        ("grasp", "lift"),
        ("collide", "close"),
        ("perceive", "look"),
        ("fall", "lift_hold"),
        ("timeout", "lift_hold"),
        ("place", "lift_hold"),  # the release phase is absent: the last one
    ],
)
def test_failure_classes(cls, phase):
    o = rd(rec(success=False, t_success=None, loss=cls, why="because"))
    assert not o["ok"] and o["failure_class"] == cls and o["why"] == f"{cls}: because"
    assert failing(o) == phase and sum(not s["ok"] for s in o["steps"]) == 1
    assert o["plan_ok"] is False and o["plan_why"] == "because"


def test_place_phase_and_facts():
    ph = {"look": 1.0, "reach_grasp": 2.0, "lift": 1.0, "release": 1.0, "open": 0.5}
    o = rd(rec(success=False, loss="place", why="", phases=ph))
    assert failing(o) == "release" and o["why"] == "place" and o["plan_ok"] is True
    assert next(s for s in o["steps"] if s["step"] == "reach_grasp")["palm_err_cm"] == 1.2


def test_error_fell_dropped():
    o = rd(rec(success=False, loss="error", error="boom", why=None, phases={}))
    assert o["failure_class"] == "error" and o["why"] == "error: boom" and o["error"] == "boom" and o["steps"] == []
    assert o["plan_ok"] is False and o["plan_why"] == "boom"
    o = rd(rec(success=False, loss="fall", fell=3.2))
    assert o["fell"] is True
    m = rec()["marks"]
    m["end"] = dict(m["end"], rise_cm=-20.0)
    assert rd(rec(success=False, loss="grasp", marks=m))["dropped"] is True
    assert rd(rec(success=True, marks=m))["dropped"] is False


def test_round_is_json():
    json.dumps(rd(rec(success=False, loss="grasp", why="x")))


# ------------------------------------------------------------------------------------------------- cards
def test_cards(tmp_path):
    rounds = [
        rd(rec()),
        B.to_round(rec(success=False, loss="grasp", why="slipped"), "XMovePickTeleop_s4", "h", {"grip": 0.3}, "s"),
        B.to_round(rec(comply={"ok": False, "why": "no_hand"}), "XMovePickTeleop_s5", "h", {}, "s"),
    ]
    f = B.cards(rounds, tmp_path / "c")
    text = f.read_text()
    assert f.name == "cards.md" and text.count("\n## ") == 3  # the segment table (rules 48) and two cards
    seg = next(x for x in text.splitlines() if x.startswith("- ") and "lost)" in x)
    assert "## Segments" in text and "lift 1/" in seg  # the grasp loss is blamed on the lift
    assert "XMovePickTeleop 1/3" in text.splitlines()[1] and "grip=0.3" in text and "failed phase: lift" in text
    assert "XMovePickTeleop_s3" not in text
    assert max(len(c.splitlines()) for c in text.split("\n## ")) <= 60


# ------------------------------------------------------------------------------------------------- pools
def pool_ids(d: Path) -> set[str]:
    return {json.loads(p.read_text())["episode_id"] for p in d.glob("*_s*.json")}


def test_pools(tmp_path):
    fam = make_fam(tmp_path)
    s = {p: B.draw(fam, p, tmp_path / p) for p in ("diag", "accept", "sealed")}
    assert [s[p]["written"] for p in s] == [60, 120, 120]
    ids = {p: pool_ids(tmp_path / p) for p in s}
    assert len(ids["diag"]) == 2 * 30 and len(ids["accept"]) == 2 * 60 and len(ids["sealed"]) == 2 * 60
    assert (
        not (ids["diag"] & ids["accept"]) and not (ids["diag"] & ids["sealed"]) and not (ids["accept"] & ids["sealed"])
    )
    assert ids["diag"] | ids["accept"] | ids["sealed"] == {f"{t}/Q/{i}" for t in TASKS for i in range(150)}
    f = json.loads((tmp_path / "diag" / "BendPickMP_s5.json").read_text())
    assert f == {"task": "BendPickMP", "episode_id": "BendPickMP/Q/5"}
    for p in ("accept", "sealed"):  # 20 per level per task
        for t in TASKS:
            lines = [int(m) for m in re.findall(rf"{t}/Q/(\d+)", " ".join(ids[p]))]
            assert [sum(lv * 50 <= i < lv * 50 + 50 for i in lines) for lv in (0, 1, 2)] == [20, 20, 20]
    assert B.hidden_ids(fam) == ids["accept"] | ids["sealed"]


def test_diag_rule_is_first_ten_per_level_in_file_order(tmp_path):
    fam = make_fam(tmp_path)
    f = Path(fam["practice_dir"]) / "BendPickMP.jsonl"  # levels interleaved: 0 1 2 0 1 2 ...
    f.write_text("".join(row(i % 3, i) + "\n" for i in range(150)))
    assert B.diag_lines(fam, "BendPickMP") == list(range(30))
    assert B.draw(fam, "diag", tmp_path / "d", tasks=["BendPickMP"])["written"] == 30


def test_split_is_deterministic_and_seeded(tmp_path):
    fam = make_fam(tmp_path)
    a, b = B.split(fam, "BendPickMP", 7), B.split(fam, "BendPickMP", 7)
    assert a == b and B.split(fam, "BendPickMP", 8) != a
    B.draw(fam, "accept", tmp_path / "x")
    B.draw(fam, "accept", tmp_path / "y")
    assert {p.name: p.read_bytes() for p in (tmp_path / "x").glob("*.json")} == {
        p.name: p.read_bytes() for p in (tmp_path / "y").glob("*.json")
    }


def test_incomplete_task_is_unavailable_and_blank_seed_refused(tmp_path):
    fam = make_fam(tmp_path)
    f = Path(fam["practice_dir"]) / "BendPickMP.jsonl"
    f.write_text("".join(f.read_text().splitlines(keepends=True)[:120]))  # level 2 stops at 20 lines
    r = B.draw(fam, "accept", tmp_path / "a")
    assert "BendPickMP" in r["unavailable"] and r["written"] == 60
    assert B.draw(fam, "diag", tmp_path / "d")["written"] == 60
    fam["pools"]["accept"] = dict(fam["pools"]["accept"], seed=None)
    fam["pools"]["sealed"] = dict(fam["pools"]["sealed"], seed=None)
    with pytest.raises(SystemExit):
        B.draw(fam, "accept", tmp_path / "z")


def test_load_real_family(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path))
    fam = BB.load("simple_q")
    assert fam["backend"] == "bench" and len(fam["tasks"]) == 5 and set(fam["rows"]) == set(fam["tasks"])
    assert all(isinstance(r["goal"], str) and r["goal"] for r in fam["rows"].values())
    assert fam["pools"]["accept"] == dict(seed=fam["pools"]["accept"]["seed"], n=150, per_level=10)
    assert fam["pools"]["sealed"]["n"] == 60 and fam["pools"]["diag"]["n"] == 30
    lay = BB.layout(fam)
    assert lay["caps"] == {**C.DEFAULT_CAPS, **fam["caps"]} and "repeat" not in lay["caps"]
    assert (
        lay["caps"]["check_min_net"] == 4 and lay["caps"]["alpha_task"] == 0.05 and lay["caps"]["manager"] == "program"
    )
    assert lay["caps"]["stop_no_progress"] == 2 and "probe_parallel" not in lay["caps"] and "min_net" not in fam["caps"]
    assert set(lay["class_suggest"]) == set(S.CLASS_SUGGEST) and lay["plan"] is None
    assert lay["skills"] == tmp_path / "simple_q" / "package" and (lay["skills"] / "__init__.py").exists()
    assert BB.base_dir(fam) != lay["skills"].resolve() and lay["prompts"].name == "prompts"
    assert not {"bed_root", "bed_dir", "bed_files", "base", "guard", "official_runner", "bed_env"} & set(fam)


def test_card_shows_where_the_grasp_sits():
    r = dict(label="BendPickMP_s1", task="BendPickMP", failure_class="grasp", why="grasp: x", steps=[],
             simple=dict(grasp_desc=[-0.066, 0.051, 0.146, -0.061, -0.27, -0.961, 0.995, 0.058, -0.08], score=0.94))  # fmt: skip
    line = next(x for x in S.card_text(r) if x.startswith("grasp chosen"))
    assert "5.1 cm left" in line and "14.6 cm up" in line and "(-0.06, -0.27, -0.96)" in line and "0.94" in line


def test_units_cover_every_episode_once_and_interleave_tasks():
    by = {t: [(f"{t}_s{i}", f"{t}/Q/{i}") for i in range(30)] for t in "ABCDE"}
    units = B._units(by, 3)
    assert sorted(x for _, items in units for x in items) == sorted(x for v in by.values() for x in v)
    assert [t for t, _ in units[:5]] == list("ABCDE")  # neighbours are different tasks
    assert max(len(i) for _, i in units) <= 13 and len(units) == 15
    small = B._units({"A": [("A_s0", "A/Q/0")] * 6}, 3)
    assert [len(i) for _, i in small] == [3, 3]
    assert B._units({"A": [("A_s0", "A/Q/0")]}, 3) == [("A", [("A_s0", "A/Q/0")])]


def test_stop_takes_the_runner_and_what_it_started(tmp_path):
    import threading
    import time

    mark = tmp_path / "pids"
    script = f"sleep 300 & echo $! > {mark}; echo $$ >> {mark}; sleep 300"
    stop = threading.Event()
    got = []
    th = threading.Thread(
        target=lambda: got.append(B._exec(["bash", "-c", script], dict(os.environ), None, stop, None, tree=True))
    )
    th.start()
    for _ in range(50):
        if mark.exists() and len(mark.read_text().split()) == 2:
            break
        time.sleep(0.1)
    pids = [int(x) for x in mark.read_text().split()]
    assert all(B._running(x) for x in pids)
    stop.set()
    th.join(30)
    assert got == [-9] and not any(B._running(x) for x in pids)


# ---------------------------------------------------------------------------------------- sandbox / leak
def test_leak_regexes():
    pats = [re.compile(x) for x in B.LEAK]
    hit = lambda s: any(p.search(s) for p in pats)  # noqa: E731
    for s in ("XMovePickTeleop/L0/e3", "use L2/e12 here", "Lhard", "EXAM_OPEN=1", "pool_accept/x", "pool_sealed"):
        assert hit(s), s
    for s in ("XMovePickTeleop/Q/3", "level 2", "BendPickMP_s14", "belle/ex"):
        assert not hit(s), s


# ------------------------------------------------------------------------- skill package and the hidden-ids convention
def test_load_skills_reads_the_parameter_names_from_the_source_without_importing(tmp_path):
    (tmp_path / "skill.py").write_text(
        "import not_a_module_that_exists\nK_ORIG = dict(stand=0.27, mode='a')\nK_CUT = dict(K_ORIG, exact_ticks=True)\n"
        "PLACING = dict(clear=0.1)\n"
    )
    p = B.load_skills(tmp_path, "x", dict(bed_keys=["feed_torque"]))
    assert p.apply_set(p, {"stand": 0.3, "stance_d": 0.1, "feed_torque": 2}) == {
        "stand": 0.3,
        "stance_d": 0.1,
        "feed_torque": 2,
    }
    assert set(p.parameters(p)) == {"stand", "mode", "exact_ticks", "clear"} and p.parameters(p)["stand"] == 0.3
    with pytest.raises(KeyError):
        p.apply_set(p, {"stnad": 1})


def test_hidden_ids_entry_is_read_back_by_load_and_drawn(tmp_path):
    fam = make_fam(tmp_path)
    fam["pools"]["accept"] = dict(n=2, seed=None, ids=None)
    ids = ["XMovePickTeleop/Q/60", "BendPickMP/Q/70"]
    fam["pools"]["accept"]["ids"] = ids
    res = B.draw(fam, "accept", tmp_path / "pool")
    assert res["written"] == 2 and sorted(p.name for p in (tmp_path / "pool").glob("*_s*.json")) == [
        "BendPickMP_s70.json",
        "XMovePickTeleop_s60.json",
    ]


# ------------------------------------------------------------------------- the seed is read-only; refused episodes
def test_the_seed_package_cannot_be_written_or_deleted(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "state"))
    fam = make_fam(tmp_path)
    seed = tmp_path / "bed" / AD["seed"]
    before = (seed / "__init__.py").read_text()
    work = B.ensure_package(fam)  # the working package is a copy, outside the seed
    assert work == tmp_path / "state" / "t_fam" / "package" and (work / "__init__.py").read_text() == before
    (work / "__init__.py").write_text("X = 2\n")  # editing the copy is fine
    assert (seed / "__init__.py").read_text() == before
    for path in (seed, seed / "__init__.py", seed / "sub" / "new.py", tmp_path / "bed"):
        with pytest.raises(B.BaseWriteRefused):
            B.guard_write(fam, path)
    with pytest.raises(B.BaseWriteRefused):
        B._write(seed / "round.json", {})
    with pytest.raises(B.BaseWriteRefused):
        B._copy(work, seed / "copy")
    with pytest.raises(B.BaseWriteRefused):
        B._cut(tmp_path / "bed", AD["seed"] + "/__init__.py")
    assert (seed / "__init__.py").read_text() == before


def test_leak_patterns_name_paths_not_the_code_folder():
    import re

    hits = lambda s: [i for i, rx in enumerate(B.LEAK) if re.search(rx, s)]  # noqa: E731
    assert hits('grep -rn "simple_eval_20261112" --include=*.py scripts bed_readonly_copy') == []
    assert hits("cat bed_readonly_copy/scripts/experiments/simple_eval_20261112/grasps_graspgen.py") == []
    assert hits("ls /workspace/code/locomanipulation-esc/outputs/simple_eval_20261112/x")
    assert hits("cat outputs/simple_eval_20261112/x/HandoverTeleop/a/dr-level-0/meta/episodes.jsonl")

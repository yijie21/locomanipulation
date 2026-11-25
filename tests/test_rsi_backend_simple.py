"""The SIMPLE backend of the RSI loop (scripts/rsi/backend_simple.py): records, pools, jobs, sandbox. No simulator."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsi"))
import backend_simple as B  # noqa: E402

TASKS = ["XMovePickTeleop", "BendPickMP"]


def row(level: int, i: int = 0) -> str:
    return json.dumps({"instruction": f"bend to pick up item {i}", "src": {"level": level, "seed": i}})


def make_practice(d: Path, tasks=TASKS, per_level: int = 50) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        d.joinpath(f"{t}.jsonl").write_text("".join(row(lv, i) + "\n" for lv in (0, 1, 2) for i in range(per_level)))


def make_fam(tmp: Path, **kw) -> dict:
    root = tmp / "bed"
    (root / "bed").mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("X = 1\n")
    practice = tmp / "practice"
    make_practice(practice)
    fam = dict(
        name="t_fam",
        backend="simple",
        tasks=TASKS,
        pools={"diag": {"seed": 0, "n": 20}, "accept": {"seed": 7, "n": 40}, "sealed": {"seed": 7, "n": 40}},
        bed_root=str(root),
        bed_dir="bed",
        bed_files=["bed/a.py"],
        track="official",
        package="pkg",
        base="pkg",
        guard="guard.py",
        practice_dir=str(practice),
        rules="rules_scene.md",
        prompts="prompts",
        plan=None,
        caps={"min_net": 3},
        sandbox={"dir": "bed_readonly_copy", "tree": [], "task_files": []},
    )
    fam.update(kw)
    (root / "bed" / "a.py").write_text("print(1)\n")
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
    fam = B.load("simple_q")
    assert fam["backend"] == "simple" and len(fam["tasks"]) == 5 and set(fam["rows"]) == set(fam["tasks"])
    assert all(isinstance(r["goal"], str) and r["goal"] for r in fam["rows"].values())
    assert fam["pools"]["accept"] == dict(seed=fam["pools"]["accept"]["seed"], n=150, per_level=10)
    assert fam["pools"]["sealed"]["n"] == 60 and fam["pools"]["diag"]["n"] == 30
    lay = B.layout(fam)
    assert lay["caps"] == {**B.FALLBACK_CAPS, **fam["caps"]}  # TODO 54: the family file holds the switches explicitly
    assert (
        lay["caps"]["check_min_net"] == 4 and lay["caps"]["alpha_task"] == 0.05 and lay["caps"]["manager"] == "program"
    )
    assert lay["caps"]["stop_no_progress"] == 2 and "probe_parallel" not in lay["caps"] and "min_net" not in fam["caps"]
    assert set(lay["class_suggest"]) == set(B.CLASSES) and lay["plan"] is None
    assert lay["skills"] == tmp_path / "simple_q" / "package" and (lay["skills"] / "__init__.py").exists()
    assert B.base_dir(fam) != lay["skills"].resolve() and lay["prompts"].name == "prompts"


# -------------------------------------------------------------------------------------------- signature
def test_signature(tmp_path):
    fam = make_fam(tmp_path)
    s = B.signature(fam, "probe")
    assert s == B.signature(fam, "mirror") and s != B.signature(fam, "gate")
    (Path(fam["bed_root"]) / "bed" / "a.py").write_text("print(2)\n")
    assert B.signature(fam, "probe") != s
    assert B.signature(dict(fam, track="truth"), "probe") != B.signature(fam, "probe")


# ------------------------------------------------------------------------------------------- run_jobs
FAKE_BED = r"""
import json, os, sys
a = sys.argv[1:]
assert a[0] == "batch"
ids = open(a[a.index("--ids") + 1]).read().split()
out = a[a.index("--out") + 1]
open(os.environ["FAKE_COUNT"], "a").write(json.dumps({"n": len(ids), "env": {k: os.environ.get(k) for k in
    ("SIMPLE_SKILLS", "SIMPLE_LOOP", "SIMPLE_PRACTICE", "SIMPLE_RUN_DIR", "EXAM_OPEN", "OMP_NUM_THREADS")},
    "setting": a[a.index("--setting") + 1]}) + "\n")
with open(out, "w") as f:
    for e in ids:
        f.write(json.dumps({"episode_id": e, "success": True, "t_success": 9.0, "loss": None, "fell": None,
            "why": "", "marks": {}, "phases": {"reach_grasp": 1.0}, "t_end": 9.0, "wall_s": 1.0,
            "comply": {"ok": True, "why": ""}}) + "\n")
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
    (root / "bed" / "fake_bed.py").write_text(FAKE_BED)
    (root / "bed" / "guard.py").write_text(FAKE_GUARD)
    fam["bed_cmd"] = [sys.executable, str(root / "bed" / "fake_bed.py")]
    count = tmp_path / "count.jsonl"
    monkeypatch.setenv("FAKE_COUNT", str(count))
    monkeypatch.setenv("EXAM_OPEN", "1")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    pool = tmp_path / "pool"
    B.draw(fam, "diag", pool, n=3)
    return fam, pool, count, root


def calls(count: Path) -> list[dict]:
    return [json.loads(x) for x in count.read_text().splitlines()] if count.exists() else []


def test_run_jobs_three_labels_then_nothing(rig, tmp_path):
    fam, pool, count, root = rig
    job = Job(root / "pkg", tmp_path / "out", {"grip": 0.5})
    assert B.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0,1") == {0: []}
    files = sorted(p.name for p in (tmp_path / "out").glob("*.json"))
    assert len(files) == 3
    r = json.loads((tmp_path / "out" / files[0]).read_text())
    assert r["ok"] and r["set"] == {"grip": 0.5} and r["motor"] == B.signature(fam, "probe")
    c = calls(count)
    assert len(c) == 1 and c[0]["n"] == 3 and json.loads(c[0]["setting"]) == {"grip": 0.5}
    e = c[0]["env"]
    assert e["SIMPLE_SKILLS"] == str((root / "pkg").resolve()) and e["SIMPLE_LOOP"] == "1" and e["EXAM_OPEN"] is None
    assert e["SIMPLE_PRACTICE"] == fam["practice_dir"] and e["OMP_NUM_THREADS"] == "2" and e["SIMPLE_RUN_DIR"]
    assert B.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0,1") == {0: []}
    assert len(calls(count)) == 1  # nothing ran again
    job2 = Job(root / "pkg", tmp_path / "out2", {}, labels=(files[0][:-5],))
    assert B.run_jobs([job, job2], fam, pool, 2, tmp_path / "log.txt", "0") == {0: [], 1: []}
    assert calls(count)[-1]["n"] == 1


def test_bed_fixed_switches_run_in_every_round_and_no_set_names_them(rig, tmp_path):
    fam, pool, count, root = rig
    before = B.signature(fam, "probe")
    fam["bed_fixed"] = {"_about": "rule 46", "feed_torque": True, "amo_arm_pos": True}
    assert B.bed_fixed(fam) == {"feed_torque": True, "amo_arm_pos": True}
    assert B.signature(fam, "probe") != before  # rounds of the old bed are not reused
    job = Job(root / "pkg", tmp_path / "outf", {"grip": 0.5, "feed_torque": False})
    assert B.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0") == {0: []}
    assert json.loads(calls(count)[0]["setting"]) == {"grip": 0.5, "feed_torque": True, "amo_arm_pos": True}
    r = json.loads(next((tmp_path / "outf").glob("*.json")).read_text())
    assert r["set"] == {"grip": 0.5, "feed_torque": False}  # the round keeps the job's own set
    (tmp_path / "pk").mkdir()
    (tmp_path / "pk" / "skill_simple.py").write_text("K_ORIG = dict(stand=0.27)\n")
    p = B.load_skills(tmp_path / "pk", "x", dict(bed_keys=["feed_torque", "see_self"], bed_fixed=fam["bed_fixed"]))
    assert p.apply_set(p, {"stand": 0.3, "see_self": True}) == {"stand": 0.3, "see_self": True}
    with pytest.raises(KeyError, match="fixed by the bed"):
        p.apply_set(p, {"feed_torque": False})


def test_the_gate_and_the_sealed_run_render_every_step(tmp_path):  # user 2026-11-23, bed_env_gate
    fam = make_fam(tmp_path)
    fam["bed_env"] = {"SIMPLE_LAZY_RENDER": "1"}
    probe, gate = B.signature(fam, "probe"), B.signature(fam, "gate")
    fam["bed_env_gate"] = {"_about": "a note", "SIMPLE_LAZY_RENDER": "0"}
    assert B.signature(fam, "probe") == probe and B.signature(fam, "gate") != gate  # the gate's rounds run again
    assert B.bed_env(fam, tmp_path, tmp_path, full=False)["SIMPLE_LAZY_RENDER"] == "1"
    e = B.bed_env(fam, tmp_path, tmp_path, full=True)
    assert e["SIMPLE_LAZY_RENDER"] == "0" and "_about" not in e


def test_card_shows_where_the_grasp_sits():
    r = dict(label="BendPickMP_s1", task="BendPickMP", failure_class="grasp", why="grasp: x", steps=[],
             simple=dict(grasp_desc=[-0.066, 0.051, 0.146, -0.061, -0.27, -0.961, 0.995, 0.058, -0.08], score=0.94))  # fmt: skip
    line = next(x for x in B.card_text(r) if x.startswith("grasp chosen"))
    assert "5.1 cm left" in line and "14.6 cm up" in line and "(-0.06, -0.27, -0.96)" in line and "0.94" in line


def test_run_dirs_differ_between_jobs(rig, tmp_path):
    fam, pool, count, root = rig
    jobs = [Job(root / "pkg", tmp_path / f"o{k}", {"k": k}) for k in range(2)]
    B.run_jobs(jobs, fam, pool, 1, tmp_path / "log.txt", "0")
    dirs = [c["env"]["SIMPLE_RUN_DIR"] for c in calls(count)]
    assert len(set(dirs)) == 2


def test_guard_failure_runs_nothing(rig, tmp_path):
    fam, pool, count, root = rig
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "__init__.py").write_text("")
    (cand / "bad.txt").write_text("x")
    job = Job(cand, tmp_path / "outg")
    assert B.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0") == {0: []}
    assert calls(count) == []
    rs = [json.loads(p.read_text()) for p in (tmp_path / "outg").glob("*.json")]
    assert len(rs) == 3 and all(not r["ok"] and r["why"] == "guard: uses truth" for r in rs)
    assert B.run_jobs([job], fam, pool, 2, tmp_path / "log.txt", "0") == {0: []} and calls(count) == []


def test_missing_records_stay_missing(rig, tmp_path):
    fam, pool, count, root = rig
    (root / "bed" / "fake_bed.py").write_text("import sys\nsys.exit(3)\n")
    out = B.run_jobs([Job(root / "pkg", tmp_path / "o")], fam, pool, 1, tmp_path / "log.txt", "0")
    assert len(out[0]) == 3


def test_gate_goes_through_official_runner(rig, tmp_path, monkeypatch):
    fam, pool, count, root = rig
    seen = []

    def fake(task, ids, package, setting, out, **kw):
        seen.append((task, list(ids), setting))
        Path(out).write_text("".join(json.dumps(dict(json.loads(FAKE_REC), episode_id=e)) + "\n" for e in ids))
        return 0

    monkeypatch.setattr(B, "official_runner", fake)
    r = B.run_jobs(
        [Job(root / "pkg", tmp_path / "og", {"a": 1})], fam, pool, 1, tmp_path / "log.txt", "0", where="gate"
    )
    assert r == {0: []} and sorted(x[0] for x in seen) == sorted({s[0] for s in seen})
    assert sum(len(x[1]) for x in seen) == 3
    rec0 = json.loads(next((tmp_path / "og").glob("*.json")).read_text())
    assert rec0["motor"] == B.signature(fam, "gate") != B.signature(fam, "probe")


def test_units_cover_every_episode_once_and_interleave_tasks():
    by = {t: [(f"{t}_s{i}", f"{t}/Q/{i}") for i in range(30)] for t in "ABCDE"}
    units = B._units(by, 3)
    assert sorted(x for _, items in units for x in items) == sorted(x for v in by.values() for x in v)
    assert [t for t, _ in units[:5]] == list("ABCDE")  # neighbours are different tasks
    assert max(len(i) for _, i in units) <= 13 and len(units) == 15
    small = B._units({"A": [("A_s0", "A/Q/0")] * 6}, 3)
    assert [len(i) for _, i in small] == [3, 3]
    assert B._units({"A": [("A_s0", "A/Q/0")]}, 3) == [("A", [("A_s0", "A/Q/0")])]


def test_official_units_run_side_by_side_in_their_own_directories(rig, tmp_path, monkeypatch):
    fam, pool, count, root = rig
    dirs, live, peak = [], [0], [0]
    lock = __import__("threading").Lock()

    def fake(task, ids, package, setting, out, env=None, **kw):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            dirs.append((Path(out).parent, env["SIMPLE_RUN_DIR"]))
        __import__("time").sleep(0.2)
        Path(out).write_text("".join(json.dumps(dict(json.loads(FAKE_REC), episode_id=e)) + "\n" for e in ids))
        with lock:
            live[0] -= 1
        return 0

    monkeypatch.setattr(B, "official_runner", fake)
    r = B.run_jobs([Job(root / "pkg", tmp_path / "ou")], fam, pool, 1, tmp_path / "log.txt", "0", where="gate")
    assert r == {0: []}
    assert all(str(d) == e for d, e in dirs) and len({d for d, _ in dirs}) == len(dirs)
    assert peak[0] == min(3, len(dirs))


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


FAKE_REC = json.dumps(
    {
        "success": True,
        "t_success": 1.0,
        "loss": None,
        "fell": None,
        "phases": {"x": 1.0},
        "t_end": 1.0,
        "comply": {"ok": True},
    }  # fmt: skip
)


# ---------------------------------------------------------------------------------------- sandbox / leak
def test_leak_regexes():
    pats = [re.compile(x) for x in B.LEAK]
    hit = lambda s: any(p.search(s) for p in pats)  # noqa: E731
    for s in ("XMovePickTeleop/L0/e3", "use L2/e12 here", "Lhard", "EXAM_OPEN=1", "pool_accept/x", "pool_sealed"):
        assert hit(s), s
    for s in ("XMovePickTeleop/Q/3", "level 2", "BendPickMP_s14", "belle/ex"):
        assert not hit(s), s


def test_sandbox_and_audit(tmp_path):
    fam = make_fam(tmp_path)
    fam["bed_files"] = ["bed/a.py"]
    fam["sandbox"] = {"dir": "bed_readonly_copy", "tree": [], "task_files": []}
    orig = tmp_path / "origin"  # what the sandbox's third_party link points at
    (orig / "SIMPLE" / "data" / "teleop_wbc" / "XMovePickTeleop").mkdir(parents=True)
    (orig / "SIMPLE" / "data" / "teleop_wbc" / "XMovePickTeleop" / "episodes.jsonl").write_text("{}\n")
    (orig / "SIMPLE" / "data" / "keep").mkdir()
    (orig / "SIMPLE" / "data" / "keep" / "f.txt").write_text("ok")
    dest = tmp_path / "sb"
    (dest / "outputs" / "simple_eval_20261112").mkdir(parents=True)
    (dest / "outputs" / "escalation_20261116" / "official" / "exam").mkdir(parents=True)
    (dest / "outputs" / "escalation_20261116" / "practice").mkdir(parents=True)
    (dest / "docs" / "handoff").mkdir(parents=True)
    (dest / "docs" / "handoff" / "escalation-plan-1.md").write_text("x")
    (dest / "docs" / "handoff" / "benchmarks-1.md").write_text("x")
    (dest / "docs" / "handoff" / "keep.md").write_text("x")
    (dest / "third_party").symlink_to(orig)
    ff = dest / "scripts" / "rsi" / "families" / "t_fam.json"
    ff.parent.mkdir(parents=True)
    ff.write_text(json.dumps(fam))
    leaked = dest / "bed_readonly_copy" / "pkg" / "__init__.py"
    assert any(x.startswith("path ") for x in B.audit(dest))  # before: forbidden paths are there
    B.sandbox(dest, fam)
    assert B.audit(dest) == []
    assert (orig / "SIMPLE" / "data" / "teleop_wbc").exists()  # the origin is untouched
    assert (dest / "third_party" / "SIMPLE" / "data" / "keep" / "f.txt").read_text() == "ok"
    assert (dest / "docs" / "handoff" / "keep.md").exists()
    nest = dest / "bed_readonly_copy"
    assert (nest / "pkg" / "__init__.py").exists() and not (nest / "bed" / "a.py").stat().st_mode & 0o222
    hidden = B.hidden_ids(fam)
    for t in TASKS:
        vis = (nest / "practice" / f"{t}.jsonl").read_text().splitlines()
        assert len(vis) == 150
        assert all((x == "{}") == (f"{t}/Q/{i}" in hidden) for i, x in enumerate(vis))
        full = dest / B.HIDDEN / "t_fam" / "practice" / f"{t}.jsonl"
        assert full.read_text() == (Path(fam["practice_dir"]) / f"{t}.jsonl").read_text()
    new = json.loads(ff.read_text())
    assert new["bed_root"] == fam["bed_root"] and new["practice_dir"] == "bed_readonly_copy/practice"
    # a leak in a visible file, an unblanked hidden row and a re-created forbidden path are all reported
    leaked.write_text("# see L1/e4 and EXAM_OPEN\n")
    (nest / "practice" / "BendPickMP.jsonl").write_text(row(1) * 1 + "\n" + "".join(row(1) + "\n" for _ in range(20)))
    (dest / "outputs" / "simple_eval_20261112").mkdir()
    out = B.audit(dest)
    assert any("pkg/__init__.py" in x and "L1/e4" in x for x in out)
    assert any("practice row not blank" in x for x in out)
    assert any("simple_eval_20261112" in x for x in out)
    # the audit does not read the hidden tree
    (dest / B.HIDDEN / "t_fam" / "note.txt").write_text("EXAM_OPEN L0/e0")
    assert not any("note.txt" in x for x in B.audit(dest))


# ------------------------------------------------------------------------- skill package and the hidden-ids convention
def test_load_skills_reads_the_parameter_names_from_the_source_without_importing(tmp_path):
    (tmp_path / "skill_simple.py").write_text(
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
    seed = tmp_path / "bed" / "pkg"
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
        B._cut(tmp_path / "bed", "pkg/__init__.py")
    assert (seed / "__init__.py").read_text() == before


def test_run_jobs_refuses_a_blank_or_missing_episode(rig, tmp_path):
    fam, pool, count, root = rig
    prac = Path(fam["practice_dir"])
    f = prac / "XMovePickTeleop.jsonl"
    lines = f.read_text().splitlines()
    lines[3] = "{}"  # a hidden row, blanked as a sandbox does
    f.write_text("\n".join(lines) + "\n")
    (pool / "XMovePickTeleop_s3.json").write_text(
        json.dumps(dict(task="XMovePickTeleop", episode_id="XMovePickTeleop/Q/3"))
    )
    (pool / "XMovePickTeleop_s999.json").write_text(
        json.dumps(dict(task="XMovePickTeleop", episode_id="XMovePickTeleop/Q/999"))
    )
    job = Job(root / "pkg", tmp_path / "out", {}, labels=("XMovePickTeleop_s3", "XMovePickTeleop_s999"))
    assert B.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0") == {0: []}
    for x in job.labels:
        r = json.loads((tmp_path / "out" / f"{x}.json").read_text())
        assert r["ok"] is False and r["why"] == "not a diagnostic episode"
    assert calls(count) == []  # the bed never started


def test_leak_patterns_name_paths_not_the_code_folder():
    import re

    hits = lambda s: [i for i, rx in enumerate(B.LEAK) if re.search(rx, s)]  # noqa: E731
    assert hits('grep -rn "simple_eval_20261112" --include=*.py scripts bed_readonly_copy') == []
    assert hits("cat bed_readonly_copy/scripts/experiments/simple_eval_20261112/grasps_graspgen.py") == []
    assert hits("ls /workspace/code/locomanipulation-esc/outputs/simple_eval_20261112/x")
    assert hits("cat outputs/simple_eval_20261112/x/HandoverTeleop/a/dr-level-0/meta/episodes.jsonl")

"""The articulated-task side of the SIMPLE backend (family simple_art): pools, blame, steps and cards. No simulator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsi"))
import backend_simple as B  # noqa: E402

ART_TASKS = ["OpenOvenTeleop", "CloseDoorTeleop", "OpenFaucetTeleop", "OpenTrashCanTeleop"]
PHASES = {"walk": 3.0, "reach": 2.0, "close": 1.0, "move": 4.0, "release": 1.0}


def art_rec(**kw) -> dict:
    base = dict(
        episode_id="OpenOvenTeleop/Q/3",
        success=False,
        t_success=None,
        loss=None,
        fell=None,
        why="",
        phases=dict(PHASES),
        t_end=11.0,
        comply={"ok": False, "why": ""},
        art=dict(joint="joint_1", sign=1, start_deg=0.0, best_deg=20.0, end_deg=15.0, need_deg=80.0, t_move=6.0),
        art_touch=dict(t_hand=4.5, hand_deg=18.0, t_body=None, body_links=[]),
    )
    base.update(kw)
    return base


def rd(r: dict) -> dict:
    return B.to_round(r, "OpenOvenTeleop_s3", "h", {}, "sig")


def blamed(o: dict) -> list[str]:
    return [s["step"] for s in o["steps"] if not s["ok"]]


def test_real_family_loads_with_four_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path))
    fam = B.load("simple_art")
    assert fam["tasks"] == ART_TASKS and set(fam["rows"]) == set(ART_TASKS)
    assert fam["pools"]["accept"]["n"] == 120 and fam["pools"]["accept"]["per_level"] == 10
    assert fam["pools"]["sealed"]["n"] == 60 and fam["pools"]["diag"]["n"] == 30
    assert B.bed_fixed(fam) == {"feed_torque": True}
    assert fam["rules"] == "scripts/rsi/rules_simple_art.md" and fam["prompts"] == "scripts/rsi/prompts_simple_art"
    assert [Path(t).name for t in fam["sandbox"]["task_files"]] == [
        f"g1_wholebody_{x}_teleop.py" for x in ("open_oven", "close_door", "open_faucet", "open_trash_can")
    ]
    assert "scripts/rsi/rules_simple_art.md" in fam["sandbox"]["keep"]
    assert "scripts/rsi/families/simple_art.json" in fam["sandbox"]["overlay"]
    assert fam["rows"]["OpenFaucetTeleop"]["goal"] == "turn the faucet"


def test_pools_have_the_right_sizes(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path))
    fam = B.load("simple_art")
    pdir = tmp_path / "practice2"
    pdir.mkdir()
    for t in ART_TASKS:
        rows = [
            json.dumps({"instruction": "x", "src": {"level": lv, "seed": i}}) for lv in (0, 1, 2) for i in range(50)
        ]
        (pdir / f"{t}.jsonl").write_text("\n".join(rows) + "\n")
    fam["practice_dir"] = str(pdir)
    got = {p: B.draw(fam, p, tmp_path / p)["written"] for p in ("diag", "accept", "sealed")}
    assert got == {"diag": 120, "accept": 120, "sealed": 240}


@pytest.mark.parametrize(
    "cls,phase", [("deliver", "reach"), ("grasp", "close"), ("short", "move"), ("collide", "move"),
                  ("fall", "release"), ("timeout", "release")]
)  # fmt: skip
def test_articulated_blame(cls, phase):
    o = rd(art_rec(loss=cls, why="because"))
    assert o["failure_class"] == cls and blamed(o) == [phase]


def test_articulated_steps_carry_the_joint_facts():
    o = rd(art_rec(loss="short", why=""))
    st = {s["step"]: s for s in o["steps"]}
    assert (st["move"]["start_deg"], st["move"]["best_deg"], st["move"]["need_deg"]) == (0.0, 20.0, 80.0)
    assert st["reach"]["t_hand"] == 4.5 and st["close"]["t_hand"] == 4.5
    o = rd(art_rec(loss="deliver", art_touch=dict(t_hand=None, hand_deg=0.0, t_body=None, body_links=[])))
    st = {s["step"]: s for s in o["steps"]}
    assert "t_hand" not in st["reach"] and st["reach"]["why"] == "deliver"


def test_compliant_success_is_a_win_with_no_blame():
    o = rd(art_rec(success=True, t_success=9.0, comply={"ok": True, "why": ""},
                   art=dict(joint="joint_2", sign=1, start_deg=0.0, best_deg=85.0, end_deg=85.0, need_deg=80.0,
                            t_move=5.0)))  # fmt: skip
    assert o["ok"] and blamed(o) == [] and o["dropped"] is False


def test_body_touch_card_and_no_grasp_line():
    r = art_rec(loss="collide", why="body touched",
                art_touch=dict(t_hand=4.5, hand_deg=2.0, t_body=3.0, body_links=["left_knee_link", "pelvis"]))  # fmt: skip
    text = "\n".join(B.card_text(rd(r)))
    assert "instruction: move forward to the oven and open it" in text
    assert "started at 0.0 deg, furthest 20.0 deg, needed 80.0 deg" in text
    assert "first touched the moving part at 4.5 s" in text and "2.0 deg were made with a hand on it" in text
    assert "left_knee_link, pelvis" in text and "from 3.0 s" in text
    assert "grasp chosen" not in text and "palm error" not in text and "mark " not in text
    assert "failed phase: move" in text


def test_card_without_a_hand_touch_and_downward_joint():
    r = art_rec(loss="deliver", art=dict(joint="door", sign=-1, start_deg=60.0, best_deg=60.0, end_deg=60.0,
                                          need_deg=-9.0, t_move=None),
                art_touch=dict(t_hand=None, hand_deg=0.0, t_body=None, body_links=[]))  # fmt: skip
    text = "\n".join(B.card_text(rd(r)))
    assert "no hand ever touched the moving part" in text and "down" in text and "failed phase: reach" in text


def test_articulated_record_missing_fields_does_not_crash():
    o = rd(art_rec(loss="short", art_touch=None, marks=None, phases={}))
    assert o["steps"] == [] and B.card_text(o)
    o = rd(art_rec(loss="short", art=dict(joint="j")))
    assert B.card_text(o)


def test_cards_and_segments_over_articulated_rounds(tmp_path):
    rounds = [rd(art_rec(loss="short")), rd(art_rec(loss="short")), rd(art_rec(loss="deliver"))]
    text = B.cards(rounds, tmp_path / "c").read_text()
    seg = next(x for x in text.splitlines() if x.startswith("- ") and "lost)" in x)
    assert "move 2/3 HOT" in seg and "reach 1/3" in seg


def test_pick_record_is_unchanged():
    ph = {"look": 2.0, "reach_grasp": 3.0, "close": 1.0, "lift": 2.0, "lift_hold": 1.0}
    r = dict(episode_id="XMovePickTeleop/Q/3", success=False, loss="grasp", why="slipped", phases=ph, t_end=9.0,
             marks={"after_lift": {"rise_cm": 1.0, "held": False}}, palm_err_cm=1.0, reach_ok=[True],
             comply={"ok": False, "why": ""})  # fmt: skip
    o = B.to_round(r, "XMovePickTeleop_s3", "h", {}, "s")
    assert blamed(o) == ["lift"] and o["steps"][3]["rise_cm"] == 1.0
    text = "\n".join(B.card_text(o))
    assert "joint" not in text and "failed phase: lift" in text
    o = B.to_round(dict(r, loss="collide"), "XMovePickTeleop_s3", "h", {}, "s")
    assert blamed(o) == ["close"]


# ----------------------------------------------------------------------------------- bench adapter (TODO 49.6a)
BENCH_REAL = Path("/workspace/code/locomanipulation-bench")
needs_bench = pytest.mark.skipif(
    not (BENCH_REAL / "src/locomani/bench/simple/guard.py").exists(), reason="no bench worktree"
)


def bench_fam(tmp_path, monkeypatch, **bench_kw) -> dict:
    """A bench family on a small fake bench root; the switches resolve without the adapter (their signature is the values)."""
    root = tmp_path / "benchroot"
    for rel, text in {
        "src/locomani/bench/options.py": "o = 1\n",
        "src/locomani/bench/simple/run.py": "r = 1\n",
        "src/locomani/bench/simple/seed/skill.py": "K_ORIG = dict(stand=0.27)\nK_ART = dict(art_obj='', art_gap=0.3)\n"
        "ART_TASK = dict(A=dict(art_obj='oven', art_pitch=90.0), B=dict(art_roll=5, art_gap=0.1))\n",
        "src/locomani/bench/simple/seed/__init__.py": "",
        "src/locomani/perception/head.py": "h = 1\n",
        "src/locomani/backends/scalebfm/motor.py": "m = 1\n",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    fam = json.loads((Path(__file__).resolve().parents[1] / "scripts/rsi/families/simple_art.json").read_text())
    bench_kw.setdefault("memo", False)  # (the family ships with it on, TODO 53; the tests below choose)
    fam["bench"] = dict(fam["bench"], root=str(root), **bench_kw)
    monkeypatch.setattr(
        B,
        "bench_record",
        lambda f: dict(
            signature="simple/" + "-".join(f"{k}:{v}" for k, v in sorted(f["bench"].get("options", {}).items()))
        ),
    )
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "pk"))
    return fam


def test_family_file_names_the_bench(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path))
    fam = B.load("simple_art")
    assert fam["bench"]["name"] == "simple" and fam["bench"]["root"] in (
        "/workspace/code/locomanipulation-bench",
        "/workspace/code/locomanipulation-bench-loop",
    )
    assert fam["bench"]["options"] == {"target": "perception", "sensor": "stereo"} and fam["bench"]["config"] == {}
    assert not {"bed_env", "bed_env_gate", "official_runner", "bed_files"} & set(fam)  # the block replaces them
    root = Path(fam["bench"]["root"])  # the bench worktree, or the detached pin the live loop uses
    assert B.bed_root(fam) == root and B.base_dir(fam) == (root / B.BENCH_SEED).resolve()


def test_bench_files_signature_and_switches(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch)
    files = B.bed_files(fam)
    assert "src/locomani/bench/simple/run.py" in files and "src/locomani/perception/head.py" in files
    assert "src/locomani/backends/scalebfm/motor.py" in files and "src/locomani/bench/options.py" in files
    s = B.signature(fam, "probe")
    assert B.signature(fam, "gate") == s  # TODO 52: every round renders lazily with warm-up, the gate included
    every = dict(fam, bench=dict(fam["bench"], render="every_step"))
    assert B.signature(every, "probe") != s and B.signature(every, "gate") == B.signature(
        every, "probe"
    )  # the explicit every-step mode is its own bed
    other = dict(fam, bench=dict(fam["bench"], options={"target": "perception", "sensor": "rgbd"}))
    assert B.signature(other, "probe") != s  # two switch sets never share a signature
    (Path(fam["bench"]["root"]) / "src/locomani/bench/simple/run.py").write_text("r = 2\n")
    assert B.signature(fam, "probe") != s  # an adapter file changed


def test_bench_requires_the_official_probe_bed_and_owns_four_keys(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="no mirror bed"):
        B.signature(dict(fam, probe_bed="mirror"), "probe")
    with pytest.raises(ValueError, match="lazy_render"):
        B.signature(dict(fam, bench=dict(fam["bench"], config={"lazy_render": True})), "probe")


def test_package_parameters_read_skill_py_with_the_art_keys(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch)
    p = B.package_parameters(B.base_dir(fam))
    assert p["stand"] == 0.27 and p["art_obj"] == "" and p["art_gap"] == 0.3  # K_ART's value wins
    assert p["art_pitch"] == 90.0 and p["art_roll"] == 5  # ART_TASK's keys, first value
    sk = B.load_skills(B.base_dir(fam), family=fam)
    assert sk.apply_set(sk, {"art_pitch": 80.0}) == {"art_pitch": 80.0}
    with pytest.raises(KeyError):
        sk.apply_set(sk, {"nope": 1})
    old = tmp_path / "old"
    old.mkdir()
    (old / "skill_simple.py").write_text("K_ORIG = dict(stand=0.27)\nK_ART = dict(art_obj='')\n")
    assert B.package_parameters(old) == {"stand": 0.27}  # the esc layout is read as before


def runner_call(fam, monkeypatch, tmp_path, full: bool, setting=None):
    seen = {}

    def fake(argv, env, log, stop, cwd, tree=False):
        seen.update(argv=argv, env=env, cwd=cwd, tree=tree)
        return 0

    monkeypatch.setattr(B, "_exec", fake)
    pkg = tmp_path / "pkg"
    pkg.mkdir(exist_ok=True)
    monkeypatch.setenv("SIMPLE_RUN_DIR", "inherited")
    monkeypatch.setenv("SIMPLE_LOOP", "1")
    B.official_runner(
        "OpenOvenTeleop",
        ["OpenOvenTeleop/Q/3"],
        pkg,
        setting or {"art_gap": 0.2},
        tmp_path / "u00" / "rows.jsonl",
        family=fam,
        full=full,
    )
    return seen


def pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_bench_runner_argv_diagnostic_and_gate(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch, config={"slots": [1, 2], "batch_tries": 2})
    (tmp_path / "u00").mkdir()
    d = runner_call(fam, monkeypatch, tmp_path, False, {"art_gap": 0.2, "stance_d": 0.05})
    a = d["argv"]
    assert a[:6] == ["/venv/main/bin/python", "-m", "locomani.bench.simple", "run", "OpenOvenTeleop", "--ids"]
    assert Path(a[6]).read_text() == "OpenOvenTeleop/Q/3\n"
    assert json.loads(pairs(a, "--setting")[0]) == {"art_gap": 0.2} and json.loads(pairs(a, "--stance")[0]) == {
        "d": 0.05
    }
    assert pairs(a, "--out") == [str(tmp_path / "u00" / "rows.jsonl")] and pairs(a, "--skills") == [
        str((tmp_path / "pkg").resolve())
    ]
    assert pairs(a, "--bench-set") == ["target=perception", "sensor=stereo"]
    sets = pairs(a, "--set")
    assert sets == [
        "kinds=Q",
        f"practice_dir={B.practice_dir(fam, False)}",
        f"run_dir={tmp_path / 'u00'}",
        "lazy_render=true",
        "lazy_warm=48",
        "slots=1,2",
        "batch_tries=2",
    ]
    assert d["cwd"] == Path(fam["bench"]["root"]) and d["tree"] is True and d["env"]["PYTHONPATH"] == "src"
    assert not [k for k in d["env"] if k.startswith("SIMPLE_")] and "EXAM_OPEN" not in d["env"]
    g = runner_call(fam, monkeypatch, tmp_path, True)["argv"]
    assert (
        "lazy_render=true" in pairs(g, "--set") and "lazy_warm=48" in pairs(g, "--set") and "--stance" not in g
    )  # (gate: lazy too)
    fam_every = dict(fam, bench=dict(fam["bench"], render="every_step"))
    ge = runner_call(fam_every, monkeypatch, tmp_path, True)["argv"]
    assert "lazy_render=false" in pairs(ge, "--set") and "lazy_warm=48" not in pairs(
        ge, "--set"
    )  # the explicit every-step mode


def test_bench_run_jobs_at_the_gate(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch)
    pdir = tmp_path / "practice"
    pdir.mkdir()
    for t in ART_TASKS:
        (pdir / f"{t}.jsonl").write_text(
            "".join(json.dumps(dict(src=dict(level=lv, seed=i))) + "\n" for lv in (0, 1, 2) for i in range(50))
        )
    fam["practice_dir"] = str(pdir)
    pool = tmp_path / "pool"
    B.draw(fam, "diag", pool, n=4)
    got = []

    def fake(task, ids, package, setting, out, env=None, full=None, **kw):
        got.append((task, full, setting, env))
        rec = dict(
            episode_id=None,
            success=True,
            comply={"ok": True, "why": ""},
            phases={"walk": 1.0},
            t_end=5.0,
            bench=dict(signature="simple/x"),
        )
        Path(out).write_text("".join(json.dumps(dict(rec, episode_id=e)) + "\n" for e in ids))
        return 0

    monkeypatch.setattr(B, "official_runner", fake)
    pk = tmp_path / "pkgjob"
    pk.mkdir()
    (pk / "skill.py").write_text("K_ORIG = dict(stand=0.27)\n")
    (pk / "__init__.py").write_text("")
    monkeypatch.setattr(B, "run_guard", lambda *a: None)
    job = type("J", (), dict(skills=pk, out=tmp_path / "o", overrides={"art_gap": 0.1}, labels=None))()
    assert B.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0", where="gate") == {0: []}
    assert got and all(g[1] is True and g[2]["feed_torque"] is True for g in got)
    assert not [k for g in got for k in g[3] if k.startswith("SIMPLE_")]
    r = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert r["bench"] == "simple/x" and r["motor"] == B.signature(fam, "gate")
    assert "bench: simple/x" in B.card_text(dict(r, ok=False, failure_class="grasp"))
    text = B.cards([dict(r, ok=False, failure_class="grasp")], tmp_path / "c").read_text()
    assert "bench: simple/x" in text.splitlines()[:4]


@needs_bench
def test_bench_guard_and_seed_are_the_real_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path))
    fam = B.load("simple_art")
    base = B.base_dir(fam)
    p = B.package_parameters(base)
    assert "art_obj" in p and "art_sweep" in p and "stand" in p
    off = dict(fam, bench=dict(fam["bench"], memo=False))
    assert B.signature(off, "probe") == B.signature(off, "gate")  # (one render mode for every round, TODO 52)
    assert fam["bench"]["memo"] is True and B.signature(fam, "gate") == B.signature(
        off, "gate"
    )  # TODO 53: on, outside the gate
    assert B.run_guard(fam, base, base) is None
    import shutil

    cand = tmp_path / "cand"
    shutil.copytree(base, cand, ignore=shutil.ignore_patterns("__pycache__"))
    assert B.run_guard(fam, base, cand) is None
    (cand / "pick.py").write_text((cand / "pick.py").read_text() + "\nx = open\n")
    assert "R4" in B.run_guard(fam, base, cand)
    (cand / "pick.py").write_text((base / "pick.py").read_text())
    (cand / "extra.py").write_text("")
    assert "R1" in B.run_guard(fam, base, cand)


def test_memo_store_is_outside_the_sandbox_and_the_hidden_tree(tmp_path, monkeypatch):
    """TODO 53: the disk store of saved states lives in the loop's scratch area: not in a sandbox a role reads, not in the
    hidden tree, not in the repository."""
    fam = bench_fam(tmp_path, monkeypatch, memo=True)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TMPDIR", str(scratch))
    store = B.memo_store_dir(fam)
    assert store.parent == scratch.resolve() and store.name == "simple_simple_art_memo"
    sandbox = tmp_path / "sandbox"
    hidden = (B.REPO / B.HIDDEN).resolve()
    for wall in (sandbox, hidden, B.REPO.resolve(), (B.REPO / "third_party").resolve().parent):
        assert wall not in store.parents and store != wall
    for inside in (
        B.REPO / "outputs" / "x",
        hidden / "pool",
        B.REPO / "tmp",
    ):  # a scratch area put inside a wall is refused
        monkeypatch.setenv("TMPDIR", str(inside))
        with pytest.raises(ValueError, match="memo store"):
            B.memo_store_dir(fam)
    monkeypatch.setenv("TMPDIR", str(scratch))
    argv = " ".join(B.bench_argv(fam, "OpenOvenTeleop", "ids", {}, None, "out", tmp_path, tmp_path / "run", False))
    assert "memo_dir=" + str(store) in argv
    full = " ".join(B.bench_argv(fam, "OpenOvenTeleop", "ids", {}, None, "out", tmp_path, tmp_path / "run", True))
    assert "--set memo" not in full  # the gate and the sealed set: neither read nor write the store


def test_bench_sandbox_copies_the_adapter_and_blanks_hidden_rows(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch)
    pdir = tmp_path / "practice"
    pdir.mkdir()
    for t in ART_TASKS:
        (pdir / f"{t}.jsonl").write_text(
            "".join(json.dumps(dict(src=dict(level=lv, seed=i))) + "\n" for lv in (0, 1, 2) for i in range(50))
        )
    fam["practice_dir"] = str(pdir)
    fam["sandbox"] = dict(fam["sandbox"], task_files=[])
    monkeypatch.setattr(B, "HIDDEN", "hid")
    dest = tmp_path / "sb"
    (dest / "scripts/rsi/families").mkdir(parents=True)
    (dest / "scripts/rsi/families/simple_art.json").write_text(json.dumps(fam))
    B.sandbox(dest, fam)
    nest = dest / "bed_readonly_copy"
    assert (nest / B.BENCH_SEED / "skill.py").exists() and (
        nest / "src/locomani/bench/simple/run.py"
    ).stat().st_mode & 0o222 == 0
    assert B.audit(dest) == []
    first = (nest / "practice" / "OpenOvenTeleop.jsonl").read_text().splitlines()
    assert first[0] != "{}"  # the diagnostic rows stay visible
    assert sum(x == "{}" for x in first) == 120


def test_render_mode_and_memo_switches_of_the_bench_family(tmp_path, monkeypatch):
    """TODO 52: bench.render is lazy (default) or every_step; bench.memo turns on saved-state replay outside the gate."""
    fam = bench_fam(tmp_path, monkeypatch)
    assert B.render_sets(fam) == ["lazy_render=true", "lazy_warm=48"] and B.memo_sets(fam, False) == []
    every = dict(fam, bench=dict(fam["bench"], render="every_step"))
    assert B.render_sets(every) == ["lazy_render=false"]
    with pytest.raises(ValueError, match="render"):
        B.check_bench(dict(fam, bench=dict(fam["bench"], render="sometimes")))
    memo = dict(fam, bench=dict(fam["bench"], memo=True))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    sets = B.memo_sets(memo, False)
    assert sets[0] == "memo=true" and sets[1:] == [
        f"memo_dir={tmp_path / 'scratch' / 'simple_simple_art_memo'}",
        f"memo_sig={B.signature(memo, 'probe')}",
        "memo_gb=20",
    ]
    assert B.memo_sets(memo, True) == []  # never at the gate or the sealed set: neither read nor written
    assert B.memo_sets(dict(fam, bench=dict(memo["bench"], memo_gb=5)), False)[-1] == "memo_gb=5"
    assert B.signature(memo, "probe") != B.signature(fam, "probe") and B.signature(memo, "gate") == B.signature(
        fam, "gate"
    )
    for k in ("memo_dir", "memo_sig", "memo_gb"):  # keys the backend sets itself
        with pytest.raises(ValueError, match=k):
            B.check_bench(dict(fam, bench=dict(fam["bench"], config={k: "x"})))
    with pytest.raises(ValueError, match="memo"):
        B.check_bench(dict(fam, bench=dict(fam["bench"], config={"memo": True})))

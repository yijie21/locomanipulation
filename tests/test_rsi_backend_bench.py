"""The loop's one generic backend for bench adapters (scripts/rsi/backend_bench.py, TODO 49.6b), on both adapters
(SIMPLE and LocoManiLab). No simulator: the adapter is a small fake bench root, the runner a fake."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

RSI = Path(__file__).resolve().parents[1] / "scripts" / "rsi"
sys.path.insert(0, str(RSI))
import backend  # noqa: E402
import backend_bench as BB  # noqa: E402
import bench_common as C  # noqa: E402
import bench_lml as BL  # noqa: E402
import bench_lml as LL  # noqa: E402
import bench_simple as B  # noqa: E402

BENCH_REAL = Path("/workspace/code/locomanipulation-bench")
needs_lml = pytest.mark.skipif(
    not (BENCH_REAL / "src/locomani/bench/locomanilab/guard.py").exists(), reason="no bench worktree"
)
FAMILIES = ("lml_q_bench", "lml_fridge_bench", "lml_drawer_bench")
ART_TASKS = ["OpenOvenTeleop", "CloseDoorTeleop", "OpenFaucetTeleop", "OpenTrashCanTeleop"]


def fake_root(tmp_path: Path, name: str) -> Path:
    """A tiny bench root: the files the adapter's table names (a few of each glob), a seed package."""
    root = tmp_path / f"root_{name}"
    files = {
        "src/locomani/bench/options.py": "o = 1\n",
        "src/locomani/perception/head.py": "h = 1\n",
        "src/locomani/backends/scalebfm/motor.py": "m = 1\n",
        "src/locomani/bench/simple/run.py": "r = 1\n",
        "src/locomani/bench/simple/body.py": "b = 1\n",
    }
    if name == "simple":
        files.update(
            {
                "src/locomani/bench/simple/seed/skill.py": "K_ORIG = dict(stand=0.27)\nK_ART = dict(art_obj='', art_gap=0.3)\n"
                "ART_TASK = dict(A=dict(art_obj='oven', art_pitch=90.0), B=dict(art_roll=5, art_gap=0.1))\n",
                "src/locomani/bench/simple/seed/__init__.py": "",
            }
        )
    else:
        L = "src/locomani/bench/locomanilab/"
        files.update({
            L + "run.py": "r = 1\n", L + "client.py": "c = 1\n", L + "comply.py": "c = 1\n", L + "loss.py": "l = 1\n",
            L + "guard.py": "g = 1\n", L + "server.py": "SERVER = 1\n", L + "truth.py": "T = 1\n", L + "config.py": "C = 1\n",
            L + "seed/skill.py": "K_ORIG = dict(stand=0.27)\nK_CUT = dict(speed=0.3, wait=1.0)\nPLACING = dict(drop=0.02)\n",
            L + "seed/__init__.py": "EXTRA = dict(lift=0.34, nav_gap=0.30)\n",
            L + "seed/pick.py": "p = 1\n",
        })  # fmt: skip
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def bench_fam(tmp_path, monkeypatch, family: str, **bench_kw) -> dict:
    """A bench family (the real family file) on a fake bench root; the switches resolve without the adapter."""
    fam = json.loads((RSI / "families" / f"{family}.json").read_text())
    root = fake_root(tmp_path, fam["bench"]["name"])
    if family == "simple_art":
        bench_kw.setdefault(
            "memo", False
        )  # (the family ships with it on, TODO 53; test_memo_sets_of_the_generic_backend)
    fam["bench"] = dict(fam["bench"], root=str(root), **bench_kw)
    rec = lambda f: dict(
        signature=f["bench"]["name"]
        + "/"
        + "-".join(f"{k}:{v}" for k, v in sorted(f["bench"].get("options", {}).items()))
    )  # noqa: E731
    monkeypatch.setattr(BB, "bench_record", rec)
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "pk"))
    return fam


def pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def practice(path: Path, tasks) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (path / f"{t}.jsonl").write_text(
            "".join(json.dumps(dict(src=dict(level=lv, seed=i))) + "\n" for lv in (0, 1, 2) for i in range(50))
        )
    return path


# ------------------------------------------------------------------------------------------- the family files
def test_the_seam_knows_the_generic_backend(monkeypatch):
    assert backend.table()["bench"] == "backend_bench"
    for n in ("simple_art", *FAMILIES):
        assert backend.of(n) is BB
    assert backend.of("simple_q") is BB  # every benchmark family is on the one backend
    assert backend.table() == {"bench": "backend_bench"}


@pytest.mark.parametrize("name", FAMILIES)
def test_lml_bench_family_files(name):
    fam = json.loads((RSI / "families" / f"{name}.json").read_text())
    assert fam["name"] == name and fam["backend"] == "bench"
    assert fam["bench"]["name"] == "locomanilab" and fam["bench"]["options"] == {}
    assert fam["bench"]["config"] == (
        {"slots": [3, 4]} if "fridge" in name else {"slots": [5, 6]} if "drawer" in name else {}
    )
    assert not {
        "bed_root",
        "bed_dir",
        "bed_files",
        "hash_files",
        "base",
        "guard",
        "runner",
        "bed_env",
        "official_slots",
    } & set(fam)
    assert fam["bed_fixed"] == {**fam["bed_fixed"], "feed_torque": True} and fam["tasks_file"].endswith("_tasks.jsonl")
    assert fam["rules"] == "scripts/rsi/rules_notes/lml.md" and (RSI.parents[1] / fam["rules"]).is_file()
    assert (RSI.parents[1] / fam["prompts"] / "engineer.md").is_file()
    sb = fam["sandbox"]
    for k in ("keep", "overlay"):
        assert f"scripts/rsi/families/{name}.json" in sb[k] and "scripts/rsi/backend_bench.py" in sb[k]
        assert all(f"scripts/rsi/{m}.py" in sb[k] for m in ("bench_common", "bench_pools", "bench_rows", "bench_lml"))


def test_rules_and_prompts_name_the_new_package():
    text = (RSI / "rules_notes" / "lml.md").read_text()
    for gone in ("skill_simple.py", "seed_pkg_lml"):
        assert gone not in text
    for f in (RSI / "prompts").rglob("*.md"):
        t = f.read_text()
        assert "skill_lml_cam" not in t and "view_lml" not in t and "import backend_lml" not in t, f.name


# --------------------------------------------------------- simple_art: the generic backend equals the old path
def test_simple_art_signature_slots_and_env(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch, "simple_art", config={"slots": [1, 2], "batch_tries": 2})
    assert fam["backend"] == "bench"
    assert BB.signature(fam, "probe") == BB.signature(fam, "gate")  # TODO 52: one render mode for every round
    assert (
        BB.slots_of(fam) == 2 and BB.slots_of(dict(fam, bench=dict(fam["bench"], config={}))) == 3
    )  # as run_jobs had it (3)
    env = BB.bed_env(fam, tmp_path / "pkg", tmp_path)
    assert env["PYTHONPATH"] == "src" and env.get("TMPDIR") == __import__("os").environ.get(
        "TMPDIR"
    )  # SIMPLE: no TMPDIR of ours


def test_simple_art_rows_stay_simple_rows(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch, "simple_art")
    rec = dict(
        episode_id="OpenOvenTeleop/Q/3",
        success=True,
        comply={"ok": True, "why": ""},
        t_end=5.0,
        bench=dict(signature="simple/x"),
    )
    a = B.to_round(rec, "OpenOvenTeleop_s3", "h", {}, "sig") | {"bench": "simple/x"}
    b = BB.to_round(rec, "OpenOvenTeleop_s3", "h", {}, "sig", fam)
    assert a == b and b["bench"] == "simple/x" and "simple" in b and "lml" not in b


# --------------------------------------------------------------------------------------------------- LocoManiLab
def lml(tmp_path, monkeypatch, name="lml_fridge_bench", **kw):
    return bench_fam(tmp_path, monkeypatch, name, **kw)


def test_lml_runner_argv(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch)
    ids = tmp_path / "ids.txt"
    a = BB.bench_argv(
        fam,
        "IThorKitchenOpenFridge_scene13",
        ids,
        {"speed": 0.2},
        {"d": 0.1},
        tmp_path / "u" / "r.jsonl",
        tmp_path / "pkg",
        tmp_path / "u",
        False,
    )
    assert a[:6] == [
        "/venv/main/bin/python",
        "-m",
        "locomani.bench.locomanilab",
        "run",
        "IThorKitchenOpenFridge_scene13",
        "--ids",
    ]
    assert json.loads(pairs(a, "--setting")[0]) == {"speed": 0.2} and json.loads(pairs(a, "--stance")[0]) == {"d": 0.1}
    assert pairs(a, "--bench-set") == []  # the family's options are empty: the adapter's profile resolves
    sets = pairs(a, "--set")
    pdir = BB.practice_dir(fam, False)
    assert sets[:5] == [
        "kinds=Q",
        f"practice_dir={pdir}",
        f"practice_dir_fridge={pdir}",
        f"practice_dir_drawer={pdir}",
        f"run_dir={tmp_path / 'u'}",
    ]
    assert sets[5].startswith("cache_dir=") and "lml_fridge_bench_cache" in sets[5] and sets[6:] == ["slots=3,4"]
    assert not [s for s in sets if s.startswith("lazy_render")]  # LocoManiLab has no render flag
    g = BB.bench_argv(fam, "IThorKitchenOpenFridge_scene13", ids, {}, None, "o", "p", tmp_path, True)
    assert [s for s in pairs(g, "--set") if s.startswith("lazy")] == [] and "--stance" not in g
    assert BB.slots_of(fam) == 2
    q = lml(tmp_path, monkeypatch, "lml_q_bench")
    assert "slots=" not in " ".join(BB.bench_config_args(q)) and BB.slots_of(q) == 2


def test_lml_runner_call_and_env(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(
        C,
        "run_command",
        lambda argv, env, log, stop, cwd, tree=False: seen.update(argv=argv, env=env, cwd=cwd, tree=tree) or 0,
    )
    monkeypatch.setenv("LML_SLOTS", "9")
    monkeypatch.setenv("SIMPLE_LOOP", "1")
    monkeypatch.setenv("EXAM_OPEN", "1")
    (tmp_path / "u").mkdir()
    BB.run_unit(
        "IThorKitchenOpenFridge_scene13",
        ["IThorKitchenOpenFridge_scene13/Q/3"],
        tmp_path / "pkg",
        {"stance_d": 0.1, "speed": 0.2},
        tmp_path / "u" / "r.jsonl",
        family=fam,
    )
    assert seen["cwd"] == Path(fam["bench"]["root"]) and seen["tree"] is True
    assert seen["env"]["PYTHONPATH"] == "src" and seen["env"]["TMPDIR"]
    assert not [k for k in seen["env"] if k.startswith(("LML_", "SIMPLE_"))] and "EXAM_OPEN" not in seen["env"]
    assert Path(pairs(seen["argv"], "--ids")[0]).read_text() == "IThorKitchenOpenFridge_scene13/Q/3\n"
    assert json.loads(pairs(seen["argv"], "--stance")[0]) == {"d": 0.1}


def test_lml_signature(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch, "lml_q_bench")
    root = Path(fam["bench"]["root"])
    s = BB.signature(fam, "probe")
    assert BB.signature(fam, "gate") == s  # one adapter, no render flag: where does not enter
    assert "src/locomani/bench/locomanilab/run.py" in BB.bed_files(
        fam
    ) and "src/locomani/bench/locomanilab/server.py" not in BB.bed_files(fam)
    assert "src/locomani/bench/locomanilab/server.py" in BB.hash_only_files(fam)  # hashed, not shown
    assert not [f for f in BB.bed_files(fam) if f.endswith(("server.py", "truth.py", "config.py"))]
    (root / "src/locomani/bench/locomanilab/server.py").write_text("SERVER = 2\n")
    s2 = BB.signature(fam, "probe")
    assert s2 != s  # a hashed-only file changed
    (root / "src/locomani/bench/locomanilab/run.py").write_text("r = 2\n")
    assert BB.signature(fam, "probe") != s2
    other = dict(fam, bench=dict(fam["bench"], options={"depth_noise": "realsense"}))
    assert BB.signature(other, "probe") != BB.signature(fam, "probe")  # two switch sets never share a signature
    assert BB.signature(dict(fam, bed_fixed={"feed_torque": False}), "probe") != BB.signature(fam, "probe")


def test_lml_benchmark_commit_enters_the_signature(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch, "lml_q_bench")
    s = BB.signature(fam, "probe")
    assert BB.bench_commit(dict(fam, bench=dict(fam["bench"], name="simple"))) == ""  # SIMPLE names no benchmark tree
    monkeypatch.setattr(BB, "bench_commit", lambda f: "abc123")
    assert BB.signature(fam, "probe") != s


def test_check_bench_errors(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not ported"):
        BB.check_bench(dict(fam, bench=dict(fam["bench"], name="nosuchbench")))
    with pytest.raises(ValueError, match="bench.root"):
        BB.check_bench(dict(fam, bench=dict(fam["bench"], root="")))
    for key in ("cache_dir", "practice_dir_drawer", "kinds"):
        with pytest.raises(ValueError, match=key):
            BB.check_bench(dict(fam, bench=dict(fam["bench"], config={key: "x"})))
    with pytest.raises(ValueError, match="lazy_render"):  # the simple adapter owns lazy_render, not this one
        BB.check_bench(dict(fam, bench=dict(fam["bench"], name="simple", config={"lazy_render": True})))
    BB.check_bench(
        dict(fam, bench=dict(fam["bench"], config={"lazy_render": True}))
    )  # LocoManiLab: the adapter will refuse it
    with pytest.raises(ValueError, match="needs a"):
        BB.check_bench(dict(fam, bench=None))


def test_lml_package_parameters_and_stance_keys(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch)
    p = BL.package_parameters(BB.base_dir(fam))
    assert p == {
        "stand": 0.27,
        "speed": 0.3,
        "wait": 1.0,
        "drop": 0.02,
        "lift": 0.34,
        "nav_gap": 0.3,
    }  # skill.py + EXTRA of __init__.py
    sk = BB.load_skills(BB.base_dir(fam), family=fam)
    assert isinstance(sk, LL.SkillPackage)
    assert sk.apply_set(sk, {"lift": 0.4, "stance_d": 0.1, "stance_here": True}) == {
        "lift": 0.4,
        "stance_d": 0.1,
        "stance_here": True,
    }
    with pytest.raises(KeyError, match="stance"):
        sk.apply_set(sk, {"stance_yaw2": 1.0})
    with pytest.raises(KeyError):
        sk.apply_set(sk, {"nope": 1})
    with pytest.raises(KeyError, match="fixed"):
        sk.apply_set(sk, {"feed_torque": False})
    art = bench_fam(tmp_path, monkeypatch, "simple_art")
    assert isinstance(BB.load_skills(BB.base_dir(art), family=art), C.SkillPackage)


def seed_job(tmp_path, with_init=True):
    pk = tmp_path / "pkgjob"
    pk.mkdir()
    (pk / "skill.py").write_text("K_CUT = dict(speed=0.3)\n")
    (pk / "__init__.py").write_text("EXTRA = dict(lift=0.3)\n")
    return pk


def test_lml_run_jobs_rows_to_rounds_and_cards(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch, "lml_q_bench")
    tasks = ["IThorKitchenApproachFridge_scene13", "IThorBathroomLiftSoapBottle_scene401"]
    fam["tasks"] = tasks
    fam["practice_dir"] = str(practice(tmp_path / "practice", tasks))
    pool = tmp_path / "pool"
    BB.draw(fam, "diag", pool, n=4)
    got = []

    def fake(task, ids, package, setting, out, family=None, env=None, full=None, **kw):
        got.append((task, full, setting, env, family["name"]))
        rec = dict(
            success=False,
            genuine=False,
            loss="grasp",
            why="the fingers closed on nothing",
            t_end=9.0,
            fell=None,
            instruction="pick up the soap bottle",
            comply={"ok": False, "why": ""},
            phases={"walk": 2.0, "close": 1.0},
            true_centre=[1.0, 2.0, 0.5],
            lml_config={"specs_file": "/secret"},
            bench=dict(signature="locomanilab/x"),
        )
        Path(out).write_text("".join(json.dumps(dict(rec, episode_id=e)) + "\n" for e in ids))
        return 0

    monkeypatch.setattr(BB, "run_unit", fake)
    monkeypatch.setattr(BB, "run_guard", lambda *a: None)
    pk = seed_job(tmp_path)
    job = type("J", (), dict(skills=pk, out=tmp_path / "o", overrides={"speed": 0.1}, labels=None))()
    assert BB.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0", where="gate") == {0: []}
    assert got and all(g[1] is True and g[2]["feed_torque"] is True and g[4] == "lml_q_bench" for g in got)
    assert not [k for g in got for k in g[3] if k.startswith(("LML_", "SIMPLE_"))]
    r = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert (
        r["bench"] == "locomanilab/x" and r["motor"] == BB.signature(fam, "gate") and "lml" in r and "simple" not in r
    )
    assert "true_centre" not in json.dumps(r) and "/secret" not in json.dumps(
        r
    )  # the allow-list drops what is not on it
    lines = BB.card_text(dict(r, ok=False, failure_class="grasp"))
    assert any(x.startswith("bench: locomanilab/x") for x in lines) and "true_centre" not in "\n".join(lines)
    text = BB.cards([dict(r, ok=False, failure_class="grasp")], tmp_path / "c").read_text()
    assert "bench: locomanilab/x" in text.splitlines()[:4]
    n = len(got)
    assert (
        BB.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0", where="gate") == {0: []} and len(got) == n
    )  # nothing runs twice


def test_lml_run_jobs_guard_and_refused_rows(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch, "lml_q_bench")
    tasks = ["IThorKitchenApproachFridge_scene13"]
    fam["tasks"] = tasks
    pdir = practice(tmp_path / "practice", tasks)
    fam["practice_dir"] = str(pdir)
    pool = tmp_path / "pool"
    BB.draw(fam, "diag", pool, n=3)
    calls = []
    monkeypatch.setattr(BB, "run_unit", lambda *a, **k: calls.append(a) or 0)
    pk = seed_job(tmp_path)
    monkeypatch.setattr(BB, "run_guard", lambda *a: "R4 pick.py: forbidden name 'qpos'")
    job = type("J", (), dict(skills=pk, out=tmp_path / "o", overrides={}, labels=None))()
    assert BB.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0") == {0: []} and not calls
    r = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert r["ok"] is False and r["why"].startswith("guard: R4") and "lml" in r and "simple" not in r
    monkeypatch.setattr(BB, "run_guard", lambda *a: None)  # a blanked practice row is refused, not run
    lines = (pdir / f"{tasks[0]}.jsonl").read_text().splitlines()
    first = sorted(pool.glob("*_s*.json"))[0].stem.rsplit("_s", 1)[1]
    lines[int(first)] = "{}"
    (pdir / f"{tasks[0]}.jsonl").write_text("\n".join(lines) + "\n")
    out2 = tmp_path / "o2"
    job2 = type("J", (), dict(skills=pk, out=out2, overrides={}, labels=None))()
    BB.run_jobs([job2], fam, pool, 1, tmp_path / "log.txt", "0")
    refused = json.loads((out2 / f"{tasks[0]}_s{first}.json").read_text())
    assert refused["failure_class"] == "refused" and "lml" in refused


# ------------------------------------------------------------------------------------------- guard and sandbox
@needs_lml
def test_lml_guard_and_seed_are_the_real_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path))
    fam = BB.load("lml_q_bench")
    assert fam["bench"]["root"] == str(BENCH_REAL)
    base = BB.base_dir(fam)
    assert base == (BENCH_REAL / "src/locomani/bench/locomanilab/seed").resolve()
    p = BL.package_parameters(base)
    assert "lift" in p and "nav_gap" in p and "stand" in p
    assert BB.run_guard(fam, base, base) is None
    cand = tmp_path / "cand"
    shutil.copytree(base, cand, ignore=shutil.ignore_patterns("__pycache__"))
    assert BB.run_guard(fam, base, cand) is None
    (cand / "pick.py").write_text((cand / "pick.py").read_text() + "\nx = open\n")
    assert "R4" in BB.run_guard(fam, base, cand)
    (cand / "pick.py").write_text((base / "pick.py").read_text())
    (cand / "flows.py").write_text(
        (cand / "flows.py").read_text() + "\nfrom locomani.bench.locomanilab import client\n"
    )
    assert "R8" in BB.run_guard(fam, base, cand)
    shutil.copy(base / "flows.py", cand / "flows.py")
    (cand / "extra.py").write_text("")
    assert "R1" in BB.run_guard(fam, base, cand)
    assert BB.signature(fam, "probe") == BB.signature(
        BB.load("lml_fridge_bench"), "probe"
    )  # the bed is one, the pools differ


@pytest.mark.parametrize("name", ("lml_q_bench", "lml_drawer_bench"))
def test_lml_sandbox_copies_the_readable_adapter_and_blanks_hidden_rows(tmp_path, monkeypatch, name):
    fam = lml(tmp_path, monkeypatch, name)
    fam["practice_dir"] = str(practice(tmp_path / "practice", fam["tasks"]))
    fam["sandbox"] = dict(fam["sandbox"], task_files=[])
    monkeypatch.setattr(C, "HIDDEN", "hid")
    dest = tmp_path / "sb"
    (dest / "scripts/rsi/families").mkdir(parents=True)
    (dest / "scripts/rsi/families" / f"{name}.json").write_text(json.dumps(fam))
    BB.sandbox(dest, fam)
    nest = dest / "bed_readonly_copy"
    L = "src/locomani/bench/locomanilab/"
    assert (nest / BB.base_rel(fam) / "skill.py").exists() and (nest / L / "run.py").stat().st_mode & 0o222 == 0
    assert (
        not (nest / L / "server.py").exists()
        and not (nest / L / "truth.py").exists()
        and not (nest / L / "config.py").exists()
    )
    assert BB.audit(dest) == []
    first = (nest / "practice" / f"{fam['tasks'][0]}.jsonl").read_text().splitlines()
    assert first[0] != "{}" and sum(x == "{}" for x in first) == 120
    (nest / L / "run.py").chmod(0o644)  # a leak in a readable file is found
    (nest / L / "client.py").chmod(0o644)
    (nest / L / "client.py").write_text("# see outputs/escalation_20261116/a_l/specs.json\n")
    shutil.copy(nest / L / "client.py", nest / "notes.txt")
    assert any("notes.txt" in x for x in BB.audit(dest))
    (nest / "notes.txt").unlink()


def test_simple_sandbox_through_the_generic_backend(tmp_path, monkeypatch):
    fam = bench_fam(tmp_path, monkeypatch, "simple_art")
    fam["practice_dir"] = str(practice(tmp_path / "practice", ART_TASKS))
    fam["sandbox"] = dict(fam["sandbox"], task_files=[])
    monkeypatch.setattr(C, "HIDDEN", "hid")
    dest = tmp_path / "sb"
    (dest / "scripts/rsi/families").mkdir(parents=True)
    (dest / "scripts/rsi/families/simple_art.json").write_text(json.dumps(fam))
    BB.sandbox(dest, fam)
    nest = dest / "bed_readonly_copy"
    assert (nest / BB.base_rel(fam) / "skill.py").exists() and (
        nest / "src/locomani/bench/simple/run.py"
    ).stat().st_mode & 0o222 == 0
    assert BB.audit(dest) == []


def test_layout_and_caps(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch, "lml_drawer_bench")
    lay = BB.layout(fam)
    assert lay["stub_file"] == "pick.py" and lay["leak"] == BL.LEAK and lay["class_suggest"] == BL.CLASS_SUGGEST
    assert lay["caps"]["sweep_rounds"] == 180 and lay["caps"]["p_max"] == 0.05
    assert Path(lay["skills"]).is_dir() and Path(lay["rules"]).name == "lml.md"
    art = bench_fam(tmp_path, monkeypatch, "simple_art")
    assert BB.layout(art)["leak"] == B.LEAK and BB.layout(art)["class_suggest"] == B.CLASS_SUGGEST


def test_cache_is_pruned_to_the_newest(tmp_path, monkeypatch):
    fam = lml(tmp_path, monkeypatch, "lml_q_bench")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    root = BB.scratch_cache(fam)
    assert (
        root == tmp_path / "tmp" / "locomanilab_lml_q_bench_cache"
        and BB.scratch_cache(bench_fam(tmp_path, monkeypatch, "simple_art")) is None
    )
    for i in range(9):
        d = root / f"k{i}"
        d.mkdir(parents=True)
        __import__("os").utime(d, (1000 + i, 1000 + i))
    BB.prune_cache(root)
    assert sorted(p.name for p in root.iterdir()) == [f"k{i}" for i in range(3, 9)]


def test_memo_sets_of_the_generic_backend(tmp_path, monkeypatch):
    """TODO 53: simple_art ships with bench.memo on; the adapter call outside the gate carries the store, the signature and the
    cap (the generic backend's signature), the gate and the sealed call carry none."""
    fam = bench_fam(tmp_path, monkeypatch, "simple_art", memo=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    shipped = json.loads((RSI / "families" / "simple_art.json").read_text())
    assert shipped["bench"]["memo"] is True
    probe = pairs(BB.bench_argv(fam, "OpenOvenTeleop", tmp_path / "ids", {}, None, "o", "p", tmp_path, False), "--set")
    gate = pairs(BB.bench_argv(fam, "OpenOvenTeleop", tmp_path / "ids", {}, None, "o", "p", tmp_path, True), "--set")
    store = tmp_path / "scratch" / "simple_simple_art_memo"
    assert "memo=true" in probe and f"memo_dir={store}" in probe and f"memo_sig={BB.signature(fam, 'probe')}" in probe
    assert "memo_gb=20" in probe
    assert not [s for s in gate if s.startswith("memo")]
    assert BB.signature(fam, "probe") != BB.signature(fam, "gate")  # the memo's rounds are their own pool
    for k in ("memo_dir", "memo_sig", "memo_gb"):  # the backend's own keys
        with pytest.raises(ValueError, match=k):
            BB.check_bench(dict(fam, bench=dict(fam["bench"], config={k: "x"})))

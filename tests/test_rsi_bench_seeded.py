"""The generic bench backend on the seed-pool adapters, BiGym 2.0 and HumanoidArena (TODO 49.7a). No simulator: the adapter is a
small fake bench root and the runner a fake; the pool tables are checked against the real adapters' config in a subprocess."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RSI = REPO / "scripts" / "rsi"
sys.path.insert(0, str(RSI))
import backend  # noqa: E402
import backend_bench as BB  # noqa: E402
import bench_common as C  # noqa: E402
import bench_pools as P  # noqa: E402
import bench_seeded as BS  # noqa: E402

FAMILIES = {"bigym2_reach_drawer": "bigym2", "humanoidarena_pp_box": "humanoidarena"}
real = pytest.mark.skipif(not (REPO / "src/locomani/bench/bigym2/guard.py").exists(), reason="no bench adapters")


def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    files = {"src/locomani/bench/options.py": "o = 1\n", "src/locomani/bench/spec.py": "s = 1\n",
             "src/locomani/config_base.py": "c = 1\n", "src/locomani/perception/sensor.py": "f = 1\n"}  # fmt: skip
    for name in ("bigym2", "humanoidarena"):
        d = f"src/locomani/bench/{name}/"
        for f in ("run", "client", "judge", "record", "guard", "server", "truth", "config", "episodes", "session"):
            files[d + f + ".py"] = f"{f} = 1\n"
        files[d + "seed/skill.py"] = "K = dict(walk_speed=0.2, grip=1.0)\n"
        files[d + "seed/__init__.py"] = ""
        files[f"configs/bench/{name}.yaml"] = "x: 1\n"
    files["src/locomani/bench/humanoidarena/gait_cycle.json"] = "{}"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def fam_of(tmp_path, monkeypatch, name="bigym2_reach_drawer", **kw) -> dict:
    for k in list(__import__("os").environ):
        if k.startswith("GIT_"):
            monkeypatch.delenv(k)
    fam = json.loads((RSI / "families" / f"{name}.json").read_text())
    fam["bench"] = dict(fam["bench"], root=str(fake_root(tmp_path)), **kw)
    fam["practice_dir"] = "outputs/_no_practice_files"
    rec = lambda f: dict(
        signature=f["bench"]["name"]
        + "/"
        + "-".join(f"{k}:{v}" for k, v in sorted(f["bench"].get("options", {}).items()))
    )  # noqa: E731
    monkeypatch.setattr(BB, "bench_record", rec)
    monkeypatch.setattr(BB, "bench_commit", lambda fam: "abc")
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path / "pk"))
    return fam


def pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


@pytest.mark.parametrize("name", FAMILIES)
def test_the_family_files(name):
    f = json.loads((RSI / "families" / f"{name}.json").read_text())
    assert f["backend"] == "bench" and f["bench"]["name"] == FAMILIES[name]
    assert backend.of(f).__name__ == "backend_bench" and BB.seeded(f)
    assert (REPO / f["rules"]).exists() and (REPO / f["prompts"]).is_dir() and (REPO / f["tasks_file"]).exists()
    assert set(f["pools"]) == {"diag", "accept", "sealed"}
    assert f["pools"]["accept"]["seed"] == f["pools"]["sealed"]["seed"]
    assert f["rules"].endswith(f"rules_notes/{FAMILIES[name]}.md") and f["prompts"] == "scripts/rsi/prompts"
    assert (REPO / f["rules"]).read_text().startswith("## ")
    BB.check_bench(f)


def test_the_tasks_are_the_seeds():
    assert json.loads((RSI / "families/bigym2_reach_drawer.json").read_text())["tasks"] == ["reach_target_single", "drawer_top_open"]  # fmt: skip
    assert json.loads((RSI / "families/humanoidarena_pp_box.json").read_text())["tasks"] == ["pp_box"]


def test_existing_families_are_unchanged():
    for n in ("lml_q_bench", "lml_fridge_bench", "lml_drawer_bench", "simple_art"):
        f = json.loads((RSI / "families" / f"{n}.json").read_text())
        assert not BB.seeded(f)
    f = json.loads((RSI / "families/lml_q_bench.json").read_text())
    assert BB.adapter(f)["stub_file"] if "stub_file" in BB.adapter(f) else True
    assert BB.ADAPTERS["locomanilab"]["kinds"] == "Q" and BB.ADAPTERS["simple"]["practice_keys"] == ("practice_dir",)
    assert "stub_file" not in BB.ADAPTERS["simple"] and "pool_mode" not in BB.ADAPTERS["locomanilab"]


@pytest.mark.parametrize("name", FAMILIES)
def test_runner_argv(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name, config={"gpu": 1})
    task = fam["tasks"][0]
    ids = tmp_path / "ids.txt"
    probe = BB.bench_argv(fam, task, ids, {"walk_speed": 0.3}, None, tmp_path / "u/r.jsonl", tmp_path / "pkg", tmp_path / "u", False)  # fmt: skip
    mod = f"locomani.bench.{FAMILIES[name]}"
    assert probe[:5] == ["/venv/main/bin/python", "-m", mod, "run", task]
    assert json.loads(pairs(probe, "--setting")[0]) == {"walk_speed": 0.3} and "--stance" not in probe
    sets = pairs(probe, "--set")
    assert sets[0].startswith("run_dir=") and "pools=P" in sets and not [s for s in sets if s.startswith(("kinds", "practice", "lazy", "cache"))]  # fmt: skip
    assert (
        any(s.startswith("lock_file=") and s.endswith(f"{FAMILIES[name]}.lock") for s in sets) and sets[-1] == "gpu=1"
    )
    gate = BB.bench_argv(fam, task, ids, {}, None, "o", "p", tmp_path, True)
    assert "pools=PAS" in pairs(gate, "--set") and "pools=P" not in pairs(gate, "--set")
    assert pairs(probe, "--bench-set") == [] and BB.slots_of(fam) == 1
    fam2 = fam_of(tmp_path, monkeypatch, name, options={"depth_noise": "none"})
    assert pairs(BB.bench_argv(fam2, task, ids, {}, None, "o", "p", tmp_path, False), "--bench-set") == ["depth_noise=none"]  # fmt: skip


@pytest.mark.parametrize("name", FAMILIES)
def test_runner_env_is_clean(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name)
    for k in ("BIGYM_X", "ISAAC_X", "SIMPLE_X", "LML_X", "EXAM_OPEN", "KEEP_ME"):
        monkeypatch.setenv(k, "1")
    env = BB.bed_env(fam, tmp_path, tmp_path)
    gone = BB.adapter(fam)["strip_env"]
    assert "KEEP_ME" in env and "EXAM_OPEN" not in env and not [k for k in env if k.startswith(gone)]
    assert "SIMPLE_X" not in env and "LML_X" not in env
    assert ("BIGYM_X" in env) == (name != "bigym2_reach_drawer") and ("ISAAC_X" in env) == (
        name != "humanoidarena_pp_box"
    )
    assert env["PYTHONPATH"] == "src" and env["TMPDIR"]


def test_config_may_not_set_the_backends_keys(tmp_path, monkeypatch):
    for key in ("pools", "run_dir", "lock_file"):
        fam = fam_of(tmp_path, monkeypatch, config={key: "x"})
        with pytest.raises(ValueError, match="backend sets"):
            BB.check_bench(fam)


def test_signature_follows_switches_and_the_adapters_own_files(tmp_path, monkeypatch):
    fam = fam_of(tmp_path, monkeypatch)
    root = Path(fam["bench"]["root"])
    s = BB.signature(fam, "probe")
    assert BB.signature(fam, "gate") == s
    assert "src/locomani/bench/bigym2/run.py" in BB.bed_files(fam) and "src/locomani/bench/bigym2/server.py" in BB.hash_only_files(fam)  # fmt: skip
    assert not [f for f in BB.bed_files(fam) if f.endswith(("server.py", "truth.py", "config.py", "episodes.py"))]
    assert not [f for f in BB.bed_files(fam) + BB.hash_only_files(fam) if "scalebfm" in f or "perception/head" in f]
    other = dict(fam, bench=dict(fam["bench"], options={"depth_noise": "realsense"}))
    assert BB.signature(other, "probe") != s  # a switch
    (root / "src/locomani/bench/humanoidarena/server.py").write_text("changed\n")
    (root / "src/locomani/bench/humanoidarena/gait_cycle.json").write_text("{1}")
    assert BB.signature(fam, "probe") == s  # a file of the other adapter
    (root / "src/locomani/bench/bigym2/server.py").write_text("changed\n")
    s2 = BB.signature(fam, "probe")
    assert s2 != s  # a hashed-only file of this adapter
    (root / "src/locomani/bench/bigym2/run.py").write_text("changed\n")
    assert BB.signature(fam, "probe") != s2
    (root / "src/locomani/bench/options.py").write_text("o = 2\n")
    ha = fam_of(tmp_path / "x", monkeypatch, "humanoidarena_pp_box")
    (Path(ha["bench"]["root"]) / "src/locomani/bench/bigym2/run.py").write_text("changed\n")
    s_ha = BB.signature(ha, "probe")
    (Path(ha["bench"]["root"]) / "src/locomani/bench/humanoidarena/gait_cycle.json").write_text("{2}")
    assert BB.signature(ha, "probe") != s_ha  # the gait is the bed's


def test_commit_enters_the_signature(tmp_path, monkeypatch):
    fam = fam_of(tmp_path, monkeypatch)
    s = BB.signature(fam, "probe")
    monkeypatch.setattr(BB, "bench_commit", lambda fam: "def")
    assert BB.signature(fam, "probe") != s
    assert BB.ADAPTERS["bigym2"]["benchmark"] == "third_party/BiGym2"
    assert BB.ADAPTERS["humanoidarena"]["benchmark"] == "third_party/HumanoidArena/repo"


@pytest.mark.parametrize("name", FAMILIES)
def test_package_parameters_skills_hash_and_guard_path(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name)
    base = BB.base_dir(fam)
    assert base == (Path(fam["bench"]["root"]) / f"src/locomani/bench/{FAMILIES[name]}/seed").resolve()
    assert BS.package_parameters(base) == {"walk_speed": 0.2, "grip": 1.0}
    pkg = BB.load_skills(base, family=fam)
    assert sorted(pkg.parameters()) == ["grip", "walk_speed"]
    pkg.apply_set(pkg, {"grip": 0.5})
    for bad in ({"nope": 1}, {"stance_d": 0.1}):
        with pytest.raises(KeyError):
            pkg.apply_set(pkg, bad)
    h = BB.skills_hash(base)
    shutil.copytree(base, tmp_path / "c")
    assert BB.skills_hash(tmp_path / "c") == h
    (tmp_path / "c/skill.py").write_text("K = dict(walk_speed=0.3)\n")
    assert BB.skills_hash(tmp_path / "c") != h
    assert BB.layout(fam)["stub_file"] == "skill.py" and BB.adapter(fam)["guard"] == f"src/locomani/bench/{FAMILIES[name]}/guard.py"  # fmt: skip
    assert BB.layout(fam)["class_suggest"].keys() >= {"fall", "comply"} and BB.layout(fam)["leak"]
    assert BB.run_guard(fam, base, base) is None
    assert "no JSON" in BB.run_guard(fam, base, tmp_path / "c")  # the fake guard is no guard: the path is the adapter's


@real
@pytest.mark.parametrize("name", FAMILIES)
def test_the_real_guard_and_seed(tmp_path, monkeypatch, name):
    monkeypatch.setenv("RSI_PACKAGE_ROOT", str(tmp_path))
    fam = BB.load(name)
    base = BB.base_dir(fam)
    p = BS.package_parameters(base)
    assert p and "walk_speed" in p or "settle_steps" in p
    cand = tmp_path / "cand"
    shutil.copytree(base, cand, ignore=shutil.ignore_patterns("__pycache__"))
    assert BB.run_guard(fam, base, cand) is None
    (cand / "skill.py").write_text((cand / "skill.py").read_text() + "\nx = open\n")
    assert "R4" in BB.run_guard(fam, base, cand)


@pytest.mark.parametrize("name", FAMILIES)
def test_pool_mapping(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name)
    fam["pools"] = json.loads(json.dumps(fam["pools"]))
    seen = {}
    for pool, letter in (("diag", "P"), ("accept", "A"), ("sealed", "S")):
        out = tmp_path / pool
        r = BB.draw(fam, pool, out)
        per = fam["pools"][pool]["per_task"]
        assert r["written"] == per * len(fam["tasks"])
        files = sorted(out.glob("*_s*.json"))
        ids = [json.loads(f.read_text())["episode_id"] for f in files]
        assert all(i.split("/")[1] == letter for i in ids) and len({*ids}) == len(ids)
        assert not [i for i in ids if BB.id_refusal(fam, i)]
        ns = [int(i.split("/")[2]) for i in ids]
        if pool == "diag":
            assert min(ns) == 0 and max(ns) < per
        else:
            assert min(ns) >= P.OFFSET_FLOOR
        seen[pool] = set(f.stem for f in files)
        # the label's task is the file's task
        assert all(json.loads(f.read_text())["task"] == f.stem.rsplit("_s", 1)[0] for f in files)
    # the draw is a function of the family's hidden seed: another seed, another block; the same seed, the same block
    again = BB.draw(dict(fam), "accept", tmp_path / "a2")
    assert {p.stem for p in (tmp_path / "a2").glob("*_s*.json")} == seen["accept"] and again["written"]
    for p in ("accept", "sealed"):
        fam["pools"][p]["seed"] = 7
    BB.draw(fam, "accept", tmp_path / "a3")
    assert {p.stem for p in (tmp_path / "a3").glob("*_s*.json")} != seen["accept"]
    # a blank seed is refused, as for the other backends
    fam["pools"]["accept"]["seed"] = fam["pools"]["sealed"]["seed"] = None
    with pytest.raises(SystemExit):
        BB.draw(fam, "accept", tmp_path / "a4")


def test_diag_n_is_the_loops_pool_size():
    for n in FAMILIES:
        f = json.loads((RSI / "families" / f"{n}.json").read_text())
        for p in f["pools"].values():
            assert p["n"] == p["per_task"] * len(f["tasks"])


@pytest.mark.parametrize("name", FAMILIES)
def test_a_family_cannot_name_the_evaluation_episodes(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name)
    t = fam["tasks"][0]
    ad = BB.adapter(fam)
    for bad in (f"{t}/E/0", f"{t}/e/0", f"{t}/PA/0", f"{t}/P/-1", f"{t}/P/x", f"{t}/P/{ad['pool_sizes']['P']}",
                f"{t}/A/{ad['pool_sizes']['A']}", f"{t}/S/{ad['pool_sizes']['S']}", "other/P/0", "P/0"):  # fmt: skip
        assert BB.id_refusal(fam, bad), bad
    assert BB.id_refusal(fam, f"{t}/P/0") is None and BB.id_refusal(fam, f"{t}/P/0", "A")
    assert BB.id_refusal(fam, f"{t}/A/5", "P") and BB.id_refusal(fam, f"{t}/S/5", "PA")
    if name.startswith("bigym2"):  # BiGym's block: no letter, and no index of any pool reaches it
        for letter, (first, size) in {"P": (1000, 9000), "A": (100000, 1000), "S": (700000, 10000)}.items():
            assert ad["pool_sizes"][letter] == size and not (first < 630000 and first + size > 620000)
        assert BB.id_refusal(fam, f"{t}/P/{620000 - 1000}")  # the index that would be the first evaluation seed
    for field, value in (("letter", "E"), ("letter", "A")):  # the diagnostic pool is P and nothing else
        f2 = json.loads(json.dumps(fam))
        f2["pools"]["diag"][field] = value
        with pytest.raises(ValueError, match="not"):
            BB.check_bench(f2)
    f3 = json.loads(json.dumps(fam))
    f3["pools"]["accept"]["ids"] = [f"{t}/E/3"]
    with pytest.raises(ValueError, match="evaluation"):
        BB.check_bench(f3)
    f4 = json.loads(json.dumps(fam))
    f4["pools"]["evaluation"] = {"per_task": 5}
    with pytest.raises(ValueError, match="pools"):
        BB.check_bench(f4)
    f5 = json.loads(json.dumps(fam))
    f5["pools"]["sealed"]["per_task"] = 10**6
    with pytest.raises(ValueError, match="per_task"):
        BB.check_bench(f5)
    f6 = json.loads(json.dumps(fam))
    f6["tasks"] = ["not_a_task"]
    with pytest.raises(ValueError, match="task"):
        BB.check_bench(f6)


@real
def test_pool_tables_and_refusal_agree_with_the_real_adapters():
    code = """
import json
from locomani.bench.bigym2 import config as bc, episodes as be
from locomani.bench.humanoidarena import config as hc, episodes as he
out = dict(
    bigym2=dict(sizes={k: v[1] for k, v in bc.POOLS.items()}, tasks=list(be.BENCHMARK_TASKS)),
    humanoidarena=dict(sizes={k: v[1] * hc.PER_GROUP for k, v in hc.POOLS.items()}, tasks=list(he.TASKS)),
)
# no id of any pool, and no id this backend would accept, is an evaluation seed
bcfg = bc.BiGym2Config(pools="PAS")
out["bigym_hits"] = sum(be.in_eval_block(be.seed_of(f"x/{k}/{n}")) for k, (lo, sz) in bc.POOLS.items() for n in range(sz))
hcfg = hc.HumanoidArenaConfig(pools="PAS")
hits = 0
for t in he.TASKS:
    ev = he.eval_seeds(t)
    assert len(ev) == 60
    for k, (lo, groups) in hc.POOLS.items():
        for n in range(groups * hc.PER_GROUP):
            hits += he.seed_of(f"{t}/{k}/{n}") in ev
out["ha_hits"] = hits
refused = []
for spec in ("drawer_top_open/E/0", "drawer_top_open/P/9000", "drawer_top_open/P/619000"):
    try:
        be.check_id(bcfg, spec)
    except Exception:
        refused.append(spec)
for spec in ("pp_box/E/0", "pp_box/P/20000"):
    try:
        he.check_id(hcfg, spec)
    except Exception:
        refused.append(spec)
out["refused"] = refused
print(json.dumps(out))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env={**__import__("os").environ, "PYTHONPATH": "src"})  # fmt: skip
    assert r.returncode == 0, r.stderr[-600:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    for n in ("bigym2", "humanoidarena"):
        assert d[n]["sizes"] == BB.ADAPTERS[n]["pool_sizes"], n
        assert set(BB.ADAPTERS[n]["tasks"]) == set(d[n]["tasks"]), n
    assert d["bigym_hits"] == 0 and d["ha_hits"] == 0 and len(d["refused"]) == 5


def _job(tmp_path, pk):
    return type("J", (), dict(skills=pk, out=tmp_path / "o", overrides={"grip": 0.5}, labels=None))()


def _pkg(tmp_path):
    pk = tmp_path / "pkgjob"
    pk.mkdir(exist_ok=True)
    (pk / "skill.py").write_text("K = dict(walk_speed=0.2, grip=1.0)\n")
    (pk / "__init__.py").write_text("")
    return pk


@pytest.mark.parametrize("name", FAMILIES)
def test_run_jobs_rows_to_rounds_and_cards(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name)
    pool = tmp_path / "pool"
    BB.draw(fam, "diag", pool, n=4)
    got = []

    def fake(task, ids, package, setting, out, family=None, env=None, full=None, **kw):
        got.append((task, full, setting, env))
        rows = []
        for e in ids:
            rows.append(dict(id=e, task=task, pool="P", seed=1234, success=False, bench_success=True, why="no_hand", fell=False,
                             steps=650, hand_steps=2, body_steps=0, body_parts=["wrist_yaw_link"], marks={"start": 30, "stand": 215},
                             flow={"notes": {"stand": {"t": 1.0}}}, wall_s=3.0, error=None,
                             bench=dict(signature="x/y"), bigym2_config={"bigym_root": "/secret"}, view="truth", package="abc"))  # fmt: skip
        Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return 0

    monkeypatch.setattr(BB, "run_unit", fake)
    monkeypatch.setattr(BB, "run_guard", lambda *a: None)
    job = _job(tmp_path, _pkg(tmp_path))
    assert BB.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0", where="probe") == {0: []}
    assert got and all(g[1] is False for g in got)
    rounds = [json.loads(f.read_text()) for f in sorted((tmp_path / "o").glob("*.json"))]
    assert len(rounds) == 4 and all(r["motor"] == BB.signature(fam, "probe") and "seeded" in r for r in rounds)
    r = rounds[0]
    assert r["ok"] is False and r["failure_class"] == "comply" and r["bench"] == "x/y" and r["task"] in fam["tasks"]
    assert [s["step"] for s in r["steps"]] == ["start", "stand", "after_stand"] and r["steps"][-1]["ok"] is False
    assert "/secret" not in json.dumps(r) and "1234" not in json.dumps(r["seeded"])  # no config, no seed
    text = BB.cards(rounds, tmp_path / "c").read_text()
    assert (
        "bench: x/y" in text.splitlines()[:4]
        and "comply" in text
        and "wrist_yaw_link" in text
        and "/secret" not in text
    )
    n = len(got)
    BB.run_jobs([job], fam, pool, 1, tmp_path / "log.txt", "0", where="probe")
    assert len(got) == n


@pytest.mark.parametrize("name", FAMILIES)
def test_probe_refuses_hidden_pools_and_gate_runs_them(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name)
    for pool in ("accept", "sealed"):
        BB.draw(fam, pool, tmp_path / pool)
    ran = []
    monkeypatch.setattr(BB, "run_guard", lambda *a: None)
    monkeypatch.setattr(BB, "run_unit", lambda task, ids, *a, **k: (ran.append((ids, k.get("full"))), 0)[1])
    job = _job(tmp_path, _pkg(tmp_path))
    out = BB.run_jobs([job], fam, tmp_path / "accept", 1, tmp_path / "log.txt", "0", where="probe")
    assert out == {0: []} and not ran  # refused: no id of a hidden pool runs in a diagnostic call
    r = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert r["failure_class"] == "refused" and r["ok"] is False and r["why"] == "not a diagnostic episode"
    BB.run_jobs([type("J", (), dict(skills=job.skills, out=tmp_path / "o2", overrides={}, labels=None))()], fam, tmp_path / "accept", 1, tmp_path / "log.txt", "0", where="gate")  # fmt: skip
    assert (
        ran
        and ran[0][1] is True
        and all(i.split("/")[1] == "A" for ids, _ in ran for i in ids)
        and {i.split("/")[0] for ids, _ in ran for i in ids} == set(fam["tasks"])
    )


@pytest.mark.parametrize("name", FAMILIES)
def test_sandbox_and_leak_audit(tmp_path, monkeypatch, name):
    fam = fam_of(tmp_path, monkeypatch, name)
    monkeypatch.setattr(C, "HIDDEN", "hid")
    fam["sandbox"] = dict(fam["sandbox"], task_files=[])
    dest = tmp_path / "sb"
    (dest / "scripts/rsi/families").mkdir(parents=True)
    (dest / "scripts/rsi/families" / f"{fam['name']}.json").write_text(json.dumps(fam))
    (dest / "third_party/BiGym2").mkdir(parents=True)
    (dest / "third_party/HumanoidArena").mkdir(parents=True)
    BB.sandbox(dest, fam)
    nest = dest / "bed_readonly_copy"
    d = f"src/locomani/bench/{FAMILIES[name]}/"
    assert (nest / BB.base_rel(fam) / "skill.py").exists() and (nest / d / "client.py").stat().st_mode & 0o222 == 0
    assert not (nest / d / "server.py").exists() and not (nest / d / "episodes.py").exists() and not (nest / d / "config.py").exists()  # fmt: skip
    assert not (dest / "third_party/BiGym2").exists() and not (dest / "third_party/HumanoidArena").exists()
    assert BB.audit(dest) == []
    # a fake practice file (a practice episode list with hidden ids and the evaluation block) in the readable copy is found
    t = fam["tasks"][0]
    (nest / "practice").mkdir(exist_ok=True)
    for text in (
        f"{t}/A/412\n",
        f"{t}/S/5\n",
        f"{t}/E/0\n",
        "seed 620017\n",
        "see third_party/BiGym2/bigym/loco/eval\n",
    ):
        (nest / "practice" / "fake.jsonl").write_text(text)
        assert any("fake.jsonl" in x for x in BB.audit(dest)), text
    (nest / "practice" / "fake.jsonl").write_text(f"{t}/P/3\n")  # a practice id is fine
    assert BB.audit(dest) == []
    (nest / "practice" / "fake.jsonl").unlink()


def test_cards_and_flavor_pick_the_seeded_rounds():
    r = BS.to_round(
        dict(id="drawer_top_open/P/0", success=True, why="", steps=500, marks={"a": 10}), "x_s0", "h", {}, "sig"
    )
    assert r["ok"] and r["failure_class"] == "" and BB.rows_module_of([r]) == "bench_seeded"
    for rec, cls in ((dict(success=False, why="fell", fell=True), "fall"), (dict(success=False, why="error", error="x"), "error"),
                     (dict(success=False, why="not_success", termination="timeout"), "timeout"),
                     (dict(success=False, why="not_success"), "short"), (dict(success=False, why="body_push"), "comply"),
                     (dict(success=False, why="not_moved"), "comply")):  # fmt: skip
        assert BS.failure_class(dict(rec, id="t/P/0")) == cls


def test_the_hidden_tree_is_not_a_forbidden_path():
    assert not [p for p in BS.FORBIDDEN_PATHS if "scene_rsi_20261016_hidden" in p]  # sandbox.py writes seeds.json there

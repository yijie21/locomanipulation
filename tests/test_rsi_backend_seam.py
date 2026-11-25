"""The loop's backend seam (scripts/rsi/backend.py): a family file without a "backend" key takes today's code, one with
"backend": "fake" (tests/fake_backend.py, registered through RSI_BACKEND_PATH) is loaded, drawn, run, carded and laid
out by the backend. No simulation. The loop, gate and family modules need the scene stack (skfmm), so those tests skip
without it."""

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"
sys.path[:0] = [str(HERE), str(REPO / "src"), str(Path(__file__).resolve().parent)]

import backend as BK  # noqa: E402
import fake_backend  # noqa: E402

CLASH = ("gate", "batch", "family", "scene_round", "digest", "loop", "calibrate", "cards")


def _rsi(name: str):
    """scripts/rsi/<name>.py under a name of its own: scripts/gate.py (the collection gate) and the warehouse loop's
    loop.py, gate.py and batch.py are `gate`, `loop` and `batch` in other test files of the same session."""
    import importlib.util

    saved = {k: sys.modules.pop(k, None) for k in CLASH}
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location(
            f"seam_{name}", HERE / f"{name}.py" if (HERE / f"{name}.py").exists() else HERE / "scene" / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"seam_{name}"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


@pytest.fixture
def fam_dir(tmp_path, monkeypatch):
    """A families/ directory with a scene-less family "fake" (backend key) and one without a key."""
    d = tmp_path / "families"
    d.mkdir()
    (d / "fake.json").write_text(
        json.dumps(dict(name="fake", backend="fake", tasks=["XTask"], pools={"diag": {"n": 2}}))
    )
    (d / "plain.json").write_text(json.dumps(dict(name="plain", scene="x", plan="docs/p.md")))
    monkeypatch.setattr(BK, "HERE", tmp_path)
    monkeypatch.setenv("RSI_BACKEND_PATH", "fake=fake_backend")
    fake_backend.CALLS.clear()
    return d


def test_of_is_none_without_the_key_and_the_module_with_it(fam_dir):
    assert BK.of("plain") is None
    assert BK.of({"name": "x"}) is None
    assert BK.of("nonexistent") is None
    assert BK.of("fake") is fake_backend
    assert BK.of({"backend": "fake"}) is fake_backend
    assert BK.table()["bench"] == "backend_bench"


def test_an_unknown_backend_name_stops_with_the_known_ones(fam_dir):
    with pytest.raises(SystemExit, match="known: bench, fake"):
        BK.of({"backend": "nope"})


def test_layout_is_todays_constants_for_scene_and_the_backends_for_fake(fam_dir):
    scene = BK.layout("plain")
    assert scene["skills"] == REPO / "src" / "locomani" / "tasks" / "scene" / "skills"
    assert scene["rules"].name == "rules_scene.md" and scene["prompts"].name == "prompts"
    assert scene["plan"] == REPO / "docs/p.md"
    assert scene["caps"] == BK.SCENE_CAPS and scene["caps"]["min_net"] == 3
    fake = BK.layout("fake")
    assert fake["skills"].name == "fake_skills" and fake["prompts"].name == "fake_prompts"
    assert (
        fake["caps"]["claims"] is False and "repeat" not in fake["caps"]
    )  # a missing key falls back to the scene value
    assert fake["caps"]["min_net"] == 2


def test_batch_delegates_run_jobs_and_signature(fam_dir):
    B = _rsi("batch")  # the warehouse loop's batch.py is `batch` too when the whole tests/ folder is collected

    j = B.Job(Path("/s"), Path("/o"))
    assert B.run_jobs([j], "fake", Path("/p"), 1, Path("/l"), "0", where="gate") == {0: []}
    assert B.run_jobs([j], "fake", Path("/p"), 1, Path("/l"), "0") == {0: []}
    assert fake_backend.CALLS == [("run_jobs", "fake", "gate", 1), ("run_jobs", "fake", "probe", 1)]
    assert B.signature("fake", "gate") == "fake-gate"
    assert B.signature("fake") == "fake-probe"


def test_family_load_and_draw_go_to_the_backend(fam_dir, tmp_path, monkeypatch):
    fam = BK.load("fake")
    assert fam["backend"] == "fake" and "meta" not in fam
    res = BK.draw(fam, "diag", tmp_path / "pool", n=3, tasks=["XTask"])
    assert res["written"] == 3 and sorted(p.stem for p in (tmp_path / "pool").glob("*.json")) == [
        "XTask_s0",
        "XTask_s1",
        "XTask_s2",
    ]
    assert ("draw", "diag", 3, ["XTask"]) in fake_backend.CALLS


def test_gate_reads_caps_for_min_net_and_sentinel(fam_dir):
    pytest.importorskip("skfmm")
    G = _rsi("gate")

    t = dict(b=2, c=0, candidate_fell=0, incumbent_fell=0, candidate_dropped=0, incumbent_dropped=0, candidate_ok=2,
             incumbent_ok=0)  # fmt: skip
    assert G.verdict_fails(t) and not G.verdict_fails(t, BK.caps("fake")["min_net"])
    inc = {"a": dict(ok=True, t_end=10.0), "b": dict(ok=False, t_end=5.0)}
    again = {"a": dict(ok=True, t_end=10.5), "b": dict(ok=False, t_end=5.0)}
    assert G.sentinel_drift(again, inc) == ["a"] and G.sentinel_drift(again, inc, False) == []
    assert G.sentinel_drift({"a": dict(ok=False, t_end=10.0)}, inc, False) == ["a"]


def test_loop_layout_caps_cards_and_leaks(fam_dir, tmp_path, monkeypatch):
    import rsi_loop_ns as NS

    L = NS.fresh_loop(scene=False)
    monkeypatch.setattr(L.BK, "HERE", tmp_path)
    monkeypatch.setattr(L, "HERE", tmp_path)
    for name in ("SKILLS", "RULES", "PROMPTS", "CAPS", "PLAN", "FAMILY", "LOOP", "HIDDEN", "STATE", "GATE_DIR"):
        monkeypatch.setattr(L, name, getattr(L, name, None), raising=False)
    L.configure("fake")
    assert L.SKILLS.name == "fake_skills" and L.PROMPTS.name == "fake_prompts" and L.CAPS["claims"] is False
    assert L.module("state").H.before_review({}, tmp_path, 1, tmp_path, False) is None  # no scene hooks for a backend
    monkeypatch.setattr(L, "STATE", tmp_path / "state")
    base = tmp_path / "base"
    base.mkdir()
    (base / "XTask_s0.json").write_text(json.dumps(dict(ok=False, label="XTask_s0", task="XTask")))
    out = L.reviewer_cards(1, base, tmp_path / "it")
    assert out.read_text() == "# fake cards\n" and ("cards", 1) in fake_backend.CALLS
    assert "backend_0" in L.leak_patterns()


def test_sandbox_hides_ids_of_a_non_diagnostic_pool(tmp_path):
    pytest.importorskip("skfmm")
    import sandbox as SB

    f = tmp_path / "scripts" / "rsi" / "families" / "fam.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(dict(pools=dict(diag=dict(ids=["a"]), accept=dict(ids=["b", "c"], n=2)))))
    assert SB.hide_seeds(tmp_path, "fam") == {"accept": {"ids": ["b", "c"]}}
    assert json.loads(f.read_text())["pools"] == dict(diag=dict(ids=["a"]), accept=dict(ids=None, n=2))
    assert json.loads((tmp_path / SB.HIDDEN / "fam" / "seeds.json").read_text()) == {"accept": {"ids": ["b", "c"]}}


def test_cards_and_calibrate_do_not_need_a_scene(tmp_path):
    pytest.importorskip("skfmm")
    sys.path.insert(0, str(HERE / "scene"))
    C = _rsi("calibrate")
    assert isinstance(C.measured_goals("no-such-sample"), set)


def test_gate_sign_test_replaces_loss_share_for_a_noisy_bed():
    G = _rsi("gate")

    t = lambda b, c: dict(b=b, c=c, p_one_sided=G.mcnemar_one_sided(b, c), candidate_fell=0, incumbent_fell=0,
                          candidate_dropped=0, incumbent_dropped=0, incumbent_ok=5, candidate_ok=5 + b - c)  # fmt: skip  # noqa: E731
    assert G.verdict_fails(t(12, 5))  # the scene rule: losses above a third of the wins
    assert not G.verdict_fails(t(12, 4), 3, 0.05) and not G.verdict_fails(t(17, 7), 3, 0.05)
    assert G.verdict_fails(t(8, 2), 3, 0.05) and G.verdict_fails(t(6, 3), 3, 0.05)
    assert G.verdict_fails(t(2, 0), 3, 0.05)  # the net threshold still holds


def test_probe_scope_takes_a_hypothesis_tasks_then_its_predicted_starts(tmp_path, monkeypatch):
    import rsi_loop_ns as NS

    L = NS.fresh_loop(scene=False)
    pool = tmp_path / "pool_diag"
    pool.mkdir()
    for t in ("A", "B", "C"):
        for i in range(2):
            (pool / f"{t}_s{i}.json").write_text("{}")
    monkeypatch.setattr(L, "LOOP", tmp_path)
    monkeypatch.setattr(L, "CAPS", dict(BK.SCENE_CAPS))
    assert L.probe_labels(dict(tasks=["A"])) is None  # a scene family: the whole pool, as before
    monkeypatch.setattr(L, "CAPS", dict(BK.SCENE_CAPS, probe_scope=True))
    assert L.probe_labels(dict(tasks=["A", "nope"])) == ("A_s0", "A_s1")
    assert L.probe_labels(dict(predicted_fixed=["B_s1"], predicted_broken=["C_s0"])) == ("B_s0", "B_s1", "C_s0", "C_s1")
    assert L.probe_labels(dict(predicted_fixed=[], claim="about A")) is None

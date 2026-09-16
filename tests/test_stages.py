"""Stages meet only through manifests (docs/handoff/infra-plan-20260916.md decision 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locomani import conf
from locomani.stages import collect, common


@pytest.fixture
def run_cfg():
    return conf.load_run(
        ["+experiment=bottle_simple_lowcom", "collect.successes=2", "collect.batch_rounds=2", "collect.max_batches=1"]
    )


def test_config_hash_covers_only_the_groups_a_stage_depends_on():
    _, a = conf.load_run(["collect.successes=8"])
    _, b = conf.load_run(["collect.successes=9"])
    _, c = conf.load_run(["collect.successes=8", "train.steps=1"])
    assert common.config_hash(a, "collect") != common.config_hash(b, "collect")
    assert common.config_hash(a, "collect") == common.config_hash(c, "collect")
    assert common.config_hash(a, "train") != common.config_hash(c, "train")


def test_upstream_manifest_is_required_and_hashed(tmp_path, run_cfg):
    run, cfg = run_cfg
    with pytest.raises(RuntimeError, match="needs a completed collect"):
        common.require_upstream(tmp_path, "convert")
    common.write_manifest(
        tmp_path,
        "collect",
        cfg,
        ok=True,
        outputs=dict(sessions="[]", manifests="[]"),
        metrics={},
        started=0.0,
        resources={},
    )
    assert common.require_upstream(tmp_path, "convert")["stage"] == "collect"
    m = common.write_manifest(tmp_path, "convert", cfg, ok=True, outputs={}, metrics={}, started=0.0, resources={})
    assert m["upstream"] == {"collect": common.manifest_hash(tmp_path, "collect")}
    assert json.loads(common.manifest_path(tmp_path, "convert").read_text())["config_hash"] == common.config_hash(
        cfg, "convert"
    )


def test_collect_stage_runs_batches_on_its_own_stack_and_writes_the_manifest(tmp_path, run_cfg, monkeypatch):
    run, cfg = run_cfg
    out = tmp_path / "run"
    events = []

    class FakeStack:
        def __init__(self, spec, *, base_env=None, record_command=None):
            self.spec, self.env = spec, dict(base_env or {})
            events.append(("new", spec.instance, spec.gpu, spec.record, spec.scene_dir))

        def up(self):
            events.append(("up",))
            return self.env

        def down(self):
            events.append(("down",))

    def fake_check(self, name, argv, env=None, cwd=None):
        events.append(("cmd", name))
        argv = [str(a) for a in argv]
        manifest = Path(argv[argv.index("--manifest") + 1])
        assert argv[argv.index("--seed") + 1] == "30700" and "--cases" in argv and "--record" in argv
        assert env["LOCOMANI_SIM_DIR"] == str(out / "scene") and env["CUDA_VISIBLE_DEVICES"] == "0"
        manifest.write_text(
            "\n".join(json.dumps(dict(case_id=f"c{i}", ok=True, produced=True)) for i in range(2)) + "\n"
        )

    monkeypatch.setattr(collect, "Stack", FakeStack)
    monkeypatch.setattr(collect, "_wait_session", lambda sessions, before, timeout=60.0: tmp_path / "session")
    monkeypatch.setattr(collect.Stage, "check", fake_check)
    m = collect.run(run, cfg, out)
    assert m["ok"] and m["metrics"]["kept"] == 2 and m["metrics"]["rounds"] == 2 and m["metrics"]["yield_"] == 1.0
    assert [e[0] for e in events] == ["new", "up", "cmd", "down"]
    assert events[0][1:] == (0, "0", True, str(out / "scene"))
    assert (out / "scene" / "scene_meta.json").exists() and (out / "task_config.json").exists()
    assert json.loads((out / "collect" / "inputs.json").read_text())["kept"] == 2
    assert "resources_at_start" in m and m["upstream"] == {}

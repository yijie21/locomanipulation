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
    patched = tmp_path / "deploy"
    patched.write_bytes(b"POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1")
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(patched))
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


@pytest.mark.parametrize("steps,save_steps", [(6, 3), (5, 3)])
def test_train_stage_delivers_the_save_interval_to_its_training_process(tmp_path, monkeypatch, steps, save_steps):
    import sys

    from locomani.stages import train

    run, cfg = conf.load_run([f"train.steps={steps}", f"train.save_steps={save_steps}", "gpu=1"])
    out = tmp_path / "run"
    ds = out / "train_dataset"
    ds.mkdir(parents=True)
    common.write_manifest(out, "collect", cfg, ok=True, outputs={}, metrics={}, started=0.0, resources={})
    common.write_manifest(
        out, "convert", cfg, ok=True, outputs=dict(train_dataset=ds), metrics={}, started=0.0, resources={}
    )
    original_check = train.Stage.check

    def training_stub(self, name, command, env=None, cwd=None):
        assert name == "train" and str(command[1]).endswith("scripts/train_bottle.sh")
        original_check(
            self,
            name,
            [
                sys.executable,
                "-c",
                'import os,json; print(json.dumps({k: os.environ[k] for k in ["MAX_STEPS", "SAVE_STEPS", "MODALITY", "CUDA_VISIBLE_DEVICES"]}))',
            ],
            env=env,
        )
        (train.checkpoint_path(out, run.train)).mkdir(parents=True)

    monkeypatch.setattr(train.Stage, "check", training_stub)
    m = train.run(run, cfg, out)
    received = json.loads((out / "train.log").read_text().splitlines()[-1])
    assert received["MAX_STEPS"] == str(steps) and received["SAVE_STEPS"] == str(min(save_steps, steps))
    assert (
        received["MODALITY"].endswith(run.embodiment.gr00t_modality_config) and received["CUDA_VISIBLE_DEVICES"] == "1"
    )
    assert m["ok"] and m["outputs"]["checkpoint"].endswith("model" if steps % save_steps else f"checkpoint-{steps}")
    assert m["upstream"] == {"convert": common.manifest_hash(out, "convert")}


def test_collect_stage_rejects_a_short_case_bank_before_building_a_stack(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from locomani.pipeline.experiment_cases import save_cases

    patched = tmp_path / "deploy"
    patched.write_bytes(b"POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1")
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(patched))
    bank = tmp_path / "cases.json"
    save_cases(bank, [], Path(conf.paths.REPO) / "outputs/scene/sim_lowcom/scene_meta.json")
    run, cfg = conf.load_run(["+experiment=bottle_simple_lowcom", f"scene.cases={bank}"])
    monkeypatch.setattr(collect, "Stack", Mock(side_effect=AssertionError("must reject before resource acquisition")))
    with pytest.raises(ValueError, match="fixed cases must cover"):
        collect.run(run, cfg, tmp_path / "run")


def test_collect_stage_rejects_a_deploy_without_the_hand_patch_before_any_stack(tmp_path, monkeypatch):
    from unittest.mock import Mock

    binary = tmp_path / "deploy"
    binary.write_bytes(b"POSE_UPPER_OWNERSHIP_V2")
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(binary))
    run, cfg = conf.load_run(["task=bottle_whole_body", "scene=lowcom", "collect=fixed8"])
    monkeypatch.setattr(collect, "Stack", Mock(side_effect=AssertionError("stack started before admission")))
    with pytest.raises(RuntimeError, match="sonic_hand_ownership.patch"):
        collect.run(run, cfg, tmp_path / "run")


def test_crate_collect_runs_the_expert_with_the_encoder_on_and_the_record_flag(tmp_path, monkeypatch):
    run, cfg = conf.load_run(
        [
            "task=crate_table30",
            "embodiment=g1_bimanual",
            "scene=crate_table30_depth04",
            "collect.successes=2",
            "collect.batch_rounds=2",
            "collect.max_batches=1",
            "collect.seed0=7000",
        ]
    )
    scene = tmp_path / "crate_table30_depth04"
    scene.mkdir()
    (scene / "scene_meta.json").write_text("{}")
    monkeypatch.setattr("locomani.tasks.crate.prepare.prepare_scene", lambda *a, **k: scene)
    seen = {}

    class FakeStack:
        def __init__(self, spec, *, base_env=None, record_command=None):
            seen["spec"] = spec
            self.env = dict(base_env or {})

        def up(self):
            return self.env

        def down(self):
            seen["down"] = True

    def fake_run(self, name, argv, env=None, cwd=None):
        argv = [str(a) for a in argv]
        seen["argv"], seen["env"], seen["name"] = argv, env, name
        Path(argv[argv.index("--manifest") + 1]).write_text(
            "\n".join(json.dumps(dict(ok=True, produced=True)) for _ in range(2)) + "\n"
        )
        return 0

    monkeypatch.setattr(collect, "Stack", FakeStack)
    monkeypatch.setattr(collect.Stage, "run", fake_run)
    monkeypatch.setattr(collect, "_wait_session", lambda sessions, before, timeout=60.0: tmp_path / "session")
    m = collect.run(run, cfg, tmp_path / "run")
    assert m["ok"] and m["metrics"]["kept"] == 2 and seen["down"]
    assert seen["name"] == "visual_00" and seen["argv"][2].endswith("tasks/crate/expert.py")
    assert "--record" in seen["argv"] and seen["argv"][seen["argv"].index("--label") + 1] == "visual_crate_s7000"
    assert seen["env"]["NO_ENCODER"] == "0" and seen["env"]["RECORD_INSTANCE"] == "1"
    assert seen["spec"].encoder and seen["spec"].scene_dir == str(scene) and seen["spec"].record

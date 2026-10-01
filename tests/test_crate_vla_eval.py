"""The crate closed-loop evaluator: action decode, failure strings, driver arguments."""

import json
from pathlib import Path

import numpy as np
import pytest

from locomani.backends.sonic.sonic_token import GRIPPER_MAX, GRIPPER_MIN, expand_hands_bimanual, hand_from_gripper
from locomani.pipeline import eval_crate_closed_loop as driver
from locomani.pipeline.closed_loop import failure_stage
from locomani.tasks.crate.vla_rounds import classify


def test_bimanual_hands_mirror_each_scalar_onto_its_own_jaws():
    left, right = expand_hands_bimanual(-0.05, 0.1)  # out of range on both sides -> clipped
    np.testing.assert_allclose(left[:2], GRIPPER_MIN)
    np.testing.assert_allclose(right[:2], GRIPPER_MAX)
    assert not left[2:].any() and not right[2:].any()
    assert hand_from_gripper(0.0).shape == (7,) and hand_from_gripper(0.0).dtype == np.float32


def test_failure_strings_follow_the_single_arm_line():
    assert classify(0.75, False, 0.0) == "[vla] walk: never closer than 0.75 m"
    assert classify(0.31, False, 0.0) == "[vla] reach: at the table (min 0.31 m) but never commanded a close"
    assert classify(0.31, True, 0.042) == "[vla] close: commanded a grasp but no lift (object moved 4 cm laterally)"
    assert failure_stage(classify(0.31, True, 0.0)) == "close"
    assert failure_stage("[stand] robot did not stabilise after reset") == "pre-vla"


def test_aggregate_counts_successes_stages_and_wilson(tmp_path):
    m = tmp_path / "closed_loop.manifest.jsonl"
    rows = [
        dict(ok=True, why="ok"),
        dict(ok=False, why=classify(0.7, False, 0)),
        dict(ok=False, why=classify(0.3, False, 0)),
    ]
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = driver.aggregate(m)
    assert (summary["rounds"], summary["successes"]) == (3, 1)
    assert summary["failure_stages"] == {"walk": 1, "reach": 1}
    lo, hi = summary["wilson95"]
    assert 0 <= lo < 1 / 3 < hi <= 1


def test_driver_refuses_instance_0_and_missing_checkpoints(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    base = ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path / "o")]
    args = driver.parse_args(base)
    assert (args.instance, args.port, args.gpu, args.serve_gpu) == (1, 5700, "1", "1")
    with pytest.raises(SystemExit):
        driver.parse_args(base + ["--instance", "0"])
    with pytest.raises(SystemExit):
        driver.parse_args(["--checkpoint", str(tmp_path / "none"), "--scene", str(tmp_path), "--out", str(tmp_path)])


def test_backbone_path_reaches_serve_policy_only_when_given(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    backbone = tmp_path / "Cosmos-Reason2-2B"
    backbone.mkdir()
    (backbone / "config.json").write_text("{}")
    base = ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path / "o")]
    assert "--backbone-path" not in driver.serve_command(driver.parse_args(base))
    cmd = driver.serve_command(driver.parse_args(base + ["--backbone-path", str(backbone)]))
    assert cmd[cmd.index("--backbone-path") + 1] == str(backbone)
    assert cmd[cmd.index("--model-path") + 1] == str(ckpt)
    with pytest.raises(SystemExit):
        driver.parse_args(base + ["--backbone-path", str(tmp_path / "missing")])


def test_serve_env_pins_the_gpu_and_unbuffers_the_listening_line(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    env = driver.serve_env(
        driver.parse_args(
            ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path), "--serve-gpu", "0"]
        )
    )
    assert (env["CUDA_VISIBLE_DEVICES"], env["PYTHONUNBUFFERED"]) == ("0", "1")


def test_the_batch_warms_up_before_the_first_recorded_round(monkeypatch, tmp_path):
    """No recorded round may be the deploy's first.

    The deploy's first round in a process passes through its reference motion's
    frame-0 crouch and steps out of it at the mode switch: 0.5 rad of knee and
    3.3 cm of pelvis height inside 0.3 s, the lurch at the start of every
    evaluation video. Every later round is transient-free, so the batch spends
    one unrecorded prelude on it (docs/handoff/crate-eval-prelude-twitch-20260917.md).
    """
    from locomani.tasks.crate import vla_rounds

    order, placements = [], []

    def fake_prelude(p, reset, meta, cfg, rng):
        placements.append(float(rng.random()))
        order.append("prelude")
        return {}, None

    monkeypatch.setattr(vla_rounds, "prelude", fake_prelude)

    rng_a, rng_b = np.random.default_rng([6000, 0xB1A5]), np.random.default_rng(6000)
    vla_rounds.warm_up(None, None, None, None, rng_a)
    order.append("round 0")
    fake_prelude(None, None, None, None, rng_b)

    assert order == ["prelude", "round 0", "prelude"]
    # The warm-up draws from its own stream, so round 0 still gets seed 6000's placement.
    assert placements[0] != placements[1]
    assert placements[1] == float(np.random.default_rng(6000).random())


def test_the_whole_eval_block_is_delivered_or_refused_but_never_silently_dropped(tmp_path):
    """2026-09-18: eval_config.json recorded `client: async` while the loop ran
    blocking, because the driver hand-picked thirteen flags and ignored the rest."""
    from locomani.pipeline.eval_config import EvalConfig

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    base = ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path / "o")]

    honoured = tmp_path / "eval_config.json"
    EvalConfig(rounds=7, seed=41, client="blocking", policy_port=5701, budget_s=42.0).dump(honoured)
    args = driver.parse_args(base + ["--eval-config", str(honoured)])
    assert (args.rounds, args.seed, args.port, args.budget_s) == (7, 41, 5701, 42.0)
    # What it cannot implement is named, not dropped.
    assert "spawn_edge" in args.ignored_eval_fields and "backend" in args.ignored_eval_fields
    assert not set(driver.HONOURED) & set(args.ignored_eval_fields)

    mirrored = tmp_path / "mirrored.json"
    EvalConfig(client="blocking", image_source="isaac").dump(mirrored)
    assert driver.parse_args(base + ["--eval-config", str(mirrored)]).image_source == "isaac"
    assert driver.parse_args(base).image_source == "mujoco"  # no run eval block: the sim's own stream
    source = (Path(driver.__file__)).read_text()
    assert "image_source=args.image_source," in source  # into vla_rounds.run, per batch

    lying = tmp_path / "lying.json"
    EvalConfig(client="async").dump(lying)
    with pytest.raises(SystemExit):
        driver.parse_args(base + ["--eval-config", str(lying)])


def test_the_crate_trace_carries_the_two_posture_channels_it_was_missing():
    """2026-09-18: the crate trace had pelvis_xy/pelvis_yaw/fallen and nothing else,
    so "did it lurch or fall at the start" could not be answered from its artifacts."""
    from locomani.tasks.crate.vla_rounds import posture

    upright = posture({"pelvis": {"pos": [0.0, 0.0, 0.788], "quat": [1.0, 0.0, 0.0, 0.0]}})
    assert upright["pelvis_z"] == pytest.approx(0.788)
    assert upright["torso_pitch"] == pytest.approx(0.0, abs=1e-9)

    # +ve pitch is leaning forward, the same sign convention as the single-arm trace.
    half = np.sin(np.radians(10) / 2), np.cos(np.radians(10) / 2)
    leaning = posture({"pelvis": {"pos": [0.0, 0.0, 0.75], "quat": [half[1], 0.0, half[0], 0.0]}})
    assert np.degrees(leaning["torso_pitch"]) == pytest.approx(10.0, abs=0.01)

    blank = posture({})
    assert blank == {"pelvis_z": None, "torso_pitch": None}


def test_each_batch_numbers_its_traces_globally_so_they_do_not_overwrite():
    """A 100-round evaluation kept 10 traces: the batch loop ran one process per batch,
    each numbering its rounds from zero into one shared `traces/` directory, so batch 1
    overwrote batch 0. Videos were always per-batch directories; traces were not.
    The driver half is checked by behaviour in test_each_batch_runs_in_this_process_on_its_own_stack."""
    from locomani import paths

    rounds = (Path(paths.REPO) / "src/locomani/tasks/crate/vla_rounds.py").read_text()
    assert '"--round-offset",' in rounds
    assert 'trace_dir / f"round_{index + round_offset:03d}.json"' in rounds


def test_clean_mode_reaches_both_the_round_loop_and_the_stack():
    """The transient was not one thing: leaving streamed mode at the end of every round
    put the NEXT round's 3.4 s settle in planner mode, where IDLE drifts. Both halves of
    the fix -- the keepalive and SONIC_AUTO_START -- hang off this one flag."""
    from locomani import paths

    rounds = (Path(paths.REPO) / "src/locomani/tasks/crate/vla_rounds.py").read_text()
    assert '"--no-clean",' in rounds and "CleanClock(config=vla_cfg)" in rounds
    assert "clean_commander_class(clock)()" in rounds

    driver_source = (Path(paths.REPO) / "src/locomani/pipeline/eval_crate_closed_loop.py").read_text()
    assert "clean_deploy=args.clean" in driver_source
    assert "clean=args.clean," in driver_source
    assert "ap.set_defaults(clean=True)" in driver_source  # clean is the default, not the option


# -- the warehouse (2026-09-27): the rounds start where the training episodes did ----------------

WAREHOUSE_META = {  # walk-in +y, the carton named by the scene, not `blue_crate`
    "robot_start": {"xy": [0.0, -1.205], "yaw_deg": 90.0},
    "target_object": "nor_cal_box",
    "objects": {"nor_cal_box": {"surface": "A", "init_quat": [1, 0, 0, 0]}},
    "surfaces": {"A": {"xy": [0.0, -0.105], "flat_half_extent": [0.6, 0.45]}},
}


class _Cmd:
    def __init__(self):
        self.holds = []

    def reset_pose(self):
        pass

    def set_hands(self, **_):
        pass

    def start_control(self):
        pass

    def hold(self, seconds, **kwargs):
        self.holds.append(kwargs)


class _P:
    def __init__(self):
        self.cmd = _Cmd()

    def obs(self):
        return {"pelvis": {"pos": [0.0, -1.455, 0.79], "quat": [1.0, 0.0, 0.0, 0.0]}}

    def pelvis_yaw(self, _state):
        return 0.0

    def pelvis_xy(self, state):
        return state["pelvis"]["pos"][:2]


class _Reset:
    def __init__(self):
        self.calls = []

    def pin_arms(self):
        pass

    def reset(self, **kwargs):
        self.calls.append(kwargs)


def test_the_rounds_place_as_the_collection_expert_does(monkeypatch):
    """Until 2026-09-27 `place` copied the synthetic-table sampling: robot yaw 0 and `blue_crate`, which in the
    logistics room faced the robot 90 deg off the walk-in and never moved the box (the sim skips the name)."""
    from locomani.tasks.crate import vla_rounds
    from locomani.tasks.crate.expert import start_placement
    from locomani.tasks.crate.visual_config import CrateVisualConfig

    monkeypatch.setattr(vla_rounds.time, "sleep", lambda _s: None)
    cfg = CrateVisualConfig(start_pose="scene", start_advance=-0.25)
    p, reset = _P(), _Reset()
    record = vla_rounds.place(p, reset, WAREHOUSE_META, cfg, np.random.default_rng(6000))

    expert = _Reset()
    placement, expected = start_placement(expert, WAREHOUSE_META, cfg, np.random.default_rng(6000))
    placement("expert")
    assert reset.calls == expert.calls
    assert set(reset.calls[0]["objects"]) == {"nor_cal_box"}
    assert reset.calls[0]["robot"]["yaw"] == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(record["start_xy"], [0.0, -1.455], atol=1e-12)
    assert {k: v for k, v in record.items() if k != "takeover"} == expected
    np.testing.assert_allclose(p.cmd.holds[0]["facing"], (0.0, 1.0, 0.0), atol=1e-12)


def test_the_bimanual_close_reaches_the_depth_its_expert_commanded():
    """The warehouse expert closes to -0.06 (visual.close_target); the bottle floor -0.02 left ~2 N per finger."""
    assert hand_from_gripper(-0.05, -0.06)[0] == pytest.approx(-0.05)
    assert hand_from_gripper(-0.09, -0.06)[0] == pytest.approx(-0.06)
    assert hand_from_gripper(-0.05)[0] == pytest.approx(GRIPPER_MIN)  # the single-arm floor is unchanged
    left, right = expand_hands_bimanual(-0.06, 0.0, -0.06)
    assert (left[0], right[0]) == (pytest.approx(-0.06), 0.0)
    rounds = (Path(driver.__file__).parents[1] / "tasks/crate/vla_rounds.py").read_text()
    assert "low = cfg.visual.close_target" in rounds and "hand_from_gripper(left[j], low)" in rounds


def _state(t, y, fallen=False):
    return {"t": t, "fallen": fallen, "pelvis": {"pos": [0.0, y, 0.79]}}


def test_a_lifted_round_is_also_scored_on_the_collection_retreat():
    from locomani.tasks.crate.visual_config import CrateVisualConfig
    from locomani.tasks.crate.vla_rounds import RetreatWatch

    visual = CrateVisualConfig(retreat_distance=0.3, lift_height=0.1)
    both = ["left", "right"]

    walked = RetreatWatch(_state(10.0, -0.40), np.pi / 2, visual)  # facing +y: backward is -y
    assert not walked.update(_state(10.7, -0.55), 0.10, both)
    assert not walked.update(_state(11.4, -0.71), 0.10, both)  # 0.31 m back: the 1 s settle starts
    assert walked.update(_state(12.5, -0.72), 0.10, both)
    assert walked.record()["reason"] == "completed" and walked.record()["walked"] == pytest.approx(0.32)

    flicker = RetreatWatch(_state(0.0, -0.40), np.pi / 2, visual)
    assert not flicker.update(_state(0.7, -0.45), 0.10, ["left"])  # one check without a hand is a flicker
    assert not flicker.update(_state(1.4, -0.50), 0.10, both)
    assert not flicker.update(_state(2.1, -0.52), 0.05, both)  # below 0.6 x lift_height: lost
    assert flicker.update(_state(2.8, -0.53), 0.04, both)
    assert flicker.record()["reason"] == "dropped"

    stood = RetreatWatch(_state(0.0, -0.40), np.pi / 2, visual)
    assert stood.update(_state(visual.retreat_timeout + 0.1, -0.41), 0.10, both)
    assert stood.record()["reason"] == "timeout"


def test_aggregate_counts_the_retreats_of_the_lifted_rounds(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rows = [
        dict(ok=True, why="ok", retreat=dict(reason="completed"), retreat_ok=True),
        dict(ok=True, why="ok", retreat=dict(reason="dropped"), retreat_ok=False),
        dict(ok=False, why="[vla] fell", retreat=None, retreat_ok=None),
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = driver.aggregate(manifest)
    assert (result["successes"], result["retreated"]) == (2, 1)
    assert result["retreat_reasons"] == {"completed": 1, "dropped": 1}
    manifest.write_text(json.dumps(dict(ok=True, why="ok")) + "\n")  # a crate run without a retreat
    assert "retreated" not in driver.aggregate(manifest)


def test_each_batch_runs_in_this_process_on_its_own_stack(monkeypatch, tmp_path):
    """Since 2026-10-01 the rounds are `vla_rounds.run` in the evaluator's process (TODO 1.16), not a
    `.venv_sim` child per batch: each batch adopts the environment of the stack it just started, and
    gets its own seed and global round offset."""
    from locomani.pipeline import run_context
    from locomani.tasks.crate import vla_rounds

    calls, adopted, started = [], [], []

    def fake_run(config, **batch):
        calls.append(batch)
        with batch["manifest"].open("a") as handle:
            for _ in range(batch["rounds"]):
                handle.write(json.dumps({"ok": True, "why": ""}) + "\n")

    def fake_start(*_a, **kw):
        started.append(kw)
        return {"LOCOMANI_SIM_DIR": kw["extra_env"]["LOCOMANI_SIM_DIR"], "TF_BATCH": str(len(started))}

    monkeypatch.setattr(vla_rounds, "run", fake_run)
    monkeypatch.setattr(run_context, "adopt_env", lambda env: adopted.append(dict(env)))
    monkeypatch.setattr(driver.stack, "start_stack", fake_start)
    monkeypatch.setattr(driver.stack, "stop_instance", lambda *_a, **_kw: None)
    monkeypatch.setattr(driver, "serve", lambda *_a, **_kw: None)
    scene, checkpoint = tmp_path / "scene", tmp_path / "ckpt"
    scene.mkdir()
    (scene / "scene_meta.json").write_text("{}")
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    args = driver.parse_args(
        [
            "--checkpoint", str(checkpoint), "--scene", str(scene), "--out", str(tmp_path / "o"),
            "--rounds", "5", "--batch-rounds", "2", "--seed", "100", "--no-clean",
        ]
    )  # fmt: skip
    driver._evaluate(args)

    assert [(c["rounds"], c["seed"], c["round_offset"]) for c in calls] == [(2, 100, 0), (2, 102, 2), (1, 104, 4)]
    assert all(c["clean"] is False and c["image_source"] == "mujoco" for c in calls)
    assert [env["TF_BATCH"] for env in adopted] == ["1", "2", "3"]
    assert json.loads((tmp_path / "o" / "closed_loop.json").read_text())["successes"] == 5


def test_the_rounds_module_is_imported_after_the_stack_scene_is_adopted(tmp_path):
    """The 1.16 gate's first try (2026-10-01): importing vla_rounds before adopting the stack's environment
    built the IK on outputs/scene/sim, and `paths.adopt_sim_dir` refused to move it. A fresh interpreter, as
    the evaluator is one; the stand-in module records which scene `locomani.paths` named when it was imported."""
    import subprocess
    import sys
    import textwrap

    from locomani import paths

    scene, checkpoint = tmp_path / "scene", tmp_path / "ckpt"
    scene.mkdir()
    (scene / "scene_meta.json").write_text("{}")
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    script = textwrap.dedent(
        f"""
        import importlib.abc, importlib.util, json, sys
        from locomani import paths
        from locomani.pipeline import eval_crate_closed_loop as driver

        seen = {{}}

        class StandIn(importlib.abc.Loader, importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                return importlib.util.spec_from_loader(name, self) if name.endswith("crate.vla_rounds") else None

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                seen["sim_dir"] = paths.SIM_DIR

                def run(config, **batch):
                    with batch["manifest"].open("a") as handle:
                        handle.write(json.dumps({{"ok": True, "why": ""}}) + "\\n")

                module.run = run

        sys.meta_path.insert(0, StandIn())
        driver.stack.start_stack = lambda *a, **kw: {{"LOCOMANI_SIM_DIR": kw["extra_env"]["LOCOMANI_SIM_DIR"]}}
        driver.stack.stop_instance = lambda *a, **kw: None
        driver.serve = lambda *a, **kw: None
        driver._evaluate(driver.parse_args(
            ["--checkpoint", "{checkpoint}", "--scene", "{scene}", "--out", "{tmp_path / "o"}", "--rounds", "1"]
        ))
        print("SEEN" + json.dumps(seen))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=paths.REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(paths.REPO) / "src")},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    seen = json.loads(out[out.index("SEEN") + 4 :].splitlines()[0])
    assert seen["sim_dir"] == str(tmp_path / "o" / "scene")

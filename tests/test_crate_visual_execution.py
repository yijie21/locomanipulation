"""Exercise the expert's camera/control loop, including reused images and final trimming."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_fixture import scene_ik

from locomani import conf
from locomani.tasks.crate import expert
from locomani.tasks.crate.perception import CrateEstimate
from locomani.tasks.crate.visual_approach import ApproachCommand
from locomani.tasks.crate.visual_low_reach import LowReachCommand


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("accepted", [True, False])
def test_reused_image_advances_control_and_only_success_publishes_training_boundary(
    tmp_path, monkeypatch, accepted, record
):
    samples, timeouts, keys = [], [], []
    state = dict(
        t=0.0,
        seq=0,
        joints=np.zeros(29),
        fallen=False,
        pelvis=dict(pos=[-1.8, 0, 0.78], quat=[1, 0, 0, 0], vel=[0, 0, 0]),
        palms={side: dict(hand_q=[-0.015] * 7) for side in ("left", "right")},
    )

    def observe():
        state["seq"] += 1
        state["t"] = state["seq"] * 0.02
        return dict(state)

    commander = SimpleNamespace(reference_observer=None, key=keys.append, hold=lambda *a, **kw: None)
    p = SimpleNamespace(cmd=commander, obs=observe, pelvis_yaw=lambda raw: 0)

    class Arm:
        def __init__(self, *args, **kwargs):
            self.q, self.root_quat = np.zeros(29), np.array([1, 0, 0, 0])

        def start(self, **kwargs):
            self._push(self.q)

        def _push(self, q, **kwargs):
            if commander.reference_observer is not None:
                commander.reference_observer("test_reference", dict(joints=q, **kwargs))

        def stop(self):
            pass

    class Body:
        def __init__(self, *args, **kwargs):
            self.count, self.velocity, self.diagnostics = 0, np.zeros(29), {}

        def step(self, wall, estimate, raw, openings, **kwargs):
            self.count += 1
            self.velocity = np.full(29, 0.1)
            samples.append((raw["t"], estimate.timestamp))
            phase = "hold" if self.count >= 3 else "reach"
            finished = self.count >= 5
            return LowReachCommand(
                np.full(29, raw["t"] * 0.1),
                self.velocity,
                np.array([1, 0, 0, 0]),
                1.0,
                phase,
                ("completed" if accepted else "unstable_hold") if finished else "",
                finished,
            )

    stamp = time.time()
    images = {name: np.ones((16, 16, 3), dtype=np.uint8) for name in ("ego_view", "left_wrist", "right_wrist")}
    camera = SimpleNamespace(read=lambda: dict(images=images, timestamps={name: stamp for name in images}))
    guidance = SimpleNamespace(
        diagnostics={},
        update=lambda images, raw, stamp: CrateEstimate(np.array([0.4, 0, -0.4]), np.eye(3), stamp, None, {}),
    )
    monkeypatch.setattr(expert, "ArmIK", lambda path: scene_ik())
    monkeypatch.setattr(expert, "reset_ready", lambda *a, **kw: (Arm(), {"initial_stance_ready": True}))
    monkeypatch.setattr(expert, "StreamedArm", Arm)
    monkeypatch.setattr(expert, "CoordinatedReach", Body)
    monkeypatch.setattr(expert, "AssistedGuidance", lambda ik: guidance)
    monkeypatch.setattr(
        expert, "ApproachPreparation", lambda *a: SimpleNamespace(step=lambda *args: (np.zeros(29), np.zeros(29)))
    )
    monkeypatch.setattr(
        expert,
        "VisualApproach",
        lambda cfg: SimpleNamespace(
            relative_speed=0,
            angular_speed=0,
            stand_attempts=[],
            step=lambda *a, **kw: ApproachCommand("settle", np.zeros(3), ready=True),
        ),
    )
    monkeypatch.setattr(expert, "evaluate", lambda *a, **kw: dict(ok=accepted, why=a[2]))
    reset = SimpleNamespace(start_video=lambda _: None, end_video=lambda: timeouts.append("video_closed"))
    meta = dict(
        objects={"blue_crate": {"surface": "A"}}, surfaces={"A": {"xy": [0, 0], "flat_half_extent": [0.5, 0.5]}}
    )
    result = expert.run_attempt(
        p,
        reset,
        camera,
        meta,
        conf.load_group("task", "crate_visual_low_reach"),
        tmp_path,
        "episode",
        None,
        record=record,
    )
    assert result.ok == accepted, result.execution
    # The exporter protocol: 'c' at task start, then 'c' saves / 'x' discards --
    # exactly one closing key per started recording, so the manifest join holds.
    assert keys == ((["c", "c"] if accepted else ["c", "x"]) if record else [])
    assert result.produced == record
    rows = [json.loads(line) for line in (tmp_path / "episode/commands.jsonl").read_text().splitlines()]
    body_commands = [row for row in rows if row["phase"] in ("reach", "hold")]
    assert len(body_commands) == (5 if accepted else 4)  # one RGB image still drives five physical control updates
    assert len({timestamp for _, timestamp in samples}) == 1
    assert np.all(np.diff([row["simulation_time"] for row in body_commands]) > 0)
    assert all(row["payload"]["reference_velocity"] == [0.1] * 29 for row in body_commands)
    metrics = result.execution["metrics"]
    assert metrics["task"] == "pick up the crate" and metrics["target"] == "blue_crate"
    if accepted:
        assert metrics["training_tail"]["end_seq"] < metrics["evaluation_state_seq"]
        # The converter's trim reads the marker where the single-arm gym puts it.
        assert metrics["execution"]["metrics"]["training_tail"] == metrics["training_tail"]
    else:
        assert "training_tail" not in metrics and "execution" not in metrics
    assert timeouts == ["video_closed"] and commander.reference_observer is None

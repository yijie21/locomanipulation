"""The real-robot inference client (src/real_robot): gripper units, observation assembly, chunk replay, CLI defaults.

No sockets: the robot model is a fake with the exporter's 43-joint layout and FK interface, the policy and the
subscribers are fakes. gear_sonic (in this venv) is imported as NVIDIA's client imports it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tyro

from locomani.backends.sonic.sonic_token import FSQ_STEP
from locomani.schema import SONIC_BOTTLE, SONIC_CRATE
from real_robot import chunk as C
from real_robot import gripper as G
from real_robot import inference as I
from real_robot.observation import build_observation, projected_gravity, to_policy_observation
from real_robot.policy_client import identity_mismatch

# --- gripper units -------------------------------------------------------------------------------------------------


def test_dex1_and_simulator_units_map_linearly_with_the_opposite_sign_and_clip():
    assert G.sim_m_to_dex1(G.SIM_OPEN_M) == pytest.approx(5.5) and G.sim_m_to_dex1(G.SIM_CLOSED_M) == pytest.approx(0)
    assert G.sim_m_to_dex1(-0.06) == 0.0  # the warehouse expert's jaw-loading command is simply 'closed'
    assert G.dex1_to_sim_m(5.5) == pytest.approx(G.SIM_OPEN_M) and G.dex1_to_sim_m(0) == pytest.approx(G.SIM_CLOSED_M)
    mid = G.sim_m_to_dex1((G.SIM_OPEN_M + G.SIM_CLOSED_M) / 2)
    assert mid == pytest.approx(2.75) and G.dex1_to_sim_m(mid) == pytest.approx((G.SIM_OPEN_M + G.SIM_CLOSED_M) / 2)
    np.testing.assert_allclose(G.to_dex1([7.0, -1.0, 3.0], "dex1"), [5.5, 0.0, 3.0])
    with pytest.raises(ValueError, match="gripper units"):
        G.to_dex1(1.0, "radians")


def test_the_wire_field_broadcasts_for_the_robot_and_writes_metres_on_the_jaw_slots_for_the_sim():
    np.testing.assert_array_equal(G.hand_field(5.5, "dex1"), np.full(7, 5.5, np.float32))
    np.testing.assert_array_equal(G.hand_field(-3.0, "dex1"), np.zeros(7, np.float32))
    sim = G.hand_field(5.5, "sim")
    assert sim.dtype == np.float32 and sim[0] == sim[1] == pytest.approx(G.SIM_OPEN_M) and not sim[2:].any()
    with pytest.raises(ValueError, match="hand wire"):
        G.hand_field(1.0, "dex3")


def test_the_hand_state_follows_the_checkpoints_units_and_the_groups_width():
    assert G.dex1_state(np.full(7, 5.39)) == pytest.approx(5.39)  # the robot broadcasts the motor position
    assert G.dex1_state([0.48]) == pytest.approx(0.48)
    np.testing.assert_array_equal(G.hand_state(5.39, 7, "dex1"), np.full(7, 5.39, np.float32))
    np.testing.assert_array_equal(G.hand_state(5.39, 1, "dex1"), np.array([5.39], np.float32))
    sim = G.hand_state(5.5, 7, "sim_m")
    assert sim[4] == sim[5] == pytest.approx(G.SIM_OPEN_M) and not np.delete(sim, [4, 5]).any()
    with pytest.raises(ValueError, match="7-wide"):
        G.hand_state(5.5, 1, "sim_m")


# --- observation ---------------------------------------------------------------------------------------------------

GROUPS = {
    "left_leg": range(0, 6),
    "right_leg": range(6, 12),
    "waist": range(12, 15),
    "left_arm": range(15, 22),
    "left_hand": range(22, 29),
    "right_arm": range(29, 36),
    "right_hand": range(36, 43),
}


class FakeRobotModel:
    """The exporter's model as the client uses it: URDF-order q and joint groups (no FK: the state has no wrist pose)."""

    num_joints = 43

    def get_configuration_from_actuated_joints(
        self, body_actuated_joint_values, left_hand_actuated_joint_values, right_hand_actuated_joint_values
    ):
        body = np.asarray(body_actuated_joint_values, float)
        q = np.zeros(43)
        q[0:15], q[15:22], q[29:36] = body[0:15], body[15:22], body[22:29]
        q[22:29], q[36:43] = left_hand_actuated_joint_values, right_hand_actuated_joint_values
        return q

    def get_joint_group_indices(self, name):
        return list(GROUPS[name])


def _state_msg(hand_value=5.39):
    return {
        "body_q": np.arange(29, dtype=float) / 100,
        "left_hand_q": np.full(7, hand_value),
        "right_hand_q": np.full(7, hand_value / 2),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }


def _images(keys=("ego_view", "left_wrist", "right_wrist")):
    return {k: np.full((480, 640, 3), i, np.uint8) for i, k in enumerate(keys)}


def test_projected_gravity_is_world_gravity_in_the_body_frame():
    np.testing.assert_allclose(projected_gravity([1, 0, 0, 0]), [0, 0, -1])
    half = np.sqrt(0.5)  # pitched 90 degrees nose-down about y: gravity points along the body's +x
    np.testing.assert_allclose(projected_gravity([half, 0, half, 0]), [1, 0, 0], atol=1e-12)


def test_the_observation_is_the_crate_schema_in_flat_keys():
    obs = build_observation(_state_msg(), _images(), "pick up the box", FakeRobotModel(), SONIC_CRATE)
    assert set(obs) == {f"state.{k}" for k in SONIC_CRATE.state_keys} | {
        f"video.{k}" for k in SONIC_CRATE.video_keys
    } | {"annotation.human.task_description"}
    np.testing.assert_allclose(obs["state.left_leg"], [np.arange(6) / 100])
    np.testing.assert_allclose(obs["state.right_arm"], [np.arange(22, 29) / 100])  # body_q's right arm, not q's
    np.testing.assert_allclose(obs["state.left_hand"], [np.full(7, 5.39)], rtol=1e-6)  # the Dex1 scalar broadcast
    np.testing.assert_allclose(obs["state.right_hand"], [np.full(7, 2.695)], rtol=1e-6)
    np.testing.assert_allclose(obs["state.projected_gravity"], [[0, 0, -1]])
    assert sum(obs[f"state.{k}"].shape[-1] for k in SONIC_CRATE.state_keys) == 46  # NVIDIA's state, nothing added
    assert obs["video.right_wrist"].shape == (1, 480, 640, 3) and obs["video.right_wrist"].dtype == np.uint8
    assert obs["video.right_wrist"][0, 0, 0, 0] == 2 and obs["annotation.human.task_description"] == "pick up the box"
    for value in obs.values():
        if isinstance(value, np.ndarray) and value.dtype != np.uint8:
            assert value.dtype == np.float32 and value.shape[0] == 1


def test_the_observation_waits_for_every_camera_the_schema_needs():
    assert build_observation(_state_msg(), _images(("ego_view",)), "p", FakeRobotModel(), SONIC_CRATE) is None
    bottle = build_observation(_state_msg(), _images(("ego_view", "right_wrist")), "p", FakeRobotModel(), SONIC_BOTTLE)
    assert set(k for k in bottle if k.startswith("video.")) == {"video.ego_view", "video.right_wrist"}
    assert {k for k in bottle if k.startswith("state.")} == {f"state.{k}" for k in SONIC_BOTTLE.state_keys}


def test_a_metres_checkpoint_sees_the_simulators_hand_layout():
    obs = build_observation(_state_msg(5.5), _images(), "p", FakeRobotModel(), SONIC_CRATE, gripper_units="sim_m")
    hand = obs["state.left_hand"][0]
    assert hand[4] == hand[5] == pytest.approx(G.SIM_OPEN_M) and not np.delete(hand, [4, 5]).any()


# --- chunk replay --------------------------------------------------------------------------------------------------


def _reply(T=40, grip_left=5.5, grip_right=0.0, token_scale=0.3):
    rng = np.random.default_rng(0)
    return {
        "action.motion_token": (rng.random((T, 64)) - 0.5) * token_scale,
        "action.left_gripper": np.full((T, 1), grip_left),
        "action.right_gripper": np.full(T, grip_right),
    }


def test_a_chunk_is_snapped_to_the_codebook_and_its_grippers_converted_to_dex1():
    chunk = C.ActionChunk.from_policy(_reply(), gripper_units="dex1")
    assert len(chunk) == 40 and chunk.token.dtype == np.float32
    np.testing.assert_allclose(chunk.token / FSQ_STEP, np.round(chunk.token / FSQ_STEP), atol=1e-6)
    token, left, right = chunk.frame(3)
    assert (left, right) == (5.5, 0.0) and token.shape == (64,)
    assert chunk.frame(999)[0] is chunk.token[39] or np.array_equal(chunk.frame(999)[0], chunk.token[39])
    raw = C.ActionChunk.from_policy(_reply(), gripper_units="dex1", snap_tokens=False)
    assert not np.allclose(raw.token / FSQ_STEP, np.round(raw.token / FSQ_STEP))
    metres = C.ActionChunk.from_policy(_reply(grip_left=0.0245, grip_right=-0.06), gripper_units="sim_m")
    assert metres.frame(0)[1:] == pytest.approx((5.5, 0.0))


def test_a_chunk_outside_the_token_bound_or_with_a_short_gripper_is_refused():
    with pytest.raises(ValueError, match="leaves"):
        C.ActionChunk.from_policy(_reply(token_scale=3.0), gripper_units="dex1")
    bad = _reply()
    bad["action.left_gripper"] = bad["action.left_gripper"][:10]
    with pytest.raises(ValueError, match="10 steps"):
        C.ActionChunk.from_policy(bad, gripper_units="dex1")
    with pytest.raises(KeyError, match="right_gripper"):
        C.ActionChunk.from_policy(
            {"motion_token": np.zeros((40, 64)), "left_gripper": np.zeros(40)}, gripper_units="dex1"
        )


# --- the inference runner's helpers (NVIDIA's functions, our formats) ----------------------------------------------


def test_the_policy_observation_is_nested_with_a_time_axis_as_gr00t_get_action_takes_it():
    flat = build_observation(_state_msg(), _images(), "pick up the box", FakeRobotModel(), SONIC_CRATE)
    nested = to_policy_observation(flat)
    assert set(nested) == {"video", "state", "language"}
    assert list(nested["video"]) == list(SONIC_CRATE.video_keys) and set(nested["state"]) == set(SONIC_CRATE.state_keys)
    assert nested["video"]["right_wrist"].shape == (1, 1, 480, 640, 3) and nested["video"]["ego_view"].dtype == np.uint8
    assert nested["state"]["left_arm"].shape == (1, 1, 7) and nested["state"]["left_arm"].dtype == np.float32
    np.testing.assert_array_equal(nested["state"]["left_leg"][0, 0], flat["state.left_leg"][0])
    assert nested["language"] == {"annotation.human.task_description": [["pick up the box"]]}


class _Camera:
    def __init__(self, images):
        self.images = images

    def read(self):
        return None if self.images is None else {"images": self.images}


class _State:
    def __init__(self, msg):
        self.msg = msg

    def get_msg(self):
        return self.msg


def test_the_observation_from_sensors_waits_for_the_camera_the_state_and_every_schema_camera():
    rm = FakeRobotModel()
    assert I.prepare_observation_from_sensors(_Camera(None), _State(_state_msg()), rm, "p") is None
    assert I.prepare_observation_from_sensors(_Camera(_images()), _State(None), rm, "p") is None
    assert I.prepare_observation_from_sensors(_Camera(_images(("ego_view",))), _State(_state_msg()), rm, "p") is None
    obs = I.prepare_observation_from_sensors(_Camera(_images()), _State(_state_msg()), rm, "lift", schema=SONIC_CRATE)
    assert obs["language"]["annotation.human.task_description"] == [["lift"]]
    np.testing.assert_allclose(obs["state"]["projected_gravity"][0, 0], [0, 0, -1], atol=1e-6)  # gear_sonic's


class _Policy:
    def __init__(self, action):
        self.action = action

    def get_action(self, observation, options=None):
        return dict(self.action), {}


def _batched_reply(**kw):
    """What Gr00tPolicy.get_action returns: unprefixed keys, (B, T, D)."""
    return {k.removeprefix("action."): np.asarray(v).reshape(1, 40, -1) for k, v in _reply(**kw).items()}


def test_inference_turns_the_batched_reply_into_a_dex1_chunk_and_skips_one_outside_the_bound():
    reply = _batched_reply(grip_left=0.0245, grip_right=-0.06) | {"task_progress": np.zeros((1, 40, 1))}
    chunk = I.run_policy_inference_and_process(_Policy(reply), {}, None, gripper_units="sim_m")
    assert isinstance(chunk, C.ActionChunk) and len(chunk) == 40 and chunk.frame(0)[1:] == pytest.approx((5.5, 0.0))
    np.testing.assert_allclose(chunk.token / FSQ_STEP, np.round(chunk.token / FSQ_STEP), atol=1e-6)
    assert I.run_policy_inference_and_process(_Policy(_batched_reply(token_scale=3.0)), {}, None) is None
    assert I.run_policy_inference_and_process(_Policy({"motion_token": np.zeros((1, 40, 64))}), {}, None) is None


def test_the_latent_message_takes_seven_wide_hands_and_the_initial_pose_hands_are_dex1():
    with pytest.raises(ValueError, match=r"shape \[7\]"):
        I.pack_latent_action_message(np.zeros(64), np.array([0]), left_hand_joints=np.zeros(6))
    assert isinstance(I.pack_latent_action_message(np.zeros(64), np.array([3]), np.zeros(7), np.zeros(7)), bytes)
    np.testing.assert_array_equal(I.initial_pose_hand_joints(False, "dex1"), np.full(7, 5.5, np.float32))
    np.testing.assert_array_equal(I.initial_pose_hand_joints(True, "dex1"), np.zeros(7, np.float32))
    assert I.get_action_field({"action.motion_token": 1}, "motion_token") == 1


# --- CLI and server identity ---------------------------------------------------------------------------------------


def test_the_defaults_are_the_official_deployments():
    s = tyro.cli(I.InferenceConfig, args=[])
    assert (s.action_publish_rate, s.action_horizon, s.rate) == (50, 40, 2.5)
    assert (s.camera_port, s.state_zmq_port, s.action_zmq_port, s.keyboard_zmq_port, s.port) == (
        5555,
        5557,
        5556,
        5580,
        5590,
    )
    assert (s.schema, s.gripper_units, s.hand_wire, s.init_token_file) == ("crate", "auto", "dex1", None)
    assert s.snap_tokens and s.identity_check and s.prompt == "pick up the box"
    s = tyro.cli(
        I.InferenceConfig,
        args=["--hand-wire", "sim", "--gripper-units", "sim_m", "--rate", "2", "--no-snap-tokens", "--prompt", "lift"],
    )
    assert (s.hand_wire, s.gripper_units, s.rate, s.snap_tokens) == ("sim", "sim_m", 2.0, False)
    for bad in (["--schema", "humanoid"], ["--gripper-units", "rad"], ["--hand-wire", "dex3"]):
        with pytest.raises(SystemExit):
            tyro.cli(I.InferenceConfig, args=bad)


def test_the_served_checkpoint_must_carry_the_schemas_field_order():
    served = {
        "modalities": {
            "video": list(SONIC_CRATE.video_keys),
            "state": list(SONIC_CRATE.state_keys),
            "action": list(SONIC_CRATE.action_keys),
        }
    }
    assert identity_mismatch(served, SONIC_CRATE) == {}
    assert set(identity_mismatch(served, SONIC_BOTTLE)) == {"video", "action"}  # one state for both tasks
    assert set(identity_mismatch({}, SONIC_CRATE)) == {"video", "state", "action"}
    # a checkpoint trained before 2026-10-07 carries the 60-d state with both wrist poses
    old = {"modalities": dict(served["modalities"], state=list(SONIC_CRATE.state_keys) + ["right_wrist_pos"])}
    assert set(identity_mismatch(old, SONIC_CRATE)) == {"state"}


# --- gripper units from the checkpoint's statistics ----------------------------------------------------------------


def test_a_training_range_names_its_gripper_units():
    assert G.units_of_range([-0.06], [0.0245]) == "sim_m"  # the simulator's metres, jaw loading included
    assert G.units_of_range([0.0], [5.5]) == "dex1" and G.units_of_range([0.48], [5.43]) == "dex1"
    for q01, q99 in (([-1.0], [1.0]), ([0.0], [0.3]), ([0.0], [50.0])):  # normalized, half-open, degrees
        with pytest.raises(ValueError, match="neither"):
            G.units_of_range(q01, q99)


def _identity(ranges):
    return {"gripper_ranges": {"action": ranges, "state": {}}}


def test_the_client_takes_the_checkpoints_gripper_units_and_refuses_a_contradiction():
    from real_robot.policy_client import resolve_gripper_units

    metres = _identity({k: {"q01": [-0.06], "q99": [0.0245]} for k in ("left_gripper", "right_gripper")})
    dex1 = _identity({k: {"q01": [0.0], "q99": [5.5]} for k in ("left_gripper", "right_gripper")})
    assert resolve_gripper_units(metres, "auto") == "sim_m" and resolve_gripper_units(dex1, "auto") == "dex1"
    assert resolve_gripper_units(metres, "sim_m") == "sim_m" and resolve_gripper_units(dex1, "dex1") == "dex1"
    with pytest.raises(ValueError, match="contradicts the checkpoint, whose gripper statistics are in sim_m"):
        resolve_gripper_units(metres, "dex1")  # the mistake that closed both grippers at start
    with pytest.raises(ValueError, match="contradicts"):
        resolve_gripper_units(dex1, "sim_m")
    mixed = _identity(
        {"left_gripper": {"q01": [-0.06], "q99": [0.0245]}, "right_gripper": {"q01": [0.0], "q99": [5.5]}}
    )
    with pytest.raises(ValueError, match="disagree"):
        resolve_gripper_units(mixed, "auto")
    # no statistics (a replay server, or a checkpoint without statistics.json): auto cannot guess, explicit passes
    with pytest.raises(ValueError, match="cannot be inferred"):
        resolve_gripper_units({"gripper_ranges": None}, "auto")
    assert resolve_gripper_units({}, "sim_m") == "sim_m"
    with pytest.raises(ValueError, match="not one of"):
        resolve_gripper_units(metres, "radians")


def test_the_server_reads_the_gripper_ranges_from_the_checkpoints_statistics(tmp_path):
    from real_robot.statistics import gripper_ranges

    stats = {
        "unitree_g1_sonic": {
            "action": {
                "motion_token": {"q01": [-0.5] * 64, "q99": [0.5] * 64},
                "left_gripper": {"min": [-0.06], "max": [0.0245], "q01": [-0.06], "q99": [0.0245]},
                "right_gripper": {"min": [-0.06], "max": [0.0245], "q01": [-0.06], "q99": [0.0245]},
            },
            "state": {"left_leg": {"q01": [0] * 6, "q99": [1] * 6}, "left_hand": {"q01": [0] * 7, "q99": [0.0245] * 7}},
        }
    }
    ckpt = tmp_path / "checkpoint-1"
    (ckpt / "experiment_cfg").mkdir(parents=True)
    assert gripper_ranges(ckpt, "unitree_g1_sonic") is None  # no statistics at all
    (ckpt / "experiment_cfg" / "dataset_statistics.json").write_text(json.dumps(stats))  # the training-time copy
    got = gripper_ranges(ckpt, "unitree_g1_sonic")
    assert set(got["action"]) == {"left_gripper", "right_gripper"} and set(got["state"]) == {"left_hand"}
    assert got["action"]["left_gripper"] == {"q01": [-0.06], "q99": [0.0245]}
    (ckpt / "statistics.json").write_text(json.dumps({"other_robot": {}}))  # the processor's copy lacks the tag
    assert gripper_ranges(ckpt, "unitree_g1_sonic") == got  # falls through to the training-time copy
    assert gripper_ranges(ckpt, "other_robot") is None

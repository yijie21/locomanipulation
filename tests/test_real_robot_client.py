"""The real-robot inference client (src/real_robot): gripper units, observation assembly, chunk replay, CLI defaults.

No sockets, no gear_sonic: the robot model is a fake with the exporter's 43-joint layout and FK interface.
"""

from __future__ import annotations

import numpy as np
import pytest

from locomani.backends.sonic.sonic_token import FSQ_STEP
from locomani.schema import SONIC_BOTTLE, SONIC_CRATE
from real_robot import chunk as C
from real_robot import gripper as G
from real_robot.inference import Settings, parse_args
from real_robot.observation import build_observation, projected_gravity
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


class Placement:
    def __init__(self, translation, rotation):
        self.translation, self.rotation = np.asarray(translation, float), np.asarray(rotation, float)


class FakeRobotModel:
    """The exporter's model as the client uses it: URDF-order q, joint groups, FK of the two wrist frames."""

    num_joints = 43

    class supplemental_info:
        hand_frame_names = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

    def __init__(self):
        self.fk_q = None

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

    def cache_forward_kinematics(self, q):
        self.fk_q = np.asarray(q, float)

    def frame_placement(self, name):
        assert self.fk_q is not None, "FK before the placement"
        side = 1.0 if name.startswith("left") else -1.0
        # a 90 degree yaw about z: quaternion (w, x, y, z) = (cos 45, 0, 0, sin 45)
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        return Placement([0.3, side * 0.2, 1.0 + self.fk_q[15]], rot)


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
    np.testing.assert_allclose(obs["state.left_wrist_pos"], [[0.3, 0.2, 1.15]], rtol=1e-6)
    np.testing.assert_allclose(obs["state.right_wrist_pos"], [[0.3, -0.2, 1.15]], rtol=1e-6)
    np.testing.assert_allclose(obs["state.right_wrist_abs_quat"], [[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], atol=1e-6)
    assert obs["video.right_wrist"].shape == (1, 480, 640, 3) and obs["video.right_wrist"].dtype == np.uint8
    assert obs["video.right_wrist"][0, 0, 0, 0] == 2 and obs["annotation.human.task_description"] == "pick up the box"
    for value in obs.values():
        if isinstance(value, np.ndarray) and value.dtype != np.uint8:
            assert value.dtype == np.float32 and value.shape[0] == 1


def test_the_observation_waits_for_every_camera_the_schema_needs():
    assert build_observation(_state_msg(), _images(("ego_view",)), "p", FakeRobotModel(), SONIC_CRATE) is None
    bottle = build_observation(_state_msg(), _images(("ego_view", "right_wrist")), "p", FakeRobotModel(), SONIC_BOTTLE)
    assert set(k for k in bottle if k.startswith("video.")) == {"video.ego_view", "video.right_wrist"}
    assert "state.left_wrist_pos" not in bottle and "state.right_wrist_pos" in bottle


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


def test_latency_compensation_and_inference_scheduling_are_nvidias():
    assert C.latency_index(0.0, 50, 40) == 0 and C.latency_index(0.07, 50, 40) == 4  # 70 ms at 50 Hz: skip 3.5 -> 4
    assert C.latency_index(0.4, 50, 40) == 20 and C.latency_index(5.0, 50, 40) == 39
    assert C.latency_index(0.07, 25, 40) == 2  # the same latency at a 25 fps dataset's publish rate
    assert C.should_infer(False, True, 0.0, 0.4)  # nothing to publish yet: ask at once
    assert not C.should_infer(True, True, 9.0, 0.4)  # never two at a time
    assert not C.should_infer(True, False, 0.3, 0.4) and C.should_infer(True, False, 0.4, 0.4)


# --- CLI and server identity ---------------------------------------------------------------------------------------


def test_the_defaults_are_the_official_deployments():
    s = parse_args([])
    assert (s.action_publish_rate, s.action_horizon, s.rate) == (50, 40, 2.5)
    assert (s.camera_port, s.state_zmq_port, s.action_zmq_port, s.keyboard_zmq_port, s.port) == (
        5555,
        5557,
        5556,
        5580,
        5590,
    )
    assert (s.schema, s.gripper_units, s.hand_wire, s.init_token_file) == ("crate", "dex1", "dex1", None)
    s = parse_args(
        ["--hand-wire", "sim", "--gripper-units", "sim_m", "--rate", "2", "--no-snap", "--prompt", "pick up the box"]
    )
    assert (s.hand_wire, s.gripper_units, s.rate, s.no_snap) == ("sim", "sim_m", 2.0, True)
    assert Settings().prompt == "pick up the box"
    for bad in (["--schema", "humanoid"], ["--gripper-units", "rad"], ["--hand-wire", "dex3"], ["--rate", "0"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_the_served_checkpoint_must_carry_the_schemas_field_order():
    served = {
        "modalities": {
            "video": list(SONIC_CRATE.video_keys),
            "state": list(SONIC_CRATE.state_keys),
            "action": list(SONIC_CRATE.action_keys),
        }
    }
    assert identity_mismatch(served, SONIC_CRATE) == {}
    assert set(identity_mismatch(served, SONIC_BOTTLE)) == {"video", "state", "action"}
    assert set(identity_mismatch({}, SONIC_CRATE)) == {"video", "state", "action"}

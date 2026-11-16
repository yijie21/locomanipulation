"""backends.sonic_onnx without the ONNX files: the deploy's resampling, merge, anchor, VR targets and input layouts."""

import json
import math
import subprocess
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from locomani.backends.sonic.sonic_decoder import ACTION_SCALE, DEFAULT_ANGLES
from locomani.backends.sonic_onnx import runtime as S
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]


def scalebfm_meta() -> dict:
    p = ROOT / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"
    text = (
        p.read_text()
        if p.exists()
        else subprocess.check_output(["git", "show", f"HEAD:{p.relative_to(ROOT)}"], cwd=ROOT)
    )
    return json.loads(text)


def yaw_quat(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


class Gains(unittest.TestCase):
    def test_gains_match_the_deploy_table(self):
        # policy_parameters.hpp:143-207 (left leg, waist, left arm), and kp / kd ratio 4 / w
        np.testing.assert_allclose(S.KP[[0, 2, 3, 4, 12, 13, 15, 20]],
                                   [99.0984, 40.1792, 99.0984, 28.5012, 40.1792, 28.5012, 14.2506, 16.7783], atol=1e-4)  # fmt: skip
        np.testing.assert_allclose(S.KD[[0, 2, 4, 15, 20]], [6.3088, 2.5579, 1.8144, 0.9072, 1.0681], atol=1e-4)
        # action scale = 0.25 x effort (training) / kp: the deploy's table agrees with these gains
        train_effort = S.EFFORT.copy()
        train_effort[[0, 1, 6, 7]] = 139.0
        np.testing.assert_allclose(ACTION_SCALE, 0.25 * train_effort / S.KP, rtol=1e-6)

    def test_meta_keeps_the_sim_order_and_maps_by_name(self):
        meta = scalebfm_meta()
        out = S.sonic_meta(meta)
        self.assertEqual(out["joint_names"], meta["joint_names"])
        for i, n in enumerate(meta["joint_names"]):
            j = MUJOCO_JOINT_NAMES.index(n)
            self.assertAlmostEqual(out["stiffness"][i], S.KP[j])
            self.assertAlmostEqual(out["default_dof_pos"][i], DEFAULT_ANGLES[j])
        self.assertEqual(out["default_dof_pos"][meta["joint_names"].index("left_shoulder_roll_joint")], 0.2)
        self.assertEqual(out["default_dof_pos"][meta["joint_names"].index("right_shoulder_roll_joint")], -0.2)


class Layouts(unittest.TestCase):
    def test_input_sizes_and_teleop_blocks(self):
        self.assertEqual(S.ENC_DIM, 1751)
        self.assertEqual(S.DEC_DIM, 994)
        self.assertEqual(S.ENC["motion_anchor_orientation_heading"], slice(644, 650))
        self.assertEqual(S.ENC["motion_joint_positions_lowerbody_10frame_step5"], slice(650, 770))
        self.assertEqual(S.ENC["vr_3point_local_target"], slice(890, 899))
        self.assertEqual(S.ENC["vr_3point_local_orn_target"], slice(899, 911))
        self.assertEqual(S.DEC["his_gravity_dir_10frame_step1"], slice(964, 994))


def policy_without_sessions(joint_names=MUJOCO_JOINT_NAMES) -> S.SonicPolicy:
    p = S.SonicPolicy.__new__(S.SonicPolicy)
    p.cfg = S.SonicConfig()
    p.to_mj = np.array([list(joint_names).index(n) for n in MUJOCO_JOINT_NAMES])
    p.from_mj = np.array([MUJOCO_JOINT_NAMES.index(n) for n in joint_names])
    p.root_source = p.root_est = None
    p.reset_state()
    return p


class Motion(unittest.TestCase):
    def planner_frames(self, n=44, v=0.3):
        qpos = np.zeros((n, 36))
        qpos[:, 0] = v * np.arange(n) / 30.0
        qpos[:, 2] = 0.79
        qpos[:, 3:7] = yaw_quat(0.0)
        qpos[:, 7:] = DEFAULT_ANGLES + 0.01 * np.arange(n)[:, None]
        return qpos

    def test_resampling_to_50_hz(self):
        m = S.from_planner(self.planner_frames(44))
        self.assertEqual(m.T, int(44 / 30 * 50))
        np.testing.assert_allclose(m.pos[:-1, 0], 0.3 * np.arange(m.T - 1) / 50.0, atol=1e-9)
        np.testing.assert_allclose(m.dq[:-1], (m.q[1:] - m.q[:-1]) * 50.0)
        np.testing.assert_allclose(m.dq[-1], m.dq[-2])
        np.testing.assert_allclose(np.linalg.norm(m.quat, axis=1), 1.0)

    def test_merge_keeps_the_old_motion_until_fgen_then_crossfades_in_8_frames(self):
        old = S.from_planner(self.planner_frames(44, 0.3))
        new = S.from_planner(self.planner_frames(44, 0.0))
        cur, fgen = 3, 5
        out = S.merge(old, new, cur, fgen)
        self.assertEqual(out.T, fgen - cur + new.T)
        start = fgen - cur
        np.testing.assert_allclose(out.q[: start + 1], old.q[cur : cur + start + 1])  # weight 0 through f = start
        np.testing.assert_allclose(out.q[start + 8 :], new.q[8 : 8 + out.T - start - 8])  # weight 1 from start + 8
        w = 0.5
        f = start + 4
        np.testing.assert_allclose(out.q[f], (1 - w) * old.q[f + cur] + w * new.q[f + cur - fgen])
        self.assertIsNone(S.merge(old, new, new.T + 5, 0))  # a plan that ends before the playing frame

    def test_sample_interpolates_between_50_hz_frames(self):
        m = S.from_planner(self.planner_frames(44))
        x = m.sample(1.5 * S.DT)
        np.testing.assert_allclose(x[:3], 0.5 * (m.pos[1] + m.pos[2]))
        np.testing.assert_allclose(x[7:], 0.5 * (m.q[1] + m.q[2]))
        np.testing.assert_allclose(m.sample(100.0)[7:], m.q[-1])


class Frames(unittest.TestCase):
    def test_heading_ignores_tilt_and_the_anchor_maps_planner_to_robot(self):
        tilt = np.array([math.cos(0.2), math.sin(0.2), 0.0, 0.0])
        q = S.qmul(yaw_quat(0.7), tilt)
        np.testing.assert_allclose(S.heading(q), yaw_quat(0.7), atol=1e-12)
        p = policy_without_sessions()
        p.apply = S.qmul(S.heading(yaw_quat(1.0)), S.qconj(S.heading(yaw_quat(0.25))))
        np.testing.assert_allclose(
            p.to_planner([math.cos(1.0), math.sin(1.0), 0.0]), [math.cos(0.25), math.sin(0.25), 0], atol=1e-12
        )

    def test_gravity_upright_and_padded(self):
        np.testing.assert_allclose(S.gravity(yaw_quat(0.4)), [0.0, 0.0, -1.0], atol=1e-12)
        p = policy_without_sessions()
        p.hist = deque(maxlen=S.HISTORY)
        entry = (np.full(3, 0.5), np.full(29, 0.1), np.full(29, 0.2), np.full(29, 0.3), np.array([0.0, 0.0, -1.0]))
        p.hist.append(entry)
        p.hist.append(entry)
        x = p.decoder_input(np.ones(64))
        g = x[S.DEC["his_gravity_dir_10frame_step1"]].reshape(10, 3)
        np.testing.assert_allclose(g[:8], np.tile([0.0, 0.0, 1.0], (8, 1)))  # zero entries in the oldest slots
        np.testing.assert_allclose(g[8:], np.tile([0.0, 0.0, -1.0], (2, 1)))
        a = x[S.DEC["his_last_actions_10frame_step1"]].reshape(10, 29)
        np.testing.assert_allclose(a[:8], 0.0)
        np.testing.assert_allclose(a[8:], 0.3)
        np.testing.assert_allclose(x[S.DEC["token_state"]], 1.0)

    def test_vr_targets_in_the_reference_pelvis_frame(self):
        pel_p, pel_q = np.array([1.0, 2.0, 0.7]), yaw_quat(math.pi / 2)
        links = {
            "left_wrist_yaw_link": (np.array([1.0, 2.3, 0.9]), pel_q),
            "right_wrist_yaw_link": (np.array([1.0, 1.7, 0.9]), pel_q),
            "torso_link": (np.array([1.0, 2.0, 0.75]), pel_q),
        }
        pos, quat = S.vr_targets(links, pel_p, pel_q)
        # the robot faces world +y: the left wrist 0.3 m ahead of the pelvis, plus the point's own offset (0.18, -0.025)
        np.testing.assert_allclose(pos[:3], [0.48, -0.025, 0.2], atol=1e-12)
        np.testing.assert_allclose(pos[3:6], [-0.12, 0.025, 0.2], atol=1e-12)
        np.testing.assert_allclose(pos[6:], [0.0, 0.0, 0.05 + 0.35], atol=1e-12)
        np.testing.assert_allclose(quat, np.tile([1.0, 0.0, 0.0, 0.0], 3), atol=1e-12)

    def test_encoder_input_teleop(self):
        p = policy_without_sessions()
        m = S.from_planner(Motion().planner_frames(44))
        p.motion, p.frame, p.apply = m, 3, yaw_quat(0.0)
        cmd = S.SonicCommand(S.Movement(), np.arange(9.0), np.arange(12.0))
        x = p.encoder_input(yaw_quat(0.0), cmd)
        np.testing.assert_allclose(x[S.ENC["encoder_mode_4"]], [1, 0, 0, 0])
        np.testing.assert_allclose(x[S.ENC["motion_anchor_orientation_heading"]], [1, 0, 0, 1, 0, 0], atol=1e-12)
        legs = x[S.ENC["motion_joint_positions_lowerbody_10frame_step5"]].reshape(10, 12)
        np.testing.assert_allclose(legs[0], m.q[3, :12])
        np.testing.assert_allclose(legs[2], m.q[13, :12])
        np.testing.assert_allclose(legs[-1], m.q[min(3 + 45, m.T - 1), :12])
        np.testing.assert_allclose(x[S.ENC["vr_3point_local_target"]], np.arange(9.0))
        others = np.ones(S.ENC_DIM, bool)
        for k in ("encoder_mode_4", "motion_anchor_orientation_heading", "motion_joint_positions_lowerbody_10frame_step5",
                  "motion_joint_velocities_lowerbody_10frame_step5", "vr_3point_local_target", "vr_3point_local_orn_target"):  # fmt: skip
            others[S.ENC[k]] = False
        self.assertEqual(float(np.abs(x[others]).sum()), 0.0)


class Readapt(unittest.TestCase):
    def test_idle_hold_follows_the_measured_legs_then_recovers(self):
        p = policy_without_sessions()
        p.motion = S.from_planner(Motion().planner_frames(44))
        f = p.motion.T - 1
        target0 = p.motion.q[f, :12].copy()
        meas = np.zeros(29)
        meas[:12] = target0 + 0.2  # 0.2 rad off: ADAPTING
        p._readapt(f, meas)
        self.assertEqual(p.readapt_state, "adapting")
        np.testing.assert_allclose(p.motion.q[f, :12], target0 + 0.02 * 0.2)
        meas[:12] = p.motion.q[f, :12]  # on target: ADAPTING -> IDLE, then IDLE -> RECOVERING
        p._readapt(f, meas)
        self.assertEqual(p.readapt_state, "idle")
        p._readapt(f, meas)
        self.assertEqual(p.readapt_state, "recovering")


if __name__ == "__main__":
    unittest.main()


class SquatHysteresis(unittest.TestCase):
    """SonicMotor's standing branch: IDLE near the standing height, IDEL_SQUAT below squat_enter until above squat_exit,
    the height rounded; a stand re-anchored at a slightly sagged pelvis must not become a squat (the 0.775 bug)."""

    def motor(self):
        from locomani.backends.scalebfm.motor import MODE_STAND, MotorConfig
        from locomani.backends.sonic_onnx.motor import SonicMotor, SonicMotorConfig

        m = SonicMotor.__new__(SonicMotor)
        m.sonic, m.cfg, m.mode, m.squatting = SonicMotorConfig(), MotorConfig(), MODE_STAND, False
        return m

    def movement(self, m, z):
        from locomani.backends.scalebfm.motor import Pose

        return m._movement(Pose(np.array([0.0, 0.0, z]), yaw_quat(0.3), np.zeros(29), np.zeros(29)))

    def test_sagged_stands_stay_idle_and_squats_round_and_hold(self):
        m = self.motor()
        for z in (0.787, 0.770, 0.746, 0.71):
            mv = self.movement(m, z)
            self.assertEqual((mv.mode, mv.height), (S.IDLE, -1.0), z)
        mv = self.movement(m, 0.611)
        self.assertEqual(mv.mode, S.IDEL_SQUAT)
        self.assertAlmostEqual(mv.height, 0.62)
        self.assertEqual(self.movement(m, 0.719).mode, S.IDEL_SQUAT)  # inside the band: still squatting
        self.assertAlmostEqual(self.movement(m, 0.719).height, 0.72)
        self.assertEqual(self.movement(m, 0.735).mode, S.IDLE)  # above squat_exit
        self.assertEqual(self.movement(m, 0.719).mode, S.IDLE)  # and the band does not re-enter from above
        self.assertAlmostEqual(self.movement(m, 0.30).height, 0.40)  # squat_min
        np.testing.assert_allclose(self.movement(m, 0.787).facing, [math.cos(0.3), math.sin(0.3), 0.0], atol=1e-12)

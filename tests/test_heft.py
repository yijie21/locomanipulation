"""backends.heft without the ONNX file: the vendored observation's layout, the gains' name matching, the motor's frame
window (delay, continuity offset), the vendored gait's end stance and gait_heft's step shapes."""

import math
import unittest

import numpy as np

from locomani.backends.heft.obs import HeftObs
from locomani.backends.heft.runtime import match
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

# policy.yaml's policy_joint_names (g1_backend checkpoints/heft/pmg; = ScaleBFM's metadata order)
POLICY_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip
HIST = [0, 1, 2, 3, 4, 8, 12, 16, 20]
STEPS = [0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16]


def heft_obs() -> HeftObs:
    return HeftObs(STEPS, HIST, HIST, HIST, HIST, 8, POLICY_ORDER, MUJOCO_JOINT_NAMES, "pelvis", 25,
                   component="context", policy_variant="pmg")  # fmt: skip


def yaw_quat(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


class Observation(unittest.TestCase):
    def test_the_four_inputs_have_policy_jsons_shapes(self):
        o = heft_obs()
        o.set_state(np.zeros(29), np.zeros(29), [1, 0, 0, 0], np.zeros(3))
        o.set_reference(np.zeros((13, 29)), np.zeros((13, 3)), np.tile([1.0, 0, 0, 0], (13, 1)))
        o.update({"action": np.zeros(29)})
        x = o.inputs()
        self.assertEqual([x[k].shape for k in HeftObs.PARTS], [(1, 1), (1, 114), (1, 806), (1, 808)])
        self.assertEqual(o.compute().shape, (1, 1729))
        np.testing.assert_array_equal(np.concatenate([x[k][0] for k in HeftObs.PARTS]), o.compute()[0])

    def test_a_reference_at_the_robot_reads_as_no_error(self):
        o = heft_obs()
        rng = np.random.default_rng(0)
        q, yaw = rng.uniform(-0.5, 0.5, 29), 0.7
        o.set_state(q, np.zeros(29), yaw_quat(yaw), np.zeros(3))
        o.set_reference(np.tile(q, (13, 1)), np.tile([2.0, -1.0, 0.78], (13, 1)), np.tile(yaw_quat(yaw), (13, 1)))
        for i in range(30):
            o.update({"action": np.zeros(29)})
            if i == 0:
                self.assertAlmostEqual(float(o.inputs()["context"][0, 0]), 24 / 25)
        x = o.inputs()
        self.assertEqual(float(x["context"][0, 0]), 0.0)  # the boot indicator is spent after 25 steps
        cmd = x["motion_command"][0]
        np.testing.assert_allclose(cmd[:36], 0.0, atol=1e-6)  # 12 root displacements from frame 0
        np.testing.assert_allclose(cmd[36:].reshape(13, 6), np.tile([1, 0, 0, 0, 1, 0], (13, 1)), atol=1e-6)  # rot6d
        tm = x["target_motion"][0]
        np.testing.assert_allclose(tm[:377].reshape(13, 29), np.tile(q[[MUJOCO_JOINT_NAMES.index(n) for n in POLICY_ORDER]], (13, 1)), atol=1e-6)  # fmt: skip
        np.testing.assert_allclose(tm[377:754], 0.0, atol=1e-6)  # reference minus current
        np.testing.assert_allclose(tm[754:767], 0.78, atol=1e-6)  # the reference pelvis height, absolute
        np.testing.assert_allclose(tm[767:].reshape(13, 3), np.tile([0, 0, -1.0], (13, 1)), atol=1e-6)

    def test_a_reference_moving_ahead_reads_in_frame_0s_heading(self):
        o = heft_obs()
        yaw = math.pi / 2  # facing +y: a +y step of the reference is "forward"
        o.set_state(np.zeros(29), np.zeros(29), yaw_quat(yaw), np.zeros(3))
        root = np.array([[0.0, 0.01 * k, 0.78] for k in STEPS])
        o.set_reference(np.zeros((13, 29)), root, np.tile(yaw_quat(yaw), (13, 1)))
        o.update({"action": np.zeros(29)})
        d = o.inputs()["motion_command"][0, :36].reshape(12, 3)
        np.testing.assert_allclose(d[:, 0], 0.01 * np.array(STEPS[1:]), atol=1e-6)
        np.testing.assert_allclose(d[:, 1:], 0.0, atol=1e-6)

    def test_previous_actions_are_kept_newest_first(self):
        o = heft_obs()
        o.set_state(np.zeros(29), np.zeros(29), [1, 0, 0, 0], np.zeros(3))
        o.set_reference(np.zeros((13, 29)), np.zeros((13, 3)), np.tile([1.0, 0, 0, 0], (13, 1)))
        for a in (1.0, 2.0, 3.0):
            o.update({"action": np.full(29, a)})
        prev = o.inputs()["proprioception"][0, -8 * 29 :].reshape(8, 29)
        np.testing.assert_array_equal(prev[:4, 0], [3.0, 2.0, 1.0, 0.0])


class Gains(unittest.TestCase):
    def test_regex_tables_match_by_full_name(self):
        kp = {".*_hip_pitch_joint": 99.1, ".*_knee_joint": 99.1, "waist_yaw_joint": 40.2, ".*_wrist_pitch_joint": 8.6}
        out = match(kp, ["left_hip_pitch_joint", "waist_yaw_joint", "right_wrist_pitch_joint"])
        np.testing.assert_allclose(out, [99.1, 40.2, 8.6])
        with self.assertRaisesRegex(ValueError, "no match"):
            match(kp, ["waist_roll_joint"])
        np.testing.assert_allclose(match(kp, ["waist_roll_joint"], strict=False), [0.0])
        with self.assertRaisesRegex(ValueError, "multiple"):
            match({".*hip.*": 1.0, ".*_hip_pitch_joint": 2.0}, ["left_hip_pitch_joint"])


class FrameWindow(unittest.TestCase):
    def motor(self, delay_f: int):
        from locomani.backends.heft.motor import HeftMotor

        m = HeftMotor.__new__(HeftMotor)
        m.steps, m.delay_f, m.k = np.array(STEPS), delay_f, 0
        m.log, m.offset, m._ref_seen = [], np.zeros(2), None
        return m

    @staticmethod
    def moving(t0: float = 0.0, x0: float = 0.0):
        from locomani.backends.scalebfm.motor import Pose

        return lambda t: Pose(np.array([x0 + (t - t0), 0.0, 0.78]), yaw_quat(0.0), np.full(29, t), np.zeros(29))

    def test_no_delay_reads_ahead_and_the_log_behind(self):
        m = self.motor(0)
        m.ref = self.moving()
        for _ in range(30):
            f = m._frames()
            m.k += 1
        t = (m.k - 1) * 0.02
        np.testing.assert_allclose(f[:, 0], [t + 0.02 * k for k in STEPS], atol=1e-9)
        np.testing.assert_allclose(f[:, 7], [t + 0.02 * k for k in STEPS], atol=1e-9)

    def test_lml_delay_reads_8_frames_back_and_nothing_newer_than_2(self):
        m = self.motor(8)
        m.ref = self.moving()
        for _ in range(40):
            f = m._frames()
            m.k += 1
        t = (m.k - 1) * 0.02
        np.testing.assert_allclose(f[:, 0], [t + 0.02 * (k - 8) for k in STEPS], atol=1e-9)
        self.assertLessEqual(f[:7, 0].max(), t - 0.04 + 1e-9)

    def test_the_log_starts_with_the_first_frame(self):
        m = self.motor(8)
        m.ref = self.moving()
        f = m._frames()
        np.testing.assert_allclose(f[:, 0], 0.0)

    def test_a_new_reference_elsewhere_is_carried_onto_the_last_frame(self):
        m = self.motor(0)
        m.ref = self.moving()
        for _ in range(10):
            m._frames()
            m.k += 1
        m.ref = self.moving(t0=m.k * 0.02, x0=5.0)  # e.g. the measured stance after a walk, 5 m away in the reference
        f = m._frames()
        np.testing.assert_allclose(m.offset, [m.k * 0.02 - 5.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(f[0, 0], m.k * 0.02, atol=1e-9)  # continuous: frame 0 where the old one would be
        np.testing.assert_allclose(np.diff(f[:7, 0]), 0.02, atol=1e-9)


class Gait(unittest.TestCase):
    def test_a_walk_ends_in_its_start_stance_at_the_goal(self):
        from geometry_fixture import scene_ik

        from locomani.backends.heft import gait as G

        lm = G.LegModel(scene_ik())
        q0 = np.zeros(29)
        q0[[0, 6]], q0[[3, 9]], q0[[4, 10]] = -0.28, 0.5, -0.23
        start = np.r_[1.0, 2.0, 0.76, yaw_quat(0.3), q0]
        info = {}
        fr = G.walk_frames(lm, start, (np.array([1.0, 2.0]), 0.3), (1.5, 2.1), 0.3, info=info)
        self.assertEqual(fr.shape[1], 36)
        np.testing.assert_allclose(fr[-1, :3], [1.5, 2.1, 0.76], atol=1e-4)
        np.testing.assert_allclose(fr[-1, 7:], q0, atol=1e-6)
        self.assertAlmostEqual(G.yaw_of(G.qmat(fr[-1, 3:7])), 0.3, places=4)
        self.assertLess(info["ik_err_cm"], 1.0)
        self.assertLessEqual(float(fr[:, 2].min()), 0.76 + 1e-6)  # never raised above the start stance
        np.testing.assert_allclose(fr[:, 7 + 12 :], np.tile(q0[12:], (len(fr), 1)), atol=1e-6)  # upper body kept


class GaitHeft(unittest.TestCase):
    """gait_heft: gait.py's plan with per-kind step shapes."""

    @classmethod
    def setUpClass(cls):
        from geometry_fixture import scene_ik

        from locomani.backends.heft import gait as G

        cls.lm = G.LegModel(scene_ik())
        cls.q0 = np.zeros(29)
        cls.q0[[0, 6]], cls.q0[[3, 9]], cls.q0[[4, 10]] = -0.28, 0.5, -0.23

    def start(self, yaw: float) -> np.ndarray:
        return np.r_[1.0, 2.0, 0.76, yaw_quat(yaw), self.q0]

    def test_hg_walks_shape_is_gait_py(self):
        from locomani.backends.heft import gait as G
        from locomani.backends.heft import gait_heft as GH

        args = (self.start(0.3), (np.array([1.0, 2.0]), 0.3), (1.5, 2.1), 0.3)
        np.testing.assert_allclose(GH.walk_frames(self.lm, *args, GH.HG_WALK), G.walk_frames(self.lm, *args), atol=1e-6)

    def test_heft_moves_end_in_the_start_stance_at_the_goal(self):
        from locomani.backends.heft import gait as G
        from locomani.backends.heft import gait_heft as GH

        for goal, yaw, kinds in (((1.0, 2.23), 0.0, {"side"}), ((1.5, 2.0), 0.0, {"walk"}),
                                 ((1.0, 2.0), math.pi / 2, {"turn"}), ((1.0, 2.5), 0.0, {"turn", "walk"})):  # fmt: skip
            info = {}
            fr = GH.walk_frames(self.lm, self.start(0.0), (np.array([1.0, 2.0]), 0.0), goal, yaw, GH.HEFT, info)
            np.testing.assert_allclose(fr[-1, :3], [*goal, 0.76], atol=1e-4)
            np.testing.assert_allclose(fr[-1, 7:], self.q0, atol=1e-6)
            self.assertAlmostEqual(G.wrap(G.yaw_of(G.qmat(fr[-1, 3:7])) - yaw), 0.0, places=4)
            self.assertEqual(set(info["kinds"]), kinds)
            self.assertLessEqual(float(fr[:, 2].min()), 0.76 + 1e-6)
            np.testing.assert_allclose(fr[:, 7 + 12 :], np.tile(self.q0[12:], (len(fr), 1)), atol=1e-6)
            if len(kinds) == 1:  # the swing ankle rises its shape's swing_h above the start stance
                z = [self.lm.feet(f[:3], f[3:7], f[7:])["left"][0][2] for f in fr]
                z0 = self.lm.feet(fr[0, :3], fr[0, 3:7], fr[0, 7:])["left"][0][2]
                self.assertAlmostEqual(max(z) - z0, getattr(GH.HEFT, kinds.pop()).swing_h, delta=0.01)

    def test_a_sidestep_leads_with_the_foot_on_its_side(self):
        from locomani.backends.heft import gait as G
        from locomani.backends.heft import gait_heft as GH

        # a 10 cm sidestep (a sidestep under both gaits' move_min) with a 0.5-degree heading correction the other way:
        # gait.py lets the correction pick the trailing foot
        for left, foot, trailing in ((0.10, "left", "right"), (-0.10, "right", "left")):
            dyaw = -math.copysign(math.radians(0.5), left)
            args = (self.start(0.0), (np.array([1.0, 2.0]), 0.0), (1.0, 2.0 + left), dyaw)
            heft, old = {}, {}
            GH.walk_frames(self.lm, *args, GH.HEFT, heft)
            G.walk_frames(self.lm, *args, info=old)
            self.assertEqual(heft["first"], foot)
            self.assertEqual(old["first"], trailing)


class ResetPose(unittest.TestCase):
    def test_the_reset_pose_is_lmls_stance_unless_heft_is_asked(self):
        from locomani.backends.heft.runtime import LML_STANCE, heft_meta

        names = POLICY_ORDER  # ScaleBFM's metadata order
        conf = {"joint_kp": {".*": 10.0}, "joint_kd": {".*": 1.0}, "default_joint_pos": {".*_knee_joint": 0.5}}
        meta = dict(joint_names=names, stiffness=[0.0] * 29, damping=[0.0] * 29, default_dof_pos=[0.0] * 29)
        out = heft_meta(meta, conf)
        self.assertEqual(out["default_dof_pos"][names.index("left_elbow_joint")], 1.2)
        np.testing.assert_allclose([out["default_dof_pos"][names.index(n)] for n in MUJOCO_JOINT_NAMES], LML_STANCE)
        self.assertEqual(heft_meta(meta, conf, "heft")["default_dof_pos"][names.index("left_knee_joint")], 0.5)
        self.assertEqual(out["stiffness"], [10.0] * 29)

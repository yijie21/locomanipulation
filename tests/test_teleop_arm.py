"""The VR 3-point channel: field gating in the commander, and the point geometry."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path[:0] = [str(Path(__file__).resolve().parent)]

from locomani.backends.sonic import agent_server
from locomani.backends.sonic.agent_server import AgentCommander, StreamMode
from locomani.backends.sonic.teleop_arm import NECK_OFFSET, VR_OFFSET, TeleopArm, vr_point


def _commander() -> AgentCommander:
    """An AgentCommander with no sockets: only `frame()`'s message building matters here."""
    c = object.__new__(AgentCommander)
    c._hand_open, c._hand_closed = agent_server.HAND_OPEN, agent_server.HAND_CLOSED
    c.heading_offset = 0.0
    c.sonic = unittest.mock.Mock()
    c.reset_pose()
    return c


class FrameFieldsTests(unittest.TestCase):
    """The deploy switches encoder mode on the PRESENCE of `vr_position`, so which
    frames carry it is a control decision, not a formatting one."""

    def sent(self, c, **kw):
        with patch.object(agent_server, "build_planner_message", return_value=b"") as bpm:
            with patch.object(AgentCommander, "manager_state"):
                c.frame(**kw)
        return bpm.call_args.kwargs

    def test_default_frame_carries_neither_vr_nor_upper_body(self):
        c = _commander()
        kw = self.sent(c)
        self.assertNotIn("vr_3pt_position", kw)
        self.assertNotIn("upper_body_position", kw)

    def test_plain_planner_and_vr3pt_stream_modes_do_not_send_vr_without_the_flag(self):
        c = _commander()
        for sm in (StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT):
            c.stream_mode = sm
            self.assertNotIn("vr_3pt_position", self.sent(c))

    def test_flag_sends_wrists_and_neck(self):
        c = _commander()
        c.vr_targets = True
        c.set_wrists(right_pos=[0.3, -0.2, -0.1])
        kw = self.sent(c)
        self.assertEqual(len(kw["vr_3pt_position"]), 9)
        self.assertEqual(len(kw["vr_3pt_orientation"]), 12)
        np.testing.assert_allclose(kw["vr_3pt_position"][3:6], [0.3, -0.2, -0.1], atol=1e-6)
        self.assertNotIn("upper_body_position", kw)

    def test_neck_target_travels_as_the_third_point(self):
        c = _commander()
        c.vr_targets = True
        c.set_neck([0.1, 0.0, 0.38], [0.99, 0.0, 0.15, 0.0])
        kw = self.sent(c)
        np.testing.assert_allclose(kw["vr_3pt_position"][6:9], [0.1, 0.0, 0.38], atol=1e-6)
        np.testing.assert_allclose(kw["vr_3pt_orientation"][8:12], [0.99, 0.0, 0.15, 0.0], atol=1e-6)

    def test_frozen_upper_body_wins_over_the_flag(self):
        c = _commander()
        c.vr_targets = True
        c.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
        c.set_upper_body(np.zeros(17))
        kw = self.sent(c)
        self.assertIn("upper_body_position", kw)
        self.assertNotIn("vr_3pt_position", kw)

    def test_reset_pose_clears_the_flag(self):
        c = _commander()
        c.vr_targets = True
        c.reset_pose()
        self.assertFalse(c.vr_targets)


class PointGeometryTests(unittest.TestCase):
    def test_vr_point_matches_the_vendor_formula(self):
        # vendor: point = R_root^T (p_link + R_link @ offset - p_root); rot = R_root^T R_link
        R_root = Rotation.from_euler("xyz", [0.05, 0.2, 1.0]).as_matrix()
        p_root = np.array([1.0, -2.0, 0.78])
        R_link = Rotation.from_euler("xyz", [0.3, -0.1, 0.4]).as_matrix()
        p_link = np.array([1.3, -2.2, 0.7])
        pt, rot = vr_point(p_link, R_link, p_root, R_root, VR_OFFSET["right"])
        expect = R_root.T @ (p_link + R_link @ VR_OFFSET["right"] - p_root)
        np.testing.assert_allclose(pt, expect, atol=1e-12)
        np.testing.assert_allclose(rot, R_root.T @ R_link, atol=1e-12)

    def test_measured_point_equals_target_point_at_entry(self):
        """Entering at the measured pose is the pico manager's per-switch calibration."""

        def wxyz(rot):
            q = rot.as_quat()
            return [q[3], q[0], q[1], q[2]]

        arm = object.__new__(TeleopArm)
        arm.side = "right"
        palm_rot = Rotation.from_euler("y", 0.4)
        state = dict(
            pelvis=dict(pos=[0.5, 0.1, 0.79], quat=wxyz(Rotation.from_euler("z", 0.7))),
            palms={"right": dict(pos=[0.8, -0.1, 0.6], quat=wxyz(palm_rot))},
        )
        arm.target = {"right": dict(pos=np.array([0.8, -0.1, 0.6]), rot=palm_rot.as_matrix())}
        m, _ = arm.measured_point("right", state)
        t, _ = arm.target_point("right", state)
        np.testing.assert_allclose(m, t, atol=1e-9)
        # and the offset really is applied in the LINK frame: the point sits
        # |offset| from the bare link position, whatever the pelvis does
        bare, _ = vr_point(
            state["palms"]["right"]["pos"],
            palm_rot.as_matrix(),
            state["pelvis"]["pos"],
            Rotation.from_euler("z", 0.7).as_matrix(),
            np.zeros(3),
        )
        self.assertAlmostEqual(float(np.linalg.norm(m - bare)), float(np.linalg.norm(VR_OFFSET["right"])), places=9)

    def test_neck_point_is_torso_plus_offset_pitched_about_pelvis_y(self):
        arm = object.__new__(TeleopArm)
        arm.torso_pf0 = (np.array([0.0, 0.0, 0.05]), np.eye(3))
        arm.lean = 0.0
        pos, _ = arm.neck_point()
        np.testing.assert_allclose(pos, [0.0, 0.0, 0.05] + NECK_OFFSET)
        arm.lean = 0.3
        pos, rot = arm.neck_point()
        self.assertGreater(pos[0], 0.1)  # a forward lean moves the neck forward ...
        self.assertLess(pos[2], 0.40)  # ... and down
        self.assertAlmostEqual(float(np.arcsin(-rot[2, 0])), 0.3, places=9)


if __name__ == "__main__":
    unittest.main()

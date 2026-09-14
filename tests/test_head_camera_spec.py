"""frames.head_camera_spec must not depend on MJCF attribute order.

MuJoCo's spec writer emits `name quat pos fovy` once build_sim_scene's
_aim_ego_camera has re-set the quaternion; the original regex assumed
`name pos quat fovy` and every freshly built scene failed with
"no head_camera" (2026-09-11), which render_wall_pilot.sh only reported as
"isaac exit 1" with no Isaac video.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomani.sim.isaac import frames  # noqa: E402

BODY = '<mujoco><worldbody><body name="torso_link">{cam}<geom size="0.1"/></body></worldbody></mujoco>'
POS, QUAT, FOVY = [0.06, 0.0, 0.45], [0.668835, 0.230083, -0.230336, -0.668332], 70.0


def _spec_of(cam_tag: str):
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(BODY.format(cam=cam_tag))
    return frames.head_camera_spec(f.name)


class HeadCameraSpecTests(unittest.TestCase):
    def test_pos_quat_fovy_order(self):
        pos, quat, fovy = _spec_of(
            '<camera name="head_camera" pos="0.06 0 0.45" quat="0.668835 0.230083 -0.230336 -0.668332" fovy="70"/>'
        )
        self.assertEqual((pos, quat, fovy), (POS, QUAT, FOVY))

    def test_quat_pos_fovy_order_as_written_by_mujoco(self):
        pos, quat, fovy = _spec_of(
            '<camera name="head_camera" quat="0.668835 0.230083 -0.230336 -0.668332" pos="0.06 0 0.45" fovy="70"/>'
        )
        self.assertEqual((pos, quat, fovy), (POS, QUAT, FOVY))

    def test_name_not_first_and_defaults(self):
        pos, quat, fovy = _spec_of('<camera fovy="70" name="head_camera"/>')
        self.assertEqual((pos, quat, fovy), ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 70.0))

    def test_other_camera_only_is_missing(self):
        with self.assertRaises(SystemExit):
            _spec_of('<camera name="left_wrist" pos="0 0 0" quat="1 0 0 0" fovy="60"/>')


def _wrist_spec_of(cam_tag: str, side: str):
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(BODY.format(cam=cam_tag))
    return frames.wrist_camera_spec(f.name, side)


class WristCameraSpecTests(unittest.TestCase):
    def test_reads_the_named_side(self):
        pos, quat, fovy = _wrist_spec_of(
            '<camera name="left_wrist" pos="0.055 0 0.085" quat="0.635149 0.310782 -0.310782 -0.635149" fovy="80"/>',
            "left",
        )
        self.assertEqual((pos, quat, fovy), ([0.055, 0.0, 0.085], [0.635149, 0.310782, -0.310782, -0.635149], 80.0))

    def test_left_and_right_are_independent(self):
        both = (
            '<camera name="left_wrist" pos="0.055 0 0.085" quat="1 0 0 0" fovy="80"/>'
            '<camera name="right_wrist" pos="-0.055 0 0.085" quat="1 0 0 0" fovy="81"/>'
        )
        self.assertEqual(_wrist_spec_of(both, "left")[0], [0.055, 0.0, 0.085])
        self.assertEqual(_wrist_spec_of(both, "right")[0], [-0.055, 0.0, 0.085])
        self.assertEqual(_wrist_spec_of(both, "right")[2], 81.0)

    def test_missing_side_is_a_clean_failure(self):
        with self.assertRaises(SystemExit):
            _wrist_spec_of('<camera name="head_camera" pos="0 0 0" quat="1 0 0 0" fovy="70"/>', "left")


if __name__ == "__main__":
    unittest.main()

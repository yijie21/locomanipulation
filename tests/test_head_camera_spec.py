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


if __name__ == "__main__":
    unittest.main()

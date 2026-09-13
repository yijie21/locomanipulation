"""isaac_render's inset layout and compositing math, independent of a live
Isaac Sim session (mount_inset_cameras itself needs isaacsim/pxr and is not
covered here).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomani.pipeline import sim_video as sv  # noqa: E402
from locomani.sim.isaac import isaac_render  # noqa: E402


class DatasetInsetLayoutTests(unittest.TestCase):
    def test_includes_all_three_dataset_cameras(self):
        # This renderer's own MAIN frame is often (via --cam-traj) a
        # third-person tracking shot, not the ego view, so the ego inset is
        # not redundant here even though it sounds like it should be.
        layout = isaac_render.dataset_inset_layout(sv.WIDTH, sv.HEIGHT)
        self.assertEqual(set(layout), {"head_camera", "left_wrist", "right_wrist"})

    def test_matches_sim_video_at_native_resolution(self):
        layout = isaac_render.dataset_inset_layout(sv.WIDTH, sv.HEIGHT)
        by_name = {name: (label, w, h, x, y) for name, label, w, h, x, y in sv.INSETS}
        self.assertEqual(layout, by_name)

    def test_scales_with_output_resolution(self):
        layout = isaac_render.dataset_inset_layout(sv.WIDTH // 2, sv.HEIGHT // 2)
        _label, w, h, x, y = layout["left_wrist"]
        self.assertEqual((w, h), (sv.WRIST_W // 2, sv.WRIST_H // 2))
        self.assertEqual((x, y), (4, 4))  # sim_video's left_wrist sits at (8, 8)


class EgoOnlyMeshPathsTests(unittest.TestCase):
    """Regression coverage for the black-dome bug: the head mesh must be
    hidden from the ego inset whenever the main camera (via --cam-traj)
    needs it visible, and left alone otherwise."""

    HIDDEN = ("torso_link/head_link_visual",)

    def test_hides_the_head_mesh_when_cam_traj_and_ego_inset_combine(self):
        paths = isaac_render.ego_only_mesh_paths(True, {"head_camera", "left_wrist"}, self.HIDDEN)
        self.assertEqual(paths, ["/visuals/torso_link/head_link_visual"])

    def test_no_toggle_without_cam_traj(self):
        # The main camera IS the ego view here; assemble_stage's own
        # stage-assembly-time hide already covers it and the inset alike.
        paths = isaac_render.ego_only_mesh_paths(False, {"head_camera"}, self.HIDDEN)
        self.assertEqual(paths, [])

    def test_no_toggle_without_an_ego_inset(self):
        # Only wrist insets requested: neither sits inside the head mesh.
        paths = isaac_render.ego_only_mesh_paths(True, {"left_wrist", "right_wrist"}, self.HIDDEN)
        self.assertEqual(paths, [])

    def test_no_toggle_with_no_insets_at_all(self):
        paths = isaac_render.ego_only_mesh_paths(True, set(), self.HIDDEN)
        self.assertEqual(paths, [])


class PasteInsetTests(unittest.TestCase):
    def test_paste_writes_the_sub_image_at_the_right_offset_and_nowhere_else(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patch = np.full((20, 30, 4), 255, dtype=np.uint8)  # (h, w, rgba), opaque white
        isaac_render._paste_inset(frame, patch, "L wrist", 30, 20, 10, 5)
        self.assertTrue((frame[10, 20] == [255, 255, 255]).all())  # inside the pasted region
        self.assertTrue((frame[50, 50] == [0, 0, 0]).all())  # untouched elsewhere


if __name__ == "__main__":
    unittest.main()

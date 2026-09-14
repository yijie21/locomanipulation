import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.perception import CrateEstimate
from locomani.tasks.crate.visible_target import VisibleTarget


def test_cached_visible_pose_stays_in_world_frame_and_does_not_refresh_age():
    cache = VisibleTarget()
    first = dict(pelvis=dict(pos=[0, 0, 0.8], quat=[1, 0, 0, 0]))
    seen = CrateEstimate(np.array([0.5, 0, -0.4]), np.eye(3), 10.0, None, {"ego_view": [[20, 30]]})
    cache.update(seen, first)
    base = Rotation.from_euler("xyz", [0.05, 0.3, 0.2]).as_matrix()
    moved = dict(pelvis=dict(pos=[0.1, 0.02, 0.77], quat=Rotation.from_matrix(base).as_quat()[[3, 0, 1, 2]]))
    cached = cache.at(moved)
    np.testing.assert_allclose(base @ cached.position + moved["pelvis"]["pos"], [0.5, 0, 0.4], atol=1e-12)
    np.testing.assert_allclose(base @ cached.rotation, np.eye(3), atol=1e-12)
    assert cached.timestamp == 10.0 and cached.detections == seen.detections
    cache.update(None, moved)
    assert cache.at(moved) is None

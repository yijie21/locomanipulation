"""visual.lift_pitch_follow_gain: the loaded phases rotate the commanded palm attitudes against the box's pitch."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.coordinated_reach import CoordinatedReach
from locomani.tasks.crate.visual_config import CrateVisualConfig


def _stub(gain, cap=25.0):
    cfg = SimpleNamespace(visual=CrateVisualConfig(lift_pitch_follow_gain=gain, lift_pitch_follow_cap_deg=cap))
    return SimpleNamespace(
        config=cfg,
        attitudes=np.array([np.eye(3), np.eye(3)]),
        closed_rotation=np.eye(3),
        pitch_follow=0.0,
        diagnostics={},
    )


def test_off_returns_reach_attitudes():
    s = _stub(0.0)
    out = CoordinatedReach.follow_attitudes(s, Rotation.from_euler("x", 15, degrees=True).as_matrix(), 0.05)
    assert np.allclose(out, s.attitudes)


def test_rotates_palms_against_box_pitch_and_caps():
    s = _stub(1.5)
    box = Rotation.from_euler("x", 10, degrees=True).as_matrix()  # box pitched +10 deg about its grasp axis
    for _ in range(200):  # let the 0.2 s low-pass settle
        out = CoordinatedReach.follow_attitudes(s, box, 0.05)
    palm = Rotation.from_matrix(out[0]).as_rotvec()
    assert np.isclose(np.degrees(palm @ box[:, 0]), -15.0, atol=0.2)  # -gain x pitch, about the box axis
    assert np.isclose(s.diagnostics["pitch_follow_command_deg"], -15.0, atol=0.2)
    s = _stub(1.5, cap=8.0)
    for _ in range(200):
        out = CoordinatedReach.follow_attitudes(s, box, 0.05)
    assert np.isclose(np.degrees(Rotation.from_matrix(out[1]).as_rotvec() @ box[:, 0]), -8.0, atol=0.2)

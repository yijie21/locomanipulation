"""visual.close_trigger=contact and the ghost-hands trace hook."""

from types import SimpleNamespace

import numpy as np

from locomani.tasks.crate.coordinated_reach import CoordinatedReach
from locomani.tasks.crate.ghost_trace import GhostTargetTrace, ghost_poses, load_ghost_trace
from locomani.tasks.crate.grasp_region import RimRegion
from locomani.tasks.crate.visual_config import CrateVisualConfig


def _stub(touching=()):
    cfg = SimpleNamespace(visual=CrateVisualConfig(close_trigger="contact"))
    return SimpleNamespace(
        config=cfg, region=RimRegion(), target=SimpleNamespace(rim_above_grasp=0.01), contact_sides=tuple(touching)
    )


CENTRES = np.array([[0.2, 0.0, 0.69], [-0.2, 0.0, 0.69]])
R = np.eye(3)


def test_contact_closes_on_jaws_around_the_wall_without_attitude():
    s = _stub()
    tools = CENTRES + np.array([[0.04, 0.05, 0.0], [-0.03, -0.02, -0.01]])  # 3-4 cm off across, slid along, at/below
    ready, err = CoordinatedReach.contact_ready(s, CENTRES, R, tools)
    assert ready and err.shape == (2, 3)


def test_contact_refuses_wall_outside_the_jaw_or_pads_above_the_rim():
    s = _stub()
    assert not CoordinatedReach.contact_ready(s, CENTRES, R, CENTRES + np.array([[0.06, 0, 0], [0, 0, 0]]))[0]
    high = CENTRES + np.array([[0, 0, 0.02], [0, 0, 0.0]])  # left grip 1 cm above the rim, not touching
    assert not CoordinatedReach.contact_ready(s, CENTRES, R, high)[0]
    assert CoordinatedReach.contact_ready(_stub(("left", "right")), CENTRES, R, high)[0]  # touching: close where it is


def test_gate_mode_still_uses_the_rim_region():
    s = _stub()
    s.config.visual = CrateVisualConfig(close_trigger="gate")
    tools = CENTRES + np.array([[0.04, 0, 0], [0, 0, 0]])
    rots = np.array([np.eye(3), np.eye(3)])
    assert not CoordinatedReach.capture_ready(s, CENTRES, R, tools, rots)[0]


def test_ghost_trace_round_trip(tmp_path):
    g = GhostTargetTrace(str(tmp_path), (0.1515, 0.0, 0.0))
    g.start("ep")
    g.row(10, {"left": [0.3, 0.2, 0.7], "right": [0.3, -0.2, 0.7]}, {"left": np.eye(3), "right": np.eye(3)})
    assert g.stop() == str(tmp_path / "ep_ghost.npz")
    table = load_ghost_trace(str(tmp_path / "ep_ghost.npz"))
    pos, quat = ghost_poses(table, 11)  # a seq apart is the same intent
    assert np.allclose(pos[0], [0.3 - 0.1515, 0.2, 0.7]) and np.allclose(quat[0], [1, 0, 0, 0])

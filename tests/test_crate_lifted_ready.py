"""visual.ready_pose lifted: the hands rise in place instead of reaching 0.42 m ahead (2026-09-24).

Three offline warehouse pilots put a fingertip on the box's near wall during arm preparation at stances under
0.51 m: the forward ready pose has the tool points 0.38 m ahead at pelvis height. The lifted pose folds the
forearms up, which the solvers' 5 deg elbow floor forbids, so it is gated on visual.elbow_min_deg.
"""

from __future__ import annotations

import numpy as np
import pytest
from geometry_fixture import scene_ik

from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.expert import lifted_ready
from locomani.tasks.crate.visual_config import CrateVisualConfig

TIP = np.array([0.194, 0.0, 0.0])  # dex1 fingertips, palm frame
HANGING = np.zeros(29)
HANGING[15:22] = [
    -0.04,
    0.35,
    -0.78,
    1.01,
    0.08,
    0.11,
    0.07,
]  # planner walking arms after a guard brake (m003b round 03)
HANGING[22:29] = [-0.04, -0.35, 0.78, 1.01, -0.08, 0.11, -0.07]


def fingertips(ik, joints):
    return np.array([p + r @ TIP for p, r in (ik.fk(joints, side) for side in ("left", "right"))])


def test_lifted_ready_needs_the_elbow_floor_lowered():
    ik = scene_ik()
    cfg = CrateVisualConfig(ready_pose="lifted", elbow_min_deg=-60.0)
    with pytest.raises(SystemExit, match="missed by"):
        lifted_ready(BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all"), cfg, HANGING)  # default 5 deg floor
    ready = lifted_ready(BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all", elbow_min_deg=-60), cfg, HANGING)
    assert np.max(ready.errors) < 0.01
    assert ready.joints[18] < 0 and ready.joints[25] < 0  # forearms folded up
    np.testing.assert_array_equal(ready.joints[12:15], 0)  # waist locked


def test_lifted_ready_keeps_the_fingertips_near_the_body_through_the_blend():
    ik = scene_ik()
    cfg = CrateVisualConfig(ready_pose="lifted", elbow_min_deg=-60.0)
    ready = lifted_ready(BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all", elbow_min_deg=-60), cfg, HANGING)
    forward = max(
        fingertips(ik, HANGING + (3 * u * u - 2 * u**3) * (ready.joints - HANGING))[:, 0].max()
        for u in np.linspace(0, 1, 41)
    )
    assert forward < 0.22  # the forward pose's fingertips sit at 0.42 m; the warehouse near wall is at 0.24-0.35 m


def test_ready_pose_config_is_validated_and_round_trips():
    with pytest.raises(ValueError, match="elbow_min_deg below 0"):
        CrateVisualConfig(ready_pose="lifted")
    with pytest.raises(ValueError, match="forward or lifted"):
        CrateVisualConfig(ready_pose="high")
    with pytest.raises(ValueError, match="three finite numbers"):
        CrateVisualConfig(ready_tool=(0.1, 0.2))
    config = CrateTaskConfig.from_dict(
        CrateTaskConfig(
            visual=CrateVisualConfig(ready_pose="lifted", elbow_min_deg=-60.0, ready_tool=(0.18, 0.16, 0.05))
        ).to_dict()
    )
    assert config.visual.ready_pose == "lifted" and tuple(config.visual.ready_tool) == (0.18, 0.16, 0.05)
    assert CrateVisualConfig().elbow_min_deg == 5.0  # the historical floor is the default

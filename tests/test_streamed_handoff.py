from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from geometry_fixture import scene_ik

from locomani.backends.sonic import streamed_arm


def test_streamed_handoff_never_emits_an_unconstrained_planner_frame(monkeypatch):
    """Exercise the actual handoff with a clock and capture its planner packets."""
    clock = [0.0]
    monkeypatch.setattr(
        streamed_arm,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=lambda dt: clock.__setitem__(0, clock[0] + dt)),
    )
    commander = Mock(left_hand=np.zeros(7), right_hand=np.zeros(7), upper_body=np.ones(17), vr_targets=False)
    unprotected = []
    commander.frame.side_effect = lambda **kw: unprotected.append(
        commander.upper_body is None and not commander.vr_targets
    )
    state = dict(joints=np.zeros(29), pelvis=dict(quat=[1, 0, 0, 0]))
    primitive = SimpleNamespace(cmd=commander, obs=lambda: state, _facing=lambda _: (1, 0, 0))
    arm = streamed_arm.StreamedArm(primitive, "right", ik=scene_ik(), left_guard=False)
    arm.start(hold_planner_pose=True)
    assert not any(unprotected), f"{sum(unprotected)} planner frames released both arm references before POSE takeover"

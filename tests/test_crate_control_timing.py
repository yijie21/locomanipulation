from types import SimpleNamespace

import numpy as np

from locomani.tasks.crate import carry_phases_lift
from locomani.tasks.crate.carry_whole_body import WholeBodyArms


def test_closing_duration_does_not_double_when_control_ticks_take_40ms(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(carry_phases_lift, "time", SimpleNamespace(time=lambda: clock[0]))
    state = dict(fallen=False, objects={"blue_crate": {"touching": ["left", "right"]}})
    closures = []
    arms = SimpleNamespace(
        arm=SimpleNamespace(q=np.zeros(29)),
        set_hands=lambda u: closures.append(u),
        push=lambda q, **kw: clock.__setitem__(0, clock[0] + 0.04),
    )
    ctx = SimpleNamespace(p=SimpleNamespace(obs=lambda: state), arms=arms, m={})
    assert carry_phases_lift._phase_close(ctx) is None
    assert closures[-1] == 1
    assert 2.2 <= clock[0] < 2.3


def test_recording_reuses_control_observation_instead_of_waiting_for_a_second_state():
    def forbidden():
        raise AssertionError("recording consumed another control tick")

    state = dict(joints=np.zeros(29), pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    recorded = []
    arms = object.__new__(WholeBodyArms)
    arms.motion = None
    arms.p = SimpleNamespace(obs=forbidden)
    arms.arm = SimpleNamespace(root_quat=np.array([1, 0, 0, 0]), _push=lambda *a, **kw: None)
    arms.motion_recording = SimpleNamespace(sample=lambda *a, **kw: recorded.append(a[3]))
    arms.push(np.zeros(29), observed=state)
    assert recorded == [state]

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.grasp_region import RimRegion


def test_valid_off_centre_rim_grasp_does_not_require_returning_to_the_slot_centre():
    region = RimRegion()
    centres = np.array([[0.1933, 0, 0.060], [-0.1933, 0, 0.060]])
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    centres = centres @ rotation.T
    wrists = [Rotation.from_euler("y", 55, degrees=True).as_matrix()] * 2
    tools = centres + [[0.070, 0, 0.02], [-0.070, 0, 0.02]]
    ready, _ = region.contains(centres, rotation, tools, wrists)
    assert ready
    target = region.nearest(centres, rotation[:, 1], tools)
    np.testing.assert_allclose(target, centres + [[0.070, 0, 0], [-0.070, 0, 0]])
    np.testing.assert_allclose(target.mean(axis=0), centres.mean(axis=0), atol=1e-12)


def test_end_wall_wrong_jaw_direction_and_high_fingers_cannot_authorize_closure():
    region = RimRegion()
    centres = np.array([[0, 0.1933, 0.060], [0, -0.1933, 0.060]])
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    wrists = [Rotation.from_euler("y", 60, degrees=True).as_matrix()] * 2
    for displacement in ([0.105, 0, 0], [0, 0.020, 0], [0, 0, 0.045], [0, 0, -0.015]):
        assert not region.contains(centres, rotation, centres + displacement, wrists)[0]
    assert not region.contains(centres, rotation, centres, [np.eye(3)] * 2)[0]


def test_carry_consumer_closes_at_valid_rim_offset_without_recentering(monkeypatch):
    from types import SimpleNamespace

    from locomani.tasks.crate import bimanual_carry as carry
    from locomani.tasks.crate import carry_phases
    from locomani.tasks.crate.carry_config import CrateCarryConfig

    clock = [0.0]
    monkeypatch.setattr(carry_phases, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(carry, "SERVO_TIMEOUT", 0.4)
    centres = dict(left=np.array([0.4, 0.1933, 0.445]), right=np.array([0.4, -0.1933, 0.445]))
    crate_rotation = Rotation.from_euler("z", 90, degrees=True)
    wrist = Rotation.from_euler("y", 60, degrees=True).as_matrix()
    state = dict(
        fallen=False,
        joints=np.zeros(29),
        wall=0.0,
        objects={"blue_crate": {"quat": crate_rotation.as_quat()[[3, 0, 1, 2]]}},
    )

    def observe():
        clock[0] += 0.02
        state["wall"] = clock[0]
        return state

    monkeypatch.setattr(
        carry, "crate_frame", lambda *a: (np.array([0.4, 0, 0.385]), centres, np.array([1, 0]), np.array([0, 1]), 0)
    )
    arms = SimpleNamespace(
        arm=SimpleNamespace(q=np.zeros(29)),
        start=lambda **kw: None,
        stop=lambda: None,
        grip_point=lambda s, side: (centres[side] + [0.07 if side == "left" else -0.07, 0, 0.02], wrist),
        solve=lambda *a, **kw: np.zeros(29),
        push=lambda *a, **kw: None,
    )
    ctx = SimpleNamespace(
        p=SimpleNamespace(obs=observe),
        dump_dir="",
        log=lambda *a: None,
        rnd=0,
        arms=arms,
        centre=np.array([0.4, 0, 0.385]),
        bars=centres,
        n=np.array([1, 0]),
        y_hat=np.array([0, 1]),
        psi=0,
        ramp=SimpleNamespace(q=np.zeros(17)),
        m={},
        t_video=0,
        config=CrateCarryConfig(reach_control="whole_body"),
        grasp_posture=None,
    )
    assert carry_phases._phase_grasp(ctx) is None
    assert ctx.m["close_trigger"] == "rim_region"
    assert clock[0] < 0.25


def test_two_individually_valid_pinches_cannot_leave_the_grasp_axis_far_from_the_centre():
    region = RimRegion()
    centres = np.array([[0, 0.1933, 0.060], [0, -0.1933, 0.060]])
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    wrists = [Rotation.from_euler("y", 60, degrees=True).as_matrix()] * 2
    ready, _ = region.contains(centres, rotation, centres + [0.060, 0, 0.02], wrists)
    assert not ready, "a shared 6 cm offset leaves the basket centre behind the grasp axis"

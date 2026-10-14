"""sim/perturb.py: the perturbation spec both controller sims share (TODO 25.5)."""

from pathlib import Path

import numpy as np

from locomani.sim import perturb


def test_delay_returns_the_command_from_delay_steps_ago():
    p = perturb.Perturbation({"delay_s": 0.02}, 3, 0.005)
    assert p.delay_steps == 4
    out = [p.delayed(k) for k in range(10)]
    assert out[:5] == [0, 0, 0, 0, 0]  # the oldest while the queue fills
    assert out[5:] == [1, 2, 3, 4, 5]


def test_off_by_default():
    p = perturb.Perturbation(None, 29, 0.005)
    assert not p.active()
    assert p.delayed("cmd") == "cmd"
    assert np.all(p.gain == 1.0)
    assert np.array_equal(p.noisy("dq", np.ones(3)), np.ones(3))


def test_gain_draw_is_seeded_and_bounded():
    a = perturb.Perturbation({"gain_scale": [0.8, 1.2], "seed": 5}, 29, 0.005)
    b = perturb.Perturbation({"gain_scale": [0.8, 1.2], "seed": 5}, 29, 0.005)
    assert np.array_equal(a.gain, b.gain)
    assert a.gain.min() >= 0.8 and a.gain.max() <= 1.2 and np.ptp(a.gain) > 0


def test_noise_is_bounded():
    p = perturb.Perturbation({"noise": {"gyro": 0.1}, "seed": 1}, 29, 0.005)
    v = p.noisy("gyro", np.zeros(1000))
    assert np.abs(v).max() <= 0.1 and np.abs(v).mean() > 0.03


def test_push_runs_out():
    class D:
        xfrc_applied = np.zeros((3, 6))

    d = D()
    perturb.request_push([50.0, 0.0, 0.0], 0.02, 0.005)
    seen = []
    for _ in range(6):
        perturb.apply_push(d, 1)
        seen.append(d.xfrc_applied[1, 0])
    assert seen == [50.0, 50.0, 50.0, 0.0, 0.0, 0.0]


def test_periodic_push_fires_every_period():
    p = perturb.Perturbation({"push_every_s": 0.05, "push_n": [60.0, 60.0], "push_s": 0.01}, 29, 0.005)
    perturb.PUSH.update(force=None, steps=0)
    fired = []
    for _ in range(25):
        p.tick()
        fired.append(perturb.PUSH["steps"] > 0 and perturb.PUSH["force"] is not None)
        if perturb.PUSH["steps"]:
            perturb.PUSH["steps"] = 0  # consume it (apply_push would)
    assert [i for i, f in enumerate(fired) if f] == [9, 19]


def test_leg_odometry_holds_a_standing_robot():
    import mujoco

    from locomani.backends.scalebfm.odometry import LegOdometry
    from locomani.paths import REPO

    xml = Path(REPO) / "outputs" / "scene" / "walk" / "scene_locomani.xml"
    if not xml.exists():
        import pytest

        pytest.skip("walk scene not built")
    odo = LegOdometry(str(xml))
    q = odo.m.qpos0.copy()
    data = mujoco.MjData(odo.m)
    data.qpos[:] = q
    mujoco.mj_kinematics(odo.m, data)
    joints, quat = q[odo.qadr], q[3:7]
    odo.reset(q[:3], joints, quat)
    for _ in range(50):
        est = odo.update(joints, quat)
    assert np.allclose(est, q[:3], atol=1e-9)

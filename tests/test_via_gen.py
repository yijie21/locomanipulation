"""selfmodel_via_20261120/via_gen.py: the random paths gen_reach.py's REACH_VIA sends as one reach command, and the cuRobo
paths REACH_VIA_REPLAY places onto the robot. No simulator."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "selfmodel_via_20261120"))
import via_gen as VG  # noqa: E402


def rot_z(a):
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def draw(seed):
    rng = np.random.default_rng(seed)
    sd = 1.0 if rng.random() < 0.5 else -1.0
    p, Rh = np.array([0.4, -0.2, 0.79]), rot_z(0.5)
    a = VG.from_h(np.array([0.12, -sd * 0.24, 0.72]), p, Rh)[0]
    b = VG.from_h(np.array([rng.uniform(0.34, 0.72), -sd * rng.uniform(0.06, 0.30), rng.uniform(0.10, 1.12)]), p, Rh)[0]
    Ra, Rb = VG.expm(rng.normal(size=3) * 0.3), VG.expm(rng.normal(size=3) * 0.8)
    sh = VG.from_h(np.array([0.0, -sd * 0.10, 1.08]), p, Rh)[0]
    return a, b, Ra, Rb, p, Rh, sd, VG.random_path(rng, a, b, Ra, Rb, p, Rh, sd, sh, None)


def test_rotation_vector_round_trip_up_to_half_a_turn():
    rng = np.random.default_rng(0)
    for angle in (1e-7, 0.3, 2.0, math.pi - 1e-7, math.pi):
        k = rng.normal(size=3)
        v = angle * k / np.linalg.norm(k)
        assert np.allclose(VG.expm(VG.logm(VG.expm(v))), VG.expm(v), atol=1e-6)


def test_heading_frame_round_trip_keeps_the_height():
    p, Rh = np.array([1.0, 2.0, 0.8]), rot_z(-1.1)
    x = np.array([[0.3, -0.2, 0.9], [1.5, 2.5, 0.1]])
    assert np.allclose(VG.from_h(VG.to_h(x, p, Rh), p, Rh), x)
    assert np.allclose(VG.to_h(x, p, Rh)[:, 2], x[:, 2])
    ahead = VG.to_h(p + Rh[:, 0] * 0.5, p, Rh)[0]
    assert np.allclose(ahead[:2], [0.5, 0.0])


def test_a_seed_gives_one_path_within_the_limits():
    bends = []
    for seed in range(300):
        a, b, Ra, Rb, p, Rh, sd, v = draw(seed)
        again = draw(seed)[-1]
        assert (
            np.array_equal(v["p"], again["p"]) and np.array_equal(v["R"], again["R"]) and v["speed"] == again["speed"]
        )
        k = len(v["p"])
        assert k <= VG.VIA_MAX and v["R"].shape == (k, 3, 3)
        assert all(np.allclose(r @ r.T, np.eye(3), atol=1e-9) and np.linalg.det(r) > 0 for r in v["R"])
        assert VG.SPEED_BOX[0] <= v["speed"] <= VG.SPEED_BOX[1]
        if k:
            h = VG.to_h(v["p"], p, Rh)
            assert (h >= VG.BOX[0] - VG.TOL).all() and (h <= VG.BOX[1] + VG.TOL).all()
        assert (k == 0) or v["gen"][0] > 0 or v["gen"][1] > 0  # via points come from a bend or a lead-in
        bends.append(int(v["gen"][0]))
    share = np.bincount(bends, minlength=4) / len(bends)
    assert np.abs(share - VG.BENDS_P).max() < 0.1


def test_the_simplified_curve_stays_within_the_tolerance():
    K = np.array([[0.1, -0.2, 0.7], [0.2, -0.45, 0.9], [0.35, -0.1, 1.1], [0.5, -0.2, 0.9]])
    X, at = VG.catmull(K)
    assert np.allclose(X[at], K)
    assert np.linalg.norm(np.diff(X, axis=0), axis=1).max() < 2 * VG.DS
    idx = VG.rdp(X, VG.TOL)
    assert idx[0] == 0 and idx[-1] == len(X) - 1 and len(idx) < len(X) / 3
    assert VG.poly_dev(X, idx) <= VG.TOL + 1e-12


def test_a_path_through_the_trunk_or_the_floor_is_not_run():
    sh = np.array([0.0, -0.1, 1.08])
    line = lambda a, b: np.linspace(a, b, 60)  # noqa: E731
    arc = lambda X: np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]  # noqa: E731
    ok = line([0.12, -0.24, 0.72], [0.5, -0.2, 0.9])
    trunk = line([0.12, -0.24, 0.72], [-0.12, 0.2, 0.9])
    floor = np.vstack([line([0.12, -0.24, 0.72], [0.4, -0.2, 0.02]), line([0.4, -0.2, 0.02], [0.5, -0.2, 0.3])])
    assert VG.REASONS[VG.blocked(ok, arc(ok), 1.0, sh, None)] == "ok"
    assert VG.REASONS[VG.blocked(trunk, arc(trunk), 1.0, sh, None)] == "trunk"
    assert VG.REASONS[VG.blocked(floor, arc(floor), 1.0, sh, None)] == "floor"


def test_a_planned_path_is_read_in_its_own_heading_frame(tmp_path):
    P, yaw = np.array([1.2, -0.7, 0.79]), 0.6
    Rw = rot_z(yaw)
    via_h = np.array([[0.2, -0.3, 0.8], [0.35, -0.35, 0.95]])
    goal_h, start_h = np.array([0.45, -0.1, 0.9]), np.array([0.11, -0.22, 0.73])
    turn = VG.expm([0.0, 0.4, 0.0])
    (tmp_path / "v").mkdir()
    (tmp_path / "p").mkdir()
    np.savez(
        tmp_path / "v" / "via_T_L0_e0.npz", p=VG.from_h(via_h, P, Rw), R=np.array([Rw @ turn, Rw]), goal=np.zeros(3)
    )
    np.savez(tmp_path / "p" / "prep_T_L0_e0.npz", pelvis_p=P, pelvis_R=Rw @ VG.expm([0.05, 0.0, 0.0]),
             jaw_p=VG.from_h(start_h, P, Rw)[0], wrist_R=Rw, goal_jaw=VG.from_h(goal_h, P, Rw)[0], goal_R=Rw @ turn)  # fmt: skip
    (r,) = VG.load_routes(f"{tmp_path / 'v'}:{tmp_path / 'p'}")
    assert r["name"] == "T_L0_e0" and r["speed"] is None and abs(r["pelvis_z"] - 0.79) < 1e-12
    assert np.allclose(r["p"], via_h, atol=1e-3) and np.allclose(r["goal"], goal_h, atol=1e-3)
    assert np.allclose(r["start"], start_h, atol=1e-3)
    assert np.allclose(r["R"][0], turn, atol=2e-3) and np.allclose(r["goal_R"], turn, atol=2e-3)
    f = VG.shape(np.vstack([r["start"], r["p"], r["goal"]]), [r["start_R"], *r["R"], r["goal_R"]])
    assert f[0] == 2 and f[2] > 1.0 and len(f) == len(VG.FEATURES)

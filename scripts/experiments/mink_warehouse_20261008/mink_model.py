"""mink warehouse study: the IK model and the hand collision proxies.

mink's CollisionAvoidanceLimit measures distances with mj_geomDistance, which on MuJoCo 3.11 returned 0.0 for
separated mesh pairs and the wrong sign for one (curobo_warehouse_20261008/cases.py, Surfaces). The IK model is
therefore the scene with, on each hand body (wrist_pitch, wrist_yaw, both Dex1 fingers), one box per collision
geom: a mesh's bounding box in its own geom frame (it contains the convex hull MuJoCo collides), a box as is, a
cylinder's or capsule's bounding box. The proxies are named `<body>_geom<original id>_proxy` and are the only hand
geoms in the collision pairs. They only make the hand
more conservative; `check_proxies` measures by how much against the sampled hull distance.

The expert's joint limits are written into the model (waist pitch <= carry.waist_pitch_max, elbow >=
visual.elbow_min_deg, shoulder yaw and wrist roll within +-50 deg, as BimanualIK), so ConfigurationLimit
enforces them.
"""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.primitives.g1_joints import ARM_MJ

SIDES = ("left", "right")
HAND_BODIES = ("wrist_pitch_link", "wrist_yaw_link", "dex1_finger_1_link", "dex1_finger_2_link")
OBSTACLES = ("nor_cal_floor", "nor_cal_xp", "nor_cal_xm", "nor_cal_yp", "nor_cal_ym", "platform_col")


def _quat_to_matrix(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def local_box(m, g):
    """(centre, half extents) in the geom frame of a box containing collision geom g."""
    kind, size = int(m.geom_type[g]), m.geom_size[g]
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = m.geom_dataid[g]
        verts = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        return (lo + hi) / 2, (hi - lo) / 2
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.zeros(3), size.copy()
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.zeros(3), np.array([size[0], size[0], size[1]])
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return np.zeros(3), np.array([size[0], size[0], size[1] + size[0]])
    raise ValueError(f"geom {g}: no proxy for type {kind}")


def build(scene_xml, waist_pitch_max, elbow_min_deg):
    """(model, proxy geom ids per side, obstacle geom ids, 17 active dof addresses, the hinge joint ids)."""
    base = mujoco.MjModel.from_xml_path(scene_xml)
    spec = mujoco.MjSpec.from_file(scene_xml)
    proxies = {s: [] for s in SIDES}
    for side in SIDES:
        for part in HAND_BODIES:
            name = f"{side}_{part}"
            b = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, name)
            body = spec.body(name)
            for g in range(base.ngeom):
                if base.geom_bodyid[g] != b or not (base.geom_contype[g] or base.geom_conaffinity[g]):
                    continue
                center, half = local_box(base, g)
                R = _quat_to_matrix(base.geom_quat[g])
                proxy = f"{name}_geom{g}_proxy"
                body.add_geom(
                    name=proxy,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=half,
                    pos=base.geom_pos[g] + R @ center,
                    quat=base.geom_quat[g],
                    contype=1,
                    conaffinity=1,
                    group=4,
                    rgba=[1, 0, 1, 0.3],
                )
                proxies[side].append(proxy)
    model = spec.compile()

    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(hinges) == 29, len(hinges)
    model.jnt_range[hinges[14], 1] = min(model.jnt_range[hinges[14], 1], waist_pitch_max)
    for side in SIDES:
        arm = ARM_MJ[side]
        j = hinges[arm[3]]
        model.jnt_range[j, 0] = max(model.jnt_range[j, 0], np.radians(elbow_min_deg))
        for index in (arm[2], arm[4]):
            j = hinges[index]
            model.jnt_range[j] = [
                max(model.jnt_range[j, 0], -np.radians(50)),
                min(model.jnt_range[j, 1], np.radians(50)),
            ]
    active = [12, 13, 14, *ARM_MJ["left"], *ARM_MJ["right"]]
    active_dofs = [int(model.jnt_dofadr[hinges[i]]) for i in active]
    # Only the planned joints are limited: a frozen leg or jaw measured a hair outside its range would otherwise
    # make ConfigurationLimit (v >= (lo - q) / dt) contradict the freezing constraint (v = 0).
    for j in range(model.njnt):
        if model.jnt_dofadr[j] not in set(active_dofs):
            model.jnt_limited[j] = 0
    ids = lambda names: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]  # noqa: E731
    return model, {s: ids(proxies[s]) for s in SIDES}, ids(OBSTACLES), active_dofs, hinges


_JITTER = np.random.default_rng(0).normal(0.0, 2e-5, (6, 3))


def robust_geom_distance(m, d, g1, g2, distmax, fromto):
    """mj_geomDistance with a retry when it returns exactly 0.0.

    On MuJoCo 3.11 it returned exactly 0.0 for box-box pairs 4 cm apart (39 of 1680 hand-proxy x obstacle pairs
    on recorded poses, 2026-10-08); shifting geom 1 by ~20 µm returns the true value. An exact zero is retried
    with 6 small shifts (median of the non-zero results, fromto from the last one); zeros on every retry are
    kept as contact.
    """
    dist = _MUJOCO.mj_geomDistance(m, d, g1, g2, distmax, fromto)
    if dist != 0.0:
        return dist
    pos = d.geom_xpos[g1].copy()
    values = []
    for shift in _JITTER:
        d.geom_xpos[g1] = pos + shift
        value = _MUJOCO.mj_geomDistance(m, d, g1, g2, distmax, fromto)
        if value != 0.0:
            values.append(value)
    d.geom_xpos[g1] = pos
    return float(np.median(values)) if values else 0.0


class _MujocoShim:
    """The mujoco module, with mj_geomDistance replaced by robust_geom_distance."""

    def __getattr__(self, name):
        return getattr(_MUJOCO, name)

    @staticmethod
    def mj_geomDistance(m, d, g1, g2, distmax, fromto):
        return robust_geom_distance(m, d, g1, g2, distmax, fromto)


_MUJOCO = mujoco


def install_robust_distance():
    """Make mink's CollisionAvoidanceLimit use robust_geom_distance (only that module sees the shim)."""
    import mink.limits.collision_avoidance_limit as cal

    cal.mujoco = _MujocoShim()


def displacement_collision_limit():
    """mink.CollisionAvoidanceLimit with its bound in the QP's units.

    mink 1.3.0 solves for the displacement dq (v = dq / dt; VelocityLimit bounds dq by dt * v_max) but
    CollisionAvoidanceLimit bounds the normal displacement by gain * (d - d_min) / dt, a velocity. At dt = 0.03 s
    that admits 33x the intended approach: one recorded step went from 12.5 mm to -15.4 mm with the row satisfied
    (2026-10-08, visual_crate_s7000_07 seq 15011). This subclass multiplies the bound by dt, i.e. dd >= -gain *
    (d - d_min) per step, which is the documented barrier.
    """
    import mink
    from mink.limits.limit import Constraint

    class DisplacementCollisionAvoidanceLimit(mink.CollisionAvoidanceLimit):
        def compute_qp_inequalities(self, configuration, dt):
            c = super().compute_qp_inequalities(configuration, dt)
            h = np.array(c.h, dtype=float)
            finite = np.isfinite(h)
            h[finite] *= dt
            return Constraint(G=c.G, h=h)

    return DisplacementCollisionAvoidanceLimit

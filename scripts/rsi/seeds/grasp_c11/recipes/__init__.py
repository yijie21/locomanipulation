"""Flows for one kind of object (rules.md 30), chosen by features, never by name. Each recipe module carries a
`SIGNATURE` (the features it is for), `match(obj) -> score` (0: not for this object) and the flow itself; `choose`
returns the best-scoring recipe or None, and the caller (pick.side_grasp) falls back to the generic pick.

`features` describes a free object as a side grasp along `ux` at jaw height `jaw_z` would meet it, from the scene file
and the loaded model's collision meshes at the object's pose now (body pose and model arrays only: no kinematics call,
so a plan that takes no recipe runs exactly as before): resting height, flat, container, mass, friction, support
height, a cover above, and the hull section across the jaw at the grasp height (its width, how far it reaches toward
the hand, and its centre)."""

from __future__ import annotations

import types

import mujoco
import numpy as np

from . import tall_narrow

RECIPES = (tall_narrow,)
JAW_OPEN_M = 0.095  # the open jaw (rules.md 12)
SECTION_BAND_M = 0.015  # the pads' half height: the section is the widest of the jaw height and this above and below
HOUSING_HALF_Z_M = 0.031  # the gripper housing's half height about the jaw axis (robot model: z -0.031..0.031)


def _quat_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def hull(world, key: str):
    """(points, triangles) of the free object's collision meshes in the world, at its pose now; None without meshes."""
    m = world.sim.m
    b = world.free[key]
    p, q = world.object_pose(key)
    Rb = _quat_mat(q)
    pts, tris, n = [], [], 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        a, k = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        local = m.mesh_vert[a : a + k] @ _quat_mat(m.geom_quat[g]).T + m.geom_pos[g]
        pts.append(local @ Rb.T + p)
        tris.append(m.mesh_face[fa : fa + fn] + n)
        n += k
    if not pts:
        return None
    return np.concatenate(pts), np.concatenate(tris)


def section(points, tris, z: float) -> np.ndarray:
    """Where the mesh edges cross the horizontal plane at `z` (k x 3)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    za, zb = points[e[:, 0], 2], points[e[:, 1], 2]
    e, za, zb = e[(za - z) * (zb - z) <= 0], za[(za - z) * (zb - z) <= 0], zb[(za - z) * (zb - z) <= 0]
    dz = zb - za
    u = np.where(np.abs(dz) > 1e-12, (z - za) / np.where(np.abs(dz) > 1e-12, dz, 1.0), 0.0)
    return points[e[:, 0]] + u[:, None] * (points[e[:, 1]] - points[e[:, 0]])


def features(world, key: str, ux, jaw_z: float, *, flat: bool, cover: bool):
    """The object's features for a side grasp along `ux` (horizontal unit, toward the object) at `jaw_z`."""
    o = world.meta["objects"][key]
    sem = o.get("semantic") or {}
    surf = world.meta["surfaces"][o["surface"]]
    p, q = world.object_pose(key)
    Ro = _quat_mat(q)
    half = np.asarray(o["box"]["half_extent"], float)
    ux = np.asarray(ux, float)
    uy = np.array([-ux[1], ux[0], 0.0])
    obj = types.SimpleNamespace(
        height=float(2 * np.abs(Ro[2]) @ half), flat=bool(flat), container=bool(o.get("container")),
        mass=sem.get("mass_kg"), friction=sem.get("friction"), support_top=float(surf["top"]), cover=bool(cover),
        width=None, margin=None, near=None, near_housing=None, aim=None,
    )  # fmt: skip
    h = hull(world, key)
    if h is None:
        return obj
    pads = [section(*h, jaw_z + dz) for dz in (-SECTION_BAND_M, 0.0, SECTION_BAND_M)]
    if any(len(c) == 0 for c in pads):
        return obj
    housing = [c for c in (section(*h, jaw_z + dz) for dz in (-HOUSING_HALF_Z_M, HOUSING_HALF_Z_M)) if len(c)] + pads
    lat = [(c - p) @ uy for c in pads]
    obj.width = float(max(v.max() - v.min() for v in lat))
    obj.margin = 0.5 * (JAW_OPEN_M - obj.width)
    centre_lat = 0.5 * float(lat[1].max() + lat[1].min())
    # the near face (behind the object's origin along ux) the pads and the housing meet
    obj.near = float(max(-np.min((c - p) @ ux) for c in pads))
    obj.near_housing = float(max(-np.min((c - p) @ ux) for c in housing))
    obj.aim = np.array([p[0], p[1], jaw_z]) + centre_lat * uy  # the section's lateral middle at the jaw height
    return obj


def choose(obj):
    """The recipe whose `match` scores highest above 0, or None (the generic pick)."""
    best, score = None, 0.0
    for r in RECIPES:
        s = float(r.match(obj))
        if s > score:
            best, score = r, s
    return best

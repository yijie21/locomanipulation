"""CuRobo warehouse study, stage 1: planning cases and the clearances of our own reference.

For every round of a mars warehouse collection run this writes one case:
- the measured state at the first `reach` frame (the planner's start: pelvis, 29 joints, jaws, box pose);
- the expert's geometric grasp points (`low_reach.observed_grips`, world and box frame) at the first reach frame;
- the expert's pre-feedback grasp target (grip point + palm attitude, both hands) at the end of `descend`,
  taken from the ghost trace, and the same target expressed in the box frame;
- per phase (reach, align, descend), the smallest signed distance between each robot body and each
  obstacle (the five carton walls and the platform) for two postures: the expert's commanded posture
  (`reference_joints` and the root attitude, pelvis translated to keep the measured feet) and the
  measured posture. Negative = penetration.

The run's scene XML is the geometry truth; distances are sampled surface-to-box SDFs (see Surfaces).

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/curobo_warehouse_20261008/cases.py \
        outputs/runs/warehouse_mars_50 outputs/curobo_warehouse_20261008/cases.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.bimanual_carry import OFFSET

PHASES = ("reach", "align", "descend")
BOX_FREE = "nor_cal_box_free"
BOX_BODY = "nor_cal_box_object"
PLATFORM_BODY = "logistics_platform"
DIST_MAX = 0.05  # m; distances beyond this are reported as DIST_MAX


SPACING = 0.002  # m, mean spacing of the surface samples


def box_sdf(points, center, R, half):
    """Signed distance of world points to a box (negative inside)."""
    q = np.abs((points - center) @ R) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


class Surfaces:
    """Surface samples of collision geoms in their own frames: the convex hull for meshes (what MuJoCo collides),
    the exact surface for boxes and cylinders. The distance to a box obstacle is the smallest box SDF over the
    samples, accurate to about SPACING; a sample inside the box means penetration.

    Not mj_geomDistance: in MuJoCo 3.11 it returned exactly 0.0 for separated meshes (100 of 11520 hand-wall
    pairs) and the wrong sign for others (-11.4 mm where the hull vertices are all 11.4 mm outside the wall,
    one pose 2026-10-08), so it cannot classify penetration here.
    """

    def __init__(self, m, geoms, spacing=SPACING, seed=0):
        self.m, self.local = m, {}
        rng = np.random.default_rng(seed)
        for g in geoms:
            self.local[g] = self._sample(int(g), spacing, rng)

    def _sample(self, g, spacing, rng):
        m, kind, size = self.m, int(self.m.geom_type[g]), self.m.geom_size[g]
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            hull = ConvexHull(verts)
            tris = verts[hull.simplices]
            return np.vstack([verts[hull.vertices], sample_triangles(tris, spacing, rng)])
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * size
            hull = ConvexHull(corners)
            return np.vstack([corners, sample_triangles(corners[hull.simplices], spacing, rng)])
        if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            r, h = size[0], size[1]
            n_side = max(8, int(2 * np.pi * r * 2 * h / spacing**2))
            a, z = rng.uniform(0, 2 * np.pi, n_side), rng.uniform(-h, h, n_side)
            side = np.c_[r * np.cos(a), r * np.sin(a), z]
            n_cap = max(8, int(np.pi * r * r / spacing**2))
            rr, aa = r * np.sqrt(rng.uniform(0, 1, 2 * n_cap)), rng.uniform(0, 2 * np.pi, 2 * n_cap)
            caps = np.c_[rr * np.cos(aa), rr * np.sin(aa), np.repeat([-h, h], n_cap)]
            return np.vstack([side, caps])
        raise ValueError(f"geom {g}: unsupported type {kind}")

    def world(self, d, g):
        return self.local[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    def to_box(self, d, g, g_box, distmax=DIST_MAX):
        """Signed distance from geom g to box geom g_box (clipped at distmax), as placed in d."""
        offset, half = box_extent(self.m, g_box)
        R = d.geom_xmat[g_box].reshape(3, 3)
        center = d.geom_xpos[g_box] + R @ offset
        if np.linalg.norm(d.geom_xpos[g] - center) > self.m.geom_rbound[g] + np.linalg.norm(half) + distmax:
            return distmax
        return float(min(distmax, box_sdf(self.world(d, g), center, R, half).min()))


def box_extent(m, g):
    """(centre offset, half extents) of a box obstacle in its geom frame. A mesh obstacle is accepted when its
    convex hull is an axis-aligned box in the geom frame (the run copy of warehouse_box_45x32x33_mars has the
    platform as such a mesh; the rebuilt scene has a box)."""
    kind = int(m.geom_type[g])
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.zeros(3), m.geom_size[g].copy()
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = m.geom_dataid[g]
        verts = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        corner_gap = np.abs(np.abs(verts - (lo + hi) / 2) - (hi - lo) / 2).max(axis=1).max()
        if corner_gap > 1e-4:
            raise ValueError(f"mesh obstacle {g} is not an axis-aligned box ({corner_gap:.4f} m off its corners)")
        return (lo + hi) / 2, (hi - lo) / 2
    raise ValueError(f"obstacle geom {g} is neither a box nor a box-shaped mesh")


def sample_triangles(tris, spacing, rng):
    """Area-weighted uniform samples on triangles, about one per spacing**2."""
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    n = max(1, int(area.sum() / spacing**2))
    pick = rng.choice(len(tris), n, p=area / area.sum())
    u, v = rng.uniform(size=(2, n))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tris[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def matrix_to_quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


class Geometry:
    """The scene's robot and obstacle collision geoms, and qpos addresses."""

    def __init__(self, xml):
        m = mujoco.MjModel.from_xml_path(xml)
        self.m, self.d = m, mujoco.MjData(m)
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i)  # noqa: E731
        self.hinge_qadr = np.array(
            [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        )
        self.jaw_qadr = np.array(
            [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
        )
        self.box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, BOX_FREE)]
        box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY)
        platform = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PLATFORM_BODY)
        collides = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]  # noqa: E731
        self.obstacles = [
            (g, name(mujoco.mjtObj.mjOBJ_GEOM, g))
            for g in range(m.ngeom)
            if collides(g) and m.geom_bodyid[g] in (box, platform)
        ]
        self.robot = [
            (g, name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
            for g in range(m.ngeom)
            if collides(g) and m.geom_bodyid[g] not in (0, box, platform)
        ]
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
        for g, _ in self.obstacles:
            box_extent(m, g)  # every obstacle must be box-shaped for the sampled distance
        self.surfaces = Surfaces(m, [g for g, _ in self.robot])

    def feet_mean(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.feet].mean(axis=0)

    def reference_qpos(self, measured, reference_joints, root_quaternion):
        """The commanded posture: reference joints and root attitude, translated so the feet stay where they were
        measured (the expert's own anchor, CoordinatedReach infers its pelvis translation the same way)."""
        q = measured.copy()
        q[self.hinge_qadr] = reference_joints
        if root_quaternion is not None:
            q[3:7] = root_quaternion
        q[:3] += self.feet_mean(measured) - self.feet_mean(q)
        return q

    def clearances(self, qpos):
        """{robot body: {obstacle geom: signed distance}} for one full qpos, closest geom per body."""
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        out: dict = {}
        for g_obs, obs_name in self.obstacles:
            for g_rob, body in self.robot:
                dist = self.surfaces.to_box(self.d, g_rob, g_obs)
                row = out.setdefault(body, {})
                row[obs_name] = min(row.get(obs_name, DIST_MAX), dist)
        return out


def merge_worst(acc, clear):
    for body, row in clear.items():
        for obs, dist in row.items():
            acc.setdefault(body, {})
            acc[body][obs] = min(acc[body].get(obs, DIST_MAX), dist)
    return acc


def worst_pair(acc):
    best = (DIST_MAX, None, None)
    for body, row in acc.items():
        for obs, dist in row.items():
            if dist < best[0]:
                best = (dist, body, obs)
    return dict(dist=best[0], body=best[1], obstacle=best[2])


def round_case(run, label, geo, stride):
    obs = [json.loads(line) for line in open(f"{run}/observations/{label}/observations.jsonl")]
    result = json.load(open(f"{run}/observations/{label}/result.json"))
    trace = np.load(f"{run}/trace/{label}_state_trace.npz")
    ghost_path = f"{run}/trace/{label}_ghost.npz"
    ghost = np.load(ghost_path) if os.path.exists(ghost_path) else None
    seqs = trace["seq"]

    def qpos_at(seq):
        i = int(np.clip(np.searchsorted(seqs, seq), 0, len(seqs) - 1))
        return trace["qpos"][i].copy(), int(seqs[i])

    phases = [o.get("phase") for o in obs]
    case = dict(
        label=label,
        ok=bool(result.get("ok")),
        why=result.get("why") or "",
        phases={p: [o["state_seq"] for o in obs if o.get("phase") == p] for p in dict.fromkeys(phases) if p},
    )
    case["phases"] = {p: [s[0], s[-1], len(s)] for p, s in case["phases"].items()}
    if "reach" not in case["phases"]:
        case["skip"] = "no reach phase"
        return case

    start_seq = case["phases"]["reach"][0]
    qpos, seq = qpos_at(start_seq)
    case["start"] = dict(seq=seq, qpos=qpos.tolist())
    first = next(o for o in obs if o.get("phase") == "reach")
    grips = np.asarray((first.get("low_reach") or {}).get("observed_grips") or [], dtype=float)
    if grips.shape == (2, 3):
        box_q = qpos[geo.box_qadr : geo.box_qadr + 7]
        Rb = quat_wxyz_to_matrix(box_q[3:])
        case["grasp_points"] = dict(world=grips.tolist(), box=((grips - box_q[:3]) @ Rb).tolist())

    descend = [o for o in obs if o.get("phase") == "descend"]
    if descend:
        end_q, end_seq = qpos_at(descend[-1]["state_seq"])
        case["descend_end"] = dict(seq=end_seq, qpos=end_q.tolist())

    if ghost is not None and descend:
        i = int(np.searchsorted(ghost["seq"], descend[-1]["state_seq"], side="right") - 1)
        box_q = qpos_at(descend[-1]["state_seq"])[0][geo.box_qadr : geo.box_qadr + 7]
        Rb, pb = quat_wxyz_to_matrix(box_q[3:]), box_q[:3]
        target = []
        for h in range(2):
            R = quat_wxyz_to_matrix(ghost["quat"][i, h])
            grip = ghost["pos"][i, h] + R @ OFFSET
            target.append(
                dict(
                    grip_world=grip.tolist(),
                    quat_world=ghost["quat"][i, h].tolist(),
                    grip_box=(Rb.T @ (grip - pb)).tolist(),
                    quat_box=matrix_to_quat_wxyz(Rb.T @ R),
                )
            )
        case["descend_end_target"] = dict(ghost_seq=int(ghost["seq"][i]), hands=target)

    for kind in ("reference", "measured"):
        per_phase = {}
        for phase in PHASES:
            frames = [o for o in obs if o.get("phase") == phase][::stride]
            acc: dict = {}
            for o in frames:
                q, _ = qpos_at(o["state_seq"])
                if kind == "reference":
                    root = (o.get("root_reference") or {}).get("quaternion")
                    q = geo.reference_qpos(q, o["reference_joints"], root)
                acc = merge_worst(acc, geo.clearances(q))
            if frames:
                per_phase[phase] = dict(frames=len(frames), worst=worst_pair(acc), bodies=acc)
        case[kind] = per_phase
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--scene", default=None, help="scene XML (default: the run's copy)")
    ap.add_argument("--stride", type=int, default=2, help="every Nth observation frame per phase")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    scene = args.scene or glob.glob(f"{args.run}/warehouse_box_*/scene_locomani.xml")[0]
    geo = Geometry(scene)
    labels = sorted(os.listdir(f"{args.run}/observations"))
    if args.limit:
        labels = labels[: args.limit]
    cases = []
    for k, label in enumerate(labels):
        cases.append(round_case(args.run, label, geo, args.stride))
        c = cases[-1]
        ref = c.get("reference", {})
        worst = min((p["worst"]["dist"] for p in ref.values()), default=DIST_MAX)
        print(f"[{k + 1}/{len(labels)}] {label} ok={c['ok']} why={c['why']} ref_worst={worst:+.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = dict(
        run=args.run,
        scene=scene,
        offset=OFFSET.tolist(),
        obstacles=[n for _, n in geo.obstacles],
        robot_bodies=sorted({b for _, b in geo.robot}),
        dist_max=DIST_MAX,
        stride=args.stride,
    )
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, cases=cases), f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

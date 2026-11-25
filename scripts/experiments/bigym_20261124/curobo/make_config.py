"""bigym_20261124: a cuRobo robot for BiGym 2.0's G1 with Dex1-1 grippers, generated from BiGym's own MJCF.

    source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh   # mujoco, trimesh, scipy, yaml
    python -I scripts/experiments/bigym_20261124/curobo/make_config.py <BiGym2 root> <out dir>
    # out dir: outputs/bigym_20261124/curobo in the 4090 checkout

What it writes (all generated; nothing in the BiGym package is edited):
  g1_29dof_with_dex1_1.urdf  the whole robot from bigym/envs/xmls/g1/g1_29dof_with_dex1_1.xml: the pelvis as the root (its
                             free joint dropped), every body a link, every hinge a revolute joint and every slide (the
                             four Dex1 finger joints) a prismatic one, with the MJCF's limits, body poses and inertials;
                             fixed links right_end_effector / left_end_effector at (0.148, 0, 0) in *_wrist_yaw_link, the
                             sites BiGym adds at load (G1Dex1._EE_SITE_POS, the pinch centre its success checks use). No
                             visual or collision elements: cuRobo collides with the spheres.
  g1_29dof_with_dex1_1.yml   SIMPLE's g1_29dof_with_dex3.yml (data/robots/g1/curobo) with the Dex3 links replaced by the
                             Dex1 links: cspace (legs, waist, arms, then the 2+2 finger joints), the collision links (both
                             arms with the wrist roll and yaw links too, the torso, both grippers), self-collision ignore
                             lists, zero buffers, collision_sphere_buffer 0.005, ee_link right_end_effector.
  spheres/g1_29dof_dex1.yml  spheres fitted to the MJCF's collision geoms (contype or conaffinity set) in each link's frame.
  fit.json                   per link: the geoms, the bloat bound and the sphere count the fit settled on.

The sphere fit. MuJoCo collides a mesh geom as its convex hull, so each collision geom is a convex piece (the hull of
the mesh's vertices, the box, the cylinder as a 64-gon prism). Per link: surface points on the pieces (less the parts
inside another piece), candidate centres on a grid inside the pieces (plus each piece's centroid and its long axis), each
candidate's radius its depth inside its piece plus the bloat bound `delta` (so no sphere reaches more than delta beyond
the geom), then a greedy set cover of the surface points; each kept sphere is shrunk to the farthest surface point it is
the nearest cover of (never below its depth). delta starts at 3 mm and grows until the link needs at most its cap.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial import ConvexHull

MJCF = "bigym/envs/xmls/g1/g1_29dof_with_dex1_1.xml"
EE_SITE_POS = (0.148, 0.0, 0.0)  # bigym.robots.configs.g1.G1Dex1._EE_SITE_POS
NAME = "g1_29dof_with_dex1_1"
SIDES = ("left", "right")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
DEX1 = ("dex1_base_link", "dex1_finger_link_1", "dex1_finger_link_2")
LEGS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
DELTAS = (0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.0125, 0.015, 0.02, 0.025)
CAP = {"torso_link": 20, "dex1_base_link": 12, "dex1_finger_link_1": 10, "dex1_finger_link_2": 10}
CAP_ARM = 10


def arm_links(side: str) -> list[str]:
    return [f"{side}_{j}_link" for j in ARM]


def dex1_links(side: str) -> list[str]:
    return [f"{side}_{n}" for n in DEX1]


def collision_links() -> list[str]:
    """SIMPLE's order: the left arm, the torso, the right arm, the left hand, the right hand."""
    return [*arm_links("left"), "torso_link", *arm_links("right"), *dex1_links("left"), *dex1_links("right")]


def cspace_joints() -> list[str]:
    j = [f"{s}_{n}_joint" for s in SIDES for n in LEGS] + list(WAIST)
    j += [f"{s}_{n}_joint" for s in SIDES for n in ARM]
    j += [f"{s}_dex1_finger_joint_{k}" for s in SIDES for k in (1, 2)]
    return j


# ----------------------------------------------------------------------------- URDF
def rpy_of(R: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (fixed axes x, y, z: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-9:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:  # gimbal lock: put it all in yaw
        roll, yaw = 0.0, float(np.arctan2(-R[0, 1], R[1, 1]))
    return roll, pitch, yaw


def quat_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def fmt(v) -> str:
    return " ".join(f"{float(x):.10g}" for x in v)


def write_urdf(m: mujoco.MjModel, path: Path) -> dict:
    robot = ET.Element("robot", name=NAME)
    pelvis = m.body("pelvis").id
    info = {"links": [], "joints": []}

    def add_link(name: str, body: int | None) -> None:
        link = ET.SubElement(robot, "link", name=name)
        if body is not None and m.body_mass[body] > 0:
            ine = ET.SubElement(link, "inertial")
            ET.SubElement(ine, "origin", xyz=fmt(m.body_ipos[body]), rpy=fmt(rpy_of(quat_mat(m.body_iquat[body]))))
            ET.SubElement(ine, "mass", value=f"{float(m.body_mass[body]):.10g}")
            ix, iy, iz = m.body_inertia[body]
            ET.SubElement(
                ine, "inertia", ixx=f"{ix:.10g}", iyy=f"{iy:.10g}", izz=f"{iz:.10g}", ixy="0", ixz="0", iyz="0"
            )
        info["links"].append(name)

    def add_joint(name, kind, parent, child, pos, R, axis=None, lo=None, hi=None, effort=None):
        j = ET.SubElement(robot, "joint", name=name, type=kind)
        ET.SubElement(j, "origin", xyz=fmt(pos), rpy=fmt(rpy_of(R)))
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        if axis is not None:
            ET.SubElement(j, "axis", xyz=fmt(axis))
            ET.SubElement(j, "limit", lower=f"{lo:.10g}", upper=f"{hi:.10g}", effort=f"{effort:.10g}", velocity="30")
        info["joints"].append(name)

    add_link("pelvis", pelvis)
    for b in range(m.nbody):
        if b in (0, pelvis):
            continue
        name, parent = m.body(b).name, m.body(m.body_parentid[b]).name
        assert parent != "world", name
        add_link(name, b)
        jids = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        pos, R = m.body_pos[b], quat_mat(m.body_quat[b])
        if not jids:
            add_joint(f"{name}_fixed_joint", "fixed", parent, name, pos, R)
            continue
        assert len(jids) == 1, name
        j = jids[0]
        assert np.allclose(m.jnt_pos[j], 0.0), f"{m.joint(j).name}: joint offset in its body"
        kind = {int(mujoco.mjtJoint.mjJNT_HINGE): "revolute", int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic"}[
            int(m.jnt_type[j])
        ]
        lo, hi = m.jnt_range[j]
        eff = float(m.jnt_actfrcrange[j][1]) if m.jnt_actfrclimited[j] else 100.0
        add_joint(m.joint(j).name, kind, parent, name, pos, R, m.jnt_axis[j], lo, hi, eff)
    for side in SIDES:
        add_link(f"{side}_end_effector", None)
        add_joint(f"{side}_end_effector_fixed_joint", "fixed", f"{side}_wrist_yaw_link", f"{side}_end_effector",
                  EE_SITE_POS, np.eye(3))  # fmt: skip
    ET.indent(robot)
    path.write_text('<?xml version="1.0"?>\n<!-- generated by bigym_20261124/curobo/make_config.py -->\n'
                    + ET.tostring(robot, encoding="unicode") + "\n")  # fmt: skip
    return info


# ----------------------------------------------------------------------------- collision pieces
def geom_points(m: mujoco.MjModel, g: int) -> np.ndarray:
    """Points whose convex hull is the geom as MuJoCo collides it, in the geom's own frame."""
    t, s = int(m.geom_type[g]), m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
        return m.mesh_vert[a : a + n].astype(float)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s[:3]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.c_[s[0] * np.cos(a), s[0] * np.sin(a)]
        return np.r_[np.c_[ring, np.full(64, -s[1])], np.c_[ring, np.full(64, s[1])]]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(3, radius=float(s[0])).vertices
    raise ValueError(f"geom type {t}")


class Piece:
    """One convex collision piece in the link frame: hull equations (A x + b <= 0 inside) and a trimesh for sampling."""

    def __init__(self, pts: np.ndarray, label: str):
        h = ConvexHull(pts)
        self.A, self.b = h.equations[:, :3], h.equations[:, 3]
        self.mesh = trimesh.Trimesh(pts, h.simplices, process=True)
        self.label = label
        self.centroid = pts[h.vertices].mean(0)
        ev, evec = np.linalg.eigh(np.cov((pts[h.vertices] - self.centroid).T))
        self.axis = evec[:, -1]

    def depth(self, X: np.ndarray) -> np.ndarray:
        """Distance inside the piece (> 0 inside, < 0 outside: then only a lower bound of the distance)."""
        return -(X @ self.A.T + self.b).max(1)


def link_pieces(m: mujoco.MjModel, body: int) -> list[Piece]:
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        X = geom_points(m, g) @ quat_mat(m.geom_quat[g]).T + m.geom_pos[g]
        t = int(m.geom_type[g])
        label = (
            m.mesh(m.geom_dataid[g]).name
            if t == mujoco.mjtGeom.mjGEOM_MESH
            else f"type{t}{m.geom_size[g].round(4).tolist()}"
        )
        out.append(Piece(X, label))
    return out


def surface_samples(pieces: list[Piece], n: int, seed: int) -> np.ndarray:
    """Area-weighted surface points of the union (points deeper than 1 mm inside another piece dropped)."""
    rng = np.random.default_rng(seed)
    areas = np.array([p.mesh.area for p in pieces])
    out = []
    for i, p in enumerate(pieces):
        k = max(200, int(n * areas[i] / areas.sum()))
        X, _ = trimesh.sample.sample_surface(p.mesh, k, seed=int(rng.integers(1 << 31)))
        V = p.mesh.vertices
        if len(V) > 400:
            V = V[rng.choice(len(V), 400, replace=False)]
        X = np.r_[X, V]
        inside_other = np.zeros(len(X), bool)
        for j, q in enumerate(pieces):
            if j != i:
                inside_other |= q.depth(X) > 1e-3
        out.append(X[~inside_other])
    return np.concatenate(out)


def depth_union(pieces: list[Piece], X: np.ndarray) -> np.ndarray:
    return np.max([p.depth(X) for p in pieces], axis=0)


def candidates(pieces: list[Piece]) -> np.ndarray:
    lo = np.min([p.mesh.bounds[0] for p in pieces], 0)
    hi = np.max([p.mesh.bounds[1] for p in pieces], 0)
    pitch = max(0.0015, float(np.prod(hi - lo) / 6000.0) ** (1 / 3))
    g = np.stack(np.meshgrid(*[np.arange(a + pitch / 2, b, pitch) for a, b in zip(lo, hi)], indexing="ij"), -1)
    G = g.reshape(-1, 3)
    G = G[depth_union(pieces, G) > 0]
    if len(G) > 5000:
        G = G[np.random.default_rng(1).choice(len(G), 5000, replace=False)]
    ax = []
    for p in pieces:  # thin pieces (the 3 mm pads) may fall between grid points: each piece's long axis too
        ext = np.abs((p.mesh.vertices - p.centroid) @ p.axis).max()
        ax.append(p.centroid + np.linspace(-ext, ext, 41)[:, None] * p.axis)
    A = np.concatenate(ax)
    return np.r_[G, A[depth_union(pieces, A) > 0]]


def fit_link(pieces: list[Piece], cap: int) -> dict:
    S = surface_samples(pieces, 4000, seed=0)
    C = candidates(pieces)
    din = depth_union(pieces, C)
    S32, C32 = S.astype(np.float32), C.astype(np.float32)
    D2 = (C32**2).sum(1)[:, None] + (S32**2).sum(1)[None] - 2 * C32 @ S32.T
    D = np.sqrt(np.maximum(D2, 0))  # (nc, ns)
    for delta in DELTAS:  # the full cover at each bound; the first within the cap (else the last bound's, over the cap)
        r = din + delta
        cover = D <= r[:, None]
        left = np.ones(len(S), bool)
        chosen = []
        while left.any():
            gain = (cover & left).sum(1)
            i = int(np.argmax(gain))
            if gain[i] == 0:
                break
            chosen.append(i)
            left &= ~cover[i]
        if len(chosen) <= cap:
            break
    # shrink: each covered surface point to the nearest chosen centre that covers it
    ch = np.array(chosen)
    Dc = D[ch]
    cov = cover[ch]
    rel = np.where(cov, Dc, np.inf)
    owner = np.where(cov.any(0), np.argmin(rel, 0), -1)
    rad = []
    for k, i in enumerate(ch):
        mine = owner == k
        need = float(Dc[k, mine].max()) if mine.any() else 0.0
        rad.append(max(need, float(din[i])))
    return dict(centers=C[ch], radii=np.array(rad), delta=delta, n_surface=len(S), covered_fit=bool(not left.any()))


# ----------------------------------------------------------------------------- yml
def self_collision_ignore(m: mujoco.MjModel, spheres: dict, buffer: float) -> dict:
    """SIMPLE's pattern (an arm's links ignore each other, the torso its shoulders, a hand's links each other) with the
    Dex1 gripper also ignoring its own wrist links, plus every pair whose spheres overlap at the zero pose."""
    ign: dict[str, set] = {n: set() for n in collision_links()}
    for side in SIDES:
        arm, hand = arm_links(side), dex1_links(side)
        for a in arm:
            ign[a] |= set(arm)
        for h in hand:
            ign[h] |= set(hand) | {f"{side}_wrist_roll_link", f"{side}_wrist_pitch_link", f"{side}_wrist_yaw_link"}
        for s in (f"{side}_shoulder_pitch_link", f"{side}_shoulder_roll_link"):
            ign[s].add("torso_link")
            ign["torso_link"].add(s)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    world = {}
    for n, sp in spheres.items():
        b = m.body(n).id
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        world[n] = (np.array([s["center"] for s in sp]) @ Rb.T + pb, np.array([s["radius"] for s in sp]) + buffer)
    zero_pairs = []
    names = list(spheres)
    for i, a in enumerate(names):
        for b_ in names[i + 1 :]:
            if b_ in ign[a] or a in ign[b_]:
                continue
            (Ca, ra), (Cb, rb) = world[a], world[b_]
            gap = np.linalg.norm(Ca[:, None] - Cb[None], axis=-1) - ra[:, None] - rb[None]
            if gap.min() < 0:
                ign[a].add(b_)
                ign[b_].add(a)
                zero_pairs.append((a, b_, round(float(gap.min()) * 1000, 1)))
    for n in ign:  # SIMPLE lists the link itself too
        ign[n].add(n)
    order = collision_links()
    return {n: [x for x in order if x in ign[n]] for n in order}, zero_pairs


def write_yml(path: Path, spheres_rel: str, ignore: dict) -> dict:
    cs = cspace_joints()
    n = len(cs)
    kin = {
        "use_usd_kinematics": False,
        "urdf_path": f"{NAME}.urdf",
        "external_robot_configs_path": None,  # set by the loader (plan_server.py / fk_check.py) to this file's folder
        "external_asset_path": None,
        "asset_root_path": None,
        "isaac_usd_path": None,
        "usd_robot_root": None,
        "usd_path": None,
        "base_link": "pelvis",
        "ee_link": "right_end_effector",
        "link_names": [
            "waist_yaw_link",
            "waist_roll_link",
            "torso_link",
            "right_end_effector",
            "right_dex1_finger_link_1",
            "right_dex1_finger_link_2",
            "left_end_effector",
            "left_dex1_finger_link_1",
            "left_dex1_finger_link_2",
        ],  # fmt: skip
        "collision_link_names": collision_links(),
        "collision_spheres": spheres_rel,
        "collision_sphere_buffer": 0.005,
        "extra_collision_spheres": {},
        "self_collision_ignore": ignore,
        "self_collision_buffer": {k: 0.0 for k in collision_links()},
        "use_global_cumul": True,
        "mesh_link_names": None,
        "cspace": {
            "joint_names": cs,
            "retract_config": [0.0] * n,
            "null_space_weight": [1.0] * n,
            "cspace_distance_weight": [1.0] * n,
            "max_jerk": 500.0,
            "max_acceleration": 10.0,
        },
    }
    head = "# generated by scripts/experiments/bigym_20261124/curobo/make_config.py from BiGym's G1 Dex1-1 MJCF\n"
    path.write_text(head + yaml.safe_dump({"robot_cfg": {"kinematics": kin}}, sort_keys=False, default_flow_style=None))
    return kin


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "spheres").mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(root / MJCF))
    info = write_urdf(m, out / f"{NAME}.urdf")
    spheres, fit = {}, {}
    for n in collision_links():
        pieces = link_pieces(m, m.body(n).id)
        key = n.split("_", 1)[1] if n.startswith(("left_dex1", "right_dex1")) else n
        cap = CAP.get(key, CAP.get(n, CAP_ARM))
        f = fit_link(pieces, cap)
        spheres[n] = [{"center": [round(float(x), 5) for x in c], "radius": round(float(r), 5)}
                      for c, r in zip(f["centers"], f["radii"])]  # fmt: skip
        fit[n] = dict(geoms=[p.label for p in pieces], n=len(spheres[n]), cap=cap, delta_mm=1000 * f["delta"],
                      covered_fit=f["covered_fit"], n_surface=f["n_surface"])  # fmt: skip
        print(f"{n:28s} {len(pieces)} geoms {fit[n]['geoms']}  -> {len(spheres[n])} spheres, delta {1000 * f['delta']:.1f} mm", flush=True)  # fmt: skip
    rel = "spheres/g1_29dof_dex1.yml"
    (out / rel).write_text("# generated by bigym_20261124/curobo/make_config.py: spheres fitted to BiGym's G1 Dex1-1 collision geoms\n"
                           + yaml.safe_dump({"collision_spheres": spheres}, sort_keys=False, default_flow_style=None))  # fmt: skip
    ignore, zero_pairs = self_collision_ignore(m, spheres, 0.005)
    write_yml(out / f"{NAME}.yml", rel, ignore)
    (out / "fit.json").write_text(json.dumps(dict(mjcf=str(root / MJCF), links=fit, zero_pose_overlaps_ignored=zero_pairs,
                                                  urdf_links=len(info["links"]), urdf_joints=len(info["joints"])), indent=1))  # fmt: skip
    print("zero-pose overlaps added to the ignore lists:", zero_pairs)
    print("wrote", out)


if __name__ == "__main__":
    main()

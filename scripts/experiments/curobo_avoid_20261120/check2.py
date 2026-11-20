"""curobo_avoid_20261120 step 2c: the planned paths checked in the replica's own geometry (not cuRobo's spheres): along each
plan, the right arm's joints set from the plan and the rest of the robot held at the reach's start reading, the distance
of the right forearm's, wrist's and hand's geoms to the table and to the rest of the robot (torso, pelvis, legs, left arm,
head), from the geoms' surface points (mesh vertices; primitives sampled): the table exactly (box distance of the points), the
rest by the nearest surface points (mj_geomDistance in mujoco 3.11 gives spurious 0s between 4-9 cm neighbours here); and the jaw centre by the replica's kinematics against cuRobo's.

    cd $REPO && SIMPLE_ROOT=$REPO/third_party/SIMPLE $PY -I check2.py <step1 dir> <step2 dir> EP
-> one JSON line
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experiments/simple_eval_20261112"))
import episodes as EP  # noqa: E402

S1, S2 = Path(sys.argv[1]), Path(sys.argv[2])
e = int(sys.argv[3])
T = "TabletopGraspMP"
name = f"{T}_L{e // 10}_e{e % 10}"
pl = S2 / f"plan_{name}.npz"
if not pl.exists() or "Q" not in np.load(pl).files:
    print(json.dumps(dict(ep=e, skip="no plan")))
    sys.exit(0)
P, pr = np.load(pl), np.load(S2 / f"prep_{name}.npz")
info = EP.build(EP.episodes(T)[e], S1 / "scene" / name)
m = mujoco.MjModel.from_xml_path(info["dir"] + "/scene_locomani.xml")
d = mujoco.MjData(m)
root = m.body_rootid[m.body("pelvis").id]
# the pose at the reach's start: the free joint from the pelvis pose, every hinge from the reading
fj = [
    j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == root and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
][0]
a0 = m.jnt_qposadr[fj]
Rp = pr["pelvis_R"]
q = np.empty(4)
mujoco.mju_mat2Quat(q, Rp.reshape(-1))
d.qpos[a0 : a0 + 3], d.qpos[a0 + 3 : a0 + 7] = pr["pelvis_p"], q
for n, v in zip([str(x) for x in pr["names"]], pr["q"]):
    d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
EP.pull_objects(m, d, info) if hasattr(EP, "pull_objects") else None
mujoco.mj_forward(m, d)


def subtree(b0):
    out, todo = set(), [b0]
    while todo:
        b = todo.pop()
        out.add(b)
        todo += [c for c in range(m.nbody) if m.body_parentid[c] == b and c != b]
    return out


arm = subtree(m.body("right_shoulder_pitch_link").id)
moving = subtree(m.body("right_elbow_link").id)  # forearm, wrist, hand
G = lambda bodies: [
    g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
]  # noqa: E731
mov_g = G(moving)
robot = {b for b in range(m.nbody) if m.body_rootid[b] == root}
rest_g = G(robot - arm)
table_g = [m.geom("table_geom").id]
names = [str(x) for x in P["names"]]
adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
wr = m.body("right_wrist_yaw_link").id
OFF = pr["OFF"]


def surface(g, n=300):
    """points on geom g's surface in its own frame: a mesh's vertices, a primitive sampled"""
    t, sz = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float)
    u = np.random.default_rng(0).normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return sz[0] * u
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.linspace(-sz[1], sz[1], n)[:, None]
        return (
            sz[0] * u + np.c_[0 * z, 0 * z, z]
        )  # spheres along the axis: the surface and some inside (harmless for a minimum)
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a_ = np.random.default_rng(1).uniform(0, 2 * np.pi, n)
        z = np.random.default_rng(2).uniform(-sz[1], sz[1], n)
        return np.c_[sz[0] * np.cos(a_), sz[0] * np.sin(a_), z]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return np.clip(u * 2, -1, 1) * sz
    raise ValueError(f"geom type {t}")


from scipy.spatial import cKDTree  # noqa: E402

SURF = {g: surface(g) for g in mov_g + rest_g}
world = lambda gs: np.concatenate([d.geom_xpos[g] + SURF[g] @ d.geom_xmat[g].reshape(3, 3).T for g in gs])  # noqa: E731
owner = lambda gs: np.concatenate([np.full(len(SURF[g]), g) for g in gs])  # noqa: E731
own_m, own_r = owner(mov_g), owner(rest_g)
tb = m.geom("table_geom").id
c_t, h_t = d.geom_xpos[tb].copy(), m.geom_size[tb].copy()
R_t = d.geom_xmat[tb].reshape(3, 3).copy()


def box_sd(X):
    dd = np.abs((X - c_t) @ R_t) - h_t
    return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


Q = P["Q"]
dt, ds, jerr = [], [], []
worst_t, worst_s = (1.0, None, -1), (1.0, None, -1)
Xr = world(rest_g)  # the rest of the robot does not move along the plan
tree_r = cKDTree(Xr)
gname = lambda g: m.geom(g).name or m.body(m.geom_bodyid[g]).name  # noqa: E731
for k, qk in enumerate(Q):
    d.qpos[adr] = qk
    mujoco.mj_kinematics(m, d)
    Xm = world(mov_g)
    sdt = box_sd(Xm)
    i = int(np.argmin(sdt))
    t_ = float(sdt[i])
    dd_, j = tree_r.query(Xm)
    i2 = int(np.argmin(dd_))
    s_ = float(dd_[i2])
    dt.append(t_)
    ds.append(s_)
    if t_ < worst_t[0]:
        worst_t = (t_, (gname(own_m[i]), "table"), k)
    if s_ < worst_s[0]:
        worst_s = (s_, (gname(own_m[i2]), gname(own_r[j[i2]])), k)
    jaw = d.xpos[wr] + d.xmat[wr].reshape(3, 3) @ OFF
    jerr.append(np.linalg.norm(jaw - P["jaw"][k]))
dt, ds = np.array(dt), np.array(ds)
left = np.flatnonzero(dt >= 0.03)
out = dict(
    ep=e,
    T=len(Q),
    table_start_cm=round(100 * dt[0], 2),
    table_min_cm=round(100 * dt.min(), 2),
    table_after_cm=round(100 * float(dt[left[0] :].min()), 2) if len(left) else None,
    table_worst=[round(100 * worst_t[0], 2), worst_t[1], worst_t[2]],
    self_start_cm=round(100 * ds[0], 2),
    self_min_cm=round(100 * ds.min(), 2),
    self_worst=[round(100 * worst_s[0], 2), worst_s[1], worst_s[2]],
    jaw_fk_mm_max=round(1000 * max(jerr), 2),
    n_mov=len(mov_g),
    n_rest=len(rest_g),
)
if "--profile" in sys.argv:
    out.update(table_cm=(100 * dt).round(1).tolist(), self_cm=(100 * ds).round(1).tolist())
print(json.dumps(out), flush=True)

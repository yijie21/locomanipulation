"""curobo_avoid_20261120 step 1: the start look's points as obstacles, against the replica's truth (TabletopGraspMP, the Isaac
look of look_dump_eyes).

    cd $REPO && SIMPLE_ROOT=$REPO/third_party/SIMPLE MUJOCO_GL=egl $PY -I obstacles.py <look dumps> <film npz> <out> EP
    $PY -I obstacles.py <look dumps> <film npz> <out> summary
Segmentation with multisampling off (offsamples 0): blended edge pixels are ids of no geom (index 320 of 259, 2026-11-20).

Per episode:
  the robot's own pixels  the replica at the look's pose (the film's first frame: simple_bench's reset and 1 s hold are
                          blind.py look's) drawn from the head camera with segmentation, robot geoms only, dilated DIL_SELF
                          px (the Isaac image sits ~3 px off the replica's). This replaces blind.py's OWN = 0.2 m cut
                          (nearer than 20 cm ahead of the pelvis is the robot), which cuts the table edge at ~14 cm.
  obstacles               the look's points (ok) less the robot's pixels and the target's (SAM's mask, dilated DIL_TGT px),
                          inside BOX (heading frame from the look's pelvis, z world), as VOX voxels
  truth                   the replica's depth (dtrue) as points with the same cuts, its table geom's pixels, its table box
metrics: the table top (perceived z on the table's true pixels minus the true top; the robot's support_of estimate; the
robot's level of the obstacle points: the densest 0.5 cm bin of z within 6 cm of support_of's, |y - target y| < 0.3, refined
by the median within 0.6 cm), the near edge (0.5 percentile of x over obstacle points within 1.2 cm of that level, the same band)
minus the true edge, seen (>= 50 such points within 3 cm of the true edge), the true edge's pixels in view / on the robot /
outside the image, the same edge with blind.py's OWN cut, the voxels against the true ones.
-> <out>/obst_<NAME>.npz, <out>/step1.json; one line per episode on stdout
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_dilation

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))  # python -I ignores PYTHONPATH
sys.path.insert(0, str(REPO / "scripts/experiments/simple_eval_20261112"))
import episodes as EP  # noqa: E402
import see as SEE  # noqa: E402

LOOK, FILMS, OUT = (Path(x) for x in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
T = "TabletopGraspMP"
DIL_SELF, DIL_TGT = 6, 4  # px
VOX = 0.02  # m
BOX = ((-0.10, 0.90), (-0.60, 0.60), (0.30, 1.50))  # m: ahead and left of the look's pelvis (heading frame), world z
EDGE_DZ, BAND, SEEN_M, SEEN_N = 0.012, 0.30, 0.03, 50
OWN = 0.2  # blind.py's


def set_pose(m, d, film_f, info, shift=None):
    """The scene at the film's first frame, joint by joint by name (the robot and the objects); `shift` (dxy, dyaw) moves the
    robot's root for an episode without a film of its own."""
    z = np.load(film_f)
    q0 = z["qpos"][0]
    names, adr, typ = [str(n) for n in z["jnt"]], z["jnt_adr"], z["jnt_type"]
    size = {0: 7, 1: 4, 2: 1, 3: 1}
    mujoco.mj_resetData(m, d)
    EP.pull_objects(m, d, info)
    for n, a, t in zip(names, adr, typ):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0 or (shift is not None and n not in ROBOT_JOINTS(m)):
            continue
        d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + size[int(t)]] = q0[a : a + size[int(t)]]
    if shift is not None:  # the root: moved by the start's difference
        j = m.body("pelvis").jntadr[0]
        a = m.jnt_qposadr[j]
        d.qpos[a : a + 2] += shift[0]
        qw, qz = d.qpos[a + 3], d.qpos[a + 6]
        yaw = 2 * np.arctan2(qz, qw) + shift[1]
        d.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)


def ROBOT_JOINTS(m):
    root = m.body_rootid[m.body("pelvis").id]
    return {m.joint(j).name for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == root}


def voxels(P):
    return np.unique(np.floor(P / VOX).astype(np.int64), axis=0)


def crop(P, pb, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rx = c * (P[:, 0] - pb[0]) + s * (P[:, 1] - pb[1])
    ry = -s * (P[:, 0] - pb[0]) + c * (P[:, 1] - pb[1])
    (x0, x1), (y0, y1), (z0, z1) = BOX
    return (rx > x0) & (rx < x1) & (ry > y0) & (ry < y1) & (P[:, 2] > z0) & (P[:, 2] < z1)


def near(A, B):
    """Of voxel set A, the share with a voxel of B within one voxel (26-neighbourhood)."""
    if len(A) == 0:
        return float("nan")
    Bs = {tuple(v) for v in B}
    off = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
    return float(np.mean([any(tuple(a + o) in Bs for o in off) for a in A]))


eps = EP.episodes(T)
rows = []
ARG = sys.argv[4]
for e in [] if ARG == "summary" else [int(ARG)]:
    name = f"{T}_L{e // 10}_e{e % 10}"
    L = np.load(LOOK / f"{name}.npz")
    info = EP.build(eps[e], OUT / "scene" / name)
    m = mujoco.MjModel.from_xml_path(info["dir"] + "/scene_locomani.xml")
    d = mujoco.MjData(m)
    film = FILMS / f"{name}_perception_direct.npz"
    note = ""
    if film.exists():
        set_pose(m, d, film, info)
    else:  # no grasp, no film: the first episode of its level's pose, moved to this start
        e0 = (e // 10) * 10
        info0 = EP.build(eps[e0], OUT / "scene" / f"{T}_L{e0 // 10}_e{e0 % 10}")
        shift = (
            np.asarray(info["start"]["xy"]) - np.asarray(info0["start"]["xy"]),
            info["start"]["yaw"] - info0["start"]["yaw"],
        )
        set_pose(m, d, FILMS / f"{T}_L{e0 // 10}_e{e0 % 10}_perception_direct.npz", info, shift)
        note = f"pose from e{e0} moved"
    cam = m.camera(EP.HEAD_CAM).id
    cam_dp = float(100 * np.linalg.norm(d.cam_xpos[cam] - L["cam_p"]))
    W, H = EP.HEAD_CAM_WH
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, H, W)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=EP.HEAD_CAM)
    seg = r.render().copy()
    geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    gid = np.where(geom, seg[..., 0], -1)
    root = m.body_rootid[m.body("pelvis").id]
    is_robot_geom = np.r_[m.body_rootid[m.geom_bodyid] == root, False]
    robot_px = geom & is_robot_geom[gid]
    table_gid = m.geom("table_geom").id
    table_px = gid == table_gid
    # truth: the table box, the replica's depth as points
    tb = info["tables"]["table"]
    top_true = float(tb["top"])
    edge_true = float(tb["pos"][0] - 0.5 * tb["size"][0])
    q = tb["quat"]
    assert abs(2 * np.arctan2(q[3], q[0])) < 1e-3, "an axis-aligned table"
    pts, ok, mask, dtrue = L["pts"], L["ok"], L["mask"], L["dtrue"]
    Pw = L["cam_p"] + SEE.cam_points(dtrue) @ L["cam_R"].T
    tok = (dtrue > 0.2) & (dtrue < 5)
    frame_off = np.median((pts - Pw)[table_px & ok & tok], axis=0)
    top_px = float(np.median(pts[table_px & ok][:, 2]) - top_true)
    top_est = float(L["top"])
    # the obstacles
    self_m = binary_dilation(robot_px, iterations=DIL_SELF)
    tgt_m = binary_dilation(mask, iterations=DIL_TGT)
    keep = ok & ~self_m & ~tgt_m
    P = pts[keep]
    pb, yaw = L["pb"], float(L["yaw"])
    P = P[crop(P, pb, yaw)]
    y_t = float(np.median(L["cloud"][:, 1]))
    zb = P[(np.abs(P[:, 1] - y_t) < BAND) & (np.abs(P[:, 2] - top_est) < 0.06), 2]
    h, b_ = np.histogram(zb, bins=np.arange(top_est - 0.06, top_est + 0.0601, 0.005))
    lvl = 0.5 * (b_[h.argmax()] + b_[h.argmax() + 1])
    level = float(np.median(zb[np.abs(zb - lvl) < 0.006]))
    sel = (np.abs(P[:, 2] - level) < EDGE_DZ) & (np.abs(P[:, 1] - y_t) < BAND)
    edge = float(np.percentile(P[sel, 0], 0.5)) if sel.sum() >= SEEN_N else None
    n_edge = int(((P[sel, 0] >= edge_true) & (P[sel, 0] < edge_true + SEEN_M)).sum())
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    edge_own = SEE.near_edge(pts, ok, top_est, ahead, float(np.array([-ahead[1], ahead[0]]) @ np.r_[np.median(L["cloud"][:, 0]), y_t]),
                             float(pb[:2] @ ahead + OWN))  # fmt: skip
    # the true edge's pixels: in the image, on the table, behind the robot
    ys = np.linspace(y_t - BAND, y_t + BAND, 61)
    E = np.c_[np.full_like(ys, edge_true + 0.005), ys, np.full_like(ys, top_true)]
    c = (E - L["cam_p"]) @ L["cam_R"]
    f = 0.5 * H / np.tan(np.radians(EP.HEAD_CAM_FOVY) / 2)
    u = np.round(c[:, 0] / -c[:, 2] * f + W / 2 - 0.5).astype(int)
    v = np.round(-c[:, 1] / -c[:, 2] * f + H / 2 - 0.5).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    on_robot = np.zeros_like(inside)
    on_table = np.zeros_like(inside)
    on_robot[inside] = robot_px[v[inside], u[inside]]
    on_table[inside] = table_px[v[inside], u[inside]]
    # the voxels against the truth's
    Pt = Pw[tok & ~robot_px & ~L["true"]]
    Pt = Pt[crop(Pt, pb, yaw)]
    Vp, Vt = voxels(P), voxels(Pt)
    zone = lambda V: V[((V[:, 0] + 0.5) * VOX > edge_true - 0.02) & ((V[:, 0] + 0.5) * VOX < edge_true + 0.25)
                       & (np.abs((V[:, 1] + 0.5) * VOX - y_t) < BAND) & (np.abs((V[:, 2] + 0.5) * VOX - top_true) < 0.10)]  # fmt: skip
    row = dict(ep=e, name=name, note=note, cam_dp_cm=round(cam_dp, 2), frame_off_cm=(100 * frame_off).round(2).tolist(),
               top_px_cm=round(100 * top_px, 2), top_est_cm=round(100 * (top_est - top_true), 2),
               level_cm=round(100 * (level - top_true), 2),
               edge_cm=None if edge is None else round(100 * (edge - edge_true), 2), n_edge=n_edge, seen=n_edge >= SEEN_N,
               edge_own_cm=None if edge_own is None else round(100 * (edge_own - edge_true), 2),
               edge_px_inside=int(inside.sum()), edge_px_table=int(on_table.sum()), edge_px_robot=int(on_robot.sum()),
               robot_px=int(robot_px.sum()), vox=len(Vp), vox_true=len(Vt), vox_prec=round(near(Vp, Vt), 3),
               vox_recall_zone=round(near(zone(Vt), Vp), 3), vox_zone_true=len(zone(Vt)))  # fmt: skip
    rows.append(row)
    np.savez_compressed(OUT / f"obst_{name}.npz", vox=((Vp + 0.5) * VOX).astype(np.float32), vox_size=VOX, pb=pb, yaw=yaw,
                        top_est=top_est, level=level, cloud=L["cloud"], robot_px=robot_px, table_px=table_px, keep=keep,
                        edge_true=edge_true, top_true=top_true)  # fmt: skip
    print(json.dumps(row), flush=True)
    json.dump(row, open(OUT / f"row_{e:02d}.json", "w"))
if ARG != "summary":
    sys.exit(0)
rows = [json.load(open(f)) for f in sorted(OUT.glob("row_*.json"))]


def med(k, rs):
    v = np.array([r[k] for r in rs if r[k] is not None], float)
    return dict(n=len(v), median_abs=round(float(np.median(np.abs(v))), 2), median=round(float(np.median(v)), 2),
                min=round(float(v.min()), 2), max=round(float(v.max()), 2))  # fmt: skip


seen = [r for r in rows if r["seen"]]
summ = dict(episodes=len(rows), seen=len(seen), top_px_cm=med("top_px_cm", rows), top_est_cm=med("top_est_cm", rows),
            level_cm=med("level_cm", rows),
            edge_cm_seen=med("edge_cm", seen), edge_cm_all=med("edge_cm", rows), edge_own_cm=med("edge_own_cm", rows),
            cam_dp_cm=med("cam_dp_cm", rows), vox_prec=med("vox_prec", rows), vox_recall_zone=med("vox_recall_zone", rows),
            edge_px_inside=med("edge_px_inside", rows), edge_px_table=med("edge_px_table", rows), edge_px_robot=med("edge_px_robot", rows))  # fmt: skip
print("SUMMARY", json.dumps(summ))
json.dump(dict(rows=rows, summary=summ), open(OUT / "step1.json", "w"), indent=1)

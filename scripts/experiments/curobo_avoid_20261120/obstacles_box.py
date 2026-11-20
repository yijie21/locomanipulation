"""curobo_avoid_20261120 step 1 for any pick task: the start look's points as obstacles (obstacles.py's voxels without its
TabletopGrasp table-edge measures).

    cd $REPO && SIMPLE_ROOT=$REPO/third_party/SIMPLE MUJOCO_GL=egl $PY -I obstacles_box.py <look dumps> <films> <clouds> <out> TASK EP

The robot's own pixels from the replica at the look's pose (the film's first frame) by segmentation, dilated DIL_SELF px; the
target: the bench's own cloud (<clouds>/<NAME>.npz, cloud_eyes_final: the look dump's mask is a table patch in 17 of
TabletopGrasp's 30 episodes) dilated in 3D by TGT_M; inside BOX (heading frame from the look's pelvis, z world); VOX voxels.
-> <out>/obst_<NAME>.npz (vox, vox_size, cloud = the target cloud), one JSON line
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experiments/simple_eval_20261112"))
import episodes as EP  # noqa: E402

LOOK, FILMS, CLOUDS, OUT = (Path(x) for x in sys.argv[1:5])
T, e = sys.argv[5], int(sys.argv[6])
OUT.mkdir(parents=True, exist_ok=True)
DIL_SELF, TGT_M, VOX = 6, 0.02, 0.02
BOX = ((-0.10, 0.90), (-0.60, 0.60), (0.30, 1.50))
name = f"{T}_L{e // 10}_e{e % 10}"
film = FILMS / f"{name}_perception_direct.npz"
if not film.exists():
    print(json.dumps(dict(ep=e, name=name, skip="no film")))
    sys.exit(0)
L = np.load(LOOK / f"{name}.npz")
info = EP.build(EP.episodes(T)[e], OUT / "scene" / name)
m = mujoco.MjModel.from_xml_path(info["dir"] + "/scene_locomani.xml")
d = mujoco.MjData(m)
z = np.load(film)
size = {0: 7, 1: 4, 2: 1, 3: 1}
mujoco.mj_resetData(m, d)
EP.pull_objects(m, d, info)
for n, a, t in zip([str(x) for x in z["jnt"]], z["jnt_adr"], z["jnt_type"]):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    if j >= 0:
        d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + size[int(t)]] = z["qpos"][0][a : a + size[int(t)]]
mujoco.mj_forward(m, d)
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
robot_px = geom & np.r_[m.body_rootid[m.geom_bodyid] == root, False][gid]
keep = L["ok"] & ~binary_dilation(robot_px, iterations=DIL_SELF)
P = L["pts"][keep]
pb, yaw = L["pb"], float(L["yaw"])
c, s = np.cos(yaw), np.sin(yaw)
rx, ry = c * (P[:, 0] - pb[0]) + s * (P[:, 1] - pb[1]), -s * (P[:, 0] - pb[0]) + c * (P[:, 1] - pb[1])
(x0, x1), (y0, y1), (z0, z1) = BOX
P = P[(rx > x0) & (rx < x1) & (ry > y0) & (ry < y1) & (P[:, 2] > z0) & (P[:, 2] < z1)]
tc = np.load(CLOUDS / f"{name}.npz")["pts"].astype(float)
P_t = P[cKDTree(tc).query(P)[0] >= TGT_M]
V = (np.unique(np.floor(P_t / VOX).astype(np.int64), axis=0) + 0.5) * VOX
np.savez_compressed(
    OUT / f"obst_{name}.npz", vox=V.astype(np.float32), vox_size=VOX, pb=pb, yaw=yaw, cloud=tc, robot_px=robot_px
)
print(json.dumps(dict(ep=e, name=name, cam_dp_cm=round(cam_dp, 2), robot_px=int(robot_px.sum()), pts=len(P), tgt_dropped=len(P) - len(P_t),
                      vox=len(V), tables={k: [round(float(v["top"]), 3), np.round(v["pos"], 3).tolist(), np.round(v["size"], 3).tolist()]
                                           for k, v in info["tables"].items()})), flush=True)  # fmt: skip

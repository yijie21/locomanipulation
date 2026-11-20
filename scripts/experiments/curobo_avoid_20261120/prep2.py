"""curobo_avoid_20261120 step 2a: each TabletopGrasp episode's state at the start of the pregrasp reach and its wanted pose,
for the cuRobo planner (which runs in SIMPLE's venv).

    cd $REPO && SIMPLE_ROOT=$REPO/third_party/SIMPLE $PY -I prep2.py <film npz dir> <trace rows dir> <step1 dir> <out> EP

From the direct film: the first frame of phase "pregrasp" (reach() holds GR.PRE_S there before the command) set into the
replica joint by joint; the robot's hinge joints by name, the pelvis pose, the right wrist-yaw link's pose and some arm
links' positions (world; a check of cuRobo's kinematics against the replica's). From the --trace row: the pregrasp's wanted
jaw centre (goal_mm) and rotation (rot_want), and the wrist-yaw link pose that puts the jaw centre there (OFF = the
REACH_HAND_OFF the bench ran with, in the wrist-yaw frame); the jaw path the reach took (the trace's pregrasp ticks).
-> <out>/prep_<NAME>.npz
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

FILMS, TRACE, S1, OUT = (Path(x) for x in sys.argv[1:5])
e = int(sys.argv[5])
OUT.mkdir(parents=True, exist_ok=True)
import os  # noqa: E402

T = os.environ.get("TASK", "TabletopGraspMP")
TRACE_FMT = os.environ.get("TRACE_FMT", "direct_{e}.jsonl")  # the --trace row of episode e in TRACE
OFF = np.array([0.0415, 0.0, 0.0])  # REACH_HAND_OFF of the bench runs
LINKS = ("right_wrist_yaw_link", "right_wrist_pitch_link", "right_elbow_link", "right_shoulder_yaw_link", "right_hand_palm_link",
         "torso_link")  # fmt: skip
name = f"{T}_L{e // 10}_e{e % 10}"
row = json.loads(open(TRACE / TRACE_FMT.format(e=e, task=T)).readline())
pre = [s for s in row["stages"] if s["stage"] == "pregrasp"]
if not pre:
    print(name, "no pregrasp (", row.get("why"), ")")
    sys.exit(0)
pre = pre[0]
z = np.load(FILMS / f"{name}_perception_direct.npz")
k = int(np.flatnonzero(z["phase"] == "pregrasp")[0])
info = EP.build(EP.episodes(T)[e], S1 / "scene" / name)
m = mujoco.MjModel.from_xml_path(info["dir"] + "/scene_locomani.xml")
d = mujoco.MjData(m)
size = {0: 7, 1: 4, 2: 1, 3: 1}
for n, a, t in zip([str(x) for x in z["jnt"]], z["jnt_adr"], z["jnt_type"]):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    if j >= 0:
        d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + size[int(t)]] = z["qpos"][k][a : a + size[int(t)]]
mujoco.mj_forward(m, d)
root = m.body_rootid[m.body("pelvis").id]
names, vals = [], []
for j in range(m.njnt):
    if m.body_rootid[m.jnt_bodyid[j]] == root and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        names.append(m.joint(j).name)
        vals.append(float(d.qpos[m.jnt_qposadr[j]]))
pel = m.body("pelvis").id
wr = m.body("right_wrist_yaw_link").id
Rw = d.xmat[wr].reshape(3, 3)
goal_jaw = np.asarray(pre["goal_mm"]) / 1000
R_want = np.asarray(pre["rot_want"])
goal_wrist = goal_jaw - R_want @ OFF
path = np.array([t[2] for t in row["trace"] if t[1] == "pregrasp"]) / 1000  # the jaw centre, world
tb = min(
    info["tables"].values(), key=lambda v: float(np.linalg.norm(np.asarray(v["pos"])[:2] - goal_jaw[:2]))
)  # the table under the grasp
links = {n: d.xpos[m.body(n).id].copy() for n in LINKS if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
np.savez_compressed(
    OUT / f"prep_{name}.npz", names=np.array(names), q=np.array(vals), pelvis_p=d.xpos[pel].copy(), pelvis_R=d.xmat[pel].reshape(3, 3).copy(),
    wrist_p=d.xpos[wr].copy(), wrist_R=Rw.copy(), jaw_p=d.xpos[wr] + Rw @ OFF, goal_jaw=goal_jaw, goal_R=R_want,
    goal_wrist=goal_wrist, path=path, pre_err_cm=pre["err_cm"], link_names=np.array(list(links)), link_p=np.array(list(links.values())),
    table_pos=np.asarray(tb["pos"], float), table_size=np.asarray(tb["size"], float), t_frame=float(z["t"][k]), OFF=OFF,
)  # fmt: skip
print(name, "frame", k, "t", round(float(z["t"][k]), 2), "joints", len(names), "pelvis", d.xpos[pel].round(3), "wrist->goal cm",
      round(100 * float(np.linalg.norm(goal_wrist - d.xpos[wr])), 1), "pregrasp err", pre["err_cm"], "path ticks", len(path))  # fmt: skip

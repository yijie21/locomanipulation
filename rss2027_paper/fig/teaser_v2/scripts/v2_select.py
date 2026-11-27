# notebook: skip render for paper Figure 1, not an experiment
"""v2_select.py: Figure 1 v2 episode choice (read-only over lmlroute's formal records).

Step 1: per controller, the decision set's "A fails, R succeeds" pairs; per task the grasp point height plan.goal z and
        the right shoulder height at the close (both arms), plus the standing shoulder height (first 10 Hz row of the
        first reach segment, before the reach command).
        Shoulder = origin of robot0_right_shoulder_pitch_link. Records keep no shoulder, so it is rebuilt from the 10 Hz
        row at or before the close (waits['close'].fire): pelvis yaw (W[:, 3]), roll / pitch from the pelvis-frame
        gravity (S[:, 58:61]), joints (S[:, :29], ScaleBFM order = the driver's qadr) on the host robot model (an MJB of
        this round's scene); placed by the recorded right hand point: shoulder = W_right_hand + FK(shoulder - hand).
        Checked on every row used: FK left hand - right hand against the recorded W left - right (and the torso point).
Step 2: keep the pairs whose grasp point is above the right shoulder at the close in both arms (A and R).
        ScaleBFM < 5 kept -> SONIC and HEFT the same way; still < 5 -> the fallback: the top 10 % grasp heights among
        the three controllers' R-only pairs (recorded as the fallback).
Step 3: in the kept set, the v1 rule: percentiles of close_dist_cm inside the kept set (kind 'mean'), score
        max(|pA - 50|, |pR - 50|), ascending; ties -> the smaller episode id.
"""

import json
import sys

import mujoco
import numpy as np

O = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126"
MJB = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/fig1_teaser_20261126/state/candle_117/alone.mjb"
SHORT = {"IThorBathroomLiftCandleFromFloor": "candle", "IThorKitchenApproachTableAndLiftApple": "apple",
         "IThorKitchenApproachTableAndLiftBowl": "bowl", "IThorKitchenApproachTableAndLiftSaltShaker": "salt",
         "IThorKitchenTakeStatueFromCabinet": "statue_cab", "IThorLivingRoomApproachTableAndLiftDumbbell": "dumbbell",
         "IThorLivingRoomApproachTableAndLiftWateringCan": "wcan"}
TASKS = list(SHORT)
DSET = {"scalebfm": ["candle", "apple", "salt", "statue_cab", "dumbbell", "wcan"],
        "sonic": ["apple", "salt", "statue_cab", "dumbbell", "wcan"],
        "heft": ["candle", "apple", "salt", "statue_cab", "dumbbell", "wcan"]}  # lrt_eval E1 (eval.txt)
ORDER = ("left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
         "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
         "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
         "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
         "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
         "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
         "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint")  # lml_selfmodel.SCALEBFM_ORDER
HAND_OFF = np.array([0.13, 0.0, 0.0])

M = mujoco.MjModel.from_binary_path(MJB)
D = mujoco.MjData(M)
jid = lambda n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "robot0_" + n)  # noqa: E731
bid = lambda n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot0_" + n)  # noqa: E731
QADR = np.array([M.jnt_qposadr[jid(n)] for n in ORDER])
RQ = int(M.jnt_qposadr[jid("floating_base_joint")])
B = {k: bid(n) for k, n in (("rw", "right_wrist_yaw_link"), ("lw", "left_wrist_yaw_link"), ("rs", "right_shoulder_pitch_link"),
                            ("ls", "left_shoulder_pitch_link"), ("torso", "torso_link"), ("pelvis", "pelvis"))}


def rot(yaw, g_p):
    v = -np.asarray(g_p, float)  # world z in the pelvis frame = Rp^T e_z = (-sp, sr cp, cr cp) for Rp = Rz Ry Rx
    v /= np.linalg.norm(v)
    p = np.arcsin(np.clip(-v[0], -1, 1))
    r = np.arctan2(v[1], v[2])
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def fk_row(S, W):
    R = rot(float(W[3]), S[58:61])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    D.qpos[:] = M.qpos0
    D.qpos[RQ:RQ + 3] = 0.0
    D.qpos[RQ + 3:RQ + 7] = q
    D.qpos[QADR] = S[:29]
    mujoco.mj_kinematics(M, D)
    hand = lambda b: D.xpos[b] + D.xmat[b].reshape(3, 3) @ HAND_OFF  # noqa: E731
    rh, lh = hand(B["rw"]), hand(B["lw"])
    W = W.astype(float)
    off = W[4:7] - rh  # place by the recorded right hand point
    out = dict(rs=D.xpos[B["rs"]] + off, ls=D.xpos[B["ls"]] + off, torso_pt=D.xpos[B["torso"]] + D.xmat[B["torso"]].reshape(3, 3) @ np.array([0, 0, 0.35]) + off,
               pelvis_fk=D.xpos[B["pelvis"]] + off)
    out["torso_tilt_deg"] = float(np.degrees(np.arccos(np.clip(D.xmat[B["torso"]].reshape(3, 3)[2, 2], -1, 1))))
    out["chk_left_cm"] = 100 * float(np.linalg.norm((lh + off) - W[7:10]))
    out["chk_torso_cm"] = 100 * float(np.linalg.norm(out["torso_pt"] - W[10:13]))
    return out


def rows_of(r, z):
    off = np.r_[0, np.cumsum(z["n_steps"])]
    fr = []
    for s in range(len(z["n_steps"])):
        f = r["d3"][s]["f_cmd"] + 5 * (np.arange(int(z["n_steps"][s])) - int(z["k0"][s]))
        fr += [(int(off[s] + i), int(x), s) for i, x in enumerate(f)]
    return fr


def arm_info(be, t, m, e):
    r = json.load(open(f"{O}/{be}/episodes/{t}/{m}/ep{e:+05d}.json"))
    z = np.load(f"{O}/{be}/segs/{t}/{m}/ep{e:+05d}.npz")
    fc = next((w["fire"] for w in r.get("waits", []) if w["key"] == "close"), None)
    info = dict(success=r["success"], reason=r.get("reason"), close_dist_cm=r.get("close_dist_cm"), close_frame=fc,
                goal=r["plan"]["goal"], host=r.get("host"))
    fr = rows_of(r, z)
    if fc is None or not fr:
        return info
    row, rf = max(((i, f) for i, f, _ in fr if f <= fc), key=lambda x: x[1])
    k = fk_row(z["S"][row], z["W"][row])
    i0 = min(fr, key=lambda x: x[1])[0]  # the first row (0.3 s before the first reach command: standing after the walk)
    k0 = fk_row(z["S"][i0], z["W"][i0])
    dv = z["W"][row, 4:7].astype(float) - k["rs"]
    info.update(arm_elev_deg=float(np.degrees(np.arctan2(dv[2], np.linalg.norm(dv[:2])))), torso_tilt_deg=k["torso_tilt_deg"],
                torso_tilt_stand_deg=k0["torso_tilt_deg"], shoulder_hand_cm=100 * float(np.linalg.norm(dv)))
    info.update(row_frame=rf, shoulder_z=float(k["rs"][2]), hand_z=float(z["W"][row, 6]), pelvis_z=float(z["W"][row, 2]),
                shoulder_z_stand=float(k0["rs"][2]), chk_left_cm=max(k["chk_left_cm"], k0["chk_left_cm"]),
                chk_torso_cm=max(k["chk_torso_cm"], k0["chk_torso_cm"]))
    return info


def q(v):
    v = np.asarray(v, float)
    return f"{np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]" if len(v) else "-"


ALL = {}
for be, tasks in DSET.items():
    P = []
    for t in TASKS:
        if SHORT[t] not in tasks:
            continue
        for e in range(30, 120):
            ra = json.load(open(f"{O}/{be}/episodes/{t}/alone/ep{e:+05d}.json"))
            rr = json.load(open(f"{O}/{be}/episodes/{t}/reachable/ep{e:+05d}.json"))
            if ra["success"] or not rr["success"]:
                continue
            a, rinfo = arm_info(be, t, "alone", e), arm_info(be, t, "reachable", e)
            P.append(dict(be=be, task=SHORT[t], task_full=t, ti=TASKS.index(t), ep=e, A=a, R=rinfo, goal_z=float(rr["plan"]["goal"][2])))
    ALL[be] = P
    chk = [max(p[g]["chk_left_cm"], 0) for p in P for g in "AR" if "chk_left_cm" in p[g]]
    chk_t = [p[g]["chk_torso_cm"] for p in P for g in "AR" if "chk_torso_cm" in p[g]]
    print(f"== {be}: decision set {tasks}; A fail & R success {len(P)}; FK check (left hand - right hand vs record) max "
          f"{max(chk):.4f} cm, torso point max {max(chk_t):.4f} cm; pairs without an A close {sum('shoulder_z' not in p['A'] for p in P)}")
    for t in tasks:
        S = [p for p in P if p["task"] == t]
        if not S:
            print(f"  {t:10s} n 0")
            continue
        g = lambda k, arm: [p[arm][k] for p in S if k in p[arm]]  # noqa: E731
        print(f"  {t:10s} n {len(S):3d} | goal z {q([p['goal_z'] for p in S])} | shoulder z at close A {q(g('shoulder_z', 'A'))} "
              f"R {q(g('shoulder_z', 'R'))} | standing {q(g('shoulder_z_stand', 'R'))} | hand z at close A {q(g('hand_z', 'A'))} R {q(g('hand_z', 'R'))} "
              f"| arm elev R {q(g('arm_elev_deg', 'R'))} deg | torso tilt R {q(g('torso_tilt_deg', 'R'))} deg | goal above shoulder (A and R) {sum(1 for p in S if 'shoulder_z' in p['A'] and p['goal_z'] > p['A']['shoulder_z'] and p['goal_z'] > p['R']['shoulder_z'])}")


def keep(p):
    return "shoulder_z" in p["A"] and "shoulder_z" in p["R"] and p["goal_z"] > p["A"]["shoulder_z"] and p["goal_z"] > p["R"]["shoulder_z"]


K = {be: [p for p in ALL[be] if keep(p)] for be in ALL}
print("step 2 kept:", {be: len(v) for be, v in K.items()})
if len(K["scalebfm"]) >= 5:
    pool, route = K["scalebfm"], "scalebfm, goal above the shoulder"
elif len(K["scalebfm"]) + len(K["sonic"]) + len(K["heft"]) >= 5:
    pool, route = K["scalebfm"] + K["sonic"] + K["heft"], "scalebfm < 5: + sonic, heft, goal above the shoulder"
else:
    allp = ALL["scalebfm"] + ALL["sonic"] + ALL["heft"]
    thr = np.quantile([p["goal_z"] for p in allp], 0.9)
    pool, route = [p for p in allp if p["goal_z"] >= thr], f"FALLBACK: top 10 % grasp heights of the three controllers' R-only pairs (goal z >= {thr:.4f}, of {len(allp)})"
print(f"step 2 pool: {len(pool)} ({route})")
for p in sorted(pool, key=lambda p: -p["goal_z"]):
    a, r = p["A"], p["R"]
    print(f"  {p['be']:8s} {p['task']:10s} ep {p['ep']:3d} goal z {p['goal_z']:.3f} | shoulder z A {a.get('shoulder_z', float('nan')):.3f} R {r.get('shoulder_z', float('nan')):.3f} "
          f"(standing {r.get('shoulder_z_stand', float('nan')):.3f}) | hand z A {a.get('hand_z', float('nan')):.3f} R {r.get('hand_z', float('nan')):.3f} | "
          f"pelvis z R {r.get('pelvis_z', float('nan')):.3f} | arm elev A {a.get('arm_elev_deg', float('nan')):.0f} R {r.get('arm_elev_deg', float('nan')):.0f} deg, torso tilt R {r.get('torso_tilt_deg', float('nan')):.0f} (stand {r.get('torso_tilt_stand_deg', float('nan')):.0f}) deg | close dist A {a['close_dist_cm']} R {r['close_dist_cm']:.3f} | A {a['reason']}")
V = [p for p in pool if p["A"]["close_dist_cm"] is not None and p["R"]["close_dist_cm"] is not None]
if V:
    VA, VR = [p["A"]["close_dist_cm"] for p in V], [p["R"]["close_dist_cm"] for p in V]
    pct = lambda v, x: 100.0 * (sum(u < x for u in v) + 0.5 * sum(u == x for u in v)) / len(v)  # noqa: E731
    for p in V:
        p["pA"], p["pR"] = pct(VA, p["A"]["close_dist_cm"]), pct(VR, p["R"]["close_dist_cm"])
        p["score"] = max(abs(p["pA"] - 50), abs(p["pR"] - 50))
    order = sorted(V, key=lambda p: (p["score"], p["ep"], ["scalebfm", "sonic", "heft"].index(p["be"]), p["ti"]))
    print(f"step 3: n {len(V)} (without an A close distance: {len(pool) - len(V)}); median close dist A {np.median(VA):.3f} R {np.median(VR):.3f}")
    for i, p in enumerate(order[:12]):
        print(f"  {i + 1:2d} score {p['score']:6.2f} {p['be']:8s} {p['task']:10s} ep {p['ep']:3d} dA {p['A']['close_dist_cm']:.3f} dR {p['R']['close_dist_cm']:.3f} "
              f"pA {p['pA']:.1f} pR {p['pR']:.1f} goal z {p['goal_z']:.3f} hosts {p['A']['host'][-5:]}/{p['R']['host'][-5:]}")
json.dump(dict(route=route, pool=[{k: v for k, v in p.items()} for p in pool]), open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1, default=float)

"""cap_check.py <scene xml> <film npz>: Figure 1 v3 step 1, read only. notebook: skip figure render, not an experiment.

How high does SIMPLE's ScaleBFM training data (selfmodel_simpledata_20261126 data3 + data3b, one-hand segments) reach,
and where is the standing shoulder?
  target z    the commanded hand point (13 cm ahead of wrist_yaw_link), world z (d3_collect: target = heading xy, world z)
  landed z    end_pos[acting hand] z, the measured hand point at the end label (world z)
  shoulder    origin of right_shoulder_pitch_link. Standing: (a) FK on the SIMPLE G1 at the command row of in-place,
              standing (pelvis_z -1), first-reach segments, placed by the recorded right hand point (v2_select's way);
              (b) the film's first frame (ScaleBFM standing in SIMPLE, before any command).
Kept segments: not fell / lost, settled_end (the SIMPLE val rule). Right hand (side 1) is reported separately.
"""

import sys

import mujoco
import numpy as np

L = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126"
ORDER = ("left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
         "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
         "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
         "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
         "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
         "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
         "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint")
HAND_OFF = np.array([0.13, 0.0, 0.0])
QS = (0.5, 5, 25, 50, 75, 95, 99.5)

M = mujoco.MjModel.from_xml_path(sys.argv[1])
D = mujoco.MjData(M)
jq = lambda n: int(M.jnt_qposadr[M.joint(n).id])  # noqa: E731
QADR = np.array([jq(n) for n in ORDER])
RQ = jq("floating_base_joint")
B = {k: M.body(n).id for k, n in (("rw", "right_wrist_yaw_link"), ("lw", "left_wrist_yaw_link"),
                                   ("rs", "right_shoulder_pitch_link"), ("pel", "pelvis"))}


def rot(yaw, g_p):
    v = -np.asarray(g_p, float)
    v /= np.linalg.norm(v)
    p = np.arcsin(np.clip(-v[0], -1, 1))
    r = np.arctan2(v[1], v[2])
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def fk_row(S, W):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot(float(W[3]), S[58:61]).ravel())
    D.qpos[:] = M.qpos0
    D.qpos[RQ:RQ + 3] = 0.0
    D.qpos[RQ + 3:RQ + 7] = q
    D.qpos[QADR] = S[:29]
    mujoco.mj_kinematics(M, D)
    rh = D.xpos[B["rw"]] + D.xmat[B["rw"]].reshape(3, 3) @ HAND_OFF
    lh = D.xpos[B["lw"]] + D.xmat[B["lw"]].reshape(3, 3) @ HAND_OFF
    off = W[4:7].astype(float) - rh
    return D.xpos[B["rs"]] + off, 100 * float(np.linalg.norm(lh + off - W[7:10]))


def q(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "n 0"
    return f"n {len(v):6d} | " + " ".join(f"p{p:g} {np.percentile(v, p):.3f}" for p in QS) + f" | max {v.max():.3f}"


def main():
    film = np.load(sys.argv[2], allow_pickle=False)
    D.qpos[:] = film["qpos"][0]
    mujoco.mj_kinematics(M, D)
    sh_film = float(D.xpos[B["rs"]][2])
    print(f"standing right shoulder z from the film's first frame (SIMPLE ScaleBFM, before any command): {sh_film:.4f} m;"
          f" pelvis z {D.xpos[B['pel']][2]:.4f}")
    pooled = {}
    for name in ("data3", "data3b"):
        z = np.load(f"{L}/{name}/data/all.npz")
        side, tg, ep = z["side"], z["target"].astype(float), z["end_pos"].astype(float)
        keep = (~z["fell"].astype(bool)) & (~z["lost"].astype(bool)) & z["settled_end"].astype(bool)
        stand = z["pelvis_z"] < 0
        hi = np.where(side == 1, 0, 1)
        land = ep[np.arange(len(side)), hi]
        n = z["n_steps"].astype(int)
        off = np.r_[0, np.cumsum(n)][:-1]
        cmd_row = off + z["k0"].astype(int)
        S, W = z["S"], z["W"]
        # standing shoulder at the command: in place, standing, first reach of the reset
        first = (z["chain"] == 0) & (~z["walked"].astype(bool)) & stand & keep
        idx = np.flatnonzero(first)
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, size=min(2000, len(idx)), replace=False)
        sh, chk = [], []
        for i in pick:
            s, c = fk_row(S[cmd_row[i]].astype(float), W[cmd_row[i]].astype(float))
            sh.append(s[2])
            chk.append(c)
        sh = np.array(sh)
        print(f"\n== {name}: {len(side)} one-hand segments, kept {keep.sum()} (not fell / lost, settled_end)")
        print(f"  standing shoulder z at the command (in place, standing, first reach; {len(pick)} drawn): {q(sh)};"
              f" FK check left hand max {max(chk):.4f} cm")
        for lab, sel in (("all kept", keep), ("right hand, kept", keep & (side == 1)),
                         ("right hand, kept, standing", keep & (side == 1) & stand)):
            print(f"  {lab}:")
            print(f"    target z: {q(tg[sel, 2])}")
            print(f"    landed z: {q(land[sel, 2])}")
        pooled.setdefault("tz", []).append(tg[keep & (side == 1)])
        pooled.setdefault("lz", []).append(land[keep & (side == 1)])
        pooled.setdefault("st", []).append(stand[keep & (side == 1)])
        pooled.setdefault("err", []).append(z["jaw_err_cm"][keep & (side == 1)].astype(float))
        pooled.setdefault("sh", []).append(sh)
        if name == "data3b":
            kind = z["kind"]
            for k in np.unique(kind):
                sel = keep & (side == 1) & (kind == k)
                print(f"    data3b kind {k} right kept: target z {q(tg[sel, 2])}")
    tg = np.concatenate(pooled["tz"])
    lz = np.concatenate(pooled["lz"])
    st = np.concatenate(pooled["st"])
    er = np.concatenate(pooled["err"])
    sh0 = float(np.median(np.concatenate(pooled["sh"])))
    print(f"\n== pooled data3 + data3b, right hand, kept: {len(tg)} segments; standing shoulder (median, FK) {sh0:.4f} m,"
          f" film {sh_film:.4f} m")
    print(f"  target z: {q(tg[:, 2])}")
    print(f"  landed z: {q(lz[:, 2])}")
    for dz in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        h = sh0 + dz
        a, b = tg[:, 2] > h, lz[:, 2] > h
        print(f"  above shoulder + {100 * dz:3.0f} cm ({h:.3f} m): target {a.sum():6d} ({100 * a.mean():5.2f} %),"
              f" landed {b.sum():6d} ({100 * b.mean():5.2f} %); standing among targets above {100 * st[a].mean() if a.any() else 0:.0f} %")
    print("  by target height band (right, kept): n, x p5-p95 (forward from the pelvis), y p5-p95, landed z median,"
          " direct error median (cm)")
    for lo in np.arange(0.95, 1.55, 0.05):
        sel = (tg[:, 2] >= lo) & (tg[:, 2] < lo + 0.05)
        if not sel.any():
            continue
        print(f"    z {lo:.2f}-{lo + 0.05:.2f}: n {sel.sum():5d} | x {np.percentile(tg[sel, 0], 5):.2f}..{np.percentile(tg[sel, 0], 95):.2f}"
              f" | y {np.percentile(tg[sel, 1], 5):+.2f}..{np.percentile(tg[sel, 1], 95):+.2f}"
              f" | landed z {np.median(lz[sel, 2]):.3f} | direct err {np.median(er[sel]):.1f} | standing {100 * st[sel].mean():.0f} %")


if __name__ == "__main__":
    main()

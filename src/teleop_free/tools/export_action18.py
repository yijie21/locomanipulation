#!/usr/bin/env python3
"""Build an 18-D action dataset from a normal (planner-mode) collection run.

    python export_action18.py <collect_dir> [--fps 25] [--out action18.npz]

The 18-D action, per LEGS (arXiv 2606.01458 sec. 3.2), is

    [ right wrist pos (3) | right wrist rotvec (3) | right grip (1)
      left  wrist pos (3) | left  wrist rotvec (3) | left  grip (1)
      base vx, vy, wz, height (4) ]                                  = 18

and every component is a COMMAND the expert actually issued, not a measured
state. The wrist poses are forward kinematics of the 17 upper-body joint
TARGETS that were streamed that frame (`agent_server.LAST_CMD["u17"]`), so
both arms are covered on every frame and the pose is exactly the one the
deploy was asked for; the grips are the hand joint targets; the base four
are the planner frame resolved into the pelvis frame. PoseSidecar writes all
of them per 50 Hz privileged frame, so this reads only
`<collect_dir>/pose_sidecar/*.jsonl` and needs no re-collection.

Base command, exactly as the deploy consumes it: `movement` and `facing` are
world-frame unit vectors and `speed` scales the movement; `height` is an
absolute pelvis height honoured only in IDEL_SQUAT (mode 4). This resolves
them to a pelvis-frame velocity and a yaw rate:

    vx, vy = speed * R(yaw)^T . movement      (0 when speed < 0)
    wz     = wrapped angle from the pelvis heading to the commanded `facing`
             (radians, an absolute heading error -- not divided by dt, which
             saturates; at deploy, facing = yaw + wz)
    h      = commanded height minus the standing pelvis height when the mode
             is IDEL_SQUAT, else 0 (so 0 means "stand", negative means
             "squat this far" -- a -1 sentinel would be a regression trap)

`--fps` resamples nearest, matching convert_to_hf_format's 50 -> 25.
"""

import argparse
import glob
import json
import math
import os

import numpy as np

MODE_SQUAT = 4
PELVIS_STAND = 0.793
# MuJoCo 29-vector slots for the 17-value planner order (primitives.upper_body_17):
# waist y/r/p then left/right-interleaved shoulder p,r,y, elbow, wrist r,p,y.
U17_TO_MJ = [12, 13, 14] + [j for i in range(7) for j in (15 + i, 22 + i)]


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotmat2rotvec(R):
    t = float(np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(t)
    if ang < 1e-8:
        return [0.0, 0.0, 0.0]
    k = ang / (2 * math.sin(ang))
    return [k * float(R[2, 1] - R[1, 2]), k * float(R[0, 2] - R[2, 0]), k * float(R[1, 0] - R[0, 1])]


def quat2rotvec(q):
    if q is None:
        return [0.0, 0.0, 0.0]
    w, x, y, z = q
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return [0.0, 0.0, 0.0]
    ang = 2 * math.atan2(n, w)
    ang = wrap(ang)
    return [ang * x / n, ang * y / n, ang * z / n]


def u17_to_mj29(u17):
    q = np.zeros(29, dtype=float)
    for k, slot in enumerate(U17_TO_MJ):
        q[slot] = float(u17[k])
    return q


def episode_actions(rows, ik, dt=0.02):
    """One sidecar file -> (T, 18) float32, plus the per-frame sim time."""
    A, T = [], []
    last = {"right": None, "left": None}
    for k, r in enumerate(rows):
        cmd = r.get("cmd") or {}
        hands = r.get("hands") or {}
        u17 = r.get("u17")
        if u17 is not None and ik is not None:
            q29 = u17_to_mj29(u17)
            for side in ("right", "left"):
                pos, rot = ik.fk(q29, side)
                last[side] = (list(map(float, pos)), rotmat2rotvec(rot))
        yaw = yaw_of(r["pelvis"][3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        mv = cmd.get("movement") or [0.0, 0.0, 0.0]
        sp = float(cmd.get("speed", -1.0))
        if sp < 0:  # no walk command on this frame
            vx = vy = 0.0
        else:
            vx = sp * (c * mv[0] + s * mv[1])
            vy = sp * (-s * mv[0] + c * mv[1])
        fac = cmd.get("facing") or [1.0, 0.0, 0.0]
        wz = float(wrap(math.atan2(fac[1], fac[0]) - yaw))
        h = float(cmd.get("height", -1.0))
        h = (h - PELVIS_STAND) if (int(cmd.get("mode", 0)) == MODE_SQUAT and h > 0) else 0.0
        row = []
        for side in ("right", "left"):
            p_r = last[side]
            row += p_r[0] if p_r else [0.0, 0.0, 0.0]
            row += p_r[1] if p_r else [0.0, 0.0, 0.0]
            hj = hands.get(side)
            row.append(float(hj[0]) if hj else 0.0)
        row += [vx, vy, wz, h]
        A.append(row)
        T.append(float(r.get("t", k * dt)))
    return np.asarray(A, dtype=np.float32), np.asarray(T, dtype=np.float64)


def resample(A, T, fps):
    if fps is None:
        return A, T
    if len(T) < 2:
        return A, T
    grid = np.arange(T[0], T[-1], 1.0 / fps)
    idx = np.searchsorted(T, grid).clip(0, len(T) - 1)
    left = (idx - 1).clip(0)
    take = np.where(np.abs(T[idx] - grid) <= np.abs(T[left] - grid), idx, left)
    return A[take], grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collect_dir")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ok-only", action="store_true", default=True)
    ap.add_argument(
        "--seq-out",
        default=None,
        metavar="NPZ",
        help="also write a seq -> 18-D table (no resampling) for convert_to_hf_format.py --a18 to join on",
    )
    a = ap.parse_args()

    man = {}
    for f in sorted(glob.glob(os.path.join(a.collect_dir, "manifest_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("produced"):
                man.setdefault(f, []).append(r)
    ok_rounds = {(os.path.basename(k), i): r["ok"] for k, v in man.items() for i, r in enumerate(v)}

    import sys as _sys

    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from arm_ik import ArmIK

    IK = ArmIK()
    eps, names, kept, skipped = [], [], 0, 0
    seq_all, a18_all = [], []
    for f in sorted(glob.glob(os.path.join(a.collect_dir, "pose_sidecar/*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        if len(rows) < 50 or "u17" not in rows[len(rows) // 2]:
            skipped += 1
            continue
        A, T = episode_actions(rows, IK)
        if a.seq_out:
            for rr, aa in zip(rows, A):
                sq = rr.get("seq")
                if sq is not None:
                    seq_all.append(int(sq))
                    a18_all.append(aa)
        A, T = resample(A, T, a.fps)
        eps.append(A)
        names.append(os.path.basename(f))
        kept += 1
    if not eps:
        raise SystemExit("no usable sidecar files (are the cmd/ik columns present?)")

    if a.seq_out:
        sq = np.asarray(seq_all, dtype=np.int64)
        aa = np.asarray(a18_all, dtype=np.float32)
        order = np.argsort(sq, kind="stable")
        sq, aa = sq[order], aa[order]
        keep = np.concatenate([[True], np.diff(sq) != 0])  # first row wins per seq
        np.savez_compressed(a.seq_out, seq=sq[keep], a18=aa[keep])
        print(f"seq table: {int(keep.sum())} unique seq -> {a.seq_out}")
    out = a.out or os.path.join(a.collect_dir, "action18.npz")
    np.savez_compressed(out, **{n: e for n, e in zip(names, eps)})
    allA = np.concatenate(eps, 0)
    lab = [
        "r_px",
        "r_py",
        "r_pz",
        "r_rx",
        "r_ry",
        "r_rz",
        "r_grip",
        "l_px",
        "l_py",
        "l_pz",
        "l_rx",
        "l_ry",
        "l_rz",
        "l_grip",
        "vx",
        "vy",
        "dyaw",
        "dz",
    ]
    print(f"{kept} episodes ({skipped} skipped), {len(allA)} frames @ {a.fps} fps -> {out}")
    print(f"{'dim':>6} {'min':>8} {'p50':>8} {'max':>8} {'std':>8}")
    for i, n in enumerate(lab):
        col = allA[:, i]
        print(f"{n:>6} {col.min():8.3f} {np.median(col):8.3f} {col.max():8.3f} {col.std():8.3f}")


if __name__ == "__main__":
    main()

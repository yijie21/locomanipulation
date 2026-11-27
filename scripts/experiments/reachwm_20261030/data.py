"""Reach world-model data (2026-10-30): loader, derived per-segment quantities and the deterministic splits for
gen_reach.py's segments (see SPEC.md).

Everything the models see is in the segment's heading frame (origin (start.x, start.y, 0), rotated by start.yaw, z the
world height). Per segment this builds the command vector u (12), the start features x0 (206), the delivered pose
(jaw position and rot6, heading frame, last row), the final joints and the final pelvis pose, and the split masks.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/data.py check          # derived error vs jaw_err_cm, split disjointness
"""

from __future__ import annotations

import glob
import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import env as E  # noqa: E402

# REACHWM_ROOT (2026-11-16): the data and model root of one controller's run (default outputs/reachwm_20261030, every
# ScaleBFM dataset); runs of other controllers in parallel each set their own
OUT = Path(os.environ.get("REACHWM_ROOT") or E.REPO / "outputs" / "reachwm_20261030")
DATA = OUT / "data"
# REACHWM_X0_REF=1 (2026-10-30 diagnosis): also give the start the controller's reference -- the jaw centre the motor
# commands at k0 (C, pelvis frame) and its offset from the measured jaw (pelvis frame). A chained reach plans from the
# previous reach's reference, not from where the hand actually is.
X0_REF = os.environ.get("REACHWM_X0_REF", "0") == "1"
# REACHWM_GOAL_REF=1 (2026-10-31, improvement 2): also give the model the reference plan reach() asks the controller to
# track (gen_reach.py records it): the final whole-body IK pose's joints, their change from the start joints, its pelvis
# position and rotation in the heading frame, and the plan's duration. The model then learns the tracking error only.
GOAL_REF = os.environ.get("REACHWM_GOAL_REF", "0") == "1"
# REACHWM_APPROACH=1 (2026-11-03, walk then reach): the start can lie before an approach walk (gen_reach.py
# REACH_APPROACH). The walk asked (forward, left, sin turn, cos turn - 1, start frame) is appended to x0, zero for the
# datasets without a walk; there the command and the plan are the ones the robot can know before walking (the reach
# through the commanded walk, the plan from the pre-walk posture). The held-out bands and the edge set are judged on the
# reach command in the frame the robot believes it stands in after the walk (target_b; = target without a walk).
APPROACH = os.environ.get("REACHWM_APPROACH", "0") == "1"
U_DIM, X0_DIM = 12, 206 + (6 if X0_REF else 0) + (68 if GOAL_REF else 0) + (4 if APPROACH else 0)
# REACHWM_SUBSET=first_still (2026-10-31, canonical starts): every split keeps only the first reach of a reset that did
# not walk -- the start is the reset's standing pose after a 0.3 s hold.
SUBSET = os.environ.get("REACHWM_SUBSET", "")
# REACHWM_TRAIN_HELD=1 (2026-11-06, a model for use, not for the bench): the bands and the edge set (37 % of the data) train too.
# They were held out to measure generalisation, and the bands sit where grasps live: commands 0.48-0.54 m ahead, 0.40-0.48 m
# high, pelvis 0.60-0.64 m. val and test_iid stay out; the band and edge splits stop being unseen.
TRAIN_HELD = os.environ.get("REACHWM_TRAIN_HELD", "0") == "1"
Q, GRAV, HEIGHT = slice(0, 29), slice(58, 61), 64  # S layout (env.py)
# REACHWM_QFIX=1 (2026-11-17, found by the independent review of sonic_g1_selfmodel_20261116): GOAL_REF's "plan joints
# minus start joints" subtracted S's joints, which are in ScaleBFM's metadata order (runtime.Sim reads dof_pos through
# sim.qadr, every controller in this harness), from the plan's joints, which are in MuJoCo order (Motor.kp.qadr): 27 of
# the 29 positions paired two different joints. With the switch the start joints are put in MuJoCo order first. Both
# raw vectors were in x0 already, so the old feature only scrambled a linear combination. Off by default: every model
# trained before keeps its own feature, plan_invert.plan_features reads the same switch, and wrist_invert.export stamps
# the pickle (.qfix) so check_qfix can refuse a model fed the other feature.
QFIX = os.environ.get("REACHWM_QFIX", "0") == "1"
# REACHWM_FDVEL=1 (2026-11-26, selfmodel_fdvel_20261126): x0's velocity inputs from the record's own positions instead
# of the simulator's velocity readings (HumanoidArena's joint_vel and root angular velocity do not match the motion of a
# standing robot; selfmodel_xfer_20261125). In each of x0's three S rows (S[k0], S[k0-1], S[k0-2]):
#   joint velocities (S 29:58)                       backward difference of the joint angles (S 0:29) / ROW_S
#   pelvis angular velocity (S 61:64, pelvis frame)  log(R_prev^T R) / ROW_S, R = Rz(yaw) Ry(pitch) Rx(roll) from W's yaw
#                                                    and S's gravity in the pelvis frame (as models.pelvis_R)
# ROW_S is the collectors' row interval (a row every TICKS = 5 frames at 50 Hz; C's time since the command steps by it).
# S holds no pelvis linear velocity (height only), and nothing else in x0 or u is a velocity. The difference of S[k0-2]
# reads the row before it inside the same segment: the collectors write three rows (PRE_S = 0.3 s) before the command,
# so k0 = 3; like the history rows themselves, nothing is filled in when that row is missing (asserted). LocoManiLab's
# and every other host's record go through the same function (derive -> fdvel_x0). Off: x0 is bit-identical to before.
FDVEL = os.environ.get("REACHWM_FDVEL", "0") == "1"
ROW_S = 0.1
S_DIM, DQ, ANGVEL = 65, slice(29, 58), slice(61, 64)  # S layout (env.py)
# x0's velocity columns, S[k0], S[k0-1], S[k0-2] in turn: 3 x 29 joint velocities and 3 x 3 angular velocities
X0_DQ_COLS = np.concatenate([np.arange(s * S_DIM + DQ.start, s * S_DIM + DQ.stop) for s in range(3)])
X0_ANGVEL_COLS = np.concatenate([np.arange(s * S_DIM + ANGVEL.start, s * S_DIM + ANGVEL.stop) for s in range(3)])


def pelvis_rot(yaw: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Pelvis rotations (N, 3, 3) from the yaw (W[:, 3], env.yaw_of) and gravity in the pelvis frame (S[:, GRAV]):
    Rz(yaw) Ry(pitch) Rx(roll), exact for env.yaw_of's yaw (models.pelvis_R; the X0_REF block above)."""
    g = np.asarray(g, np.float64)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    th, ph, yw = np.arcsin(np.clip(g[:, 0], -1, 1)), np.arctan2(-g[:, 1], -g[:, 2]), np.asarray(yaw, np.float64)
    cz, sz, cy, sy, cx, sx = np.cos(yw), np.sin(yw), np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
    return np.stack([np.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], -1),
                     np.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], -1),
                     np.stack([-sy, cy * sx, cy * cx], -1)], -2)  # fmt: skip


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vectors (N, 3) of rotations (N, 3, 3) below pi: angle from atan2 (accurate near 0), axis from the skew
    part."""
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    s2 = np.linalg.norm(v, axis=1)  # 2 sin(angle)
    th = np.arctan2(s2, np.trace(R, axis1=1, axis2=2) - 1.0)
    k = np.where(s2 > 1e-12, th / np.maximum(s2, 1e-300), 0.5)
    return v * k[:, None]


def fd_rows(S: np.ndarray, W: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Velocities at global rows `rows` from the row before each: joint angle differences / ROW_S (N, 29) and the
    pelvis-frame angular velocity log(R_prev^T R) / ROW_S (N, 3)."""
    rows = np.asarray(rows)
    dq = (S[rows, Q].astype(np.float64) - S[rows - 1, Q]) / ROW_S
    Rp, Rc = pelvis_rot(W[rows - 1, 3], S[rows - 1, GRAV]), pelvis_rot(W[rows, 3], S[rows, GRAV])
    w = so3_log(np.einsum("nji,njk->nik", Rp, Rc)) / ROW_S
    return dq, w


def fdvel_x0(D: dict, x0: np.ndarray) -> np.ndarray:
    """REACHWM_FDVEL's x0: `x0` (D's segments) with its 96 velocity columns (X0_DQ_COLS, X0_ANGVEL_COLS) replaced by
    fd_rows of S[k0], S[k0-1], S[k0-2]; every other column unchanged."""
    n, k0 = D["n_steps"].astype(int), D["k0"].astype(int)
    assert k0.min() >= 3, f"the difference of S[k0-2] needs the row before it in the segment (k0 >= 3), min k0 {k0.min()}"
    r0 = np.r_[0, np.cumsum(n)[:-1]] + k0
    out = np.array(x0, np.float32, copy=True)
    for s in range(3):
        dq, w = fd_rows(D["S"], D["W"], r0 - s)
        out[:, s * S_DIM + DQ.start : s * S_DIM + DQ.stop] = dq
        out[:, s * S_DIM + ANGVEL.start : s * S_DIM + ANGVEL.stop] = w
    return out


def s_to_mujoco() -> np.ndarray:
    """Index array: S's 29 joints (ScaleBFM metadata order) -> MuJoCo order (locomani.primitives.g1_joints)."""
    import json

    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # env.py put REPO/src on the path

    names = json.load(open(E.REPO / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"))["joint_names"]
    idx = np.array([names.index(n) for n in MUJOCO_JOINT_NAMES])
    assert len(idx) == 29 and sorted(idx) == list(range(29))
    return idx


S2MJ = s_to_mujoco() if QFIX else None


def start_joints_like_plan(q0: np.ndarray) -> np.ndarray:
    """Start joints (S order, rows or one vector) in the order the plan's joints use: MuJoCo under QFIX, else unchanged."""
    return np.asarray(q0)[..., S2MJ] if QFIX else q0


def check_qfix(model) -> None:
    """A pickled reach model (wrist_invert.export) was trained with one form of the joint feature; refuse the other."""
    trained = bool(getattr(model, "qfix", False))
    assert trained == QFIX, (
        f"model trained with REACHWM_QFIX={int(trained)}, this process runs REACHWM_QFIX={int(QFIX)}"
    )


FRACS = (0.1, 0.3, 1.0)
SEG_KEYS = ("n_steps", "k0", "side", "dyaw", "pitch", "pelvis_z", "walked", "chain", "fell", "ok", "jaw_err_cm",
            "att_err_deg", "ik_err_cm", "reach_s", "reset", "target", "rot6", "start")  # fmt: skip


def files(spec=None) -> list[Path]:
    """spec: None (all.npz if present, else every chunk), a list of paths, or a glob string (several: comma-separated)."""
    if isinstance(spec, str) and "," in spec:
        return [p for g in spec.split(",") for p in files(g)]
    if spec is None:
        a = DATA / "all.npz"
        return [a] if a.exists() else sorted(DATA.glob("w[0-9]*_[0-9]*.npz"))
    if isinstance(spec, (str, Path)):
        return [Path(p) for p in sorted(glob.glob(str(spec)))]
    return [Path(p) for p in spec]


def hv(v: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """World-frame vectors (N, 3) into the heading frame of per-row yaw: Rz(yaw)^T v."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)


def hash_bucket(reset: np.ndarray, salt: str, mod: int) -> np.ndarray:
    return np.array([zlib.crc32(f"{salt}{int(r)}".encode()) % mod for r in reset])


def load(spec=None) -> dict:
    """Concatenate the chunks and add the derived arrays. Per-step arrays are S, W, J, C (+ O odometry); per-segment
    ones are SEG_KEYS plus the derived ones below."""
    zs = []
    for p in files(spec):
        try:
            z = dict(np.load(p))  # each key once; a chunk still being written fails here and is skipped
        except Exception as e:  # noqa: BLE001
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        if APPROACH:
            z.setdefault("walk", np.zeros((len(z["n_steps"]), 3), np.float32))
            z.setdefault("target_b", z["target"])
        zs.append(z)
    assert zs, "no data chunks"
    keys = set.intersection(*(set(z) for z in zs))  # datasets of different dates: only the arrays they all have
    D = {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if k in keys}
    D["files"] = [str(p) for p in files(spec)]
    derive(D)
    return D


def derive(D: dict) -> None:
    n = D["n_steps"].astype(int)
    N = len(n)
    so = np.r_[0, np.cumsum(n)[:-1]]
    last, k0 = so + n - 1, so + D["k0"].astype(int)
    side = D["side"].astype(np.float32)
    start, yaw0 = D["start"].astype(np.float64), D["start"][:, 3].astype(np.float64)
    org = np.c_[start[:, :2], np.zeros(N)]
    seg = np.repeat(np.arange(N), n)
    W, J, S = D["W"].astype(np.float64), D["J"].astype(np.float64), D["S"]
    right = side > 0

    def acting(rows):
        """Acting jaw position (world) and rotation (first two columns, world) at the given global rows."""
        pos = np.where(right[:, None], W[rows, 4:7], W[rows, 7:10])
        return pos, J[rows]

    def head(pos, rot, yaw, o):
        return hv(pos - o, yaw), np.concatenate([hv(rot[:, :3], yaw), hv(rot[:, 3:], yaw)], -1)

    pf, rf = head(*acting(last), yaw0, org)
    p0, r0 = head(*acting(k0), yaw0, org)
    D["pos_f"], D["rot_f"] = pf.astype(np.float32), rf.astype(np.float32)
    D["jaw0"], D["rot0"] = p0.astype(np.float32), r0.astype(np.float32)
    # final pelvis pose in the heading frame (position, yaw change, gravity in the pelvis frame) and joints
    D["pel_f"] = hv(W[last, :3] - org, yaw0).astype(np.float32)
    D["dyaw_f"] = E_wrap(W[last, 3] - yaw0).astype(np.float32)
    D["g_f"] = S[last, GRAV].astype(np.float32)
    D["q_f"], D["q_0"] = S[last, Q].astype(np.float32), S[k0, Q].astype(np.float32)
    D["start_z"] = start[:, 2].astype(np.float32)
    # delivered error against the command
    D["err_cm"] = (100 * np.linalg.norm(D["pos_f"] - D["target"], axis=1)).astype(np.float32)
    # settled: max acting-jaw speed over the last 0.5 s (5 intervals of 0.1 s) below 2 cm/s
    jp = np.where(right[seg, None], W[:, 4:7], W[:, 7:10])
    sp = np.linalg.norm(jp[1:] - jp[:-1], axis=1) / 0.1
    sp = np.r_[sp, 0.0]
    D["settled"] = np.array([sp[last[i] - 5 : last[i]].max() < 0.02 if n[i] > 5 else False for i in range(N)]) & ~D[
        "fell"
    ].astype(bool)
    # command vector and start features
    stand = (D["pelvis_z"] == -1).astype(np.float32)
    D["u"] = np.c_[side, D["target"], D["rot6"], stand, np.where(stand > 0, 0.0, D["pelvis_z"])].astype(np.float32)
    D["x0"] = np.c_[S[k0], S[k0 - 1], S[k0 - 2], D["jaw0"], D["rot0"], D["walked"], D["chain"]].astype(np.float32)
    if X0_REF:
        g = S[k0][:, GRAV].astype(np.float64)
        th, ph = np.arcsin(np.clip(g[:, 0], -1, 1)), np.arctan2(-g[:, 1], -g[:, 2])
        yw = W[k0, 3]
        cz, sz, cy, sy, cx, sx = np.cos(yw), np.sin(yw), np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
        Rp = np.stack([np.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], -1),
                       np.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], -1),
                       np.stack([-sy, cy * sx, cy * cx], -1)], -2)  # fmt: skip
        jaw_w = np.where(right[:, None], W[k0, 4:7], W[k0, 7:10])
        meas = np.einsum("nji,nj->ni", Rp, jaw_w - W[k0, :3])
        ref = D["C"][k0, :3].astype(np.float64)
        D["x0"] = np.c_[D["x0"], ref, ref - meas].astype(np.float32)
    if GOAL_REF:
        w, x, y, z = (D["ref_pel"][:, 3 + i].astype(np.float64) for i in range(4))
        c0 = np.stack(
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1
        )  # quat wxyz -> columns 0, 1
        c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
        rq = D["ref_q"].astype(np.float32)
        D["x0"] = np.c_[D["x0"], rq, rq - start_joints_like_plan(D["q_0"]), hv(D["ref_pel"][:, :3] - org, yaw0), hv(c0, yaw0), hv(c1, yaw0),
                        D["ref_s"]].astype(np.float32)  # fmt: skip
    if APPROACH:
        D["x0"] = np.c_[D["x0"], walk_features(D["walk"])].astype(np.float32)
    if FDVEL:
        D["x0"] = fdvel_x0(D, D["x0"])
    # odometry of every transition t -> t+1 (zero on each segment's last row), world pelvis rows
    O = np.zeros((len(W), 4), np.float32)
    d = hv(W[1:, :3] - W[:-1, :3], W[:-1, 3])
    O[:-1, :3] = d
    O[:-1, 3] = E_wrap(W[1:, 3] - W[:-1, 3])
    O[last] = 0.0
    D["O"] = O
    D["so"] = so
    # splits
    t, dy, pz = D["target_b" if APPROACH else "target"], np.degrees(D["dyaw"]), D["pelvis_z"]
    D["band_x"] = (t[:, 0] >= 0.48) & (t[:, 0] < 0.54)
    D["band_z"] = (t[:, 2] >= 0.40) & (t[:, 2] < 0.48)
    D["band_yaw"] = (dy >= 30) & (dy < 42)
    D["band_pel"] = (pz >= 0.60) & (pz < 0.64)
    D["edge"] = (t[:, 0] > 0.85) | (t[:, 2] < 0.08) | (side * t[:, 1] < -0.50)
    D["bucket"] = hash_bucket(D["reset"], "reach", 10)
    D["frac_rank"] = hash_bucket(D["reset"], "frac", 1 << 20) / float(1 << 20)


def E_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def walk_features(walk: np.ndarray) -> np.ndarray:
    """REACHWM_APPROACH's x0 tail for walks (N, 3) = (forward, left, turn): forward, left, sin turn, cos turn - 1."""
    w = np.asarray(walk, np.float64).reshape(-1, 3)
    return np.c_[w[:, 0], w[:, 1], np.sin(w[:, 2]), np.cos(w[:, 2]) - 1.0].astype(np.float32)


def splits(D: dict, frac: float = 1.0) -> dict[str, np.ndarray]:
    """Index arrays: train (nested by frac), val, test_iid, the four bands and the edge set (bands/edge may overlap)."""
    held = D["band_x"] | D["band_z"] | D["band_yaw"] | D["band_pel"] | D["edge"]
    keep = subset_mask(D)
    rest = ~held & keep
    out = dict(
        train=np.where((keep if TRAIN_HELD else rest) & (D["bucket"] < 8) & (D["frac_rank"] < frac + 1e-9))[0],
        val=np.where(rest & (D["bucket"] == 8))[0],
        test_iid=np.where(rest & (D["bucket"] == 9))[0],
    )
    for k in ("band_x", "band_z", "band_yaw", "band_pel", "edge"):
        out[k] = np.where(D[k] & keep)[0]
    return out


def subset_mask(D: dict) -> np.ndarray:
    if SUBSET == "first_still":
        return (D["chain"] == 0) & (D["walked"] == 0)
    assert SUBSET == "", f"unknown REACHWM_SUBSET {SUBSET}"
    return np.ones(len(D["n_steps"]), bool)


def pad(D: dict) -> dict:
    """Per-segment padded step tensors for the step models: S (N, L, 65), O (N, L, 4), tc (N, L) = time since the
    command at the row AFTER each transition (the action of the step t -> t+1), -1 before the command."""
    n, so = D["n_steps"].astype(int), D["so"]
    L = int(n.max())
    N = len(n)
    Sp, Op, Tp = np.zeros((N, L, 65), np.float32), np.zeros((N, L, 4), np.float32), np.full((N, L), -1.0, np.float32)
    tc = D["C"][:, 3]
    for i in range(N):
        a, m = so[i], n[i]
        Sp[i, :m], Op[i, :m] = D["S"][a : a + m], D["O"][a : a + m]
        Sp[i, m:] = Sp[i, m - 1]
        Tp[i, : m - 1] = tc[a + 1 : a + m]
    return dict(S=Sp, O=Op, tc=Tp)


def check(spec=None) -> None:
    D = load(spec)
    n = len(D["n_steps"])
    print(f"{n} segments from {len(D['files'])} files, {int(D['n_steps'].sum())} steps, fell {int(D['fell'].sum())}, "
          f"settled {int(D['settled'].sum())}")  # fmt: skip
    ok = D["settled"] & (D["jaw_err_cm"] >= 0)
    diff = D["err_cm"][ok] - D["jaw_err_cm"][ok]
    print(f"derived final error - jaw_err_cm on {int(ok.sum())} settled segments: mean {diff.mean():+.3f} cm, "
          f"median |d| {np.median(np.abs(diff)):.3f}, p90 |d| {np.percentile(np.abs(diff), 90):.3f}, "
          f"p99 |d| {np.percentile(np.abs(diff), 99):.3f}, max |d| {np.abs(diff).max():.3f}")  # fmt: skip
    print("seg  side  target(x,y,z)           derived_cm  jaw_err_cm  settled")
    for i in np.where(ok)[0][:8]:
        t = D["target"][i]
        print(f"{i:4d} {int(D['side'][i]):+d}  ({t[0]:.2f},{t[1]:+.2f},{t[2]:.2f})  {D['err_cm'][i]:9.2f}  "
              f"{D['jaw_err_cm'][i]:9.2f}  {bool(D['settled'][i])}")  # fmt: skip
    bad = np.where(ok & (np.abs(D["err_cm"] - D["jaw_err_cm"]) > 0.3))[0]
    print(f"settled segments off by more than 0.3 cm: {len(bad)} of {int(ok.sum())}")
    for i in bad[:5]:
        print(f"  seg {i}: derived {D['err_cm'][i]:.2f} vs {D['jaw_err_cm'][i]:.2f} cm, ok={bool(D['ok'][i])}, "
              f"reach_s {D['reach_s'][i]:.2f}, n_steps {D['n_steps'][i]}")  # fmt: skip
    sp = splits(D, 1.0)
    held = D["band_x"] | D["band_z"] | D["band_yaw"] | D["band_pel"] | D["edge"]
    print("split sizes:", {k: len(v) for k, v in sp.items()})
    for f in FRACS:
        print(f"  train frac {f}: {len(splits(D, f)['train'])}")
    pairs = [(a, b) for a in ("train", "val", "test_iid") for b in ("train", "val", "test_iid") if a < b]
    print("disjoint train/val/test_iid:", all(len(np.intersect1d(sp[a], sp[b])) == 0 for a, b in pairs))
    print("train segments inside any band or edge:", int(held[sp["train"]].sum()), " val:", int(held[sp["val"]].sum()),
          " test_iid:", int(held[sp["test_iid"]].sum()))  # fmt: skip
    rs = lambda k: set(D["reset"][sp[k]].tolist())  # noqa: E731
    print("reset ids shared train/val:", len(rs("train") & rs("val")), " train/test_iid:", len(rs("train") & rs("test_iid")),
          " val/test_iid:", len(rs("val") & rs("test_iid")))  # fmt: skip
    nested = set(splits(D, 0.1)["train"]) <= set(splits(D, 0.3)["train"]) <= set(sp["train"])
    print("nested fractions:", nested)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(sys.argv[2] if len(sys.argv) > 2 else None)

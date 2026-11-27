"""selfmodel_jref 38.10.3: films with the motor's reference (simple_bench.py --film-ref) -> a dataset that
selfmodel_via_20261120/via_point.py trains and evaluates as it is (table.npz + arm.npz in via_model.build's format).

    cd $REPO && SIMPLE_ROOT=... PYTHONPATH=src $PY -I scripts/experiments/simple_eval_20261112/jref_data.py \
        build <out dir> <scene dir> <film dir> <rows.jsonl ...> [--subset all|seen|held] [--holdout 3]
    ... jref_data.py range <check dataset dir> <training dataset dir> [<training dataset dir> ...]

Two sources. --collect rows (stage "jref", one per sample, phase jref<k> of the film), and the bench's J arm rerun with
--film-ref (stage "pregrasp" with joint_src, phase "pregrasp"). One row per reach that has a plan and no fall.

The reference is the motor's own (Motor.reach_joints: the plan's arm joints, the start's gap faded out, the rest of the
body held), its spheres by the scene robot's kinematics with the acting hand at its finger COMMAND (the hand the
controller is told to hold; the 38.10.2 measure used the plan's closed hand and so had a ~4.8 cm floor). The reference
pelvis is carried onto the true one at the command (the odometry's belief before the reach is not the controller's doing);
the real spheres are the simulated robot's with its real fingers. Everything is in the heading frame of the true pelvis at
the command, x ahead, as via_model. Times: the real sweep resampled at 10 Hz from the command to the end of the reach's phase
(gen_reach.POST_S after it); the reference over its own duration (reach_joints' dur_s).

x is via_model.build's layout with no via points (their fields 0), speed 0.15, standing, the right hand; the goal and its
attitude are the reference's last pose (the plan's end). Splits: --collect rows by layout, the training layouts dealt in
crc32 order to buckets 0-9 (8 each, the same whichever rows a build gets: a dataset per task joins into the same split);
route = layout, so the bootstrap resamples layouts. Bench rows: bucket 9, route -1
(one path each), and --subset keeps all of them, those of the layouts the data saw, or those it left out.
range: the check's joints at the command against the training paths' 0.5-99.5 % band, joint by joint (29), as
bench_via_table.py does for the via data.

--frame root (selfmodel_rootframe_20261125; the default, --frame true, is the above, unchanged): the motor's reference as it
is (its own root ref[0:7], not carried onto the true pelvis), everything in the heading frame of that root at the command
(origin at the root's xy, x ahead, heights as they are), the labels the real spheres minus the reference's as before, and
no simulator truth in x (Belief.of):
  S3  as it is (the measured joints and joint velocities, the gravity reading, the gyro, the leg odometry's pelvis height);
  W   replaced by the believed pelvis (xy and heading: the stance anchor, which the reference root is; height: S3's) and
      the right jaw centre by the measured joints from it, with the gravity reading's tilt;
  J   replaced by the wrist's attitude the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))
sys.path.insert(0, str(Path.cwd() / "src"))
import via_model as VM  # noqa: E402
from arm_fk import FINGERS, ArmFK  # noqa: E402

TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")  # simple_bench.PICK_TASKS
SPEED = 0.15


def layout(task: str, level, index) -> int:
    return 30 * TASKS.index(task) + 10 * int(level) + int(index)


def held_out(task: str, level, index, k: int) -> bool:  # simple_bench.held_out
    return zlib.crc32(f"{task}:{level}:{index}".encode()) % k == 0


def quat_R(q) -> np.ndarray:
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def r6m(r) -> np.ndarray:
    return VM.r6m(r)


def yaw_of(R) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


class Belief:
    """--frame root: what the robot itself knows of its pelvis and right jaw at the command.

    S3 is kept as it is: the measured joints and joint velocities, the gravity reading, the gyro and the pelvis height
    the controller is told (simple_bench.py runs BFM_ROOT=odo: env.self_state's height is leg_odo.OdoRoot's, the stance
    anchor plus the leg odometry since; not the simulator's). The believed pelvis: xy and heading the stance anchor
    (the row's bel: after the walk the belief the goals are placed by stops, and the motor anchors its reference root
    there, so the root is bel.pt, yt), height S3's, tilt the gravity reading; the jaw centre and the wrist by the scene
    robot's kinematics from it with the measured joints."""

    def __init__(self, fk: ArmFK):
        import mujoco

        self.fk, self.mj = fk, mujoco

    @staticmethod
    def level(grav) -> np.ndarray:  # leg_odo.level: body -> level frame (rows: forward on the floor, left, up)
        up = -np.asarray(grav, float)
        up /= np.linalg.norm(up)
        x = np.array([1.0, 0.0, 0.0]) - up[0] * up
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(up, x), up])

    def of(self, S3, root) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S3 (3, 65) as recorded, root the reference root at the command (7) -> S3, the believed W (pelvis xyz, yaw, right
        jaw centre) and J (the right wrist's first two columns), world."""
        S3 = np.asarray(S3, float)
        yaw = yaw_of(quat_R(root[3:7]))
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        p = np.r_[root[0], root[1], S3[0, 64]]
        q = np.zeros(4)
        self.mj.mju_mat2Quat(q, (Rz @ self.level(S3[0, 58:61])).reshape(-1))
        _, jaw = self.fk.spheres("right", p, q, S3[0, :29], "meta", np.zeros(len(FINGERS)))  # no finger moves the jaw
        Rw = self.fk.d.xmat[self.fk.wrist["right"]].reshape(3, 3)
        return S3, np.r_[p, yaw, jaw], np.r_[Rw[:, 0], Rw[:, 1]]


def samples(rows_files: list[str]):
    """-> (row, stage, phase, t_cmd, dur) for every usable reach."""
    for f in rows_files:
        for line in open(f):
            r = json.loads(line)
            for s in r["stages"]:
                if s["stage"] == "jref" and "skip" not in s and s.get("fell") is None and "t_cmd" in s:
                    yield r, s, f"jref{s['k']}", float(s["t_cmd"]), float(s["dur_s"])
                elif s["stage"] == "pregrasp" and s.get("joint_src") and "t_cmd" in s and r.get("fell") is None:
                    yield r, s, "pregrasp", float(s["t_cmd"]), float(s["joint_dur_s"])


def build(a) -> None:
    fk = ArmFK(Path(a.scene))
    bel = Belief(fk) if a.frame == "root" else None
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    link = list(fk.links)
    key = [link.index(x) for x in VM.KEY_LINKS]
    u = np.linspace(0.0, 1.0, VM.Q)
    Xall, rows, films = [], dict(x=[], ref=[], off=[], a=[], b=[], reset=[], route=[], held=[], sid=[], k=[], S0=[]), {}
    n_skip = dict(film=0, short=0, subset=0)
    for r, s, ph, t_cmd, dur in samples(a.rows):
        held = held_out(r["task"], r["level"], r["index"], a.holdout)
        if (a.subset == "seen" and held) or (a.subset == "held" and not held):
            n_skip["subset"] += 1
            continue
        tag = f"_r{r['rep']}" if "rep" in r else ""
        fp = Path(a.films) / f"{r['task']}_L{r['level']}_e{r['index']}_{r['column']}_{r['method']}{tag}.npz"
        if not fp.exists():
            n_skip["film"] += 1
            continue
        if fp not in films:
            films.clear()
            films[fp] = dict(np.load(fp))
        z = films[fp]
        jn = [str(x) for x in z["jnt"]]
        adr = np.asarray(z["jnt_adr"])
        qa = [int(adr[jn.index(n)]) for n in MUJOCO_JOINT_NAMES]
        fa = [int(adr[jn.index(f"right_hand_{f}_joint")]) for f in FINGERS]
        sel = (z["phase"] == ph) & (z["t"] >= t_cmd - 0.041)
        T, QP, RF = z["t"][sel].astype(float), z["qpos"][sel].astype(float), z["ref"][sel].astype(float)
        if len(T) < 10 or T[-1] - t_cmd < dur + 0.6:
            n_skip["short"] += 1
            continue
        W, J = np.asarray(s["W"], float), np.asarray(s["J"], float)
        S3 = np.asarray(s["S3"], float)
        j0 = int(np.argmin(np.abs(T - t_cmd)))  # the film's frame at the command
        if bel is not None:  # --frame root: the robot's own S3, W and J, the reference root's heading frame
            S3, W, J = bel.of(S3, RF[j0, 0:7].astype(float))
        p0, yaw = W[:3], float(W[3])
        Rh = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        head = lambda x: np.c_[(x[..., :2].reshape(-1, 2) - p0[:2]) @ Rh[:2, :2], x[..., 2].reshape(-1)].reshape(
            x.shape
        )  # noqa: E731
        Xa = np.array([fk.spheres("right", q[0:3], q[3:7], q[qa], "mj", q[fa])[0] for q in QP])
        XR, jaw_e, Rw_e = [], None, None
        if bel is None:  # the reference pelvis carried onto the true one at the command
            Rt, Rr = quat_R(QP[j0, 3:7]), quat_R(RF[j0, 3:7])
            Ra, pa = Rt @ Rr.T, QP[j0, 0:3] - Rt @ Rr.T @ RF[j0, 0:3]
            for f in RF:
                x, jw = fk.spheres("right", f[0:3], f[3:7], f[7:36], "mj", f[36:43])
                XR.append(x @ Ra.T + pa)
                jaw_e, Rw_e = Ra @ jw + pa, Ra @ fk.d.xmat[fk.wrist["right"]].reshape(3, 3)
        else:  # --frame root: the motor's reference as it is
            for f in RF:
                x, jw = fk.spheres("right", f[0:3], f[3:7], f[7:36], "mj", f[36:43])
                XR.append(x)
                jaw_e, Rw_e = jw, fk.d.xmat[fk.wrist["right"]].reshape(3, 3).copy()
        XR = head(np.array(XR))
        RT = T - t_cmd
        t10 = 0.1 * np.arange(int(np.floor((T[-1] - t_cmd) / 0.1 + 1e-9)) + 1)
        X10 = VM.interp(RT, head(Xa), t10)
        ref = VM.interp(RT, XR, u * dur)
        act = VM.interp(t10, X10, u * dur)
        off = np.concatenate([act - ref, (X10[-VM.TAIL :].mean(0) - ref[-1])[None]])
        jaw = np.r_[(W[4:6] - p0[:2]) @ Rh[:2, :2], W[6]]
        R0 = Rh.T @ r6m(J)
        Rg = Rh.T @ Rw_e
        tgt = head(jaw_e[None])[0]
        rows["x"].append(np.r_[
            S3.ravel(), jaw, R0[:, 0], R0[:, 1], tgt, Rg[:, 0], Rg[:, 1], np.zeros(48 + 96 + 16), 0.0, SPEED, 0.0, 0.0, 1.0, dur,
            ref[::3, key].ravel(),
        ].astype(np.float32))  # fmt: skip
        b0 = sum(len(x) for x in Xall)
        Xall.append(X10.astype(np.float32))
        rows["ref"].append(ref.astype(np.float32))
        rows["off"].append(off.astype(np.float32))
        rows["a"].append(b0)
        rows["b"].append(b0 + len(X10))
        lid = layout(r["task"], r["level"], r["index"])
        rows["reset"].append(lid)
        rows["route"].append(lid if "rep" in r else -1)
        rows["held"].append(held)
        rows["sid"].append(int(r.get("sid", -1)))
        rows["k"].append(int(s.get("k", 0)))
        rows["S0"].append(S3[0, :29].astype(np.float32))
    n = len(rows["x"])
    assert n, f"no rows ({n_skip})"
    reset = np.array(rows["reset"])
    collect = (np.array(rows["route"]) >= 0).all()
    if collect:  # every training layout of the four tasks (present here or not) dealt in crc32 order to 10 buckets, so
        # datasets built task by task split as one built from all rows would
        lids = [
            layout(t, lv, i) for t in TASKS for lv in range(3) for i in range(10) if not held_out(t, lv, i, a.holdout)
        ]
        lids = sorted(lids, key=lambda v: zlib.crc32(f"jref{v}".encode()))
        deal = {v: k % 10 for k, v in enumerate(lids)}
        bucket = np.array([deal[v] for v in reset])
    else:
        bucket = np.full(n, 9)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T_ = dict(x=np.array(rows["x"]), ref=np.array(rows["ref"]), off=np.array(rows["off"]), seg=np.arange(n), a=np.array(rows["a"]),
              b=np.array(rows["b"]), reset=reset, side=np.ones(n), pelvis_z=np.full(n, -1.0), bends=np.zeros(n),
              route=np.array(rows["route"]), walked=np.zeros(n), phi=np.full(n, -1.0), sector=np.zeros(n, bool), bucket=bucket,
              palm=np.array([x == "hand_palm_link" for x in link]), held=np.array(rows["held"]), sid=np.array(rows["sid"]),
              k=np.array(rows["k"]), S0=np.array(rows["S0"]))  # fmt: skip
    np.savez_compressed(out / "table.npz", **T_)
    np.savez_compressed(out / "arm.npz", X=np.concatenate(Xall), radius=fk.radius, link=np.array(link))
    e0 = np.linalg.norm(T_["off"], axis=-1)
    print(f"{out}: {n} paths ({n_skip} skipped), layouts {len(set(reset.tolist()))}, buckets "
          f"{np.bincount(bucket, minlength=10).tolist()}, offset median {100 * np.median(e0):.2f} cm, palm path median "
          f"{100 * np.median(np.linalg.norm(np.diff(T_['ref'][:, :, key[-1]], axis=1), axis=2).sum(1)):.1f} cm")  # fmt: skip


def joint_range(a) -> None:
    X = np.concatenate([(lambda t: t["x"][t["bucket"] < 8][:, :29])(np.load(Path(d) / "table.npz")) for d in a.train])
    lo, hi = np.percentile(X, [0.5, 99.5], axis=0)
    ck = np.load(Path(a.check) / "table.npz")
    out = ((ck["S0"] < lo) | (ck["S0"] > hi)).sum(1)
    print(f"joints at the command outside the training paths' 0.5-99.5 % band ({len(out)} reaches): median "
          f"{np.median(out):.1f}, mean {out.mean():.2f}, max {out.max()}, counts {np.bincount(out).tolist()}")  # fmt: skip
    for name, m in (("seen layouts", ~ck["held"]), ("held-out layouts", ck["held"])):
        if m.any():
            print(f"  {name}: {int(m.sum())} reaches, median {np.median(out[m]):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("scene")
    b.add_argument("films")
    b.add_argument("rows", nargs="+")
    b.add_argument("--subset", choices=("all", "seen", "held"), default="all")
    b.add_argument("--holdout", type=int, default=3)
    b.add_argument("--frame", choices=("true", "root"), default="true",
                   help="true: the reference carried onto the true pelvis, its heading frame (as before); root: the motor's "
                   "reference as it is, the reference root's heading frame, no simulator truth in x")  # fmt: skip
    r = sub.add_parser("range")
    r.add_argument("check")
    r.add_argument("train", nargs="+")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else joint_range(a)


if __name__ == "__main__":
    main()

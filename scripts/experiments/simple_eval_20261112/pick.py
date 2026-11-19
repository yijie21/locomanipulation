"""The pick tasks of SIMPLE's evaluation set with ScaleBFM and the Dex3 hand: walk up, reach, close three fingers, lift.

Privileged baseline: the object's pose is read from the simulator and the servo reads the palm's true place. The grasp is
either SIMPLE's own stored Dex3 grasp for the object (--grasp stored) or GraspGen-X's palm poses from the head camera's
cloud (--grasp graspgen, after see.py and grasps_graspgen.py): the best-scored one the right hand can take, which is
one with the fingers pointing ahead of the robot, the palm facing left to down, and the hand clear of the table.

    BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src python pick.py XMovePickTeleop --side right --stand 0.27
-> one line per episode and "success k of n"; with --out, a json of the rows
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reachwm_20261030"))
import episodes as EP  # noqa: E402
import leg_odo as LO  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import WRIST, Motor, MotorConfig  # noqa: E402

GRASPNET = EP.SIMPLE / "data" / "assets" / "graspnet"
PALM = (0.0415, 0.0, 0.0)  # the Dex3 palm link in the wrist-yaw frame (the y offset of 3 mm is left out)
SIDE_OFF = -0.22  # m, the palm's target to the left of the pelvis at the stance (SIMPLE's grasp sits 0.22 to the right)
AHEAD = 0.7  # a hand pose the right arm takes: the cosine of the fingers with straight ahead of the robot,
ROLL = (-30.0, 110.0)  # and, in deg, the palm facing between left (the back of the hand up) and down
CLEAR = 0.06  # m, the palm link above what the object stands on: the middle finger hangs 4 cm below the palm link
PRE = 0.08  # m, GraspGen-X's pregrasp: back from the grasp against the palm's normal (the palm link's y)
BACK = 0.06  # m, the reach starts this far behind the pregrasp, on the line from the grasp through the pregrasp


def T(p, q) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3], M[:3, :3] = p, R.mat(np.asarray(q, float))
    return M


def stored_grasp(side: str, obj: str = "0"):
    """SIMPLE's Dex3 grasp for a GraspNet object: palm poses in the object's frame and the 7 finger joints,
    as (grasp, pregrasp, fingers at the grasp, fingers at the pregrasp)."""

    def load(f):
        v = torch.load(GRASPNET / f"dex_grasp/dex3_{side}/{obj}/{f}", map_location="cpu", weights_only=True)
        return v.numpy().ravel().astype(float)

    g, pre = load(f"{obj}__0.pt"), load(f"{obj}__0_pregrasp.pt")  # obj: the GraspNet uid ("10" the chips can)
    st = np.load(GRASPNET / f"stable/{obj}_stable.npy", allow_pickle=True)[0]
    T_ob = np.linalg.inv(T([0.4, 0.0, st[2]], st[3:]))  # the grasps are stored with the object at this pose
    return T_ob @ T(g[:3], g[3:7]), T_ob @ T(pre[:3], pre[3:7]), g[7:], pre[7:]


def graspgen_grasp(
    name: str, T_wo: np.ndarray, yaw: float, clear: float = CLEAR, roll_range=ROLL, cloud: str = "cloud"
):
    """GraspGen-X's best-scored palm pose that the right hand can take, with the palm link `clear` above the object's
    foot and the palm's facing within `roll_range`, in the object's frame, or None."""
    z = np.load(EP.OUT / cloud / f"grasp_{name}.npz")
    support = np.percentile(
        np.load(EP.OUT / cloud / f"{name}.npz")["pts"][:, 2], 2
    )  # the cloud's foot: what it stands on
    ahead, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    for T_og, score in zip(z["T_op"], z["score"]):
        Rw = (T_wo @ T_og)[:3, :3]
        roll = np.degrees(np.arctan2(-Rw[2, 1], Rw[:, 1] @ left))  # of the palm's normal: 0 faces left, 90 faces down
        if (
            Rw[:, 0] @ ahead >= AHEAD
            and roll_range[0] <= roll <= roll_range[1]
            and (T_wo @ T_og)[2, 3] >= support + clear
        ):
            T_op = T_og.copy()
            T_op[:3, 3] -= PRE * T_og[:3, 1]
            return T_og, T_op, z["f_close"], z["f_open"], float(score)
    return None


def touching(sim, body: int) -> list:
    """The hand links in contact with the body."""
    m, d, out = sim.m, sim.d, set()
    for i in range(d.ncon):
        a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        other = b if a == body else a if b == body else None
        if other is not None and ("_hand_" in m.body(other).name or "wrist" in m.body(other).name):
            out.add(m.body(other).name.split("_hand_")[-1])
    return sorted(out)


class Body:
    """What leg_odo's feeds read of the robot: joints, gyro and gravity readings. `pelvis` is the simulator's and is read
    once, to say where the episode starts."""

    def __init__(self, sim, mo):
        self.sim, self.m, self.odo_root = sim, mo, None

    def pelvis(self):
        q = self.sim.d.qpos
        return q[:3].copy(), R.mat(q[3:7])

    def self_state(self) -> np.ndarray:
        st = self.sim.state()
        grav = self.pelvis()[1].T @ np.array([0.0, 0.0, -1.0])  # the IMU's gravity reading
        return np.concatenate([st["dof_pos"], st["dof_vel"], grav, st["ang_vel"], [0.0]]).astype(np.float32)


def cont(mo, side: str) -> dict:
    """Motor.reach keywords that start a reach where the last one's command ended, in its posture family. A reach that
    starts at the measured palm instead drops the servo's lead in one step, and the hand recoils (it knocked objects)."""
    p, q = mo.last_fut[WRIST[side]]
    return dict(start=p[0] + R.mat(q[0]) @ R.OFF, start_rot=R.mat(q[0]), ik_from_ref=True)


FINGERS = {"thumb": [0, 1, 2], "index": [3, 4], "middle": [5, 6]}  # a finger's joints in runtime.DEX3_JOINTS
PRESS = 0.3  # rad, how far past the angle at which it touched a finger's joints are driven


def close_on(sim, side: str, body: int, f_close, f_pre, seconds: float = 1.0) -> None:
    """Close the hand finger by finger: each finger stops PRESS past where it first touches the body (the Dex3's
    fingertip pressure sensors give that touch), so fingers that keep curling do not turn the object in the hand."""
    m, d = sim.m, sim.d
    q = sim.gq[7 * ("left", "right").index(side) :][:7]
    cmd, free = np.array(f_pre, float), dict(FINGERS)
    for u in np.linspace(0, 1, int(seconds / 0.02)):
        touched = set()
        for i in range(d.ncon):
            a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            other = b if a == body else a if b == body else None
            if other is not None:
                touched |= {f for f in free if f"{side}_hand_{f}_" in m.body(other).name}
        for f in touched:
            j = free.pop(f)
            cmd[j] = d.qpos[q[j]] + PRESS * np.sign(f_close[j] - f_pre[j])
        for j in free.values():
            cmd[j] = f_pre[j] + u * (f_close[j] - f_pre[j])
        sim.fingers_cmd[side] = cmd.copy()
        yield


def close_by_feel(sim, side: str, f_close, f_pre, seconds: float = 1.0, lag: float = 0.4, slow: float = 0.3, log=None):
    """Close the hand with the finger joints' readings alone: a finger whose joints stop following the closing command
    (they moved less than `slow` of what a free finger moves in 0.1 s; a free joint follows with a time constant of
    `lag`) is pressing the object against the other fingers, and every finger stops PRESS past where it is."""
    q = sim.gq[7 * ("left", "right").index(side) :][:7]
    f_close, f_pre = np.asarray(f_close, float), np.asarray(f_pre, float)
    cmd, free, hist = f_pre.copy(), dict(FINGERS), []
    for k, u in enumerate(np.linspace(0, 1, int(seconds / 0.02))):
        now = sim.d.qpos[q].copy()  # the finger encoders
        hist.append(now)
        if k >= 10:
            moved = np.abs(now - hist[k - 5])
            want = np.abs(f_close - f_pre) / seconds * 0.1 * (1 - np.exp(-(k - 2.5) * 0.02 / lag))
            for f, j in list(free.items()):
                jj = [i for i in j if want[i] > 2e-3]
                if log is not None:
                    log.append((k, f, float(moved[jj].sum() / max(1e-9, want[jj].sum())) if jj else None))
                if jj and moved[jj].sum() < slow * want[jj].sum():
                    for j in free.values():  # the object is held between this finger and the others: all stop
                        cmd[j] = now[j] + PRESS * np.sign(f_close[j] - f_pre[j])
                    free.clear()
                    break
        for j in free.values():
            cmd[j] = f_pre[j] + u * (f_close[j] - f_pre[j])
        sim.fingers_cmd[side] = cmd.copy()
        yield


def pick(
    mo, sim, side: str, Tg, Tp, f_close, f_pre, squeeze: float = 0.6, close_s: float = 0.5, body=None, feel=None
) -> dict:
    """Reach to the pregrasp, then the grasp, close the fingers, lift 15 cm. The fingers are driven past the closed
    posture by `squeeze`, or with `body` given each one stops where it touches that body (close_on)."""
    step = Tp[:3, 3] - Tg[:3, 3]
    sim.fingers_cmd[side] = f_pre
    mo.reach(side, Tp[:3, 3] + BACK * step / max(1e-6, np.linalg.norm(step)), Tp[:3, :3])
    mo.reach(side, Tp[:3, 3], Tp[:3, :3], speed=0.1, **cont(mo, side))
    mo.reach(side, Tg[:3, 3], Tg[:3, :3], speed=0.08, **cont(mo, side))
    p, Rm = mo.jaw_centre(side)  # the true palm, for the record
    err = 100 * np.linalg.norm(p - Tg[:3, 3])
    ang = np.degrees(np.arccos(np.clip((np.trace(Tg[:3, :3].T @ Rm) - 1) / 2, -1, 1)))
    if feel is not None:
        for _ in close_by_feel(sim, side, f_close, f_pre, log=feel):
            mo.hold(0.02)
    elif body is not None:
        for _ in close_on(sim, side, body, f_close, f_pre):
            mo.hold(0.02)
    for u in np.linspace(0, 1, int(close_s / 0.02)) if body is None and feel is None else ():
        sim.fingers_cmd[side] = f_pre + u * (f_close + squeeze * (f_close - f_pre) - f_pre)
        mo.hold(0.02)
    mo.hold(0.5)
    p, Rm = (getattr(mo, "jaw_source", None) or mo.jaw_centre)(side)
    mo.reach(side, p + [0, 0, 0.15], Rm, speed=0.1, **cont(mo, side))
    mo.hold(1.0)
    return dict(palm_cm=float(err), palm_deg=float(ang))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--side", default="right")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stand", type=float, default=0.27, help="m, the palm's target ahead of the pelvis at the stance")
    ap.add_argument("--grasp", default="stored")
    ap.add_argument("--side-off", type=float, default=SIDE_OFF, help="m, the palm's target to the left of the pelvis")
    ap.add_argument("--squeeze", type=float, default=0.6, help="how far past the closed posture the fingers are driven")
    ap.add_argument("--table-gap", type=float, default=0.2, help="m, the pelvis stays this far before the table's edge")
    ap.add_argument(
        "--odo", action="store_true", help="no pelvis or hand truth after the start: leg odometry and joints"
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    R.OFF[:] = PALM
    pol = R.Policy("cuda", ScaleBfmBackendConfig())
    rows = []
    for ep in EP.episodes(a.task)[: a.n]:
        info = EP.build(ep)
        sim = R.Sim(pol.meta, backend=ScaleBfmBackendConfig(scene_dir=info["dir"], object_name=""))
        mo = Motor(sim, pol, MotorConfig(), model_xml=info["dir"] + "/scene_locomani.xml")
        if a.odo:  # as grasp_room.make_env with BFM_ROOT=odo: ScaleBFM is told the pelvis position from leg odometry
            body = Body(sim, mo)
            mo.cfg = dataclasses.replace(mo.cfg, stop_ki=0.0, stop_timeout=mo.cfg.settle_s)
            body.odo_root = LO.OdoRoot(body)
            feed = LO.JawFeed(body)
        mo.reset(info["start"]["xy"], info["start"]["yaw"])
        EP.pull_objects(sim.m, sim.d, info)
        b = sim.m.body(info["target"]).id
        mo.hold(1.0)
        if (
            a.odo
        ):  # the look happens here: the target is known in the frame of the pelvis now, followed by odometry after
            p0, R0 = body.pelvis()
            yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
            feed.begin()
        z0 = float(sim.d.xpos[b][2])
        T_wo = T(sim.d.xpos[b], sim.d.xquat[b])
        yaw = info["start"]["yaw"]
        row = dict(level=ep["level"], index=ep["index"])
        if a.grasp == "stored":
            T_og, T_op, f_close, f_pre = stored_grasp(a.side)
            Tg, Tp = T_wo @ T_og, T_wo @ T_op
            stance = (Tg[0, 3] - a.stand, sim.d.qpos[1])
        else:
            found = graspgen_grasp(f"{a.task}_L{ep['level']}_e{ep['index']}", T_wo, yaw)
            if found is None:
                rows.append(dict(row, success=False, none=True))
                print(rows[-1], flush=True)
                continue
            T_og, T_op, f_close, f_pre, row["score"] = found
            Tg, Tp = T_wo @ T_og, T_wo @ T_op
            c, s = np.cos(yaw), np.sin(yaw)
            stance = Tg[:2, 3] - np.array([[c, -s], [s, c]]) @ [a.stand, a.side_off]
        edge = min(t["pos"][0] - 0.5 * t["size"][0] for t in info["tables"].values())  # the tables lie square to x
        stance = (min(stance[0], edge - a.table_gap), stance[1])
        row["stand"] = float(Tg[0, 3] - stance[0])
        mo.walk_to(tuple(stance), yaw, face="fixed")
        mo.hold(0.5)
        if a.odo:
            c, s = np.cos(yaw0), np.sin(yaw0)
            p1 = p0 + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ feed.odo.p
            row["walk_cm"] = float(100 * np.linalg.norm((p1 - sim.d.qpos[:3])[:2]))  # the belief against the truth
            feed.anchor(p1, yaw0 + feed.odo.yaw, WRIST, R.OFF)
            mo.jaw_source = feed.jaw
        row.update(pick(mo, sim, a.side, Tg, Tp, f_close, f_pre, squeeze=a.squeeze))
        z = float(sim.d.xpos[b][2])
        row.update(touch=touching(sim, b), rise_cm=100 * (z - z0), success=bool(EP.lifted(a.task, z0, z)),
                   fell=mo.fell_at, t=float(mo.t))  # fmt: skip
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    print("success", sum(r["success"] for r in rows), "of", len(rows))
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()

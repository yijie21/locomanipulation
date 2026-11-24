"""The BiGym 2.0 environment server (TODO 49.5): one standalone file, run under BiGym's own venv.

    third_party/BiGym2/.venv/bin/python server.py SOCK [--refuse-lo N --refuse-hi N] [--depth] [--shape H W]

It imports nothing of locomani (BiGym's venv has MuJoCo 3.8.x, ours 3.13), listens on one unix socket and answers pickled
messages from one client (client.ServerLink). It dies with the client. The messages (a tuple, the first item the verb):

    ("hello",)                                   -> versions, task list
    ("reset", task, seed)                        -> the first state, the instruction, the model's names
    ("step", cmd)                                -> one 50 Hz control step; cmd = dict(vx, vy, height, wz, pitch, left, right, grip)
    ("state",)                                   -> the robot's own readings (joints, hand sites, pelvis) and the clock
    ("ik", side, pos, quat|None, q|None)         -> the arm's joint angles that put the hand site at `pos` (world frame)
    ("rgbd", camera)                             -> rgb, metric depth, fovy, the camera in the pelvis frame (needs --depth)
    ("truth",)                                   -> the simulator's truth: scene bodies' poses, movable joints, task facts
    ("verts", name)                              -> the world vertices of a body's collision geoms
    ("outcome",)                                 -> the benchmark's own verdict and the episode's contact record
    ("close",)

Seeds. `reset` refuses a seed in [refuse_lo, refuse_hi) (BiGym's evaluation block, 620000-629999) and any negative one: the
client's checks come first, this is the last of the layers (tests/test_bigym2_episodes.py). The depth patch (`--depth`) is applied
here, at run time, to `bigym.loco.env.CameraConfig`, which the environment builds its cameras with (it hard-codes depth=False,
bigym/loco/env.py:669): BiGym's files are not edited. With it the head camera renders colour and depth at `--shape`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROBOT = "g1_29dof_with_dex1_1/"
ROLL_OUT = 0.12  # rad, the least outward shoulder roll the IK solutions use (left; mirrored for the right)
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
INSTRUCTIONS = {
    "reach_target_single": "Touch the red sphere with your left hand.",
    "drawer_top_open": "Open the top drawer of the kitchen cabinet.",
}


def install_depth_patch(shape: tuple[int, int]) -> None:
    """Make every camera the environment builds render depth too, at `shape` (H, W). Run before `make`."""
    import bigym.loco.env as E

    original = E.CameraConfig

    def with_depth(name, rgb=True, depth=False, resolution=(84, 84)):
        return original(name=name, rgb=True, depth=True, resolution=tuple(shape))

    E.CameraConfig = with_depth


def rot_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def pose4(pos, mat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T


class Server:
    def __init__(self, refuse: tuple[int, int], depth: bool, shape: tuple[int, int]):
        self.refuse, self.depth, self.shape = refuse, depth, shape
        self.env = None
        self.task = None
        if depth:
            install_depth_patch(shape)

    # ---- the environment

    def build(self, task: str):
        from bigym.loco import make

        if self.env is not None and self.task == task:
            return
        if self.env is not None:
            self.env.close()
        kw = dict(camera_keys=("head",)) if self.depth else {}  # one rendered camera: the depth patch makes it cost
        self.env = make(task, **kw)
        self.task = task
        ie = self.env.inner_env
        self.m, self.d = ie.model, ie.data
        m = self.m
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}" for i in range(m.nbody)]
        robot_root = next(i for i, n in enumerate(self.body_names) if n.startswith(ROBOT))
        self.is_robot = np.zeros(m.nbody, bool)
        for i in range(m.nbody):
            b = i
            while b > 0:
                if b == robot_root or self.body_names[b].startswith(ROBOT):
                    self.is_robot[i] = True
                    break
                b = int(m.body_parentid[b])
        # a scene body that has joints of its own (free, slide, hinge): what a task moves. Geoms of its children count for it.
        self.owner = np.full(m.nbody, -1, int)
        for i in range(m.nbody):
            b = i
            while b > 0 and not self.is_robot[i]:
                if int(m.body_dofnum[b]) > 0:
                    self.owner[i] = b
                    break
                b = int(m.body_parentid[b])
        self.movable = sorted(set(int(x) for x in self.owner if x >= 0))
        self.pitch = bool(self.env.lowerbody.pitch_cmd_enabled)
        self.arm_adr = {}
        self.arm_dof = {}
        for side in ("left", "right"):
            js = [m.joint(f"{ROBOT}{side}_{j}_joint") for j in ARM_JOINTS]
            self.arm_adr[side] = np.array([int(j.qposadr[0]) for j in js])
            self.arm_dof[side] = np.array([int(j.dofadr[0]) for j in js])
        self.arm_range = {
            s: np.array([m.jnt_range[m.joint(f"{ROBOT}{s}_{j}_joint").id] for j in ARM_JOINTS])
            for s in ("left", "right")
        }
        # the shoulder's roll may not swing the arm into the torso (the yaw link meets torso_link at roll < ~0): the IK keeps it outward
        self.arm_range["left"][1, 0] = max(self.arm_range["left"][1, 0], ROLL_OUT)
        self.arm_range["right"][1, 1] = min(self.arm_range["right"][1, 1], -ROLL_OUT)
        self.site = {s: m.site(f"{ROBOT}{s}_end_effector").id for s in ("left", "right")}
        self.pelvis_body = m.body(f"{ROBOT}pelvis").id

    def reset(self, task: str, seed: int):
        lo, hi = self.refuse
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or lo <= seed < hi:
            raise ValueError(f"seed {seed!r} refused: negative or inside the evaluation block [{lo}, {hi})")
        self.build(task)
        ts = self.env.reset(seed=int(seed))
        self.steps, self.contacts, self.over = 0, {}, None
        self.fell = self.success = False
        self.trunc = None
        return dict(
            instruction=INSTRUCTIONS.get(task, task.replace("_", " ")),
            names=self.body_names, movable=[self.body_names[b] for b in self.movable],
            pitch=self.pitch, budget=int(self.env._time_limit), dt=0.02,
            state=self.state(), rgb_obs_shape=tuple(ts.rgb_obs.shape),
        )  # fmt: skip

    def step(self, cmd: dict):
        env = self.env
        raw = np.array(env.raw_hold_action(), float)
        o = 5 if self.pitch else 4
        raw[0], raw[1], raw[3] = cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("wz", 0.0)
        if cmd.get("height") is not None:
            raw[2] = cmd["height"]
        if self.pitch and cmd.get("pitch") is not None:
            raw[4] = cmd["pitch"]
        if cmd.get("left") is not None:
            raw[o : o + 7] = cmd["left"]
        if cmd.get("right") is not None:
            raw[o + 7 : o + 14] = cmd["right"]
        if cmd.get("grip") is not None:
            raw[o + 14 : o + 16] = cmd["grip"]
        ts = env.step(env.normalize_action(raw))
        self.steps += 1
        self.record_contacts()
        self.success = env.episode_succeeded()
        self.fell = env.episode_fell()
        self.trunc = env.last_truncation_reason()
        self.over = bool(ts.last() or self.success or self.trunc)
        return dict(
            over=self.over,
            success=self.success,
            fell=self.fell,
            truncation=self.trunc,
            steps=self.steps,
            state=self.state(),
        )

    def record_contacts(self):
        """Per step, every robot body touching a scene body that has joints of its own: {(robot body, object): [steps]}."""
        m, d = self.m, self.d
        seen = set()
        for c in d.contact[: d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for r, o in ((b1, b2), (b2, b1)):
                if self.is_robot[r] and not self.is_robot[o] and self.owner[o] >= 0:
                    seen.add((self.body_names[r], self.body_names[int(self.owner[o])]))
        for pair in seen:
            self.contacts.setdefault(pair, []).append(self.steps)

    # ---- readings

    def state(self) -> dict:
        d = self.d
        pel = self.body_pose_of(self.pelvis_body)
        out = dict(t=self.steps * 0.02, pelvis=pel, tilt=float(np.hypot(*self.tilt())))
        for s in ("left", "right"):
            out[f"{s}_q"] = d.qpos[self.arm_adr[s]].copy()
            out[f"{s}_hand"] = d.site_xpos[self.site[s]].copy()
            out[f"{s}_hand_R"] = d.site_xmat[self.site[s]].reshape(3, 3).copy()
        out["grip_q"] = {
            s: [float(d.qpos[self.m.joint(f"{ROBOT}{s}_dex1_finger_joint_{k}").qposadr[0]]) for k in (1, 2)]
            for s in ("left", "right")
        }
        return out

    def tilt(self):
        R = self.d.xmat[self.pelvis_body].reshape(3, 3)
        return R[2, 0], R[2, 1]

    def body_pose_of(self, b: int) -> np.ndarray:
        return pose4(self.d.xpos[b], self.d.xmat[b])

    def ik(self, side: str, pos, quat, q_init, iters=200, tol=1e-4):
        """Damped least squares on the arm's seven joints so the hand site reaches `pos` (and `quat`, a world-frame
        orientation, when given), from the live pelvis pose. Joint limits are kept. -> (q, position error m)."""
        m, d0 = self.m, self.d
        d = mujoco.MjData(m)
        d.qpos[:] = d0.qpos
        adr, dof = self.arm_adr[side], self.arm_dof[side]
        q = np.array(q_init if q_init is not None else d0.qpos[adr], float)
        lo, hi = self.arm_range[side][:, 0], self.arm_range[side][:, 1]
        sid = self.site[side]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err_p = np.inf
        for _ in range(iters):
            d.qpos[adr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e = np.asarray(pos, float) - d.site_xpos[sid]
            err_p = float(np.linalg.norm(e))
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J, r = jacp[:, dof], e
            if quat is not None:
                cur = rot_to_quat(d.site_xmat[sid])
                dq = np.zeros(3)
                neg = np.zeros(4)
                mujoco.mju_negQuat(neg, cur)
                rel = np.zeros(4)
                mujoco.mju_mulQuat(rel, np.asarray(quat, float), neg)
                mujoco.mju_quat2Vel(dq, rel, 1.0)
                J, r = np.vstack([J, 0.5 * jacr[:, dof]]), np.concatenate([e, 0.5 * dq])
            if np.linalg.norm(r) < tol:
                break
            step = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(J.shape[0]), r)
            n = np.linalg.norm(step)
            if n > 0.2:
                step *= 0.2 / n
            q = np.clip(q + step, lo, hi)
        return q, err_p

    def rgbd(self, camera: str = "head"):
        if not self.depth:
            raise RuntimeError("the server was started without --depth")
        ie = self.env.inner_env
        v = ie.get_visual_obs()
        cid = self.m.camera(f"{ROBOT}{camera}").id
        Rcv = self.d.cam_xmat[cid].reshape(3, 3) @ np.diag(
            [1.0, -1.0, -1.0]
        )  # MuJoCo's camera (x right, y up, z back) -> x right, y down, z forward
        Tw = pose4(self.d.cam_xpos[cid], Rcv)
        Tb = np.linalg.inv(self.body_pose_of(self.pelvis_body)) @ Tw
        rgb = np.moveaxis(v[f"rgb_{camera}"], 0, -1).copy()
        return dict(
            rgb=rgb,
            depth=np.asarray(v[f"depth_{camera}"], np.float32).copy(),
            fovy=float(self.m.cam_fovy[cid]),
            T_body_cam=Tb,
            t=self.steps * 0.02,
        )

    def truth(self) -> dict:
        d, m = self.d, self.m
        bodies = {self.body_names[b]: self.body_pose_of(b) for b in range(1, m.nbody) if not self.is_robot[b]}
        joints = {}
        for j in range(m.njnt):
            if self.is_robot[int(m.jnt_bodyid[j])]:
                continue
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            b = int(m.jnt_bodyid[j])
            joints[nm] = dict(q=float(d.qpos[int(m.jnt_qposadr[j])]), range=m.jnt_range[j].tolist(), body=self.body_names[b],
                              axis_world=(d.xmat[b].reshape(3, 3) @ m.jnt_axis[j]).tolist(), type=int(m.jnt_type[j]))  # fmt: skip
        facts = {}
        ie = self.env.inner_env
        if hasattr(ie, "targets"):
            facts["targets"] = [
                dict(pos=t.get_position().tolist(), hands=[str(h.value).lower() for h in t.config.target_hands])
                for t in ie.targets
            ]
        if hasattr(ie, "cabinet_drawers"):
            facts["drawers"] = np.asarray(ie.cabinet_drawers.get_state(), float).tolist()
        for part in ("cabinet_wall", "cabinet_door_left", "cabinet_door_right"):
            if hasattr(ie, part):
                facts[part] = np.asarray(getattr(ie, part).get_state(), float).tolist()
        # a handle: a child body (of a body with joints of its own) with collision geoms of its own; its box and mesh corners' extent
        facts["handles"] = {}
        for b in self.movable:
            for c in range(1, m.nbody):
                if int(m.body_parentid[c]) == b and not self.is_robot[c] and int(m.body_dofnum[c]) == 0:
                    v = self.verts(self.body_names[c])
                    if len(v):
                        facts["handles"].setdefault(self.body_names[b], []).append(
                            dict(body=self.body_names[c], lo=v.min(0).tolist(), hi=v.max(0).tolist())
                        )
        return dict(
            bodies=bodies, joints=joints, facts=facts, pelvis=self.body_pose_of(self.pelvis_body), t=self.steps * 0.02
        )

    def verts(self, name: str) -> np.ndarray:
        m, d = self.m, self.d
        b0 = self.body_names.index(name)
        out = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            top = b
            while top > b0 and top > 0:
                top = int(m.body_parentid[top])
            if top != b0 or int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
                continue
            size, R, p = m.geom_size[g], d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                c = m.mesh_vert[a : a + n]
            else:
                r = float(size[0])
                c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(
                    [r, r, size[2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else r]
                )
            out.append(c @ R.T + p)
        return np.vstack(out) if out else np.zeros((0, 3))

    def outcome(self) -> dict:
        return dict(success=self.success, fell=self.fell, truncation=self.trunc, steps=self.steps,
                    termination=self.env.episode_termination(), contacts={f"{a}|{b}": v for (a, b), v in self.contacts.items()})  # fmt: skip

    def handle(self, msg):
        verb = msg[0]
        if verb == "hello":
            import bigym.loco.env as E
            from bigym.loco.fingerprint import SUBSTRATE_VERSION

            return dict(
                mujoco=mujoco.__version__,
                substrate=SUBSTRATE_VERSION,
                depth=self.depth,
                python=sys.version.split()[0],
                env=E.__name__,
            )
        if verb == "reset":
            return self.reset(msg[1], msg[2])
        if verb == "step":
            return self.step(msg[1])
        if verb == "state":
            return self.state()
        if verb == "ik":
            q, e = self.ik(msg[1], msg[2], msg[3], msg[4])
            return dict(q=q, err=e)
        if verb == "rgbd":
            return self.rgbd(msg[1])
        if verb == "truth":
            return self.truth()
        if verb == "verts":
            return dict(v=self.verts(msg[1]))
        if verb == "outcome":
            return self.outcome()
        raise ValueError(f"unknown verb {verb!r}")


def die_with_parent():
    import ctypes

    ctypes.CDLL("libc.so.6").prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sock")
    ap.add_argument("--refuse-lo", type=int, required=True)
    ap.add_argument("--refuse-hi", type=int, required=True)
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--shape", type=int, nargs=2, default=(240, 320))
    a = ap.parse_args(argv)
    die_with_parent()
    srv = Server((a.refuse_lo, a.refuse_hi), a.depth, tuple(a.shape))
    with Listener(a.sock, family="AF_UNIX") as lis:
        conn = lis.accept()
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                return 0
            if msg[0] == "close":
                conn.send(dict(ok=True))
                return 0
            try:
                conn.send(srv.handle(msg))
            except Exception as e:  # noqa: BLE001 - the client re-raises it
                conn.send(dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:]))


if __name__ == "__main__":
    raise SystemExit(main())

"""The bed's side of the socket to SIMPLE's environment (TODO 49.2): the ScaleBFM runtime's Sim with SIMPLE's own environment
(server.py) as the physics, the environment's step count and judge as the clock, and the episode facade a skill package sees.

Moved from simple_eval_20261112/official.py (OffSim, EpisodeOver) and escalation_20261116/bed_simple_official.py (OffCfgSim,
OClock, OEnv; `Env`'s fields folded into OEnv; PHASE_BOX not ported). `OffSim.step` sends ScaleBFM's joint targets with its
gains to SIMPLE's motors and writes the joint readings and base state that come back into this model, which then serves as
the robot's own kinematic model. The scene's objects in this model never move: nothing may be read from them. `eyes` gives the
head camera's colour image (Isaac Sim) and the depth worked out from its stereo pair (stereo_server.py).
"""

import contextlib
import copy
import dataclasses

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Motor
from locomani.bench.simple import judge
from locomani.perception import head

DT, CAP_S = 0.02, 60.0
LIMIT = dict(BendPickMP=400, TabletopGraspMP=800, XMovePickTeleop=800, XMoveBendPickTeleop=800, HandoverTeleop=800,
             LocomotionPickBetweenTablesTeleop=1200,
             OpenOvenTeleop=1000, CloseDoorTeleop=450, OpenFaucetTeleop=1000, OpenTrashCanTeleop=800)  # max_episode_steps of each task file; XMoveBendPick: HITCH table VII  # fmt: skip


class BedMotor(Motor):
    """The Motor with the episode's clock in its tick: every 20 ms step the Motor takes goes through `clock.tick`, which counts
    it, asks the environment's judge and raises Done when the episode is over (the old bed replaced `mo.tick` on the instance).
    `reach` also records how far the palm is from each reach's goal when the reach returns (record only: row["reach_err_cm"])."""

    clock = None
    row: dict | None = None

    def raw_tick(self) -> bool:
        return super().tick()

    def tick(self) -> bool:
        return self.raw_tick() if self.clock is None else self.clock.tick()

    def reach(self, side, p, Rm, **kw):
        out = super().reach(side, p, Rm, **kw)
        if self.row is not None:
            self.row.setdefault("reach_err_cm", []).append(
                round(float(100 * np.linalg.norm(self.jaw_centre(side)[0] - np.asarray(p))), 2)
            )
        return out


class EpisodeOver(Exception):
    pass


class OffSim(R.Sim):
    rgbd = None  # sensor=rgbd: a perception.rgbd.RgbdSensor (see `use_rgbd`); None = the stereo pair through the depth service

    def __init__(self, meta, backend, link, depth_link, film: bool = False):
        super().__init__(meta, backend=backend)
        self.link, self.depth_link = (
            link,
            depth_link,
        )  # multiprocessing connections, shared by the run's episodes (depth_link: None under sensor=rgbd)
        self.counted, self.over, self.film, self.first, self.steps = False, None, film, None, 0

    def pull(self, st: dict | None = None) -> None:
        st = st or self.first
        d = self.d
        d.qpos[:7], d.qpos[self.oq] = st["qpos"][:7], st["qpos"][7:]
        d.qvel[:6], d.qvel[self.ov] = st["qvel"][:6], st["qvel"][6:]
        mujoco.mj_forward(self.m, d)

    def reset(self, spawn_xy, spawn_yaw, crate_dy):
        """Motor.reset calls this before its start hold: the local model's default pose (runtime.Sim.reset), then the
        environment's reset state on top, so the hold starts from the episode's real pose. Without the pull the hold
        ran on the default pose (pelvis 0.762 m instead of 0.797 m): dropped in the TODO 49.2 port, found by the oven
        follow-up of the parity run (old 46/50, new 29/40), restored from simple_eval_20261112/official.py."""
        super().reset(spawn_xy, spawn_yaw, crate_dy)
        self.pull()

    def step(self, tgt):
        if self.over is not None:
            raise EpisodeOver
        q = np.zeros(43)
        q[self.body_slot], q[self.hand_slot] = tgt, self.grip_target
        self.link.send(("step", q, self.kp43, self.kd43, self.counted, self.film))
        st = self.link.recv()
        self.pull(st)
        if st["terminated"] or st["truncated"]:
            self.over = dict(success=st["success"], steps=st["steps"], truncated=st["truncated"])
        self.steps = st["steps"]

    def use_rgbd(self, depth_noise: str, seed: int) -> None:
        """sensor=rgbd: the camera view's depth is the LEFT head camera's rendered depth (the monocular RGB-D input), as is
        (depth_noise=none) or through the RealSense-like noise model, whose occlusion shadows use the RIGHT camera's rendered
        depth (one baseline to the right). `seed` makes a rerun of an episode see the same noise. The server attaches
        Isaac's distance_to_image_plane annotator to the two head cameras on this message only."""
        from locomani.perception.rgbd import RgbdSensor

        self.link.send(("rgbd_setup",))
        self.link.recv()
        self._right = None
        self.rgbd = RgbdSensor(self._exact_rgbd, depth_noise, seed, depth_right=lambda: self._right)

    def _exact_rgbd(self):
        """One render of both head cameras: the left image and the left / right rendered depth (m along the optical axis,
        NaN where nothing is hit), as a Frame. The camera pose is the robot's own forward kinematics."""
        from locomani.perception.sensor import Frame

        self.link.send(("eyes_d",))
        e = self.link.recv()
        dl, dr = (
            np.where(np.isfinite(e[k]) & (e[k] > 0), e[k], np.nan).astype(np.float32)
            for k in ("depth_left", "depth_right")
        )
        self._right = dr
        if getattr(self, "_kin", None) is None:
            self._kin = head.HeadKin(self.m, self.qadr)
        p, Rc = self._kin(self.d.qpos[self.qadr])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rc, p
        self._last_raw = dict(left=e["left"], right=e["right"], depth_left=dl, depth_right=dr)
        return Frame(
            rgb=np.asarray(e["left"]),
            depth=dl,
            K=head.intrinsics(),
            T_body_cam=T,
            source="rgbd",
            t=float(self.steps) * 0.02,
        )

    def eyes(self):
        if self.rgbd is not None:
            f = self.rgbd.frame()
            self.last_eyes = dict(
                self._last_raw, depth=f.depth, sg=f.depth, lb=f.depth
            )  # lb: what the see_self rules read as depth
            return f.rgb, f.depth
        self.link.send(("eyes",))
        e = self.link.recv()
        self.depth_link.send((e["left"], e["right"]))
        depth, sg, lb = self.depth_link.recv()
        self.last_eyes = dict(e, depth=lb, sg=sg, lb=lb)
        return e[
            "left"
        ], lb  # LingBot-Depth's output given the matched depth: raw matching scatters 2 cm at 0.85 m (a quarter pixel)

    def video(self, path: str) -> int:
        self.link.send(("video", path))
        return self.link.recv()


def spatial_key(a: dict, kind: str) -> str:
    return f"container_{a['uid']}" if kind == "container" else str(a["uid"])


def state_cfg(cfg: dict) -> dict:
    """The environment configuration with the `layout` block's poses and distractors written into the places SIMPLE's reset
    reads. For a released episode (layout == spatial) this changes nothing."""
    cfg = copy.deepcopy(cfg)
    dr, A = cfg["dr_state_dict"], cfg["layout"]["actors"]
    sp = dr["spatial"]
    for key, a in A.items():
        if "asset" not in a:  # the table
            continue
        kind = "container" if key == "container" else "target" if key == "target" else "distractor"
        if key == "articulated":  # the articulated tasks' asset: SIMPLE's spatial key is "articulated_<uid>"
            sp[f"articulated_{a['uid']}"] = dict(
                position=list(a["pose"]["position"]), quaternion=list(a["pose"]["quaternion"])
            )
            continue
        sp[spatial_key(a, kind)] = dict(position=list(a["pose"]["position"]), quaternion=list(a["pose"]["quaternion"]))
    dr["distractors"] = {str(a["uid"]): a["asset"] for k, a in A.items() if k.startswith("distractor")}
    return cfg


class OffCfgSim(OffSim):
    """OffSim whose reset takes a configuration, and which keeps the last reply (success flag, reward) of the environment."""

    last: dict | None = None
    zoff: float = (
        0.0  # the AMO tasks' world lies this far lower than the mirror's (ground at -zoff): added to the root height
    )

    def pull(self, st: dict | None = None) -> None:
        st = st or self.first
        d = self.d
        d.qpos[:7], d.qpos[self.oq] = st["qpos"][:7], st["qpos"][7:]
        d.qpos[2] += self.zoff
        d.qvel[:6], d.qvel[self.ov] = st["qvel"][:6], st["qvel"][6:]
        mujoco.mj_forward(self.m, d)

    def begin_cfg(self, task: str, state: dict) -> dict:
        self.link.send(("reset_cfg", task, state))
        st = self.link.recv()
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in st["names"]]
        assert min(ids) >= 0, "a SIMPLE joint is missing in the robot's model"
        self.oq, self.ov = self.m.jnt_qposadr[ids], self.m.jnt_dofadr[ids]
        slot = {int(a): k for k, a in enumerate(self.oq)}
        self.body_slot = np.array([slot[int(a)] for a in self.qadr])
        self.hand_slot = np.array([slot[int(a)] for a in self.gq])
        self.kp43, self.kd43 = np.zeros(43), np.zeros(43)
        self.kp43[self.body_slot], self.kd43[self.body_slot] = self.kp, self.kd
        self.kp43[self.hand_slot], self.kd43[self.hand_slot] = np.tile(R.DEX3_KP, 2), R.DEX3_KD
        self.counted, self.over, self.first, self.last = False, None, st, None
        self.limits = dict(theirs=st["torque_limit"][self.body_slot], ours=self.tlim)
        return st

    def step(self, tgt):
        q = np.zeros(43)
        q[self.body_slot], q[self.hand_slot] = tgt, self.grip_target
        self.link.send(("step", q, self.kp43, self.kd43, self.counted, self.film))
        st = self.link.recv()
        self.pull(st)
        self.last = st
        if self.counted and (st["terminated"] or st["truncated"]):
            self.over = dict(success=st["success"], steps=st["steps"], truncated=st["truncated"])
        self.steps = st["steps"]

    def truth(self) -> dict:
        self.link.send(("truth",))
        return self.link.recv()


class Done(Exception):
    """The episode ended: its reason is the message (fell, success, success_late, cap)."""


class OClock:
    """The environment's own step count and judge, wrapped round Motor.tick; counted seconds per phase."""

    def __init__(self, env):
        self.env, self.steps, self.counting, self.t_success, self.label, self.phases = env, 0, False, None, "start", {}
        self.late = False  # past the limit: raw steps, only to see whether the episode would have succeeded
        self.pick = (
            env.task in judge.PICK
        )  # the three picks end on a held run of 0.2 s (comply_simple), not on the check
        self.t_check = None  # the first counted step the environment's check held
        self.official: dict | None = None
        self.reward_last, self.reward_max = 0.0, 0.0
        self._tick = env.mo.raw_tick
        env.mo.clock = self  # BedMotor.tick calls this clock (the old bed assigned `mo.tick` itself)

    def start(self) -> None:
        self.env.sim.counted, self.counting = True, True

    @contextlib.contextmanager
    def phase(self, name: str):
        """Counted seconds are booked to the phase named while the block runs. (The old PHASE_BOX record, the true box pose at
        the end of every phase, was a default-off hand switch and is not ported.)"""
        prev, self.label = self.label, name
        try:
            yield
        finally:
            self.label = prev

    def tick(self) -> bool:
        ok = self._tick()
        if not self.counting:
            return ok
        if not ok:
            raise Done("fell")
        sim = self.env.sim
        self.steps += 1
        self.phases[self.label] = self.phases.get(self.label, 0) + 1
        st = sim.last
        if "reward" in st:
            self.reward_last = st["reward"]
            self.reward_max = max(self.reward_max, st["reward"])
        if not self.late and sim.over is not None:
            ov = sim.over
            self.official = dict(ov, step=self.steps)
            if not ov["truncated"] or st["terminated"]:  # the environment's check_success held
                if not self.pick:
                    self.t_success = self.steps * DT
                    raise Done("success")
                if self.t_check is None:
                    self.t_check = self.steps * DT
                sim.over = None  # the episode goes on until a held run reaches 0.2 s
                if st.get("held_t") is not None:
                    self.t_success = st["held_t"]
                    raise Done("success")
            else:
                self.late, sim.over, sim.counted = True, None, False  # truncated: on without the judge
        elif self.late and st["success"]:
            self.t_success = self.steps * DT
            raise Done("success_late")
        if not self.late and not self.env.limit and self.steps >= self.env.limit_eff:  # no TimeLimit in the environment
            self.late, self.official = True, dict(truncated=True, steps=self.steps, step=self.steps)
        if self.steps * DT >= CAP_S:
            raise Done("cap")
        return ok


@dataclasses.dataclass
class OEnv:
    """What a skill package gets of the episode: task, layout `info`, the (local) sim and Motor, the Belief, the parameters K,
    the stance, the record `row`, the clock, the view, and `mark`. Poses for the marks are read from the environment's
    server, never from the local model (bed_simple.Env's other half). `sam3`, `cloud_dir` and `mask_post` are what the
    camera view needs of the run: SAM 3's address, the look cache folder, and the see_self mask filter (or None)."""

    task: str
    info: dict
    sim: object
    mo: object
    bel: object
    K: dict
    stance: dict | None
    sam3: tuple = ("127.0.0.1", 18832)
    cloud_dir: object = None
    mask_post: object = None
    self_filter: object = None
    row: dict = dataclasses.field(default_factory=dict)
    marks: dict = dataclasses.field(default_factory=dict)
    clock: object = None
    view: object = None

    limit: int | None = None  # the environment's own max_episode_steps (None: it has no TimeLimit)
    limit_eff: int = 0  # the limit this bed applies: the environment's, else bed_simple's table

    def truth(self) -> dict:
        return self.sim.truth()

    def snapshot(self) -> dict:
        if (
            self.task in judge.ART
        ):  # the articulated tasks have no target to follow: the record carries rec["art"] instead
            return dict(t=self.clock.steps * DT, p=np.zeros(3), up=np.array([0.0, 0.0, 1.0]), held=False, others={})
        t = self.truth()
        o = t["objects"]
        Rm = np.zeros(9)
        mujoco.mju_quat2Mat(Rm, np.asarray(o["target"][3:7], float))
        s = dict(t=self.clock.steps * DT, p=np.asarray(o["target"][:3], float), up=Rm.reshape(3, 3)[:, 2].copy(),
                 held=bool(t["held"]))  # fmt: skip
        if "container" in o:
            s["basket"] = np.asarray(o["container"][:3], float)
        s["others"] = {k: np.asarray(o[k][:3], float) for k in self.info["objects"] if k not in ("target", "container") and k in o}  # fmt: skip
        return s

    def watch(self) -> None:
        pass

    def mark(self, name: str) -> None:
        self.marks[name] = self.snapshot()
        if name == "before_close" and self.sim is not None:  # record only: where the box really is against the palm
            t = self.truth()["objects"]["target"][:3] + [0.0, 0.0, self.sim.zoff]
            p, Rm = self.mo.jaw_centre("right")
            self.row["close_geom"] = dict(target=np.round(t, 3).tolist(), palm=np.round(p, 3).tolist(),
                                          target_in_palm=np.round(Rm.T @ (t - p), 3).tolist(), dist_cm=round(float(100 * np.linalg.norm(t - p)), 1))  # fmt: skip

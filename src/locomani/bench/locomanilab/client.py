"""The bed's side of the socket to LocoManiLab's environment (TODO 49.4): the ScaleBFM runtime's Sim with the benchmark's own
environment (server.py) as the physics, the environment's step count and check as the clock, and the episode facade a skill
package sees.

Moved from bed_lml.py (link / ask / Clock / Env / sim_body), lml_sim.py (LmlSim, scene_of) and the facade parts of
bed_lml_cam.py. `LmlSim.step` sends ScaleBFM's joint targets with its gains to the environment's 43 motors and writes the
whole state that comes back (robot, objects, doors) into this model, a copy of the environment's compiled scene (scene.mirror:
the robot first, Unitree's names, the free joint at the pelvis). So this model is the robot's kinematic model and, for the bed,
the place the scene is read from. A skill package therefore never gets the Sim: it gets `PackageEnv` (a few fields), `Hand`
(the fingers' commands and readings) and, through the Belief, the head camera's image.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from multiprocessing.connection import Client
from pathlib import Path

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.motor import Motor
from locomani.bench.locomanilab import comply as CL
from locomani.bench.locomanilab.config import LocoManiLabConfig
from locomani.bench.locomanilab.scene import PREFIX, mirror
from locomani.perception import head
from locomani.perception.sensor import Frame

DT = 0.02
HERE = Path(__file__).resolve().parent
SERVER = HERE / "server.py"
CAM = "head_camera"  # the robot model's name of the head camera (the server's own is robot0_head_camera)
CAM_WH = (320, 240)  # the server's camera size (server.py: camera_widths=320, camera_heights=240)


class Done(Exception):
    """The episode ended: its reason is the message (fell, success...)."""


class EpisodeOver(Exception):
    """The environment's own verdict is in (its check held, or its step limit): the sim takes no more steps."""


# ---- the environment server of this process


def die_with_parent() -> None:
    """In the server just before it starts: SIGTERM when this process goes (a crashed run left its server behind,
    2 GB each, six of them on 11-17)."""
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def server_command(cfg: LocoManiLabConfig, sock: str, srv_dir: Path) -> list[str]:
    """The server's command line: its venv's python, the file, and the arguments that carry what used to be the environment
    (the benchmark's root and the seed wall: LocoManiLabConfig.seed_floor)."""
    return [cfg.lml_python, cfg.server_file or str(SERVER), sock, str(srv_dir), cfg.lml_root, str(cfg.seed_floor()), "1" if cfg.kinds == "Q" else "0"]  # fmt: skip


class ServerLink:
    """This process's LocoManiLab environment server: started on first use, one client, dies with this process."""

    def __init__(self, cfg: LocoManiLabConfig, srv_dir: Path):
        self.cfg, self.srv_dir, self.conn, self.proc = cfg, Path(srv_dir), None, None

    def link(self):
        if self.conn is None:
            sock = f"{self.cfg.sock_dir or tempfile.gettempdir()}/lml_{os.getpid()}.sock"
            self.srv_dir.mkdir(parents=True, exist_ok=True)
            if os.path.exists(sock):  # (a server of this process that died: its socket file is still there)
                os.unlink(sock)
            lml = self.cfg.lml_root
            env = dict(
                os.environ, PYTHONPATH=f"{lml}/external/g1_backend:{lml}/external/humanoid_benchmark", MUJOCO_GL="egl"
            )
            log = open(self.srv_dir / f"server_{os.getpid()}.log", "w")
            proc = subprocess.Popen(server_command(self.cfg, sock, self.srv_dir), env=env, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=lml, preexec_fn=die_with_parent)  # fmt: skip
            t0 = time.time()
            while not os.path.exists(sock):
                if proc.poll() is not None or time.time() - t0 >= self.cfg.server_ready_s:
                    proc.terminate()
                    raise RuntimeError("the LocoManiLab server did not come up")
                time.sleep(0.5)
            self.conn, self.proc = Client(sock, family="AF_UNIX"), proc
        return self.conn

    def reset_link(self) -> None:
        """Drop this process's environment server (dead or in an unknown state): the next `ask` starts a new one."""
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.conn = self.proc = None

    def dead(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None

    def ask(self, *msg) -> dict:
        c = self.link()
        for attempt in range(3):
            c.send(msg)
            rep = c.recv()
            # two servers building one scene for the first time read each other's half-written arena file (seen once,
            # 11-17: "ParseError: unclosed token"): a reset is tried again
            if "error" not in rep or msg[0] != "reset" or attempt == 2 or "PermissionError" in rep["error"]:
                break
            time.sleep(3.0)
        if "error" in rep:
            raise AssertionError(rep.get("error"))
        return rep


def scene_of(reply: dict, out: Path) -> Path:
    """The scene directory for a reset's exported model (written once per task)."""
    if not (out / "scene_locomani.xml").exists():
        mirror(reply["xml"], out)
    return out


# ---- the robot's model with the environment as its physics


class LmlSim(R.Sim):
    rgbd = None  # a perception.rgbd.RgbdSensor over the one head camera (set by `use_rgbd`)
    last_pose = None  # record only: the true camera pose of the last image (for the recorder's cam_err; never given to a package)

    def __init__(self, meta, link, first: dict, scene_dir: Path):
        super().__init__(meta, backend=ScaleBfmBackendConfig(scene_dir=str(scene_dir), object_name=""))
        self.link, self.first, self.over, self.steps, self.success = link, first, None, 0, False
        self.up = np.array(json.loads((scene_dir / "scene_meta.json").read_text())["lml"]["root_up"])
        m = self.m
        cq, sq, cv, sv = [], [], [], []
        for jt in first["joints"]:  # the environment's joint -> this model's, by name
            name, kind, qadr, dadr = jt["name"], jt["type"], jt["qpos_adr"], jt["dof_adr"]
            j = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, name[len(PREFIX) :] if name.startswith(PREFIX) else name
            )
            assert j >= 0, f"joint {name} of the environment is missing in the mirrored scene"
            nq, nv = (7, 6) if int(kind) == 0 else (1, 1)
            cq += range(m.jnt_qposadr[j], m.jnt_qposadr[j] + nq)
            sq += range(int(qadr), int(qadr) + nq)
            cv += range(m.jnt_dofadr[j], m.jnt_dofadr[j] + nv)
            sv += range(int(dadr), int(dadr) + nv)
        self.cq, self.sq, self.cv, self.sv = (np.array(a) for a in (cq, sq, cv, sv))
        assert len(cq) == m.nq and len(cv) == m.nv, "the two models differ in their joints"
        act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, a["joint"][len(PREFIX) :]) for a in first["actuators"]]
        slot = {int(m.jnt_qposadr[j]): k for k, j in enumerate(act)}  # this model's qpos address -> their motor
        self.body_slot = np.array([slot[int(a)] for a in self.qadr])
        self.hand_slot = np.array([slot[int(a)] for a in self.gq])
        self.kp43, self.kd43 = np.zeros(len(act)), np.zeros(len(act))
        self.kp43[self.body_slot], self.kd43[self.body_slot] = self.kp, self.kd
        self.kp43[self.hand_slot], self.kd43[self.hand_slot] = np.tile(R.DEX3_KP, 2), R.DEX3_KD
        self.limits = dict(
            theirs=np.array([a["ctrlrange"][1] for a in first["actuators"]])[self.body_slot], ours=self.tlim
        )

    def pull(self, st: dict | None = None) -> None:
        """The environment's state into this model; the root moves from their floor-level frame to the pelvis."""
        st, d = st or self.first, self.d
        d.qpos[self.cq], d.qvel[self.cv] = np.asarray(st["qpos"])[self.sq], np.asarray(st["qvel"])[self.sv]
        Rm = R.mat(d.qpos[3:7])
        d.qvel[:3] += Rm @ np.cross(
            d.qvel[3:6], self.up
        )  # the pelvis's velocity from the root's (angular in body frame)
        d.qpos[:3] += Rm @ self.up
        mujoco.mj_forward(self.m, d)

    def start(self) -> tuple:
        """Where the environment put the robot: (xy, yaw) of the pelvis."""
        q = np.asarray(self.first["qpos"])
        a = next(int(j["qpos_adr"]) for j in self.first["joints"] if j["name"] == PREFIX + "floating_base_joint")
        return q[a : a + 2].copy(), float(R.yaw_of(q[a + 3 : a + 7]))

    def reset(self, spawn_xy, spawn_yaw, crate_dy):
        super().reset(spawn_xy, spawn_yaw, crate_dy)
        self.pull()

    def step(self, tgt):
        if self.over is not None:
            raise EpisodeOver
        q = np.zeros(len(self.kp43))
        q[self.body_slot], q[self.hand_slot] = tgt, self.grip_target
        self.link.send(("step", q, self.kp43, self.kd43))
        st = self.link.recv()
        self.pull(st)
        self.steps, self.success = int(st["steps"]), bool(st["success"])
        if st["success"] or st["truncated"]:
            self.over = dict(success=bool(st["success"]), steps=self.steps, truncated=bool(st["truncated"]))

    def eyes(self) -> dict:
        self.link.send(("eyes",))
        return self.link.recv()

    def use_rgbd(self, depth_noise: str, seed: int) -> None:
        """sensor=rgbd: the camera's image and the renderer's exact depth, as is (depth_noise=none: the loops so far) or through
        the RealSense-like noise model. The one camera has no second imager, so the model casts no occlusion shadows (it is
        given no right-camera depth); `seed` makes a rerun of an episode see the same noise."""
        from locomani.perception.rgbd import RgbdSensor

        self.rgbd = RgbdSensor(self._exact_rgbd, depth_noise, seed)

    def _exact_rgbd(self) -> Frame:
        """One render: the image, the renderer's metric depth (m along the optical axis) and the field of view of this frame, as a
        Frame (K from that field of view, the camera pose from the robot's own forward kinematics)."""
        e = self.eyes()
        self.last_pose = (np.asarray(e["cam_pos"], float), np.asarray(e["cam_mat"], float))  # record only
        self.last_fovy = float(e["fovy"])
        if getattr(self, "_kin", None) is None:
            self._kin = head.HeadKin(self.m, self.qadr, CAM)
        p, Rc = self._kin(self.d.qpos[self.qadr])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rc, p
        return Frame(
            rgb=e["rgb"],
            depth=e["depth"],
            K=head.intrinsics(CAM_WH, self.last_fovy),
            T_body_cam=T,
            source="rgbd",
            t=self.steps * DT,
        )

    def camera_image(self) -> dict:
        """What the Belief reads of the head camera: colour, depth and the field of view. (The camera's true pose is kept in
        `last_pose` for the recorder; nothing else of the reply leaves this class.)"""
        f = self.rgbd.frame()
        return dict(rgb=f.rgb, depth=f.depth, fovy=self.last_fovy)


class BedMotor(Motor):
    """The Motor with the episode's clock in its tick: every 20 ms step the Motor takes goes through `clock.tick`, which counts it,
    feeds the judge and raises Done when the robot falls (the old bed replaced `mo.tick` on the instance)."""

    clock = None

    def raw_tick(self) -> bool:
        return super().tick()

    def tick(self) -> bool:
        return self.raw_tick() if self.clock is None else self.clock.tick()


# ---- the episode


def sim_body(env) -> int:
    return env.sim.m.body(env.info["target"]).id


def root_of(m, name: str) -> str:
    """The body that carries the object's free joint: `name` or the nearest body above it that has a joint."""
    b = m.body(name).id
    while m.body_jntnum[b] == 0 and m.body_parentid[b] != 0:
        b = int(m.body_parentid[b])
    return m.body(b).name


class Clock:
    """Counted seconds per phase; the step count and the verdict are the environment's. Also feeds comply's Watcher once per
    control step with the local model's contacts, right after the environment's check of that step."""

    def __init__(self, env):
        self.env, self.label, self.phases, self.bk = env, "start", {}, None
        self.watcher, self.watch_error, self._ids = CL.Watcher(DT, strict=env.spec["family"] in CL.ART), None, None
        self._tick = env.mo.raw_tick
        env.mo.clock = self  # BedMotor.tick calls this clock (the old bed assigned `mo.tick` itself)

    @property
    def steps(self) -> int:
        return self.env.sim.steps

    @contextmanager
    def phase(self, name: str):
        prev, self.label = self.label, name
        try:
            yield
        finally:
            self.label = prev

    def feed(self) -> None:
        """One counted step: the contacts of the target, the benchmark's check of this step, whether the robot is down."""
        env, sim = self.env, self.env.sim
        try:
            if self._ids is None:  # (the target's body, the receptacle's body or None) in this model
                rb = (env.spec.get("target") or {}).get("body")
                self._ids = (sim_body(env), sim.m.body(rb).id if rb else None)
            ck = bool(sim.success)
            nav = env.spec["family"] in CL.NAV
            art = (
                env.spec["family"] in CL.ART
            )  # a door or drawer: the target is the moving part's body, a fall counts at any step
            snap = None if nav else CL.snapshot(sim.m, sim.d, self._ids[0], self._ids[1])
            self.watcher.feed(sim.steps * DT, snap, ck, fallen=bool(sim.fallen()) if art else ck and bool(sim.fallen()))
            if art and getattr(env, "art", None) is not None:
                env.art.step(sim.steps * DT)
        except Exception as e:  # noqa: BLE001  counted in the record, the episode goes on
            self.watch_error = self.watch_error or f"{type(e).__name__}: {e}"[:120]

    def tick(self) -> bool:
        ok = self._tick()
        if not ok:
            raise Done("fell")
        self.phases[self.label] = self.phases.get(self.label, 0) + 1
        self.feed()
        return ok


class Hand:
    """What a skill package may do with the fingers: command them (`cmd`: the Sim's own dict of the sides' seven joint targets),
    read their joint encoders and the grip target. Not the Sim: nothing else of it is reachable from here."""

    def __init__(self, sim):
        self._sim = sim

    @property
    def cmd(self) -> dict:
        return self._sim.fingers_cmd

    def q(self, side: str) -> np.ndarray:
        """The seven finger joints' readings of one hand."""
        sim = self._sim
        return sim.d.qpos[sim.gq[7 * ("left", "right").index(side) :][:7]].copy()

    def grip_target(self, side: str) -> np.ndarray:
        return self._sim.grip_target[7 * ("left", "right").index(side) :][:7]


class PhaseClock:
    """The part of the clock a skill package uses: `phase(name)` books the counted seconds to a phase."""

    def __init__(self, clock):
        self._clock = clock

    def phase(self, name: str):
        return self._clock.phase(name)

    @property
    def steps(self) -> int:
        return self._clock.steps


class GraspRecorded:
    """The package's view as the skill sees it, with the chosen grasp's distance from the truth track's written to the record (record
    only: `grasp_vs_truth_cm`). Everything else is the view's own."""

    def __init__(self, view, on_grasp):
        self.__dict__["_view"], self.__dict__["_on_grasp"] = view, on_grasp

    def grasp(self, tgt, yaw: float):
        g = self._view.grasp(tgt, yaw)
        self._on_grasp(tgt, g)
        return g

    def __getattr__(self, name):
        return getattr(self._view, name)

    def __setattr__(self, name, value):
        setattr(self._view, name, value)


class PackageEnv:
    """What a skill package gets of the episode (the old bed handed it the whole `Env`): the parameters `K`, the stance, the record
    `row`, the parsed instruction `want`, SAM 3's address and the look cache folder, the Belief, the Motor, the view, `hand`, the clock's phases, `mark` and `flow_done`."""

    def __init__(self, env: Env):
        self._env = env
        self.K, self.stance, self.row, self.want = env.K, env.stance, env.row, env.want
        self.sam3, self.cloud_dir = env.sam3, env.cloud_dir  # what the camera view needs of the run
        self.bel, self.mo, self.hand = env.bel, env.mo, Hand(env.sim)
        self.clock = PhaseClock(env.clock)

    @property
    def view(self):
        return self._env.view

    def mark(self, name: str) -> None:
        self._env.mark(name)

    def flow_done(self, name: str) -> None:
        """A flow of the package ended (`approach` | `pick`): the bed's recorder reads its own numbers (record only)."""
        self._env.flow_done(name)


@dataclasses.dataclass
class Env:
    """The bed's whole view of one episode: the sim, the judge's inputs, the record. Not given to a skill package (`facade`)."""

    task: str
    info: dict
    sim: object
    mo: object
    bel: object
    K: dict
    stance: dict | None
    row: dict = dataclasses.field(default_factory=dict)
    marks: dict = dataclasses.field(default_factory=dict)
    clock: Clock = None
    view: object = None
    bodies: set = None  # the target and the bodies below it
    spec: dict = None  # the task's spec (specs.json)
    art: object = None  # the recorder of an articulated task, record.ArtRecorder
    rec: object = None  # the recorder, record.Recorder
    want: dict = None  # the parsed instruction
    prompt: str = ""
    skill: object = None  # the package's skill module (the bed's close_desc record reads its `grasp_desc`)
    sam3: tuple = ("127.0.0.1", 18832)  # SAM 3's address
    cloud_dir: object = None  # the look cache folder of this package and these parameters
    marked: object = None  # hook called with the name after every mark (record-only comparisons beyond the recorder's)

    def facade(self) -> PackageEnv:
        return PackageEnv(self)

    def snapshot(self) -> dict:
        sim, b = self.sim, sim_body(self)
        m, d = sim.m, sim.d
        held = any(({int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} & self.bodies)
                   and any("_hand_" in m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2))
                   for c in d.contact[: d.ncon])  # fmt: skip
        return dict(
            t=self.clock.steps * DT,
            p=d.xpos[b].copy(),
            up=d.xmat[b].reshape(3, 3)[:, 2].copy(),
            held=bool(held),
            others={},
        )

    def mark(self, name: str) -> None:
        self.marks[name] = self.snapshot()
        if self.marked is not None:
            self.marked(name)

    def flow_done(self, name: str) -> None:
        if self.rec is not None:
            {"approach": self.rec.after_approach, "pick": self.rec.after_pick}[name]()


def free(env: Env) -> None:
    env.mo.clock = None  # the clock holds the Motor and the Motor's tick is the clock's: break the ring, or the scene's
    env.sim = env.mo = env.bel = env.view = env.clock = None  # compiled model (about 180 MB) waits for the cycle
    import gc

    gc.collect()  # collector and a worker runs out of memory

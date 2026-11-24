"""The bed's side of the environment server, and what a skill package gets of an episode (TODO 49.5b).

`ServerLink` starts the environment server (server.py, the benchmark's own venv) on first use, speaks to it over one unix socket
and kills it when this process ends. `Episode` is the bed's whole view of one episode: the link, the clock, the judge's inputs. The
skill package never gets it: `PackageEnv` is the facade, built from it:

    env.K, env.task, env.instruction          the package's parameters and what the task says
    env.view                                  the truth view (truth.TruthView): `env.view.goal()`
    env.me                                    the robot's own readings and arm kinematics (`Me`)
    env.drive(...)                            one 50 Hz control step in the native 40-D action space; raises Done when the episode is over
    env.mark(name)                            a named moment (recorded)
    env.cameras                               `Cameras`: `rgbd()`
    env.gait                                  the walking cycle of the benchmark's example motions (`Gait`)

The native action (vla_robot_current_local_runtime_v3.py of the benchmark): root x/y delta in the reference heading frame (2), root
height (1), root rotation relative to the episode's start heading as rot6d (6), the 29 canonical joint positions (29), the two hand
open/close values (2). The TWIST2 tracker follows it. The facade holds no link, no Privileged and no outcome.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from locomani.bench.humanoidarena.config import HumanoidArenaConfig
from locomani.perception.sensor import Frame

DT = 0.02  # the benchmark's control step, 50 Hz


class Done(Exception):
    """The episode is over (success, a fall, the budget). Carries nothing a package may read."""


class ServerError(RuntimeError):
    pass


def server_command(cfg: HumanoidArenaConfig, sock: str) -> list[str]:
    return [
        "taskset", "-c", cfg.cores, cfg.ha_python, cfg.server_script(), sock,
        "--root", f"{cfg.ha_root}/repo/isaaclab_twist2_g1", "--kit-gpu", str(cfg.gpu),
    ]  # fmt: skip


def server_env(cfg: HumanoidArenaConfig) -> dict:
    usd = f"{cfg.ha_root}/repo/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2_thumd.usd"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg.gpu), OMNI_KIT_ACCEPT_EULA="YES", PYTHONNOUSERSITE="1",
               LD_LIBRARY_PATH=f"{cfg.ha_root}/cdds/lib", TMPDIR=f"{cfg.ha_root}/tmp",
               PYTHONPATH=f"{cfg.ha_root}/repo/isaaclab_twist2_g1", ROBOT_USD_OVERRIDE=usd)  # fmt: skip
    return env


def die_with_parent():
    import ctypes

    ctypes.CDLL("libc.so.6").prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM


class ServerLink:
    """This process's HumanoidArena environment server: started on first use, one client, dies with this process."""

    def __init__(self, cfg: HumanoidArenaConfig):
        self.cfg, self.conn, self.proc, self.log = cfg, None, None, None

    def link(self):
        if self.conn is None and self.cfg.attach:
            self.conn = Client(self.cfg.attach, family="AF_UNIX")
        if self.conn is None:
            run = Path(self.cfg.run_dir or tempfile.gettempdir())
            run.mkdir(parents=True, exist_ok=True)
            sock = str(run / f"humanoidarena_{os.getpid()}.sock")
            if os.path.exists(sock):
                os.unlink(sock)
            self.log = open(run / f"humanoidarena_server_{os.getpid()}.log", "w")
            self.proc = subprocess.Popen(server_command(self.cfg, sock), env=server_env(self.cfg), stdout=self.log, stderr=subprocess.STDOUT,
                                         cwd=self.cfg.ha_root, preexec_fn=die_with_parent)  # fmt: skip
            t0 = time.time()
            while not os.path.exists(sock):
                if self.proc.poll() is not None or time.time() - t0 >= self.cfg.server_ready_s:
                    self.proc.terminate()
                    raise ServerError("the HumanoidArena server did not come up (see its log in the run folder)")
                time.sleep(0.2)
            self.conn = Client(sock, family="AF_UNIX")
        return self.conn

    def ask(self, *msg):
        c = self.link()
        c.send(
            tuple(m.tolist() if isinstance(m, np.ndarray) else m for m in msg)
        )  # the server's numpy (1.26) cannot unpickle ours (2.x)
        rep = c.recv()
        if isinstance(rep, dict) and "error" in rep:
            raise ServerError(rep["error"])
        return rep

    def close(self) -> None:
        if self.cfg.attach:  # a server the bed did not start is left running
            self.conn = None
            return
        if self.conn is not None:
            try:
                self.conn.send(("close",))
                self.conn.recv()
            except (OSError, EOFError):
                pass
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log is not None:
            self.log.close()
        self.conn = self.proc = self.log = None


def yaw_of(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rot6d_of_yaw(yaw: float) -> np.ndarray:
    """The benchmark's rot6d of a rotation about z: the matrix's first two columns, row-major (identity -> [1 0 0 1 0 0]; checked against
    the policy state the server reports at the start)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c, -s, s, c, 0.0, 0.0], np.float32)


class Me:
    """The robot's own readings and kinematics: the root, the wrists, the arm joints and their Jacobians. These are what the robot
    knows of itself (self_pose=truth: in the benchmark's world frame); nothing of the scene."""

    def __init__(self, ep: Episode):
        self._ep = ep

    def _st(self) -> dict:
        return self._ep.state

    def pelvis(self) -> tuple[np.ndarray, float]:
        r = np.asarray(self._st()["root"], float)
        return r[:3].copy(), yaw_of(r[3:7])

    def wrist(self, side: str) -> np.ndarray:
        return np.asarray(self._st()["hands"][side], float).copy()

    def policy_state(self) -> np.ndarray:
        """The 64-D state of the benchmark's policy: root rot6d (6), canonical joint positions (29), velocities (29)."""
        return np.asarray(self._st()["policy_state"], float).copy()

    def joints(self) -> np.ndarray:
        return self.policy_state()[6:35]

    def arm_ids(self, side: str) -> list[int]:
        """Indices of the arm's seven joints in the canonical 29 (what the action's joint slots use)."""
        return list(self._ep.info["arm_canon"][side])

    def jac(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(3 x 7 position Jacobian of the wrist in the world frame over the arm's joints, the wrist position)."""
        r = self._ep.link.ask("jac", side)
        return np.asarray(r["J"], float), np.asarray(r["hand"], float)


class Cameras:
    """bench/channels.py `Cameras` on HumanoidArena: the one front camera (640 x 480, D435 pose), no stereo."""

    def __init__(self, ep: Episode):
        self._ep = ep

    def stereo(self):
        raise NotImplementedError("HumanoidArena has one front camera; no stereo pair")

    def rgbd(self) -> Frame:
        """The front camera's image with the renderer's exact depth (metres along the optical axis). K is the camera's own;
        T_body_cam is x right, y down, z forward in the pelvis frame."""
        r = self._ep.link.ask("rgbd")
        return Frame(rgb=r["rgb"], depth=r["depth"], K=np.asarray(r["K"], float), T_body_cam=np.asarray(r["T_body_cam"], float),
                     source="rgbd", t=float(r["t"]))  # fmt: skip


class Gait:
    """The walking cycle of the benchmark's own example motions in the action's joint order (gait_build.py). The tracker needs the legs of its
    reference to show a gait: moving the reference root alone leaves the robot standing. Not scene knowledge; any package may read it."""

    def __init__(self, path=None):
        import json

        d = json.loads(Path(path or Path(__file__).with_name("gait_cycle.json")).read_text())
        self.cycle = np.asarray(d["cycle"], float)  # (n, 29) canonical joints over one stride
        self.period, self.speed, self.n = (
            float(d["period_s"]),
            float(d["speed"]),
            int(d["n"]),
        )  # s, the cycle's mean forward speed m/s

    def pose(self, phase: float) -> np.ndarray:
        """The 29 joint positions at `phase` in [0, 1) of the stride (linear between samples)."""
        k = (phase % 1.0) * self.n
        k0 = int(k) % self.n
        f = k - int(k)
        return (1 - f) * self.cycle[k0] + f * self.cycle[(k0 + 1) % self.n]


@dataclass
class Episode:
    """The bed's whole view of one episode. Not given to a skill package (`facade`)."""

    task: str
    link: ServerLink
    info: dict  # the reset reply
    K: dict
    state: dict = field(default_factory=dict)
    steps: int = 0
    over: dict | None = None
    marks: dict = field(default_factory=dict)
    notes: dict = field(
        default_factory=dict
    )  # what the package wrote at its marks (readings of its own, for the record)
    max_steps: int = 0
    view: object = None
    on_mark: object = (
        None  # the bed's hook at a package's mark (a picture for a person to look at); the package cannot reach it
    )

    def facade(self) -> PackageEnv:
        return PackageEnv(self)

    def step(self, action) -> dict:
        if self.over is not None:
            raise Done()
        rep = self.link.ask("step", np.asarray(action, np.float32))
        self.state, self.steps = rep["state"], int(rep["steps"])
        if rep["over"] or (self.max_steps and self.steps >= self.max_steps):
            self.over = dict(success=bool(rep["success"]), fell=bool(rep["fell"]), steps=self.steps)
        return rep


class PackageEnv:
    """What a skill package gets of the episode (see the module docstring)."""

    def __init__(self, ep: Episode):
        self._ep = ep
        self.K, self.task, self.instruction = ep.K, ep.task, ep.info["instruction"]
        self.view = ep.view
        self.me = Me(ep)
        self.cameras = Cameras(ep)
        self.gait = Gait()
        self.leg_ids = list(ep.info["leg_canon"])  # the legs' joints in the 29 (hips, knees, ankles)
        self.start_height = float(ep.info["start_height"])  # the pelvis height the episode starts standing at (m)

    @property
    def t(self) -> float:
        return self._ep.steps * DT

    def drive(self, action) -> None:
        """One control step with a 40-D native action (see the module docstring); `hold()` gives one that keeps the robot as it is."""
        self._ep.step(action)
        if self._ep.over is not None:
            raise Done()

    def hold(self, joints=None, dxy=(0.0, 0.0), yaw=0.0, height=None, hands=(0.0, 0.0)) -> np.ndarray:
        """A native action: stand where the robot is. `joints` the 29 canonical joint targets (None: the present ones), `dxy` the
        reference root's step (m, in the heading frame), `yaw` the heading relative to the episode's start (rad), `height` the pelvis
        height (m), `hands` the two hand open/close values (0 open .. 1 closed)."""
        ps = self.me.policy_state()
        q = ps[6:35] if joints is None else np.asarray(joints, float)
        h = self.start_height if height is None else height
        return np.concatenate([np.asarray(dxy, float), [h], rot6d_of_yaw(yaw), q, np.asarray(hands, float)]).astype(
            np.float32
        )

    def mark(self, name: str, note=None) -> None:
        """A named moment (recorded with the step number); `note` is any plain value the package wants kept with it."""
        self._ep.marks.setdefault(name, self._ep.steps)
        if self._ep.on_mark is not None:
            self._ep.on_mark(name)
        if note is not None:
            self._ep.notes.setdefault(name, note)

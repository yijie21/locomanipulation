"""The bed's side of the environment server, and what a skill package gets of an episode (TODO 49.5).

`ServerLink` starts the environment server (server.py, BiGym's own venv) on first use, speaks to it over one unix socket and
kills it when this process ends. `Episode` is the bed's whole view of one episode: the link, the clock, the judge's inputs. The
skill package never gets it: `PackageEnv` is the facade, built from it:

    env.K, env.task, env.instruction          the package's parameters and what the task says
    env.view                                  the truth view (truth.TruthView): `env.view.goal()`
    env.me                                    the robot's own readings and kinematics (`Me`)
    env.drive(...)                            one 50 Hz control step; raises Done when the episode is over
    env.mark(name)                            a named moment (recorded)
    env.cameras                               `Cameras`: `rgbd()` (needs depth on)

The facade holds no link, no Privileged and no outcome. The environment's own limits (budget, fall, success) end an episode by `Done`.
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

from locomani.bench.bigym2.config import EVAL_BLOCK, BiGym2Config
from locomani.perception.sensor import Frame

DT = 0.02  # BiGym's control step, 50 Hz


class Done(Exception):
    """The episode is over (success, a fall that ended it, the budget, a physics error). Carries nothing a package may read."""


class ServerError(RuntimeError):
    pass


def server_command(cfg: BiGym2Config, sock: str) -> list[str]:
    cmd = [
        cfg.bigym_python,
        cfg.server_script(),
        sock,
        "--refuse-lo",
        str(EVAL_BLOCK[0]),
        "--refuse-hi",
        str(EVAL_BLOCK[1]),
    ]
    if cfg.depth:
        cmd += ["--depth", "--shape", str(cfg.render_shape[0]), str(cfg.render_shape[1])]
    return cmd


def die_with_parent():
    import ctypes

    ctypes.CDLL("libc.so.6").prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM


class ServerLink:
    """This process's BiGym environment server: started on first use, one client, dies with this process."""

    def __init__(self, cfg: BiGym2Config):
        self.cfg, self.conn, self.proc, self.log = cfg, None, None, None

    def link(self):
        if self.conn is None:
            run = Path(self.cfg.run_dir or tempfile.gettempdir())
            run.mkdir(parents=True, exist_ok=True)
            sock = str(run / f"bigym2_{os.getpid()}.sock")
            if os.path.exists(sock):
                os.unlink(sock)
            env = dict(os.environ, MUJOCO_GL="egl", EGL_DEVICE_ID=str(self.cfg.gpu), PYTHONNOUSERSITE="1")
            env.pop("PYTHONPATH", None)  # BiGym's venv resolves its own packages
            self.log = open(run / f"bigym2_server_{os.getpid()}.log", "w")
            self.proc = subprocess.Popen(server_command(self.cfg, sock), env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                         cwd=self.cfg.bigym_root, preexec_fn=die_with_parent)  # fmt: skip
            t0 = time.time()
            while not os.path.exists(sock):
                if self.proc.poll() is not None or time.time() - t0 >= self.cfg.server_ready_s:
                    self.proc.terminate()
                    raise ServerError("the BiGym server did not come up (see its log in the run folder)")
                time.sleep(0.2)
            self.conn = Client(sock, family="AF_UNIX")
        return self.conn

    def ask(self, *msg):
        c = self.link()
        c.send(msg)
        rep = c.recv()
        if isinstance(rep, dict) and "error" in rep:
            raise ServerError(rep["error"])
        return rep

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.send(("close",))
                self.conn.recv()
            except (OSError, EOFError):
                pass
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log is not None:
            self.log.close()
        self.conn = self.proc = self.log = None


class Me:
    """The robot's own readings and kinematics: joints, the pinch centres, the pelvis, an arm solution for a hand target. These are
    what the robot knows of itself (self_pose=truth: in BiGym's world frame); nothing of the scene."""

    def __init__(self, ep: Episode):
        self._ep = ep

    def _st(self) -> dict:
        return self._ep.state

    def pelvis(self) -> tuple[np.ndarray, float]:
        T = self._st()["pelvis"]
        return T[:3, 3].copy(), float(np.arctan2(T[1, 0], T[0, 0]))

    def hand(self, side: str) -> np.ndarray:
        return np.asarray(self._st()[f"{side}_hand"], float).copy()

    def hand_R(self, side: str) -> np.ndarray:
        return np.asarray(self._st()[f"{side}_hand_R"], float).copy()

    def arm(self, side: str) -> np.ndarray:
        return np.asarray(self._st()[f"{side}_q"], float).copy()

    def tilt(self) -> float:
        return float(self._st()["tilt"])

    def ik(self, side: str, pos, quat=None, q0=None) -> tuple[np.ndarray, float]:
        """The arm's seven joint angles that put the pinch centre at `pos` (world), facing `quat` (w x y z) when given. -> (q, error m)"""
        rep = self._ep.link.ask(
            "ik", side, np.asarray(pos, float), None if quat is None else np.asarray(quat, float), q0
        )
        return np.asarray(rep["q"], float), float(rep["err"])


class Cameras:
    """bench/channels.py `Cameras` on BiGym: one head camera, no stereo."""

    def __init__(self, ep: Episode):
        self._ep = ep

    def stereo(self):
        raise NotImplementedError("BiGym 2.0 has no stereo head camera")

    def rgbd(self) -> Frame:
        """The head camera's image with the renderer's exact depth (metres along the optical axis). Needs `BiGym2Config.depth`
        (the server's depth patch). K is a pinhole from the camera's vertical field of view; T_body_cam is x right, y down, z forward in the pelvis frame."""
        r = self._ep.link.ask("rgbd", "head")
        h, w = r["depth"].shape
        f = 0.5 * h / np.tan(np.radians(r["fovy"]) / 2)
        K = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]])
        return Frame(
            rgb=r["rgb"],
            depth=r["depth"],
            K=K,
            T_body_cam=np.asarray(r["T_body_cam"], float),
            source="rgbd",
            t=float(r["t"]),
        )


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
    max_steps: int = 0
    view: object = None

    def facade(self) -> PackageEnv:
        return PackageEnv(self)

    def step(self, cmd: dict) -> dict:
        if self.over is not None:
            raise Done()
        rep = self.link.ask("step", cmd)
        self.state, self.steps = rep["state"], int(rep["steps"])
        if rep["over"] or (self.max_steps and self.steps >= self.max_steps):
            self.over = dict(
                success=bool(rep["success"]), fell=bool(rep["fell"]), truncation=rep["truncation"], steps=self.steps
            )
        return rep


class PackageEnv:
    """What a skill package gets of the episode (see the module docstring)."""

    def __init__(self, ep: Episode):
        self._ep = ep
        self.K, self.task, self.instruction = ep.K, ep.task, ep.info["instruction"]
        self.view = ep.view
        self.me = Me(ep)
        self.cameras = Cameras(ep)
        self.pitch = bool(ep.info["pitch"])  # the task has a torso-pitch command

    @property
    def t(self) -> float:
        return self._ep.steps * DT

    def drive(self, vx=0.0, vy=0.0, wz=0.0, height=None, pitch=None, left=None, right=None, grip=None) -> None:
        """One control step. vx, vy (m/s) and wz (rad/s) are the lower body's command, `height` the pelvis height (m, 0.4-1.0), `left` and
        `right` the arms' seven joint targets (rad; None keeps the arm where it is), `grip` the two grippers (0 open .. 1 closed)."""
        cmd = dict(
            vx=float(vx), vy=float(vy), wz=float(wz), height=height, pitch=pitch, left=left, right=right, grip=grip
        )
        self._ep.step(cmd)
        if self._ep.over is not None:
            raise Done()

    def mark(self, name: str) -> None:
        self._ep.marks.setdefault(name, self._ep.steps)

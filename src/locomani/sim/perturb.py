"""Sim-side perturbations for the SONIC -> ScaleBFM comparison (TODO 25.5, plan §4.6), one spec for both sims.

The spec is a dict (on the SONIC sim: the JSON in env LOCOMANI_PERTURB, read by run_sim_locomani; on ScaleBFM:
`runtime.Sim(perturb=...)`); every key is optional and off when absent:

* `delay_s`    b -- action delay: the PD targets and gains the motors apply are the ones the controller sent `delay_s`
                    of simulated time earlier (rounded to the physics step).
* `noise`      c -- {"gyro": rad/s, "dq": rad/s, "q": rad}: uniform +-noise added, independently every tick, to the
                    pelvis angular velocity and the joint velocities / positions the controller reads.
* `friction`   d -- sliding friction of the floor and of the feet (MuJoCo takes the larger of a pair).
* `gain_scale` f -- [lo, hi]: every body joint's kp and kd multiplied by one factor drawn uniformly in [lo, hi] per
                    episode (SONIC: every episode-server reset, ScaleBFM: every Sim.reset).
* `push_every_s`, `push_n`, `push_s`
               g -- a horizontal push in a random direction on the pelvis every push_every_s of the episode, magnitude
                    uniform in push_n ([lo, hi] N), push_s long.
* `seed`          -- the draws.
Single pushes are requests: `push(force_xyz, seconds)` puts the force on the pelvis for that long (SONIC:
EpisodeControlClient.push -> the episode server -> PUSH). Payload (e) is the object's mass, a scene change.
"""

from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

ENV = "LOCOMANI_PERTURB"
PUSH = {"force": None, "steps": 0}  # the pending pelvis push, applied by the torque hook every physics step
CURRENT: list = []  # the SONIC sim's Perturbation (run_sim_locomani), redrawn by on_reset()


def from_env() -> dict | None:
    raw = os.environ.get(ENV, "").strip()
    return json.loads(raw) if raw else None


class Perturbation:
    def __init__(self, spec: dict | None, n_joints: int, physics_dt: float):
        self.spec = dict(spec or {})
        self.rng = np.random.default_rng(self.spec.get("seed", 0))
        self.delay_steps = int(round(float(self.spec.get("delay_s", 0.0)) / physics_dt))
        self.noise = dict(self.spec.get("noise") or {})
        self.n = n_joints
        self.gain = np.ones(n_joints)
        self.redraw()
        self.queue: deque = deque(maxlen=self.delay_steps + 1)
        self.dt = physics_dt
        self.steps = 0  # physics steps since the episode began (the periodic push's clock)

    def tick(self) -> None:
        """Every physics step: the periodic push (`push_every_s`, magnitude uniform in `push_n` N, horizontal,
        random direction, `push_s` long; the first one push_every_s after the episode starts)."""
        self.steps += 1
        every = self.spec.get("push_every_s")
        if not every:
            return
        n = int(round(float(every) / self.dt))
        if self.steps % n == 0:
            lo, hi = self.spec.get("push_n", [50.0, 100.0])
            a = self.rng.uniform(0, 2 * np.pi)
            f = self.rng.uniform(lo, hi)
            request_push([f * np.cos(a), f * np.sin(a), 0.0], float(self.spec.get("push_s", 0.2)), self.dt)

    def redraw(self) -> None:
        lo_hi = self.spec.get("gain_scale")
        self.gain = self.rng.uniform(lo_hi[0], lo_hi[1], self.n) if lo_hi else np.ones(self.n)

    def active(self) -> bool:
        return bool(self.delay_steps or self.noise or self.spec.get("gain_scale") or self.spec.get("friction"))

    def delayed(self, command):
        """Push this step's command (any object) and return the one `delay_steps` physics steps old (the oldest
        available while the queue fills)."""
        if not self.delay_steps:
            return command
        self.queue.append(command)
        return self.queue[0]

    def noisy(self, key: str, value):
        a = float(self.noise.get(key, 0.0))
        if not a:
            return value
        v = np.asarray(value, dtype=float)
        return v + self.rng.uniform(-a, a, v.shape)

    def describe(self) -> str:
        return json.dumps(
            dict(self.spec, gain_drawn=[round(float(g), 3) for g in self.gain] if self.spec.get("gain_scale") else None)
        )


def on_reset() -> None:
    """An episode reset on the SONIC sim: new gain draw, empty delay queue, no pending push."""
    for p in CURRENT:
        p.redraw()
        p.queue.clear()
        p.steps = 0
    PUSH["force"], PUSH["steps"] = None, 0


def set_friction(model, friction: float, foot_bodies) -> list[int]:
    """Floor (world-body planes) and foot collision geoms get `friction` as their sliding coefficient."""
    import mujoco

    feet = {model.body(b).id for b in foot_bodies}
    changed = []
    for g in range(model.ngeom):
        world_plane = model.geom_bodyid[g] == 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        if world_plane or (model.geom_bodyid[g] in feet and model.geom_contype[g]):
            model.geom_friction[g, 0] = friction
            changed.append(g)
    return changed


def request_push(force, seconds: float, physics_dt: float) -> None:
    PUSH["force"] = [float(v) for v in force]
    PUSH["steps"] = max(1, int(round(seconds / physics_dt)))


def apply_push(data, pelvis_body: int) -> None:
    """Every physics step: the pending push on the pelvis, cleared when its steps run out."""
    if PUSH["steps"] > 0 and PUSH["force"] is not None:
        data.xfrc_applied[pelvis_body, :3] = PUSH["force"]
        PUSH["steps"] -= 1
        if PUSH["steps"] == 0:
            data.xfrc_applied[pelvis_body, :3] = 0.0
            PUSH["force"] = None

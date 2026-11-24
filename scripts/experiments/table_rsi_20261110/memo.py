"""TODO 46.1: a memoised simulator. Rounds that start the same way share the simulation of their common beginning.

A round is a sequence of top-level motor commands (walk_to, reach, grip ...). The simulation is deterministic, so two
rounds from the same start whose first n commands are identical are in the same state after them. `install(world)`
wraps the motor's primitives: each command's key chains the start's label, every earlier command and its own arguments;
on a hit the state after the command is restored (the MuJoCo data, the policy's history, the motor's own fields) and
the recorded outcome returned, without stepping. Skill code runs unchanged: all its Python runs again, only the
stepping is skipped. A variant that changes the grasp replays the walk for free.

The chain breaks (no more hits or stores for the round) when the simulation was stepped outside a primitive, as a
hand program does through `tick`. Set `memo.label` before a round: the motor's `reset` starts the chain from it; `clear()` drops the stored states (about 8 MB
each on the ADT apartment).
"""

from __future__ import annotations

import copy
import hashlib
import json

import mujoco
import numpy as np

PRIMITIVES = ("walk_to", "turn", "sidestep", "posture", "reach", "adopt_hand", "grip", "release", "hold")
KEEP = ("sim", "pol", "cfg", "kp", "ik", "jacobian", "track", "foot_bodies", "fingers", "reach_note", "reach_offset",
        "reach_route", "trace", "calls", "tick", "reset", "_call_depth", *PRIMITIVES)  # fmt: skip


def _plain(x):
    if isinstance(x, np.ndarray):
        return ["nd", x.shape, x.astype(float).tobytes().hex()]
    if isinstance(x, (np.floating, float)):
        return float(x).hex()
    if isinstance(x, (np.integer, int, bool, str, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items())}
    return repr(x)


class Memo:
    def __init__(self, world):
        self.world, self.m = world, world.motor
        self.store: dict[str, dict] = {}
        self.key, self.live, self.k_seen, self.depth = "", False, 0, 0
        self.hits = self.misses = self.saved_ticks = 0
        for name in PRIMITIVES:
            setattr(self.m, name, self._wrap(name, getattr(self.m, name)))
        self.label, reset = "", self.m.reset

        def fresh(*args, **kw):
            reset(*args, **kw)
            self.begin(self.label)

        self.m.reset = fresh

    def begin(self, label: str) -> None:
        self.key = hashlib.sha256(f"start:{label}".encode()).hexdigest()
        self.live, self.k_seen = True, self.m.k

    def clear(self) -> None:
        self.store.clear()

    def _snapshot(self, out, n_trace: int, n_calls: int) -> dict:
        m, sim, pol = self.m, self.m.sim, self.m.pol
        data = mujoco.MjData(sim.m)
        mujoco.mj_copyData(data, sim.m, sim.d)
        return dict(
            data=data,
            grip_target=sim.grip_target.copy(),
            buf={k: v.copy() for k, v in pol.buf.items()},
            mode=pol.mode.clone(),
            motor={k: copy.deepcopy(v) for k, v in vars(m).items() if k not in KEEP},
            trace=m.trace[n_trace:],
            calls=copy.deepcopy(m.calls[n_calls:]),
            out=copy.deepcopy(out),
        )

    def _restore(self, s: dict):
        m, sim, pol = self.m, self.m.sim, self.m.pol
        self.saved_ticks += s["motor"]["k"] - m.k
        mujoco.mj_copyData(sim.d, sim.m, s["data"])
        sim.grip_target[:] = s["grip_target"]
        pol.buf = {k: v.copy() for k, v in s["buf"].items()}
        pol.mode = s["mode"].clone()
        for k, v in s["motor"].items():
            setattr(m, k, copy.deepcopy(v))
        m.trace.extend(s["trace"])
        m.calls.extend(copy.deepcopy(s["calls"]))
        return copy.deepcopy(s["out"])

    def _wrap(self, name: str, fn):
        def call(*args, **kw):
            m = self.m
            if self.depth or not self.live:
                return fn(*args, **kw)
            if m.k != self.k_seen:  # stepped outside a primitive: the state is no longer a function of the commands
                self.live = False
                return fn(*args, **kw)
            self.key = hashlib.sha256(json.dumps([self.key, name, _plain(args), _plain(kw)]).encode()).hexdigest()
            hit = self.store.get(self.key)
            if hit is not None:
                self.hits += 1
                out = self._restore(hit)
                self.k_seen = m.k
                return out
            self.misses += 1
            n_trace, n_calls = len(m.trace), len(m.calls)
            self.depth += 1
            try:
                out = fn(*args, **kw)
            finally:
                self.depth -= 1
            self.store[self.key] = self._snapshot(out, n_trace, n_calls)
            self.k_seen = m.k
            return out

        return call


def install(world) -> Memo:
    world.memo = Memo(world)
    return world.memo

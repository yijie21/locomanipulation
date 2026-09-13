"""The upper-body reference channel: what the k-planner has, and how to take
it over without a step.

`g1_deploy_onnx_ref.cpp:781` replaces the reference motion's 17 upper-body
joints with `upper_body_position` verbatim -- in every one of the 10 lookahead
frames -- on any tick whose planner message carries that key
(`zmq_manager.hpp:259` sets the flag from the key's mere presence).  So the
switch into PLANNER_FROZEN_UPPER_BODY swaps the *source* of the reference's
upper body in a single tick, and whatever the first frozen command says lands
in the encoder's observation as a step.  The encoder reads the reference, not
the measurement, so a step there is a step in the recorded token.

Two things follow, and this module exists for both.

**Anchor on `body_q_target`, not on the measured joints.**  Seeding the first
frozen command from `upper_body_from_mj(obs["joints"])` looks continuous but is
not: the measured pose is not the quantity being replaced, and the two differ
by the whole tracking error of the frozen RL decoder.  The quantity being
replaced is `current_motion_->JointPositions(current_frame)`, which the deploy
publishes as `body_q_target` (g1_debug key 22, 29 doubles, MuJoCo order).  Two
details make it usable:

* it carries NO default-angle offset (`output_interface.hpp:192`), unlike
  `body_q`/`body_q_measured` -- it is absolute radians straight from the
  motion, the same space `upper_body_position` is written in, so
  `upper_body_from_mj` applies to it unchanged;
* the substitution writes into the OBSERVATION buffer and never back into
  `current_motion_`, so `body_q_target` keeps reporting the k-planner's own
  arms even while we are overriding them -- which is what makes a release ramp
  possible as well as a takeover one.

**Rate-limit the command, and always send its derivative.**  `UpperRamp` does
both; see its docstring for why omitting `upper_body_velocity` does not mean
zero.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

from locomani.backends.sonic.config import DEFAULT_BACKEND, SonicBackendConfig
from locomani.sim import ports

DT = 0.02

# How far the 17-value command may move in one tick.  The reference dataset's
# task arm (the human-teleop session, motion_smoothness.py) has a p99 joint
# speed of 0.938 rad/s = 1.07 deg/tick at 50 Hz, so a cap in that neighbourhood
# keeps every upper-body command change inside the human envelope.  2.0 deg is
# deliberately a touch above it: the rise has to finish during the approach
# walk or it turns into recorded standing at the table, and 2 deg moves 4 of
# the token's 64 dims where the 20 deg step it replaces moved 27
# (sonic_encoder.py --selfcheck).
UPPER_SLEW_DEG = DEFAULT_BACKEND.upper_slew_deg
MAX_STEP = np.radians(UPPER_SLEW_DEG)


class RefTap:
    """Latest `body_q_target` off the deploy's g1_debug stream.

    A background thread, because the message is CONFLATE and the callers are
    50 Hz control loops that must not block on it.  `None` from `q29`/`upper17`
    means "no fresh reference" and callers must have a fallback -- a missing
    tap has to LOOK missing rather than silently seed from something else.
    """

    def __init__(self, port: int | None = None):
        self.port = ports.out() if port is None else int(port)
        self._q: np.ndarray | None = None
        self._root: np.ndarray | None = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._run = False
        self._th: threading.Thread | None = None
        self._sub = None

    def start(self) -> "RefTap":
        if self._run:
            return self
        try:
            from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
        except ImportError:  # not on the sim venv's path
            import sys

            from locomani import paths

            sys.path.insert(0, paths.SONIC)
            from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
        self._sub = ZMQStateSubscriber(host="localhost", port=self.port)
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

        return self

    def _loop(self) -> None:
        while self._run:
            m = self._sub.get_msg(clear=True)
            if m is not None:
                q = m.get("body_q_target")
                if q is not None and len(q) == 29:
                    with self._lock:
                        self._q = np.asarray(q, dtype=np.float32)
                        root = np.asarray(m.get("base_quat_target", []), dtype=np.float32)
                        self._root = (
                            root
                            if root.shape == (4,) and np.isfinite(root).all() and np.linalg.norm(root) > 0.5
                            else None
                        )
                        self._t = time.time()
            time.sleep(0.004)

    def stop(self) -> None:
        self._run = False
        if self._th is not None:
            self._th.join(timeout=2.0)
            self._th = None

    def q29(self, max_age: float = 0.5) -> np.ndarray | None:
        """The k-planner's 29-joint reference, MuJoCo order, absolute rad."""
        with self._lock:
            if self._q is None or time.time() - self._t > max_age:
                return None

            return self._q.copy()

    def pose(self):
        """One fresh joint/root reference packet, before frozen-upper substitution."""
        with self._lock:
            if self._q is None or self._root is None or time.time() - self._t > 0.2:
                return None
            return self._q.copy(), self._root.copy()

    def upper17(self, max_age: float = 0.5) -> np.ndarray | None:
        """The same reference as a 17-value upper-body command."""
        from locomani.backends.sonic.primitives import upper_body_from_mj

        q = self.q29(max_age)

        return None if q is None else upper_body_from_mj(q)

    def wait(self, timeout: float = 3.0) -> bool:
        """Block until one `body_q_target` has arrived. False on timeout.

        The field is published only once the deploy has a motion
        (`output_interface.hpp` gates it on `has_joint_data`), so a cold tap
        before `start_control` is expected, not an error.
        """
        end = time.time() + timeout
        while time.time() < end:
            if self.q29(max_age=1e9) is not None:
                return True
            time.sleep(0.02)

        return False


_SHARED: RefTap | None = None


def shared() -> RefTap:
    """One tap per process -- every caller wants the same latest frame."""
    global _SHARED
    if _SHARED is None:
        _SHARED = RefTap().start()

    return _SHARED


class UpperRamp:
    """A slew-limited upper-body command that carries its own derivative.

    Two jobs `set_upper_body` alone cannot do:

    * **Rate limit.**  The deploy substitutes the 17 values into the reference
      verbatim, so a jump in what we send is a jump in what the encoder
      observes.  grasp_gym's rise is parameterised by DISTANCE REMAINING, which
      is not a rate: a round that starts its approach already close to the
      stand evaluates the ramp at u=0.3-0.6 on its very first tick and commands
      a third of the whole rise in 20 ms.  `max_step` caps the per-tick change
      whatever the parameter does.

    * **Velocity, explicitly, every tick.**  `upper_body_velocity` is a
      separate optional key, and the deploy substitutes the last one it
      received for as long as position data keeps arriving --
      `input_interface.hpp:381-390` reads the buffer with no staleness cutoff
      and `g1_deploy_onnx_ref.cpp:2985` discards the validity flag.  So
      omitting the key does not mean "zero" once anything has been sent, and
      never sending it at all means the encoder reads a hard zero reference
      velocity while the positions sweep -- a self-contradictory reference
      either way.  `push` writes the actual derivative of what it just sent,
      and `settle` writes a real zero.
    """

    def __init__(self, cmd, q0, *, max_step: float | None = None, config: SonicBackendConfig = DEFAULT_BACKEND):
        self.cmd = cmd
        # float64 state, f32 only on the wire: the slew is applied hundreds
        # of times per round and f32 accumulation drifts the cap.
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.max_step = float(np.radians(config.upper_slew_deg) if max_step is None else max_step)
        self._t: float | None = None
        self.cmd.set_upper_body(self.q)
        self.cmd.set_upper_body_vel(np.zeros(17, dtype=np.float32))

    def push(self, q_want) -> float:
        """Step at most `max_step` toward `q_want`. Returns the residual."""
        want = np.asarray(q_want, dtype=np.float64)
        d = np.clip(want - self.q, -self.max_step, self.max_step)
        now = time.time()
        dt = DT if self._t is None else float(np.clip(now - self._t, 0.005, 0.1))
        self._t = now
        self.q = self.q + d
        self.cmd.set_upper_body(self.q)
        self.cmd.set_upper_body_vel((d / dt).astype(np.float32))

        return float(np.max(np.abs(want - self.q)))

    def settle(self) -> None:
        """Hold the current command with an honest zero velocity."""
        self._t = None
        self.cmd.set_upper_body(self.q)
        self.cmd.set_upper_body_vel(np.zeros(17, dtype=np.float32))


def _selfcheck() -> None:
    """Exercise UpperRamp against a stub commander -- no stack needed."""

    class Stub:
        def __init__(self):
            self.q, self.v = [], []

        def set_upper_body(self, q):
            self.q.append(np.asarray(q, dtype=np.float64).copy())

        def set_upper_body_vel(self, v):
            self.v.append(np.asarray(v, dtype=np.float64).copy())

    q0 = np.zeros(17)
    want = np.full(17, np.radians(60.0))
    c = Stub()
    r = UpperRamp(c, q0, max_step=np.radians(2.0))

    assert np.allclose(c.q[0], q0), "the first command must BE the anchor"
    assert np.allclose(c.v[0], 0.0), "the anchor carries a zero velocity"

    n = 0
    while n < 1000:
        n += 1
        if r.push(want) <= 1e-9:
            break
    assert n == 30, f"60 deg at 2 deg/tick should take 30 ticks, took {n}"

    steps = np.max(np.abs(np.diff(np.stack(c.q), axis=0)), axis=1)
    assert steps.max() <= np.radians(2.0) + 1e-9, f"slew breached: {np.degrees(steps.max()):.3f} deg"

    # The velocity must be the derivative of what was actually sent, and the
    # settle must write a real zero (omitting the key does not clear it).
    for k in range(1, len(c.q)):
        dq = c.q[k] - c.q[k - 1]
        vk = c.v[k]
        moving = np.abs(dq) > 1e-12
        assert np.all(np.sign(vk[moving]) == np.sign(dq[moving])), "sign"
        assert np.all(vk[~moving] == 0.0), "a still joint must read zero"
    r.settle()
    assert np.allclose(c.v[-1], 0.0), "settle must write zeros"

    # A step SMALLER than the cap passes through untouched.
    c2 = Stub()
    r2 = UpperRamp(c2, q0, max_step=np.radians(2.0))
    small = np.full(17, np.radians(0.5))
    assert r2.push(small) < 1e-9, "a sub-cap move must land in one tick"

    print(f"[PASS] UpperRamp: {n} ticks for 60 deg, max step {np.degrees(steps.max()):.3f} deg, velocity consistent")


if __name__ == "__main__":
    import sys

    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        tap = RefTap().start()
        print(f"waiting for body_q_target on :{tap.port} ...")
        if not tap.wait(5.0):
            raise SystemExit("no body_q_target -- is the deploy up and started?")
        from locomani.backends.sonic.primitives import upper_body_from_mj  # noqa: F401

        q = tap.q29(max_age=1e9)
        print("body_q_target (MuJoCo order, deg):")
        print(f"  waist {np.round(np.degrees(q[12:15]), 1)}")
        print(f"  l_arm {np.round(np.degrees(q[15:22]), 1)}")
        print(f"  r_arm {np.round(np.degrees(q[22:29]), 1)}")
        tap.stop()

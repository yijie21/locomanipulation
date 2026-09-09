"""`CleanClock` + `install_clean_shim`, split out of vla_round.py (refactor
step 4b family 3, pure extraction; decision 9's file-size cap). Every
statement here is unchanged from the original inline body -- only the
packaging moved, plus one read that has to stay LIVE against the `vla_round`
module object: `CleanClock.__init__` reads `DATASET_FPS`, which
`eval_closed_loop.py` overrides by assignment (`vla_round.DATASET_FPS =
cfg.backend.dataset_fps`) before this class is ever constructed. A plain
`from vla_round import DATASET_FPS` here would bind to the value at THIS
module's import time (before the override runs) and never see it; reading
`vla_round.DATASET_FPS` through a deferred import (inside `__init__`, not at
this module's top level -- vla_round.py imports `CleanClock` from here for
re-export, so a top-level import in the other direction would be circular)
gets the same live read the original bare name had when this class lived in
vla_round.py itself. See vla_round.py's own module docstring for the rest of
this monkeypatch inventory.
"""

from __future__ import annotations

import time

from locomani.backends.sonic.sonic_token import expand_hands


class CleanClock:
    """Wall-clock counterpart of LockstepClock for a deploy engaged by
    SONIC_AUTO_START ("clean deploy": never a planner command in the whole
    process lifetime -- docs/lockstep.md '校准诊断': one planner hold before
    enter_streamed leaves the deploy executing walking tokens at ~1/3).

    Keeps the standing token flowing at DATASET_FPS whenever the VLA round is
    not publishing, primes streamed mode after every deploy (re)start, and
    serialises every publish on the ONE commander socket (the wall-clock
    commander BINDS the deploy's SUB port, so there is no second socket)."""

    def __init__(self):
        import threading

        from locomani.backends.sonic import vla_round as vr
        from locomani.backends.sonic.initial_pose import initial_token
        from locomani.backends.sonic.sonic_token import GRIPPER_MAX

        self.init_tok = initial_token()
        self.l_open, self.r_open = expand_hands(GRIPPER_MAX)
        self.cmd = None
        self.lock = threading.RLock()
        self.dt = 1.0 / vr.DATASET_FPS
        self._run = threading.Event()
        self._stop = threading.Event()
        self._priming = False
        self.primed = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def attach(self, cmd) -> None:
        self.cmd = cmd

    def _send_init(self) -> None:
        with self.lock:
            self.cmd.token_frame(self.init_tok, right_hand=self.r_open, left_hand=self.l_open, frame_index=0)

    def prime(self) -> None:
        """Fresh CONTROL-state deploy -> streamed mode on the standing token
        (the lockstep _prime, at wall-clock pace)."""
        with self.lock:
            self._priming = True
            try:
                for _ in range(3):
                    self._send_init()
                    time.sleep(0.02)
                self.cmd.enter_streamed()
                for _ in range(4):
                    self._send_init()
                    time.sleep(0.02)
            finally:
                self._priming = False
        self.primed = True
        print("[clean-clock] deploy primed (streamed, standing token)")

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._run.is_set() or self.cmd is None or not self.primed:
                self._run.wait(0.05)
                continue
            t = time.time()
            try:
                self._send_init()
            except Exception as e:
                print(f"[clean-clock] publish failed: {e}"[:160])
                time.sleep(0.2)
            rest = self.dt - (time.time() - t)
            if rest > 0:
                time.sleep(rest)

    def resume(self) -> None:
        self._run.set()

    def pause(self) -> None:
        self._run.clear()
        time.sleep(self.dt * 1.5)  # let an in-flight publish finish

    def hold(self, seconds: float) -> None:
        self.resume()
        time.sleep(seconds)

    def stop(self) -> None:
        self._stop.set()
        self.th.join(timeout=2.0)


def install_clean_shim(clock: CleanClock) -> None:
    """Wall-clock 'clean deploy' commander shim: planner commands dropped,
    hold() = standing-token time, enter_streamed from the round = hand the
    socket to the round (deploy is already streamed), exit_streamed = take it
    back (never back to planner mode). Every publish goes through the clock's
    lock. Installed at the class before grasp_gym builds its Primitives."""
    from locomani.backends.sonic import agent_server

    C = agent_server.AgentCommander
    orig_init, orig_command = C.__init__, C.command
    orig_enter, orig_token = C.enter_streamed, C.token_frame

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if clock.cmd is None:
            clock.attach(self)

    def command(self, *, start: bool, stop: bool, planner: bool) -> None:
        if planner and not stop:
            return  # never a planner command
        with clock.lock:
            orig_command(self, start=start, stop=stop, planner=planner)

    def enter_streamed(self) -> None:
        if clock._priming:
            orig_enter(self)
            return
        clock.pause()  # the round publishes from here on

    def exit_streamed(self) -> None:
        clock.resume()  # the round is over; keep standing

    def hold(self, seconds: float, hz: float = 50.0, **frame_kwargs) -> None:
        clock.hold(seconds)

    def frame(self, *a, **kw) -> None:
        return

    def token_frame(self, *a, **kw):
        with clock.lock:
            return orig_token(self, *a, **kw)

    C.__init__, C.command, C.enter_streamed, C.exit_streamed = __init__, command, enter_streamed, exit_streamed
    C.hold, C.frame, C.token_frame = hold, frame, token_frame
    print(
        "[clean-deploy] commander shim installed: planner commands dropped, "
        "hold() = standing token, streamed mode kept for the process lifetime"
    )

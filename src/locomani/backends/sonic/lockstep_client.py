"""Lockstep client: simulated-latency, simulation-clock evaluation, split out
of vla_round.py (refactor step 4b family 3, pure extraction; decision 9's
file-size cap). Every statement here is unchanged from the original inline
body -- only the packaging moved, plus `make_vla_policy_lockstep`'s reads of
`DATASET_FPS` / `TIMEOUT_S` / `RTC_ON` / `LOCKSTEP_NO_SEED` / `ENSEMBLE_N` /
`NEAR_M` / `LIFT_M`, which have to stay LIVE against the `vla_round` module
object the same way `stream_reach.toppled` did in family 2 (see that
module's docstring for the general pattern, and vla_round.py's own for this
family's full monkeypatch inventory) -- `eval_closed_loop.py` overrides
several of these by assignment before building this round. The deferred
import is aliased `vr`, not `vla_round`, because the nested closure this
module builds is itself named `vla_round` (matching every other channel's
`vla_round(p, reset, meta, ...)` factory-return convention) and would shadow
a same-named module import.

The sim (lockstep_sim.py) advances only on request, so wall-clock time is
no longer part of the experiment. Every quantity the async client measured
with a clock is re-expressed in 25 fps FRAMES:
  - K, the frames the body keeps executing the old plan while the brain
    "thinks", is DRAWN from the empirical wall-clock distribution (or fixed
    with --latency-frames) instead of being whatever the GPU happened to
    deliver;
  - chunk t0 is a frame number; the ensemble aligns by frame; RTC's
    elapsed == frozen == K exactly.
Same obs -> same chunk (policy.seed) -> same physics: a round replays
bit-for-bit, and a GPU shared with three other servers changes nothing but
the wall time.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from locomani.backends.sonic.sonic_token import expand_hands
from locomani.backends.sonic.vla_client import _ensemble, _grip_gain, _stalled
from locomani.backends.sonic.vla_obs import ObsBuilder, PolicyLink, _obj_extra

LOCKSTEP_PORT = int(os.environ.get("TF_LOCKSTEP_PORT", 5583 + 10 * int(os.environ.get("INSTANCE", "0"))))


class LockstepLink:
    """REQ client for lockstep_sim.py's step server. One socket, one lock:
    the clock pump and the round loop share it."""

    def __init__(self, host="localhost", port=LOCKSTEP_PORT, timeout_ms=60000):
        import threading

        import msgpack
        import msgpack_numpy as mnp
        import zmq

        self._msgpack, self._mnp = msgpack, mnp
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{host}:{port}")
        self._addr, self._timeout_ms = f"tcp://{host}:{port}", timeout_ms
        self.lock = threading.Lock()

    def _reconnect(self) -> None:
        """A REQ socket that lost a reply (timeout, sim hiccup) refuses the
        next send ("Operation cannot be accomplished in current state");
        rebuild it so the round fails cleanly and the next one can run."""
        import zmq

        try:
            self.sock.close(linger=0)
        except Exception:
            pass
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self._addr)

    def _call(self, req: dict) -> dict:
        with self.lock:
            try:
                self.sock.send(self._msgpack.packb(req, default=self._mnp.encode))
                return self._msgpack.unpackb(self.sock.recv(), object_hook=self._mnp.decode, raw=False)
            except Exception:
                self._reconnect()
                raise

    def ping(self) -> bool:
        try:
            return bool(self._call({"cmd": "ping"}).get("ok"))
        except Exception:
            return False

    def begin(self) -> dict:
        return self._call({"cmd": "begin"})

    def step(self, frames, ticks_per_frame: int = 2, want_obs: bool = False) -> dict:
        """frames: [(token|None, left|None, right|None), ...]"""
        fr = []
        for tok, l, r in frames:
            fr.append(
                {
                    "token": None if tok is None else np.asarray(tok, dtype=np.float32),
                    "left": None if l is None else np.asarray(l, dtype=np.float32),
                    "right": None if r is None else np.asarray(r, dtype=np.float32),
                }
            )
        rep = self._call(
            {"cmd": "step", "frames": fr, "ticks_per_frame": int(ticks_per_frame), "want_obs": bool(want_obs)}
        )
        if not rep.get("ok"):
            raise RuntimeError(f"lockstep step: {rep.get('error')}")
        return rep


class LockstepClock:
    """The agent's side of the simulation clock, outside the VLA round.

    Two jobs. (1) A background pump: while running it requests one tick per
    20 ms of wall time, so the legacy gym code that sleeps between actions
    (deploy restart, reset settle, p.obs polls) sees a world that keeps
    moving. (2) The streamed-token bring-up after every lockstep SESSION
    (deploy restart, or a reset -- the sim reopens a session whenever
    mjData.time goes backwards, and the deploy's ResetLockstepSession clears
    the external token state and re-anchors heading): prime a few plain
    tokens, enter streamed mode, then keep the standing token flowing as
    envelope frames. There is NO planner under lockstep (its wall-clock
    thread deadlocks the gate), so every idle second is standing-token time.

    hold_frames(n) is the deterministic replacement for p.cmd.hold(): exactly
    n control ticks of the standing token, pump paused.
    """

    def __init__(self, link: LockstepLink, cmd, hz: float = 50.0):
        import threading

        from locomani.backends.sonic.initial_pose import initial_token
        from locomani.backends.sonic.sonic_token import GRIPPER_MAX as _G_OPEN

        self.link = link
        self.cmd = cmd
        self.init_tok = initial_token()
        self.l_open, self.r_open = expand_hands(_G_OPEN)
        self.dt = 1.0 / hz
        self._run = threading.Event()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._err_last = 0.0
        self.session = None  # session the deploy is primed for
        self.primed = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    # -- one tick, with session tracking ------------------------------------
    def _tick(self, frames, tpf=1, want_obs=False):
        rep = self.link.step(frames, ticks_per_frame=tpf, want_obs=want_obs)
        sess = rep.get("session")
        if sess != self.session:
            self.session, self.primed = sess, False
        return rep

    def _prime(self):
        """Bring the fresh session into streamed mode on the standing token."""
        for _ in range(3):
            self.cmd.token_frame(self.init_tok, right_hand=self.r_open, left_hand=self.l_open, frame_index=0)
            self._tick([(None, None, None)])
        self.cmd.enter_streamed()
        self._tick([(None, None, None)] * 2)
        self._tick([(self.init_tok, self.l_open, self.r_open)] * 2)
        self.primed = True
        print(f"[lockstep-clock] session {self.session} primed (streamed, standing token)")

    def ensure_primed(self, timeout: float = 240.0):
        """Block until a session exists and is primed (deploy restarts take
        ~30 s; each failed attempt costs one ack timeout)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.session is None:
                    self._tick([(None, None, None)])
                if not self.primed:
                    self._prime()
                return True
            except Exception as e:
                if time.time() - self._err_last > 5.0:
                    print(f"[lockstep-clock] waiting for the deploy: {e}"[:160])
                    self._err_last = time.time()
                time.sleep(0.5)
        return False

    def _loop(self):
        while not self._stop.is_set():
            if not self._run.is_set():
                self._idle.set()
                self._run.wait(0.05)
                continue
            self._idle.clear()
            t = time.time()
            try:
                if not self.primed:
                    self.ensure_primed(timeout=10.0)
                else:
                    self._tick([(self.init_tok, self.l_open, self.r_open)])
            except Exception as e:
                if time.time() - self._err_last > 5.0:
                    print(f"[lockstep-clock] tick failed: {e}"[:160])
                    self._err_last = time.time()
                time.sleep(0.2)
            rest = self.dt - (time.time() - t)
            if rest > 0:
                time.sleep(rest)

    def resume(self):
        self._run.set()

    def pause(self):
        """Stop pumping and wait until the in-flight tick has returned."""
        self._run.clear()
        self._idle.wait(10.0)

    def hold_frames(self, n: int):
        """Exactly n ticks of the standing token (deterministic hold)."""
        was = self._run.is_set()
        self.pause()
        try:
            self.ensure_primed()
            while n > 0:
                k = min(n, 50)
                self._tick([(self.init_tok, self.l_open, self.r_open)] * k)
                n -= k
        finally:
            if was:
                self.resume()

    def stop(self):
        self._stop.set()
        self._run.set()
        self.th.join(timeout=2.0)


def install_lockstep_shim(clock: LockstepClock) -> None:
    """Under lockstep the commander must never ask for PLANNER mode (the
    deploy's planner thread deadlocks the lockstep gate) and wall-clock holds
    must become tick counts. Patched at the class so grasp_gym's Primitives
    pick it up: start_control / exit_streamed / pause_control (planner=True
    commands) become no-ops, hold() becomes hold_frames(), planner frame()
    is dropped. enter_streamed and token_frame pass through."""
    from locomani.backends.sonic import agent_server

    C = agent_server.AgentCommander
    orig_command = C.command

    def command(self, *, start: bool, stop: bool, planner: bool) -> None:
        if planner and not stop:
            return  # no planner under lockstep
        orig_command(self, start=start, stop=stop, planner=planner)

    def hold(self, seconds: float, hz: float = 50.0, **frame_kwargs) -> None:
        clock.hold_frames(int(round(seconds * 50.0)))

    def frame(self, *a, **kw) -> None:
        return

    C.command = command
    C.hold = hold
    C.frame = frame
    print("[lockstep] commander shim installed: planner commands dropped, hold() = N standing-token ticks")


def load_latency_dist(path: str | None, fixed: int | None) -> dict:
    """{'first': [frames...], 'steady': [frames...]} to draw K from."""
    if fixed is not None:
        return {"first": [int(fixed)], "steady": [int(fixed)], "source": f"fixed {fixed}"}
    if not path:
        raise SystemExit("lockstep client needs --latency-frames K or --latency-dist <json from tools/latency_dist.py>")
    d = json.load(open(path))
    assert d.get("first") and d.get("steady"), f"{path}: empty distribution"
    d["source"] = path
    return d


def make_vla_policy_lockstep(
    link: PolicyLink,
    ls: LockstepLink,
    clock: LockstepClock,
    task: str,
    latency: dict,
    seed: int = 0,
    trace_dir: str | None = None,
):
    """Simulation-clock round with simulated inference latency.

    Cycle (all in 25 fps frames, f = frames since the round started):
        obs at frame f  ->  chunk (t0 = f)  ->  draw K  ->  advance the sim K
        frames on the OLD plan (ensemble of buffered chunks, or the latest
        chunk at index f' - t0)  ->  the new chunk joins the buffer at f + K.
    That is the async client's worker/main-thread dance with the clock
    replaced by arithmetic: K plays both the inference latency and the
    spacing between observations (the worker re-observes as soon as the
    previous inference returns when inference is slower than the 0.4 s
    interval, which it is).
    """
    from locomani.backends.sonic import vla_round as vr
    from locomani.backends.sonic.initial_pose import initial_token
    from locomani.backends.sonic.sonic_token import GRIPPER_MAX as _G_OPEN

    counter = {"n": 0}
    RATE = int(round(vr.DATASET_FPS))
    TIMEOUT_F = int(vr.TIMEOUT_S * RATE)
    LIFT_HOLD_F = 2 * RATE  # 2 s of sustained lift, as async
    STATUS_EVERY_F = max(1, RATE // 2)  # ~2 Hz predicates, as async

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record, replace=None, spawn_xy=None, round_id=None):
        from locomani.backends.sonic.primitives import LocomotionMode

        INIT_TOK = initial_token()
        l_open, r_open = expand_hands(_G_OPEN)
        rng = np.random.default_rng([int(seed), counter["n"]])
        draw = lambda key: int(rng.choice(latency[key]))  # noqa: E731

        obs_src = ObsBuilder(task)
        trace, ok, why = [], False, "[vla] timeout"
        commanded_close = False
        lat_rec = {"K": []}
        prelude = {}
        try:
            # -- prelude -------------------------------------------------------
            # The clock pump already primed this session (plain tokens ->
            # enter_streamed) and has been holding the standing token; the
            # round takes the clock over and keeps standing while the first
            # observation is taken.
            clock.pause()
            if not clock.ensure_primed():
                raise RuntimeError("deploy never became ready (lockstep)")
            ls.step([(INIT_TOK, l_open, r_open)] * 5, ticks_per_frame=2)

            def observe(rep):
                for _ in range(20):  # deploy state from the out port
                    obs_src._poll()
                    if obs_src._last_proprio is not None:
                        break
                    time.sleep(0.01)
                # AFTER the poll: _poll also drains the async 5555 stream,
                # whose newest frame can predate the boundary render.
                obs_src._last_img = rep["images"]
                obs = obs_src.build()
                if obs is None:
                    raise RuntimeError("[vla] lockstep observation incomplete")
                return obs

            def infer(obs, f, elapsed, frozen):
                # LOCKSTEP_NO_SEED=1: leave the server's torch RNG alone (A/B
                # against the deterministic per-inference seed)
                if not vr.LOCKSTEP_NO_SEED:
                    obs["policy.seed"] = int(rng.integers(0, 2**31 - 1))
                if vr.RTC_ON:
                    obs["rtc.reset"] = 1 if f == 0 else 0
                    obs["rtc.elapsed"] = int(elapsed)
                    obs["rtc.frozen"] = int(frozen)
                return link.get_chunk(obs)

            rep = ls.step([(INIT_TOK, l_open, r_open)], 2, want_obs=True)
            s0 = p.obs()
            obj0 = np.asarray(p.object_pos(obj_key, s0))
            f = 0
            K0 = draw("first")
            tok0, grip0 = infer(observe(rep), 0, 0, K0)
            # hold the standing token through the (simulated) first inference,
            # then a 1 s blend into the first chunk -- the async prelude
            ls.step([(INIT_TOK, l_open, r_open)] * K0, 2)
            f += K0
            blend = []
            for i in range(RATE):
                a = (i + 1) / RATE
                blend.append(((1 - a) * INIT_TOK + a * tok0[0], l_open, r_open))
            ls.step(blend, 2)
            f += RATE
            lat_rec["K"].append(K0)
            prelude["prelude_frames"] = f
            buf = [(tok0, grip0, 0)]
            latest = (tok0, grip0, 0)
            lifted_at = None
            next_status = 0

            def frame_cmd(fi):
                if vr.ENSEMBLE_N > 1:
                    b = _ensemble(buf, fi, 1.0)
                    if b is None:
                        tok_j, grip_j = latest[0][-1], float(latest[1][-1])
                    else:
                        tok_j, grip_j = b
                else:
                    tok, grip, t0 = latest
                    j = min(max(fi - t0, 0), len(tok) - 1)
                    tok_j, grip_j = tok[j], float(grip[j])
                return tok_j, grip_j

            # one frame of the first chunk, and the observation for cycle 1
            tok_j, grip_raw = frame_cmd(f)
            lh, rh = expand_hands(_grip_gain(grip_raw))
            rep = ls.step([(tok_j, lh, rh)], 2, want_obs=True)
            f += 1
            # -- main cycle ----------------------------------------------------
            while f < TIMEOUT_F:
                # K: frames until the next observation (worker cycle);
                # A <= K: frames until this chunk EXISTS (inference time) --
                # from then on the ensemble already blends it, as the wall-
                # clock worker appends to the buffer the moment inference
                # returns, before the next observation is taken.
                K = draw("steady")
                A = min(draw("adopt"), K) if latency.get("adopt") else K
                obs = observe(rep)
                t_obs = f
                tok, grip = infer(obs, t_obs, elapsed=t_obs - latest[2], frozen=A)
                lat_rec["K"].append(K)
                lat_rec.setdefault("A", []).append(A)
                # what did the policy predict? (diagnostics for the table
                # stall: does the chunk itself contain a reach/close?)
                lat_rec.setdefault("chunk_grip_min", []).append(float(np.min(grip)))
                lat_rec.setdefault("chunk_tok_span", []).append(
                    float(np.abs(np.asarray(tok)[-1] - np.asarray(tok)[0]).mean())
                )
                frames = []
                grips = []
                for i in range(K):
                    if i == A:
                        buf.append((tok, grip, t_obs))
                        del buf[: -max(vr.ENSEMBLE_N, 1)]
                        latest = (tok, grip, t_obs)
                    tok_j, grip_raw = frame_cmd(f + i)
                    g = _grip_gain(grip_raw)
                    if g < 0.005:
                        commanded_close = True
                    lh, rh = expand_hands(g)
                    frames.append((tok_j, lh, rh))
                    grips.append((g, grip_raw))
                if A >= K:
                    buf.append((tok, grip, t_obs))
                    del buf[: -max(vr.ENSEMBLE_N, 1)]
                    latest = (tok, grip, t_obs)
                rep = ls.step(frames, 2, want_obs=True)
                f += K
                if rep.get("fallen"):
                    why = "[vla] fell"
                    break
                if f >= next_status:
                    next_status = f + STATUS_EVERY_F
                    st = p.obs()
                    obj = np.asarray(p.object_pos(obj_key, st))
                    pel = np.asarray(p.pelvis_xy(st))
                    rise = float(obj[2] - obj0[2])
                    palm = np.asarray(st["palms"]["right"]["pos"], dtype=float)
                    g, graw = grips[-1]
                    pq = st.get("pelvis", {}).get("quat")
                    pitch = None
                    if pq is not None:
                        w_, x_, y_, z_ = (float(v) for v in pq)
                        pitch = float(np.arcsin(float(np.clip(2.0 * (w_ * y_ - z_ * x_), -1.0, 1.0))))
                    trace.append(
                        dict(
                            t=f / RATE,
                            frame=f,
                            K=K,
                            rise=rise,
                            dist=float(np.linalg.norm(obj[:2] - pel)),
                            fallen=bool(st["fallen"]),
                            grip=float(g),
                            grip_raw=float(graw),
                            pelvis_xy=[float(v) for v in pel],
                            pelvis_yaw=float(p.pelvis_yaw(st)),
                            pelvis_z=float(st["pelvis"]["pos"][2]),
                            torso_pitch=pitch,
                            obj_xy=[float(v) for v in obj[:2]],
                            palm_z=float(palm[2]),
                            palm_obj=float(np.linalg.norm(palm - obj)),
                            **_obj_extra(st, obj_key),
                        )
                    )
                    if st["fallen"]:
                        why = "[vla] fell"
                        break
                    if rise > vr.LIFT_M:
                        if lifted_at is None:
                            lifted_at = f
                        elif f - lifted_at > LIFT_HOLD_F:
                            ok, why = True, "ok"
                            break
                    else:
                        lifted_at = None
                    if _stalled(trace, commanded_close):
                        why = "[vla] reach: stalled (--stall-abort)"
                        break
        except RuntimeError as e:
            why = f"[vla] lockstep: {e}"
        finally:
            try:
                p.cmd.exit_streamed()
            except Exception:
                pass
            clock.resume()
            p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
            obs_src.close()

        if not ok and why == "[vla] timeout" and trace:
            min_dist = min(r["dist"] for r in trace)
            if min_dist > vr.NEAR_M:
                why = f"[vla] walk: never closer than {min_dist:.2f} m"
            elif not commanded_close:
                why = f"[vla] reach: at the table (min {min_dist:.2f} m) but never commanded a close"
            else:
                moved = float(np.linalg.norm(np.asarray(p.object_pos(obj_key, p.obs()))[:2] - obj0[:2]))
                why = f"[vla] close: commanded a grasp but no lift (object moved {moved * 100:.0f} cm laterally)"

        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            # Name the trace by the CALLER's round number, not by this closure's
            # own call count: a round that fails before the VLA stage (a [stand]
            # collapse, say) never reaches here, so the private counter silently
            # drifts from the manifest -- the visible200 eval's traces were all
            # off by one against closed_loop.manifest.jsonl, and every analysis
            # that joined the two by round number was reading mismatched pairs.
            # `round` is written into the file too, so a join never has to trust
            # the filename.
            rid = counter["n"] if round_id is None else int(round_id)
            with open(os.path.join(trace_dir, f"round_{rid:03d}.json"), "w") as fh:
                json.dump(
                    dict(
                        round=rid,
                        ok=ok,
                        why=why,
                        clock="lockstep",
                        latency_source=latency.get("source"),
                        **prelude,
                        latency=lat_rec,
                        trace=trace,
                    ),
                    fh,
                )
        counter["n"] += 1
        return ok, why

    return vla_round

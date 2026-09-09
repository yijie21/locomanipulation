"""Sim-side lockstep mode: the world advances only when the agent says so.

    LOCKSTEP=1 bash scripts/stack/start_stack.sh ...

Under wall-clock (the default) the MuJoCo loop in BaseSimulator.start paces
itself against real time and the C++ deploy runs its own 50 Hz control loop
off the LowState stream. Under lockstep the deploy is launched with
`--sim-lockstep` (SIMPLE's patch, built at /workspace/deps/
gear_sonic_deploy_lockstep): it computes exactly one control tick per
envelope-stamped LowState and acknowledges it on the three command topics,
and the Python side (vendored `lockstep_vendor`) will not step physics until
that acknowledgement is in hand. This module is the piece in between: it
replaces the simulator's free-running loop with a request/reply server on
TF_LOCKSTEP_PORT that the agent drives.

Protocol (msgpack over ZMQ REP):

    {"cmd": "step", "frames": [{"token": [64]|null, "left": [7]|null,
                                "right": [7]|null, "frame_index": int}, ...],
     "ticks_per_frame": 2, "want_obs": bool}
        -> {"ok": true, "ticks": n, "time": mj_time, "sim_step": seq,
            "fallen": bool, "images": {"ego_view": u8[H,W,3], ...}}
        Each frame is held for `ticks_per_frame` control ticks (a 25 fps
        chunk frame = 2 x 20 ms ticks). token null = LOCKSTEP_NO_ACTION: the
        deploy runs its planner input instead (how the scripted spawn /
        settle / turn phases keep working -- the agent's commander still
        publishes planner frames over the usual PUB socket, the deploy reads
        them in Input() every tick).
    {"cmd": "begin"} -> new lockstep session (after a deploy restart).
    {"cmd": "ping"}  -> {"ok": true}

Everything the wall-clock loop did per 20 ms tick still happens per tick
here -- update_reward (privileged publisher, video frame, :5582 reset poll)
-- so grasp_gym's predicates, the third-person mp4 and the teleport channel
are unchanged. A reset (mj_resetData) sends mjData.time back to zero; the
runner notices and opens a fresh session, which the deploy handles as
ResetLockstepSession.

Images come back IN the reply, rendered at the frozen boundary, so the
policy's observation is exactly the state its chunk will be applied to. The
async 5555 publisher keeps running for the stack's own readiness checks.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

LOCKSTEP_PORT = int(os.environ.get("TF_LOCKSTEP_PORT", 5583 + 10 * int(os.environ.get("INSTANCE", "0"))))
CONTROL_DT = 0.02


def enabled() -> bool:
    # Receiving end of the shell boundary (decision 8): this runs inside the
    # sim's own OS process, launched by start_stack.sh with no channel to
    # receive SonicBackendConfig.lockstep.enabled other than inherited env --
    # stack.py is the sole writer of LOCKSTEP (whitelisted, scripts/lint_env.py).
    return os.environ.get("LOCKSTEP", "0") == "1"


def install(sim_module) -> None:
    """Swap the DDS bridge for the lockstep one and stamp `time` into obs.

    Must run before SimWrapper is constructed: BaseSimulator.__init__ builds
    the bridge, and DefaultEnv.prepare_obs is what the barrier validates
    (`proprio["time"]` must be the exact mjData.time).
    """
    from lockstep_vendor.lockstep_unitree_bridge import LockstepUnitreeBridge

    base_sim = sim_module

    def init_unitree_bridge(self):
        self.unitree_bridge = LockstepUnitreeBridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    base_sim.BaseSimulator.init_unitree_bridge = init_unitree_bridge

    orig_prepare = base_sim.DefaultEnv.prepare_obs

    def prepare_obs(self):
        obs = orig_prepare(self)
        obs["time"] = float(self.mj_data.time)
        return obs

    base_sim.DefaultEnv.prepare_obs = prepare_obs
    print(f"[lockstep] installed: bridge swapped, obs carries mjData.time; RPC on :{LOCKSTEP_PORT}")


class LockstepRunner:
    def __init__(self, simulator, zmq_port: int, out_port: int):
        import zmq
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
        from lockstep_vendor.lockstep import LockstepBarrier, LockstepTimeout

        self.sim = simulator
        self.env = simulator.sim_env
        self.bridge = simulator.unitree_bridge
        self._pack = pack_pose_message
        self._Timeout = LockstepTimeout
        self.ctx = zmq.Context.instance()
        # The deploy is a SUB that connects to ONE address (--zmq-port), and
        # under wall-clock the agent's commander is the PUB bound there. Under
        # lockstep two publishers need that endpoint -- the commander (mode
        # switches, plain prime tokens) and this runner (envelope-stamped
        # tokens per boundary) -- so the sim owns the ONLY PUB, bound on the
        # deploy's port, fed by a PULL on LOCKSTEP_PUSH_PORT that every
        # publisher PUSHes into (agent_server honours TF_SONIC_PUB_CONNECT).
        # Not an XSUB/XPUB proxy: the deploy's idle 'command' subscriber
        # reconnects every 100 ms of silence, and through XPUB that flaps the
        # upstream subscription so commands get dropped. PUSH/PULL has no
        # subscription state; the deploy's SUBs sit one hop from the PUB, as
        # under wall-clock. The sim starts before the deploy, so the SUBs
        # find the PUB bound.
        import threading

        push_port = int(os.environ.get("TF_LOCKSTEP_XSUB_PORT", zmq_port + 28))
        pull = self.ctx.socket(zmq.PULL)
        pull.bind(f"tcp://*:{push_port}")
        pub = self.ctx.socket(zmq.PUB)
        pub.bind(f"tcp://*:{zmq_port}")
        threading.Thread(target=zmq.proxy, args=(pull, pub), daemon=True).start()
        self.tok = self.ctx.socket(zmq.PUSH)
        self.tok.connect(f"tcp://localhost:{push_port}")
        print(f"[lockstep] token relay: PULL :{push_port} -> PUB :{zmq_port}")
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind(f"tcp://*:{LOCKSTEP_PORT}")
        self.barrier = LockstepBarrier(
            self.bridge,
            self._send_token,
            physics_dt=float(self.sim.sim_dt),
            control_dt=CONTROL_DT,
            # SonicBackendConfig.lockstep.ack_timeout's default (decision 8); never
            # overridden in any script today, so this constant IS that default.
            ack_timeout=5.0,
        )
        self._last_time = None
        self._ticks = 0
        self._err_last = 0.0
        # Kinematic warm-up after every session begin. The deploy clears its
        # observation history on ResetLockstepSession (logger_size=0 in its
        # log) and the policy's first ~1 s of output from an empty history
        # knocks the robot over -- SIMPLE's own eval pins the seeded stance
        # kinematically through this window. Here: for WARMUP ticks after a
        # session opens, physics runs but the robot is restored to its held
        # pose after every substep, so the deploy fills its history with a
        # static, self-consistent robot before it is allowed to move.
        # SonicBackendConfig.lockstep.{warmup_ticks,cmd_delay}'s defaults
        # (decision 8); neither is overridden in any script today.
        self._warmup = 60
        self._cmd_delay = 0
        self._cmd_hist = []
        self._hold_qpos = None
        self._hold_left = 0
        # LOCKSTEP_JITTER=1: reproduce the wall-clock stack's timing noise,
        # deterministically. Under wall-clock the deploy's 50 Hz control tick
        # is not phase-locked to the 200 Hz physics loop (LowState age
        # 4-6 ms and drifting in its logs), so the physics advanced per
        # control tick is 3-5 substeps, not 4; and the agent's 25 fps token
        # loop sleeps with jitter, so a frame is occasionally held 1 or 3
        # ticks. Both are mean-preserving. Seeded per session so a round
        # replays exactly.
        # SonicBackendConfig.lockstep.{jitter,jitter_seed}'s defaults (decision 8);
        # neither is overridden in any script today.
        self._jitter = False
        self._jseed = 0
        self._jrng = np.random.default_rng(self._jseed)
        self._img_every = max(1, int(round(self.sim.image_dt / CONTROL_DT)))

    # -- wire ---------------------------------------------------------------
    def _send_token(self, token, left, right, envelope) -> None:
        data = {
            "token_state": np.asarray(token, dtype=np.float32).reshape(-1),
            "frame_index": np.array([envelope.action_seq], dtype=np.int64),
            "lockstep_session": np.array([envelope.session_id], dtype=np.int64),
            "lockstep_sim_step": np.array([envelope.sim_step_seq], dtype=np.int64),
            "lockstep_action_seq": np.array([envelope.action_seq], dtype=np.int64),
        }
        if left is not None:
            data["left_hand_joints"] = np.asarray(left, dtype=np.float32).reshape(-1)
        if right is not None:
            data["right_hand_joints"] = np.asarray(right, dtype=np.float32).reshape(-1)
        self.tok.send(self._pack(data, topic="pose", version=4))

    # -- physics ------------------------------------------------------------
    def _substep(self) -> None:
        """One physics step from the acknowledged command. sim_step() minus
        PublishLowState (the barrier publishes, with the envelope) and minus
        the elastic band (disabled in run_sim_locomani)."""
        import mujoco

        env = self.env
        body_torques = env.compute_body_torques()
        hand_torques = env.compute_hand_torques()
        env.torques[env.body_joint_index - 1] = body_torques
        if env.num_hand_dof > 0:
            env.torques[env.left_hand_index - 1] = hand_torques[: env.num_hand_dof]
            env.torques[env.right_hand_index - 1] = hand_torques[env.num_hand_dof :]
        env.torques = np.clip(env.torques, -env.torque_limit, env.torque_limit)
        if env.config["FREE_BASE"]:
            env.mj_data.ctrl = np.concatenate((np.zeros(6), env.torques))
        else:
            env.mj_data.ctrl = env.torques
        mujoco.mj_step(env.mj_model, env.mj_data)
        env.check_fall()

    def _begin(self) -> None:
        obs = self.env.prepare_obs()
        self.barrier.begin_session(obs, mj_time=float(self.env.mj_data.time))
        self._last_time = float(self.env.mj_data.time)
        self._hold_qpos = self.env.mj_data.qpos.copy()
        self._hold_left = self._warmup
        self._jrng = np.random.default_rng([self._jseed, int(self.barrier.session_id or 0)])
        print(f"[lockstep] session {self.barrier.session_id} active at t={self._last_time:.3f}")

    def _tick(self, token, left, right) -> None:
        """One 20 ms control tick: boundary -> ack -> 4 substeps -> hooks."""
        t = float(self.env.mj_data.time)
        if self.barrier.state != "ACTIVE" or self._last_time is None or t < self._last_time - 1e-9:
            # first tick, a failed session, or a reset wound the clock back
            self._begin()
        self.barrier.commit_boundary(token, left, right)
        # LOCKSTEP_CMD_DELAY=n (diagnostic): apply the LowCmd acknowledged n
        # ticks ago for this tick's substeps -- an inner-loop (WBC->sim)
        # delay knob, calibration against wall-clock's DDS path
        br = self.env.unitree_bridge
        if self._cmd_delay > 0:
            import copy

            self._cmd_hist.append(copy.deepcopy(br.low_cmd))
            del self._cmd_hist[: -(self._cmd_delay + 1)]
            live_cmd = br.low_cmd
            br.low_cmd = self._cmd_hist[0] if len(self._cmd_hist) > self._cmd_delay else br.low_cmd
        if self._jitter:
            # set AFTER the commit (which checks last tick's count) and
            # before this tick's substeps; the next commit checks against it
            self.barrier.substeps_per_control = int(self._jrng.choice([3, 4, 5], p=[0.25, 0.5, 0.25]))
        for _ in range(self.barrier.substeps_per_control):
            self._substep()
            if self._hold_left > 0:
                import mujoco

                self.env.mj_data.qpos[:] = self._hold_qpos
                self.env.mj_data.qvel[:] = 0.0
                mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
            self.barrier.cache_substep(self.env.prepare_obs(), mj_time=float(self.env.mj_data.time))
        if self._cmd_delay > 0:
            br.low_cmd = live_cmd
        if self._hold_left > 0:
            self._hold_left -= 1
        self._last_time = float(self.env.mj_data.time)
        # what the wall-clock loop ran every REWARD_DT: privileged publisher,
        # video frame, :5582 reset poll (a reset applied here is what makes
        # the next tick's time go backwards and reopen the session)
        self.env.update_reward()
        self._ticks += 1
        if self._ticks % self._img_every == 0:
            self.env.update_render_caches()

    def _render(self) -> dict:
        caches = self.env.update_render_caches()
        return {
            k[: -len("_image")]: np.ascontiguousarray(v, dtype=np.uint8)
            for k, v in caches.items()
            if k.endswith("_image")
        }

    # -- server -------------------------------------------------------------
    def run(self) -> None:
        import msgpack
        import msgpack_numpy as mnp

        print(f"[lockstep] serving on :{LOCKSTEP_PORT} (deploy must be --sim-lockstep); waiting for the first request")
        # start_stack's readiness check wants frames on :5555 before any
        # client ticks; render the resting scene a few times so it has some
        # (the publisher logs "Image publish frequency" every 50 frames, and
        # that line is what start_stack greps for)
        for _ in range(60):
            self.env.update_render_caches()
            time.sleep(0.03)
        while True:
            req = msgpack.unpackb(self.rep.recv(), object_hook=mnp.decode, raw=False)
            try:
                rep = self._handle(req)
            except self._Timeout as e:
                rep = {"ok": False, "error": f"ack timeout: {e.diagnostics}"}
                if time.time() - self._err_last > 5.0:
                    print(f"[lockstep] {rep['error']}", file=sys.stderr)
                    self._err_last = time.time()
                # a dead deploy leaves the barrier FAILED; a new session is
                # opened on the next tick automatically
            except Exception as e:  # noqa: BLE001 - reply, never die silently
                import traceback

                traceback.print_exc()
                rep = {"ok": False, "error": repr(e)}
            self.rep.send(msgpack.packb(rep, default=mnp.encode))

    def _handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "time": float(self.env.mj_data.time)}
        if cmd == "begin":
            self._begin()
            return {"ok": True, "session": self.barrier.session_id}
        if cmd == "lowcmd":
            # debug: the acknowledged motor command the physics is running on
            lc = self.bridge.low_cmd
            if not lc:
                return {"ok": True, "have": False}
            n = self.bridge.num_body_motor
            return {
                "ok": True,
                "have": True,
                "kp": [float(lc.motor_cmd[i].kp) for i in range(n)],
                "kd": [float(lc.motor_cmd[i].kd) for i in range(n)],
                "q": [float(lc.motor_cmd[i].q) for i in range(n)],
                "tau": [float(lc.motor_cmd[i].tau) for i in range(n)],
                "qpos": [float(v) for v in self.env.mj_data.qpos[7 : 7 + n]],
                "qvel": [float(v) for v in self.env.mj_data.qvel[6 : 6 + n]],
                "dq": [float(lc.motor_cmd[i].dq) for i in range(n)],
                "ctrl": [float(v) for v in self.env.mj_data.ctrl],
                "use_sensor": bool(self.bridge.use_sensor),
                "body_joint_index": [int(v) for v in self.env.body_joint_index],
                "reserve": [int(v) for v in lc.reserve],
            }
        if cmd != "step":
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        tpf = int(req.get("ticks_per_frame", 2))
        n = 0
        for fr in req.get("frames") or [{"token": None}]:
            tok = fr.get("token")
            tok = None if tok is None else np.asarray(tok, dtype=np.float32)
            left, right = fr.get("left"), fr.get("right")
            n_ticks = tpf
            if self._jitter and tpf >= 2:
                n_ticks = tpf + int(self._jrng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15]))
            for j in range(n_ticks):
                # Wire-level parity with wall-clock: a 25 fps frame reaches
                # the deploy ONCE, and the second 50 Hz tick runs on the
                # deploy's own starvation clamp (last received frame). Sending
                # the token again on every tick is a different input stream
                # (two frames per frame) and the policy walks in place on it.
                if j == 0:
                    self._tick(tok, left, right)
                else:
                    self._tick(None, None, None)
                n += 1
                if bool(getattr(self.env, "fall", False)):
                    break
        rep = {
            "ok": True,
            "ticks": n,
            "time": float(self.env.mj_data.time),
            "sim_step": int(self.barrier.sim_step_seq),
            "session": int(self.barrier.session_id or 0),
            "fallen": bool(getattr(self.env, "fall", False)),
        }
        if req.get("want_obs"):
            rep["images"] = self._render()
        return rep


def run(simulator, zmq_port: int, out_port: int) -> None:
    LockstepRunner(simulator, zmq_port, out_port).run()

"""One closed-loop VLA round: observation assembly + token streaming + verdict.

This module is what grasp_gym.VLA_POLICY gets installed with. A VLA round
therefore shares EVERYTHING with a scripted round -- the round sampling, the
spawn bisection, the teleport, the re-anchor, the appearance seed, the video --
and differs only from the point the robot stands settled at its spawn. That
sharing is what makes the result readable: the reference line is the scripted
policy's 200/210 = 95.2% on this exact distribution, measured by the
collection run itself.

OBSERVATION FIDELITY

Every observation channel is built the way the EXPORTER built it during data
collection, mostly by running the same code:

  images       taken verbatim from the camera SUB (:5555) message, no colour
               conversion -- the exporter wrote frame_data[feature] =
               images[key] with no conversion either, so whatever channel
               order training saw, this hands the policy the same one.
  joint state  the same 43-slot layout via
               robot_model.get_configuration_from_actuated_joints, sliced by
               the same group boundaries.
  wrist pose   the same pinocchio FK on the same 'right_wrist_yaw_link'
               frame, quaternion scalar-first, pelvis frame.
  gravity      the same compute_projected_gravity(base_quat).

The one intentional difference: the exporter recorded at 50 Hz and the dataset
was resampled to 30 Hz; here the policy is queried directly at the re-plan
points, each time with the freshest camera/state message.

TOKEN DELIVERY

Chunks come back 40 x 65. The token half is snapped to the FSQ grid
(sonic_token.snap -- training, this driver and the eval all snap the same
way), the gripper scalar becomes the two 7-wide protocol-v4 hand fields
(expand_hands), and frames go out at 30 Hz -- the chunk's native rate. Between
frames the token is zero-order held by the deploy itself (CONFLATE + its own
50 Hz loop). Only the first `execution_horizon` frames of each chunk are
streamed before re-planning, matching the open-loop metric's definition.

VERDICT

Same physical predicate the gym's scripted rounds use: the object's privileged
z (sim stream :5581, independent of the exporter) must rise > 6 cm above its
episode-start height and still be up when the stream ends. Failures are
CLASSIFIED, not just counted -- walk / reach / close / fell / timeout -- because
"the VLA fails where the script fails" (close) and "the VLA cannot even get
there" (walk) are different follow-ups: the first is inherited data bias, the
second is Q5's missing-history ablation trigger.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

from sonic_token import expand_hands, snap  # noqa: E402

TOKEN_DIM = 64
LIFT_M = 0.06            # the gym's own success threshold
NEAR_M = 0.60            # "got to the table" for failure classification
TIMEOUT_S = 90.0

# --- optional early abort for rounds that are going nowhere -------------------
# OFF by default, and it must stay that way for any number that gets published.
#
# A failing round burns the whole 90 s budget (measured over round 3's 100
# rounds: failures 89.5-90.0 s, successes a median of 19.6 s), and the robot in
# those rounds is not idle -- it shuffles. Over the last 30 s of a failed round
# the pelvis walks 0.25 m of path for 0.013 m of net displacement, 19x. So the
# usable predicate is NET displacement over a window, not speed.
#
# Measured on round 3's 100 traces, aborting the first time a window of
# STALL_ABORT_S carries less than STALL_ABORT_M of net pelvis motion with no
# close yet commanded:
#
#   window   fires   successes lost   wall clock saved
#     15 s   76/100        6              ~44%
#     20 s   73/100        4              ~40%
#     25 s   71/100        2              ~37%
#     30 s   70/100        1              ~33%
#
# Those are this function's OWN numbers, produced by replaying it frame by
# frame over the 100 saved traces -- not an offline approximation of it. The
# calibration and the implementation are the same code.
#
# The cost is small but real and one-directional: every round it gets wrong is
# a success scored as a failure, so the bias is always downward. 5.9 / 13.7 /
# 28.0 were all measured on the full budget, and a number measured with this on
# is not on that ladder. Use it for diagnostic sweeps -- checkpoint scans,
# ablations, frozen-DiT -- never for a headline.
STALL_ABORT_S = 0.0        # 0 disables; 30.0 is the measured sweet spot
STALL_ABORT_M = 0.03       # net pelvis displacement over the window, metres


def _stalled(trace, commanded_close) -> bool:
    """True when the pelvis has gone nowhere for STALL_ABORT_S with no close.

    Reads the trace that is already being recorded rather than keeping its own
    history: one source of truth, and the offline pass that calibrated the
    thresholds ran over exactly these rows.
    """
    if STALL_ABORT_S <= 0.0 or commanded_close or not trace:
        return False
    now = trace[-1]["t"]
    if now < STALL_ABORT_S:
        return False
    ref = None
    for row in trace:                    # oldest row still inside the window
        if now - row["t"] <= STALL_ABORT_S:
            ref = row
            break
    if ref is None or now - ref["t"] < STALL_ABORT_S * 0.9:
        return False
    a, b = ref["pelvis_xy"], trace[-1]["pelvis_xy"]
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 < STALL_ABORT_M
REPLAN_HZ = 30.0         # the chunk's native frame rate


class ObsBuilder:
    """Assemble the policy's flat observation from the live stack."""

    def __init__(self, task: str):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
        from gear_sonic.data.features_sonic_vla import get_g1_robot_model
        from gear_sonic.utils.data_collection.zmq_state_subscriber import (
            ZMQStateSubscriber,
        )

        cam_port = int(os.environ.get("TF_CAMERA_PORT", 5555))
        state_port = int(os.environ.get("TF_OUT_PORT", 5557))
        self.cam = ComposedCameraClientSensor(server_ip="localhost", port=cam_port)
        self.state = ZMQStateSubscriber(host="localhost", port=state_port)
        self.rm = get_g1_robot_model()
        self.task = task
        self._last_img = None
        self._last_proprio = None

    def _poll(self):
        m = self.cam.read()
        if m is not None:
            self._last_img = m["images"]
        s = self.state.get_msg()
        if s is not None:
            self._last_proprio = s

    def wait_ready(self, timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self._poll()
            if self._last_img is not None and self._last_proprio is not None:
                have = set(self._last_img)
                if {"ego_view", "right_wrist"} <= have:
                    return True
            time.sleep(0.05)
        return False

    def build(self) -> dict | None:
        from gear_sonic.utils.data_collection.transforms import (
            compute_projected_gravity,
        )
        from scipy.spatial.transform import Rotation as R

        self._poll()
        if self._last_img is None or self._last_proprio is None:
            return None
        pr = self._last_proprio
        body = np.asarray(pr["body_q"], dtype=np.float64)
        lh = np.asarray(pr["left_hand_q"], dtype=np.float64)
        rh = np.asarray(pr["right_hand_q"], dtype=np.float64)
        base_quat = np.asarray(pr["base_quat"], dtype=np.float64)

        whole = self.rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body,
            left_hand_actuated_joint_values=lh,
            right_hand_actuated_joint_values=rh)
        self.rm.cache_forward_kinematics(whole)
        pl = self.rm.frame_placement(self.rm.supplemental_info.hand_frame_names["right"])
        rw_pos = pl.translation[:3]
        rw_quat = R.from_matrix(pl.rotation).as_quat(scalar_first=True)

        f32 = lambda a: np.asarray(a, dtype=np.float32).reshape(1, -1)  # noqa: E731
        obs = {
            "state.left_leg": f32(body[0:6]),
            "state.right_leg": f32(body[6:12]),
            "state.waist": f32(body[12:15]),
            "state.left_arm": f32(body[15:22]),
            "state.right_arm": f32(body[22:29]),
            "state.left_hand": f32(lh),
            "state.right_hand": f32(rh),
            "state.projected_gravity": f32(compute_projected_gravity(base_quat)),
            "state.right_wrist_pos": f32(rw_pos),
            "state.right_wrist_abs_quat": f32(rw_quat),
            "annotation.human.task_description": self.task,
        }
        for key in ("ego_view", "right_wrist"):
            img = np.asarray(self._last_img[key])
            obs[f"video.{key}"] = img[None].astype(np.uint8)
        return obs

    def close(self):
        try:
            self.cam.close()
        except Exception:
            pass
        try:
            self.state.close()
        except Exception:
            pass


class PolicyLink:
    """Tiny msgpack/zmq client for serve_policy.py -- no gr00t imports."""

    def __init__(self, host="localhost", port=5590, timeout_ms=30000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        self._msgpack, self._mnp = msgpack, mnp
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def _call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(self._msgpack.packb(req, default=self._mnp.encode))
        rep = self._msgpack.unpackb(self.sock.recv(), object_hook=self._mnp.decode)
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(f"policy server: {rep['error']}")
        return rep

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except Exception:
            return False

    def get_chunk(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """-> (token [40,64] snapped, gripper [40])."""
        out = self._call("get_action_raw", {"obs": obs})
        tok = np.asarray(out["action.motion_token"], dtype=np.float64)
        grip = np.asarray(out["action.right_gripper"], dtype=np.float64).reshape(-1)
        return snap(tok), grip


def make_vla_policy(link: PolicyLink, task: str, execution_horizon: int = 16,
                    trace_dir: str | None = None):
    """Build the callable grasp_gym.VLA_POLICY expects.

    Called with the robot settled at its spawn, facing the object, video and
    (optionally) the exporter already recording. Must return (ok, why).
    """
    counter = {"n": 0}

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record):
        from primitives import LocomotionMode

        obs_src = ObsBuilder(task)
        if not obs_src.wait_ready():
            obs_src.close()
            return False, "[vla] camera/state streams never became ready"

        s0 = p.obs()
        obj0 = np.asarray(p.object_pos(obj_key, s0))
        trace = []
        ok, why = False, "[vla] timeout"
        dt = 1.0 / REPLAN_HZ
        commanded_close = False

        # Prime the buffer BEFORE the mode switch, then hold the first token
        # across it -- streamed mode over an empty buffer stops control
        # (g1_deploy_onnx_ref.cpp:618), and the first v4 frame is also what
        # flips is_using_encoder_ off. Same order diag_token_replay validated.
        obs = obs_src.build()
        tok0, grip0 = link.get_chunk(obs)
        l0, r0 = expand_hands(grip0[0])
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)
        p.cmd.enter_streamed()
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)

        t_start = time.time()
        frame_index = 0
        lifted_at = None
        try:
            while time.time() - t_start < TIMEOUT_S:
                obs = obs_src.build()
                if obs is None:
                    why = "[vla] observation stream dropped"
                    break
                tok, grip = link.get_chunk(obs)
                n = min(execution_horizon, len(tok))
                for j in range(n):
                    lhj, rhj = expand_hands(grip[j])
                    if grip[j] < 0.005:
                        commanded_close = True
                    p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj,
                                      frame_index=frame_index)
                    frame_index += 1
                    time.sleep(dt)

                st = p.obs()
                obj = np.asarray(p.object_pos(obj_key, st))
                pel = np.asarray(p.pelvis_xy(st))
                rise = float(obj[2] - obj0[2])
                dist = float(np.linalg.norm(obj[:2] - pel))
                trace.append(dict(t=time.time() - t_start, rise=rise, dist=dist,
                                  fallen=bool(st["fallen"]),
                                  grip=float(grip[:n].min()),
                                  # the stall poses are the whole point of the
                                  # trace: they are what a targeted recovery
                                  # collection spawns the expert at
                                  pelvis_xy=[float(v) for v in pel],
                                  pelvis_yaw=float(p.pelvis_yaw(st)),
                                  obj_xy=[float(v) for v in obj[:2]]))
                if st["fallen"]:
                    why = "[vla] fell"
                    break
                if rise > LIFT_M:
                    if lifted_at is None:
                        lifted_at = time.time()
                    elif time.time() - lifted_at > 2.0:   # held, not a bounce
                        ok, why = True, "ok"
                        break
                else:
                    lifted_at = None
                if _stalled(trace, commanded_close):
                    why = "[vla] reach: stalled (--stall-abort)"
                    break
        finally:
            p.cmd.exit_streamed()
            p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
            obs_src.close()

        # Classify the failure by how far it got, in the manifest's own words.
        if not ok and not why.startswith("[vla] fell") and trace:
            min_dist = min(r["dist"] for r in trace)
            if min_dist > NEAR_M:
                why = f"[vla] walk: never closer than {min_dist:.2f} m"
            elif not commanded_close:
                why = f"[vla] reach: at the table (min {min_dist:.2f} m) but never commanded a close"
            else:
                moved = float(np.linalg.norm(
                    np.asarray(p.object_pos(obj_key, p.obs()))[:2] - obj0[:2]))
                why = (f"[vla] close: commanded a grasp but no lift "
                       f"(object moved {moved * 100:.0f} cm laterally)")

        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            with open(os.path.join(trace_dir, f"round_{counter['n']:03d}.json"), "w") as f:
                json.dump(dict(ok=ok, why=why, trace=trace), f)
        counter["n"] += 1
        return ok, why

    return vla_round


def make_vla_policy_async(link: PolicyLink, task: str,
                          inference_interval: float = 0.4,
                          trace_dir: str | None = None):
    """The official streaming pattern: never stop publishing.

    NVIDIA's own client (gear_sonic/scripts/run_vla_inference.py) does not
    execute-a-block-then-replan the way make_vla_policy above does. It runs
    inference in a WORKER THREAD on a fresh observation every 0.4 s, keeps a
    cached chunk, and the main loop publishes from that cache at the action
    rate WITHOUT EVER PAUSING -- when a fresh chunk lands it starts at a
    LATENCY-COMPENSATED index (inference_delay x rate), so the robot is always
    executing an action whose timestamp matches "now", not the moment the
    camera frame was taken.

    Why the difference matters here: the blocking client freezes the deploy on
    a held token for 0.3-0.6 s out of every ~0.9 s. Training data contains no
    such stop-and-go (the expert streamed continuously), and our dominant
    failure mode is a standing-still attractor -- a client that periodically
    freezes the intent is plausibly feeding it. This variant removes that
    difference; the A/B against the blocking client is the measurement.

    Differences from NVIDIA's client kept ON PURPOSE:
      - 30 Hz publish, because OUR dataset is 30 fps (theirs is 50 both sides;
        each is self-consistent, mixing them would replay motions fast).
      - FSQ snap stays (sonic_token.snap): NVIDIA sends raw floats with only a
        |x| > 1.25 sanity bound. Snapping to the codebook cannot hurt a
        near-grid prediction; dropping it is a separate ablation.
    """
    import threading

    counter = {"n": 0}
    RATE = 30.0

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record):
        from primitives import LocomotionMode

        obs_src = ObsBuilder(task)
        if not obs_src.wait_ready():
            obs_src.close()
            return False, "[vla] camera/state streams never became ready"

        s0 = p.obs()
        obj0 = np.asarray(p.object_pos(obj_key, s0))
        trace = []
        ok, why = False, "[vla] timeout"
        commanded_close = False

        # -- worker: obs -> inference -> (chunk, started_at), forever ---------
        # The worker owns the camera/state sockets and the policy link; the
        # main thread owns the commander and the privileged predicates. ZMQ
        # sockets are not thread-safe, so the split is load-bearing.
        latest = {"chunk": None, "t0": None}
        lock = threading.Lock()
        stop = threading.Event()

        def worker():
            while not stop.is_set():
                t0 = time.time()
                obs = obs_src.build()
                if obs is not None:
                    try:
                        tok, grip = link.get_chunk(obs)
                    except Exception:
                        time.sleep(0.1)
                        continue
                    with lock:
                        latest["chunk"] = (tok, grip)
                        latest["t0"] = t0
                rest = inference_interval - (time.time() - t0)
                if rest > 0:
                    time.sleep(rest)

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        # wait for the first chunk, prime, switch -- same order the replay
        # diag validated (buffer before mode switch, cpp:618)
        t_wait = time.time()
        while True:
            with lock:
                ready = latest["chunk"] is not None
            if ready or time.time() - t_wait > 15:
                break
            time.sleep(0.05)
        if not ready:
            stop.set()
            obs_src.close()
            return False, "[vla] first inference never returned"

        with lock:
            tok0, grip0 = latest["chunk"]
        l0, r0 = expand_hands(grip0[0])
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)
        p.cmd.enter_streamed()
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)

        t_start = time.time()
        frame_index = 0
        lifted_at = None
        chunk_t0, idx = None, 0
        tok, grip = tok0, grip0
        last_status = 0.0
        try:
            while time.time() - t_start < TIMEOUT_S:
                tick = time.time()
                # adopt a fresh chunk at a latency-compensated index
                with lock:
                    if latest["t0"] is not None and latest["t0"] != chunk_t0:
                        tok, grip = latest["chunk"]
                        chunk_t0 = latest["t0"]
                        idx = min(int((time.time() - chunk_t0) * RATE),
                                  len(tok) - 1)
                j = min(idx, len(tok) - 1)
                lhj, rhj = expand_hands(grip[j])
                if grip[j] < 0.005:
                    commanded_close = True
                p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj,
                                  frame_index=frame_index)
                frame_index += 1
                idx = min(idx + 1, len(tok) - 1)

                # predicates at ~2 Hz, off the hot path
                if tick - last_status > 0.5:
                    last_status = tick
                    st = p.obs()
                    obj = np.asarray(p.object_pos(obj_key, st))
                    pel = np.asarray(p.pelvis_xy(st))
                    rise = float(obj[2] - obj0[2])
                    trace.append(dict(t=tick - t_start, rise=rise,
                                      dist=float(np.linalg.norm(obj[:2] - pel)),
                                      fallen=bool(st["fallen"]),
                                      grip=float(grip[j]),
                                      pelvis_xy=[float(v) for v in pel],
                                      pelvis_yaw=float(p.pelvis_yaw(st)),
                                      obj_xy=[float(v) for v in obj[:2]]))
                    if st["fallen"]:
                        why = "[vla] fell"
                        break
                    if rise > LIFT_M:
                        if lifted_at is None:
                            lifted_at = tick
                        elif tick - lifted_at > 2.0:
                            ok, why = True, "ok"
                            break
                    else:
                        lifted_at = None
                    if _stalled(trace, commanded_close):
                        why = "[vla] reach: stalled (--stall-abort)"
                        break

                rest = 1.0 / RATE - (time.time() - tick)
                if rest > 0:
                    time.sleep(rest)
        finally:
            stop.set()
            p.cmd.exit_streamed()
            p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
            th.join(timeout=1.0)
            obs_src.close()

        if not ok and why == "[vla] timeout" and trace:
            min_dist = min(r["dist"] for r in trace)
            if min_dist > NEAR_M:
                why = f"[vla] walk: never closer than {min_dist:.2f} m"
            elif not commanded_close:
                why = (f"[vla] reach: at the table (min {min_dist:.2f} m) "
                       f"but never commanded a close")
            else:
                moved = float(np.linalg.norm(
                    np.asarray(p.object_pos(obj_key, p.obs()))[:2] - obj0[:2]))
                why = (f"[vla] close: commanded a grasp but no lift "
                       f"(object moved {moved * 100:.0f} cm laterally)")

        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            with open(os.path.join(trace_dir, f"round_{counter['n']:03d}.json"), "w") as f:
                json.dump(dict(ok=ok, why=why, trace=trace), f)
        counter["n"] += 1
        return ok, why

    return vla_round

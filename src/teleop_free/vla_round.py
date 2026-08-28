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

# NVIDIA's own latency compensation, used verbatim so this tracks their
# semantics rather than a lookalike of ours (docs/official_parity_audit.md A2).
from gear_sonic.utils.inference.vla_utils import (  # noqa: E402
    calculate_latency_compensated_index as latency_index,
)

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

# FSQ snap on the predicted token before it goes on the wire. Ours has always
# snapped; NVIDIA's own client sends raw floats (run_vla_inference.py, only a
# whole-chunk |x|>1.25 rejection). Round 3's open-loop measured the snap
# COSTING ~16% nmse_token (0.1592 snapped vs 0.1369 raw, checkpoint-9000 h16).
# This switch exists to run the closed-loop half of that ablation; the default
# stays on because 5.9/13.7/28.0 were measured with it on.
SNAP = True

# --- temporal ensembling (ACT-style), OFF by default -------------------------
# The async client replaces the cached chunk wholesale: a fresh 40-step
# prediction lands, the old one is dropped, and execution jumps to a
# latency-compensated index. Consecutive predictions overlap by ~28 of 40
# frames and never see each other, so every adoption is a small discontinuity
# in the commanded intent.
#
# Ensembling averages, for the CURRENT wall-clock instant, every buffered
# chunk's prediction of that same instant. Chunk c predicted absolute time
# c.t0 + i/RATE at index i, so index i_c = (now - c.t0) * RATE aligns them.
# This is the standard fix from ACT / Diffusion Policy and it targets exactly
# this round's dominant failure: 44 of 52 reach failures are "arm up, palm
# never converges on the object", which is what accumulated error looks like.
#
# ENSEMBLE_M weights by chunk AGE, 0 = newest:
#     m > 0  favour the freshest observation (we already latency-compensate)
#     m = 0  uniform mean
#     m < 0  favour older predictions, the direction ACT's paper uses
#
# Averaging snapped tokens lands off-grid, which is fine and probably helps:
# NVIDIA's own client sends raw floats, and averaging quantised values
# recovers sub-grid precision.
#
# Default OFF (N=1) so the 5.9/13.7/28.0/47.5 ladder stays byte-comparable.
ENSEMBLE_N = 1        # chunks to blend; 1 = off, current behaviour
ENSEMBLE_M = 0.3      # exponential weight decay per chunk of age


def _ensemble(buf, now, rate):
    """Blend every buffered chunk's prediction of `now`. -> (tok, grip) or None.

    buf is newest-last: [(tok, grip, t0), ...]. A chunk whose aligned index
    has run past its end is exhausted and contributes nothing.
    """
    picks, ws = [], []
    for age, (tok, grip, t0) in enumerate(reversed(buf)):     # age 0 = newest
        # round, not truncate (same half-frame bias the main path had) -- but
        # NOT the official latency_index here: it CLAMPS to the last frame,
        # which would silently drag exhausted chunks back into the blend
        # instead of dropping them. Dropping is the ensemble's semantics.
        i = int(round((now - t0) * rate))
        if i < 0 or i >= len(tok):
            continue
        picks.append((tok[i], grip[i]))
        ws.append(np.exp(-ENSEMBLE_M * age))
    if not picks:
        return None
    w = np.asarray(ws, dtype=np.float64)
    w /= w.sum()
    tok = np.sum([p[0] * wi for p, wi in zip(picks, w)], axis=0)
    grip = float(np.sum([p[1] * wi for p, wi in zip(picks, w)]))
    return tok, grip
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
# The chunk's native frame rate. It MUST equal the fps of the dataset the
# checkpoint was trained on -- the chunk is a sequence of frames, and publishing
# it at a different rate replays the motion at the wrong speed.
#
# Default 30 because every checkpoint that exists today was trained on 30 fps
# data. `convert_to_hf_format.py` now writes 25 (see the DST_FPS comment there
# for why), so the FIRST checkpoint trained on re-converted data has to run with
# TF_DATASET_FPS=25. Kept as an env var rather than flipped now, because the
# wrong value is silent -- the robot just moves at 1.2x or 0.83x speed -- and a
# default that silently becomes wrong on the next retrain is exactly the class
# of failure docs/reference.md is a list of.
DATASET_FPS = float(os.environ.get("TF_DATASET_FPS", "30"))
REPLAN_HZ = DATASET_FPS


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
        """-> (token [40,64], gripper [40]). Snapped unless SNAP is off."""
        out = self._call("get_action_raw", {"obs": obs})
        tok = np.asarray(out["action.motion_token"], dtype=np.float64)
        grip = np.asarray(out["action.right_gripper"], dtype=np.float64).reshape(-1)
        return (snap(tok) if SNAP else tok), grip


def make_vla_policy(link: PolicyLink, task: str, execution_horizon: int = 16,
                    trace_dir: str | None = None):
    """Build the callable grasp_gym.VLA_POLICY expects.

    Called with the robot settled at its spawn, facing the object, video and
    (optionally) the exporter already recording. Must return (ok, why).
    """
    counter = {"n": 0}

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record,
                  replace=None, spawn_xy=None):
        from primitives import LocomotionMode

        obs_src = ObsBuilder(task)
        if not obs_src.wait_ready():
            obs_src.close()
            return False, "[vla] camera/state streams never became ready"

        s0 = p.obs()
        t_entry = time.time()
        obj0 = np.asarray(p.object_pos(obj_key, s0))
        trace = []
        ok, why = False, "[vla] timeout"
        dt = 1.0 / REPLAN_HZ
        commanded_close = False

        # Prime the buffer BEFORE the mode switch, then hold the first token
        # across it -- streamed mode over an empty buffer stops control
        # (g1_deploy_onnx_ref.cpp:618), and the first v4 frame is also what
        # flips is_using_encoder_ off. Same order diag_token_replay validated.
        if replace is not None and os.environ.get(
                "LOCOMANI_REPLACE", "0") == "1":
            replace()   # legacy sim-only re-teleport, off by default
        obs = obs_src.build()
        tok0, grip0 = link.get_chunk(obs)
        l0, r0 = expand_hands(grip0[0])

        def _mark(tag):   # see the async client's comment: prelude forensics
            try:
                sm = p.obs()
                p.log(f"[vla] prelude {tag}: pelvis "
                      f"{np.round(p.pelvis_xy(sm), 3)} "
                      f"grip0 {float(grip0[0]):+.4f}")
            except Exception:
                pass

        _mark("chunk-ready")
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)
        _mark("primed")
        p.cmd.enter_streamed()
        for _ in range(30):
            p.cmd.token_frame(tok0[0], right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(0.02)
        _mark("streamed")

        t_start = time.time()
        s_ts = p.obs()
        # Where the robot ACTUALLY is when the VLA starts being measured.
        # Added after the stand-start finding: in ~2/3 of rounds something
        # moves the robot from its settled spawn to the stand pose during
        # the prelude, so t=0 of the trace is NOT the settled spawn. These
        # three fields make that drift measurable per round.
        prelude = dict(
            entry_pelvis=[round(float(v), 4) for v in p.pelvis_xy(s0)],
            tstart_pelvis=[round(float(v), 4) for v in p.pelvis_xy(s_ts)],
            prelude_s=round(t_start - t_entry, 2))
        if spawn_xy is not None:
            # The fix's acceptance criterion, IN the data: pelvis-to-spawn
            # distance at the moment measurement starts. docs promise
            # "within 0.1 m of the spawn"; record it every round and say so
            # loudly when it does not hold, so a regression of the
            # re-teleport can never hide behind a proxy metric again.
            off0 = float(np.linalg.norm(
                np.asarray(p.pelvis_xy(s_ts)) - np.asarray(spawn_xy)))
            prelude["tstart_spawn_off"] = round(off0, 4)
            if off0 > 0.12:
                p.log(f"[vla] WARNING tstart {off0:.2f} m off the spawn "
                      f"(acceptance is 0.1) -- stand-start drift is back?")
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
                palm = np.asarray(st["palms"]["right"]["pos"], dtype=float)
                trace.append(dict(t=time.time() - t_start, rise=rise, dist=dist,
                                  fallen=bool(st["fallen"]),
                                  grip=float(grip[:n].min()),
                                  palm_z=float(palm[2]),
                                  palm_obj=float(np.linalg.norm(palm - obj)),
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
                json.dump(dict(ok=ok, why=why, **prelude, trace=trace), f)
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
      - publish at DATASET_FPS, because OUR dataset is not 50 fps (theirs is
        50 on both sides; each is self-consistent, mixing them would replay
        motions fast).
      - FSQ snap stays (sonic_token.snap): NVIDIA sends raw floats with only a
        |x| > 1.25 sanity bound. Snapping to the codebook cannot hurt a
        near-grid prediction; dropping it is a separate ablation.
    """
    import threading

    counter = {"n": 0}
    RATE = DATASET_FPS

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record,
                  replace=None, spawn_xy=None):
        from primitives import LocomotionMode
        from gear_sonic.utils.inference.initial_poses import (
            LATENT_INITIAL_MOTION_TOKEN as INIT_TOK)
        from sonic_token import GRIPPER_MAX as _G_OPEN

        # FIRST act: get off the planner channel. The stand-start drift is a
        # planner-mode disease (IDLE limit cycle, ~0.15 m/s in drift-prone
        # spawn configs); the token channel measures rock-stable
        # (0.013 m net / 30 s). NVIDIA's initial-pose mechanism, adapted:
        # prime the deploy with the predefined standing token (no inference
        # needed), enter streamed mode, and keep that token flowing from a
        # keepalive thread while the slow setup below runs. Works unchanged
        # on real hardware -- no sim reset involved. The commander socket
        # belongs to the keepalive until _teardown(); the main thread must
        # not touch p.cmd before then.
        l_open, r_open = expand_hands(_G_OPEN)
        for _ in range(30):
            p.cmd.token_frame(INIT_TOK, right_hand=r_open, left_hand=l_open,
                              frame_index=0)
            time.sleep(0.02)
        p.cmd.enter_streamed()
        t_hold = time.time()
        keep_stop = threading.Event()

        def _keepalive():
            while not keep_stop.is_set():
                p.cmd.token_frame(INIT_TOK, right_hand=r_open,
                                  left_hand=l_open, frame_index=0)
                time.sleep(1.0 / RATE)

        keep_th = threading.Thread(target=_keepalive, daemon=True)
        keep_th.start()

        def _teardown():
            keep_stop.set()
            keep_th.join(timeout=1.0)

        obs_src = ObsBuilder(task)
        if not obs_src.wait_ready():
            _teardown()
            obs_src.close()
            return False, "[vla] camera/state streams never became ready"

        s0 = p.obs()
        t_entry = time.time()
        obj0 = np.asarray(p.object_pos(obj_key, s0))
        trace = []
        ok, why = False, "[vla] timeout"
        commanded_close = False

        # -- worker: obs -> inference -> (chunk, started_at), forever ---------
        # The worker owns the camera/state sockets and the policy link; the
        # main thread owns the commander and the privileged predicates. ZMQ
        # sockets are not thread-safe, so the split is load-bearing.
        latest = {"chunk": None, "t0": None}
        buf = []                      # [(tok, grip, t0)], newest last
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
                        if ENSEMBLE_N > 1:
                            buf.append((tok, grip, t0))
                            del buf[:-ENSEMBLE_N]
                rest = inference_interval - (time.time() - t0)
                if rest > 0:
                    time.sleep(rest)

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        # Wait for a chunk computed from an observation of the HELD robot
        # (t0 > t_hold): a chunk from the planner era would encode a stale
        # intent. The keepalive keeps the standing token flowing meanwhile.
        ready = False
        t_wait = time.time()
        while time.time() - t_wait < 15:
            with lock:
                ready = latest["t0"] is not None and latest["t0"] > t_hold
            if ready:
                break
            time.sleep(0.05)
        if not ready:
            _teardown()
            stop.set()
            obs_src.close()
            return False, ("[vla] initial-pose hold: no fresh chunk within "
                           "15 s -- policy worker stalled; round aborted")

        if replace is not None and os.environ.get(
                "LOCOMANI_REPLACE", "0") == "1":
            replace()   # legacy sim-only re-teleport, off by default

        with lock:
            tok0, grip0 = latest["chunk"]
        l0, r0 = expand_hands(grip0[0])

        # Hand the commander socket back to this thread, then BLEND from the
        # standing token into the first chunk frame (1 s linear, the
        # official client's default) so the handover has no step in it.
        _teardown()
        BLEND_S = 1.0
        n_bl = max(1, int(BLEND_S * RATE))
        for i in range(n_bl):
            a = (i + 1) / n_bl
            tk = (1.0 - a) * INIT_TOK + a * np.asarray(tok0[0])
            p.cmd.token_frame(tk, right_hand=r_open, left_hand=l_open,
                              frame_index=0)
            time.sleep(1.0 / RATE)

        # Prelude segmentation marks. The stand-start gap (~0.35 m of
        # table-ward motion before t_start) has two candidates: (a) a creep
        # smeared over the whole prelude, (b) the 1.2 s priming below
        # executing the held first token (the pre-trace jaw close/open seen
        # in every eval video says priming DOES execute poses). One pelvis
        # fix per segment decides it.
        def _mark(tag):
            try:
                sm = p.obs()
                p.log(f"[vla] prelude {tag}: pelvis "
                      f"{np.round(p.pelvis_xy(sm), 3)} "
                      f"grip0 {float(grip0[0]):+.4f}")
            except Exception as e:  # a missing mark must LOOK missing:
                p.log(f"[vla] prelude {tag}: MARK FAILED ({e}) -- do not "
                      f"read this segment as drift-free")

        _mark("streamed")

        t_start = time.time()
        s_ts = p.obs()
        # See the blocking client's comment: t=0 of the trace is not the
        # settled spawn in ~2/3 of rounds; these fields measure the gap.
        prelude = dict(
            entry_pelvis=[round(float(v), 4) for v in p.pelvis_xy(s0)],
            tstart_pelvis=[round(float(v), 4) for v in p.pelvis_xy(s_ts)],
            prelude_s=round(t_start - t_entry, 2))
        if spawn_xy is not None:
            # The fix's acceptance criterion, IN the data: pelvis-to-spawn
            # distance at the moment measurement starts. docs promise
            # "within 0.1 m of the spawn"; record it every round and say so
            # loudly when it does not hold, so a regression of the
            # re-teleport can never hide behind a proxy metric again.
            off0 = float(np.linalg.norm(
                np.asarray(p.pelvis_xy(s_ts)) - np.asarray(spawn_xy)))
            prelude["tstart_spawn_off"] = round(off0, 4)
            if off0 > 0.12:
                p.log(f"[vla] WARNING tstart {off0:.2f} m off the spawn "
                      f"(acceptance is 0.1) -- stand-start drift is back?")
        frame_index = 0
        lifted_at = None
        chunk_t0, idx = None, 0
        tok, grip = tok0, grip0
        last_status = 0.0
        try:
            while time.time() - t_start < TIMEOUT_S:
                tick = time.time()
                if ENSEMBLE_N > 1:
                    # Blend every overlapping prediction of THIS instant.
                    # Indexing is by absolute time, which also removes the
                    # drift the incrementing idx below accumulates when the
                    # loop does not land exactly on 1/RATE.
                    with lock:
                        blended = _ensemble(buf, tick, RATE)
                        if blended is None and latest["chunk"] is not None:
                            blended = (latest["chunk"][0][-1],
                                       float(latest["chunk"][1][-1]))
                    tok_j, grip_j = blended
                else:
                    # adopt a fresh chunk at a latency-compensated index,
                    # using NVIDIA's own function. Ours was semantically
                    # identical except it TRUNCATED where the official rounds
                    # -- a systematic half-frame (~17 ms at 30 Hz) lag, always
                    # in the same direction. docs/official_parity_audit.md A2.
                    with lock:
                        if latest["t0"] is not None and latest["t0"] != chunk_t0:
                            tok, grip = latest["chunk"]
                            chunk_t0 = latest["t0"]
                            idx = latency_index(time.time() - chunk_t0,
                                                RATE, len(tok))
                    j = min(idx, len(tok) - 1)
                    tok_j, grip_j = tok[j], float(grip[j])
                    idx = min(idx + 1, len(tok) - 1)

                lhj, rhj = expand_hands(grip_j)
                if grip_j < 0.005:
                    commanded_close = True
                p.cmd.token_frame(tok_j, right_hand=rhj, left_hand=lhj,
                                  frame_index=frame_index)
                frame_index += 1

                # predicates at ~2 Hz, off the hot path
                if tick - last_status > 0.5:
                    last_status = tick
                    st = p.obs()
                    obj = np.asarray(p.object_pos(obj_key, st))
                    pel = np.asarray(p.pelvis_xy(st))
                    rise = float(obj[2] - obj0[2])
                    # palm state answers the question round 3's traces could
                    # not: of the "at the table, never closed" failures, did
                    # the arm ever come up, and how near did the hand get?
                    palm = np.asarray(st["palms"]["right"]["pos"], dtype=float)
                    trace.append(dict(t=tick - t_start, rise=rise,
                                      dist=float(np.linalg.norm(obj[:2] - pel)),
                                      fallen=bool(st["fallen"]),
                                      grip=float(grip_j),
                                      pelvis_xy=[float(v) for v in pel],
                                      pelvis_yaw=float(p.pelvis_yaw(st)),
                                      obj_xy=[float(v) for v in obj[:2]],
                                      palm_z=float(palm[2]),
                                      palm_obj=float(np.linalg.norm(palm - obj))))
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
                json.dump(dict(ok=ok, why=why, **prelude, trace=trace), f)
        counter["n"] += 1
        return ok, why

    return vla_round

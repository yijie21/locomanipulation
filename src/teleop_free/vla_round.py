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
from collections import deque
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

from sonic_token import (  # noqa: E402
    GRIPPER_MAX, GRIPPER_MIN, expand_hands, snap,
)

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

# --- gripper gain, OFF by default (1.0 = passthrough) ------------------------
# The one discriminator that reproduces across every batch measured so far:
# a successful round drives the commanded gripper to GRIPPER_MIN (-0.0200,
# the clip floor); a `reach` failure parks it at +0.0115..+0.0195 and 0% of
# them ever cross 0.010. The training gripper is hard bimodal -- open >0.020
# is 53.3% of frames, shut <0.005 is 43.5%, and the band between them is
# 3.3%. Failing rounds sit in that hole for the whole episode.
#
# A hard threshold is the obvious fix and it is WRONG here: the model does not
# separate "near" from "at". Measured on one failed round, the gripper was
# already down at +0.0189 while the palm was still 0.195 m from the bottle, and
# only reached +0.0178 when the palm got to 0.136 m. A threshold placed to
# catch the second fires on the first too, and closes the jaw on air 20 cm out.
#
# This instead applies a GAIN to the deviation from fully-open, which preserves
# the model's own ordering and only amplifies it:
#
#     g' = clip(GRIPPER_MAX - k * (GRIPPER_MAX - g), GRIPPER_MIN, GRIPPER_MAX)
#
# At k = 3, against the measured failure band (p25 0.0159 / p50 0.0168 /
# p75 0.0178) and the far-field value 0.0245:
#
#     0.0245 (far)              -> 0.0245   unchanged, stays open
#     0.0189 (palm 0.195 m out) -> 0.0077   still open   <- correctly no fire
#     0.0178 (p75)              -> 0.0044   CLOSES
#     0.0168 (median)           -> 0.0014   CLOSES
#
# so k = 3 puts the effective trigger at 0.0178 -- the p75 of the failure band
# -- and in the round above it fires at palm 0.136 m and not at 0.195 m.
#
# Default 1.0: every number on record was measured without it.
GRIP_GAIN = 1.0



def _tstart_view(p, st, obj_key, prelude, gx=0.80, gy=0.90):
    """Is the target still in the ego frame when the trace starts?

    The gym gates ego visibility at the SPAWN -- after the settle, before the
    VLA fork (`[view] object not in ego view at spawn`, ~12% of rounds at
    --spawn-edge 1.1 1.3). Nothing re-checked it AFTER the prelude, and the
    prelude is where the 18-D path holds its pose open-loop, so a round could
    start blind and then be scored as a policy failure. This records where the
    target actually sits in the frame at t=0 (normalised, frame edge 1.0, live
    from the privileged publisher's camera) so such a round is identifiable
    offline instead of assumed away.

    Warned, not aborted: aborting here would have to unwind the feeder threads
    and the running recorder, and the honest treatment of a blind round in a
    PAIRED comparison is to drop it from both arms, which is an offline
    decision. tools/failure_taxonomy.py buckets it.

    Margin, measured on the pinned eval scene over 62 cover400 spawns
    (src/teleop_free/diag_spawn_view.py, 2026-09-04): the bottle leaves the image at a
    pelvis->object range of ~1.95 m and the conservative central-90% gate
    breaks at 1.68 m, while real spawns sit at 1.17-1.47 m (u_y 0.61-0.81).
    Sideways there is 78-105 cm of slack (|u_x| <= 0.40 against a 0.80 gate),
    so only BACKWARD drift can lose the target -- and the settle's measured
    direction is table-ward, which gains margin rather than spending it.
    """
    cam = ((st.get("objects") or {}).get(obj_key) or {}).get("cam")
    prelude["tstart_cam"] = (None if cam is None
                             else [round(float(v), 3) for v in cam])
    if cam is None:
        p.log("[vla] WARNING target is BEHIND the ego camera at trace t=0 -- "
              "this round is blind and is not a policy test")
    elif abs(cam[0]) > gx or abs(cam[1]) > gy:
        p.log(f"[vla] WARNING target outside the ego gate at trace t=0 "
              f"(cam=[{cam[0]:+.2f} {cam[1]:+.2f}], gate {gx}/{gy}) -- the "
              f"prelude drifted it off frame; not a policy test")


def _obj_extra(st, obj_key):
    """Per-tick object facts for the failure taxonomy (2026-09-03): tilt from
    upright in degrees, which robot links touch the object, and its height.
    Old traces lack these; tools/failure_taxonomy.py degrades to xy displacement."""
    o = (st.get("objects") or {}).get(obj_key) or {}
    q = o.get("quat"); tilt = None
    if q is not None:
        w_, x_, y_, z_ = (float(v) for v in q)
        r22 = 1.0 - 2.0 * (x_ * x_ + y_ * y_)
        tilt = float(np.degrees(np.arccos(float(np.clip(r22, -1.0, 1.0)))))
    pos = o.get("pos")
    return dict(obj_tilt=tilt, touch=list(o.get("touching") or []),
                touch_links=list(o.get("touch_links") or []),   # every robot link, not just the hands
                obj_z=(None if pos is None else float(pos[2])))

def _grip_gain(g: float) -> float:
    """Amplify the predicted gripper's deviation from fully-open."""
    if GRIP_GAIN == 1.0:
        return g
    return float(np.clip(GRIPPER_MAX - GRIP_GAIN * (GRIPPER_MAX - g),
                         GRIPPER_MIN, GRIPPER_MAX))

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
# RTC (real-time chunking, test-time inpainting variant). When on, the async
# worker tells serve_policy how many frames of the previous chunk have elapsed
# (rtc.elapsed) and how many the NEXT inference will eat (rtc.frozen, rolling
# p90 of measured latency); the server freezes that prefix of the new chunk to
# the old chunk's overlapping rows and ramps the rest. Async client only.
RTC_ON = False
# TF_ACTION18=1: the policy emits the 18-D command (wrist SE(3) + grip per arm
# + base) instead of a 64-D SONIC token, and this round drives the robot the
# way the scripted expert does -- IK to 17 upper-body joint targets on the
# PLANNER channel, never streamed mode. Incompatible with --clean-deploy (that
# shim drops planner commands) and needs the deploy's encoder loaded, i.e. the
# stack must start WITHOUT NO_ENCODER. See action18_client.py.
ACTION18 = os.environ.get("TF_ACTION18", "0") == "1"
# TF_GRIP_DELAY (frames at DATASET_FPS): send the gripper command from N frames
# ago. Diagnosis 2026-09-04 (replay_reach): the token path executes the ARM
# ~0.2 s behind the recorded motion (palm 0.5-5 cm short at the recorded close
# frame, always on the approach side), while the hand joints go straight to
# the hand PD with no lag -- so the policy closes on time onto a hand that is
# not there yet, and the jaw clips the bottle (obj_in_palm y = -4 cm in 42/44
# base100 knock-overs). Delaying only the gripper channel re-aligns the two.
GRIP_DELAY = int(os.environ.get("TF_GRIP_DELAY", "0"))


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
        self.last_rtc = out.get("rtc.applied")  # [overlap, frozen] | None
        dump = os.environ.get("TF_OBS_DUMP")
        if dump:   # diagnostic: what the policy saw + what it answered
            os.makedirs(dump, exist_ok=True)
            n = getattr(self, "_dump_n", 0); self._dump_n = n + 1
            np.savez_compressed(os.path.join(dump, f"obs_{n:05d}.npz"),
                                wall=time.time(), tok=np.asarray(out["action.motion_token"]),
                                grip=np.asarray(out["action.right_gripper"]),
                                **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v))
                                   for k, v in obs.items()})
        if ACTION18:
            a = np.asarray(out["action.a18"], dtype=np.float64)
            # the right grip rides along as the second return so every
            # predicate downstream (commanded_close, grip traces) is unchanged
            return a, a[:, 6].reshape(-1)
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

        def _mark(tag):   # see the async client's comment: prelude forensics
            try:
                sm = p.obs()
                p.log(f"[vla] prelude {tag}: pelvis "
                      f"{np.round(p.pelvis_xy(sm), 3)}")
            except Exception:
                pass

        # Entry on the STANDING token, not chunk-0 (port of the async
        # client's bbc207d fix; blocking kept the old sequence and paid for
        # it with ~1.2 s of "start walking" executed before t_start). The
        # sequential client needs NO keepalive thread: during the first
        # obs+inference below the deploy starves and clamps -- on a STANDING
        # frame, which is stable. Clamp-on-walking is what used to pitch the
        # robot over; clamp-on-standing is just standing.
        from initial_pose import initial_token
        INIT_TOK = initial_token()
        l_open, r_open = expand_hands(GRIPPER_MAX)
        for _ in range(30):
            p.cmd.token_frame(INIT_TOK, right_hand=r_open, left_hand=l_open,
                              frame_index=0)
            time.sleep(0.02)
        _mark("primed")
        p.cmd.enter_streamed()
        t_obs0 = time.time()
        obs = obs_src.build()
        tok0, grip0 = link.get_chunk(obs)
        _mark("chunk-ready")
        l0, r0 = expand_hands(grip0[0])
        # 1 s linear blend from the standing token into chunk-0's first
        # frame (the official client's handover, same as async).
        n_bl = max(1, int(1.0 * REPLAN_HZ))
        for i in range(n_bl):
            a = (i + 1) / n_bl
            tk = (1.0 - a) * INIT_TOK + a * np.asarray(tok0[0])
            p.cmd.token_frame(tk, right_hand=r0, left_hand=l0, frame_index=0)
            time.sleep(dt)
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
            # The acceptance criterion, IN the data: pelvis distance from
            # the HANDOVER pose (grasp_gym's s_fork, post-settle -- since
            # 2026-08-31; earlier runs measured from the raw teleport
            # landing, so their recorded values sit ~0.13 m higher and are
            # not comparable across the change). What this bounds is the
            # VLA prelude itself: the token hold plus the 1 s blend, during
            # which chunk-0 is already partially executing. > 0.12 m means
            # the prelude is again moving the robot before trace t=0.
            off0 = float(np.linalg.norm(
                np.asarray(p.pelvis_xy(s_ts)) - np.asarray(spawn_xy)))
            prelude["tstart_spawn_off"] = round(off0, 4)
            if off0 > 0.12:
                p.log(f"[vla] WARNING tstart {off0:.2f} m off the handover "
                      f"pose (acceptance is 0.1) -- prelude drift is back?")
        _tstart_view(p, s_ts, obj_key, prelude)
        frame_index = 0
        lifted_at = None

        # GAP FEEDER (2026-08-31). The blocking loop stops publishing for the
        # whole [trace bookkeeping + obs build + inference] span (~0.4-1.0 s).
        # The deploy consumes streamed frames at its 50 Hz control tick and,
        # on starvation, CLAMPS the reference to the last received frame
        # (zmq_endpoint_interface.hpp "Clamping to last frame") -- a frozen
        # mid-stride pose with non-zero reference velocities that the RL
        # controller keeps leaning into. At spawn 1.1-1.3 the first chunk
        # commands a vigorous walk, so the first gap pitched the robot onto
        # its face. The fix is the async client's core idea in one thread:
        # keep feeding the REMAINDER of the current chunk (frames n..40)
        # during the gap, and hand frame_index continuity back on join.
        # Socket ownership stays partitioned: the feeder owns p.cmd for
        # exactly the span where the main thread only touches obs/policy
        # sockets.
        import threading as _th
        _feed = {"fi": 0}

        def _start_feeder(ftok, fgrip, j0, fi0):
            stop = _th.Event()

            def _run():
                j, fi = j0, fi0
                while not stop.is_set():
                    jj = min(j, len(ftok) - 1)  # exhausted -> explicit hold
                    lhj, rhj = expand_hands(fgrip[jj])
                    p.cmd.token_frame(ftok[jj], right_hand=rhj, left_hand=lhj,
                                      frame_index=fi)
                    _feed["fi"] = fi
                    j += 1
                    fi += 1
                    time.sleep(dt)

            th = _th.Thread(target=_run, daemon=True)
            th.start()
            return stop, th

        # Seed the first in-loop gap with chunk-0: the blend above ended on
        # tok0[0], so the feeder continues tok0 from frame 0 while the loop's
        # first obs+inference runs -- continuous motion from the handover on.
        cur = (tok0, grip0, 0)   # (tok, grip, next frame) of the running chunk
        feeder = None

        def _stop_feeder():
            nonlocal feeder, frame_index
            if feeder is not None:
                feeder[0].set()
                feeder[1].join(timeout=1.0)
                frame_index = _feed["fi"] + 1
                feeder = None

        try:
            while time.time() - t_start < TIMEOUT_S:
                if cur is not None:
                    feeder = _start_feeder(cur[0], cur[1], cur[2], frame_index)
                t_obs = time.time()
                obs = obs_src.build()
                if obs is None:
                    _stop_feeder()
                    why = "[vla] observation stream dropped"
                    break
                tok, grip = link.get_chunk(obs)
                _stop_feeder()
                # Latency-compensated start (the async client's adoption
                # rule, missing here since day one): the chunk was planned
                # from the state at t_obs, and the feeder has meanwhile
                # executed the gap on the OLD chunk. Starting at frame 0
                # re-executes ~T_inf of the past every cycle -- the visible
                # per-chunk stagger. Start where the plan says "now" is.
                skip = latency_index(time.time() - t_obs, REPLAN_HZ, len(tok))
                n = max(1, min(execution_horizon, len(tok) - skip))
                for j in range(skip, skip + n):
                    lhj, rhj = expand_hands(grip[j])
                    if grip[j] < 0.005:
                        commanded_close = True
                    p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj,
                                      frame_index=frame_index)
                    frame_index += 1
                    time.sleep(dt)
                cur = (tok, grip, skip + n)

                st = p.obs()
                obj = np.asarray(p.object_pos(obj_key, st))
                pel = np.asarray(p.pelvis_xy(st))
                rise = float(obj[2] - obj0[2])
                dist = float(np.linalg.norm(obj[:2] - pel))
                palm = np.asarray(st["palms"]["right"]["pos"], dtype=float)
                trace.append(dict(t=time.time() - t_start, rise=rise, dist=dist,
                                  fallen=bool(st["fallen"]),
                                  grip=float(grip[skip:skip + n].min()),
                                  palm_z=float(palm[2]),
                                  palm_obj=float(np.linalg.norm(palm - obj)),
                                      **_obj_extra(st, obj_key),
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
            _stop_feeder()   # an exception mid-gap must not leave a thread
            #                  publishing token frames after exit_streamed
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
        # OURS, not NVIDIA's: theirs holds the arms forward and bent with the
        # torso pitched, which is outside the distribution every episode was
        # trained in, so the handover used to lean forward and then snap
        # upright over the blend. TF_INIT_TOKEN=official restores it.
        # See initial_pose.py for the measurement and why a medoid.
        from initial_pose import initial_token
        INIT_TOK = initial_token()
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
        sender18, _fv0 = None, None
        if ACTION18:
            # 18-D drives the PLANNER channel, so there is nothing to get off
            # and nothing to prime: hold the arm where it is and the base IDLE.
            from action18_client import Action18Sender
            from primitives import LocomotionMode as _LM, upper_body_from_mj as _u17
            sender18 = Action18Sender(p)
            _s0 = p.obs()
            _fv0 = p._facing(_s0)
            _anchor = (np.asarray(p.pelvis_xy(_s0), dtype=float).copy(),
                       float(p.pelvis_yaw(_s0)))
            for _ in range(30):
                sender18.hold(_anchor[0], _anchor[1], l_open, r_open)
                time.sleep(0.02)
        else:
            for _ in range(30):
                p.cmd.token_frame(INIT_TOK, right_hand=r_open, left_hand=l_open,
                                  frame_index=0)
                time.sleep(0.02)
            p.cmd.enter_streamed()
        t_hold = time.time()
        keep_stop = threading.Event()

        def _keepalive():
            while not keep_stop.is_set():
                if ACTION18:
                    # closed on BOTH heading and position against the handover
                    # pose; measured 3.5 cm vs 12.7 cm for planner IDLE alone
                    # (diag_station_keep.py). See action18_client.hold.
                    sender18.hold(_anchor[0], _anchor[1], l_open, r_open)
                else:
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
        # Empirical latency record for the lockstep client (tools/
        # latency_dist.py): per-inference obs->chunk seconds and the spacing
        # between consecutive obs (the K the sim advances per cycle).
        lat_rec = {"infer_s": [], "cycle_s": []}

        def worker():
            rtc_prev_t0 = None    # obs time of the last SUCCESSFUL inference
            rtc_lat = []          # last 20 measured obs->chunk latencies (s)
            rtc_first = True
            while not stop.is_set():
                t0 = time.time()
                obs = obs_src.build()
                if obs is not None:
                    if RTC_ON:
                        if rtc_first:
                            obs["rtc.reset"] = 1
                        # elapsed 0 on the first request: no injection server-
                        # side, but the meta's presence makes it CACHE the
                        # chunk, so request 2 has a prefix to freeze.
                        e = (0 if rtc_prev_t0 is None
                             else int(round((t0 - rtc_prev_t0) * RATE)))
                        lat = sorted(rtc_lat)
                        p90 = lat[int(0.9 * (len(lat) - 1))] if lat else 0.5
                        obs["rtc.elapsed"] = e
                        obs["rtc.frozen"] = max(1, min(20, int(np.ceil(
                            p90 * RATE)) + 1))
                    try:
                        tok, grip = link.get_chunk(obs)
                    except Exception:
                        time.sleep(0.1)
                        continue
                    lat_rec["infer_s"].append(time.time() - t0)
                    if latest["t0"] is not None:
                        lat_rec["cycle_s"].append(t0 - latest["t0"])
                    lat_rec.setdefault("chunk_grip_min", []).append(float(np.min(grip)))
                    lat_rec.setdefault("chunk_tok_span", []).append(
                        float(np.abs(np.asarray(tok)[-1] - np.asarray(tok)[0]).mean()))
                    if RTC_ON:
                        rtc_first = False
                        rtc_prev_t0 = t0
                        rtc_lat.append(time.time() - t0)
                        del rtc_lat[:-20]
                        applied = getattr(link, "last_rtc", None)
                        if applied is not None:
                            counter.setdefault("rtc_applied", 0)
                            counter["rtc_applied"] += 1
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
        s_blend = p.obs()   # where the HOLD left the robot; the blend's own
        #                     displacement is chunk-0 already executing, not
        #                     drift (diag_token_hold: both tokens <= 0.016 m
        #                     over 30 s of pure hold)
        BLEND_S = 1.0
        n_bl = max(1, int(BLEND_S * RATE))
        for i in range(n_bl):
            a = (i + 1) / n_bl
            if ACTION18:
                # nothing to interpolate FROM in token space: the 18-D action
                # is a pose command, so the blend is a ramp of the wrist target
                # from where the hold left the arm to the policy's first target.
                a18_0 = np.asarray(tok0[0], dtype=float).copy()
                cur = sender18.ik.fk(sender18.q, "right")[0]
                a18_0[0:3] = (1.0 - a) * np.asarray(cur) + a * a18_0[0:3]
                a18_0[14:18] = a * a18_0[14:18]      # ease the base command in
                sender18.send(a18_0, frame_index=0)
            else:
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
            # Split accounting (2026-08-31). Reference is the HANDOVER pose
            # (grasp_gym's s_fork, post-settle); earlier runs measured from
            # the raw teleport landing and sit ~0.13 m higher -- not
            # comparable. hold_off is the only segment that can DRIFT (the
            # keepalive holding INIT_TOK; diag_token_hold measured 0.010 m
            # median over 30 s of PURE hold), so the acceptance warning
            # watches it alone -- but hold_off also contains the
            # planner->streamed mode-entry step (the reference snaps from
            # the planner stance to INIT_TOK's pose and the robot takes a
            # small physical step: probe 2026-08-31 median 0.084, max
            # 0.183, every round, both tokens). The threshold sits above
            # that floor so it is quiet when healthy and loud only when
            # something genuinely walks off. blend_off is chunk-0 already
            # executing through the 1 s blend (probe median 0.033) -- the
            # policy's own first action, recorded, not warned about.
            # tstart_spawn_off stays as the total for continuity.
            hold_off = float(np.linalg.norm(
                np.asarray(p.pelvis_xy(s_blend)) - np.asarray(spawn_xy)))
            blend_off = float(np.linalg.norm(
                np.asarray(p.pelvis_xy(s_ts)) - p.pelvis_xy(s_blend)))
            off0 = float(np.linalg.norm(
                np.asarray(p.pelvis_xy(s_ts)) - np.asarray(spawn_xy)))
            prelude["hold_off"] = round(hold_off, 4)
            prelude["blend_off"] = round(blend_off, 4)
            prelude["tstart_spawn_off"] = round(off0, 4)
            if hold_off > 0.25:
                p.log(f"[vla] WARNING hold drifted {hold_off:.2f} m off the "
                      f"handover pose (entry-step floor is ~0.18) -- "
                      f"prelude drift is back?")
        _tstart_view(p, s_ts, obj_key, prelude)
        frame_index = 0
        grip_hist = deque(maxlen=GRIP_DELAY + 1)
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

                grip_raw_j = grip_j
                grip_j = _grip_gain(grip_j)
                if GRIP_DELAY > 0:
                    grip_hist.append(grip_j)
                    grip_j = grip_hist[0]          # value from GRIP_DELAY frames ago (deque is full after warm-up)
                lhj, rhj = expand_hands(grip_j)
                if grip_j < 0.005:
                    commanded_close = True
                if ACTION18:
                    a_j = np.asarray(tok_j, dtype=float).copy()
                    a_j[6] = grip_j        # honour --grip-gain / TF_GRIP_DELAY
                    sender18.send(a_j, frame_index=frame_index)
                else:
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
                    # pelvis_z and torso_pitch, added 2026-08-30 for the
                    # deployment parameter study. Without them a trace cannot
                    # tell "raised the arm but never lowered the body" from
                    # "did the whole reach": palm_z alone is useless, because
                    # in a SUCCESS its maximum is the height the bottle gets
                    # LIFTED to (median 1.080 m) and not the grasp (0.718 m),
                    # so comparing max palm_z across outcomes just restates
                    # the outcome. The descent is the slow, sustained,
                    # position-like half of the reach and is the half the
                    # chunk-fragmentation hypothesis predicts should die.
                    # Palm ORIENTATION, and the object expressed IN the palm
                    # frame. palm_obj is a scalar distance and cannot tell a
                    # grasp pose from a palm the same distance away but
                    # rotated: measured 2026-08-30, a FAILED round reached
                    # palm_obj 0.135 m and dwelt there 5 s while two SUCCESSES
                    # only managed 0.148 and 0.137 -- so distance is provably
                    # not the discriminator and orientation is the next gross
                    # variable to rule in or out.
                    pquat = st["palms"]["right"].get("quat")
                    pquat = None if pquat is None else [float(v) for v in pquat]
                    obj_in_palm = None
                    if pquat is not None:
                        w_, x_, y_, z_ = pquat
                        R = np.array([
                            [1-2*(y_*y_+z_*z_), 2*(x_*y_-z_*w_), 2*(x_*z_+y_*w_)],
                            [2*(x_*y_+z_*w_), 1-2*(x_*x_+z_*z_), 2*(y_*z_-x_*w_)],
                            [2*(x_*z_-y_*w_), 2*(y_*z_+x_*w_), 1-2*(x_*x_+y_*y_)]])
                        obj_in_palm = [float(v) for v in R.T @ (obj - palm)]
                    pp = st.get("pelvis", {}).get("pos")
                    pq = st.get("pelvis", {}).get("quat")
                    pitch = None
                    if pq is not None:
                        w_, x_, y_, z_ = (float(v) for v in pq)
                        # pitch about the body y axis, +ve = leaning forward
                        sp = 2.0 * (w_ * y_ - z_ * x_)
                        pitch = float(np.arcsin(float(np.clip(sp, -1.0, 1.0))))
                    trace.append(dict(t=tick - t_start, rise=rise,
                                      dist=float(np.linalg.norm(obj[:2] - pel)),
                                      fallen=bool(st["fallen"]),
                                      grip=float(grip_j),
                                      grip_raw=float(grip_raw_j),
                                      pelvis_xy=[float(v) for v in pel],
                                      pelvis_yaw=float(p.pelvis_yaw(st)),
                                      pelvis_z=(None if pp is None
                                                else float(pp[2])),
                                      torso_pitch=pitch,
                                      obj_xy=[float(v) for v in obj[:2]],
                                      palm_z=float(palm[2]),
                                      palm_quat=pquat,
                                      obj_in_palm=obj_in_palm,
                                      palm_obj=float(np.linalg.norm(palm - obj)),
                                      **_obj_extra(st, obj_key)))
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
                json.dump(dict(ok=ok, why=why, **prelude, latency=lat_rec,
                               trace=trace), f)
        counter["n"] += 1
        return ok, why

    return vla_round


# =============================================================================
# Lockstep client: simulated-latency, simulation-clock evaluation
# =============================================================================
# The sim (lockstep_sim.py) advances only on request, so wall-clock time is
# no longer part of the experiment. Every quantity the async client measured
# with a clock is re-expressed in 25 fps FRAMES:
#   - K, the frames the body keeps executing the old plan while the brain
#     "thinks", is DRAWN from the empirical wall-clock distribution (or fixed
#     with --latency-frames) instead of being whatever the GPU happened to
#     deliver;
#   - chunk t0 is a frame number; the ensemble aligns by frame; RTC's
#     elapsed == frozen == K exactly.
# Same obs -> same chunk (policy.seed) -> same physics: a round replays
# bit-for-bit, and a GPU shared with three other servers changes nothing but
# the wall time.
LOCKSTEP_PORT = int(os.environ.get(
    "TF_LOCKSTEP_PORT", 5583 + 10 * int(os.environ.get("INSTANCE", "0"))))


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
                return self._msgpack.unpackb(self.sock.recv(),
                                             object_hook=self._mnp.decode, raw=False)
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
            fr.append({"token": None if tok is None else np.asarray(tok, dtype=np.float32),
                       "left": None if l is None else np.asarray(l, dtype=np.float32),
                       "right": None if r is None else np.asarray(r, dtype=np.float32)})
        rep = self._call({"cmd": "step", "frames": fr,
                          "ticks_per_frame": int(ticks_per_frame),
                          "want_obs": bool(want_obs)})
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
        from initial_pose import initial_token
        from sonic_token import GRIPPER_MAX as _G_OPEN
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
        self.session = None          # session the deploy is primed for
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
            self.cmd.token_frame(self.init_tok, right_hand=self.r_open,
                                 left_hand=self.l_open, frame_index=0)
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
    import agent_server
    C = agent_server.AgentCommander
    orig_command = C.command

    def command(self, *, start: bool, stop: bool, planner: bool) -> None:
        if planner and not stop:
            return                      # no planner under lockstep
        orig_command(self, start=start, stop=stop, planner=planner)

    def hold(self, seconds: float, hz: float = 50.0, **frame_kwargs) -> None:
        clock.hold_frames(int(round(seconds * 50.0)))

    def frame(self, *a, **kw) -> None:
        return

    C.command = command
    C.hold = hold
    C.frame = frame
    print("[lockstep] commander shim installed: planner commands dropped, "
          "hold() = N standing-token ticks")


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
        from initial_pose import initial_token
        from sonic_token import GRIPPER_MAX
        self.init_tok = initial_token()
        self.l_open, self.r_open = expand_hands(GRIPPER_MAX)
        self.cmd = None
        self.lock = threading.RLock()
        self.dt = 1.0 / DATASET_FPS
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
            self.cmd.token_frame(self.init_tok, right_hand=self.r_open,
                                 left_hand=self.l_open, frame_index=0)

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
        time.sleep(self.dt * 1.5)      # let an in-flight publish finish

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
    import agent_server
    C = agent_server.AgentCommander
    orig_init, orig_command = C.__init__, C.command
    orig_enter, orig_token = C.enter_streamed, C.token_frame

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if clock.cmd is None:
            clock.attach(self)

    def command(self, *, start: bool, stop: bool, planner: bool) -> None:
        if planner and not stop:
            return                      # never a planner command
        with clock.lock:
            orig_command(self, start=start, stop=stop, planner=planner)

    def enter_streamed(self) -> None:
        if clock._priming:
            orig_enter(self)
            return
        clock.pause()                   # the round publishes from here on

    def exit_streamed(self) -> None:
        clock.resume()                  # the round is over; keep standing

    def hold(self, seconds: float, hz: float = 50.0, **frame_kwargs) -> None:
        clock.hold(seconds)

    def frame(self, *a, **kw) -> None:
        return

    def token_frame(self, *a, **kw):
        with clock.lock:
            return orig_token(self, *a, **kw)

    C.__init__, C.command, C.enter_streamed, C.exit_streamed = __init__, command, enter_streamed, exit_streamed
    C.hold, C.frame, C.token_frame = hold, frame, token_frame
    print("[clean-deploy] commander shim installed: planner commands dropped, "
          "hold() = standing token, streamed mode kept for the process lifetime")


def load_latency_dist(path: str | None, fixed: int | None) -> dict:
    """{'first': [frames...], 'steady': [frames...]} to draw K from."""
    if fixed is not None:
        return {"first": [int(fixed)], "steady": [int(fixed)], "source": f"fixed {fixed}"}
    if not path:
        raise SystemExit("lockstep client needs --latency-frames K or "
                         "--latency-dist <json from tools/latency_dist.py>")
    d = json.load(open(path))
    assert d.get("first") and d.get("steady"), f"{path}: empty distribution"
    d["source"] = path
    return d


def make_vla_policy_lockstep(link: PolicyLink, ls: LockstepLink, clock: LockstepClock,
                             task: str, latency: dict, seed: int = 0,
                             trace_dir: str | None = None):
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
    from initial_pose import initial_token
    from sonic_token import GRIPPER_MAX as _G_OPEN

    counter = {"n": 0}
    RATE = int(round(DATASET_FPS))
    TIMEOUT_F = int(TIMEOUT_S * RATE)
    LIFT_HOLD_F = 2 * RATE                  # 2 s of sustained lift, as async
    STATUS_EVERY_F = max(1, RATE // 2)      # ~2 Hz predicates, as async

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record,
                  replace=None, spawn_xy=None):
        from primitives import LocomotionMode
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
                for _ in range(20):          # deploy state from the out port
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
                if os.environ.get("LOCKSTEP_NO_SEED", "0") != "1":
                    obs["policy.seed"] = int(rng.integers(0, 2**31 - 1))
                if RTC_ON:
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
                if ENSEMBLE_N > 1:
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
                    float(np.abs(np.asarray(tok)[-1] - np.asarray(tok)[0]).mean()))
                frames = []
                grips = []
                for i in range(K):
                    if i == A:
                        buf.append((tok, grip, t_obs))
                        del buf[:-max(ENSEMBLE_N, 1)]
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
                    del buf[:-max(ENSEMBLE_N, 1)]
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
                    trace.append(dict(t=f / RATE, frame=f, K=K, rise=rise,
                                      dist=float(np.linalg.norm(obj[:2] - pel)),
                                      fallen=bool(st["fallen"]),
                                      grip=float(g), grip_raw=float(graw),
                                      pelvis_xy=[float(v) for v in pel],
                                      pelvis_yaw=float(p.pelvis_yaw(st)),
                                      pelvis_z=float(st["pelvis"]["pos"][2]),
                                      torso_pitch=pitch,
                                      obj_xy=[float(v) for v in obj[:2]],
                                      palm_z=float(palm[2]),
                                      palm_obj=float(np.linalg.norm(palm - obj)),
                                      **_obj_extra(st, obj_key)))
                    if st["fallen"]:
                        why = "[vla] fell"
                        break
                    if rise > LIFT_M:
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
            with open(os.path.join(trace_dir, f"round_{counter['n']:03d}.json"), "w") as fh:
                json.dump(dict(ok=ok, why=why, clock="lockstep",
                               latency_source=latency.get("source"),
                               **prelude, latency=lat_rec, trace=trace), fh)
        counter["n"] += 1
        return ok, why

    return vla_round

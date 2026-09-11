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

Split across five files (refactor step 4b family 3, pure extraction;
decision 9's file-size cap): this file keeps `make_vla_policy` (the blocking
client) and every module-level knob below; `vla_obs.py` has `ObsBuilder` /
`PolicyLink` / `_tstart_view` / `_obj_extra`; `vla_client.py` has
`_grip_gain` / `_ensemble` / `_stalled` / `make_vla_policy_async` (the
official streaming client); `lockstep_client.py` has the simulation-clock
evaluation cluster (`LockstepLink`, `LockstepClock`, `install_lockstep_shim`,
`load_latency_dist`, `make_vla_policy_lockstep`); `clean_clock.py` has
`CleanClock` / `install_clean_shim`. Every name that used to live here and
moved is re-exported below, so `from locomani.backends.sonic.vla_round
import X` still works.

MONKEYPATCH INVENTORY (why this matters more here than in a typical split):
`eval_closed_loop.py` overrides nearly every constant below by assignment
before building a round (`vla_round.DATASET_FPS = cfg.backend.dataset_fps`,
`.GRIP_GAIN`, `.RTC_ON`, `.TRANSITION_COMMIT`, `.TRANSITION_DECODER`,
`.ENSEMBLE_N`, `.ENSEMBLE_M`, `.TIMEOUT_S`, `.SNAP`, `.STALL_ABORT_S`), the
RTC tools replace `vla_round.PolicyLink.get_chunk` and
`vla_round.make_vla_policy_async` wholesale, and `eval_closed_loop.py` also
constructs `vla_round.PolicyLink`/`.LockstepLink`/`.LockstepClock`/
`.CleanClock` and calls `.install_lockstep_shim`/`.install_clean_shim`/
`.load_latency_dist`/`.make_vla_policy_lockstep`/`.make_vla_policy` directly
off this module. Every one of those names is satisfied by staying defined
here (`make_vla_policy`, the constants) or by a plain top-level re-export
(everything else, since a class-attribute patch like `PolicyLink.get_chunk`
mutates the shared class object regardless of which module holds a
reference to it, and a whole-function patch like `make_vla_policy_async =
...` only has to land on the attribute this module's own callers already
read through). The one place a plain re-export is NOT enough: code that
moved to a DIFFERENT file and reads one of these constants during its own
execution (not just at the moment a caller reaches in) has to read it back
through a deferred, module-qualified `from locomani.backends.sonic import
vla_round as vr; vr.CONST` instead of an imported name -- a plain import
would bind to the value at THAT file's import time and never see a later
override. `_grip_gain`, `_ensemble`, `_stalled`, `make_vla_policy_async`,
`PolicyLink.get_chunk`, `CleanClock.__init__` and `make_vla_policy_lockstep`
all do this; see each file's own docstring. (The alias is `vr`, not
`vla_round`, in every closure-building factory below, because each builds a
nested function itself named `vla_round` -- matching this module's own
`vla_round(p, reset, meta, ...)` convention -- which would shadow a
same-named module import.)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

# NVIDIA's own latency compensation, used verbatim so this tracks their
# semantics rather than a lookalike of ours (docs/official_parity_audit.md A2).
from gear_sonic.utils.inference.vla_utils import (  # noqa: E402
    calculate_latency_compensated_index as latency_index,
)

from locomani.backends.sonic.clean_clock import CleanClock, install_clean_shim
from locomani.backends.sonic.lockstep_client import (
    LOCKSTEP_PORT,
    LockstepClock,
    LockstepLink,
    install_lockstep_shim,
    load_latency_dist,
    make_vla_policy_lockstep,
)
from locomani.backends.sonic.sonic_token import GRIPPER_MAX, expand_hands
from locomani.backends.sonic.vla_client import _ensemble, _grip_gain, _stalled, make_vla_policy_async
from locomani.backends.sonic.vla_obs import ObsBuilder, PolicyLink, _obj_extra, _tstart_view

TOKEN_DIM = 64
LIFT_M = 0.06  # the gym's own success threshold
NEAR_M = 0.60  # "got to the table" for failure classification
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
STALL_ABORT_S = 0.0  # 0 disables; 30.0 is the measured sweet spot

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
ENSEMBLE_N = 1  # chunks to blend; 1 = off, current behaviour
ENSEMBLE_M = 0.3  # exponential weight decay per chunk of age
# RTC (real-time chunking, test-time inpainting variant). When on, the async
# worker tells serve_policy how many frames of the previous chunk have elapsed
# (rtc.elapsed) and how many the NEXT inference will eat (rtc.frozen, rolling
# p90 of measured latency); the server freezes that prefix of the new chunk to
# the old chunk's overlapping rows and ramps the rest. Async client only.
RTC_ON = False
TRANSITION_COMMIT = False
TRANSITION_DECODER = None
# TF_GRIP_DELAY (frames at DATASET_FPS): send the gripper command from N frames
# ago. Diagnosis 2026-09-04 (replay_reach): the token path executes the ARM
# ~0.2 s behind the recorded motion (palm 0.5-5 cm short at the recorded close
# frame, always on the approach side), while the hand joints go straight to
# the hand PD with no lag -- so the policy closes on time onto a hand that is
# not there yet, and the jaw clips the bottle (obj_in_palm y = -4 cm in 42/44
# base100 knock-overs). Delaying only the gripper channel re-aligns the two.
# Module-level toggle (decision 8), same pattern as GRIP_GAIN/ENSEMBLE_N below:
# SonicBackendConfig.grip_delay's default; a caller assigns vla_round.GRIP_DELAY
# to override, instead of TF_GRIP_DELAY.
GRIP_DELAY = 0

STALL_ABORT_M = 0.03  # net pelvis displacement over the window, metres

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
#
# Module-level default (decision 8): SonicBackendConfig.dataset_fps; a caller
# assigns vla_round.DATASET_FPS to override, instead of TF_DATASET_FPS (must be
# a single name -- REPLAN_HZ used to be a second copy of the same value,
# snapshotted at import time, which silently went stale if only one of the two
# was reassigned; every REPLAN_HZ use below is DATASET_FPS itself now).
DATASET_FPS = 30.0

# TF_OBS_DUMP's replacement (decision 8): SonicBackendConfig.obs_dump's default is
# None (off); a caller assigns vla_round.OBS_DUMP = "<path>" for the diagnostic dump.
OBS_DUMP = None
# LOCOMANI_REPLACE's replacement (decision 8): SonicBackendConfig.replace; legacy
# sim-only re-teleport, off by default.
REPLACE = False
# LOCKSTEP_NO_SEED's replacement (decision 8): SonicBackendConfig.lockstep.no_seed;
# never overridden in any script today, so this constant IS that default.
LOCKSTEP_NO_SEED = False


def make_vla_policy(link: PolicyLink, task: str, execution_horizon: int = 16, trace_dir: str | None = None):
    """Build the callable grasp_gym.VLA_POLICY expects.

    Called with the robot settled at its spawn, facing the object, video and
    (optionally) the exporter already recording. Must return (ok, why).
    """
    counter = {"n": 0}

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record, replace=None, spawn_xy=None, round_id=None):
        from locomani.backends.sonic.primitives import LocomotionMode

        obs_src = ObsBuilder(task)
        if not obs_src.wait_ready():
            obs_src.close()
            return False, "[vla] camera/state streams never became ready"

        s0 = p.obs()
        t_entry = time.time()
        obj0 = np.asarray(p.object_pos(obj_key, s0))
        trace = []
        ok, why = False, "[vla] timeout"
        dt = 1.0 / DATASET_FPS
        commanded_close = False

        # Prime the buffer BEFORE the mode switch, then hold the first token
        # across it -- streamed mode over an empty buffer stops control
        # (g1_deploy_onnx_ref.cpp:618), and the first v4 frame is also what
        # flips is_using_encoder_ off. Same order diag_token_replay validated.
        if replace is not None and REPLACE:
            replace()  # legacy sim-only re-teleport, off by default

        def _mark(tag):  # see the async client's comment: prelude forensics
            try:
                sm = p.obs()
                p.log(f"[vla] prelude {tag}: pelvis {np.round(p.pelvis_xy(sm), 3)}")
            except Exception:
                pass

        # Entry on the STANDING token, not chunk-0 (port of the async
        # client's bbc207d fix; blocking kept the old sequence and paid for
        # it with ~1.2 s of "start walking" executed before t_start). The
        # sequential client needs NO keepalive thread: during the first
        # obs+inference below the deploy starves and clamps -- on a STANDING
        # frame, which is stable. Clamp-on-walking is what used to pitch the
        # robot over; clamp-on-standing is just standing.
        from locomani.backends.sonic.initial_pose import initial_token

        INIT_TOK = initial_token()
        l_open, r_open = expand_hands(GRIPPER_MAX)
        for _ in range(30):
            p.cmd.token_frame(INIT_TOK, right_hand=r_open, left_hand=l_open, frame_index=0)
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
        n_bl = max(1, int(1.0 * DATASET_FPS))
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
            prelude_s=round(t_start - t_entry, 2),
        )
        if spawn_xy is not None:
            # The acceptance criterion, IN the data: pelvis distance from
            # the HANDOVER pose (grasp_gym's s_fork, post-settle -- since
            # 2026-08-31; earlier runs measured from the raw teleport
            # landing, so their recorded values sit ~0.13 m higher and are
            # not comparable across the change). What this bounds is the
            # VLA prelude itself: the token hold plus the 1 s blend, during
            # which chunk-0 is already partially executing. > 0.12 m means
            # the prelude is again moving the robot before trace t=0.
            off0 = float(np.linalg.norm(np.asarray(p.pelvis_xy(s_ts)) - np.asarray(spawn_xy)))
            prelude["tstart_spawn_off"] = round(off0, 4)
            if off0 > 0.12:
                p.log(
                    f"[vla] WARNING tstart {off0:.2f} m off the handover "
                    f"pose (acceptance is 0.1) -- prelude drift is back?"
                )
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
                    p.cmd.token_frame(ftok[jj], right_hand=rhj, left_hand=lhj, frame_index=fi)
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
        cur = (tok0, grip0, 0)  # (tok, grip, next frame) of the running chunk
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
                skip = latency_index(time.time() - t_obs, DATASET_FPS, len(tok))
                n = max(1, min(execution_horizon, len(tok) - skip))
                for j in range(skip, skip + n):
                    lhj, rhj = expand_hands(grip[j])
                    if grip[j] < 0.005:
                        commanded_close = True
                    p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj, frame_index=frame_index)
                    frame_index += 1
                    time.sleep(dt)
                cur = (tok, grip, skip + n)

                st = p.obs()
                obj = np.asarray(p.object_pos(obj_key, st))
                pel = np.asarray(p.pelvis_xy(st))
                rise = float(obj[2] - obj0[2])
                dist = float(np.linalg.norm(obj[:2] - pel))
                palm = np.asarray(st["palms"]["right"]["pos"], dtype=float)
                trace.append(
                    dict(
                        t=time.time() - t_start,
                        rise=rise,
                        dist=dist,
                        fallen=bool(st["fallen"]),
                        grip=float(grip[skip : skip + n].min()),
                        palm_z=float(palm[2]),
                        palm_obj=float(np.linalg.norm(palm - obj)),
                        **_obj_extra(st, obj_key),
                        # the stall poses are the whole point of the
                        # trace: they are what a targeted recovery
                        # collection spawns the expert at
                        pelvis_xy=[float(v) for v in pel],
                        pelvis_yaw=float(p.pelvis_yaw(st)),
                        obj_xy=[float(v) for v in obj[:2]],
                    )
                )
                if st["fallen"]:
                    why = "[vla] fell"
                    break
                if rise > LIFT_M:
                    if lifted_at is None:
                        lifted_at = time.time()
                    elif time.time() - lifted_at > 2.0:  # held, not a bounce
                        ok, why = True, "ok"
                        break
                else:
                    lifted_at = None
                if _stalled(trace, commanded_close):
                    why = "[vla] reach: stalled (--stall-abort)"
                    break
        finally:
            _stop_feeder()  # an exception mid-gap must not leave a thread
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
            with open(os.path.join(trace_dir, f"round_{rid:03d}.json"), "w") as f:
                json.dump(dict(round=rid, ok=ok, why=why, **prelude, trace=trace), f)
        counter["n"] += 1
        return ok, why

    return vla_round

"""Blocking VLA round and public client imports.

Runtime options are explicit immutable VlaConfig values.
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

from locomani.backends.sonic.clean_clock import CleanClock, clean_commander_class, install_clean_shim
from locomani.backends.sonic.lockstep_client import (
    LOCKSTEP_PORT,
    LockstepClock,
    LockstepLink,
    install_lockstep_shim,
    load_latency_dist,
    lockstep_commander_class,
    make_vla_policy_lockstep,
)
from locomani.backends.sonic.sonic_token import GRIPPER_MAX, expand_hands
from locomani.backends.sonic.vla_client import _ensemble, _grip_gain, _stalled, make_vla_policy_async
from locomani.backends.sonic.vla_config import DEFAULT_VLA, LIFT_M, NEAR_M, STALL_ABORT_M, VlaConfig
from locomani.backends.sonic.vla_obs import ObsBuilder, PolicyLink, _obj_extra, _tstart_view


def make_vla_policy(
    link: PolicyLink,
    task: str,
    execution_horizon: int = 16,
    trace_dir: str | None = None,
    *,
    config: VlaConfig = DEFAULT_VLA,
):
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
        dt = 1.0 / config.backend.dataset_fps
        commanded_close = False

        # Prime the buffer BEFORE the mode switch, then hold the first token
        # across it -- streamed mode over an empty buffer stops control
        # (g1_deploy_onnx_ref.cpp:618), and the first v4 frame is also what
        # flips is_using_encoder_ off. Same order diag_token_replay validated.
        if replace is not None and config.backend.replace:
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
        n_bl = max(1, int(1.0 * config.backend.dataset_fps))
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
            while time.time() - t_start < config.budget_s:
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
                skip = latency_index(time.time() - t_obs, config.backend.dataset_fps, len(tok))
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
                if _stalled(trace, commanded_close, config=config):
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

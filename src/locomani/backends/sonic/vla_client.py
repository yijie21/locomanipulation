"""Asynchronous token streaming, ensembling and trace predicates.

Runtime options are explicit immutable VlaConfig values.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

import numpy as np

# NVIDIA's own latency compensation, used verbatim so this tracks their
# semantics rather than a lookalike of ours (docs/official_parity_audit.md A2).
from gear_sonic.utils.inference.vla_utils import (  # noqa: E402
    calculate_latency_compensated_index as latency_index,
)

from locomani.backends.sonic.sonic_token import GRIPPER_MAX, GRIPPER_MIN, expand_hands
from locomani.backends.sonic.vla_config import DEFAULT_VLA, LIFT_M, NEAR_M, STALL_ABORT_M, VlaConfig
from locomani.backends.sonic.vla_obs import ObsBuilder, PolicyLink, _obj_extra, _tstart_view


def _grip_gain(g: float, *, config: VlaConfig = DEFAULT_VLA) -> float:
    """Amplify the predicted gripper's deviation from fully-open."""

    if config.grip_gain == 1.0:
        return g
    return float(np.clip(GRIPPER_MAX - config.grip_gain * (GRIPPER_MAX - g), GRIPPER_MIN, GRIPPER_MAX))


def _ensemble(buf, now, rate, *, config: VlaConfig = DEFAULT_VLA):
    """Blend every buffered chunk's prediction of `now`. -> (tok, grip) or None.

    buf is newest-last: [(tok, grip, t0), ...]. A chunk whose aligned index
    has run past its end is exhausted and contributes nothing.
    """

    picks, ws = [], []
    for age, (tok, grip, t0) in enumerate(reversed(buf)):  # age 0 = newest
        # round, not truncate (same half-frame bias the main path had) -- but
        # NOT the official latency_index here: it CLAMPS to the last frame,
        # which would silently drag exhausted chunks back into the blend
        # instead of dropping them. Dropping is the ensemble's semantics.
        i = int(round((now - t0) * rate))
        if i < 0 or i >= len(tok):
            continue
        picks.append((tok[i], grip[i]))
        ws.append(np.exp(-config.ensemble_m * age))
    if not picks:
        return None
    w = np.asarray(ws, dtype=np.float64)
    w /= w.sum()
    tok = np.sum([p[0] * wi for p, wi in zip(picks, w)], axis=0)
    grip = float(np.sum([p[1] * wi for p, wi in zip(picks, w)]))
    return tok, grip


def _stalled(trace, commanded_close, *, config: VlaConfig = DEFAULT_VLA) -> bool:
    """True when the pelvis has gone nowhere for STALL_ABORT_S with no close.

    Reads the trace that is already being recorded rather than keeping its own
    history: one source of truth, and the offline pass that calibrated the
    thresholds ran over exactly these rows.
    """

    if config.stall_abort_s <= 0.0 or commanded_close or not trace:
        return False
    now = trace[-1]["t"]
    if now < config.stall_abort_s:
        return False
    ref = None
    for row in trace:  # oldest row still inside the window
        if now - row["t"] <= config.stall_abort_s:
            ref = row
            break
    if ref is None or now - ref["t"] < config.stall_abort_s * 0.9:
        return False
    a, b = ref["pelvis_xy"], trace[-1]["pelvis_xy"]
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 < STALL_ABORT_M


def make_vla_policy_async(
    link: PolicyLink,
    task: str,
    inference_interval: float = 0.4,
    trace_dir: str | None = None,
    *,
    config: VlaConfig = DEFAULT_VLA,
):
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
    RATE = config.backend.dataset_fps

    def vla_round(p, reset, meta, obj_key, psi, fv, grip_za, record, replace=None, spawn_xy=None, round_id=None):
        # OURS, not NVIDIA's: theirs holds the arms forward and bent with the
        # torso pitched, which is outside the distribution every episode was
        # trained in, so the handover used to lean forward and then snap
        # upright over the blend. TF_INIT_TOKEN=official restores it.
        # See initial_pose.py for the measurement and why a medoid.
        from locomani.backends.sonic.initial_pose import initial_token
        from locomani.backends.sonic.primitives import LocomotionMode

        INIT_TOK = initial_token()
        from locomani.backends.sonic.sonic_token import GRIPPER_MAX as _G_OPEN

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
            p.cmd.token_frame(INIT_TOK, right_hand=r_open, left_hand=l_open, frame_index=0)
            time.sleep(0.02)
        p.cmd.enter_streamed()
        t_hold = time.time()
        keep_stop = threading.Event()

        def _keepalive():
            while not keep_stop.is_set():
                p.cmd.token_frame(INIT_TOK, right_hand=r_open, left_hand=l_open, frame_index=0)
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
        latest = {"chunk": None, "t0": None, "capture_chunk_id": None}
        buf = []  # [(tok, grip, t0)], newest last
        lock = threading.Lock()
        stop = threading.Event()
        # Empirical latency record for the lockstep client (tools/
        # latency_dist.py): per-inference obs->chunk seconds and the spacing
        # between consecutive obs (the K the sim advances per cycle).
        lat_rec = {"infer_s": [], "cycle_s": []}
        commitment = None
        if config.transition_commit:
            from locomani.backends.sonic.transition_execution import TransitionCommitment

            commitment = TransitionCommitment(RATE, config.transition_decoder)
            lat_rec["transition_commit_events"] = commitment.events
            lat_rec["transition_commit_errors"] = commitment.errors

        def worker():
            rtc_prev_t0 = None  # obs time of the last SUCCESSFUL inference
            rtc_lat = []  # last 20 measured obs->chunk latencies (s)
            rtc_first = True
            while not stop.is_set():
                t0 = time.time()
                obs = obs_src.build()
                if obs is not None:
                    if config.rtc:
                        if rtc_first:
                            obs["rtc.reset"] = 1
                        # elapsed 0 on the first request: no injection server-
                        # side, but the meta's presence makes it CACHE the
                        # chunk, so request 2 has a prefix to freeze.
                        e = 0 if rtc_prev_t0 is None else int(round((t0 - rtc_prev_t0) * RATE))
                        lat = sorted(rtc_lat)
                        p90 = lat[int(0.9 * (len(lat) - 1))] if lat else 0.5
                        obs["rtc.elapsed"] = e
                        obs["rtc.frozen"] = max(1, min(20, int(np.ceil(p90 * RATE)) + 1))
                        if commitment is not None:
                            obs["rtc.frozen"] = commitment.window.frozen(t0, obs["rtc.frozen"])
                            obs["rtc.commit"] = int(t0 < commitment.window.until)
                    try:
                        tok, grip = link.get_chunk(obs)
                    except Exception:
                        time.sleep(0.1)
                        continue
                    lat_rec["infer_s"].append(time.time() - t0)
                    if commitment is not None:
                        try:
                            commitment.consider(t0, tok)
                        except Exception as exc:
                            commitment.errors.append(repr(exc))
                    if latest["t0"] is not None:
                        lat_rec["cycle_s"].append(t0 - latest["t0"])
                    lat_rec.setdefault("chunk_grip_min", []).append(float(np.min(grip)))
                    lat_rec.setdefault("chunk_tok_span", []).append(
                        float(np.abs(np.asarray(tok)[-1] - np.asarray(tok)[0]).mean())
                    )
                    if config.rtc:
                        rtc_first = False
                        rtc_prev_t0 = t0
                        rtc_lat.append(time.time() - t0)
                        del rtc_lat[:-20]
                        applied = getattr(link, "last_rtc", None)
                        lat_rec.setdefault("rtc_requests", []).append(
                            dict(
                                wall=t0,
                                elapsed=e,
                                frozen=int(obs["rtc.frozen"]),
                                commit=bool(obs.get("rtc.commit", False)),
                                applied=None if applied is None else np.asarray(applied).tolist(),
                            )
                        )
                        if applied is not None:
                            counter.setdefault("rtc_applied", 0)
                            counter["rtc_applied"] += 1
                    with lock:
                        latest["chunk"] = (tok, grip)
                        latest["capture_chunk_id"] = getattr(getattr(link, "capture_context", None), "chunk_id", None)
                        latest["t0"] = t0
                        if config.ensemble_n > 1:
                            buf.append((tok, grip, t0))
                            del buf[: -config.ensemble_n]
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
            if commitment is not None:
                commitment.close()
            obs_src.close()
            return False, (
                "[vla] initial-pose hold: no fresh chunk within 15 s -- policy worker stalled; round aborted"
            )

        if replace is not None and config.backend.replace:
            replace()  # legacy sim-only re-teleport, off by default

        with lock:
            tok0, grip0 = latest["chunk"]
        l0, r0 = expand_hands(grip0[0])

        # Hand the commander socket back to this thread, then BLEND from the
        # standing token into the first chunk frame (1 s linear, the
        # official client's default) so the handover has no step in it.
        _teardown()
        s_blend = p.obs()  # where the HOLD left the robot; the blend's own
        #                     displacement is chunk-0 already executing, not
        #                     drift (diag_token_hold: both tokens <= 0.016 m
        #                     over 30 s of pure hold)
        BLEND_S = 1.0
        n_bl = max(1, int(BLEND_S * RATE))
        for i in range(n_bl):
            a = (i + 1) / n_bl
            tk = (1.0 - a) * INIT_TOK + a * np.asarray(tok0[0])
            p.cmd.token_frame(tk, right_hand=r_open, left_hand=l_open, frame_index=0)
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
                p.log(f"[vla] prelude {tag}: pelvis {np.round(p.pelvis_xy(sm), 3)} grip0 {float(grip0[0]):+.4f}")
            except Exception as e:  # a missing mark must LOOK missing:
                p.log(f"[vla] prelude {tag}: MARK FAILED ({e}) -- do not read this segment as drift-free")

        _mark("streamed")

        t_start = time.time()
        s_ts = p.obs()
        # See the blocking client's comment: t=0 of the trace is not the
        # settled spawn in ~2/3 of rounds; these fields measure the gap.
        prelude = dict(
            entry_pelvis=[round(float(v), 4) for v in p.pelvis_xy(s0)],
            tstart_pelvis=[round(float(v), 4) for v in p.pelvis_xy(s_ts)],
            prelude_s=round(t_start - t_entry, 2),
        )
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
            hold_off = float(np.linalg.norm(np.asarray(p.pelvis_xy(s_blend)) - np.asarray(spawn_xy)))
            blend_off = float(np.linalg.norm(np.asarray(p.pelvis_xy(s_ts)) - p.pelvis_xy(s_blend)))
            off0 = float(np.linalg.norm(np.asarray(p.pelvis_xy(s_ts)) - np.asarray(spawn_xy)))
            prelude["hold_off"] = round(hold_off, 4)
            prelude["blend_off"] = round(blend_off, 4)
            prelude["tstart_spawn_off"] = round(off0, 4)
            if hold_off > 0.25:
                p.log(
                    f"[vla] WARNING hold drifted {hold_off:.2f} m off the "
                    f"handover pose (entry-step floor is ~0.18) -- "
                    f"prelude drift is back?"
                )
        _tstart_view(p, s_ts, obj_key, prelude)
        frame_index = 0
        grip_hist = deque(maxlen=config.backend.grip_delay + 1)
        lifted_at = None
        chunk_t0, idx = None, 0
        capture_chunk_id = None
        tok, grip = tok0, grip0
        last_status = 0.0
        try:
            while time.time() - t_start < config.budget_s:
                tick = time.time()
                trace_source = dict(kind="ensemble", chunk_id=None, action_index=None)
                if config.ensemble_n > 1:
                    # Blend every overlapping prediction of THIS instant.
                    # Indexing is by absolute time, which also removes the
                    # drift the incrementing idx below accumulates when the
                    # loop does not land exactly on 1/RATE.
                    with lock:
                        blended = _ensemble(buf, tick, RATE, config=config)
                        if blended is None and latest["chunk"] is not None:
                            blended = (latest["chunk"][0][-1], float(latest["chunk"][1][-1]))
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
                            capture_chunk_id = latest["capture_chunk_id"]
                            idx = latency_index(time.time() - chunk_t0, RATE, len(tok))
                    j = min(idx, len(tok) - 1)
                    tok_j, grip_j = tok[j], float(grip[j])
                    trace_source = dict(
                        kind="chunk",
                        chunk_id=capture_chunk_id,
                        action_index=j,
                        observation_wall=chunk_t0,
                        grip_delay_frames=config.backend.grip_delay,
                    )
                    idx = min(idx + 1, len(tok) - 1)

                grip_raw_j = grip_j
                grip_j = _grip_gain(grip_j, config=config)
                if config.backend.grip_delay > 0:
                    grip_hist.append(grip_j)
                    grip_j = grip_hist[0]  # value from GRIP_DELAY frames ago (deque is full after warm-up)
                lhj, rhj = expand_hands(grip_j)
                if grip_j < 0.005:
                    commanded_close = True
                p.cmd.token_frame(
                    tok_j, right_hand=rhj, left_hand=lhj, frame_index=frame_index, trace_source=trace_source
                )
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
                        R = np.array(
                            [
                                [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                                [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                                [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)],
                            ]
                        )
                        obj_in_palm = [float(v) for v in R.T @ (obj - palm)]
                    pp = st.get("pelvis", {}).get("pos")
                    pq = st.get("pelvis", {}).get("quat")
                    pitch = None
                    if pq is not None:
                        w_, x_, y_, z_ = (float(v) for v in pq)
                        # pitch about the body y axis, +ve = leaning forward
                        sp = 2.0 * (w_ * y_ - z_ * x_)
                        pitch = float(np.arcsin(float(np.clip(sp, -1.0, 1.0))))
                    trace.append(
                        dict(
                            t=tick - t_start,
                            rise=rise,
                            dist=float(np.linalg.norm(obj[:2] - pel)),
                            fallen=bool(st["fallen"]),
                            grip=float(grip_j),
                            grip_raw=float(grip_raw_j),
                            pelvis_xy=[float(v) for v in pel],
                            pelvis_yaw=float(p.pelvis_yaw(st)),
                            pelvis_z=(None if pp is None else float(pp[2])),
                            torso_pitch=pitch,
                            obj_xy=[float(v) for v in obj[:2]],
                            palm_z=float(palm[2]),
                            palm_quat=pquat,
                            obj_in_palm=obj_in_palm,
                            palm_obj=float(np.linalg.norm(palm - obj)),
                            **_obj_extra(st, obj_key),
                        )
                    )
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
                    if _stalled(trace, commanded_close, config=config):
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
            if commitment is not None:
                commitment.close()
            obs_src.close()

        if not ok and why == "[vla] timeout" and trace:
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
                json.dump(dict(round=rid, ok=ok, why=why, **prelude, latency=lat_rec, trace=trace), f)
        counter["n"] += 1
        return ok, why

    return vla_round

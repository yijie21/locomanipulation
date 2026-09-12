"""Small crossed-prefix pilot, preserving the measured handover posture.

ee/ev use the same scripted prefix and recorded creep termination. ve/vv use
the same clean-token prefix with an encoder-loaded deploy. ve restores the
planner interface only at handover. Every interface change is logged; these
are feasibility pilots, not interchangeable benchmark success rates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from locomani import paths

sys.path.insert(0, paths.SONIC)
from locomani.backends.sonic import agent_server
from locomani.backends.sonic import vla_round as vr
from locomani.rtc.transition_capture import Capture
from locomani.tasks.bottle import execution as sr
from locomani.tasks.bottle import grasp_gym as gym


class Finished(Exception):
    def __init__(self, result):
        self.result = result


def suffix(
    p,
    cap,
    obj_key,
    port,
    task,
    budget=60,
    takeover=False,
    takeover_crouched=False,
    policy_seed=None,
    takeover_wait=10.0,
    prepare_observation=None,
    rtc=False,
):
    """Same 25Hz, 0.4s async/latency-index rule; no standing-token prelude."""
    s0 = p.obs()
    z0 = s0["objects"][obj_key]["pos"][2]
    debug = cap.latest_debug
    if debug is None or "token_state" not in debug:
        raise RuntimeError("No measured token at handover")
    tok = np.asarray(debug["token_state"]).copy()
    grip = float(p.cmd.right_hand[0])
    latest = {}
    lock = threading.Lock()
    stop = threading.Event()
    errors = []
    cap.event("vla_suffix_enter", state=s0, init_token=tok, interface="external_token", budget=budget)
    # Prime from the current deploy token, not the standing initialization.
    lh, rh = vr.expand_hands(grip)
    p.cmd.token_frame(tok, left_hand=lh, right_hand=rh, frame_index=0)
    p.cmd.enter_streamed()

    def worker():
        obs = vr.ObsBuilder(task)
        link = vr.PolicyLink(port=port)
        inference_index = 0
        previous_observation = None
        latencies = []
        try:
            if not obs.wait_ready():
                raise RuntimeError("No policy observation")
            if prepare_observation is not None:
                prepare_observation(obs)
            while not stop.is_set():
                started = time.time()
                x = obs.build()
                if x is not None:
                    if policy_seed is not None:
                        x["policy.seed"] = int(policy_seed + inference_index)
                    if rtc:
                        if inference_index == 0:
                            x["rtc.reset"] = 1
                        x["rtc.elapsed"] = (
                            0 if previous_observation is None else int(round((started - previous_observation) * 25))
                        )
                        ordered = sorted(latencies)
                        p90 = ordered[int(0.9 * (len(ordered) - 1))] if ordered else 0.5
                        x["rtc.frozen"] = max(1, min(20, int(np.ceil(p90 * 25)) + 1))
                    if inference_index == 0:
                        cap.event("first_policy_observation", state=cap.latest_priv)
                    inference_index += 1
                    t, g = link.get_chunk(x)
                    previous_observation = started
                    latencies = (latencies + [time.time() - started])[-20:]
                    with lock:
                        latest.update(t0=started, token=t, grip=g)
                stop.wait(max(0, 0.4 - (time.time() - started)))
        except Exception as exc:
            errors.append(repr(exc))
        finally:
            obs.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    start = time.time()
    ready_since = None
    lifted = None
    counter = 0
    last_state = 0
    result = (False, "vla timeout")
    reason = "budget"
    try:
        while time.time() - start < budget:
            now = time.time()
            if errors:
                raise RuntimeError(errors)
            with lock:
                if latest:
                    j = vr.latency_index(now - latest["t0"], 25, len(latest["token"]))
                    tok = latest["token"][j]
                    grip = float(latest["grip"][j])
            lh, rh = vr.expand_hands(grip)
            p.cmd.token_frame(tok, left_hand=lh, right_hand=rh, frame_index=counter)
            counter += 1
            if now - last_state > 0.1:
                st = p.obs()
                last_state = now
                obj = np.array(st["objects"][obj_key]["pos"])
                dist = np.linalg.norm(obj[:2] - p.pelvis_xy(st))
                if st["fallen"]:
                    result = (False, "fell")
                    reason = "fell"
                    break
                if obj[2] - z0 > 0.06:
                    if lifted is None:
                        lifted = now
                    if now - lifted > 2:
                        result = (True, "lifted")
                        reason = "success"
                        break
                else:
                    lifted = None
                if takeover:
                    # Intervention trigger, NOT success criterion: arrived,
                    # upright, no close for 10s. Log the actual geometry.
                    height = st["pelvis"]["pos"][2]
                    ready = (
                        dist < (0.55 if takeover_crouched else 0.45)
                        and grip >= 0.005
                        and (height < 0.73 if takeover_crouched else height > 0.73)
                        and (not takeover_crouched or not sr.toppled(st, obj_key))
                    )
                    if not ready:
                        ready_since = None
                    elif ready_since is None:
                        ready_since = now
                    elif now - ready_since > takeover_wait:
                        result = (False, "handover")
                        reason = "handover"
                        break
            time.sleep(max(0, 0.04 - (time.time() - now)))
    finally:
        stop.set()
        # Keep the last action alive while an in-flight inference returns.
        while thread.is_alive():
            thread.join(timeout=0.04)
            p.cmd.token_frame(tok, left_hand=lh, right_hand=rh, frame_index=counter)
            counter += 1
        cap.event("vla_suffix_exit", reason=reason, result=result, state=p.obs())
    return result


def expert_grasp(p, reset, meta, obj_key, cap):
    """Production grasp servo, replanning against the actual handover state."""
    from collect_episode import obj_top

    from locomani.backends.sonic.primitives import Primitives
    from locomani.tasks.bottle import adaptive_grasp, grasping, reach_map

    surface = meta["surfaces"][meta["objects"][obj_key].get("surface", "A")]

    def make(wrist, rot):
        st = p.obs()
        grip = np.array(st["objects"][obj_key]["pos"])
        grip[2] = obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM
        yaw = p.pelvis_yaw(st)
        pel = np.array(st["pelvis"]["pos"])
        anchor = pel + np.array([0.31 * np.sin(yaw), -0.31 * np.cos(yaw), 0.44])
        d = grip[:2] - p.pelvis_xy(st)
        d = d / max(np.linalg.norm(d), 1e-6)
        return adaptive_grasp.plan(
            grip,
            wrist,
            rot,
            surface["top"],
            box=meta["objects"][obj_key]["box"],
            obj_rot=Primitives._rot(st["objects"][obj_key]["quat"]),
            azimuth_from=anchor,
            avoid_dir=np.r_[d, 0],
            avoid_sign=1.0,
        )

    st = p.obs()
    grip = np.array(st["objects"][obj_key]["pos"])
    grip[2] = obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM
    plan = make(st["palms"]["right"]["pos"], Primitives._rot(st["palms"]["right"]["quat"]))
    if not plan.ok:
        return False, plan.why
    p.hold_yaw = p.pelvis_yaw(st)
    cap.event("expert_suffix_enter", state=st, plan=str(plan))
    result = sr.grasp_streamed(
        p,
        obj_key,
        grip,
        plan,
        side="right",
        surface_top=surface["top"],
        replan=make,
        nudge=False,
        envelope=(reach_map.ReachMap(), 0.02, np.deg2rad(36.5)),
    )
    cap.event("expert_suffix_exit", state=p.obs(), result=result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ee", "ev", "ve", "vv"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--port", type=int, default=6590)
    ap.add_argument("--record", action="store_true", help="EE only: save successful continuous expert trajectories")
    ap.add_argument("--until-ok", type=int, default=None)
    a = ap.parse_args()
    root = Path(a.out).resolve()
    if a.record and a.arm != "ee":
        ap.error("--record is only supported for the expert EE arm")
    if root.exists():
        raise SystemExit(f"Refusing overwrite: {root}")
    cap = Capture(root)
    cap.instrument()
    np.random.seed(a.seed)
    cap.event(
        "cross_start",
        arm=a.arm,
        seed=a.seed,
        creep_obs=os.environ.get("GYM_CREEP_OBS", "1"),
        caveat="Interface differs at E/V switch; inspect handover drift and command tracking.",
    )
    original_round = gym._run_round

    def run(*args, **kwargs):
        cap.event("round_start", arm=a.arm)
        try:
            result = original_round(*args, **kwargs)
        except Finished as done:
            result = done.result
        cap.event("round_end", result=result)
        return result

    gym._run_round = run
    original_creep = sr._creep_to_band

    def creep(p, *args, **kw):
        cap.event("creep_enter", state=p.obs())
        r = original_creep(p, *args, **kw)
        cap.event("creep_exit", state=p.obs(), gx=r)
        if a.arm == "ev" and r.status == sr.ApproachStatus.READY:
            raise Finished(suffix(p, cap, "water_bottle", a.port, "pick up the water bottle"))
        return r

    sr._creep_to_band = creep
    original_squat = sr.PlannerArm.squat

    def squat(arm, *args, **kw):
        cap.event("squat_enter", state=arm.p.obs(), drop=args[0] if args else kw.get("drop_m"))
        result = original_squat(arm, *args, **kw)
        cap.event("squat_exit", state=arm.p.obs(), delivered=result)
        return result

    sr.PlannerArm.squat = squat
    clock = None
    runtime = gym.GymRuntime()
    if a.arm in ["ve", "vv"]:
        C = agent_server.AgentCommander
        names = ["__init__", "command", "enter_streamed", "exit_streamed", "hold", "frame", "token_frame"]
        originals = {k: getattr(C, k) for k in names}
        clock = vr.CleanClock()
        vr.install_clean_shim(clock)
        old_fresh = gym.fresh_deploy

        def fresh(p):
            # Restore clean wrappers for subsequent rounds after expert mode.
            for k, v in clean_methods.items():
                setattr(C, k, v)
            clock.pause()
            clock.primed = False
            result = old_fresh(p)
            if result:
                clock.prime()
                clock.resume()
            return result

        gym.fresh_deploy = fresh
        old_init = gym.Primitives.__init__

        def init(p, *args, **kw):
            old_init(p, *args, **kw)
            clock.prime()
            clock.resume()

        gym.Primitives.__init__ = init
        clean_methods = {k: getattr(C, k) for k in names}

        def policy(p, reset, meta, obj_key, **kw):
            result = suffix(p, cap, obj_key, a.port, "pick up the water bottle", budget=60, takeover=a.arm == "ve")
            if result[1] == "handover":
                clock.pause()
                cap.event("token_to_planner", state=p.obs())
                for k, v in originals.items():
                    setattr(C, k, v)
                p.cmd.exit_streamed()
                # The mode-change safety reset consumes the first start
                # command in a clean auto-started deploy. Send start again
                # after that reset, so handlePlannerInput can initialize the
                # planner and release the previous external token.
                time.sleep(0.2)
                p.cmd.exit_streamed()
                cap.event("planner_start_reasserted", state=p.obs())
                p.cmd.heading_offset = p.pelvis_yaw(p.obs())
                # Every planner reference seeds from the measured current pose.
                from locomani.backends.sonic.primitives import upper_body_from_mj

                p.cmd.set_upper_body(upper_body_from_mj(p.obs()["joints"]))
                p.cmd.set_upper_body_vel(np.zeros(17))
                p.cmd.stream_mode = agent_server.StreamMode.PLANNER_FROZEN_UPPER_BODY
                result = expert_grasp(p, reset, meta, obj_key, cap)
            return result

        runtime = gym.GymRuntime(policy=policy)
    sys.argv = [
        "grasp_gym.py",
        "--rounds",
        str(a.rounds),
        "--seed",
        str(a.seed),
        "--targets",
        "water_bottle",
        "--edge-jitter",
        ".20",
        "--spawn-edge",
        "1.1",
        "1.3",
        "--face-object",
        "--spawn-yaw-err",
        "20",
        "--spawn-lateral",
        ".20",
        "--one-shot",
        "--mid-band",
        ".20",
        "--manifest",
        str(root / "manifest.jsonl"),
    ]
    if a.record:
        sys.argv.append("--record")
    if a.until_ok is not None:
        sys.argv.extend(["--until-ok", str(a.until_ok)])
    try:
        gym.main(runtime=runtime)
    finally:
        if clock:
            clock.stop()
        cap.event("cross_end")
        cap.close()


if __name__ == "__main__":
    main()

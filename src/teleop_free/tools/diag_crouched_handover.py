"""Aligned-start VLA / in-place expert experiment, not a benchmark.

Only align robot XY before the policy starts observing. Never teleport,
walk, or repeat the squat at the expert handover. Record channel-switch
drift separately; a failed handover is not proof of unreachable geometry.
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

sys.path.insert(0, paths.SONIC)
import agent_server
import grasp_gym as gym
import stream_reach as sr
import vla_round as vr
from diag_transition_cross import suffix
from transition_capture import Capture


def in_place_expert(
    p,
    meta,
    key,
    cap,
    originals,
    clock,
    transport_strategy="baseline",
    transport_feedback=False,
    transport_pose_feedback=False,
    planner_active=False,
    active_arm=None,
):
    """Run the expert; an active_arm is borrowed and remains caller-owned.

    Borrowing keeps its reference, facing, height and control channel. The
    caller must first verify crouched stability and must always clean it up.
    """
    import adaptive_grasp
    import grasping
    from collect_episode import obj_top
    from primitives import Primitives, upper_body_from_mj

    owns_arm = active_arm is None
    if not owns_arm and (
        not planner_active
        or active_arm.p is not p
        or active_arm.side != "right"
        or active_arm.mode != agent_server.LocomotionMode.IDEL_SQUAT
    ):
        raise ValueError("active_arm must be this robot's crouched right Planner controller")
    before = p.obs()
    z = float(before["pelvis"]["pos"][2])
    cap.event(
        "handover_before",
        state=before,
        planner_active=planner_active,
        heading_offset=p.cmd.heading_offset,
        reused_arm=not owns_arm,
    )
    if sr.toppled(before, key):
        return False, "object already toppled; not an intact-grasp handover"
    clock.pause()
    if owns_arm:
        for name, method in originals.items():
            setattr(agent_server.AgentCommander, name, method)
        if not planner_active:
            p.cmd.heading_offset = p.pelvis_yaw(before)
        p.cmd.set_upper_body(upper_body_from_mj(before["joints"]))
        p.cmd.set_upper_body_vel(np.zeros(17))
        p.cmd.stream_mode = agent_server.StreamMode.PLANNER_FROZEN_UPPER_BODY
        if not planner_active:
            p.cmd.exit_streamed()
            time.sleep(0.2)
            p.cmd.exit_streamed()
        arm = sr.PlannerArm(p, "right", send_vel=True)
        arm.mode = agent_server.LocomotionMode.IDEL_SQUAT
        arm.height = float(np.clip(z - sr.SQUAT_HEIGHT_BIAS, 0.40, 0.78))
    else:
        arm = active_arm
    arm.guard = key
    surface = meta["surfaces"][meta["objects"][key].get("surface", "A")]

    def plan(wrist, rot, observed=None):
        st = p.obs() if observed is None else observed
        target = np.array(st["objects"][key]["pos"])
        target[2] = obj_top(meta, key) - grasping.GRIP_BELOW_RIM
        yaw = p.pelvis_yaw(st)
        pel = np.array(st["pelvis"]["pos"])
        anchor = pel + np.array([0.31 * np.sin(yaw), -0.31 * np.cos(yaw), 0.44])
        d = target[:2] - p.pelvis_xy(st)
        d /= max(np.linalg.norm(d), 1e-6)
        return adaptive_grasp.plan(
            target,
            wrist,
            rot,
            surface["top"],
            box=meta["objects"][key]["box"],
            obj_rot=Primitives._rot(st["objects"][key]["quat"]),
            azimuth_from=anchor,
            avoid_dir=np.r_[d, 0],
            avoid_sign=1.0,
        )

    try:
        if owns_arm:
            arm.start()
            cap.event("handover_transient", state=p.obs())
            # Same height request throughout: let the mode-change transient
            # settle instead of mistaking a 0.6 s intermediate pose for delivery.
            arm.hold(sr.SQUAT_SETTLE)
        st = p.obs()
        drift = float(np.linalg.norm(np.array(st["pelvis"]["pos"][:2]) - before["pelvis"]["pos"][:2]))
        dz = float(st["pelvis"]["pos"][2] - z)
        cap.event(
            "handover_after",
            state=st,
            xy_drift=drift,
            z_drift=dz,
            height_command=arm.height,
            geometry_preserved=drift <= 0.03 and abs(dz) <= 0.03,
        )
        if st["fallen"]:
            return False, "expert handover fell"
        if sr.toppled(st, key):
            return False, "object toppled during handover"
        candidate = plan(st["palms"]["right"]["pos"], Primitives._rot(st["palms"]["right"]["quat"]), observed=st)
        cap.event("expert_plan", ok=candidate.ok, why=candidate.why, plan=str(candidate), state=st)
        if not candidate.ok:
            return False, candidate.why
        from execution_record import ExecutionRecord
        from transport_guard import TransportGuard

        execution = ExecutionRecord()
        guard = None
        if transport_strategy != "baseline":
            guard = TransportGuard(
                arm.ik, key, meta["objects"][key]["body"], [s["body"] for s in meta["surfaces"].values()]
            )
        cap.event("transport_start", strategy=transport_strategy, state=p.obs())
        if transport_strategy == "planned":
            from diagnostic_transport import execute_transport

            moved, reason = execute_transport(
                p,
                arm,
                candidate,
                guard,
                cap,
                key,
                feedback=transport_feedback,
                pose_feedback=transport_pose_feedback,
                preserve_reference=not owns_arm,
            )
            if not moved:
                cap.event("expert_servo_exit", result=(False, reason), state=p.obs(), execution=execution.snapshot())
                return False, reason
        result = sr._servo_continuous(
            p,
            arm,
            key,
            candidate,
            "right",
            replan=None if transport_strategy == "planned" else plan,
            surface_top=surface["top"],
            execution=execution,
            transport_guard=guard,
            start_at_pregrasp=transport_strategy == "planned",
        )
        cap.event("expert_servo_exit", result=result, state=p.obs(), execution=execution.snapshot())
        if result[0]:
            initial_offset = None
            max_change = 0.0
            for _ in range(100):
                arm._push(arm.q)
                st = p.obs()
                off = np.array(st["objects"][key]["pos"]) - st["palms"]["right"]["pos"]
                if initial_offset is None:
                    initial_offset = off.copy()
                max_change = max(max_change, float(np.linalg.norm(off - initial_offset)))
            held = max_change < 0.04 and st["objects"][key]["pos"][2] > before["objects"][key]["pos"][2] + 0.06
            cap.event("expert_hold_exit", held=held, max_offset_change=max_change, state=st)
            return held, "expert lifted and held" if held else "expert slipped during hold"
        return result
    finally:
        if owns_arm:
            arm.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["vla", "expert"], required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=6200)
    ap.add_argument("--port", type=int, default=6590)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--rtc", action="store_true", help="Diagnostic RTC prefix condition")
    ap.add_argument("--takeover-wait", type=float, default=25.0)
    ap.add_argument("--transport-strategy", choices=["baseline", "guard", "planned"], default="baseline")
    ap.add_argument(
        "--transport-feedback",
        action="store_true",
        help="Diagnostic bounded joint-error compensation; planned strategy only",
    )
    ap.add_argument(
        "--transport-pose-feedback",
        action="store_true",
        help="Diagnostic measured palm-pose feedback after planned transport",
    )
    a = ap.parse_args()
    if a.transport_pose_feedback and (a.transport_strategy != "planned" or a.transport_feedback):
        ap.error("--transport-pose-feedback requires planned strategy without joint feedback")
    if a.transport_feedback and a.transport_strategy != "planned":
        ap.error("--transport-feedback requires --transport-strategy planned")
    if a.out.exists():
        ap.error("refusing overwrite")
    cap = Capture(a.out.resolve())
    cap.instrument()
    np.random.seed(a.seed)
    cap.event(
        "experiment",
        mode=a.mode,
        seed=a.seed,
        aligned=not a.no_align,
        rtc=a.rtc,
        takeover_wait=a.takeover_wait,
        transport_strategy=a.transport_strategy,
        transport_feedback=a.transport_feedback,
        transport_pose_feedback=a.transport_pose_feedback,
        note="encoder loaded; paired diagnostic, not old benchmark configuration",
    )
    C = agent_server.AgentCommander
    names = ["__init__", "command", "enter_streamed", "exit_streamed", "hold", "frame", "token_frame"]
    originals = {k: getattr(C, k) for k in names}
    clock = vr.CleanClock()
    vr.install_clean_shim(clock)
    clean = {k: getattr(C, k) for k in names}
    old_init = gym.Primitives.__init__

    def init(p, *args, **kw):
        old_init(p, *args, **kw)
        clock.prime()
        clock.resume()

    gym.Primitives.__init__ = init
    old_fresh = gym.fresh_deploy

    def fresh(p):
        for k, v in clean.items():
            setattr(C, k, v)
        clock.pause()
        clock.primed = False
        result = old_fresh(p)
        if result:
            clock.prime()
            clock.resume()
        return result

    gym.fresh_deploy = fresh
    original_round = gym._run_round
    current = {}
    sig = inspect.signature(original_round)

    def run(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        d = bound.arguments
        fv = np.array([np.cos(d["psi"]), np.sin(d["psi"])])
        current["xy"] = np.asarray(d["spawn_xy"]) - d["spawn_off"] * fv + d["spawn_lat"] * np.array([-fv[1], fv[0]])
        current["round"] = current.get("round", -1) + 1
        cap.event("round_start", round=current["round"], target_xy=current["xy"])
        result = original_round(*args, **kwargs)
        cap.event("round_end", result=result)
        return result

    gym._run_round = run

    def policy(p, reset, meta, obj_key, **kwargs):
        def prepare(obs):
            # Own the ZMQ control socket in this worker. Align only after
            # camera/state initialization, immediately before observation.
            from privileged_pub import EpisodeControlClient

            ctl = EpisodeControlClient()
            before = cap.latest_priv
            ctl.align_robot_xy(current["xy"])
            deadline = time.time() + 2
            while time.time() < deadline:
                st = cap.latest_priv
                if st is not None and np.linalg.norm(np.array(st["pelvis"]["pos"][:2]) - current["xy"]) < 0.005:
                    break
                time.sleep(0.01)
            else:
                ctl.close()
                raise RuntimeError("initial placement did not arrive")
            time.sleep(0.12)  # camera cycles after placement, before build()
            obs._poll()
            st = cap.latest_priv
            err = float(np.linalg.norm(np.array(st["pelvis"]["pos"][:2]) - current["xy"]))
            ctl.close()
            cap.event("initial_alignment", before=before, after=st, xy_error=err)
            if err > 0.02:
                raise RuntimeError("initial observation outside 2 cm tolerance")

        result = suffix(
            p,
            cap,
            obj_key,
            a.port,
            "pick up the water bottle",
            budget=60,
            takeover=a.mode == "expert",
            takeover_crouched=True,
            takeover_wait=a.takeover_wait,
            prepare_observation=None if a.no_align else prepare,
            rtc=a.rtc,
            policy_seed=a.seed * 10000 + current["round"] * 1000,
        )
        if result[1] == "handover":
            return in_place_expert(
                p,
                meta,
                obj_key,
                cap,
                originals,
                clock,
                a.transport_strategy,
                a.transport_feedback,
                a.transport_pose_feedback,
            )
        return result

    gym.VLA_POLICY = policy
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
        str(a.out / "manifest.jsonl"),
    ]
    try:
        gym.main()
    finally:
        clock.stop()
        cap.event("experiment_end")
        cap.close()


if __name__ == "__main__":
    main()

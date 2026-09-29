#!/usr/bin/env python3
"""Run a primitive plan against a live stack, one episode per round (design §7; TODO 7.6).

    $VENV_SIM_PY scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml \\
        --gpu 1 --instance 1 --rounds 3 --name skills_smoke_bottle_20260928 [--plan plan.json | --agent]

The experiment file names the scene, the backend and the task (target, goal); the plan is a JSON list of
`{"primitive": name, "args": {...}}`. Without `--plan` the runner derives the obvious one from the task:
`approach → grasp_single [→ approach(goal) → place]` for a single-hand target, `approach → ready →
grasp_bimanual` for a bimanual one. With `--agent` there is no fixed plan: `skills.planner.ClaudePlanner`
(`claude -p`, JSON in and out, no tools) names one primitive at a time and reads the executor's entry back,
up to `--max-steps` calls; every ask is logged to `agent_io_NN.jsonl`. Each round: teleport reset (the
round loop's spawn, without its sampling jitter), the calls through `skills.Executor`, the task criterion
from the privileged state, one manifest line and one `agent_trace_NN.jsonl`, one video. The stack is this
process's (instance/GPU from the arguments), brought down at the end unless `--keep-stack`. No exporter,
no recording: this is the regression / smoke entry. `--profile` overrides the profile the scene picks for
the target (a new object's calibration, docs/skills.md). `--inject` names a JSON failure injection
(`skills/injection.py`): a world event fired once per round after the first successful call of a named
primitive, identical for the fixed plan and the agent on the same seed (agentic-plan Q11; docs/skills.md
§7); `scripts/injection_report.py` pairs the two runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument(
        "--instance", type=int, default=1, help="stack instance (port block); 1 keeps clear of a recording stack"
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="object yaw / placement rng seed for round 0; +1 per round")
    ap.add_argument(
        "--name", default=None, help="run directory name under outputs/runs; default <experiment name>_plan"
    )
    ap.add_argument("--plan", type=Path, default=None, help="JSON list of steps; default derived from the task")
    ap.add_argument("--agent", action="store_true", help="let claude -p choose each primitive instead of a fixed plan")
    ap.add_argument("--agent-model", default="sonnet", help="model alias for claude -p (sonnet, opus, haiku)")
    ap.add_argument("--max-steps", type=int, default=12, help="agent mode: primitive calls per round before abort")
    ap.add_argument("--profile", default=None, help="profile name for the target instead of the scene's choice")
    ap.add_argument("--spawn-off", type=float, default=1.2, help="single-arm: metres behind the planned stand")
    ap.add_argument("--inject", type=Path, default=None, help="JSON failure injection (configs/injections/)")
    ap.add_argument("--keep-stack", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan, scene and stack; start nothing")
    return ap.parse_args(argv)


def default_plan(scene, task: dict, profile: str | None = None, profiles: dict | None = None) -> list[dict]:
    """The obvious plan for the task. A coupled single-arm profile places from the grasp stand (its measured
    geometry, pick_place ledger §12: 3/4 there vs 2/5 after a carried approach); the simple one walks to the goal first."""
    target = task.get("target") or scene.meta.get("target_object")
    obj = scene.objects[target]
    profile = profile or scene.profile_for(target)
    prof = (profiles or {}).get(profile)
    strategy = getattr(getattr(getattr(prof, "config", None), "reach", None), "strategy", "simple")
    if obj.bimanual:
        steps = [
            {"primitive": "approach", "args": {"target": target, "profile": profile}},
            {"primitive": "ready", "args": {"profile": profile}},
            {"primitive": "grasp_bimanual", "args": {"target": target, "profile": profile}},
        ]
    else:
        steps = [
            {"primitive": "approach", "args": {"target": target, "profile": profile}},
            {"primitive": "grasp_single", "args": {"target": target, "profile": profile}},
        ]
        if task.get("goal"):
            if strategy != "coupled":
                steps.append({"primitive": "approach", "args": {"target": task["goal"], "profile": profile}})
            steps.append({"primitive": "place", "args": {"container": task["goal"], "profile": profile}})
    return steps


def scripted_round(executor, steps: list[dict], injector=None) -> list[dict]:
    """`Executor.run_plan` with the injector between calls: stops at the first rejected or failed step."""
    entries = []
    for step in steps:
        entry = executor.step(step["primitive"], step.get("args"))
        entries.append(entry)
        if "rejected" in entry or not entry["outcome"]["ok"]:
            break
        if injector is not None:
            injector.after(entry, executor)
    return entries


def agent_round(executor, planner, session, scene, steps_max: int, log, injector=None) -> list[dict]:
    """Ask, execute, feed back, until the planner says done/abort, the round fails, or `steps_max` calls."""
    from locomani.skills.planner import scene_view
    from locomani.skills.session import snapshot

    entries, last = [], None
    for _ in range(steps_max):
        state = session.stack.obs()
        decision = planner.next_step(
            session=snapshot(session), scene=scene_view(session.stack, scene, state), last=last
        )
        log(f"[agent] -> {decision['next']} {json.dumps(decision.get('args') or {})} ({decision['rationale'][:120]})")
        if decision["next"] in ("done", "abort"):
            entries.append(executor.abort(decision["next"] + ": " + decision["rationale"]))
            break
        last = executor.step(decision["next"], decision.get("args"))
        entries.append(last)
        if injector is not None:
            injector.after(last, executor)  # the agent is not told; the next ask shows the state
        if (
            "outcome" in last
            and not last["outcome"]["ok"]
            and last["outcome"]["reason"].startswith(("[carry]", "[put_down]"))
        ):
            continue  # the agent may recover; a fallen robot ends below through the criterion
        if session.stack.obs().get("fallen"):
            break
    return entries


def spawn_single(session, p, reset, meta, target, profile, *, robot_nom, spawn_off, rng, log):
    """The bottle round's spawn (round_phases._phase_spawn_and_settle) at the planned stand, no jitter."""
    import numpy as np

    from locomani.backends.sonic import upper_ref
    from locomani.backends.sonic.agent_server import LocomotionMode, StreamMode
    from locomani.skills.catalogue.bottle import stand_plan
    from locomani.tasks.bottle.execution import reanchor

    init_xy = np.asarray(meta["objects"][target]["init_pos"][:2], dtype=float)
    _rel, psi, stand, _hint = stand_plan(meta, target, init_xy, profile.config.reach, session.scene.reach)
    fv2 = np.array([np.cos(psi), np.sin(psi)])
    spawn = stand - spawn_off * fv2
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    stale = bool(getattr(p, "_gym_started", False))
    if stale:
        yaw = float(p.pelvis_yaw(p.obs()))
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=(float(np.cos(yaw)), float(np.sin(yaw)), 0.0))
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.set_ghost()
    object_yaw = float(rng.uniform(-np.pi, np.pi))
    reset.reset(
        objects={target: {"dxy": [0.0, 0.0], "yaw": object_yaw}},
        robot={"dxy": [float(v) for v in spawn - robot_nom], "yaw": float(psi)},
    )
    time.sleep(0.3)
    landed = np.asarray(p.pelvis_xy(p.obs()))
    err = float(np.linalg.norm(landed - spawn))
    if err > 0.10:
        return False, f"[stand] teleport landed {err:.2f} m off"
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.start_control()
    p._gym_started = True
    if not upper_ref.shared().wait(3.0):
        log("[plan] WARNING: no body_q_target 3 s after start_control")
    facing = (float(fv2[0]), float(fv2[1]), 0.0)
    if stale:
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE)
        reanchor(p)
    p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE, facing=facing)
    s = p.obs()
    if s["fallen"]:
        return False, "[stand] robot did not stabilise after reset"
    drift = float(np.linalg.norm(np.asarray(p.pelvis_xy(s)) - landed))
    if drift > 0.35:
        return False, f"[stand] walked {drift:.2f} m off the spawn while standing up"
    session.heading = float(psi)
    p.hold_yaw = float(psi)
    log(f"[plan] spawn {np.round(spawn, 2)} psi {np.degrees(psi):.0f} deg -> settled {np.round(p.pelvis_xy(s), 2)}")
    return True, ""


def spawn_bimanual(session, p, reset, meta, profile, *, rng, log):
    """The crate round's start: sampled placement + stand_up with the planner's arms (expert.reset_hanging's head)."""
    from locomani.tasks.crate import expert
    from locomani.tasks.crate.visual_approach import approach_heading

    cfg = profile.config.visual
    placement, record = expert.start_placement(reset, meta, cfg, rng)
    heading = approach_heading(meta)
    expert.stand_up(p, reset, placement=placement, whole_body=True, phase=lambda *_a, **_k: None, heading=heading)
    placement("prepared_placement")
    time.sleep(0.3)
    session.heading = heading
    log(f"[plan] placement {json.dumps(record, default=float)[:160]}")
    return True, ""


def criterion_for(scene, task: dict, steps: list[dict] | None = None):
    """The task's success predicate over the privileged state: the template criterion when a goal is set and
    the plan ends in a place; "resting on the surface, nothing held" when it ends in put_down_bimanual; else
    "holding the target" (a carry / navigate plan is judged on the load, not on the goal)."""
    from locomani.skills.scene_model import TaskInstance

    target = task.get("target") or scene.meta["target_object"]
    goal = task.get("goal")
    last = steps[-1] if steps else None
    if goal and (last is None or last["primitive"] in ("place", "release", "abort")):
        inst = next(i for i in scene.instances() if (i.obj, i.target) == (target, goal))
        check = scene.criterion(inst)
        return inst.text, lambda state, session: check(state)
    if last is not None and last["primitive"] == "put_down_bimanual":
        surface = last.get("args", {}).get("surface") or scene.objects[target].on_surface
        phrase = scene.objects[target].phrase
        inst = TaskInstance(
            "put <obj> on <surface>",
            target,
            surface,
            "on",
            scene.profile_for(target),
            f"put {phrase} down on {surface}",
        )
        on_top = scene.criterion(inst)

        def put_down(state, session):
            if session.held is not None:
                return False, "[put_down] still held at the end of the plan"
            return on_top(state)

        return inst.text, put_down

    def held(state, session):
        if session.held is None or session.held.key != target:
            return False, "[hold] nothing held at the end of the plan"
        return True, ""

    return f"pick up {scene.objects[target].phrase}", held


def main(argv=None) -> int:
    args = parse(argv)
    # Every locomani port default is read at import time, and an exported TF_* (scripts/paths.sh exports
    # instance 0's block when INSTANCE is unset) wins over INSTANCE: pin the whole block here, first.
    from locomani.sim.ports import instance_ports

    os.environ["INSTANCE"] = str(args.instance)
    os.environ.update({k: str(v) for k, v in instance_ports(args.instance).items()})
    from locomani import paths
    from locomani.pipeline import run_context

    cfg, run_dir = run_context.open_run(
        args.experiment, args.name or f"{args.experiment.stem}_plan", write=not args.dry_run
    )
    scene_dir = Path(cfg["scene"]["dir"])
    scene_dir = scene_dir if scene_dir.is_absolute() else REPO / scene_dir
    paths.adopt_sim_dir(str(scene_dir))
    from locomani.skills import SceneModel, load_profiles

    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    task = cfg["task"]
    scene = SceneModel(meta)
    profiles = load_profiles()
    steps = json.loads(args.plan.read_text()) if args.plan else default_plan(scene, task, args.profile, profiles)
    if args.profile:  # a calibration profile replaces the plan's profile arguments wholesale (docs/skills.md §5)
        for step in steps:
            if "profile" in step.get("args", {}):
                step["args"]["profile"] = args.profile
    text, _ = criterion_for(scene, task, steps)
    injection = None
    if args.inject:
        from locomani.skills.injection import load_injection

        injection = load_injection(args.inject)
    print(
        f"[plan] run {run_dir}\n[plan] scene {scene_dir}\n[plan] task: {text}\n"
        + (
            f"[plan] agent: claude -p ({args.agent_model}), up to {args.max_steps} calls"
            if args.agent
            else f"[plan] steps {json.dumps(steps)}"
        )
        + (f"\n[plan] inject {injection}" if injection else ""),
        flush=True,
    )
    if args.dry_run:
        print(f"[plan] stack: instance {args.instance} gpu {args.gpu} no exporter; rounds {args.rounds}")
        return 0

    from locomani.stack.manager import Stack, spec_from_config

    video_dir = run_dir / "videos"  # the sim writes one file per reset.start_video/end_video pair
    video_dir.mkdir(parents=True, exist_ok=True)
    spec = dataclasses.replace(
        spec_from_config(
            cfg, run_dir=run_dir, gpu=args.gpu, scene_dir=str(scene_dir), record=False, video_dir=video_dir
        ),
        instance=args.instance,
    )
    stack = Stack(spec, run_dir=run_dir, mode="collect")
    env = stack.up()
    os.environ.update(env)
    try:
        return rounds(args, cfg, run_dir, meta, scene, steps, profiles, injection)
    finally:
        if not args.keep_stack:
            stack.down()


def rounds(args, cfg, run_dir, meta, scene, steps, profiles, injection=None) -> int:
    import numpy as np

    from locomani.backends.sonic.agent_server import LocomotionMode
    from locomani.backends.sonic.primitives import Primitives
    from locomani.pipeline.execution_record import ExecutionRecord
    from locomani.sim import ports
    from locomani.sim.privileged_pub import EpisodeControlClient
    from locomani.skills import Executor, Session, snapshot
    from locomani.skills.catalogue import build_catalogue
    from locomani.skills.injection import Injector
    from locomani.tasks.bottle import reach_map
    from locomani.tasks.bottle.grasp_gym import start_control_once

    task = cfg["task"]
    target = task.get("target") or meta["target_object"]
    bimanual = scene.objects[target].bimanual
    profile = profiles[args.profile or scene.profile_for(target)]
    planner = None
    if args.agent:
        from locomani.skills.planner import ClaudePlanner

        planner = ClaudePlanner(model=args.agent_model)
    text, criterion = criterion_for(scene, task, steps)
    out = run_dir / "skills"
    out.mkdir(parents=True, exist_ok=True)
    camera = None
    if bimanual:
        if cfg.get("backend", {}).get("render", "live") == "live":
            from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

            camera = ComposedCameraClientSensor(server_ip="localhost", port=ports.port("TF_CAMERA_PORT"))
        else:  # backend.render offline: the assisted expert runs on placeholder frames (expert.run does the same)
            from locomani.sim.dataset_cameras import PlaceholderCamera

            camera = PlaceholderCamera()
    p, reset = Primitives(meta=meta), EpisodeControlClient()
    scene.reach = reach_map.ReachMap()
    wins = 0
    try:
        p.cmd.reset_pose()
        reset.pin_arms()
        reset.reset()
        time.sleep(0.3)
        robot_nom = np.array(p.obs()["pelvis"]["pos"][:2])
        start_control_once(p)
        p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
        for rnd in range(args.rounds):
            t0 = time.time()
            rng = np.random.default_rng(args.seed + rnd)
            session = Session(
                stack=p,
                scene=scene,
                record=ExecutionRecord(),
                reset=reset,
                camera=camera,
                directory=out / f"round_{rnd:02d}",
            )
            executor = Executor(session, build_catalogue(profiles), trace_path=out / f"agent_trace_{rnd:02d}.jsonl")
            injector = Injector(injection, log=p.log) if injection is not None else None
            session.record.enter("spawn")
            if bimanual:
                ok, why = spawn_bimanual(session, p, reset, meta, profile, rng=rng, log=p.log)
            else:
                ok, why = spawn_single(
                    session,
                    p,
                    reset,
                    meta,
                    target,
                    profile,
                    robot_nom=robot_nom,
                    spawn_off=args.spawn_off,
                    rng=rng,
                    log=p.log,
                )
            entries = []
            if ok:
                reset.start_video(f"round_{rnd:02d}")
                if planner is not None:
                    planner.log_path = out / f"agent_io_{rnd:02d}.jsonl"
                    planner.start_episode(
                        task=text,
                        menu=[f"{i.text} (profile {i.profile})" for i in scene.instances()],
                        catalogue=executor.catalogue,
                        profiles=profiles,
                    )
                    entries = agent_round(executor, planner, session, scene, args.max_steps, p.log, injector)
                    ok, why = criterion(p.obs(), session)
                    if (
                        ok
                        and entries
                        and entries[-1].get("primitive") == "abort"
                        and entries[-1]["reason"].startswith("abort")
                    ):
                        ok, why = False, "[plan] agent aborted: " + entries[-1]["reason"][6:]
                else:
                    entries = scripted_round(executor, steps, injector)
                    last = entries[-1] if entries else None
                    if last is None or "rejected" in last:
                        ok, why = False, "[plan] " + (json.dumps(last["rejected"]) if last else "empty plan")
                    elif not last["outcome"]["ok"]:
                        ok, why = False, last["outcome"]["reason"]
                    else:
                        state = p.obs()
                        ok, why = criterion(state, session)
            if session.arm_owner.value == "pose_stream":
                executor.step("hand_back")  # leave the planner in charge before the next spawn
            try:
                reset.end_video()
            except Exception:  # noqa: BLE001 -- no video started when the spawn failed
                pass
            row = dict(
                round=rnd,
                seed=args.seed + rnd,
                ok=bool(ok),
                why=why,
                task=text,
                steps=[e["primitive"] for e in entries],
                completed=[e["primitive"] for e in entries if "outcome" in e and e["outcome"]["ok"]],
                agent=bool(planner is not None),
                agent_calls=planner.calls if planner is not None else None,
                injection=injector.fired if injector is not None else None,
                injected=bool(injector is not None and injector.fired is not None),
                seconds=round(time.time() - t0, 1),
                session=executor.session
                and __import__("locomani.skills.session", fromlist=["snapshot"]).snapshot(session),
            )
            wins += int(ok)
            with (out / "manifest.jsonl").open("a") as f:
                f.write(json.dumps(row, default=float) + "\n")
            print(f"[plan] round {rnd}: {'OK' if ok else 'FAIL'} {why} ({row['seconds']} s)", flush=True)
            p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
        print(f"[plan] {wins}/{args.rounds} OK -> {out / 'manifest.jsonl'}", flush=True)
    finally:
        for client in (camera, reset, p):
            if client is not None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
    return 0 if wins == args.rounds else 1


if __name__ == "__main__":
    sys.exit(main())

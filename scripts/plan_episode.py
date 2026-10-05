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
the target (a new object's calibration, docs/skills.md). `--record` makes it a collection entry instead
(scene_discovery plan decision 13): the stack runs the exporter (instance 0, the only one that records), every
round is an episode that starts after the spawn settles ('c') and is saved when the task criterion holds ('c')
or discarded ('x'), and `<run>/collect/manifest_00.jsonl` + the run summary's `collect` entry pair that episode
with its instruction -- one of the task's `prompts`, drawn per round, or the criterion text -- so
`scripts/convert_data.sh <run>` converts it like any collection run. `--inject` names a JSON failure injection
(`skills/injection.py`): a world event fired once per round after the first successful call of a named
primitive, identical for the fixed plan and the agent on the same seed (agentic-plan Q11; docs/skills.md
§7); `scripts/injection_report.py` pairs the two runs. `--think-s` gives a fixed plan an agent's decision
latency: the runner waits that long after every successful call, re-sending the current command (or nothing,
`--think-silent`), and traces what a single-arm load did meanwhile (TODO 8.3.5).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
    ap.add_argument(
        "--until-injected",
        type=int,
        default=None,
        help="with --inject: keep adding rounds until this many fired (--rounds is then the cap)",
    )
    ap.add_argument(
        "--think-s",
        type=float,
        default=0.0,
        help="fixed plan: wait this long after every successful call, as an agent's decision does (claude -p "
        "answers in 5-9 s); a single-arm load's palm-frame drift is traced as a `think` event",
    )
    ap.add_argument(
        "--think-silent",
        action="store_true",
        help="with --think-s: send nothing during the wait (the agent runner's gap; the deploy times the planner "
        "out after 1 s) instead of re-sending the current command",
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="record every round as a training episode (exporter on instance 0); fixed plans only",
    )
    ap.add_argument("--keep-stack", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan, scene and stack; start nothing")
    args = ap.parse_args(argv)
    if args.record:
        # what a training episode may contain is decided here, once: a think pause would be idle frames the
        # dataset has to cut (agentic-plan Q5, not built), an injection is a world event no policy should imitate
        refused = [
            flag for flag, on in (("--agent", args.agent), ("--inject", args.inject), ("--think-s", args.think_s)) if on
        ]
        if refused:
            ap.error(f"--record collects with fixed plans only; drop {', '.join(refused)}")
        if args.instance != 0:
            ap.error("--record needs --instance 0: only instance 0 runs the exporter")
    return args


def default_plan(scene, task: dict, profile: str | None = None, profiles: dict | None = None) -> list[dict]:
    """The obvious plan for the task. A coupled single-arm profile places from the grasp stand (its measured
    geometry, pick_place ledger §12: 3/4 there vs 2/5 after a carried approach); the simple one walks to the goal first.
    A goal that is a surface walks to a spot on it and puts the object down there (put_down_single)."""
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
        goal = task.get("goal")
        if goal and goal in scene.surfaces and goal not in scene.objects:
            steps.append({"primitive": "approach", "args": {"target": goal, "profile": profile}})
            steps.append({"primitive": "put_down_single", "args": {"surface": goal, "profile": profile}})
        elif goal:
            if strategy != "coupled":
                steps.append({"primitive": "approach", "args": {"target": task["goal"], "profile": profile}})
            steps.append({"primitive": "place", "args": {"container": task["goal"], "profile": profile}})
    return steps


def more_rounds(done: int, injected: int, rounds: int, until_injected: int | None) -> bool:
    """Another round? Plain runs stop at `rounds`; `--until-injected N` runs until N injections fired, capped at
    `rounds` (the injection fires only after a successful primitive, so a 50 % grasp rate halves the pair count)."""
    if done >= rounds:
        return False
    return until_injected is None or injected < until_injected


def scripted_round(
    executor, steps: list[dict], injector=None, think_s: float = 0.0, capture=None, keepalive: bool = True
) -> list[dict]:
    """`Executor.run_plan` with the injector between calls: stops at the first rejected or failed step.
    `think_s` > 0 waits after every successful call (`think`); `capture(k)` names the k-th wait's npz."""
    entries = []
    for k, step in enumerate(steps):
        entry = executor.step(step["primitive"], step.get("args"))
        entries.append(entry)
        if "rejected" in entry or not entry["outcome"]["ok"]:
            break
        if injector is not None:
            injector.after(entry, executor)
        if think_s > 0:
            think(executor, think_s, capture(k) if capture is not None else None, keepalive=keepalive)
    return entries


def think(executor, seconds: float, npz: Path | None = None, keepalive: bool = True) -> dict:
    """An agent's decision latency between two calls, through `Executor.idle` (`keepalive=False` is the agent
    runner's old gap, which the deploy times out on after 1 s). With a single-arm load, trace its palm-frame
    drift and contact, and keep the configuration stream (sim time + qpos) in `npz`."""
    import numpy as np

    from locomani.skills.catalogue.locomotion import LoadWatch
    from locomani.tasks.bottle.grasp_gym import palm_frame_offset

    session = executor.session
    p, held = session.stack, session.held
    watch = LoadWatch(p, held, palm_frame_offset) if held is not None and len(held.sides) == 1 else None
    rows, qpos = [], []

    def record(s):
        if watch is None:
            return
        _ok, drift = watch.load_ok(s)
        obj = s["objects"][held.key]
        rows.append([float(s["t"]), drift, float(obj["pos"][2]), held.sides[0] in (obj.get("touching") or ())])
        qpos.append([float(s["t"]), *s["qpos"]])

    executor.idle(seconds, keepalive=keepalive, on_tick=record)
    event = {"event": "think", "seconds": seconds, "keepalive": keepalive, "held": None if held is None else held.key}
    if rows:
        t0, t1 = rows[0][0], rows[-1][0]
        event.update(
            drift_m=round(rows[-1][1], 4),
            max_drift_m=round(watch.max_drift_m, 4),
            drift_rate_cm_s=round(100 * rows[-1][1] / max(t1 - t0, 1e-3), 3),
            dz_m=round(rows[-1][2] - rows[0][2], 4),
            touching_end=bool(rows[-1][3]),
            samples=[[round(r[0] - t0, 2), round(r[1], 4), round(r[2], 4)] for r in rows[:: max(1, len(rows) // 40)]],
        )
        if npz is not None:
            np.savez_compressed(npz, qpos=np.asarray(qpos), drift=np.asarray([r[:3] for r in rows]))
            event["npz"] = str(npz)
    return executor.note(event)


def agent_round(executor, planner, session, scene, steps_max: int, log, injector=None) -> list[dict]:
    """Ask, execute, feed back, until the planner says done/abort, the round fails, or `steps_max` calls."""
    from locomani.skills.planner import scene_view
    from locomani.skills.session import snapshot

    entries, last = [], None
    with ThreadPoolExecutor(max_workers=1) as pool:
        for _ in range(steps_max):
            state = session.stack.obs()
            # claude -p answers in 5-9 s; this thread keeps commanding meanwhile (Executor.idle), the ZMQ
            # sockets stay on it, and only the subprocess wait runs in the worker
            ask = pool.submit(
                planner.next_step, session=snapshot(session), scene=scene_view(session.stack, scene, state), last=last
            )
            executor.idle(ask.done)
            decision = ask.result()
            log(
                f"[agent] -> {decision['next']} {json.dumps(decision.get('args') or {})} ({decision['rationale'][:120]})"
            )
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


SINGLE_AXIS_YAW_DEG = 20.0  # capability.yaml reach_single_arm.grasp_yaw: |a| <= 20 deg 6/9, 20-40 2/5, > 40 0/18


def spawn_object_yaw(half_extent, psi: float, rng) -> float:
    """The target's spawn yaw: uniform, except for an object that fits the open jaw one way only (the coffee canister,
    8.2 x 12.3 cm), whose long side is kept within SINGLE_AXIS_YAW_DEG of the approach heading `psi`, either way
    round -- the existing grasp closes across whatever faces it and never holds the long side (TODO 9.12.11)."""
    import numpy as np

    from locomani.skills.scene_model import DEX1_OPENING_M

    if half_extent is None or 2 * max(float(half_extent[0]), float(half_extent[1])) <= DEX1_OPENING_M:
        return float(rng.uniform(-np.pi, np.pi))
    a = np.radians(rng.uniform(-SINGLE_AXIS_YAW_DEG, SINGLE_AXIS_YAW_DEG)) + np.pi * int(rng.integers(2))
    long_side = 0.0 if half_extent[0] >= half_extent[1] else np.pi / 2  # body-frame direction of the long side
    return float(np.angle(np.exp(1j * (psi + a - long_side))))


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
    if meta.get("room"):  # a scene_discovery room: walls and furniture behind the stand
        spawn_off = clear_spawn_off(meta, stand, fv2, spawn_off)
        if spawn_off is None:
            return False, "[stand] no clear spawn behind the planned stand"
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
    object_yaw = spawn_object_yaw((meta["objects"][target].get("box") or {}).get("half_extent"), psi, rng)
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
    log(
        f"[plan] spawn {np.round(spawn, 2)} psi {np.degrees(psi):.0f} deg -> settled {np.round(p.pelvis_xy(s), 2)}"
        f", {target} yaw {np.degrees(object_yaw):.0f} deg"
    )
    return True, ""


SPAWN_OFFS = (1.2, 1.0, 0.8, 0.6)  # m behind the stand, longest first: the approach was calibrated from 1.2
# m from every surface edge: the hanging hands reach ~0.25 m to the side and standing up drifts the pelvis 0.11-0.19 m
# (2026-10-03: a canister spawn 0.295 m from the round table settled with the right hand over its plate, and the
# approach's clearance check stopped 3/3 rounds at the first tick; docs/handoff/put-down-single-20261003.md §4)
SPAWN_EDGE_CLEAR = 0.45


def clear_spawn_off(meta, stand, fv2, wanted: float, _maps={}) -> float | None:  # noqa: B006  (per-scene cache)
    """The longest spawn distance (from `wanted` down to 0.6 m) the robot can stand at and walk in from.

    The spawn itself must be free in the inflated map (the body fits there) and SPAWN_EDGE_CLEAR from every
    surface edge (the hands fit there after the stand-up drift). The straight walk to the stand is
    checked against the raw footprints only, and not over its last 0.15 m: a stand is a pelvis pose 0.2-0.3 m
    from the table edge, inside the inflation by design (navigator docstring), and the guard walk brakes there."""
    import numpy as np

    from locomani.primitives.navigator import OccupancyMap

    key = id(meta)
    if key not in _maps:
        _maps[key] = OccupancyMap(meta)
    occ = _maps[key]
    from locomani.skills.catalogue.bottle import surfaces_edge_distance

    edge = surfaces_edge_distance(meta)

    def blocked(xy) -> bool:
        c = occ.cell_of(xy)
        return not occ.in_bounds(c) or bool(occ.occupied[c[0], c[1]])

    for off in (o for o in SPAWN_OFFS if o <= wanted + 1e-9):
        if not occ.is_free(stand - off * fv2) or edge(stand - off * fv2) < SPAWN_EDGE_CLEAR:
            continue
        walk = [stand - d * fv2 for d in np.arange(off, 0.15, -occ.res)]
        if not any(blocked(xy) for xy in walk):
            return off
    return None


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
    the plan ends in a place / put_down_single (a surface goal also needs the hand empty and off the object);
    "resting on the surface, nothing held" when it ends in put_down_bimanual; else "holding the target"
    (a carry / navigate plan is judged on the load, not on the goal)."""
    from locomani.skills.scene_model import TaskInstance

    target = task.get("target") or scene.meta["target_object"]
    goal = task.get("goal")
    last = steps[-1] if steps else None
    if goal and (last is None or last["primitive"] in ("place", "put_down_single", "release", "abort")):
        inst = next(i for i in scene.instances() if (i.obj, i.target) == (target, goal))
        check = scene.criterion(inst)
        if inst.kind != "on":
            return inst.text, lambda state, session: check(state)

        def put_on(state, session):  # "on" accepts the underside up to 0.30 m above the top: a held object passes
            if session.held is not None or (state["objects"][target].get("touching") or ()):
                return False, "[put_on] still in the hand at the end of the plan"
            return check(state)

        return inst.text, put_on
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
        + (f"\n[plan] inject {injection}" if injection else "")
        + (
            f"\n[plan] think {args.think_s:g} s after every call ({'silent' if args.think_silent else 'keepalive'})"
            if args.think_s and not args.agent
            else ""
        ),
        flush=True,
    )
    if args.dry_run:
        exporter = "exporter -> collect/manifest_00.jsonl" if args.record else "no exporter"
        print(f"[plan] stack: instance {args.instance} gpu {args.gpu} {exporter}; rounds {args.rounds}")
        if args.record:
            print(f"[plan] instructions: {cfg['task'].get('prompts') or [text]}")
        return 0

    from locomani.stack.manager import Stack, spec_from_config

    video_dir = run_dir / "videos"  # the sim writes one file per reset.start_video/end_video pair
    video_dir.mkdir(parents=True, exist_ok=True)
    spec = dataclasses.replace(
        spec_from_config(
            cfg, run_dir=run_dir, gpu=args.gpu, scene_dir=str(scene_dir), record=args.record, video_dir=video_dir
        ),
        instance=args.instance,
    )
    stack = Stack(spec, run_dir=run_dir, mode="collect")
    env = stack.up()
    os.environ.update(env)
    recording = None
    if args.record:
        from locomani.pipeline import batches

        collect_root = run_dir / "collect"
        collect_root.mkdir(parents=True, exist_ok=True)
        recording = Recording(batches.open_exporter_session(stack, collect_root, 0), collect_root / "manifest_00.jsonl")
    try:
        code = rounds(args, cfg, run_dir, meta, scene, steps, profiles, injection, recording)
        if recording is not None:
            recording.finish(run_dir)  # before stack.down(): the exporter may still be writing the last episode
        return code
    finally:
        if not args.keep_stack:
            stack.down()


class Recording:
    """One exporter session and its manifest: what makes this run convertible (`scripts/convert_data.sh`)."""

    TAIL_HOLD_S = 1.0  # a plan-end training_tail must sit inside the recorded frames, not on the last one
    SAVE_HOLD_S = 4.0  # as the bottle line: let the exporter write before the next spawn

    def __init__(self, session: str, manifest: Path):
        self.session, self.manifest = session, manifest
        self.rows: list[dict] = []

    def start(self, p) -> None:
        p.cmd.key("c")  # IDLE -> RECORDING; the first frame is the first frame of the task

    def stop(self, p, session, ok: bool, facing) -> None:
        """Save a success ('c'), discard a failure ('x'). A success gets the training_tail the converter trims
        at: the last primitive's own (grasp_single and place set one before their verification hold), else
        the plan's end."""
        from locomani.backends.sonic.agent_server import LocomotionMode

        if ok:
            if "training_tail" not in session.record.metrics:
                session.record.metrics["training_tail"] = dict(end_seq=int(p.obs()["seq"]), reason="plan_end")
                p.cmd.hold(self.TAIL_HOLD_S, mode=LocomotionMode.IDLE, facing=facing)
            p.cmd.key("c")
            p.cmd.hold(self.SAVE_HOLD_S, mode=LocomotionMode.IDLE, facing=facing)
        else:
            p.cmd.key("x")
            p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=facing)

    def write(self, row: dict) -> None:
        self.rows.append(row)
        with self.manifest.open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def finish(self, run_dir: Path) -> None:
        from locomani.pipeline import batches, run_context

        batches.wait_saved(self.session, self.manifest)
        kept = batches.kept_in(self.manifest, True)
        run_context.summarise(
            run_dir,
            "collect",
            attempts=len(self.rows),
            kept=kept,
            batches=1,
            yield_rate=round(kept / len(self.rows), 3) if self.rows else 0.0,
            failures=sorted({str(r.get("why") or "")[:70] for r in self.rows if not r.get("ok")}),
            sessions=[self.session],
            manifests=[str(self.manifest)],
        )
        print(f"[plan] recorded {kept} episode(s) -> {self.session}", flush=True)


def rounds(args, cfg, run_dir, meta, scene, steps, profiles, injection=None, recording=None) -> int:
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
    wins, injected, rnd = 0, 0, 0
    try:
        p.cmd.reset_pose()
        reset.pin_arms()
        reset.reset()
        time.sleep(0.3)
        robot_nom = np.array(p.obs()["pelvis"]["pos"][:2])
        start_control_once(p)
        p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
        while more_rounds(rnd, injected, args.rounds, args.until_injected):
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
            # the settle waits keep commanding: the drop_held event fires with the load in a frozen hand
            injector = Injector(injection, log=p.log, sleep=executor.idle) if injection is not None else None
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
            prompt = text
            recorded = False
            if ok and recording is not None:
                # its own stream, so recording does not shift the spawn draws of `rng` and a recorded run
                # meets the same scenes as an unrecorded one with the same seed
                prompts = task.get("prompts") or [text]
                prompt = str(np.random.default_rng([args.seed + rnd, 1]).choice(prompts))
                recording.start(p)
                recorded = True
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
                    entries = scripted_round(
                        executor,
                        steps,
                        injector,
                        think_s=args.think_s,
                        capture=lambda k, r=rnd: out / f"think_{r:02d}_{k}.npz",
                        keepalive=not args.think_silent,
                    )
                    last = entries[-1] if entries else None
                    if last is None or "rejected" in last:
                        ok, why = False, "[plan] " + (json.dumps(last["rejected"]) if last else "empty plan")
                    elif not last["outcome"]["ok"]:
                        ok, why = False, last["outcome"]["reason"]
                    else:
                        state = p.obs()
                        ok, why = criterion(state, session)
            if recorded:
                yaw = float(p.pelvis_yaw(p.obs()))
                recording.stop(p, session, bool(ok), (float(np.cos(yaw)), float(np.sin(yaw)), 0.0))
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
                task=prompt,  # the converter's per-episode instruction
                criterion=text,
                target=target,
                attempt=0,
                produced=recorded,
                plan=steps if planner is None else None,
                execution=session.record.snapshot() if recorded else None,
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
            injected += int(row["injected"])
            rnd += 1
            with (out / "manifest.jsonl").open("a") as f:
                f.write(json.dumps(row, default=float) + "\n")
            if recording is not None:
                recording.write(row)
            print(f"[plan] round {row['round']}: {'OK' if ok else 'FAIL'} {why} ({row['seconds']} s)", flush=True)
            p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
        print(f"[plan] {wins}/{rnd} OK ({injected} injected) -> {out / 'manifest.jsonl'}", flush=True)
    finally:
        for client in (camera, reset, p):
            if client is not None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
    return 0 if wins == rnd else 1


if __name__ == "__main__":
    sys.exit(main())

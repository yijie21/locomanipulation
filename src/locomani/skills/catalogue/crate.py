"""Bimanual black boxes wrapped from the crate line: approach (walk-in + brake) and grasp_bimanual.

Both run the pieces `expert.run_attempt` runs, in the same order, through `tasks/crate/attempt.py`; the
objects the offline tests monkeypatch on `expert.*` are looked up on that module at call time, so the same
fakes drive these adapters. What is left out is episode plumbing that belongs to the runner: the
teleported reset (`stand_up` / `start_placement`), the video and the exporter keys.

The walk-in ends with the arms wherever the profile's approach mode leaves them: hanging (guard mode,
`HangingWalk`) or readied (depth mode, `PlannerArm`), hence `approach` yields either planner owner. The
reach enters the POSE stream itself (`expert.handoff`) and, on success, leaves the session there holding
the box; a failed reach hands the stream back so the next primitive starts from the planner.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from locomani.skills.contract import Outcome, Violation
from locomani.skills.session import ArmOwner, HeldObject, Session, Stance, needs_heading, stance_violations

QUIET_SECONDS = 1.0  # reset_hanging's quiet admission before the first step


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def near_edge_for(meta: dict, key: str):
    """`visual_approach.near_edge` for an explicit target instead of the scene's `target_object`."""
    from locomani.tasks.crate.visual_approach import approach_heading

    surface = meta["surfaces"][meta["objects"][key]["surface"]]
    heading = approach_heading(meta)
    axis = np.array([np.cos(heading), np.sin(heading)])
    half = surface.get("flat_half_extent") or surface["half_extent"]
    reach = abs(axis[0]) * half[0] + abs(axis[1]) * half[1]
    return np.asarray(surface["xy"][:2], dtype=float) - axis * reach, axis


@dataclass
class CrateAttempt:
    """Everything the walk-in built that the reach needs; rides in `session.scratch["crate"]`."""

    key: str
    profile: str
    config: object
    cfg: object
    model: object
    solver: object
    coordinated_target: object
    perception: object
    recorder: object
    ghost: object
    ticks: object
    guard: dict | None
    heading: float
    body_top: float
    initial: dict
    grasp_points: object

    def close(self, p) -> None:
        if hasattr(p, "cmd"):
            p.cmd.reference_observer = None
        for cleanup in (self.recorder.close, self.ghost.stop):
            try:
                cleanup()
            except Exception:  # noqa: BLE001 -- closing artefacts must not mask the primitive's verdict
                pass


def _hanging_walk(p, heading, cfg, expert):
    """reset_hanging without its teleport: arms to the walking policy, stand quiet, then the HangingWalk."""
    from locomani.backends.sonic.agent_server import HAND_OPEN, StreamMode

    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(None)
    p.hold_yaw = float(heading)
    walk = expert.HangingWalk(p, heading)
    state = p.obs()
    began, quiet_since = float(state["t"]), None
    while float(state["t"]) - began < cfg.walk_timeout and not state["fallen"]:
        _w, x, y, _z = state["pelvis"]["quat"]
        upright = 1 - 2 * (x * x + y * y) >= np.cos(np.radians(cfg.upright_tolerance_deg))
        quiet = upright and state["pelvis"]["pos"][2] >= 0.72 and np.linalg.norm(state["pelvis"]["vel"]) < 0.06
        quiet_since = (float(state["t"]) if quiet_since is None else quiet_since) if quiet else None
        if quiet_since is not None and float(state["t"]) - quiet_since >= QUIET_SECONDS:
            return walk, True
        walk._push()
        state = p.obs()
    return walk, False


class CrateApproach:
    """Walk in along the scene heading with the planner's arms, brake (guard) or stop by depth, face the box."""

    name = "approach"
    requires = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})
    yields = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})

    def __init__(self, profiles):
        self._profiles = profiles

    def preconditions(self, session: Session, *, target: str, profile: str) -> list[Violation]:
        out = []
        prof = self._profiles.get(profile)
        if prof is None or prof.hands != "bimanual":
            out.append(Violation("profile", profile, note="bimanual approach needs a bimanual profile"))
        else:
            cfg, carry = prof.config.visual, prof.config.carry
            if cfg.pose_source != "assisted" or carry.reach_control != "whole_body":
                out.append(Violation("profile", profile, note="skills wrap the assisted whole-body line only"))
        obj = session.scene.objects.get(target)
        if obj is None:
            out.append(Violation("target", target, note=f"known: {sorted(session.scene.objects)}"))
        elif not obj.bimanual:
            out.append(Violation("target", target, note="not flagged bimanual in the scene"))
        if session.camera is None:
            out.append(Violation("camera", None, note="the bimanual line admits the target through the cameras"))
        if session.held is not None:
            out.append(Violation("held", session.held.key, note="approach walks; carry_to is unvalidated"))
        return out

    def run(self, session: Session, *, target: str, profile: str) -> Outcome:
        from locomani import paths
        from locomani.sim.dataset_cameras import PlaceholderCamera
        from locomani.tasks.crate import attempt, expert
        from locomani.tasks.crate.bimanual_ik import BimanualIK
        from locomani.tasks.crate.grasp_region import GRASP_BELOW_RIM_M, RimRegion, grasp_points_for
        from locomani.tasks.crate.visual_approach import approach_heading

        t0 = session.clock()
        p, meta = session.stack, session.scene.meta
        config = self._profiles[profile].config
        cfg = config.visual
        model = expert.ArmIK(paths.SCENE_XML)
        solver = BimanualIK(
            model,
            waist_pitch_max=config.carry.waist_pitch_max,
            waist_dofs=config.carry.waist_dofs,
            elbow_min_deg=cfg.elbow_min_deg,
        )
        key, target_meta = target, meta["objects"][target]
        grasp_points = grasp_points_for(target_meta, cfg.box_grasp_below_rim)
        rim_above_grasp = cfg.box_grasp_below_rim if target_meta.get("box") else GRASP_BELOW_RIM_M
        capture_region = RimRegion(across=cfg.capture_across) if cfg.capture_across > 0 else RimRegion()
        coordinated_target = expert.CoordinatedTarget(
            object_key=key, grasp_points_local=grasp_points, rim_above_grasp=rim_above_grasp, region=capture_region
        )
        approach = expert.VisualApproach(cfg)
        heading = approach_heading(meta)  # the guard line and the walk-in axis are the scene's, as in run_attempt
        guard = None
        if cfg.approach_stop == "guard":
            edge_xy, walk_axis = near_edge_for(meta, key)
            guard = dict(
                watch=expert.FootWatch(paths.SCENE_XML),
                rule=expert.GuardBrake(
                    extra=cfg.guard_extra,
                    margin=cfg.guard_margin,
                    stride0=cfg.guard_stride0,
                    stride_mode=cfg.guard_stride_mode,
                ),
                axis=walk_axis,
                line=float(walk_axis @ edge_xy),
                closest=-np.inf,
                walk_closest=-np.inf,
                brake=None,
            )
        session.record.enter("approach.prepare", mode=cfg.approach_stop, profile=profile)
        if guard is not None:
            arm, quiet = _hanging_walk(p, heading, cfg, expert)
            preparation = None
        else:
            arm, prepared = expert.prepare_stationary_ready(
                p,
                session.reset,
                cfg,
                solver,
                placement=lambda _reset_id: None,
                whole_body=True,
                heading=heading,
                after_brake=True,
            )
            quiet = bool(prepared["initial_stance_ready"])
            preparation = expert.ApproachPreparation(solver, cfg, arm.q)
        session.heading, p.hold_yaw = heading, heading
        if not quiet:
            self._leave(session, arm, guard)
            return Outcome(False, "[stand] initial_stance_unstable", duration_s=session.clock() - t0)
        initial = p.obs()
        body_top = np.inf
        if guard is not None and target_meta.get("box"):
            guard["line"] = min(guard["line"], expert.box_front_line(initial, key, target_meta["box"], guard["axis"]))
            box = target_meta["box"]
            body_top = float(initial["objects"][key]["pos"][2]) + float(box["centre"][2]) + float(box["half_extent"][2])
        perception = expert.AssistedGuidance(
            expert.ArmIK(paths.SCENE_XML),
            key=key,
            box=target_meta.get("box"),
            rendered=not isinstance(session.camera, PlaceholderCamera),
        )
        base = (
            Path(session.directory) if session.directory is not None else Path(tempfile.mkdtemp(prefix="skills_crate_"))
        )
        label = f"{key}_{len(session.trace):03d}"
        recorder = expert.VisualRecording(base / label)
        ghost = expert.GhostTargetTrace(str(base / "trace"), solver.offset)
        target_cache = expert.VisibleTarget()
        if hasattr(p, "cmd"):
            p.cmd.reference_observer = recorder.observe_reference
        started = session.clock()
        ticks = attempt.TickSource(
            p,
            session.camera,
            perception,
            target_cache,
            recorder,
            cfg,
            assisted=True,
            solver=solver,
            initial=initial,
            started=started,
        )
        walk = attempt.WalkState(
            p=p,
            approach=approach,
            arm=arm,
            guard=guard,
            preparation=preparation,
            solver=solver,
            cfg=cfg,
            assisted=True,
            perception=perception,
            stand_attempts=approach.stand_attempts,
            probe_factory=lambda raw, joints: expert.CoordinatedReach(
                solver,
                config,
                raw,
                reference_joints=joints,
                reference_quaternion=raw["pelvis"]["quat"],
                target=coordinated_target,
            ),
            ticks=ticks,
            last_motion={"movement": np.zeros(3), "speed": -1.0},
        )
        att = CrateAttempt(
            key,
            profile,
            config,
            cfg,
            model,
            solver,
            coordinated_target,
            perception,
            recorder,
            ghost,
            ticks,
            guard,
            heading,
            body_top,
            initial,
            grasp_points,
        )
        reason, ready = "no_camera_frames", False
        try:
            while ticks.elapsed() < cfg.walk_timeout + 2:
                tick = ticks.next()
                if tick is None:
                    continue
                if isinstance(tick, str):
                    reason = tick
                    break
                if tick.hold_only:
                    arm._push(arm.q, **walk.last_motion)
                    continue
                command, navigation, nav = attempt.walk_step(tick, walk)
                attempt.guard_track(guard, tick.raw, hanging=isinstance(arm, expert.HangingWalk), body_top=body_top)
                if command.phase != ticks.last_phase:
                    ticks.last_phase = command.phase
                    session.record.enter(f"approach.{command.phase}")
                if tick.fresh_image:
                    attempt.record_tick(
                        recorder, tick, command, navigation, low=None, arm=arm, perception=perception, assisted=True
                    )
                if command.finished:
                    reason = command.reason
                    break
                if nav.ready:
                    ready = True
                    break
            if ready and guard is not None and cfg.face_after_brake != "none":
                raw = p.obs()
                seen = target_cache.at(raw) or ticks.estimate
                turn = expert.face_target(p, arm, cfg, seen, raw, grasp_points)
                heading = att.heading = float(turn["heading"])
                session.record.enter("approach.turn", **{k: v for k, v in turn.items() if k != "heading"})
                ticks.touch()
        except Exception:
            att.close(p)
            self._leave(session, arm, guard)
            raise
        if not ready:
            att.close(p)
            self._leave(session, arm, guard)
            return Outcome(
                False,
                f"[approach] {reason}",
                duration_s=session.clock() - t0,
                metrics=dict(brake=guard and guard["brake"]),
            )
        raw = p.obs()
        est = target_cache.at(raw)
        stance = Stance(
            target=key,
            profile=profile,
            depth_m=float(est.position[0]) if est is not None else float("nan"),
            lateral_m=float(est.position[1]) if est is not None else float("nan"),
            yaw_err_rad=_wrap(p.pelvis_yaw(raw) - heading),
            quiet=True,
        )
        session.heading, p.hold_yaw = heading, heading
        self._leave(session, arm, guard)
        session.stance, session.scratch["crate"] = stance, att
        session.hands = {"left": 0.0, "right": 0.0}
        metrics = dict(brake=guard and guard["brake"], stand_attempts=list(approach.stand_attempts))
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=metrics, produced=stance)

    @staticmethod
    def _leave(session: Session, arm, guard) -> None:
        """Owner bookkeeping for how the walk-in left the arms: hanging (guard) or readied (depth)."""
        if guard is not None:
            session.arm, session.arm_owner = None, ArmOwner.PLANNER_SWING
        else:
            session.arm, session.arm_owner = arm, ArmOwner.PLANNER_FROZEN


class GraspBimanual:
    """POSE-stream handoff, then the coordinated reach to `hold`; the crate line's evaluation decides."""

    name = "grasp_bimanual"
    requires = frozenset({ArmOwner.PLANNER_FROZEN})
    yields = ArmOwner.POSE_STREAM

    def __init__(self, profiles):
        self._profiles = profiles

    def preconditions(self, session: Session, *, target: str, profile: str) -> list[Violation]:
        out = []
        v = needs_heading(session)
        if v is not None:
            out.append(v)
        stance = session.stance
        if stance is None or stance.target != target:
            out.append(
                Violation("stance", None if stance is None else stance.target, note="approach this target first")
            )
        elif stance.profile != profile:
            out.append(
                Violation("profile", stance.profile, note=f"approached with {stance.profile}, grasp asks {profile}")
            )
        else:
            out.extend(stance_violations(stance, self._profiles[profile].stance))
        att = session.scratch.get("crate")
        if att is None or att.key != target:
            out.append(Violation("prepared_attempt", None, note="no walk-in state for this target"))
        if session.arm is None or type(session.arm).__name__ != "PlannerArm":
            out.append(
                Violation("arm", type(session.arm).__name__ if session.arm else None, note="ready the arms first")
            )
        if session.held is not None:
            out.append(Violation("held", session.held.key))
        return out

    def run(self, session: Session, *, target: str, profile: str) -> Outcome:
        from locomani.tasks.crate import attempt, expert

        t0 = session.clock()
        att: CrateAttempt = session.scratch.pop("crate")
        p, cfg, config, ticks, arm = session.stack, att.cfg, att.config, att.ticks, session.arm
        ticks.touch()
        low, reason, streaming = None, "no_camera_frames", False
        began = float(ticks.final["t"])
        session.record.enter("grasp_bimanual.handoff")
        try:
            while float(ticks.final["t"]) - began < cfg.timeout_seconds + 2:
                tick = ticks.next()
                if tick is None:
                    continue
                if isinstance(tick, str):
                    reason = tick
                    break
                if tick.hold_only:
                    arm._push(arm.q)
                    continue
                if low is None:
                    if tick.estimate is None:
                        arm._push(arm.q)
                        continue
                    att.recorder.context["phase"] = "handoff"
                    if tick.fresh_image:
                        att.recorder.append(
                            tick.images,
                            dict(
                                wall_time=tick.now,
                                image_timestamps=tick.message["timestamps"],
                                state_seq=tick.raw.get("seq"),
                                simulation_time=tick.raw.get("t"),
                                joints=tick.joints,
                                base_quaternion=tick.raw["pelvis"]["quat"],
                                base_position=tick.raw["pelvis"]["pos"],
                                reference_joints=arm.q,
                                phase="handoff",
                            ),
                        )
                    arm, low = expert.handoff(
                        p,
                        att.model,
                        cfg,
                        config,
                        att.solver,
                        att.coordinated_target,
                        assisted=True,
                        geometry=None,
                        ghost=att.ghost,
                        label=att.key,
                    )
                    session.arm, session.arm_owner, streaming = arm, ArmOwner.POSE_STREAM, True
                    continue
                command = attempt.reach_step(
                    tick, arm=arm, low=low, solver=att.solver, cfg=cfg, assisted=True, recorder=att.recorder
                )
                attempt.guard_track(att.guard, tick.raw, hanging=False, body_top=att.body_top)
                if command.phase != ticks.last_phase:
                    ticks.last_phase = command.phase
                    session.record.enter(f"grasp_bimanual.{command.phase}")
                if tick.fresh_image:
                    attempt.record_tick(
                        att.recorder, tick, command, None, low=low, arm=arm, perception=att.perception, assisted=True
                    )
                if command.phase == "hold" and att.recorder.training_tail is None:
                    att.recorder.mark_training_end(tick.raw)
                if command.finished:
                    reason = command.reason
                    break
            result = expert.evaluate(
                att.initial,
                ticks.final,
                reason,
                lift_height=cfg.lift_height,
                max_tilt_deg=cfg.max_tilt_deg,
                key=att.key,
            )
        except Exception:
            att.close(p)
            self._release(session, arm, streaming)
            raise
        att.close(p)
        metrics = dict(reason=reason, evaluation={k: v for k, v in result.items() if k != "ok"})
        if getattr(low, "retries", None) is not None:
            metrics["retries"] = dict(low.retries)
        if not result["ok"]:
            self._release(session, arm, streaming)
            return Outcome(False, f"[reach] {result['why']}", duration_s=session.clock() - t0, metrics=metrics)
        held = HeldObject(
            key=att.key,
            sides=("left", "right"),
            grip_in_palm={},
            hold_offset=None,
            start_z=float(att.initial["objects"][att.key]["pos"][2]),
        )
        session.held, session.hands = held, {"left": 1.0, "right": 1.0}
        session.arm, session.arm_owner = arm, ArmOwner.POSE_STREAM
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=metrics, produced=held)

    @staticmethod
    def _release(session: Session, arm, streaming: bool) -> None:
        """A failed reach leaves the stream: hand back at the measured pose so the planner owns the arms again."""
        if streaming:
            try:
                arm.stop()
            except Exception:  # noqa: BLE001 -- the verdict is the reach's, not the hand-back's
                pass
        session.arm, session.arm_owner = None, ArmOwner.PLANNER_FROZEN

"""Single-arm black boxes wrapped from the bottle line: approach (guard walk), grasp_single, place.

Each adapter builds the `RoundContext` the bottle round phases expect from the session and the
`bottle_simple` profile, then calls the existing phase functions unchanged. Hand-over between the
approach and the grasp (the prepared `SimpleGraspPolicy`) rides in `session.scratch["bottle"]`.

Two facts a caller has to know (design §9a): the simple expert's `execute()` hands the POSE stream back
in its own `finally`, so `grasp_single` yields the frozen planner channel with the object held; and the
place phase drives a `PlannerArm` (frozen channel), so `place` runs from that same channel and needs no
`take_over` before it.

With something held, `approach` is not the guard walk (that hands the arms to the walking policy and would
swing the load): it is `carry_to` to the stand the round sampler would draw for the target, then a standing
turn to the table normal, and it writes the same `Stance`. That is the "walk to the container" step a
cross-surface pick-and-place needs before `place` (TODO 7.7).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from locomani.skills.catalogue.locomotion import CarryTo, LoadWatch, load_violations, turn_with_load
from locomani.skills.contract import Outcome, Violation
from locomani.skills.session import ArmOwner, HeldObject, Session, Stance, needs_heading, stance_violations

TURN_TOL = 0.16  # the spawn phase's tolerance for facing the table normal
WRAPPED_STRATEGIES = frozenset({"simple", "coupled"})  # bottle strategies the approach/grasp adapters have run live
CORRECT_TOL = 0.05  # loaded approach: residual (m) between the target's pelvis-frame position and the aimed one
STRAFE_MAX = 1.2  # loaded approach: within this of the stand and already facing the table, walk there without turning
LATERAL_TOL = 0.12  # approach: a stand more than this off the approach line is walked to first (holding the facing)
LEAD_BACK = 0.60  # approach: the pre-walk aims this far behind the stand along the table normal


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def surfaces_edge_distance(meta: dict):
    """grasp_gym's signed clearance to the nearest table footprint (negative inside), over every surface."""
    rects = [
        (np.asarray(s["xy"][:2], dtype=float), np.asarray(s.get("half_extent", [0.6, 0.6])[:2], dtype=float))
        for s in meta["surfaces"].values()
    ]

    def edge_dist(xy):
        best = np.inf
        for rc, rh in rects:
            dxy = np.abs(np.asarray(xy, dtype=float)[:2] - rc) - rh
            out = np.maximum(dxy, 0.0)
            best = min(best, float(np.hypot(out[0], out[1])) if (dxy > 0).any() else float(max(dxy)))
        return best

    return edge_dist


def stand_plan(meta: dict, target: str, object_xy, reach_config, rm, rng=None):
    """The stand the round sampler would draw for this target: (rel, psi, stand_xy, squat_hint).

    Same geometry as `round_loop` (stand = object − R(psi)·rel, psi = the table normal the object hugs);
    the round's ±0.14 rad yaw jitter and the 1 cm stand jitter are left out (rng seeded 0 unless given).
    """
    from locomani.tasks.bottle.plan import plan_for_target
    from locomani.tasks.bottle.ready_grasp_policy import sample_stand

    plan = plan_for_target(meta, target, rm, reach_config)
    rel, squat_hint = sample_stand(plan, rng if rng is not None else np.random.default_rng(0), reach_config)
    normal = np.zeros(2)
    normal[plan.ax] = -plan.sgn
    psi = float(np.arctan2(normal[1], normal[0]))
    c, s = np.cos(psi), np.sin(psi)
    stand = np.asarray(object_xy, dtype=float)[:2] - np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    return np.asarray(rel, dtype=float), psi, stand, squat_hint


def round_context(session: Session, profile, target: str, *, rm=None):
    """A `RoundContext` for the robot where it stands now (no spawn): the approach and grasp phases read it."""
    from locomani.tasks.bottle import grasp_gym, grasping
    from locomani.tasks.bottle.round_phases import RoundContext
    from locomani.tasks.bottle.tabletop_grasp import TabletopTarget

    p, meta, cfg = session.stack, session.scene.meta, profile.config
    if rm is None:
        from locomani.tasks.bottle import reach_map

        rm = reach_map.ReachMap()
    state = p.obs()
    obj_xy = np.asarray(state["objects"][target]["pos"][:2], dtype=float)
    rel, psi, stand, squat_hint = stand_plan(meta, target, obj_xy, cfg.reach, rm)
    surf_a = meta["surfaces"][meta["objects"][target].get("surface", "A")]
    ctx = RoundContext(
        p=p,
        reset=session.reset,
        meta=meta,
        obj_key=target,
        surf_a=surf_a,
        edge_dist=surfaces_edge_distance(meta),
        d_can=obj_xy,
        d_robot=np.asarray(p.pelvis_xy(state), dtype=float),
        psi=psi,
        side="right",
        rel=rel,
        rm=rm,
        spawn_xy=stand,
        squat_hint=squat_hint,
        execution=session.record,
        reach_config=cfg.reach,
        goal_key=None,
        place_config=cfg.place,
        task_config=cfg,
        arm_pin=cfg.arm_pin,
        edge_min=grasp_gym.EDGE_MIN,
        raise_lead=grasp_gym.RAISE_LEAD,
        raise_finish_max_s=grasp_gym.RAISE_FINISH_MAX_S,
        palm_drift_max=grasp_gym.PALM_DRIFT_MAX,
        tabletop_grasp_cls=grasp_gym.TabletopGrasp,
    )
    ctx.fv = (float(np.cos(psi)), float(np.sin(psi)), 0.0)
    ctx.task_grasp = grasp_gym.TabletopGrasp(
        p,
        TabletopTarget(
            key=target,
            side="right",
            below_center=cfg.reach.grip_below_center if cfg.reach.ready_grasp_recovery else None,
            grip_z=grasp_gym.obj_top(meta, target) - grasping.GRIP_BELOW_RIM,
            surface_top=surf_a["top"],
            box=meta["objects"][target]["box"],
            body=meta["objects"][target]["body"],
            surface_bodies=tuple(s["body"] for s in meta["surfaces"].values()),
        ),
        meta=meta,
        config=cfg,
    )
    ctx.grip_za = float(ctx.task_grasp.grip(state)[2])
    return ctx


class BottleApproach:
    """Face the table normal, neutral waist, then the simple expert's guard walk, brake and settle."""

    name = "approach"
    requires = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})
    # the guard walk hands the arms to the walking policy (SWING); a loaded approach keeps them frozen
    yields = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})

    def __init__(
        self,
        profiles,
        *,
        phases=None,
        prepare_ready=None,
        context=round_context,
        carry=None,
        palm_offset=None,
        stand=None,
    ):
        self._profiles, self._context = profiles, context
        self._carry = carry or CarryTo(palm_offset=palm_offset)
        self._stand_plan = stand or stand_plan
        if phases is None:
            from locomani.tasks.bottle.round_phases import _phase_navigate
            from locomani.tasks.bottle.round_phases_approach import _phase_approach

            phases = (_phase_navigate, _phase_approach)
        if prepare_ready is None:
            from locomani.tasks.bottle.ready_grasp_policy import prepare_ready
        self._phases, self._prepare_ready = phases, prepare_ready

    def preconditions(self, session: Session, *, target: str, profile: str, drift_max: float = 0.03) -> list[Violation]:
        out = []
        prof = self._profiles.get(profile)
        if prof is None or prof.hands != "single":
            out.append(Violation("profile", profile, note="single-arm approach needs a single-hand profile"))
        elif getattr(prof.config.reach, "strategy", "simple") not in WRAPPED_STRATEGIES:
            # the adapters run the round's navigate/approach phases and `execute_grasp`; these two strategies
            # are the ones measured through them (pick_place ledger §11-12, 2026-09-28)
            out.append(
                Violation(
                    "profile",
                    profile,
                    note=f"strategy {prof.config.reach.strategy!r}: wrapped are {sorted(WRAPPED_STRATEGIES)}",
                )
            )
        if target not in session.scene.objects:
            out.append(Violation("target", target, note=f"known: {sorted(session.scene.objects)}"))
        elif session.held is None and not session.scene.objects[target].single_hand_graspable:
            out.append(Violation("target", target, note="not single-hand graspable (bimanual, container or too wide)"))
        if session.held is not None:
            if session.arm_owner is not ArmOwner.PLANNER_FROZEN:
                out.append(Violation("held", session.held.key, note="a loaded approach starts from planner_frozen"))
            if target == session.held.key:
                out.append(Violation("target", target, note="cannot approach the object being carried"))
            out.extend(load_violations(session))
            v = needs_heading(session)
            if v is not None:
                out.append(v)
        return out

    def run(self, session: Session, *, target: str, profile: str, drift_max: float = 0.03) -> Outcome:
        if session.held is not None:
            return self._loaded(session, target=target, profile=profile, drift_max=drift_max)
        t0 = session.clock()
        p = session.stack
        ctx = self._context(session, self._profiles[profile], target)
        session.record.enter("approach.face", psi=ctx.psi)
        state = p.obs()
        if abs(_wrap(p.pelvis_yaw(state) - ctx.psi)) > TURN_TOL:
            from locomani.backends.sonic.primitives import LocomotionMode

            if not p.turn_to(ctx.psi, tol=TURN_TOL, timeout=20.0, mode=LocomotionMode.SLOW_WALK):
                return Outcome(False, "[stand] could not face the table", duration_s=session.clock() - t0)
        session.heading = ctx.psi
        p.hold_yaw = ctx.psi
        prewalk = self._prewalk(session, ctx)
        if prewalk is not None and not prewalk["ok"]:
            return Outcome(
                False,
                "[stand] could not reach the approach line",
                duration_s=session.clock() - t0,
                metrics=dict(prewalk=prewalk),
            )
        if not self._prepare_ready(p, arms=False):
            return Outcome(False, "[stand] neutral waist not delivered", duration_s=session.clock() - t0)
        try:
            for phase in self._phases:
                result = phase(ctx)
                if result is not None:
                    ok, why = result
                    session.arm_owner = ArmOwner.PLANNER_SWING
                    self._close(ctx)
                    return Outcome(
                        False, why if why.startswith("[") else f"[approach] {why}", duration_s=session.clock() - t0
                    )
        except Exception:
            self._close(ctx)
            raise
        from locomani.tasks.bottle.round_phases import _rel_now

        state = p.obs()
        rel_now = _rel_now(ctx, state)
        stats = getattr(ctx.coordinated, "stats", {}) or {}
        stance = Stance(
            target=target,
            profile=profile,
            depth_m=float(rel_now[0]),
            lateral_m=float(rel_now[1]),
            yaw_err_rad=_wrap(p.pelvis_yaw(state) - ctx.psi),
            quiet=bool((stats.get("stand") or {}).get("settled", True)),
        )
        session.stance, session.arm, session.arm_owner = stance, None, ArmOwner.PLANNER_SWING
        session.hands = {"left": 0.0, "right": 0.0}
        session.scratch["bottle"] = ctx
        metrics = {"park": session.record.metrics.get("park"), "first_stop": session.record.metrics.get("first_stop")}
        if prewalk is not None:
            metrics["prewalk"] = prewalk
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=metrics, produced=stance)

    @staticmethod
    def _prewalk(session: Session, ctx) -> dict | None:
        """The guard walk closes depth, not a lateral offset: with the stand more than LATERAL_TOL off the approach
        line (2026-09-29, ledger §13: a 0.6 m sideways move ended 2.5 s later in "arrival left the object out of
        reach"), walk to a lead point LEAD_BACK behind the stand first, holding the table normal as the facing."""
        stand = getattr(ctx, "spawn_xy", None)
        if stand is None:
            return None
        p = session.stack
        stand = np.asarray(stand, dtype=float)[:2]
        fv = np.array([np.cos(ctx.psi), np.sin(ctx.psi)])
        perp = np.array([-fv[1], fv[0]])
        here = np.asarray(p.pelvis_xy(p.obs()), dtype=float)
        lateral = float(np.dot(here - stand, perp))
        if abs(lateral) <= LATERAL_TOL:
            return None
        lead = stand - LEAD_BACK * fv
        session.record.enter("approach.prewalk", lateral=lateral, lead=lead.tolist())
        ok = bool(p.walk_to(lead, stop_dist=0.10, timeout=15.0, face_target=False, face_yaw=float(ctx.psi)))
        after = np.asarray(p.pelvis_xy(p.obs()), dtype=float)
        return {
            "ok": ok,
            "lateral_m": lateral,
            "lead": lead.round(3).tolist(),
            "residual_m": float(np.dot(after - stand, perp)),
        }

    def _loaded(self, session: Session, *, target: str, profile: str, drift_max: float = 0.03) -> Outcome:
        """Carry the load to the target's planned stand, face the table normal, write the stance."""
        t0 = session.clock()
        p, meta, cfg = session.stack, session.scene.meta, self._profiles[profile].config
        rm = session.scene.reach
        if rm is None:
            from locomani.tasks.bottle import reach_map

            rm = session.scene.reach = reach_map.ReachMap()
        state = p.obs()
        obj_xy = np.asarray(state["objects"][target]["pos"][:2], dtype=float)
        _rel, psi, stand, _hint = self._stand_plan(meta, target, obj_xy, cfg.reach, rm)
        stand = np.asarray(stand, dtype=float)
        want = np.asarray(_rel, dtype=float)[:2]
        if session.scene.objects[target].container and getattr(cfg, "place", None) is not None:
            # A container is approached for `place`, whose turn phase wants it inside the placing band
            # (pelvis-frame band_x x band_y); the grasp stand from `stand_plan` parks it 0.18-0.24 m to the
            # right, outside band_y (2026-09-28: 2/10 rounds died in "[turn] not in the placing band").
            want = np.array([float(np.mean(cfg.place.band_x)), float(np.mean(cfg.place.band_y))])
            c, sn = np.cos(psi), np.sin(psi)
            stand = obj_xy - np.array([c * want[0] - sn * want[1], sn * want[0] + c * want[1]])
        # Already facing this table edge and within STRAFE_MAX of the stand: walk there holding the facing instead
        # of turning with the load (2026-09-29, ledger §13: 7/7 recovered grasps were lost in that turn).
        here = np.asarray(p.pelvis_xy(state), dtype=float)
        strafe = bool(
            abs(_wrap(float(p.pelvis_yaw(state)) - float(psi))) <= TURN_TOL
            and float(np.linalg.norm(stand - here)) <= STRAFE_MAX
        )
        session.record.enter("approach.carry", target=target, stand=stand.tolist(), psi=float(psi), strafe=strafe)
        out = self._carry.run(
            session, xy=stand, stop_dist=0.06, timeout=30.0, turn_first=not strafe, drift_max=drift_max
        )
        if not out.ok:
            return Outcome(
                False,
                out.reason.replace("[carry]", "[approach]", 1),
                duration_s=session.clock() - t0,
                metrics=out.metrics,
            )
        watch = LoadWatch(p, session.held, self._carry._palm_offset, drift_max=drift_max)
        state = p.obs()
        if abs(_wrap(p.pelvis_yaw(state) - psi)) > TURN_TOL:
            session.record.enter("approach.face", psi=float(psi))
            _turned, lost = turn_with_load(p, watch, float(psi), TURN_TOL, "slow_walk")
            if lost:
                session.heading = float(p.pelvis_yaw(p.obs()))
                return Outcome(False, lost.replace("[carry]", "[approach]", 1), duration_s=session.clock() - t0)
        session.heading, p.hold_yaw = float(psi), float(psi)
        # The standing turn drifts the pelvis (measured 2026-09-28: the container ended 0.1-0.27 m right of the
        # aimed point after a ~90 deg slow_walk turn), so close the residual with short guarded moves, facing held.
        corrections = []
        for _ in range(2):
            state = p.obs()
            rel = np.asarray(p.world_to_root(state["objects"][target]["pos"], state), dtype=float)
            residual = rel[:2] - want
            if float(np.hypot(*residual)) <= CORRECT_TOL:
                break
            session.record.enter("approach.correct", residual=residual.tolist())
            out = self._carry.run(
                session,
                relative=[float(residual[0]), float(residual[1])],
                stop_dist=0.04,
                timeout=10.0,
                turn_first=False,
                drift_max=drift_max,
            )
            corrections.append([residual.round(3).tolist(), bool(out.ok)])
            if not out.ok:
                return Outcome(
                    False,
                    out.reason.replace("[carry]", "[approach]", 1),
                    duration_s=session.clock() - t0,
                    metrics=out.metrics,
                )
            session.heading, p.hold_yaw = float(psi), float(psi)
        state = p.obs()
        rel = np.asarray(p.world_to_root(state["objects"][target]["pos"], state), dtype=float)
        stance = Stance(
            target=target,
            profile=profile,
            depth_m=float(rel[0]),
            lateral_m=float(rel[1]),
            yaw_err_rad=_wrap(p.pelvis_yaw(state) - psi),
            quiet=True,
        )
        session.stance = stance
        metrics = dict(out.metrics, rel=rel[:2].tolist(), stand=stand.tolist(), corrections=corrections, strafe=strafe)
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=metrics, produced=stance)

    @staticmethod
    def _close(ctx) -> None:
        if ctx.coordinated is not None and hasattr(ctx.coordinated, "close"):
            ctx.coordinated.close()


class GraspSingle:
    """The simple expert's takeover → ready → reach → entry → close → lift → recover → hand-back, as one call."""

    name = "grasp_single"
    requires = frozenset({ArmOwner.PLANNER_SWING})
    yields = ArmOwner.PLANNER_FROZEN  # SimpleGraspPolicy.execute() releases the stream in its own finally

    def __init__(self, profiles, *, execute_grasp=None, palm_offset=None):
        self._profiles = profiles
        if execute_grasp is None:
            from locomani.tasks.bottle.execution import execute_grasp
        if palm_offset is None:
            from locomani.tasks.bottle.grasp_gym import palm_frame_offset as palm_offset
        self._execute, self._palm_offset = execute_grasp, palm_offset

    def preconditions(self, session: Session, *, target: str, profile: str, side: str = "right") -> list[Violation]:
        out = []
        if side != "right":
            out.append(Violation("side", side, note="the single-arm expert is calibrated for the right hand (D7)"))
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
        ctx = session.scratch.get("bottle")
        if ctx is None or ctx.obj_key != target or ctx.coordinated is None:
            out.append(Violation("prepared_policy", None, note="no prepared grasp policy for this target"))
        if session.held is not None:
            out.append(Violation("held", session.held.key))
        return out

    def run(self, session: Session, *, target: str, profile: str, side: str = "right") -> Outcome:
        t0 = session.clock()
        p, ctx = session.stack, session.scratch.pop("bottle")
        state = p.obs()
        start_z = float(state["objects"][target]["pos"][2])
        if session.reset is not None:
            session.reset.pin_arms()
        session.record.enter("grasp_plan")
        grip = ctx.task_grasp.grip(state)
        try:
            result = self._execute(
                p,
                target,
                grip,
                None,
                side=side,
                surface_top=ctx.surf_a["top"],
                replan=ctx.task_grasp.replan,
                ghost=None,
                nudge=False,
                execution=session.record,
                config=ctx.reach_config,
                coordinated=ctx.coordinated,
                transport_guard_factory=lambda ik: ctx.task_grasp.transport_guard(ik, ctx.reach_config),
                pin=(session.reset.pin_arms if ctx.arm_pin and session.reset is not None else None),
                squat_hint=ctx.squat_hint,
                plan_r=float(np.hypot(ctx.rel[0], ctx.rel[1])),
                stand_rel=None,
                envelope=None,
            )
        finally:
            BottleApproach._close(ctx)
            session.arm, session.arm_owner, session.stance = None, ArmOwner.PLANNER_FROZEN, session.stance
        stats = dict(getattr(ctx.coordinated, "stats", None) or {})  # the simple policy's phase/entry/failure evidence
        metrics = {"visibility": result.visibility, "stats": stats}
        if not result.ok:
            return Outcome(False, f"[grasp] {result.why}", duration_s=session.clock() - t0, metrics=metrics)
        s = p.obs()
        held = HeldObject(
            key=target,
            sides=(side,),
            grip_in_palm={side: np.asarray(self._palm_offset(s, target, side), dtype=float)},
            hold_offset=None,
            start_z=start_z,
        )
        session.held, session.hands[side] = held, 1.0
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=metrics, produced=held)


class Place:
    """Raise, turn to the container, carry over, descend, release, lift away, back off, verify (place_phase)."""

    name = "place"
    requires = frozenset({ArmOwner.PLANNER_FROZEN})
    yields = ArmOwner.PLANNER_FROZEN

    def __init__(self, profiles, *, place=None, arm_factory=None):
        self._profiles = profiles
        if place is None:
            from locomani.tasks.bottle.place_phase import _phase_place as place
        self._place, self._arm_factory = place, arm_factory or _planner_arm

    def preconditions(
        self, session: Session, *, container: str, profile: str, side: str = "right", palm_drift_max: float = 0.03
    ) -> list[Violation]:
        out = []
        prof = self._profiles.get(profile)
        if prof is None or prof.hands != "single":
            out.append(Violation("profile", profile, note="place is the single-arm line's"))
        held = session.held
        if held is None or side not in held.sides:
            out.append(Violation("held", None if held is None else held.key, note=f"nothing held in the {side} hand"))
        obj = session.scene.objects.get(container)
        if obj is None or not obj.container:
            out.append(Violation("container", container, note="not a container in this scene"))
        elif held is not None and container == held.key:
            out.append(Violation("container", container, note="cannot place an object into itself"))
        cam = session.stack.obs()["objects"].get(container, {}).get("cam") if obj is not None else None
        if cam is None:
            out.append(Violation("view", None, note="container not in the ego view"))
        return out

    def run(
        self, session: Session, *, container: str, profile: str, side: str = "right", palm_drift_max: float = 0.03
    ) -> Outcome:
        t0 = session.clock()
        p, meta, cfg = session.stack, session.scene.meta, self._profiles[profile].config
        key = session.held.key
        arm = self._arm_factory(p, side, cfg)
        arm.start()
        ctx = SimpleNamespace(
            p=p,
            obj_key=key,
            side=side,
            goal_key=container,
            place_config=cfg.place,
            surf_a=meta["surfaces"][meta["objects"][container].get("surface", "A")],
            meta=meta,
            execution=session.record,
            palm_drift_max=float(palm_drift_max),
            coordinated=SimpleNamespace(arm=arm),
        )
        session.arm = arm
        try:
            ok, why = self._place(ctx)
        finally:
            try:
                arm.stop()
            except Exception:  # noqa: BLE001 -- a stopped PlannerArm must not mask the place verdict
                pass
            session.arm, session.arm_owner, session.stance = None, ArmOwner.PLANNER_FROZEN, None
        session.hands[side] = 0.0
        session.held = None  # released whether or not the place verified
        metrics = dict(session.record.metrics.get("place") or {})
        reason = "" if ok else (why if why.startswith("[") else f"[place] {why}")
        return Outcome(ok, reason, duration_s=session.clock() - t0, metrics=metrics)


def _planner_arm(p, side, cfg):
    from locomani.backends.sonic.streamed_arm import PlannerArm

    return PlannerArm(p, side, send_vel=cfg.arm_vel, left_guard=cfg.left_guard)

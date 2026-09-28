"""Locomotion primitives: turn in place, free walk, backward retreat with a load, carry, navigate.

Each adapter wraps one existing entry (`Primitives.turn_to` / `walk_to` / the `Navigator` field,
`tasks/crate/retreat.retreat`) and translates the session in and the result out; the wrapped code is
unchanged. Dependencies are injected through the constructor so a test drives the adapter with a fake stack.
`LoadWatch` is the one per-tick load rule every walking-with-a-load primitive shares (carry_to,
navigate_to, and walk_to / approach when something is held).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from locomani.skills.contract import Outcome, Violation
from locomani.skills.session import ArmOwner, Session, needs_heading


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


WALKING = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})
TURN_MODES = frozenset({"idle", "slow_walk"})  # Primitives.turn_to's mode: IDLE re-aims in place, SLOW_WALK steps round


def _turn_mode(name: str):
    from locomani.backends.sonic.agent_server import LocomotionMode

    return LocomotionMode.SLOW_WALK if name == "slow_walk" else LocomotionMode.IDLE


class TurnTo:
    """Rotate on the spot to a world yaw and hold it as the new heading anchor."""

    name = "turn_to"
    requires = WALKING
    yields = None  # unchanged

    def preconditions(self, session: Session, *, yaw: float, tol: float = 0.15, mode: str = "idle") -> list[Violation]:
        out = []
        v = needs_heading(session)
        if v is not None:
            out.append(v)
        if tol < 0.15:
            out.append(Violation("tol", tol, (0.15, None), "the controller's heading dead zone is 7-8 deg"))
        if mode not in TURN_MODES:
            out.append(Violation("mode", mode, note=f"one of {sorted(TURN_MODES)}"))
        return out

    def run(self, session: Session, *, yaw: float, tol: float = 0.15, mode: str = "idle") -> Outcome:
        t0 = session.clock()
        session.record.enter("turn_to", yaw=float(yaw), mode=mode)
        session.stack.hold_yaw = float(yaw)
        ok = bool(session.stack.turn_to(float(yaw), tol=tol, mode=_turn_mode(mode)))
        session.stance = None
        if ok:
            session.heading = float(yaw)
        reason = "" if ok else "[turn] turn_to did not reach the yaw"
        return Outcome(ok, reason, duration_s=session.clock() - t0, metrics={"yaw": float(yaw)})


class WalkTo:
    """Straight planner walk to a pelvis xy. With a load it is `carry_to` (frozen upper body, load watched)."""

    name = "walk_to"
    requires = WALKING
    yields = None

    def __init__(self, carry=None):
        self._carry = carry or CarryTo()

    def preconditions(self, session: Session, *, xy, stop_dist: float = 0.05, timeout: float = 30.0) -> list[Violation]:
        out = []
        if session.held is not None:
            if session.arm_owner is not ArmOwner.PLANNER_FROZEN:
                out.append(
                    Violation("held", session.held.key, note="walk with a load from planner_frozen (hand_back first)")
                )
            out.extend(self._carry.preconditions(session, xy=xy, stop_dist=stop_dist, timeout=timeout))
        if len(xy) != 2:
            out.append(Violation("xy", len(xy), (2, 2), "world xy"))
        return out

    def run(self, session: Session, *, xy, stop_dist: float = 0.05, timeout: float = 30.0) -> Outcome:
        if session.held is not None:
            return self._carry.run(session, xy=xy, stop_dist=stop_dist, timeout=timeout)
        t0 = session.clock()
        session.record.enter("walk_to", xy=[float(v) for v in xy])
        ok = bool(session.stack.walk_to(np.asarray(xy, dtype=float), stop_dist=stop_dist, timeout=timeout))
        session.stance = None
        reason = "" if ok else "[walk] walk_to did not arrive"
        return Outcome(ok, reason, duration_s=session.clock() - t0)


class Retreat:
    """Hand the streamed arm back as it is and walk backward along the heading with the load held.

    Wraps `tasks/crate/retreat.retreat`; the wrapped function itself performs the hand-back, so this
    primitive enters on the POSE stream and yields the frozen planner channel.
    """

    name = "retreat"
    requires = frozenset({ArmOwner.POSE_STREAM})
    yields = ArmOwner.PLANNER_FROZEN

    def __init__(self, retreat_fn=None):
        if retreat_fn is None:
            from locomani.tasks.crate.retreat import retreat as retreat_fn
        self._retreat = retreat_fn

    def preconditions(
        self, session: Session, *, distance: float, speed: float = 0.25, timeout: float = 20.0, lift_height: float = 0.1
    ) -> list[Violation]:
        out = []
        v = needs_heading(session)
        if v is not None:
            out.append(v)
        if session.held is None:
            out.append(Violation("held", None, note="retreat checks the load; nothing is held"))
        elif session.held.start_z is None:
            out.append(Violation("held.start_z", None, note="the load floor is relative to the grasp height"))
        if distance <= 0:
            out.append(Violation("distance", distance, (0.0, None)))
        return out

    def run(
        self, session: Session, *, distance: float, speed: float = 0.25, timeout: float = 20.0, lift_height: float = 0.1
    ) -> Outcome:
        t0 = session.clock()
        session.record.enter("retreat", distance=float(distance))
        cfg = SimpleNamespace(
            retreat_distance=float(distance),
            retreat_speed=float(speed),
            retreat_timeout=float(timeout),
            lift_height=lift_height,
        )
        reason, diag, _state = self._retreat(
            session.stack, session.arm, cfg, session.held.key, session.heading, session.held.start_z
        )
        session.arm = None  # retreat() released the stream with preserve_reference=True
        session.arm_owner = ArmOwner.PLANNER_FROZEN
        session.stance = None
        ok = reason == "completed"
        return Outcome(ok, "" if ok else f"[retreat] {reason}", duration_s=session.clock() - t0, metrics=dict(diag))


class LoadWatch:
    """The per-tick load rule of every walk with something held (design §5 carry_to; bottle ledger §41).

    Single arm: the palm-frame offset of the object must stay within `drift_max` of where the grasp left it and
    that hand must keep touching it. Bimanual: the object stays at least `lift_floor_m` above its grasp height
    and both hands touch it. A contact gap shorter than `contact_gap_s` is tolerated (contacts flicker on a
    moving grasp); a palm-frame slip past `drift_max` is not. Usable as `Primitives.walk_to`'s `on_tick`.
    """

    def __init__(self, stack, held, palm_offset, *, drift_max=0.03, contact_gap_s=0.3, lift_floor_m=0.06):
        self.p, self.key, self.single, self.side = stack, held.key, len(held.sides) == 1, held.sides[0]
        self.drift_max, self.contact_gap_s, self._palm_offset = drift_max, contact_gap_s, palm_offset
        s0 = stack.obs()
        self.ref = np.asarray(palm_offset(s0, held.key, self.side), dtype=float) if self.single else None
        self.floor = None if self.single else float(held.start_z) + lift_floor_m
        self.max_drift_m, self.min_load_z = 0.0, np.inf
        self.samples: list = []
        self.lost_since, self.why, self._next_sample = None, None, -1.0

    def load_ok(self, state) -> tuple[bool, float]:
        obj = state["objects"][self.key]
        touching = set(obj.get("touching") or ())
        if self.single:
            drift = float(
                np.linalg.norm(np.asarray(self._palm_offset(state, self.key, self.side), dtype=float) - self.ref)
            )
            self.max_drift_m = max(self.max_drift_m, drift)
            return drift < self.drift_max and self.side in touching, drift
        z = float(obj["pos"][2])
        self.min_load_z = min(self.min_load_z, z)
        return z >= self.floor and {"left", "right"} <= touching, z

    def __call__(self, state, dist) -> bool:
        now = float(state["t"])
        ok, measure = self.load_ok(state)
        if self.single and measure >= self.drift_max:
            self.why = f"[carry] load slipped {measure * 100:.1f} cm in the palm frame"
            return False
        self.lost_since = None if ok else (now if self.lost_since is None else self.lost_since)
        if self.lost_since is not None and now - self.lost_since >= self.contact_gap_s:
            self.why = "[carry] load dropped (contact lost)" if self.single else "[carry] load dropped"
            return False
        if now >= self._next_sample:
            self._next_sample = now + 0.1
            self.samples.append([round(now, 3), round(float(dist), 3), round(float(measure), 4)])
        return True

    def metrics(self) -> dict:
        return dict(
            max_drift_m=self.max_drift_m if self.single else None,
            min_load_z=None if self.single else self.min_load_z,
            samples=self.samples,
        )


def _palm_offset_default():
    from locomani.tasks.bottle.grasp_gym import palm_frame_offset

    return palm_frame_offset


class CarryTo:
    """Walk to a world xy with the upper body frozen and the load watched every tick.

    The legs are the planner's (`Primitives.walk_to`); the arms stay pinned where the hand-back left them
    (PLANNER_FROZEN_UPPER_BODY keeps the last 17-joint reference and the closed hands). The load rule is
    the single-arm line's palm-frame drift (`PALM_DRIFT_MAX`, a standing criterion until measured here) or
    the crate line's retreat rule (height floor + both hands touching, contact flicker tolerated).
    `relative=[forward, left]` in the current heading frame is accepted instead of `xy`.

    Heading changes are made standing, before the walk (`turn_first`, via `Primitives.turn_to`), and the walk
    then holds that line (`face_yaw`). Measured 2026-09-28 with the bottle (docs/ledger/bottle.md §41): straight
    or backward 1-2 m, drift 0.3-0.8 cm (6/6); a 90 deg IDLE turn then 1.5 m, 3/3; a 180 deg turn is where the
    load goes -- IDLE in one go 0/2, two IDLE 90 deg steps 1/2, SLOW_WALK in one go 4/4 (drift 0.3-0.9 cm) --
    hence `turn_mode="slow_walk"`. `walk_to` flipping the facing while walking (`face_target=True`) slipped
    3 cm within 0.4 s (0/2) and is kept only as the raw behaviour.
    """

    name = "carry_to"
    requires = frozenset({ArmOwner.PLANNER_FROZEN})
    yields = ArmOwner.PLANNER_FROZEN

    def __init__(self, palm_offset=None):
        self._palm_offset = palm_offset or _palm_offset_default()

    def preconditions(
        self,
        session: Session,
        *,
        xy=None,
        relative=None,
        stop_dist: float = 0.08,
        timeout: float = 30.0,
        turn_first: bool = True,
        turn_tol: float = 0.2,
        turn_mode: str = "slow_walk",
        face_target: bool = False,
        drift_max: float = 0.03,
        contact_gap_s: float = 0.3,
        lift_floor_m: float = 0.06,
    ) -> list[Violation]:
        out = []
        v = needs_heading(session)
        if v is not None:
            out.append(v)
        out.extend(load_violations(session))
        if (xy is None) == (relative is None):
            out.append(Violation("target", None, note="give exactly one of xy (world) or relative ([forward, left] m)"))
        elif len(xy if xy is not None else relative) != 2:
            out.append(Violation("target", None, note="two coordinates"))
        if turn_mode not in TURN_MODES:
            out.append(Violation("turn_mode", turn_mode, note=f"one of {sorted(TURN_MODES)}"))
        return out

    def run(
        self,
        session: Session,
        *,
        xy=None,
        relative=None,
        stop_dist: float = 0.08,
        timeout: float = 30.0,
        turn_first: bool = True,
        turn_tol: float = 0.2,
        turn_mode: str = "slow_walk",
        face_target: bool = False,
        drift_max: float = 0.03,
        contact_gap_s: float = 0.3,
        lift_floor_m: float = 0.06,
    ) -> Outcome:
        t0 = session.clock()
        p = session.stack
        s0 = p.obs()
        yaw0 = float(p.pelvis_yaw(s0))
        start_xy = np.asarray(p.pelvis_xy(s0), dtype=float)
        target = relative_target(start_xy, yaw0, xy, relative)
        session.record.enter("carry_to", target=target.tolist(), face_target=face_target)
        watch = LoadWatch(
            p,
            session.held,
            self._palm_offset,
            drift_max=drift_max,
            contact_gap_s=contact_gap_s,
            lift_floor_m=lift_floor_m,
        )
        bearing = float(np.arctan2(*(target - start_xy)[::-1]))
        turned = None
        if turn_first and not face_target and abs(_wrap(bearing - yaw0)) > turn_tol:
            turned, why = turn_with_load(p, watch, bearing, turn_tol, turn_mode)
            if why:
                session.stance, session.heading = None, float(p.pelvis_yaw(p.obs()))
                return Outcome(False, why, duration_s=session.clock() - t0, metrics=dict(turned=turned))
        face_yaw = None if face_target else (bearing if turned else yaw0)
        arrived = bool(
            p.walk_to(
                target, stop_dist=stop_dist, timeout=timeout, face_target=face_target, face_yaw=face_yaw, on_tick=watch
            )
        )
        s = p.obs()
        session.stance = None
        session.heading = float(p.pelvis_yaw(s))
        p.hold_yaw = session.heading
        metrics = dict(
            turned=turned,
            walked_m=float(np.linalg.norm(np.asarray(p.pelvis_xy(s), dtype=float) - start_xy)),
            final_error_m=float(np.linalg.norm(target - np.asarray(p.pelvis_xy(s), dtype=float))),
            **watch.metrics(),
        )
        if not arrived:
            why = watch.why or f"[carry] {getattr(p, 'aborted', None) or 'walk_to failed'}"
            if watch.why is None and s.get("fallen"):
                why = "[carry] fell while carrying"
            return Outcome(False, why, duration_s=session.clock() - t0, metrics=metrics)
        still, _ = watch.load_ok(s)
        if not still:
            return Outcome(False, "[carry] load not held at arrival", duration_s=session.clock() - t0, metrics=metrics)
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=metrics)


def load_violations(session: Session) -> list[Violation]:
    """What a walk with a load needs from the session: something held, and its floor for the bimanual rule."""
    if session.held is None:
        return [Violation("held", None, note="this primitive moves a load; nothing is held")]
    if len(session.held.sides) > 1 and session.held.start_z is None:
        return [Violation("held.start_z", None, note="the bimanual load floor is relative to the grasp height")]
    return []


def relative_target(start_xy, yaw0: float, xy, relative) -> np.ndarray:
    """World target from `xy`, or from `relative=[forward, left]` in the heading frame at `start_xy`."""
    if xy is not None:
        return np.asarray(xy, dtype=float)
    fwd, left = (float(v) for v in relative)
    return start_xy + fwd * np.array([np.cos(yaw0), np.sin(yaw0)]) + left * np.array([-np.sin(yaw0), np.cos(yaw0)])


def turn_with_load(p, watch: LoadWatch, bearing: float, tol: float, mode: str) -> tuple[bool, str]:
    """Turn standing (no guard callback exists inside `turn_to`), then check the load once. -> (turned, why)."""
    p.hold_yaw = bearing
    turned = bool(p.turn_to(bearing, tol=tol, timeout=15.0, mode=_turn_mode(mode)))
    ok, _measure = watch.load_ok(p.obs())
    return turned, "" if ok else "[carry] load lost while turning"


class NavigateTo:
    """Walk to a world xy (or a ring around an object) through the scene's occupancy map, one waypoint at a time.

    The route is `Navigator`'s geodesic field (`Primitives.planner`, built from `scene_meta` when the stack has a
    map): each segment steers at `waypoint(here, step)` and is walked with `Primitives.walk_to`. Free-handed, the
    segment faces its waypoint as it walks (`face_target=True`, the `navigate` behaviour). With a load the segment
    follows carry_to's measured rule: turn standing first (SLOW_WALK, `turn_tol`), then walk the segment holding
    that line with `LoadWatch` on every tick; walking while turning is what slipped the bottle (ledger §41).
    """

    name = "navigate_to"
    requires = WALKING
    yields = None

    def __init__(self, palm_offset=None):
        self._palm_offset = palm_offset or _palm_offset_default()

    def preconditions(
        self,
        session: Session,
        *,
        xy=None,
        target: str | None = None,
        stand_off: float = 0.6,
        stop_dist: float = 0.10,
        timeout: float = 60.0,
        step: float = 0.45,
        turn_tol: float = 0.35,
        turn_mode: str = "slow_walk",
        drift_max: float = 0.03,
        contact_gap_s: float = 0.3,
        lift_floor_m: float = 0.06,
    ) -> list[Violation]:
        out = []
        if getattr(session.stack, "planner", None) is None:
            out.append(Violation("map", None, note="the stack has no occupancy map (Primitives built without meta)"))
        if (xy is None) == (target is None):
            out.append(Violation("target", None, note="give exactly one of xy (world) or target (an object key)"))
        elif target is not None and (session.scene is None or target not in session.scene.objects):
            out.append(Violation("target", target, note="not an object of this scene"))
        elif xy is not None and len(xy) != 2:
            out.append(Violation("target", None, note="two coordinates"))
        if session.held is not None:
            if session.arm_owner is not ArmOwner.PLANNER_FROZEN:
                out.append(Violation("held", session.held.key, note="navigate with a load from planner_frozen"))
            out.extend(load_violations(session))
        if turn_mode not in TURN_MODES:
            out.append(Violation("turn_mode", turn_mode, note=f"one of {sorted(TURN_MODES)}"))
        return out

    def run(
        self,
        session: Session,
        *,
        xy=None,
        target: str | None = None,
        stand_off: float = 0.6,
        stop_dist: float = 0.10,
        timeout: float = 60.0,
        step: float = 0.45,
        turn_tol: float = 0.35,
        turn_mode: str = "slow_walk",
        drift_max: float = 0.03,
        contact_gap_s: float = 0.3,
        lift_floor_m: float = 0.06,
    ) -> Outcome:
        t0 = session.clock()
        p, nav = session.stack, session.stack.planner
        s = p.obs()
        here = np.asarray(p.pelvis_xy(s), dtype=float)
        start_xy = here.copy()
        if target is not None:
            goal = np.asarray(s["objects"][target]["pos"][:2], dtype=float)
            planned = nav.set_goal_ring(goal, max(stand_off - 0.07, 0.0), stand_off + 0.07, here_xy=here)
        else:
            goal = np.asarray(xy, dtype=float)
            planned = nav.set_goal_point(goal, here_xy=here)
        session.record.enter("navigate_to", goal=goal.tolist(), target=target)
        if not planned or (not nav.reachable(here) and not (nav.replan(here) and nav.reachable(here))):
            session.stance = None
            return Outcome(False, "[navigate] no route to the goal", duration_s=session.clock() - t0)
        watch = None
        if session.held is not None:
            watch = LoadWatch(
                p,
                session.held,
                self._palm_offset,
                drift_max=drift_max,
                contact_gap_s=contact_gap_s,
                lift_floor_m=lift_floor_m,
            )
        tol = stop_dist + float(getattr(nav, "snap", 0.0))
        segments, why = [], None
        deadline = session.clock() + timeout
        best_d, since_progress = np.inf, session.clock()
        while why is None:
            s = p.obs()
            here = np.asarray(p.pelvis_xy(s), dtype=float)
            if s.get("fallen"):
                why = "[navigate] fell while navigating"
                break
            if not nav.reachable(here):
                nav.replan(here)
            d = float(nav.distance_at(here))
            if d < tol:
                break
            if d < best_d - 0.05:
                best_d, since_progress = d, session.clock()
            elif session.clock() - since_progress > 8.0:
                why = f"[navigate] no progress for 8 s, {d:.2f} m of field left"
                break
            if session.clock() >= deadline:
                why = f"[navigate] timed out with {d:.2f} m of field left"
                break
            wp = nav.waypoint(here, min(step, max(d, 0.15)))
            if wp is None:
                why = "[navigate] boxed in"
                break
            wp = np.asarray(wp, dtype=float)
            bearing = float(np.arctan2(*(wp - here)[::-1]))
            yaw = float(p.pelvis_yaw(s))
            seg_timeout = max(2.0, min(12.0, deadline - session.clock()))
            if watch is None:
                ok = bool(p.walk_to(wp, stop_dist=0.08, timeout=seg_timeout, face_target=True))
                segments.append([wp.round(3).tolist(), bool(ok)])
            else:
                turned = None
                if abs(_wrap(bearing - yaw)) > turn_tol:
                    turned, lost = turn_with_load(p, watch, bearing, turn_tol, turn_mode)
                    if lost:
                        why = lost
                        break
                ok = bool(
                    p.walk_to(
                        wp, stop_dist=0.08, timeout=seg_timeout, face_target=False, face_yaw=bearing, on_tick=watch
                    )
                )
                segments.append([wp.round(3).tolist(), bool(ok), turned])
                if not ok and watch.why:
                    why = watch.why
                    break
            if not ok and s.get("fallen"):
                why = "[navigate] fell while navigating"
                break
        s = p.obs()
        session.stance = None
        session.heading = float(p.pelvis_yaw(s))
        p.hold_yaw = session.heading
        metrics = dict(
            segments=segments,
            walked_m=float(np.linalg.norm(np.asarray(p.pelvis_xy(s), dtype=float) - start_xy)),
            field_left_m=float(nav.distance_at(np.asarray(p.pelvis_xy(s), dtype=float))),
            **(watch.metrics() if watch is not None else {}),
        )
        if why is None and watch is not None and not watch.load_ok(s)[0]:
            why = "[carry] load not held at arrival"
        return Outcome(why is None, why or "", duration_s=session.clock() - t0, metrics=metrics)

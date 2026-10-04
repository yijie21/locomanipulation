"""`put_down_single(surface)`: put the object held in one hand down on a surface (scene_discovery decision 11).

Built by Claude under the same acceptance bar as a hand-built skill: spec, gate and record in
docs/handoff/put-down-single-20261003.md. It runs the bottle line's place phase (`tasks/bottle/place_phase.py`)
with a surface and a spot instead of a container, on the same `PlannerArm`; the spot comes from the loaded
`approach(surface)` (`catalogue/surface_spot.py`).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from locomani.skills.catalogue.bottle import BAND_TOL, band_violation
from locomani.skills.catalogue.surface_spot import SPOT_MARGIN, _radius, reachable_spot, surface_occupants
from locomani.skills.contract import Outcome, Violation
from locomani.skills.session import ArmOwner, Session

_IK = []  # one ArmIK per process, built before any load is in the hand


def _shared_ik():
    if not _IK:
        from locomani.primitives.arm_ik import ArmIK

        _IK.append(ArmIK())
    return _IK[0]


def _planner_arm_shared_ik(p, side, cfg):
    """`bottle._planner_arm` without loading the IK model at the hand-over. Building ArmIK (0.2-0.3 s idle, more
    under load) between approach(surface) and the first arm frame let the deploy's 1 s planner timeout release the
    frozen arm and the hand: the canister was flung 1.4 m (2026-10-03, acceptance v2 round 11). PlannerArm only
    re-applies the same waist limits to the shared solver."""
    from locomani.backends.sonic.streamed_arm import PlannerArm

    return PlannerArm(p, side, send_vel=cfg.arm_vel, left_guard=cfg.left_guard, ik=_shared_ik())


class PutDownSingle:
    """Put the object held in one hand down on a surface, at the spot `approach(surface)` chose: raise, turn the spot
    into the placing band, squat when the spot is lower than the standing hand reaches, carry over, descend until it
    lands (`drop_m` above its resting height, or stopped sinking), open, lift away, stand, back off, verify that it
    stands on the surface, upright and untouched. Same place phase as `place` (docs/handoff/put-down-single-20261003.md)."""

    name = "put_down_single"
    requires = frozenset({ArmOwner.PLANNER_FROZEN})
    yields = ArmOwner.PLANNER_FROZEN

    def __init__(self, profiles, *, place=None, arm_factory=None):
        self._profiles = profiles
        if place is None:
            from locomani.tasks.bottle.place_phase import _phase_place as place
        if arm_factory is None:
            _shared_ik()  # the catalogue is built before the round's spawn: load the model now, not mid-carry
        self._place, self._arm_factory = place, arm_factory or _planner_arm_shared_ik

    def preconditions(
        self,
        session: Session,
        *,
        surface: str,
        profile: str,
        side: str = "right",
        drop_m: float = 0.01,
        palm_drift_max: float = 0.03,
    ) -> list[Violation]:
        out = []
        prof = self._profiles.get(profile)
        if prof is None or prof.hands != "single":
            out.append(Violation("profile", profile, note="put_down_single is the single-arm line's"))
        held = session.held
        if held is None or side not in held.sides:
            out.append(Violation("held", None if held is None else held.key, note=f"nothing held in the {side} hand"))
        if surface not in session.scene.surfaces:
            out.append(Violation("surface", surface, note="not a surface in this scene"))
        if not 0.0 <= drop_m <= 0.05:
            out.append(Violation("args", drop_m, (0.0, 0.05), "drop_m: release height above the resting height"))
        spot = session.scratch.get("spot")
        if spot is None or spot.get("surface") != surface:
            on = None if spot is None else spot.get("surface")
            out.append(Violation("spot", on, note=f"approach({surface}) first: it picks the spot and walks to it"))
        elif prof is not None and held is not None:
            state = session.stack.obs()
            aim = [*spot["xy"], float(session.scene.surfaces[surface].top)]
            band = band_violation(prof.config.place, session.stack.world_to_root(aim, state), BAND_TOL, surface)
            if band is not None:
                out.append(band)
            occupants = surface_occupants(session.scene.meta, surface, state)
            r = _radius(session.scene.meta["objects"][held.key].get("box")) + SPOT_MARGIN
            for key, (xy, rad) in occupants.items():
                gap = float(np.linalg.norm(np.asarray(spot["xy"]) - xy)) - r - rad
                if key != held.key and gap < 0:
                    out.append(
                        Violation("spot_clear", key, (0.0, None), f"{key} now stands {-gap:.2f} m into the spot")
                    )
        return out

    def run(
        self,
        session: Session,
        *,
        surface: str,
        profile: str,
        side: str = "right",
        drop_m: float = 0.01,
        palm_drift_max: float = 0.03,
    ) -> Outcome:
        t0 = session.clock()
        p, meta, cfg = session.stack, session.scene.meta, self._profiles[profile].config
        key = session.held.key
        spot = session.scratch.pop("spot")
        surf = meta["surfaces"][surface]
        pose = lambda st: [*(round(float(v), 3) for v in p.pelvis_xy(st)[:2]), round(float(p.pelvis_yaw(st)), 3)]  # noqa: E731
        poses = {"entry": pose(p.obs())}  # the base shifted 0.23-0.26 m before the first pulse in 3 of 4 v2/v3 rounds
        # Hold the yaw the robot has, not the approach's table normal: PlannerArm.start sends `p._facing` (hold_yaw), and
        # a 4 deg difference made the planner take a turning step: 0.11 m and 7 deg during the 0.6 s start
        # (2026-10-03, acceptance v4 round 6, `poses`).
        yaw_now = float(p.pelvis_yaw(p.obs()))
        p.hold_yaw, session.heading = yaw_now, yaw_now
        arm = self._arm_factory(p, side, cfg)
        setup_s = session.clock() - t0  # entry to the first arm frame: the deploy drops the arm after 1 s of silence
        # Take over the frozen channel at the reference it already holds (PlannerArm.start's "owned" entry), not at the
        # measured joints: under the load the two differ, and re-seeding moved the base 0.13 m with a 0.4 deg facing
        # difference (2026-10-03, acceptance v4 round 14, `poses`).
        ref = getattr(getattr(p, "cmd", None), "upper_body", None)
        owned = ref is not None and np.size(ref) == 17 and bool(np.all(np.isfinite(ref)))
        if owned:
            arm.start(upper_reference=np.asarray(ref, dtype=float), settle_seconds=0.0)
        else:
            arm.start()
        poses["after_start"] = pose(p.obs())
        ctx = SimpleNamespace(
            p=p,
            obj_key=key,
            side=side,
            goal_key=None,
            surface={
                "key": surface,
                "top": float(surf["top"]),
                "xy": [float(v) for v in surf["xy"][:2]],
                "half_extent": [float(v) for v in (surf.get("flat_half_extent") or surf["half_extent"])[:2]],
                "yaw": float(surf.get("yaw", 0.0)),
            },
            spot=spot["xy"],
            respot=lambda st, aim: reachable_spot(meta, surface, key, surface_occupants(meta, surface, st), aim),
            drop_m=float(drop_m),
            place_config=cfg.place,
            surf_a=surf,
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
            except Exception:  # noqa: BLE001 -- a stopped PlannerArm must not mask the verdict
                pass
            session.arm, session.arm_owner, session.stance = None, ArmOwner.PLANNER_FROZEN, None
        # a failure before the release leaves the load in the hand: read it back rather than assume it is gone
        if ok or side not in (p.obs()["objects"].get(key, {}).get("touching") or ()):
            session.hands[side], session.held = 0.0, None
        # place_servo: where each servo move started, ended and aimed (pelvis frame); kept on failure too
        metrics = dict(
            session.record.metrics.get("place_prep") or {}, **(session.record.metrics.get("place") or {}), spot=spot
        )
        metrics["servo"] = session.record.metrics.get("place_servo") or []
        metrics["setup_s"] = setup_s
        metrics["poses"] = poses  # pelvis [x, y, yaw]
        metrics["takeover"] = "owned_reference" if owned else "measured_joints"
        metrics["spot_final"] = session.record.metrics.get("place_spot_final")
        metrics["band_pulses"] = session.record.metrics.get("place_pulses") or []  # [x, y, s, [x, y] after]
        reason = "" if ok else (why if why.startswith("[") else f"[place] {why}")
        return Outcome(ok, reason, duration_s=session.clock() - t0, metrics=metrics)

"""`put_down_bimanual(surface)`: lower a box held in both hands onto a surface, open, lift the hands away.

The first implementation of the design's §5 "missing" entry. It runs on the POSE stream the bimanual grasp
left (`StreamedArm` with both hands closed, 29-joint reference `arm.q`): the two tool points are taken from
the reference through the profile's `BimanualIK` (pelvis-relative, the frame the reach solved in), lowered
together at `lower_speed` until the object rests on the surface (its z within `rest_tolerance` of the
resting height for `settle_seconds`, or the commanded drop is exhausted), then the hands ramp open over
`release_seconds`, then the tools rise `raise_m` and pull back `back_m` so the fingers clear the walls.
Every pushed frame re-solves the arms from the previous solution; the legs and waist stay where the
reference had them. No feedback on the measured tool points: the object is on the surface when its own
height says so, which is the verdict, not the hands' positions.
"""

from __future__ import annotations

import numpy as np

from locomani.skills.catalogue.arms import _bimanual_solver
from locomani.skills.contract import Outcome, Violation
from locomani.skills.session import ArmOwner, Session

DT = 0.02  # the POSE stream's frame period (backends/sonic/primitives_tuning.DT)


def _hand_open():
    from locomani.backends.sonic.agent_server import HAND_OPEN

    return np.asarray(HAND_OPEN, dtype=np.float32)


def resting_height(scene, key: str, surface: str) -> float:
    """World z of the object's origin when its box bottom sits on the surface top."""
    obj, surf = scene.objects[key], scene.surfaces[surface]
    return float(surf.top - obj.centre[2] + obj.half_extent[2])


def tilt_deg(quat_wxyz) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    zz = 1 - 2 * (x * x + y * y)  # third diagonal element of the rotation matrix
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


class PutDownBimanual:
    name = "put_down_bimanual"
    requires = frozenset({ArmOwner.POSE_STREAM})
    yields = ArmOwner.POSE_STREAM

    def __init__(self, profiles, solver_factory=_bimanual_solver, hand_open=None, sleep=None):
        self._profiles, self._solver = profiles, solver_factory
        self._open = hand_open if hand_open is not None else _hand_open()
        self._sleep = sleep  # None = the stream paces itself (StreamedArm._push sleeps to DT)

    def preconditions(
        self,
        session: Session,
        *,
        profile: str,
        surface: str | None = None,
        lower_speed: float = 0.05,
        clearance_m: float = 0.0,
        release_seconds: float = 1.0,
        raise_m: float = 0.10,
        back_m: float = 0.04,
        rest_tolerance: float = 0.01,
        settle_seconds: float = 0.3,
        timeout: float = 20.0,
    ) -> list[Violation]:
        out = []
        prof = self._profiles.get(profile)
        if prof is None or prof.hands != "bimanual":
            out.append(Violation("profile", profile, note="put_down_bimanual needs a bimanual profile"))
        held = session.held
        if held is None or len(held.sides) != 2:
            out.append(Violation("held", None if held is None else held.key, note="nothing held in both hands"))
        else:
            obj = session.scene.objects.get(held.key)
            if obj is None or obj.half_extent is None:
                out.append(Violation("target", held.key, note="the held object has no box in the scene"))
            key = surface if surface is not None else (obj.on_surface if obj is not None else None)
            if key not in session.scene.surfaces:
                out.append(Violation("surface", key, note=f"known: {sorted(session.scene.surfaces)}"))
        arm = session.arm
        if arm is None or type(arm).__name__ != "StreamedArm" or getattr(arm, "_stream_released", False):
            out.append(Violation("arm", type(arm).__name__ if arm else None, note="needs the live POSE stream"))
        if lower_speed <= 0 or raise_m < 0 or release_seconds <= 0:
            out.append(Violation("args", None, note="lower_speed > 0, raise_m >= 0, release_seconds > 0"))
        return out

    def run(
        self,
        session: Session,
        *,
        profile: str,
        surface: str | None = None,
        lower_speed: float = 0.05,
        clearance_m: float = 0.0,
        release_seconds: float = 1.0,
        raise_m: float = 0.10,
        back_m: float = 0.04,
        rest_tolerance: float = 0.01,
        settle_seconds: float = 0.3,
        timeout: float = 20.0,
    ) -> Outcome:
        t0 = session.clock()
        p, arm, key = session.stack, session.arm, session.held.key
        prof = self._profiles[profile]
        cfg, carry = prof.config.visual, prof.config.carry
        solver = self._solver(cfg, carry)
        surface = surface if surface is not None else session.scene.objects[key].on_surface
        rest_z = resting_height(session.scene, key, surface)
        s = p.obs()
        z0 = float(s["objects"][key]["pos"][2])
        drop = max(0.0, z0 - rest_z) + clearance_m
        q = np.asarray(arm.q, dtype=float).copy()
        posture = q.copy()
        tools0, rots0 = solver.tool_poses(q)
        session.record.enter("put_down.lower", drop_m=drop, rest_z=rest_z, surface=surface)
        stats = dict(drop_m=drop, rest_z=rest_z, lowered_m=0.0, min_tilt_deg=0.0, max_tilt_deg=0.0)

        def push(targets):
            nonlocal q
            sol = solver.solve(q, targets, rots0, iterations=8, posture=posture, posture_weight=0.01)
            q = np.asarray(sol.joints, dtype=float)
            arm.q = q
            arm._push(q)
            if self._sleep is not None:
                self._sleep(DT)

        def watch(state) -> str | None:
            if state.get("fallen"):
                return "[put_down] robot fell"
            tilt = tilt_deg(state["objects"][key]["quat"])
            stats["max_tilt_deg"] = max(stats["max_tilt_deg"], tilt)
            if tilt > cfg.max_tilt_deg:
                return f"[put_down] box tilted {tilt:.0f} deg"
            return None

        # 1. lower until the object rests
        n = max(1, int(round(max(drop / lower_speed, 0.5) / DT)))
        rest_since, lowered = None, 0.0
        for k in range(1, n + 1):
            s = p.obs()
            why = watch(s)
            if why:
                return self._fail(session, why, stats, t0)
            if session.clock() - t0 > timeout:
                return self._fail(session, "[put_down] timed out lowering", stats, t0)
            z = float(s["objects"][key]["pos"][2])
            touching = set(s["objects"][key].get("touching") or ())
            if not {"left", "right"} <= touching and z > rest_z + rest_tolerance:
                return self._fail(session, "[put_down] load dropped while lowering", stats, t0)
            if abs(z - rest_z) <= rest_tolerance:
                rest_since = rest_since if rest_since is not None else float(s["t"])
                if float(s["t"]) - rest_since >= settle_seconds:
                    break
            else:
                rest_since = None
            u = k / n
            lowered = u * drop
            push(tools0 - np.array([0.0, 0.0, lowered]))
        stats["lowered_m"] = float(lowered)
        stats["rest_z_error_m"] = float(p.obs()["objects"][key]["pos"][2] - rest_z)
        # 2. open both hands
        session.record.enter("put_down.release")
        start = {side: np.asarray(arm.hands[side], dtype=np.float32).copy() for side in ("left", "right")}
        tools_low = tools0 - np.array([0.0, 0.0, lowered])
        m = max(1, int(round(release_seconds / DT)))
        for k in range(1, m + 1):
            u = k / m
            arm.hands = {side: (start[side] + u * (self._open - start[side])).astype(np.float32) for side in start}
            push(tools_low)
        # 3. lift the hands away and back
        session.record.enter("put_down.raise", raise_m=raise_m, back_m=back_m)
        r = max(1, int(round(max(raise_m, back_m) / 0.10 / DT)))
        for k in range(1, r + 1):
            u = k / r
            push(tools_low + u * np.array([-back_m, 0.0, raise_m]))
            why = watch(p.obs())
            if why:
                return self._fail(session, why, stats, t0)
        # 4. verdict from the object alone
        s = p.obs()
        obj = s["objects"][key]
        z = float(obj["pos"][2])
        touching = set(obj.get("touching") or ())
        stats.update(final_z_error_m=z - rest_z, final_tilt_deg=tilt_deg(obj["quat"]), touching=sorted(touching))
        session.held = None
        session.hands = {"left": 0.0, "right": 0.0}
        if abs(z - rest_z) > 3 * rest_tolerance:
            return Outcome(
                False,
                f"[put_down] object {z - rest_z:+.3f} m off the resting height",
                duration_s=session.clock() - t0,
                metrics=stats,
            )
        if stats["final_tilt_deg"] > cfg.max_tilt_deg:
            return Outcome(
                False,
                f"[put_down] object tilted {stats['final_tilt_deg']:.0f} deg",
                duration_s=session.clock() - t0,
                metrics=stats,
            )
        if touching & {"left", "right"}:
            return Outcome(
                False, "[put_down] a hand still touches the object", duration_s=session.clock() - t0, metrics=stats
            )
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=stats)

    @staticmethod
    def _fail(session: Session, why: str, stats: dict, t0: float) -> Outcome:
        """A failure mid-way leaves the stream and the hands as they are; the load state is re-read from the sim."""
        s = session.stack.obs()
        obj = s["objects"][session.held.key]
        if not (set(obj.get("touching") or ()) & {"left", "right"}):
            session.held = None
            session.hands = {"left": 0.0, "right": 0.0}
        return Outcome(False, why, duration_s=session.clock() - t0, metrics=stats)

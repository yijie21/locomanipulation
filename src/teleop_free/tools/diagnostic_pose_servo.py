"""Opt-in fixed-world palm feedback with measured delivery and guard checks."""

import time

import numpy as np
from pose_feedback import PoseFeedback
from primitives import Primitives


def execute_pose_feedback(
    p, arm, target_pos, target_rot, guard, emit, *, timeout=25.0, settle=1.0, speed=0.5, origin_state=None
):
    initial = p.obs()
    anchor = initial if origin_state is None else origin_state
    origin = np.asarray(anchor["pelvis"]["pos"])
    objects = {k: np.asarray(v["pos"]) for k, v in anchor["objects"].items()}
    controller = PoseFeedback(arm.ik, arm.q, arm.side, speed=speed)
    began = time.monotonic()
    arrived_since = None
    best = float("inf")
    improved = began
    emit(
        "pose_feedback_start",
        speed=speed,
        excursion=0.35,
        settle=settle,
        timeout=timeout,
        reference=arm.q.tolist(),
        origin=origin.tolist(),
        state=initial,
        target_pos=np.asarray(target_pos).tolist(),
        target_rot=np.asarray(target_rot).tolist(),
    )
    while time.monotonic() - began < timeout:
        state = p.obs()
        now = time.monotonic()

        def finish(ok, reason, **metrics):
            emit("pose_feedback_exit", ok=ok, reason=reason, state=state, **metrics)
            return ok, reason

        if state["fallen"]:
            return finish(False, "fallen")
        if time.time() - state["wall"] > 0.2:
            return finish(False, "stale observation")
        drift = float(np.linalg.norm(np.asarray(state["pelvis"]["pos"]) - origin))
        if drift > 0.04:
            return finish(False, "base drift exceeded 4 cm", base_drift=drift)
        if any(np.linalg.norm(np.asarray(state["objects"][k]["pos"]) - pos) > 0.02 for k, pos in objects.items()):
            return finish(False, "object moved beyond 2 cm")
        root = Primitives._rot(state["pelvis"]["quat"])
        base = np.asarray(state["pelvis"]["pos"])
        palm = state["palms"][arm.side]
        candidate, metrics = controller.step(
            arm.q,
            root.T @ (np.asarray(palm["pos"]) - base),
            root.T @ Primitives._rot(palm["quat"]),
            root.T @ (target_pos - base),
            root.T @ target_rot,
            0.02,
        )
        checked = candidate.copy()
        checked[:12] = state["joints"][:12]
        collision = guard.check(state, checked, arm.side)
        if not collision.clear:
            return finish(False, "clearance rejected", collision=collision.to_dict(), **metrics)
        score = metrics["position_error"] / 0.045 + metrics["rotation_error"] / 0.25
        if score < best - 0.05:
            best = score
            improved = now
        arrived = metrics["position_error"] < 0.045 and metrics["rotation_error"] < 0.25
        if arrived:
            if arrived_since is None:
                arrived_since = now
            if now - arrived_since >= settle:
                return finish(True, "measured pose held", **metrics)
        else:
            arrived_since = None
        if not arrived and now - improved > 8.0:
            return finish(False, "no pose progress for 8 s", **metrics)
        emit("pose_feedback_tick", state=state, reference=candidate.tolist(), base_drift=drift, **metrics)
        arm.q = candidate
        arm._push(candidate, observed=state)
    emit("pose_feedback_exit", ok=False, reason="pose feedback timeout", state=p.obs())
    return False, "pose feedback timeout"

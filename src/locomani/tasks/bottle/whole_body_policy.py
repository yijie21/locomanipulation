"""Camera-admitted bottle reach through one supported descent/closure/lift reference."""

import json
import time
from collections import deque
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.backends.sonic.pose_preview import PosePreview
from locomani.backends.sonic.primitives import (
    HAND_CLOSED_RIGHT,
    HAND_OPEN,
    LocomotionMode,
    Primitives,
    upper_body_from_mj,
)
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.contacts import bilateral_dex1_contact
from locomani.primitives.supported_motion import SupportedMotion
from locomani.tasks.bottle.assisted_target import AssistedBottleTarget
from locomani.tasks.bottle.grasp_capture import closure_geometry
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ready_q, templates, terminal_rotation


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class WholeBodyGraspPolicy:
    def __init__(self, robot, task, config, arm_ramp, execution):
        require_pose_ownership(hand_handoff=True)
        self.p, self.task, self.config, self.ramp, self.execution = robot, task, config, arm_ramp, execution
        self.cfg = config.whole_body
        self.ik = ArmIK()
        self.guard = task.transport_guard(self.ik, config)
        self.guard.finger_only_target_contact = True
        self.target = AssistedBottleTarget(self.ik, task, self.cfg.vision_timeout)
        self.error = None
        self.arm = self.motion = None
        self.completed = False
        self.last_walk = None
        self.stats = execution.metrics.setdefault(
            "whole_body", dict(samples=[], references=[], source="assisted", clock="simulation")
        )
        self.previous_observer = robot.cmd.reference_observer
        robot.cmd.reference_observer = self.observe_reference
        self.context = {}
        self.closed = False

    def observe_reference(self, channel, payload):
        if self.previous_observer is not None:
            self.previous_observer(channel, payload)
        # Current consumed labels remain exporter-owned. Log sent reference
        # identity separately, including buffered future samples during handoff.
        if channel == "pose":
            # A preview contains ~50 future frames on every tick. Record its
            # identity/endpoints, not quadratic copies of the same trajectory;
            # samples below retain the current joint reference and velocity.
            payload = dict(payload)
            payload["preview_frames"] = len(payload["frame_index"])
            for key in ("joint_pos", "joint_vel", "body_quat", "frame_index"):
                payload[key] = [payload[key][0], payload[key][-1]]
        self.stats["references"].append(dict(channel=channel, sent_wall=time.time(), **self.context, payload=payload))

    def close(self):
        if not self.closed:
            self.target.close()
            self.p.cmd.reference_observer = self.previous_observer
            self.closed = True
            normalized = json.loads(
                json.dumps(
                    self.stats, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else float(value)
                )
            )
            self.stats.clear()
            self.stats.update(normalized)

    def walk_tick(self, state):
        target = self.target.observe(state)
        self.context = dict(seq=state["seq"], t=state["t"], phase="prepare", camera=self.target.diagnostics.copy())
        q = ready_q()
        if target is not None:
            depth = self.p.world_to_root_full(target, state)[0]
            u = smooth((0.65 - depth) / 0.35)
            q[14] = 0.10 * u
            q[22] -= 0.18 * u
        stamp = float(state["t"])
        if self.last_walk is not None and stamp <= self.last_walk:
            return True
        self.last_walk = stamp
        wanted = upper_body_from_mj(q)
        self.ramp.push(wanted)
        return True

    def _hold_planner(self, state):
        self.p.cmd.frame(
            mode=LocomotionMode.IDLE, height=-1.0, speed=0.0, movement=(0, 0, 0), facing=self.p._facing(state)
        )

    def _brake(self):
        start, quiet, last_progress = float(self.p.obs()["t"]), None, time.monotonic()
        previous = start
        while time.monotonic() - last_progress < 2:
            state = self.p.obs()
            stamp = float(state["t"])
            if state["fallen"] or stamp - start > 3:
                return False
            self.walk_tick(state)
            self._hold_planner(state)
            if stamp > previous:
                previous, last_progress = stamp, time.monotonic()
                quiet = (
                    (stamp if quiet is None else quiet) if np.linalg.norm(state["pelvis"]["vel"][:2]) < 0.06 else None
                )
                if quiet is not None and stamp - quiet >= 0.25:
                    return True
            time.sleep(0.01)
        return False

    def _measured_tools(self, state):
        return np.array(
            [
                np.asarray(state["palms"][side]["pos"]) + Primitives._rot(state["palms"][side]["quat"]) @ OFFSET
                for side in ("left", "right")
            ]
        )

    def _clear(self, state, candidate, contact, candidate_base=None):
        for side in ("left", "right", "body"):
            hit = self.guard.check(
                state, candidate, side, allow_target_contact=contact and side == "right", candidate_base=candidate_base
            )
            if not hit.clear:
                return f"whole-body clearance: {side}:{hit.link}:{hit.obstacle}"
        return None

    def execute(self, *, grip_world, squat_hint, ghost):
        ok = False
        try:
            self.execution.enter("whole_body.brake")
            if not self._brake():
                return False, "whole-body base did not stabilize"
            ok, why = self._reach()
            self.completed = ok
            self.stats["outcome"] = dict(ok=ok, reason=why, state=self.p.obs())
            return ok, why
        finally:
            if self.arm is not None:
                # Success returns to walking only after measured upright recovery.
                # Failure retains a bounded stopped reference; never rise through
                # an obstructed outstretched hand as a cleanup operation.
                if ok:
                    self.arm.stop(preserve_reference=True)
                else:
                    self.arm._push(self.arm.q, reference_velocity=np.zeros(29))
            # Serialization and camera teardown belong to round cleanup. Doing
            # them while carrying can starve planner commands past its timeout.
            self.p.cmd.reference_observer = self.previous_observer

    def prepare_carry(self):
        """The same reference already recovered upright with the bottle secured."""
        return self.completed

    def _reach(self):
        state = self.p.obs()
        target = self.target.observe(state)
        if target is None:
            return False, "whole-body target not visible in student cameras"
        self.arm = StreamedArm(self.p, "right", ik=self.ik, left_guard=False)
        physical = self.ik.model.jnt_range[self.ik.jids]
        self.ik.lo, self.ik.hi = physical[:, 0].copy(), physical[:, 1].copy()
        self.ik.hi[14] = min(self.ik.hi[14], self.cfg.waist_pitch_max)
        self.arm.preview = PosePreview(acceleration=self.cfg.joint_acceleration, joint_limits=(self.ik.lo, self.ik.hi))
        self.context["phase"] = "handoff"
        self.arm.start(hold_planner_pose=True)
        state = self.p.obs()
        self.motion = SupportedMotion(
            self.ik, state, self.cfg, self.arm.q, self.arm.root_quat, tool_offset=OFFSET, hand_weights=(0.35, 1.0)
        )
        motion = self.motion
        initial = self._measured_tools(state)
        rotations = np.array([Primitives._rot(state["palms"][side]["quat"]) for side in ("left", "right")])
        root = Primitives._rot(state["pelvis"]["quat"])
        rel = self.p.world_to_root_full(target, state)
        _, _, local_rot = min(templates(), key=lambda row: np.linalg.norm(row[1] - rel))
        desired_rot = terminal_rotation(root, local_rot, self.config)
        # Upright bottle: keep both entry and jaw axes level in world space.
        # Carrying the pelvis pitch into the wrist puts its housing into the neck.
        desired_rot = Rotation.from_euler("z", np.arctan2(desired_rot[1, 0], desired_rot[0, 0])).as_matrix()
        turn = Slerp([0, 1], Rotation.from_matrix([rotations[1], desired_rot]))
        gap = max(0.10, self.config.pregrasp_gap)
        approach_part = "clear"
        lean_target = float(np.clip((initial[1, 2] - target[2]) / 0.6, 0.0, 0.55))
        self.stats["initial"] = dict(
            state=state, reference=self.arm.q, root=self.arm.root_quat, support_offset=motion.support_offset
        )
        phase, progress, closure, stable_contact = "reach", 0.0, 0.0, None
        window = deque()
        began = previous = float(state["t"])
        last_wall = time.monotonic()
        obj0 = np.asarray(state["objects"][self.task.target.key]["pos"]).copy()
        targets = initial.copy()
        last_sample, ready_since = -1.0, None
        self.execution.enter("grasp.whole_body_reach")
        while time.monotonic() - last_wall < 2:
            state = self.p.obs()
            stamp = float(state["t"])
            dt = stamp - previous
            if dt == 0:
                time.sleep(0.005)
                continue
            if dt < 0 or dt > 0.5:
                return False, "whole-body physical state clock gap"
            previous, last_wall = stamp, time.monotonic()
            if stamp - began > self.config.recovery_timeout:
                return False, f"whole-body {phase} timed out"
            obj = state["objects"][self.task.target.key]
            actual = self._measured_tools(state)
            if (
                state["fallen"]
                or np.linalg.norm(np.asarray(obj["pos"])[:2] - obj0[:2]) > 0.05
                and phase in ("reach", "close")
            ):
                return False, "whole-body disturbed target or base"
            target = self.target.observe(state)
            if target is None and phase in ("reach", "close"):
                return False, "whole-body target visibility expired"
            self.context = dict(seq=state["seq"], t=stamp, phase=phase, camera=self.target.diagnostics.copy())
            torso_target = None
            if phase == "reach":
                outside = target - gap * desired_rot[:, 0]
                if approach_part == "clear":
                    goal = outside.copy()
                    goal[2] = max(initial[1, 2], target[2] + 0.12)
                    if np.linalg.norm((actual[1] - outside)[:2]) < 0.035:
                        approach_part = "descend"
                elif approach_part == "descend":
                    goal = outside.copy()
                    if abs(actual[1, 2] - target[2]) < 0.03:
                        approach_part = "insert"
                else:
                    # Stop inside the capture region, leaving room before the
                    # palm housing rather than centring the bottle against it.
                    goal = target - 0.010 * desired_rot[:, 0]
                delta = goal - targets[1]
                speed_limit = min(self.cfg.hand_speed, 0.02 + 0.5 * np.linalg.norm(goal - actual[1]))
                targets[1] += delta * min(1.0, speed_limit * min(dt, 0.1) / max(np.linalg.norm(delta), 1e-9))
                progress = min(1.0, progress + min(dt, 0.1) / 1.0)
                rotations[1] = turn(smooth(progress)).as_matrix()
                torso_target = motion.body.heading @ Rotation.from_euler("y", lean_target).as_matrix()
                self.context["approach_part"] = approach_part
                geometry = closure_geometry(state, "right", target)
                relative = target - actual[1]
                window.append((stamp, relative))
                while len(window) > 2 and stamp - window[1][0] > 0.2:
                    window.popleft()
                speed = (
                    np.inf
                    if stamp == window[0][0]
                    else np.linalg.norm(relative - window[0][1]) / (stamp - window[0][0])
                )
                if geometry["ready"] and speed < 0.08:
                    # Close while stabilizing the measured capture region. The
                    # slow finger ramp and bilateral-contact gate replace a
                    # separate stationary wait at an already graspable bottle.
                    phase = "close"
                    targets = actual.copy()
                    rotations = np.array([Primitives._rot(state["palms"][side]["quat"]) for side in ("left", "right")])
                    motion.body.put(motion.x)
                    capture_torso = self.ik.data.xmat[motion.body.torso].reshape(3, 3).copy()
                    motion.posture = motion.joints
                    self.execution.enter("grasp.close")
                    self.stats["capture"] = dict(t=stamp, geometry=geometry, relative_speed=speed)
            elif phase == "close":
                torso_target = capture_torso
                closure = min(1.0, closure + min(dt, 0.1) / self.cfg.close_seconds)
                contact = bilateral_dex1_contact(obj, "right")
                stable_contact = (stamp if stable_contact is None else stable_contact) if contact else None
                if closure >= 1:
                    if stable_contact is None or stamp - stable_contact < 0.08:
                        return False, "whole-body no sustained bilateral contact"
                    phase, progress = "lift", 0.0
                    lift_start = motion.body.tools(motion.x).copy() - self.cfg.tracking_gain * motion.feedback
                    relative_load = np.asarray(obj["pos"]) - actual[1]
                    retract = motion.body.heading @ np.array([-0.06, 0.0, 0.0])
                    self.execution.enter("grasp.lift_and_recover")
            else:
                torso_target = motion.body.heading @ Rotation.from_euler("y", lean_target).as_matrix()
                if np.linalg.norm(np.asarray(obj["pos"]) - actual[1] - relative_load) > 0.04:
                    return False, "whole-body bottle slipped during recovery"
                progress = min(1.0, progress + min(dt, 0.1) / 3.0)
                targets[1] = lift_start[1] + [0, 0, 0.16 * smooth(progress)]
                targets[1] += retract * smooth((progress - 0.25) / 0.75)
                if float(obj["pos"][2]) - obj0[2] >= 0.035:
                    torso_target = (
                        motion.body.heading @ Rotation.from_euler("y", lean_target * (1 - smooth(progress))).as_matrix()
                    )
                motion.body.put(
                    np.r_[
                        state["pelvis"]["pos"],
                        Rotation.from_matrix(Primitives._rot(state["pelvis"]["quat"])).as_euler("xyz")[1],
                        state["joints"],
                    ]
                )
                torso = motion.body.ik.data.xmat[motion.body.torso].reshape(3, 3)
                lean = np.arccos(np.clip(torso[2, 2], -1, 1))
                ready = progress >= 1 and float(obj["pos"][2]) - obj0[2] >= 0.06 and lean < np.radians(12)
                ready_since = (stamp if ready_since is None else ready_since) if ready else None
                if ready_since is not None and stamp - ready_since >= 0.2:
                    self.stats["loaded_upright"] = dict(
                        t=stamp, lean_deg=np.degrees(lean), rise=float(obj["pos"][2]) - obj0[2]
                    )
                    return True, "continuous whole-body grasp and upright recovery"
            previous_x, previous_root = motion.x.copy(), motion.root.copy()
            drift = motion.measured_foot_drift(state)
            if max(drift) > 0.05:
                self.stats["failure"] = dict(state=state, measured_foot_drift=drift)
                return False, "whole-body measured foot drift"
            pose = motion.step(
                targets,
                rotations,
                actual,
                dt,
                torso_rotation=torso_target,
                measured_rotations=np.array(
                    [Primitives._rot(state["palms"][side]["quat"]) for side in ("left", "right")]
                ),
            )
            # SONIC references include systematic decoder tracking error. Sweep
            # the next reference increment from the measured body, not a nominal
            # reference body placed inside the real scene as if it were measured.
            predicted = deepcopy(state)
            predicted["joints"] = np.asarray(state["joints"]) + pose.joints - previous_x[4:]
            delta_rotation = Primitives._rot(motion.root) @ Primitives._rot(previous_root).T
            predicted["pelvis"] = dict(
                state["pelvis"],
                pos=np.asarray(state["pelvis"]["pos"]) + pose.position - previous_x[:3],
                quat=Rotation.from_matrix(delta_rotation @ Primitives._rot(state["pelvis"]["quat"])).as_quat()[
                    [3, 0, 1, 2]
                ],
            )
            reason = self._clear(
                state,
                predicted["joints"],
                phase != "reach" or approach_part == "insert",
                candidate_base=[*predicted["pelvis"]["pos"], *predicted["pelvis"]["quat"]],
            )
            if reason:
                self.stats["failure"] = dict(
                    state=state,
                    predicted=predicted,
                    reference=pose.joints,
                    target=target,
                    planned_tool=targets[1],
                    phase=phase,
                    reason=reason,
                )
                return False, reason
            if max(pose.foot_errors) > 0.025 or np.linalg.norm(pose.com_shift) > 0.04:
                self.stats["failure"] = dict(
                    state=state,
                    predicted=predicted,
                    reference=pose.joints,
                    foot_error=pose.foot_errors,
                    com_shift=pose.com_shift,
                    phase=phase,
                )
                return False, "whole-body support reference outside tolerance"
            self.arm.q, self.arm.root_quat = pose.joints, motion.root
            self.arm.hands["right"] = HAND_OPEN + closure * (HAND_CLOSED_RIGHT - HAND_OPEN)
            self.arm._push(pose.joints, reference_velocity=motion.v[4:], reference_time=stamp)
            if stamp - last_sample >= 0.1:
                self.stats["samples"].append(
                    dict(
                        seq=state["seq"],
                        t=stamp,
                        phase=phase,
                        progress=progress,
                        target=targets[1].tolist(),
                        tool=actual[1].tolist(),
                        object=list(obj["pos"]),
                        waist=list(state["joints"][12:15]),
                        reference=list(pose.joints),
                        velocity=list(motion.v[4:]),
                        **motion.diagnostics,
                    )
                )
                last_sample = stamp
        return False, "whole-body physical state stopped"

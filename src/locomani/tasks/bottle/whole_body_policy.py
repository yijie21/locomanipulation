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
)
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.contacts import bilateral_dex1_contact
from locomani.primitives.supported_motion import SupportedMotion
from locomani.tasks.bottle.assisted_target import BottleTargetObserver
from locomani.tasks.bottle.grasp_capture import closure_geometry
from locomani.tasks.bottle.ready_grasp_policy import OFFSET
from locomani.tasks.bottle.walking_reach import WalkingReach


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def entry_step(command, measured, target, entry_axis, dt, hand_speed):
    """One continuous, cap-clearing step toward the bottle's capture region."""
    above = max(measured[2], command[2]) - target[2]
    gap = 0.018 + 0.10 * smooth((above - 0.018) / 0.04)
    goal = target - gap * entry_axis
    delta = goal - command
    behind = min(float((target - hand) @ entry_axis) for hand in (measured, command))
    if delta[2] < 0 and above > 0.035:
        delta[2] *= smooth((behind - 0.05) / 0.05)
    speed = min(hand_speed, 0.02 + 0.5 * np.linalg.norm(goal - measured))
    step = delta * min(1.0, speed * min(dt, 0.1) / max(np.linalg.norm(delta), 1e-9))
    return command + step, gap


class WholeBodyGraspPolicy:
    def __init__(self, robot, task, config, arm_ramp, execution):
        require_pose_ownership(hand_handoff=True)
        self.p, self.task, self.config, self.ramp, self.execution = robot, task, config, arm_ramp, execution
        self.cfg = config.whole_body
        self.ik = ArmIK()
        self.guard = task.transport_guard(self.ik, config)
        self.guard.finger_only_target_contact = True
        self.target = BottleTargetObserver(self.ik, task, self.cfg.vision_timeout)
        self.error = None
        self.arm = self.motion = None
        self.completed = False
        self.walking = WalkingReach(self.ik, self.cfg, arm_ramp.q)
        self.stats = execution.metrics.setdefault(
            "whole_body", dict(samples=[], walking=[], references=[], source="assisted", clock="simulation")
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
        if self.walking.last == float(state["t"]):
            return self.error is None
        started = time.perf_counter()
        target = self.target.observe(state)
        observed = time.perf_counter()
        self.context = dict(seq=state["seq"], t=state["t"], phase="prepare", camera=self.target.diagnostics.copy())
        previous = self.walking.q.copy()
        try:
            wanted, velocity = self.walking.step(state, target)
        except ValueError as exc:
            self.error = str(exc)
            self.stats["failure"] = dict(state=state, previous_t=self.walking.last, reason=self.error)
            return False
        planned = time.perf_counter()
        predicted = np.asarray(state["joints"], dtype=float).copy()
        delta = wanted - previous
        predicted[12:15] += delta[:3]
        predicted[15:22] += delta[3::2]
        predicted[22:29] += delta[4::2]
        self.error = self._clear(state, predicted, False)
        if self.error:
            self.stats["failure"] = dict(state=state, reference=wanted, reason=self.error)
            return False
        self.p.cmd.set_upper_body(wanted)
        self.p.cmd.set_upper_body_vel(velocity)
        elapsed = dict(camera=observed - started, planning=planned - observed, clearance=time.perf_counter() - planned)
        self.stats["last_walk_timing"] = dict(t=state["t"], **elapsed)
        if self.walking.diagnostics.get("plan_updated"):
            self.stats["walking"].append(dict(**self.walking.diagnostics, camera=self.target.diagnostics.copy()))
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
            if not self.walk_tick(state):
                return False
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
                return False, self.error or "whole-body base did not stabilize"
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
        desired_rot = self.walking.rotation
        if desired_rot is None:
            desired_rot = Rotation.from_euler("z", np.arctan2(root[1, 0], root[0, 0])).as_matrix()
        turn = Slerp([0, 1], Rotation.from_matrix([rotations[1], desired_rot]))
        # Preparation already moved the hand: lean follows bottle height, not
        # the remaining hand error (which would erase the bend after good preparation).
        lean_target = float(np.clip((state["pelvis"]["pos"][2] + 0.12 - target[2]) / 0.6, 0.0, 0.55))
        motion.body.put(motion.x)
        torso_reference = self.ik.data.xmat[motion.body.torso].reshape(3, 3).copy()
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
                targets[1], entry_gap = entry_step(
                    targets[1], actual[1], target, desired_rot[:, 0], dt, self.cfg.hand_speed
                )
                progress = min(1.0, progress + min(dt, 0.1) / 1.0)
                rotations[1] = turn(smooth(progress)).as_matrix()
                torso_target = motion.body.heading @ Rotation.from_euler("y", lean_target).as_matrix()
                self.context.update(approach_part="direct", entry_gap=float(entry_gap))
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
                if geometry["ready"] and geometry["depth_m"] >= 0.005 and speed < 0.08:
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
            if torso_target is not None:
                turn_delta = Rotation.from_matrix(torso_target @ torso_reference.T).as_rotvec()
                turn_delta *= min(1.0, 0.25 * dt / max(np.linalg.norm(turn_delta), 1e-9))
                torso_reference = Rotation.from_rotvec(turn_delta).as_matrix() @ torso_reference
            pose = motion.step(
                targets,
                rotations,
                actual,
                dt,
                torso_rotation=torso_reference,
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
                phase != "reach" or abs(actual[1, 2] - target[2]) < 0.035,
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

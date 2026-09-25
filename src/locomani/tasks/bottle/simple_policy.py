"""Simple single-arm bottle expert: walk in with hanging arms, stop, wrap the hand round the bottle, close, lift.

The shape of the crate line (walk → stand → reach → close → lift). The arms hang
as the deploy left them for the whole walk; the takeover waits for the planner
to finish its stop; then one continuous hand path goes up beside the body (after
a pull-back if the hand is in the table-top band at the edge), over the edge and
down to a pre-grasp on the bottle's outer side, and a single entry brings the
jaw round the bottle with the fingers pointing across the body, the way a person
wraps a hand round a bottle standing to one side. Each phase is an open-loop
Cartesian motion executed by the supported whole-body IK with its own rate
limits; the only gates are measured hard facts (collision sweep, fall, the
bottle knocked away, pad contact after the close, bottle rise after the lift)
and timeouts. No terminal observer, no stance planner, no quiet windows and no
progress governor: the controller tracks the hand to a few centimetres, so the
grasp is made tolerant of that instead of being gated on millimetres.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.pose_preview import PosePreview
from locomani.backends.sonic.primitives import HAND_CLOSED_RIGHT, HAND_OPEN, LocomotionMode, Primitives, StreamMode
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.contacts import bilateral_dex1_contact
from locomani.primitives.supported_motion import SupportedMotion
from locomani.tasks.bottle.assisted_target import BottleTargetObserver
from locomani.tasks.bottle.edge_guard import EdgeGuard, facing_edge
from locomani.tasks.bottle.ready_grasp_policy import OFFSET
from locomani.tasks.bottle.walking_reach import WalkingReach
from locomani.tasks.crate.bimanual_ik import BimanualIK

# Arm meshes that legitimately touch the bottle in the middle-pad grasp: the Dex1
# housing sits 0-4 mm from the held bottle (housing_clearance_review.json, 2026-09-16).
GRASP_ENVELOPE_MESHES = ("Dex1_base_link",)
PHASES = ("backoff", "ready", "reach", "settle", "entry", "retreat", "close", "lift", "recover")
CAP_CLEARING_CONTACT_HEIGHT_M = 0.035  # finger/target contact is allowed once the tool is near grip height
HAND_FORWARD_EXTENT_M = (
    0.05  # hanging Dex1 geometry reaches this far ahead of its tool point (guard reproduction, 2026-09-16)
)
ANKLE_BODIES = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}


def native(value):
    """Recursively convert numpy containers to JSON-native values for the round record."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def minjerk(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def moving_start(u, head=0.15):
    """Arc-length fraction for the reach: moves from the first tick, minimum-jerk stop.

    A minimum-jerk start leaves the hand at rest for the first 10% of the
    profile; after the stop that is a still hanging-arm phase a policy learns
    as "stop at the table". The first `head` of the arc is covered linearly at
    the profile's mean speed, the rest with a minimum-jerk blend to rest.
    """
    u = float(np.clip(u, 0.0, 1.0))
    head = float(np.clip(head, 0.0, 0.9))
    if u <= head:
        return u
    return head + (1.0 - head) * minjerk((u - head) / (1.0 - head))


def wrap_direction(pelvis_xy, grip, forward, yaw_per_m, yaw_max_rad):
    """Finger direction (palm x) for a wrap from the bottle's outer side.

    A bottle straight ahead is taken with the fingers forward. The further it
    sits to the robot's right, the more the fingers turn left across the body
    (yaw = `yaw_per_m` x lateral offset, capped), so the pre-grasp lands on the
    bottle's outer side and the elbow stays out: the bearing-aligned grasp of
    the earlier line pointed the right hand's fingers 28-40 deg outward and
    twisted the shoulder and wrist (stage4_hang8f review, 2026-09-16).
    """
    f = np.asarray(forward, dtype=float)[:2]
    f = f / float(np.linalg.norm(f))
    right = np.array([f[1], -f[0]])
    lateral = float((np.asarray(grip, dtype=float)[:2] - np.asarray(pelvis_xy, dtype=float)[:2]) @ right)
    yaw = float(np.clip(float(yaw_per_m) * lateral, 0.0, float(yaw_max_rad)))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * f[0] - s * f[1], s * f[0] + c * f[1], 0.0])


def grasp_rotation(direction):
    """Fingers level and pointing along the entry line."""
    direction = np.asarray(direction, dtype=float)
    return Rotation.from_euler("z", float(np.arctan2(direction[1], direction[0]))).as_matrix()


def pregrasp_point(grip, direction, back_m, up_m):
    return np.asarray(grip, dtype=float) - float(back_m) * np.asarray(direction, dtype=float) + [0.0, 0.0, float(up_m)]


def raise_height(grip_z, table_top, above_m, table_margin_m):
    """Tool height for the raise: above the grip, and above the table top by the hand's extent plus sag."""
    return max(float(grip_z) + float(above_m), float(table_top) + float(table_margin_m))


def raise_point(start_tool, forward, height, forward_m, outboard_m=0.0, right=None):
    """Where the hand goes up: ahead and outboard of its hanging spot, at the raise height.

    Outboard keeps the wrist camera off the torso: straight up beside the
    body it looked at the hip from 1-2 cm (ray review, 2026-09-16).
    """
    forward = np.asarray(forward, dtype=float)
    if right is None:
        right = np.array([forward[1], -forward[0], 0.0])
    point = (
        np.asarray(start_tool, dtype=float)
        + float(forward_m) * forward
        + float(outboard_m) * np.asarray(right, dtype=float)
    )
    point[2] = float(height)
    return point


def hanging_arms(reference17, simple):
    """The walking upper-body reference: the ramp's waist with both arms in the configured hang.

    Joint order per arm: shoulder pitch, roll, yaw, elbow, wrist roll, pitch,
    yaw; the left arm mirrors roll and both yaws.
    """
    q = np.asarray(reference17, dtype=float).copy()
    right = np.array(
        [
            simple.hang_shoulder_pitch_rad,
            -abs(simple.hang_shoulder_roll_rad),
            simple.hang_shoulder_yaw_rad,
            simple.hang_elbow_rad,
            simple.hang_wrist_roll_rad,
            simple.hang_wrist_pitch_rad,
            simple.hang_wrist_yaw_rad,
        ]
    )
    left = right * np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    q[3::2], q[4::2] = left, right
    return q


def needs_retract(tool_z, hand_edge_m, table_top, table_margin_m, edge_margin_m):
    """The hanging hand sits in the table-top band and within `edge_margin_m` of the edge.

    Then a straight raise sweeps it into the top and the hand is pulled back
    along the heading first. Both quantities come from the measured stand, so
    a higher table (hand below it, no contact possible on the way up) or a
    farther stand skips the pull-back.
    """
    in_band = float(tool_z) < float(table_top) + float(table_margin_m)
    return bool(in_band and float(hand_edge_m) < float(edge_margin_m))


def polyline_point(points, s):
    """The point at arc length `s` along the polyline `points` (clamped to the ends)."""
    pts = [np.asarray(q, dtype=float) for q in points]
    remaining = max(0.0, float(s))
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        if remaining <= length:
            return a if length < 1e-9 else a + (remaining / length) * (b - a)
        remaining -= length
    return pts[-1].copy()


def polyline_length(points):
    pts = [np.asarray(q, dtype=float) for q in points]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))


def entry_command(command, grip, direction, speed, dt, overshoot_m):
    """Advance the tool command along the entry line, capped at grip + overshoot (negative = stand-off)."""
    command = np.asarray(command, dtype=float).copy()
    grip = np.asarray(grip, dtype=float)
    direction = np.asarray(direction, dtype=float)
    limit = float((grip - command) @ direction) + float(overshoot_m)
    step = min(float(speed) * float(dt), max(0.0, limit))
    return command + step * direction


def entry_captured(measured_tool, grip, direction, capture_depth_m):
    """The measured tool point is within capture depth of the grip along the entry line."""
    depth = float((np.asarray(grip, dtype=float) - np.asarray(measured_tool, dtype=float)) @ np.asarray(direction))
    return depth <= float(capture_depth_m)


def object_in_palm(palm_rotation, tool, object_position):
    """Bottle position relative to the tool point in the palm frame (x entry, y jaw axis, z up)."""
    return np.asarray(palm_rotation, dtype=float).T @ (
        np.asarray(object_position, dtype=float) - np.asarray(tool, dtype=float)
    )


def entry_aligned(offset_in_palm, lateral_tolerance_m, height_tolerance_m):
    """Both pads will land on the bottle: it sits inside the jaw laterally and at pad height."""
    offset = np.asarray(offset_in_palm, dtype=float)
    return abs(float(offset[1])) <= float(lateral_tolerance_m) and abs(float(offset[2])) <= float(height_tolerance_m)


def lateral_correction(shift, palm_rotation, offset_in_palm, speed, dt, max_m):
    """Move the entry command sideways toward the measured jaw-axis offset, rate- and range-limited.

    The correction rides along with the advance instead of pausing at a
    stand-off: the hand wanders centimetres while it moves, and the jaw has
    2 mm of clearance, so the command follows the bottle continuously.
    """
    shift = np.asarray(shift, dtype=float).copy()
    wanted = shift + np.asarray(palm_rotation, dtype=float)[:, 1] * float(np.asarray(offset_in_palm, dtype=float)[1])
    delta = wanted - shift
    norm = float(np.linalg.norm(delta))
    step = min(norm, float(speed) * float(dt))
    if norm > 1e-9:
        shift += delta * (step / norm)
    total = float(np.linalg.norm(shift))
    if total > float(max_m):
        shift *= float(max_m) / total
    return shift


def retry_target(target, direction, palm_rotation, offset_in_palm, retreat_m):
    """Retreat along the entry line and shift the command by the measured jaw-axis/height offset."""
    palm = np.asarray(palm_rotation, dtype=float)
    offset = np.asarray(offset_in_palm, dtype=float)
    shift = palm[:, 1] * float(offset[1]) + np.array([0.0, 0.0, float(offset[2])])
    return np.asarray(target, dtype=float) - float(retreat_m) * np.asarray(direction, dtype=float) + shift


def finger_touching(obj, side):
    prefix = "R" if side == "right" else "L"
    links = set(obj.get("touch_links", []))
    return any(link.startswith(f"{prefix}.dex1_finger") for link in links)


def lean_angle(pelvis_pos, target, lean_max, forward=None):
    """Torso lean for the supported reach: 0 at 0.30 m of forward depth to the grip, the cap at 0.60 m.

    Forward depth only when `forward` (the pelvis heading) is given. The stand
    puts the bottle 0.24-0.30 m to the robot's right at 0.30 m of depth, so the
    straight-line distance is 0.37-0.52 m and leaned the torso 0.13-0.40 rad for
    a lateral offset no lean can shorten; that put a hip at the table in 4/8
    rounds (stage4_simple8, 2026-09-16). The whole-body line's height term was
    dropped earlier for the same reason.
    """
    delta = np.asarray(target[:2], dtype=float) - np.asarray(pelvis_pos[:2], dtype=float)
    depth = float(np.linalg.norm(delta)) if forward is None else float(delta @ np.asarray(forward, dtype=float)[:2])
    return float(np.clip((depth - 0.30) / 0.30 * float(lean_max), 0.0, float(lean_max)))


def torso_offset(ik, motion, point):
    """A world point expressed in the reference torso frame of the commanded body state."""
    motion.body.put(motion.x)
    frame = ik.data.xmat[motion.body.torso].reshape(3, 3)
    return frame.T @ (np.asarray(point, dtype=float) - ik.data.xpos[motion.body.torso])


def lift_target(ik, motion, offset, rise):
    """The torso-frame offset in the current commanded torso, raised by `rise` in world z.

    The held hand rides the reference torso: a world point plus torso recovery
    folded the elbow to the chin (2026-09-14), so the recovery itself raises the
    hand and `rise` only adds the explicit lift.
    """
    motion.body.put(motion.x)
    frame = ik.data.xmat[motion.body.torso].reshape(3, 3)
    return ik.data.xpos[motion.body.torso] + frame @ np.asarray(offset, dtype=float) + [0.0, 0.0, float(rise)]


def stance_geometry(ik, joints):
    """Foot separation in the body frame (forward, lateral) from the measured joints."""
    left, _ = ik.body_pose(np.asarray(joints, dtype=float), ANKLE_BODIES["left"])
    right, _ = ik.body_pose(np.asarray(joints, dtype=float), ANKLE_BODIES["right"])
    return dict(forward_m=float(left[0] - right[0]), lateral_m=float(left[1] - right[1]))


class SimpleGraspPolicy:
    def __init__(self, robot, task, config, arm_ramp, execution):
        require_pose_ownership(hand_handoff=True)
        self.p, self.task, self.config, self.ramp, self.execution = robot, task, config, arm_ramp, execution
        self.cfg = config.whole_body  # walking-phase law, IK limits and tracking compensation limits
        self.simple = config.simple
        self.ik = ArmIK()
        self.guard = task.transport_guard(self.ik, config)
        self.guard.finger_only_target_contact = True
        self.guard.allow_target_contact_meshes(GRASP_ENVELOPE_MESHES)
        self.target = BottleTargetObserver(self.ik, task, self.cfg.vision_timeout)
        # Local walking preparation only: no stance planner, no feasibility callback.
        walking_config = replace(self.cfg, natural_walk_prep=False)
        self.walking = WalkingReach(
            self.ik,
            walking_config,
            arm_ramp.q,
            brake_depth=config.visible_brake_depth,
            brake_edge=config.visible_brake_edge,
        )
        self.error = None
        self.arm = None
        self.completed = False
        self._stream_released = True
        self._walk_logged = -np.inf
        self._walk_q = np.asarray(arm_ramp.q, dtype=float).copy()  # ramps to the configured hang during the walk
        self._walk_stamp = None
        self._edge_anchor = None  # (pelvis-to-edge distance, pelvis xy) at the last walk tick that had one
        self.edge_guard = None  # approach_stop guard: built by prepare_walk, read by the approach loop
        self.context = {}
        self.stats = execution.metrics.setdefault(
            "simple", dict(samples=[], walking=[], phases=[], source="assisted", clock="simulation")
        )

    # ------------------------------------------------------------------ walking
    def set_approach_settlement_observer(self, observer):
        self._settlement_observer = observer

    def prepare_walk(self, heading=None, surface=None):
        """Standing, before the first step: both jaws open, both arms in the walking hang.

        Ramping the arms during the walk recorded a wrist turn with no visual
        cause (stage4_wrap8j: wrist roll 0.86 -> 1.26 over the last 1.5 s of
        the walk) and left the ramp still moving at the takeover, where the
        mode switch replays the residual as a jump. The commander's cached
        hand targets are also the previous round's closed jaw, which the
        planner packet re-applies for a tick at the switch: open them here.
        """
        p, simple = self.p, self.simple
        p.cmd.set_hands(left=HAND_OPEN.copy(), right=HAND_OPEN.copy())
        if simple.approach_stop == "guard":
            return self._prepare_guard_walk(heading, surface)
        goal = hanging_arms(self._walk_q, simple)
        state = p.obs()
        began = float(state["t"])
        previous = began
        while True:
            state = p.obs()
            stamp = float(state["t"])
            if stamp == previous:
                time.sleep(0.005)
                continue
            dt = max(0.0, stamp - previous)
            previous = stamp
            step = simple.hang_ramp_rad_s * dt
            self._walk_q += np.clip(goal - self._walk_q, -step, step)
            p.cmd.set_upper_body(self._walk_q.copy())
            p.cmd.set_upper_body_vel(np.zeros(17))
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(state))
            measured = np.asarray(state["joints"], dtype=float)
            arms_err = float(np.max(np.abs(np.r_[measured[15:22], measured[22:29]] - np.r_[goal[3::2], goal[4::2]])))
            reached = float(np.max(np.abs(goal - self._walk_q))) < 1e-6
            if (reached and arms_err <= simple.hang_reached_rad) or stamp - began >= simple.hang_prepare_timeout_s:
                self.stats["prepare_walk"] = dict(seconds=stamp - began, arm_error_rad=arms_err, reached=bool(reached))
                return

    def _prepare_guard_walk(self, heading, surface):
        """approach_stop guard: the arms go back to the walking policy, as in the warehouse walk
        (crate/expert.py HangingWalk) -- no upper-body joints on the wire, so the planner's own swing drives them --
        and the brake watches the table edge the heading walks into."""
        if heading is None or surface is None or "half_extent" not in surface:
            raise ValueError("simple guard walk needs the approach heading and the table footprint (half_extent)")
        p = self.p
        p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
        p.cmd.upper_body = None
        p.cmd.set_upper_body_vel(None)
        edge = facing_edge(surface["xy"], surface["half_extent"], heading)
        self.edge_guard = EdgeGuard(self.ik.model, edge, float(self.task.target.surface_top), self.simple)
        self.stats["prepare_walk"] = dict(arms="walking policy", edge_axis=edge.axis.tolist(), edge_line=edge.line)

    def _hold_walk_arms(self):
        """Keep referencing the walking hang; with the guard walk the walking policy owns the arms instead."""
        if self.edge_guard is None:
            self.p.cmd.set_upper_body(self._walk_q.copy())
            self.p.cmd.set_upper_body_vel(np.zeros(17))

    def walk_tick(self, state, *, edge_distance=None, navigation_heading=None, navigation_target=None):
        """Hold the hanging arms for the whole walk; only the sweep and the target observer run.

        The crate pattern: nothing reaches while the base moves. The hand-follow
        of the earlier walk-in was the phase the 4-5 cm tracking wander defeated,
        and it put the arm's first motion on a timer instead of on what the
        cameras see. The arm now moves after the stop, from a state the ego
        camera distinguishes: table edge at the bottom of the frame, bottle at
        the stand-off.
        """
        stamp = float(state["t"])
        if self.walking.last == stamp:
            return self.error is None
        self.walking.last = stamp
        if edge_distance is not None:
            self._edge_anchor = (float(edge_distance), np.asarray(self.p.pelvis_xy(state), dtype=float).copy())
        target = self.target.observe(state)
        # The arms hold the walking hang taken in prepare_walk: nothing moves during the walk.
        self.error = self._clear(state, np.asarray(state["joints"], dtype=float), True)
        if self.error:
            reference = None if self.edge_guard is not None else self._walk_q.copy()
            self.stats["failure"] = dict(state=state, reference=reference, reason=self.error)
            return False
        self._hold_walk_arms()
        if stamp - self._walk_logged >= 1.0 / self.cfg.walk_plan_hz:
            self._walk_logged = stamp
            self.stats["walking"].append(
                dict(
                    t=stamp,
                    target=None if target is None else [float(v) for v in target],
                    ego_px=self.target.diagnostics.get("ego_height_px"),
                    edge_distance=None if edge_distance is None else float(edge_distance),
                    chosen_stop_depth=self.walking.chosen_stop_depth,
                    chosen_stop_edge=self.walking.chosen_stop_edge,
                )
            )
        return True

    def settle_for_admission(self):
        """Hold the hanging arms until the planner's stop has ended: legs and base at rest.

        The takeover used to follow the stop request by about a second, which
        is exactly when the gait is frozen mid-step; the expert cannot step,
        so the feet stayed one ahead of the other for the whole grasp
        (stage4_hang8f review, 2026-09-16). Only the planner can square the
        stance, so wait for it. A stance that does not settle in time is
        recorded, not failed.
        """
        p, simple = self.p, self.simple
        previous = None
        quiet, began = 0.0, None
        leg_speed = pelvis_speed = float("nan")
        still = False
        while True:
            state = p.obs()
            stamp = float(state["t"])
            if previous is not None and stamp == previous[0]:
                time.sleep(0.005)
                continue
            joints = np.asarray(state["joints"], dtype=float)
            pelvis = np.asarray(state["pelvis"]["pos"], dtype=float)
            self.target.observe(state)  # keep the camera observer fed while standing
            if self.edge_guard is not None:
                self.edge_guard.band_gap(state)  # the step after IDLE is where the band front comes closest
            if began is None:
                began = stamp
            if previous is not None:
                dt = stamp - previous[0]
                if dt <= 0 or dt > 0.5:
                    self.stats["stand"] = dict(settled=False, reason="clock gap", dt=dt)
                    return True
                leg_speed = float(np.max(np.abs(joints[:12] - previous[1][:12])) / dt)
                pelvis_speed = float(np.linalg.norm(pelvis[:2] - previous[2][:2]) / dt)
                still = leg_speed <= simple.stand_leg_speed_rad_s and pelvis_speed <= simple.stand_pelvis_speed_m_s
                quiet = quiet + dt if still else 0.0
            previous = (stamp, joints, pelvis)
            self._hold_walk_arms()
            settled = still and quiet >= simple.stand_settle_s
            if settled or stamp - began >= simple.stand_settle_timeout_s:
                if self.edge_guard is not None:
                    # Written here, not at the takeover: the stand gates after the approach can end the round first.
                    self.edge_guard.finish(state)
                    self.stats["guard"] = self.edge_guard.record()
                feet = None
                try:
                    feet = stance_geometry(self.ik, joints)
                except KeyError:
                    pass
                self.stats["stand"] = dict(
                    settled=bool(settled),
                    seconds=float(stamp - began),
                    leg_speed_rad_s=leg_speed,
                    pelvis_speed_m_s=pelvis_speed,
                    feet=feet,
                    edge_distance=None if self._edge_anchor is None else self._edge_anchor[0],
                )
                return True

    # ------------------------------------------------------------------ helpers
    def _clear(
        self, state, candidate, contact, candidate_base=None, *, attached=False, sides=("left", "right", "body")
    ):
        for side in sides:
            hit = self.guard.check(
                state,
                candidate,
                side,
                allow_target_contact=contact and side == "right",
                candidate_base=candidate_base,
                ignore_target=attached and side == "right",
            )
            if not hit.clear:
                return f"simple clearance: {side}:{hit.link}:{hit.obstacle}"
        return None

    @staticmethod
    def _measured_tools(state):
        return np.array(
            [
                np.asarray(state["palms"][side]["pos"], dtype=float)
                + Primitives._rot(state["palms"][side]["quat"]) @ OFFSET
                for side in ("left", "right")
            ]
        )

    def _fail(self, reason, **evidence):
        self.stats["failure"] = dict(reason=reason, **evidence)
        return False, reason

    # ------------------------------------------------------------------ grasp
    def execute(self, *, grip_world, squat_hint, ghost):
        ok, why = False, "simple expert did not start"
        try:
            ok, why = self._run()
        finally:
            self._release(ok)
        self.completed = ok
        self.stats["outcome"] = dict(ok=ok, reason=why)
        # The round record is written with the plain json module.
        for key in list(self.stats):
            self.stats[key] = native(self.stats[key])
        return ok, why

    def _release(self, ok):
        if self.arm is None or self._stream_released:
            return
        try:
            if not ok:
                self.arm._push(self.arm.q, reference_velocity=np.zeros(29))
            self.execution.enter("simple.handback")
            self.arm.stop(preserve_reference=True)
        finally:
            self._stream_released = True

    def prepare_carry(self):
        return self.completed

    def close(self):
        self.target.close()

    def _plan(self, state, forward):
        """Grip, wrap direction and pre-grasp from the current measured state."""
        grip = np.asarray(self.task.grip(state), dtype=float)
        direction = wrap_direction(
            self.p.pelvis_xy(state), grip, forward, self.simple.wrap_yaw_per_m, self.simple.wrap_yaw_max_rad
        )
        pregrasp = pregrasp_point(grip, direction, self.simple.pregrasp_back_m, self.simple.pregrasp_up_m)
        return grip, direction, pregrasp

    def _lift_ready(self):
        """approach_stop guard, right after the takeover: the warehouse's lifted ready (crate/expert.py
        prepare_stationary_ready). One whole-arm pose solved from the walking arms (BimanualIK.lifted_ready), then a
        smoothstep joint blend with the waist locked and the legs and root held, so the forearms fold up close to the
        body. A straight Cartesian rise kept the hands where they hung and swung the shoulders out instead (53-68 deg,
        elbows 13-16 cm outboard; user review of bottle_simple_guard_straight_20260925). None once the hands are up,
        else the failed round."""
        p, simple = self.p, self.simple
        solver = BimanualIK(
            self.ik,
            waist_pitch_max=self.cfg.waist_pitch_max,
            waist_dofs="all",
            elbow_min_deg=simple.ready_elbow_min_deg,
        )
        start = np.asarray(self.arm.q, dtype=float).copy()
        tool = (simple.ready_tool_forward_m, simple.ready_tool_lateral_m, simple.ready_tool_up_m)
        ready = solver.lifted_ready(tool, simple.ready_pitch_deg, start)
        record = self.stats["ready"] = dict(
            tool=list(tool),
            pitch_deg=simple.ready_pitch_deg,
            errors_m=ready.errors.tolist(),
            start=start[12:29].tolist(),
            reference=ready.joints[12:29].tolist(),
        )
        if float(np.max(ready.errors)) > 0.02:
            # The warehouse stops the run here (the pose must be delivered, not approximated); a stand is per round.
            return self._fail("simple lifted ready pose missed by the solver", errors_m=ready.errors.tolist())
        key = self.task.target.key
        delta = ready.joints - start
        state = p.obs()
        obj0 = np.asarray(state["objects"][key]["pos"], dtype=float).copy()
        began = previous = float(state["t"])
        self.stats["phases"].append(dict(phase="ready", t=began, seq=state.get("seq")))
        self.execution.enter("simple.ready")
        wall_progress = time.monotonic()
        while True:
            state = p.obs()
            stamp = float(state["t"])
            if stamp == previous:
                if time.monotonic() - wall_progress > 2.0:
                    return self._fail("simple physical clock stalled", phase="ready")
                time.sleep(0.005)
                continue
            wall_progress = time.monotonic()
            if stamp < previous or stamp - previous > 0.5:
                return self._fail("simple physical state clock gap", dt=stamp - previous, phase="ready")
            previous = stamp
            if state.get("fallen", False):
                return self._fail("simple fell during the ready lift")
            moved = float(np.linalg.norm(np.asarray(state["objects"][key]["pos"])[:2] - obj0[:2]))
            if moved > simple.disturb_limit_m:
                return self._fail("simple bottle knocked before the close", moved_m=moved, phase="ready")
            u = min(1.0, (stamp - began) / simple.ready_seconds)
            q = start + (3 * u * u - 2 * u**3) * delta
            velocity = (6 * u - 6 * u * u) / simple.ready_seconds * delta
            candidate = np.asarray(state["joints"], dtype=float).copy()
            candidate[12:29] = q[12:29]
            reason = self._clear(state, candidate, False)
            if reason:
                return self._fail(reason, phase="ready", progress=u)
            self.arm.q = q
            self.arm._push(q, reference_velocity=velocity, reference_time=stamp)
            if stamp - began >= simple.ready_seconds + simple.ready_hold_s:
                record.update(
                    seconds=stamp - began,
                    tools=self._measured_tools(state).tolist(),
                    joints=np.asarray(state["joints"], dtype=float)[12:29].tolist(),
                )
                return None

    def _run(self):
        p, cfg, simple = self.p, self.cfg, self.simple
        key, side = self.task.target.key, "right"
        state = p.obs()
        seen_at = time.monotonic()
        while self.target.observe(state) is None and self.target.age() > simple.target_grace_s:
            # At the stand the bottle can sit below the ego frame (depth < 0.22 m);
            # what counts is that the cameras admitted it on the way in, within
            # the grace. Otherwise wait for a fresh admission before failing.
            if time.monotonic() - seen_at > simple.target_grace_s:
                return self._fail("simple target not visible in student cameras", camera=dict(self.target.diagnostics))
            self._hold_walk_arms()
            time.sleep(0.02)
            state = p.obs()
        self.stats["target_age_s"] = self.target.age()
        self.arm = StreamedArm(p, side, ik=self.ik, left_guard=False)
        physical = self.ik.model.jnt_range[self.ik.jids]
        self.ik.lo, self.ik.hi = physical[:, 0].copy(), physical[:, 1].copy()
        self.ik.hi[14] = min(self.ik.hi[14], cfg.waist_pitch_max)
        self.arm.preview = PosePreview(
            acceleration=cfg.joint_acceleration,
            joint_limits=(self.ik.lo, self.ik.hi),
            continuous_handoff=True,
        )
        self.execution.enter("simple.takeover")
        self._stream_released = False
        # The streamed arm inherits the commander's cached hand targets, which
        # are the previous round's closed jaw after its handback; the first
        # streamed frames then closed the open jaw and the grasp loop reopened
        # it (user review, 2026-09-16). Nothing is held at the takeover: open.
        self.arm.hands = {"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()}
        p.cmd.set_hands(left=HAND_OPEN.copy(), right=HAND_OPEN.copy())  # the switch replays the last planner packet
        self.arm.start(hold_planner_pose=True, inherit_planner_velocity=True)
        self.stats["takeover"] = dict(switch=dict(self.arm.switch_diagnostics), braking=self.arm.brake_to_stop())

        state = p.obs()
        guard_walk = self.edge_guard is not None
        if guard_walk:
            gap = self.edge_guard.finish(state)
            self.stats["guard"] = self.edge_guard.record()
            if gap < 0:
                # The ready lift starts from here: from past the edge it would sweep the top, and there is no pull-back.
                return self._fail(
                    "simple body past the table edge at the takeover",
                    band_gap_m=gap,
                    body=self.edge_guard.final["band_body"],
                )
            failure = self._lift_ready()
            if failure is not None:
                return failure
            state = p.obs()
        actual = self._measured_tools(state)
        targets = actual.copy()
        posture = np.asarray(self.arm.q, dtype=float).copy()
        posture[12:14] = 0.0
        posture[25:29] = self.walking.comfortable[25:29]
        weights = np.full(29, 0.04)
        weights[12:14] = cfg.comfort_waist_weight
        weights[25:29] = cfg.comfort_elbow_wrist_weight
        motion = SupportedMotion(
            self.ik,
            state,
            cfg,
            self.arm.q,
            self.arm.root_quat,
            tool_offset=OFFSET,
            hand_weights=(0.35, 1.0),
            posture_reference=posture,
            posture_weights=weights,
        )
        heading = motion.body.heading
        forward = heading[:, 0]
        grip, direction, pregrasp = self._plan(state, forward)
        rotations = np.array([Primitives._rot(state["palms"]["left"]["quat"]), grasp_rotation(direction)])
        obj0 = np.asarray(state["objects"][key]["pos"], dtype=float).copy()
        grip_offset = grip - obj0
        start_tool = actual[1].copy()
        pelvis_xy = np.asarray(p.pelvis_xy(state), dtype=float)
        table_top = float(self.task.target.surface_top)
        height = raise_height(grip[2], table_top, simple.raise_above_grip_m, simple.raise_table_margin_m)
        raised = raise_point(start_tool, forward, height, simple.raise_forward_m)
        hand_ahead = float((start_tool[:2] - pelvis_xy) @ forward[:2]) + HAND_FORWARD_EXTENT_M
        # The walk's last edge distance predates the stop overshoot (up to 0.26 m):
        # carry it forward by the base motion since that tick.
        pelvis_edge = (
            float("nan")
            if self._edge_anchor is None
            else self._edge_anchor[0] - float((pelvis_xy - self._edge_anchor[1]) @ forward[:2])
        )
        hand_edge = pelvis_edge - hand_ahead
        retract = (
            not guard_walk
            and self._edge_anchor is not None
            and needs_retract(
                start_tool[2], hand_edge, table_top, simple.raise_table_margin_m, simple.retract_edge_margin_m
            )
        )
        # A hand at the edge backs off at constant speed first (a minimum-jerk
        # start left it 5 ticks at rest while the wander closed the last 4 mm,
        # stage4_wrap8b); the continuous reach starts from the backed-off point.
        backoff_to = start_tool - simple.retract_m * forward if retract else start_tool
        raised = raise_point(backoff_to, forward, height, simple.raise_forward_m, simple.raise_outboard_m)
        path = [backoff_to, raised, pregrasp]
        ready = None
        if guard_walk:
            # The hands are already up (_lift_ready): the right one slides on from where it measured, the left one
            # holds its ready point for the whole grasp.
            ready = actual.copy()
            raised = actual[1].copy()
            path = [raised, pregrasp]
        # Fingers stay down while the hand goes up (a vertical lift of the whole
        # hand, no palm swing); they turn level on the slide to the pre-grasp.
        hanging_rotation = Primitives._rot(state["palms"][side]["quat"])
        rotations[1] = hanging_rotation
        reach_seconds = simple.pregrasp_seconds if guard_walk else simple.raise_seconds + simple.pregrasp_seconds
        raise_arc = polyline_length(path[:-1])
        self.stats["plan"] = dict(
            grip=grip.tolist(),
            direction=direction.tolist(),
            wrap_yaw_rad=float(np.arctan2(direction[1], direction[0]) - np.arctan2(forward[1], forward[0])),
            raised=raised.tolist(),
            pregrasp=pregrasp.tolist(),
            start_tool=start_tool.tolist(),
            pelvis=[float(v) for v in state["pelvis"]["pos"]],
            depth=float((grip[:2] - pelvis_xy) @ forward[:2]),
            lean=lean_angle(state["pelvis"]["pos"], grip, simple.lean_max_rad, forward),
            table_top=table_top,
            pelvis_edge_m=pelvis_edge,
            hand_edge_m=hand_edge,
            retract=bool(retract),
            ready=None if ready is None else ready.tolist(),
            path=[q.tolist() for q in path],
            reach_seconds=reach_seconds,
        )
        phase, phase_began = ("backoff" if retract else "reach"), float(state["t"])
        self.execution.enter(f"simple.{phase}")
        closure, lean, hold_offset, lean_at_lift = 0.0, 0.0, None, 0.0
        attempts, retreat_from, retreat_to, retry_shift = 1, None, None, np.zeros(3)
        replanned = False
        entry_cmd, shift = None, np.zeros(3)
        self.stats["entries"] = []
        previous = None
        began = float(state["t"])
        last_sample = -np.inf
        wall_progress = time.monotonic()
        while True:
            state = p.obs()
            stamp = float(state["t"])
            if previous is not None and stamp == previous:
                if time.monotonic() - wall_progress > 2.0:
                    return self._fail("simple physical clock stalled")
                time.sleep(0.005)
                continue
            wall_progress = time.monotonic()
            dt = 0.02 if previous is None else stamp - previous
            if dt < 0 or dt > 0.5:
                return self._fail("simple physical state clock gap", dt=dt)
            previous = stamp
            if state.get("fallen", False):
                return self._fail("simple fell during the grasp")
            if stamp - began > self.config.recovery_timeout:
                return self._fail(f"simple {phase} timed out")
            obj = state["objects"][key]
            actual = self._measured_tools(state)
            bilateral = bilateral_dex1_contact(obj, side)
            elapsed = stamp - phase_began
            if phase in ("backoff", "reach", "settle", "entry", "retreat"):
                knocked = float(np.linalg.norm(np.asarray(obj["pos"])[:2] - obj0[:2]))
                if knocked > simple.disturb_limit_m:
                    return self._fail("simple bottle knocked before the close", moved_m=knocked, phase=phase)

            def enter(name):
                nonlocal phase, phase_began
                self.stats["phases"].append(dict(phase=name, t=stamp, seq=state.get("seq")))
                self.execution.enter(f"simple.{name}")
                phase, phase_began = name, stamp

            if phase == "backoff":
                # Both hands hang at the edge; the left one stays back for the grasp.
                step = min(simple.retract_speed_m_s * dt, float((targets[1] - backoff_to) @ forward))
                targets[1] = targets[1] - max(0.0, step) * forward
                targets[0] = targets[0] - max(0.0, step) * forward
                if float((targets[1] - backoff_to) @ forward) <= 1e-6:
                    targets[1] = backoff_to.copy()
                    enter("reach")
            elif phase == "reach":
                # One continuous path: (pull-back,) raise, slide to the pre-grasp,
                # with a single minimum-jerk arc-length profile so the hand does
                # not stop at the corners.
                arc = moving_start(elapsed / reach_seconds) * polyline_length(path)
                if not replanned and arc >= raise_arc:
                    # Re-plan from the settled base once the hand is up: the
                    # brake's forward creep (1-2 cm over the first second) is over.
                    replanned = True
                    grip, direction, pregrasp = self._plan(state, forward)
                    rotations[1] = grasp_rotation(direction)  # fingers turn level on the slide
                    path[-1] = pregrasp
                    self.stats["plan"]["replanned"] = dict(
                        t=stamp,
                        grip=grip.tolist(),
                        direction=direction.tolist(),
                        pregrasp=pregrasp.tolist(),
                        tool=actual[1].tolist(),
                        pelvis=[float(v) for v in state["pelvis"]["pos"]],
                    )
                targets[1] = polyline_point(path, arc)
                lean = lean_angle(state["pelvis"]["pos"], grip, simple.lean_max_rad, forward)
                if elapsed >= reach_seconds:
                    enter("settle")
            elif phase == "settle":
                # Only if the hand has not caught up with the command (coarse, centimetres).
                settled = float(np.linalg.norm(actual[1] - targets[1])) <= simple.settle_tolerance_m
                if settled or elapsed >= simple.settle_timeout_s:
                    self.stats["settle"] = dict(
                        t=stamp, settled=settled, error_m=float(np.linalg.norm(actual[1] - targets[1]))
                    )
                    entry_cmd, shift = targets[1].copy(), np.zeros(3)
                    enter("entry")
            elif phase == "entry":
                palm_rotation = Primitives._rot(state["palms"][side]["quat"])
                # The grip sits `grip_below_center` off the bottle centre by design:
                # align on the grip point, not the centre (stage4_wrap8b: -3 cm
                # of intended height offset failed the 3 cm window).
                offset = object_in_palm(palm_rotation, actual[1], np.asarray(obj["pos"], dtype=float) + grip_offset)
                aligned = entry_aligned(offset, simple.entry_lateral_tolerance_m, simple.entry_height_tolerance_m)
                # One advance at the dash speed with the jaw-axis correction
                # riding along; close on arrival or on contact.
                entry_cmd = entry_command(
                    entry_cmd, grip, direction, simple.dash_speed_m_s, dt, simple.entry_overshoot_m
                )
                shift = lateral_correction(
                    shift,
                    palm_rotation,
                    offset,
                    simple.lateral_correction_speed_m_s,
                    dt,
                    simple.lateral_correction_max_m,
                )
                targets[1] = entry_cmd + shift
                arrived = entry_captured(actual[1], grip, direction, simple.entry_capture_depth_m)
                touching = finger_touching(obj, side)
                if arrived or bilateral:
                    self.stats["entries"].append(
                        dict(
                            t=stamp,
                            attempt=attempts,
                            arrived=arrived,
                            touching=touching,
                            bilateral=bilateral,
                            offset_in_palm=offset.tolist(),
                            aligned=aligned,
                            shift_m=float(np.linalg.norm(shift)),
                        )
                    )
                    if aligned or bilateral:
                        # Hold where the hand is: the frozen compensation keeps
                        # the reference continuous, the closure does the rest.
                        # Hold the measured tool through the closure with the
                        # translation compensation still active: frozen at the
                        # close, the 3-4 cm the hand sags while moving came back
                        # as a rise once it stopped, and the pads closed 6 cm
                        # above the centre (stage4_wrap8d). Rotation is frozen.
                        targets[:] = actual
                        motion.freeze_rotation_feedback()
                        enter("close")
                    elif attempts >= simple.entry_attempts:
                        return self._fail(
                            "simple entry could not align the jaw on the bottle",
                            offset_in_palm=offset.tolist(),
                            attempts=attempts,
                        )
                    else:
                        # Last resort: back straight off (a sideways shift at the
                        # bottle knocked it over, stage4_wrap8), then re-enter
                        # with the measured jaw-axis offset as the starting correction.
                        attempts += 1
                        retreat_from = targets[1].copy()
                        retreat_to = retreat_from - simple.retreat_m * direction
                        retry_shift = lateral_correction(
                            np.zeros(3), palm_rotation, offset, 1.0, 1.0, simple.lateral_correction_max_m
                        )
                        enter("retreat")
                elif elapsed > simple.entry_timeout_s:
                    return self._fail(
                        "simple entry timed out without capture",
                        depth_m=float((grip - actual[1]) @ direction),
                        tool=actual[1].tolist(),
                    )
            elif phase == "retreat":
                targets[1] = retreat_from + minjerk(elapsed / simple.retreat_seconds) * (retreat_to - retreat_from)
                if elapsed >= simple.retreat_seconds:
                    entry_cmd, shift = targets[1].copy(), retry_shift.copy()
                    enter("entry")
            elif phase == "close":
                closure = min(1.0, elapsed / simple.close_seconds)
                if elapsed >= simple.close_seconds:
                    if bilateral:
                        motion.freeze_translation_feedback()
                        hold_offset = torso_offset(self.ik, motion, targets[1])
                        lean_at_lift = lean
                        motion.resume_rotation_feedback()
                        self.stats["capture"] = dict(t=stamp, tool=actual[1].tolist(), object=list(obj["pos"]))
                        enter("lift")
                    elif elapsed >= simple.close_seconds + simple.contact_timeout_s:
                        return self._fail("simple no bilateral pad contact after the close")
            elif phase == "lift":
                targets[1] = lift_target(
                    self.ik, motion, hold_offset, simple.lift_height_m * minjerk(elapsed / simple.lift_seconds)
                )
                if elapsed >= simple.lift_seconds:
                    enter("recover")
            elif phase == "recover":
                lean = lean_at_lift * (1.0 - minjerk(elapsed / simple.recover_seconds))
                targets[1] = lift_target(self.ik, motion, hold_offset, simple.lift_height_m)
                if elapsed >= simple.recover_seconds:
                    rise = float(obj["pos"][2] - obj0[2])
                    self.stats["upright"] = dict(t=stamp, rise_m=rise, bilateral=bilateral, lean=lean)
                    if not bilateral:
                        return self._fail("simple lost pad contact during the lift", rise_m=rise)
                    if rise < simple.min_rise_m:
                        return self._fail("simple lift did not raise the bottle", rise_m=rise)
                    return True, "simple grasp lifted and upright"

            torso = heading @ Rotation.from_euler("y", lean).as_matrix()
            previous_x, previous_root = motion.x.copy(), motion.root.copy()
            pose = motion.step(
                targets,
                rotations,
                actual,
                dt,
                torso_rotation=torso,
                measured_rotations=np.array([Primitives._rot(state["palms"][s]["quat"]) for s in ("left", "right")]),
            )
            predicted = deepcopy(state)
            predicted["joints"] = np.asarray(state["joints"], dtype=float) + pose.joints - previous_x[4:]
            delta_rotation = Primitives._rot(motion.root) @ Primitives._rot(previous_root).T
            predicted["pelvis"] = dict(
                state["pelvis"],
                pos=np.asarray(state["pelvis"]["pos"], dtype=float) + pose.position - previous_x[:3],
                quat=Rotation.from_matrix(delta_rotation @ Primitives._rot(state["pelvis"]["quat"])).as_quat()[
                    [3, 0, 1, 2]
                ],
            )
            near_grip = abs(float(actual[1][2] - grip[2])) < CAP_CLEARING_CONTACT_HEIGHT_M
            # The back-off starts from a hand that may already be within the
            # sweep margin of the table (that is why it backs off); the sweep
            # flags the start pose whatever the direction, so only the body is
            # screened while the hands move straight away from the edge.
            reason = self._clear(
                state,
                predicted["joints"],
                phase in ("settle", "entry", "retreat", "close") or near_grip,
                candidate_base=[*predicted["pelvis"]["pos"], *predicted["pelvis"]["quat"]],
                attached=phase in ("lift", "recover"),
                # During the closure the hand is held where it is and the bottle
                # stands on the table: a finger brushing the top as the tool sags
                # (stage4_wrap8i round 4, 5 cm) is not a transport collision.
                sides=("body",)
                if phase == "backoff"
                else ("left", "body")
                if phase == "close"
                else ("left", "right", "body"),
            )
            if reason:
                return self._fail(reason, phase=phase, tool=actual[1].tolist(), target=targets[1].tolist())
            if max(pose.foot_errors) > 0.025 or float(np.linalg.norm(pose.com_shift)) > 0.04:
                return self._fail(
                    "simple support reference outside tolerance",
                    foot_error=[float(v) for v in pose.foot_errors],
                    com_shift=[float(v) for v in pose.com_shift],
                    phase=phase,
                )
            self.arm.q, self.arm.root_quat = pose.joints, motion.root
            self.arm.hands[side] = HAND_OPEN + closure * (HAND_CLOSED_RIGHT - HAND_OPEN)
            self.arm._push(
                pose.joints,
                reference_velocity=motion.v[4:],
                reference_time=stamp,
                root_angular_velocity=Rotation.from_matrix(delta_rotation).as_rotvec() / dt,
            )
            if stamp - last_sample >= 0.1:
                last_sample = stamp
                self.stats["samples"].append(
                    dict(
                        seq=state.get("seq"),
                        t=stamp,
                        phase=phase,
                        closure=closure,
                        lean=lean,
                        tool=actual[1].tolist(),
                        target=targets[1].tolist(),
                        object=[float(v) for v in obj["pos"]],
                        bilateral=bilateral,
                        touching=finger_touching(obj, side),
                        feedback=float(np.linalg.norm(motion.feedback[1])),
                        rotation_feedback=float(np.linalg.norm(motion.rotation_feedback[1])),
                    )
                )

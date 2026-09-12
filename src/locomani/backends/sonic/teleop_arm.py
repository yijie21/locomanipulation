"""`TeleopArm`: the arm channel the SONIC checkpoints were built for.

The official PICO "bend and grasp" flow (`PLANNER_VR_3PT`) never solves an arm
IK. It streams three POSES in the pelvis frame -- left wrist point, right
wrist point, neck point -- inside the ordinary planner message, and the deploy
switches its encoder to mode 1 ("teleop"), whose inputs are the k-planner's
LOWER-body reference plus those three targets (observation_config.yaml,
`encoder_modes[1]`). The policy itself puts the hands there, waist and steps
included; `sonic_v1_1` was trained with `randomize_wrist_poses: true` on
exactly this interface (policy/sonic_v1_1/config.yaml).

Everything in `PlannerArm` / `StreamedArm` -- our own DLS IK, the frozen
17-joint pin, the waist locked at zero, the predicted squat retreat -- is a
re-implementation of the part the checkpoint provides, calibrated on the
release checkpoint's tracking errors (ledger bottle.md 21-34, memory
v11-swap-is-an-interface-mismatch). This class replaces that layer with what
the human teleoperator does: command the wrist pose, watch where the hand
went, and move the command.

Geometry, from the vendor (gear_sonic_deploy g1_deploy_onnx_ref.cpp:1150-1210,
vr3pt_pose_visualizer.py:79-95, gear_sonic mdp/observations.py:1348-1442):

  point_i  = R_pelvis^T (p_link_i + R_link_i @ OFFSET_i - p_pelvis)
  quat_i   = inv(q_pelvis) * q_link_i                      (w, x, y, z)
  OFFSET   = left_wrist_yaw_link  [0.18, -0.025, 0]
             right_wrist_yaw_link [0.18, +0.025, 0]
             torso_link           [0, 0, 0.35]

with the FULL pelvis rotation (not heading-only) on both position and
orientation. `state["palms"][side]` IS the wrist_yaw_link pose
(privileged_pub.PALM_BODY), so the measured point needs no FK; the neck does
(torso_link, via `ArmIK.body_pose`).

Protocol facts that decide whether the channel is live at all (deploy source,
all three built trees carry it):
  * the field's PRESENCE is the switch -- see `AgentCommander.vr_targets`;
  * the first target must be the MEASURED pose (the pico manager recalibrates
    against the robot's FK on every entry, :2094-2096), or the arm jumps;
  * a safety reset drops the mode back to "g1" (zmq_manager.hpp:498), so the
    channel is entered per episode, after `start_control`, never held across.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.primitives import (
    DT,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    HAND_OPEN,
    LocomotionMode,
    Primitives,
    StreamMode,
    upper_body_from_mj,
)
from locomani.primitives.arm_ik import ArmIK

VR_OFFSET = {
    "left": np.array([0.18, -0.025, 0.0]),
    "right": np.array([0.18, 0.025, 0.0]),
}
NECK_OFFSET = np.array([0.0, 0.0, 0.35])
TORSO_BODY = "torso_link"

# Feedback on the servoed side: command = target + KP * clip(target - measured).
# Proportional on the CURRENT residual only -- accumulating corrections
# diverged on transients in the streamed channel (memory
# sonic-streamed-motion-channel). KP=0 is the open-loop control leg.
KP = 0.6
CLIP_M = 0.06
# Same idea for the attitude: the probe delivered 0.88 of a 60 deg turn (2.2 deg
# residual at rest, 7-10 deg while moving), and closure_geometry's jaw-tilt gate
# is 0.30 rad -- the first pilot rounds sat at 0.30-0.33 on the measured palm
# with the target upright, so the residual is served too. rotvec, rad.
KP_ROT = 0.6
CLIP_RAD = 0.35
# Cartesian rate limit on the commanded target, m/s and rad/s. The human
# moves a controller at this order; asking the policy to jump is what the
# pico manager's per-entry calibration exists to avoid.
RATE_M = 0.25
RATE_RAD = 1.2
ENTRY_HOLD_S = 0.4  # frames at the measured pose before anything moves


def _mat2quat_wxyz(rot) -> np.ndarray:
    q = Rotation.from_matrix(np.asarray(rot, dtype=float)).as_quat()  # x, y, z, w
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def vr_point(pos_w, rot_w, pelvis_pos, pelvis_rot, offset):
    """World link pose -> (point, rot) in the pelvis frame, offset applied in the link frame."""
    rot_pf = np.asarray(pelvis_rot).T @ np.asarray(rot_w)
    pf = np.asarray(pelvis_rot).T @ (np.asarray(pos_w, dtype=float) - np.asarray(pelvis_pos, dtype=float))
    return pf + rot_pf @ np.asarray(offset), rot_pf


class TeleopArm:
    """Own the VR 3-point channel: world-frame wrist targets, pelvis-frame commands."""

    def __init__(self, p: Primitives, side: str, kp: float = KP, clip_m: float = CLIP_M, kp_rot: float = KP_ROT):
        self.p, self.side = p, side
        self.kp, self.clip_m = float(kp), float(clip_m)
        self.kp_rot = float(kp_rot)
        self.ik = ArmIK()  # FK only here (torso point); no solve
        self.mode = LocomotionMode.IDLE
        self.height = -1.0
        self._face = (1.0, 0.0, 0.0)
        # world wrist_yaw_link pose per side: {"pos": (3,), "rot": (3,3)}
        self.target: dict[str, dict] = {}
        self.torso_pf0 = None  # (pos, rot) of torso_link in the pelvis frame at entry
        self.lean = 0.0  # rad, forward pitch asked of the neck point (0 = upright)
        self.hands = {
            "left": np.array(p.cmd.left_hand, dtype=np.float32),
            "right": np.array(p.cmd.right_hand, dtype=np.float32),
        }
        self.next_t = None
        self.i = 0
        self.stats = dict(ticks=0, err_sum=0.0, err_max=0.0)

    # -- plumbing ----------------------------------------------------------

    def start(self) -> dict:
        """Enter the channel at the measured pose. -> the entry observation."""
        p = self.p
        s = p.obs()
        self._face = p._facing(s)
        for sd in ("left", "right"):
            palm = s["palms"][sd]
            self.target[sd] = dict(pos=np.array(palm["pos"], dtype=float), rot=Primitives._rot(palm["quat"]))
        self.torso_pf0 = self.ik.body_pose(np.asarray(s["joints"], dtype=float), TORSO_BODY)
        self.lean = 0.0
        # Drop the frozen upper-body pin: with `upper_body_position` present the
        # deploy substitutes those joints into the reference (cpp:780) -- the
        # pico manager sends one field or the other, never both.
        p.cmd.upper_body = None
        p.cmd.set_upper_body_vel(None)
        p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
        self._write_targets(s, feedback=False)
        p.cmd.vr_targets = True  # the rising edge: encoder -> "teleop"
        self.next_t = time.time()
        self.hold(ENTRY_HOLD_S)
        return s

    def stop(self) -> None:
        """Leave the channel: pin the upper body where it is, drop the VR fields."""
        p = self.p
        if self.mode != LocomotionMode.IDLE:
            self.stand()
        s = p.obs()
        p.cmd.set_upper_body(upper_body_from_mj(s["joints"]))
        p.cmd.set_upper_body_vel(np.zeros(17, dtype=np.float32))
        p.cmd.set_hands(left=self.hands["left"], right=self.hands["right"])
        p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
        p.cmd.vr_targets = False  # the falling edge: encoder -> "g1"
        p.cmd.hold(0.2, mode=LocomotionMode.IDLE, facing=self._face)

    def measured_point(self, sd: str, s: dict):
        palm = s["palms"][sd]
        return vr_point(
            palm["pos"],
            Primitives._rot(palm["quat"]),
            s["pelvis"]["pos"],
            Primitives._rot(s["pelvis"]["quat"]),
            VR_OFFSET[sd],
        )

    def target_point(self, sd: str, s: dict):
        t = self.target[sd]
        return vr_point(t["pos"], t["rot"], s["pelvis"]["pos"], Primitives._rot(s["pelvis"]["quat"]), VR_OFFSET[sd])

    def neck_point(self):
        """Neck target in the pelvis frame: the entry torso pose pitched forward by `lean`."""
        pos0, rot0 = self.torso_pf0
        lean = Rotation.from_rotvec([0.0, self.lean, 0.0]).as_matrix()
        rot = lean @ rot0
        return lean @ pos0 + rot @ NECK_OFFSET, rot

    def _write_targets(self, s: dict, feedback: bool = True) -> float:
        """Set the commander's three targets from the world targets. -> servo residual (m)."""
        err = 0.0
        for sd in ("left", "right"):
            pt, rot_pf = self.target_point(sd, s)
            if feedback and sd == self.side and (self.kp > 0 or self.kp_rot > 0):
                meas, meas_rot = self.measured_point(sd, s)
                resid = pt - meas
                err = float(np.linalg.norm(resid))
                pt = pt + self.kp * np.clip(resid, -self.clip_m, self.clip_m)
                if self.kp_rot > 0:
                    # rotation still needed, in the pelvis frame: target * measured^T
                    rv = Rotation.from_matrix(rot_pf @ meas_rot.T).as_rotvec()
                    n = float(np.linalg.norm(rv))
                    if n > CLIP_RAD:
                        rv *= CLIP_RAD / n
                    rot_pf = Rotation.from_rotvec(self.kp_rot * rv).as_matrix() @ rot_pf
            self.p.cmd.set_wrists(**{f"{sd}_pos": pt.astype(np.float32), f"{sd}_quat": _mat2quat_wxyz(rot_pf)})
        neck_pos, neck_rot = self.neck_point()
        self.p.cmd.set_neck(neck_pos.astype(np.float32), _mat2quat_wxyz(neck_rot))
        return err

    def tick(self, s: dict | None = None, *, movement=(0.0, 0.0, 0.0), speed: float = -1.0) -> dict:
        """One 50 Hz frame: re-express the world targets in the live pelvis frame and send."""
        s = s or self.p.obs()
        err = self._write_targets(s)
        self.p.cmd.set_hands(left=self.hands["left"], right=self.hands["right"])
        self.p.cmd.frame(mode=self.mode, facing=self._face, height=self.height, movement=movement, speed=speed)
        self.stats["ticks"] += 1
        self.stats["err_sum"] += err
        self.stats["err_max"] = max(self.stats["err_max"], err)
        self.i += 1
        self.next_t += DT
        lag = self.next_t - time.time()
        if lag > 0:
            time.sleep(lag)
        else:
            self.next_t = time.time()
        return s

    def hold(self, seconds: float) -> None:
        for _ in range(int(round(seconds / DT))):
            self.tick()

    # -- behaviours --------------------------------------------------------

    def move_to(self, pos_w, rot_w=None, *, side: str | None = None, rate_m: float = RATE_M, settle: float = 0.0):
        """Slide the world target to a pose at a bounded rate, ticking all the way."""
        sd = side or self.side
        pos1 = np.asarray(pos_w, dtype=float)
        t = self.target[sd]
        pos0, rot0 = t["pos"].copy(), t["rot"].copy()
        rot1 = rot0 if rot_w is None else np.asarray(rot_w, dtype=float)
        d_pos = float(np.linalg.norm(pos1 - pos0))
        d_rot = float(np.linalg.norm(Rotation.from_matrix(rot0.T @ rot1).as_rotvec()))
        dur = max(d_pos / rate_m, d_rot / RATE_RAD, DT)
        n = max(1, int(round(dur / DT)))
        r0, r1 = Rotation.from_matrix(rot0), Rotation.from_matrix(rot1)
        rv = (r0.inv() * r1).as_rotvec()
        for k in range(1, n + 1):
            u = k / n
            sm = 3 * u * u - 2 * u**3
            t["pos"] = pos0 + sm * (pos1 - pos0)
            t["rot"] = (r0 * Rotation.from_rotvec(sm * rv)).as_matrix()
            self.tick()
        if settle > 0:
            self.hold(settle)

    def residual(self, s: dict | None = None, side: str | None = None) -> float:
        """Distance between the target point and the measured point, m."""
        s = s or self.p.obs()
        sd = side or self.side
        return float(np.linalg.norm(self.target_point(sd, s)[0] - self.measured_point(sd, s)[0]))

    def squat(self, drop_m: float, settle: float = 1.6) -> float:
        """Crouch through IDEL_SQUAT while the wrist targets stay WORLD-fixed. -> delivered drop.

        The legs are the k-planner's in this channel too (encoder mode 1 reads
        the planner's lower-body reference), so the depth comes from the
        planner message's absolute `height`, as in PlannerArm.squat. The
        difference is the arm: the target is re-expressed in the live pelvis
        frame every tick, so whatever the squat does to the pelvis -- the
        retreat `project_squat` under-predicted by 3.5 cm, the +16 deg pitch it
        predicted as zero (ledger 32/33) -- is compensated by measurement, not
        by a constant.
        """
        from locomani.backends.sonic.streamed_arm import SQUAT_DROP_MAX, SQUAT_HEIGHT_BIAS

        s = self.p.obs()
        z0 = float(s["pelvis"]["pos"][2])
        target_z = z0 - float(np.clip(drop_m, 0.0, SQUAT_DROP_MAX))
        if z0 - target_z < 0.02:
            return 0.0
        self.height = float(np.clip(target_z - SQUAT_HEIGHT_BIAS, 0.40, 0.78))
        self.mode = LocomotionMode.IDEL_SQUAT
        self.hold(settle)
        return float(z0 - self.p.obs()["pelvis"]["pos"][2])

    def stand(self, settle: float = 1.5) -> None:
        self.mode = LocomotionMode.IDLE
        self.height = -1.0
        self.hold(settle)

    def close_hand(self, obj_key: str, ramp: float = 2.0, squeeze: float = 0.6, full_close: bool = True) -> bool:
        """Close the servoed hand on the object with the servo's proven rule
        (servo_phases._phase_close): a 2 s ramp, contact latched only when
        SUSTAINED for 3 ticks (both dex1 jaws when available), then `squeeze`
        more of the ramp past the latch. The first teleop pilot latched a
        one-tick graze at alpha ~0.1 and stopped at a 6.6 cm jaw gap on a 4.5 cm
        bottle: 'touching', zero clamp, and the lift raised nothing.

        `full_close` (the default, and what the servo does for dex1 whenever
        `require_bilateral` is on) runs the ramp to alpha 1.0 instead of
        stopping `squeeze` past the latch: run 6 round az30:4 stopped early,
        lifted 8.7 cm, and then dropped the bottle 79.9 cm in the palm frame
        during the hold. A partial clamp survives the lift and not the carry.
        """
        from locomani.primitives.contacts import bilateral_dex1_contact

        closed = (HAND_CLOSED_RIGHT if self.side == "right" else HAND_CLOSED_LEFT).astype(np.float32)
        contact_alpha = None
        touch_ticks = 0
        alpha = 0.0
        n = int(round(ramp / DT))
        for k in range(n + 1):
            alpha = k / n
            self.hands[self.side] = HAND_OPEN + alpha * (closed - HAND_OPEN)
            s = self.tick()
            obj = s["objects"][obj_key]
            try:
                contact = bool(bilateral_dex1_contact(obj, self.side))
            except Exception:
                contact = self.side in obj.get("touching", [])
            touch_ticks = touch_ticks + 1 if contact else 0
            if contact_alpha is None and touch_ticks >= 3:
                contact_alpha = alpha
            if not full_close and contact_alpha is not None and alpha >= min(1.0, contact_alpha + squeeze):
                break
        self.hold(0.6)
        self.last_alpha = alpha
        self.contact_alpha = contact_alpha
        return contact_alpha is not None and self.side in self.p.obs()["objects"][obj_key]["touching"]

    def open_hand(self, ramp: float = 0.8) -> None:
        start = self.hands[self.side].copy()
        n = int(round(ramp / DT))
        for k in range(1, n + 1):
            self.hands[self.side] = start + (k / n) * (HAND_OPEN - start)
            self.tick()

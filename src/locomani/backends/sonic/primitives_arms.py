"""`Primitives`'s arms/hands mixin, split out of primitives.py (refactor step
4b family 4, pure extraction; decision 9's file-size cap). Every statement
here is unchanged from the original inline body -- only the packaging moved.
"""

from __future__ import annotations

import time

import numpy as np

from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.backends.sonic.primitives_tuning import (
    DT,
    GRASP_CENTRE_OFFSET,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    NAV_HAND_SPEED,
    REACH_BOX_HI,
    REACH_BOX_LO,
    REACH_GAIN,
    REACH_GLIDE_MIN,
    REACH_MAX_STEP,
    REACH_PROGRESS,
    REACH_QUAT_SETTLE,
    REACH_SEED_HOLD,
    REACH_SETTLE_AFTER_GLIDE,
    REACH_STALL_S,
    REACH_UPDATE_HZ,
    REACH_X_GAIN,
    REACH_X_OFFSET,
    REACH_Z_GAIN,
    REACH_Z_OFFSET,
    SERVO_DEPTH_TOL,
    SERVO_GAIN,
    SERVO_GATE_TOL,
    SERVO_MAX_STEP,
    SERVO_SAFE_GAP,
    _opposed,
)
from locomani.primitives import wrist_map


class PrimitivesArmsMixin:
    """`reach`/`servo_grasp`/`grasp`/`release` -- driving a palm to a world
    point and closing the hand on it."""

    @staticmethod
    def _legacy_seed(local) -> np.ndarray:
        """The hand-fitted inverse, kept only as a fallback.

        Measured against 64 samples it is wrong by 24.6 cm on average and 47.4
        at worst — z has the wrong sign and y is not modelled at all. It stands
        in only until diag_map.py has been run on a given deploy, and anything
        it produces should be treated as a starting guess, not a calibration.
        """
        seed = np.asarray(local, dtype=float).copy()
        seed[0] = (seed[0] + REACH_X_OFFSET) / REACH_X_GAIN
        seed[2] = (seed[2] + REACH_Z_OFFSET) / REACH_Z_GAIN

        return seed

    def reach(
        self,
        world_pos,
        side: str = "right",
        tol: float = 0.05,
        settle: float = 0.8,
        timeout: float = 10.0,
        height: float = -1.0,
        palm_offset=None,
        resume: bool = False,
        quat=None,
        glide: float | None = None,
        speed: float = NAV_HAND_SPEED,
    ) -> bool:
        """Drive the palm to `world_pos` by integrating the measured error.

        Commanding the target directly does not work. Measured on the live
        stack (locomani/diag_reach.py), the wrist target's y and z track to
        within ~3 cm but x lands roughly `0.75 * x - 0.09`: ask for 0.65 m
        ahead and the palm stops at 0.40 m. Raising `vr_3pt_compliance` changes
        nothing at all — 0.5 and 1.0 give identical results to the millimetre,
        so it is not a tracking gain.

        Rather than fit that mapping and re-fit it whenever the pose changes,
        integrate: nudge the commanded point by the error between where the
        palm *is* and where it should be. The privileged stream already carries
        the true palm pose, so the loop closes on the real quantity and absorbs
        whatever the mapping happens to be.

        Two phases, not one. The integrator alone produces the motion the
        videos showed — the commanded point teleports to the seed and the arm
        lunges after it, then holds for REACH_SEED_HOLD, then creeps. The seed
        is computed by dividing by the tracking gains (0.55 in x, 0.50 in z),
        so a 0.20 m change of target becomes a 0.40 m step of command in a
        single tick. So: glide the command from wherever the arm is already
        holding to the seed at a bounded speed, then let the integrator take
        over. The arm follows a trajectory instead of chasing a jump.

        That is not cosmetic. These trajectories are training data, and a
        policy finetuned on lunges will lunge.
        """
        end = time.time() + timeout
        ok_since = None
        # `resume` continues from the command the previous reach left the arm
        # holding, instead of re-seeding from the inverse model. A refinement
        # pass must resume: re-seeding jumps the command discontinuously and
        # REACH_SEED_HOLD then pins the arm at that jump for 2.5 s, which is a
        # swipe, not a correction. Measured, adding a naive second descent pass
        # took "orange knocked away" from 1/10 to 6/10 of episodes.
        cmd_local = self.last_cmd.copy() if (resume and self.last_cmd is not None) else None
        palm_rot = None
        err = float("inf")
        best = float("inf")
        stalled_since = time.time()
        next_log = time.time()
        next_update = time.time() + (0.0 if cmd_local is not None else REACH_SEED_HOLD)
        # Glide phase state. `start` is what the arm is being told to hold
        # right now, which is where a smooth move has to begin.
        start = np.array(self.cmd.right_wrist_pos if side == "right" else self.cmd.left_wrist_pos, dtype=float)
        seed = None
        glide_t0 = glide_s = None
        if quat is not None:
            # Command the orientation and let it ARRIVE before anything else
            # reads the palm. The loop below freezes the palm rotation on its
            # first tick and uses it to turn the grip point into a wrist point;
            # freeze it while the hand is still rotating and the offset is
            # taken from the old orientation, which for the tilted grasp is a
            # 23 deg error on a 0.12 m lever — 5 cm of target, silently.
            self.cmd.set_wrists(**{f"{side}_quat": np.asarray(quat, np.float32)})
            settle_end = time.time() + REACH_QUAT_SETTLE
            while time.time() < settle_end:
                st = self.obs()
                self._maintain_crouch(st)
                self.cmd.frame(mode=LocomotionMode.IDLE, height=height, facing=self._facing(st))
                time.sleep(DT)
            end += REACH_QUAT_SETTLE

        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while reaching"
                return False

            # `world_pos` is where the *grasp centre* should end up, but what
            # the loop can steer is the wrist, 11 cm behind it. Converting one
            # into the other needs the palm's orientation — and that
            # orientation must be FROZEN at the start of the leg, not tracked.
            #
            # Tracking it live is a positive feedback loop, and it is what has
            # been failing every grasp. Measured on the descent: the palm came
            # down 0.167 m while the target it was chasing rose 0.117 m and
            # moved 0.218 m forward, because descending rotates the wrist and
            # rotating the wrist swings the offset. The error sat at 0.217 m
            # for the whole stall window and then reported "out of the arm's
            # workspace", which it never was.
            #
            # Frozen, the target is a fixed point the integrator can converge
            # on. The cost is that any subsequent wrist rotation shows up as a
            # position error of at most the offset length; that is bounded and
            # measurable, whereas a moving target is neither.
            goal = np.asarray(world_pos, dtype=float)
            if palm_offset is not None:
                if palm_rot is None:
                    palm_rot = self._rot(s["palms"][side]["quat"])
                goal = goal - palm_rot @ palm_offset

            desired_local = self.world_to_root_full(goal, s)
            palm_local = self.world_to_root_full(s["palms"][side]["pos"], s)
            if seed is None:
                seed = (
                    wrist_map.command_for(desired_local) if wrist_map.available() else self._legacy_seed(desired_local)
                )
                seed = np.clip(seed, REACH_BOX_LO, REACH_BOX_HI)
                # Time the glide by distance, so a short correction is quick
                # and a long move is not a lunge. Floored, because even a tiny
                # step wants a moment of ramp at each end.
                span = float(np.linalg.norm(seed - start))
                glide_s = glide if glide is not None else max(REACH_GLIDE_MIN, span / max(speed, 1e-3))
                glide_t0 = time.time()
                if cmd_local is None:
                    cmd_local = start.copy()

            if glide_t0 is not None:
                u = (time.time() - glide_t0) / max(glide_s, 1e-3)
                if u < 1.0:
                    # smoothstep: zero velocity at both ends, so the arm eases
                    # out of rest and eases into the target
                    e = u * u * (3.0 - 2.0 * u)
                    cmd_local = start + (seed - start) * e
                else:
                    cmd_local = seed.copy()
                    glide_t0 = None
                    # The whole-body controller squats to follow a low wrist
                    # target and that takes a couple of seconds; hold before
                    # integrating, or the loop reads the transient as failure.
                    next_update = time.time() + REACH_SETTLE_AFTER_GLIDE

            delta = desired_local - palm_local
            err = float(np.linalg.norm(delta))
            if glide_t0 is None and time.time() >= next_update:
                step = np.clip(REACH_GAIN * delta, -REACH_MAX_STEP, REACH_MAX_STEP)
                cmd_local = np.clip(cmd_local + step, REACH_BOX_LO, REACH_BOX_HI)
                next_update = time.time() + 1.0 / REACH_UPDATE_HZ

            if side == "right":
                self.cmd.set_wrists(right_pos=cmd_local)
            else:
                self.cmd.set_wrists(left_pos=cmd_local)
            self._maintain_crouch(s)
            self.cmd.frame(mode=LocomotionMode.IDLE, height=height, facing=self._facing(s))

            if err < tol and glide_t0 is None:
                ok_since = ok_since or time.time()
                if time.time() - ok_since > settle:
                    # How much the palm turned since the offset was frozen.
                    # This is the exact size of the error the freeze buys, and
                    # it is the only unexplained term left between "the reach
                    # converged" and "the object is 13 cm off the pinch axis".
                    now = self._rot(s["palms"][side]["quat"])
                    drift = np.linalg.norm((now - palm_rot) @ palm_offset) if palm_offset is not None else 0.0
                    self.log(
                        f"reach {side} done (err {err:.3f} m, "
                        f"cmd {np.round(cmd_local, 3)}, "
                        f"offset drift since freeze {drift:.3f} m)"
                    )
                    self.last_cmd = cmd_local
                    return True
            else:
                ok_since = None

            # Saturation: the command keeps growing but the palm has stopped
            # improving, i.e. the point is outside the arm's workspace. Say so
            # instead of burning the whole timeout.
            if self.verbose and time.time() >= next_log:
                # `want` and the pelvis height are what separate the two ways
                # this stalls. If want_z stays far below the palm the arm is
                # genuinely out of travel; if want_z climbs towards the palm as
                # the body squats, the target is being reached by the legs and
                # the arm only has to hold station. Without both numbers the
                # trace cannot tell them apart, which is how "out of the arm's
                # workspace" got printed for both.
                self.log(
                    f"    reach err {err:.3f} cmd {np.round(cmd_local, 3)} "
                    f"palm {np.round(palm_local, 3)} "
                    f"want {np.round(desired_local, 3)} "
                    f"pelvis_z {s['pelvis']['pos'][2]:.3f}"
                )
                next_log = time.time() + 1.0

            if err < best - REACH_PROGRESS or glide_t0 is not None:
                best, stalled_since = min(best, err), time.time()
            elif time.time() - stalled_since > REACH_STALL_S:
                self.aborted = f"reach {side} stalled at {err:.3f} m — target looks out of the arm's workspace"
                self.last_cmd = cmd_local
                return False

            time.sleep(DT)

        self.aborted = f"reach {side} timed out (err {err:.3f} m)"
        self.last_cmd = cmd_local
        return False

    def servo_grasp(
        self,
        obj_key: str,
        side: str = "right",
        offset=GRASP_CENTRE_OFFSET,
        tol: float = 0.018,
        align_tol: float = 0.006,
        settle: float = 0.6,
        timeout: float = 16.0,
        nudge_max: float = 0.05,
        entry_axis: int = 1,
    ) -> bool:
        """Close the last few centimetres by servoing on the palm-frame error.

        The reach loop puts the *wrist* at a world point derived from a frozen
        palm rotation. Two separate errors then land on top of each other: the
        loop's own residual (3.7 cm measured) and whatever the wrist turned
        since the freeze (4 cm). Eight centimetres, into a hand that opens 6.1
        — so the fingers close beside the object, which is exactly what the
        pre-close logs showed.

        Neither error is interesting. The quantity that decides the grasp is
        where the object sits *in the palm's own frame*, and the privileged
        stream reports both poses, so servo on that directly. Rotation drift
        cancels: it moves the palm frame and the measurement together.

        If the hand moves by d (world), the object's palm coordinates shift by
        -d, so to take the object from q to the grip centre g the hand moves by
        R_palm @ (q - g).

        Two phases, and the split is the whole trick. Correcting across the jaw
        *while* advancing spends the clearance before the alignment converges,
        and the fingertip hits the object head-on. So: line up first, holding
        the object SERVO_SAFE_GAP out along the entry axis, and only advance
        once it is centred.

        `entry_axis` is the palm's y by default, not its x. Along x the middle
        fingertip fouls the object 10.5 cm before the grip centre reaches it,
        so a straight push shoves the object for the last tenth of a metre;
        the thumb side is clear the whole way in.
        """
        offset = np.asarray(offset, dtype=float)
        start = self.object_pos(obj_key)
        end = time.time() + timeout
        ok_since = None
        next_update = time.time()
        next_log = time.time()
        cmd_local = np.array(self.cmd.right_wrist_pos if side == "right" else self.cmd.left_wrist_pos, dtype=float)
        err = float("inf")
        phase = "align"

        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while closing on the object"
                return False
            moved = float(np.linalg.norm(self.object_pos(obj_key, s) - start))
            if moved > nudge_max:
                # Name the link. "Shoved 5 cm" is the same message whether a
                # fingertip caught the object on the way in or the palm did,
                # and the two want opposite fixes.
                hit = s["objects"][obj_key].get("touch_links") or ["nothing"]
                # Toppled or shoved? A tall object that has been knocked over
                # reports a large displacement with nothing touching it, since
                # the contact ended long before it finished falling. Reading
                # that as "the hand pushed it sideways" points the fix in the
                # wrong direction, so check the object's own up-axis.
                up = self._rot(s["objects"][obj_key]["quat"])[:, 2]
                how = "toppled" if abs(float(up[2])) < 0.7 else "shoved"
                self.aborted = (
                    f"{obj_key} {how} {moved * 100:.0f} cm while closing in ({phase}, touched {', '.join(hit)})"
                )
                return False

            rot = self._rot(s["palms"][side]["quat"])
            q = rot.T @ (self.object_pos(obj_key, s) - np.array(s["palms"][side]["pos"]))
            full = q - offset
            others = [i for i in range(3) if i != entry_axis]
            lat = float(np.linalg.norm(full[others]))
            if phase == "align":
                # Hold the object at a depth where the fingertips are clear of
                # it, and correct across the jaw there. Leaving the depth to
                # whatever the approach reach happened to leave is what broke
                # the previous version: that reach lands 4-8 cm out, and the
                # fingertips lead the grip centre by 11 cm, so the hand could
                # arrive already resting on the object — and then even a purely
                # sideways correction shoves it.
                delta_palm = full.copy()
                delta_palm[entry_axis] = full[entry_axis] - SERVO_SAFE_GAP
                if lat < align_tol and abs(delta_palm[entry_axis]) < SERVO_DEPTH_TOL:
                    phase = "advance"
                    self.log(
                        f"servo_grasp aligned ({np.round(full, 3)} in the "
                        f"palm frame); sliding in "
                        f"{abs(full[entry_axis]) * 100:.0f} cm"
                    )
            if phase == "advance":
                # Gate the advance on how well the jaw is lined up, rather than
                # committing to it once. The clearance either side of the
                # object is (aperture - width) / 2 — 5.5 mm for a 5 cm can in a
                # 6.1 cm hand — so a centimetre of drift acquired *during* the
                # 15 cm advance is enough to catch a fingertip on it, which a
                # one-shot alignment check cannot see coming. Scaling the depth
                # step by the alignment makes the hand stop and re-centre by
                # itself whenever it starts to wander.
                # SERVO_GATE_TOL, not align_tol. Gating on the same threshold
                # the alignment phase had to cross means the gate is exactly
                # shut at the moment alignment completes — the advance never
                # starts and the whole thing times out, which is what every
                # "timed out in the advance phase" was.
                gate = float(np.clip(1.0 - lat / SERVO_GATE_TOL, 0.0, 1.0))
                delta_palm = full.copy()
                delta_palm[entry_axis] = full[entry_axis] * gate
            err = float(np.linalg.norm(full if phase == "advance" else delta_palm))

            if phase == "advance" and err < tol:
                ok_since = ok_since or time.time()
                if time.time() - ok_since > settle:
                    self.log(
                        f"servo_grasp {obj_key}: object at "
                        f"{np.round(q, 3)} in the palm frame "
                        f"(want {np.round(offset, 3)}, err {err:.3f})"
                    )
                    self.last_cmd = cmd_local
                    return True
            else:
                ok_since = None

            if time.time() >= next_update:
                # Same cadence as reach: the arm lags about a second, so a
                # faster loop winds the command up through the lag.
                # palm frame -> world -> pelvis frame, which is the frame the
                # wrist command is interpreted in
                in_root = (self._rot(s["pelvis"]["quat"]).T @ rot) @ delta_palm
                step = np.clip(SERVO_GAIN * in_root, -SERVO_MAX_STEP, SERVO_MAX_STEP)
                cmd_local = np.clip(cmd_local + step, REACH_BOX_LO, REACH_BOX_HI)
                next_update = time.time() + 1.0 / REACH_UPDATE_HZ

            self.cmd.set_wrists(**{f"{side}_pos": cmd_local})
            self._maintain_crouch(s)
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))
            if self.verbose and time.time() >= next_log:
                self.log(
                    f"    servo[{phase}] err {err:.3f} object@palm "
                    f"{np.round(q, 3)} cmd {np.round(cmd_local, 3)} "
                    f"nudged {moved * 100:.0f} cm"
                )
                next_log = time.time() + 1.0
            time.sleep(DT)

        self.aborted = f"servo_grasp {obj_key} timed out in the {phase} phase (err {err:.3f} m)"
        self.last_cmd = cmd_local
        return False

    def grasp(
        self, obj_key: str, side: str = "right", hold: float = 1.5, squeeze: float = 0.08, ramp: float = 1.6
    ) -> bool:
        """Close until the fingers touch, then a little more. Then require it.

        Not a single jump to a fixed closed pose. That pose is sized for
        nothing in particular, and against a 4 cm can it buries the fingers
        27 mm inside it — a position-controlled hand really does drive there,
        so the object is squeezed out and lands on the floor, which the episode
        then reports as "no contact after closing". Closing progressively and
        stopping just past first contact needs no model of the object's width:
        the same loop grips a can, a bottle or a fork.

        Measured offline (locomani/tasks/bottle/diag/diag_grasp_pose.py), this hand first
        opposes a 4 cm cylinder at 70% of the closed pose, with 6 mm of
        penetration; at 100% it is 27 mm.

        The stopping test is OPPOSED contact — a thumb link and a finger link
        at once — not merely "the hand is touching it". Stopping at any contact
        stops far too early: two episodes latched at 34% and 38%, where only
        the middle finger was on the object, squeezed 8% further, pushed it off
        the table, and reported a confirmed grasp on the way down.
        """
        closed = HAND_CLOSED_RIGHT if side == "right" else HAND_CLOSED_LEFT
        touched_at = None
        start = time.time()
        while True:
            t = time.time() - start
            alpha = min(t / ramp, 1.0)
            if touched_at is not None:
                alpha = min(touched_at + squeeze, 1.0)
            self.cmd.set_hands(**{side: (HAND_OPEN + alpha * (closed - HAND_OPEN)).astype(np.float32)})
            s = self.obs()
            self._maintain_crouch(s)
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))
            if touched_at is None and _opposed(s, obj_key, side):
                touched_at = alpha
                self.log(
                    f"grasp {obj_key}: opposed contact at {alpha:.2f} of "
                    f"the closed pose; squeezing to "
                    f"{min(alpha + squeeze, 1.0):.2f}"
                )
                start = time.time() - ramp  # freeze the ramp, start the hold
            if touched_at is not None and time.time() - start > hold:
                break
            if touched_at is None and alpha >= 1.0 and t > ramp + hold:
                break
            time.sleep(DT)

        s = self.obs()
        links = s["objects"][obj_key].get("touch_links", [])
        if not _opposed(s, obj_key, side):
            self.aborted = (
                f"grasp {obj_key}: closed to {alpha:.2f} without "
                f"opposed contact (touching {', '.join(links) or 'nothing'})"
            )
            return False
        self.log(f"grasp {obj_key} with {side} confirmed by opposed contact ({', '.join(links)})")

        return True

    def release(self, side: str = "right", hold: float = 1.0) -> None:
        self.cmd.set_hands(**{side: HAND_OPEN})
        deadline = time.time() + hold
        while time.time() < deadline:
            self.cmd.frame(mode=LocomotionMode.IDLE)
            time.sleep(DT)
        self.log(f"release {side}")

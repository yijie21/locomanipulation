"""`Primitives`'s crouch/track/tuck mixin, split out of primitives.py
(refactor step 4b family 4, pure extraction; decision 9's file-size cap).
Every statement here is unchanged from the original inline body -- only the
packaging moved.
"""

from __future__ import annotations

import collections
import time

import numpy as np

from locomani.backends.sonic.agent_server import LocomotionMode, StreamMode
from locomani.backends.sonic.primitives_tuning import (
    ARM_REACH_LIMIT,
    CROUCH_HOLD_X,
    CROUCH_HOLD_Y,
    CROUCH_STEP,
    CROUCH_TOL,
    CROUCH_WRIST_MAX,
    CROUCH_WRIST_MIN,
    DT,
    PELVIS_HOLD_GAIN,
    REACH_BOX_HI,
    REACH_BOX_LO,
    READY_ARM,
    READY_UPPER,
    READY_WRISTS,
    SHOULDER_OFFSET,
    STANDING_PELVIS_Z,
    TRACK_BIAS_GAIN,
    TRACK_BIAS_MAX,
    TRACK_LAG,
    TRACK_MIN_S,
    TRACK_ROT_BOOST,
    TRACK_ROT_GAIN,
    TRACK_SPEED,
    TUCK_WRISTS,
    _quat_of,
    _rot_lerp,
    _rot_scale,
    upper_body_from_mj,
)
from locomani.primitives import wrist_map


class PrimitivesCrouchMixin:
    """`crouch`/`stand_up`/`track`/`crouch_needed`/`_hold_pelvis`/
    `_maintain_crouch`/`tuck_arms`/`carry_pose` -- squatting, streaming a
    wrist pose, and holding a posture while another loop runs."""

    def crouch(
        self, target_pelvis_z: float, hold_side: str = "left", timeout: float = 12.0, settle: float = 0.8
    ) -> bool:
        """Lower the body until the pelvis sits at `target_pelvis_z`.

        The grasp needs the grip point *above* the pelvis — with the palm held
        horizontal, which is the only orientation this controller offers, the
        arm cannot reach below it (locomani/tasks/bottle/diag/diag_reach_map.py). At a waist
        height table that is impossible standing up, and no amount of arm
        tuning changes it. Squatting moves the pelvis instead of the hand.

        The lever is the wrist target, not the planner: `height` is inert
        (measured again in locomani/diag_squat.py — 0.70, 0.60 and 0.50 all
        leave the pelvis at 0.536), while a low wrist target takes it from
        0.782 down to 0.536, 25 cm.

        Which is why the crouch is held with the OTHER arm. Driving it with the
        grasping arm is a positive feedback loop: once that arm rises to a grip
        point now above the pelvis, the body stands back up and takes the grip
        point out of range again. The free arm holds a low target beside the
        body and the working arm is left free to do the task — which is also
        roughly what a person does.
        """
        pos = np.array([CROUCH_HOLD_X, CROUCH_HOLD_Y if hold_side == "left" else -CROUCH_HOLD_Y, -0.10])
        end = time.time() + timeout
        ok_since = None
        if self.hold_yaw is None:
            self.hold_yaw = self.pelvis_yaw()
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while crouching"
                return False
            z = float(s["pelvis"]["pos"][2])
            err = z - target_pelvis_z
            if abs(err) < CROUCH_TOL:
                ok_since = ok_since or time.time()
                if time.time() - ok_since > settle:
                    self.crouch_cmd = pos.copy()
                    self.crouch_hold = (hold_side, target_pelvis_z, pos.copy())
                    self.log(f"crouch to {z:.3f} m (wanted {target_pelvis_z:.3f}, wrist z {pos[2]:+.2f})")
                    return True
            else:
                ok_since = None
                # The mapping from wrist target to pelvis height is steeply
                # non-linear — flat to -0.25, then 0.10 m of pelvis per 0.15 of
                # command — so close the loop on the pelvis instead of fitting
                # it.
                pos[2] = float(np.clip(pos[2] - np.sign(err) * CROUCH_STEP, CROUCH_WRIST_MIN, CROUCH_WRIST_MAX))
            self.cmd.set_wrists(**{f"{hold_side}_pos": pos})
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))
            time.sleep(DT)

        self.aborted = f"crouch timed out at {z:.3f} m, wanted {target_pelvis_z:.3f}"
        return False

    def stand_up(self, hold_side: str = "left") -> None:
        """Release the crouch by putting the holding arm back at rest."""
        self.crouch_cmd = None
        self.crouch_hold = None
        self.hold_yaw = None
        self.cmd.set_wrists(**{f"{hold_side}_pos": TUCK_WRISTS[hold_side]})

    def track(
        self,
        target_pos,
        target_rot=None,
        side: str = "right",
        speed: float = TRACK_SPEED,
        tol: float = 0.03,
        settle: float = 0.5,
        extra: float = 6.0,
        obj_key: str | None = None,
        nudge_max: float = 0.10,
    ) -> bool:
        """Stream a wrist pose along a straight line, and let the body follow.

        This replaces reach()'s search with a trajectory. reach() treated the
        interface as a black box to be probed: seed the command from an inverse
        model, hold it still for 2.5 s so the transient dies, then nudge at 3 Hz
        and watch what happens. Eight and a half seconds of that is commanded
        waiting before the first correction, and it is most of why the videos
        show a robot freezing and then lurching.

        But SONIC is a motion TRACKER. Its training data is VR teleoperation —
        continuous, human, whole-body references — and a reference that stands
        still for two seconds and then jumps is the one thing it never saw. So
        give it what it expects: a smooth pose reference, updated every control
        tick, moving at a hand's pace from where the hand is to where it has to
        be. The body then follows a trajectory instead of chasing a setpoint.

        Two corrections make that work on this deploy:

        * Position does not track 1:1 — ask for 0.65 m ahead and the palm stops
          at 0.40. So the command carries a bias, seeded from the measured
          mapping and then trimmed continuously rather than in 3 Hz steps.
        * The bias is integrated against a DELAYED copy of the reference. The
          arm lags the command by about a second; comparing the palm to where
          the reference is *now* would read that lag as error, wind the bias up
          through the whole move and overshoot at the end. Comparing it to
          where the reference was a second ago leaves only the real error.

        Orientation gets the same treatment. The wrist channel is attenuated
        ~4.5x (diag_pitch.py), so the commanded rotation is the wanted one
        scaled up about its own axis, and then trimmed by the same lagged loop.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        s = self.obs()
        start = np.array(s["palms"][side]["pos"], dtype=float)
        span = float(np.linalg.norm(target_pos - start))
        duration = max(TRACK_MIN_S, span / max(speed, 1e-3))

        # Bias seed: what the command has to be for the palm to end up at the
        # target, according to the measured mapping. The integrator only has to
        # trim this, so it starts near zero error instead of metres away.
        end_local = self.world_to_root_full(target_pos, s)
        seed = wrist_map.command_for(end_local) if wrist_map.available() else self._legacy_seed(end_local)
        bias = np.clip(seed - end_local, -TRACK_BIAS_MAX, TRACK_BIAS_MAX)

        rot_bias = np.eye(3)
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=float).reshape(3, 3)

        obj_start = self.object_pos(obj_key, s) if obj_key else None
        free_side = "left" if side == "right" else "right"
        pelvis_z0 = float(s["pelvis"]["pos"][2])
        lag = collections.deque()  # (t, reference position)
        t0 = time.time()
        end_t = t0 + duration + extra
        ok_since = None
        err = float("inf")
        next_log = t0

        while time.time() < end_t:
            now = time.time()
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while tracking"
                return False
            if obj_key is not None:
                moved = float(np.linalg.norm(self.object_pos(obj_key, s) - obj_start))
                if moved > nudge_max:
                    hit = s["objects"][obj_key].get("touch_links") or ["nothing"]
                    self.aborted = (
                        f"{obj_key} moved {moved * 100:.0f} cm while "
                        f"the hand was on its way in (touched "
                        f"{', '.join(hit)})"
                    )
                    return False

            # --- reference: smoothstep along the straight line --------------
            u = min(1.0, (now - t0) / duration)
            e = u * u * (3.0 - 2.0 * u)
            ref = start + (target_pos - start) * e

            lag.append((now, ref.copy()))
            while len(lag) > 1 and now - lag[0][0] > TRACK_LAG:
                lag.popleft()
            ref_then = lag[0][1]

            palm = np.array(s["palms"][side]["pos"], dtype=float)
            err_vec = ref_then - palm
            err = float(np.linalg.norm(target_pos - palm))
            bias = np.clip(bias + TRACK_BIAS_GAIN * DT * err_vec, -TRACK_BIAS_MAX, TRACK_BIAS_MAX)

            cmd = np.clip(self.world_to_root_full(ref, s) + bias, REACH_BOX_LO, REACH_BOX_HI)

            kw = {f"{side}_pos": cmd}
            if target_rot is not None:
                measured = self._rot(s["palms"][side]["quat"])
                # Trim the orientation bias against the same lagged idea of
                # where the palm should be pointing by now.
                slip = target_rot @ measured.T
                rot_bias = _rot_lerp(rot_bias, slip @ rot_bias, TRACK_ROT_GAIN * DT)
                want = rot_bias @ target_rot
                # Pre-scale about the axis of the remaining turn: the channel
                # only delivers a fraction of what it is asked for.
                kw[f"{side}_quat"] = _quat_of(_rot_scale(want @ measured.T, TRACK_ROT_BOOST) @ measured)

            self.cmd.set_wrists(**kw)
            if self.crouch_hold is not None:
                self._maintain_crouch(s)
            else:
                self._hold_pelvis(s, pelvis_z0, free_side)
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))

            if self.verbose and now >= next_log:
                self.log(
                    f"    track u={u:.2f} err {err:.3f} palm "
                    f"{np.round(self.world_to_root_full(palm, s), 3)} "
                    f"cmd {np.round(cmd, 3)} bias {np.round(bias, 3)} "
                    f"pelvis_z {s['pelvis']['pos'][2]:.3f}"
                )
                next_log = now + 1.0

            if u >= 1.0 and err < tol:
                ok_since = ok_since or now
                if now - ok_since > settle:
                    self.log(f"track {side} done (err {err:.3f} m after {now - t0:.1f} s, span {span:.3f} m)")
                    self.last_cmd = cmd
                    return True
            else:
                ok_since = None

            time.sleep(DT)

        self.aborted = (
            f"track {side} did not arrive: {err:.3f} m short of "
            f"{np.round(target_pos, 3)} after {duration + extra:.1f} s"
        )
        self.last_cmd = cmd

        return False

    def crouch_needed(self, plan, side: str = "right", state=None):
        """Highest pelvis that still puts the wrist target inside the arm.

        `plan` carries the wrist pose outright now, so there is nothing to
        derive: the adaptive planner already turned the grip point into the
        place the wrist has to be. Walk the pelvis down from standing and stop
        at the first height where the shoulder can make it.

        Returns None when no height works, which is a stand-pose problem rather
        than a crouch problem: squatting moves the shoulder down but also back,
        so past a point it stops helping and the answer is to stand closer.
        """
        s = state or self.obs()
        here = self.pelvis_xy(s)
        yaw = self.pelvis_yaw(s)
        rot = self._rot([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        shoulder = SHOULDER_OFFSET * np.array([1.0, 1.0 if side == "right" else -1.0, 1.0])
        wrist = np.asarray(plan.wrist, dtype=float)

        for pelvis_z in np.arange(STANDING_PELVIS_Z, 0.60, -0.01):
            local = rot.T @ np.array([wrist[0] - here[0], wrist[1] - here[1], wrist[2] - pelvis_z])
            if float(np.linalg.norm(local - shoulder)) <= ARM_REACH_LIMIT:
                return float(pelvis_z)

        return None

    def _hold_pelvis(self, s, target_z: float, free_side: str) -> None:
        """Keep the pelvis at `target_z` using the free hand as the lever.

        _maintain_crouch only runs when a crouch was commanded, so during a
        reach from a standing pose NOTHING regulated the pelvis — and the
        controller squats on its own whenever the arm is asked for something
        demanding. Measured on tracks with no crouch at all: 0.779 -> 0.641 m,
        14 cm of sink nobody asked for. That is not cosmetic. The wrist command
        is in the pelvis frame, so a sinking pelvis moves the target away from
        the hand, the bias integrator winds up chasing it, and the sink gets
        worse. It is the same runaway the crouch used to have, minus the
        crouch.

        The lever is the same one crouch uses: the free hand's height. Pushing
        it down makes squatting cheap for the optimizer, lifting it makes
        standing cheap, and a proportional term on the deviation holds station.
        """
        base = np.asarray(TUCK_WRISTS[free_side], dtype=float)
        rise = float(s["pelvis"]["pos"][2]) - target_z
        cmd = base.copy()
        cmd[2] = float(np.clip(base[2] - PELVIS_HOLD_GAIN * rise, CROUCH_WRIST_MIN, CROUCH_WRIST_MAX))
        self.cmd.set_wrists(**{f"{free_side}_pos": cmd})

    def _maintain_crouch(self, s) -> None:
        """Keep the squat held while another loop runs. Call once per tick."""
        if self.crouch_hold is None:
            return
        side, target_z, base = self.crouch_hold
        rise = float(s["pelvis"]["pos"][2]) - target_z
        cmd = base.copy()
        # Correct BOTH directions. Pushing the holding hand down when the
        # pelvis rises makes standing up a losing trade for the optimizer;
        # clamping that correction at zero, as this did, left the other
        # direction completely uncontrolled — and the other direction is the
        # one that was breaking every grasp.
        #
        # Measured over a whole batch: the crouch lands on target (0.657 m for
        # a 0.651 m ask) and then, once the reaching arm is asked for a point
        # it cannot make, the controller squats to lengthen the arm — pelvis
        # 0.658 -> 0.439 m in one reach, 5 episodes out of 5. That sinking
        # moves the shoulder down and back, so the fixed world target recedes
        # in the pelvis frame (want_x 0.36 -> 0.64), so the command grows, so
        # it squats further. The arm's error converged to 0.14 m and then
        # climbed back to 0.31 m purely from the body falling out from under
        # it. A symmetric term turns that runaway into a regulator.
        cmd[2] = float(np.clip(base[2] - 1.5 * rise, CROUCH_WRIST_MIN, CROUCH_WRIST_MAX))
        self.cmd.set_wrists(**{f"{side}_pos": cmd})

    def tuck_arms(self, hold: str | None = None) -> None:
        """Bring the wrists in close before walking anywhere.

        The default rest pose leaves the hands out where they catch table
        corners the pelvis clears by a comfortable margin — see TUCK_WRISTS.
        This is issued once; the wrist target persists until something else
        sets it.

        `hold` names an arm that is already carrying something and has its own
        pose (carry_pose): re-commanding it mid-carry would swing the load.

        The walking pose is READY_WRISTS — forearms level, hands above
        tabletop height — so the grasp never has to raise the hand at the
        table; the arms travel the whole episode in the posture the reach
        starts from.
        """
        kw = {f"{s}_pos": READY_WRISTS[s] for s in ("left", "right") if s != hold}
        # Never overwrite an arm that is holding a crouch — that arm's low
        # target is the only thing keeping the body down.
        if self.crouch_cmd is not None:
            kw.pop("left_pos", None)
        self.cmd.set_wrists(**kw)
        if self.cmd.stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
            if hold is None:
                self.cmd.set_upper_body(READY_UPPER)
            else:
                # Freeze the holding arm where it is (it has the object);
                # only the free arm goes to READY.
                v = upper_body_from_mj(self.obs()["joints"])
                free = "left" if hold == "right" else "right"
                col = 0 if free == "left" else 1
                for k in range(7):
                    v[3 + 2 * k + col] = READY_ARM[free][k]
                self.cmd.set_upper_body(v)

    def carry_pose(self, side: str = "right") -> None:
        """Tuck the loaded arm in front of the chest before walking.

        An arm left extended is both a collision risk and, at walking speed, a
        large disturbance torque on a balancing humanoid.
        """
        pos = np.array([0.28, -0.12 if side == "right" else 0.12, 0.05])
        if side == "right":
            self.cmd.set_wrists(right_pos=pos)
        else:
            self.cmd.set_wrists(left_pos=pos)

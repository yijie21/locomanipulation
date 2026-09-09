"""`Primitives`'s walking/turning mixin, split out of primitives.py (refactor
step 4b family 4, pure extraction; decision 9's file-size cap). Every
statement here is unchanged from the original inline body -- only the
packaging moved.
"""

from __future__ import annotations

import time

import numpy as np

from locomani.backends.sonic.agent_server import LocomotionMode
from locomani.backends.sonic.primitives_tuning import (
    APPROACH_LATERAL_MAX,
    DT,
    GRASP_BEARING,
    MIN_STAND_OFF,
    NAV_ARRIVE,
    NAV_EASE,
    NAV_PROGRESS,
    NAV_REPLAN_S,
    NAV_RING_BAND,
    NAV_SPEED,
    NAV_SPEED_MIN,
    NAV_STALL_S,
    NAV_STEP,
    NAV_TURN_IN_PLACE,
    NAV_TURN_STRAIGHT,
    STAND_OFF,
    wrap,
)


class PrimitivesWalkMixin:
    """`walk_to`/`turn_to`/`navigate`/`goto_object`/`face_for_grasp`/`approach`
    -- moving and turning the pelvis."""

    def walk_to(
        self,
        target_xy,
        stop_dist: float = 0.05,
        timeout: float = 30.0,
        mode: LocomotionMode = LocomotionMode.SLOW_WALK,
        face_target: bool = True,
        face_yaw: float | None = None,
        on_tick=None,
    ) -> bool:
        """Drive the pelvis to within `stop_dist` of `target_xy`.

        `movement` is a world-frame direction with magnitude in [0, 1] acting as
        a speed scale, so it is recomputed every tick from the live position
        error — which is what makes this robust to the perturbed start poses
        P2 samples.

        `on_tick(state, dist)` runs once per control tick before the frame is
        sent, so a caller can shape the upper body WHILE the legs walk — that
        is what lets the arm rise on the way in instead of after arriving.
        """
        target = np.asarray(target_xy, dtype=float)
        end = time.time() + timeout
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                # Where it fell is the whole question — the orbit, the radial
                # leg in, and the corrective re-aim all use walk_to, and they
                # fail for different reasons.
                self.aborted = f"fell while walking at {np.round(self.pelvis_xy(s), 2)} heading for {target.round(2)}"
                return False
            err = target - self.pelvis_xy(s)
            dist = float(np.linalg.norm(err))
            if dist < stop_dist:
                self.cmd.frame(mode=LocomotionMode.IDLE)
                self.log(f"walk_to {target.round(2)} done ({dist:.3f} m)")
                return True

            if on_tick is not None:
                on_tick(s, dist)
            direction = err / max(dist, 1e-6)
            # Ease off over the last 0.5 m so it does not overshoot the stop.
            gain = float(np.clip(dist / 0.5, 0.25, 1.0))
            if face_target:
                facing = direction
            elif face_yaw is None:
                facing = np.array([np.cos(self.pelvis_yaw(s)), np.sin(self.pelvis_yaw(s))])
            else:
                # EASE toward a world heading while walking, the way a person
                # curves onto a line rather than turning first and then
                # translating. Stepped like turn_to, because a facing command
                # far from the robot's actual yaw does not turn it faster.
                #
                # Without this the plain `face_target=False` branch above
                # commands the robot's OWN current yaw -- "hold whatever you
                # have" -- so the residual the spawn turn leaves (the
                # controller's 7-8 deg heading dead zone, unavoidable) is
                # LOCKED IN for the whole walk instead of decaying. At a 3 m
                # spawn that residual is not small: rel is a pelvis-frame
                # bearing, so 9 deg of heading error reads as 0.48 m of
                # lateral error, the deadband approach then has to crawl
                # sideways to erase it, and it stalls trying. Measured on one
                # such round: pre-walk rel_y -0.56 with yaw_err +9, ending at
                # yaw_err +17 and out of the ego frame.
                err = wrap(face_yaw - self.pelvis_yaw(s))
                goal = self.pelvis_yaw(s) + np.clip(err, -0.5, 0.5)
                facing = np.array([np.cos(goal), np.sin(goal)])
            # Same as navigate: the deploy normalises the movement vector, so
            # the gain has to go into `speed` to have any effect at all.
            self.cmd.frame(
                mode=mode,
                speed=float(np.clip(NAV_SPEED * gain, NAV_SPEED_MIN, NAV_SPEED)),
                movement=(direction[0], direction[1], 0.0),
                facing=(facing[0], facing[1], 0.0),
            )
            time.sleep(DT)

        self.aborted = f"walk_to {target.round(2)} timed out"
        return False

    def turn_to(
        self, yaw: float, tol: float = 0.15, timeout: float = 15.0, mode: LocomotionMode = LocomotionMode.IDLE
    ) -> bool:
        """Turn on the spot to a world yaw.

        `tol` cannot be tightened below ~0.15 rad. The controller has a
        heading DEAD ZONE of about 7-8 deg: every probe in diag_turn.py, in
        both modes and at every angle from 15 to 120 deg, stops with a
        residual of 6.5-8.1 deg and then holds there. A tol inside that band
        does not turn harder, it just spins out the timeout -- which is how a
        -70 deg spawn turn came to fail at tol=0.12.

        `mode` matters more than the angle does. Measured (diag_turn.py,
        seconds to converge, +/- pairs):

            deg      IDLE          SLOW_WALK
             15    1.2 / 1.3       0.5 / 2.1
             30    8.8 / 4.3       1.0 / 3.0
             45   13.0 / 4.7       2.9 / 5.0
             60   FAIL / 4.4       4.0 / 6.7
             90    5.2 / 3.0       4.0 / 4.0
            120   FAIL / 4.5       5.8 / 6.1

        SLOW_WALK converged on all 12 probes, never slower than 6.7 s, and
        is barely affected by the angle -- 120 deg costs about as much as
        45. IDLE managed 10 of 12, erratically, and both of its failures are
        dead-zone near-misses (residual 7.9-8.1 deg) rather than stalls. This also
        replaces the older note here -- "an IDLE facing more than ~60 deg
        away stalls the controller (68 deg turns timing out while 30 deg
        turns finish in a second)" -- which read a tolerance artefact as an
        angle limit. Stepping the commanded facing is still right and still
        done; it is just not what was rescuing the 30 deg case.
        """
        end = time.time() + timeout
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while turning"
                return False
            err = wrap(yaw - self.pelvis_yaw(s))
            if abs(err) < tol:
                self.cmd.frame(mode=LocomotionMode.IDLE, facing=(np.cos(yaw), np.sin(yaw), 0.0))
                self.log(f"turn_to {np.rad2deg(yaw):.0f}deg done")
                return True
            # Walk the facing command toward the goal instead of jumping it
            # there — keep the commanded heading within a step the policy
            # actually takes. See the docstring for what this does and does
            # not buy.
            goal = self.pelvis_yaw(s) + np.clip(err, -0.5, 0.5)
            self.cmd.frame(mode=mode, movement=(0.0, 0.0, 0.0), facing=(np.cos(goal), np.sin(goal), 0.0))
            time.sleep(DT)

        self.aborted = f"turn_to {np.rad2deg(yaw):.0f}deg timed out"
        return False

    def navigate(
        self, timeout: float = 60.0, stop_dist: float = NAV_ARRIVE, tuck: bool = True, hold: str | None = None
    ) -> bool:
        """Follow the planner's distance field to whichever goal it holds.

        The field is the plan: `waypoint` returns the best point one stride
        away, and steering at it is the whole controller. What is *not*
        automatic is the humanoid part — a walk command whose direction is 90
        degrees off the robot's heading makes it crab sideways, which is how
        most of the falls happened. Hence the three bands: turn in place when
        badly misaligned, walk slower while correcting, walk freely when
        aligned.
        """
        if self.planner is None or self.planner.dist is None:
            self.aborted = "navigate: no plan (set a goal first)"
            return False
        if tuck:
            self.tuck_arms(hold)

        here = self.pelvis_xy()
        if not self.planner.reachable(here):
            # One attempt at re-marching from where the robot actually is; the
            # goal set may simply have no route to it, which is a real answer
            # and much better than walking into a wall for 60 s.
            self.planner.replan(here)
        if not self.planner.reachable(here):
            self.aborted = f"navigate: no route from {np.round(here, 2)} to the goal set"
            return False

        # A goal that had to be snapped is not where it was asked for, so
        # arriving at it means arriving `snap` metres short of the request.
        tol = stop_dist + self.planner.snap
        end = time.time() + timeout
        next_replan = time.time() + NAV_REPLAN_S
        next_log = time.time()
        best_d, since_progress = float("inf"), time.time()
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = f"fell while navigating at {np.round(self.pelvis_xy(s), 2)}"
                return False

            here = self.pelvis_xy(s)
            # Two ways the deploy's odometry can put the robot somewhere the
            # plan cannot help with. After a fall its state estimator never
            # relearns, so `facing` is interpreted in a rotated frame and the
            # robot walks confidently in the wrong direction; one run ended at
            # x = -6.97, five metres outside the map, and only then reported
            # "boxed in". Name it where it happens.
            if not self.occ.in_bounds(self.occ.cell_of(here)):
                self.aborted = (
                    f"navigate: pelvis left the map at "
                    f"{np.round(here, 2)} — odom has probably "
                    f"diverged, restart the stack"
                )
                return False

            d = self.planner.distance_at(here)
            if d < best_d - NAV_PROGRESS:
                best_d, since_progress = d, time.time()
            elif time.time() - since_progress > NAV_STALL_S:
                self.aborted = (
                    f"navigate: no progress for {NAV_STALL_S:.0f}s at {np.round(here, 2)}, {d:.2f} m of field left"
                )
                return False
            # (re-set below while arcing to align: the steering arc trades
            # field distance for heading on purpose, and counting that as a
            # stall killed every episode that spawned facing away)
            if d < tol:
                self.cmd.frame(mode=LocomotionMode.IDLE)
                self.log(f"navigate done ({d:.2f} m of field left, tol {tol:.2f})")
                return True
            # Re-march on a timer, and immediately whenever the field has gone
            # flat underfoot. The goal ring at a table sits right on the edge
            # of the inflation, so a robot arriving at it routinely lands one
            # cell inside — where the last wavefront never reached, because its
            # escape corridor was carved somewhere else entirely. That is a
            # stale plan, not an unreachable goal, and the first batch failed
            # every episode by treating it as the latter.
            flat = not self.planner.reachable(here)
            if flat or time.time() >= next_replan:
                self.planner.replan(here)
                next_replan = time.time() + NAV_REPLAN_S
            if flat and not self.planner.reachable(here):
                self.aborted = f"navigate: field went flat at {np.round(here, 2)}"
                return False
            if flat:
                d = self.planner.distance_at(here)
                if d < tol:
                    self.cmd.frame(mode=LocomotionMode.IDLE)
                    self.log(f"navigate done after replan ({d:.2f} m left)")
                    return True

            wp = self.planner.waypoint(here, min(NAV_STEP, max(d, 0.15)))
            if wp is None:
                self.aborted = f"navigate: boxed in at {np.round(here, 2)}"
                return False

            bearing = float(np.arctan2(*(wp - here)[::-1]))
            yaw = self.pelvis_yaw(s)
            dth = abs(wrap(bearing - yaw))
            if dth > NAV_TURN_IN_PLACE:
                gain = 0.0
            elif dth > NAV_TURN_STRAIGHT:
                gain = float(np.clip(1.0 - dth / NAV_TURN_IN_PLACE, 0.3, 1.0))
            else:
                gain = 1.0
            gain *= float(np.clip(d / NAV_EASE, 0.3, 1.0))
            # A facing goal more than ~60 deg away stalls the controller
            # outright, and a pure in-place turn barely rotates at all
            # (~6 deg/s, drifting the while) — SONIC turns properly only
            # while WALKING. So: carrot the heading half a radian ahead, and
            # when badly misaligned walk a slow steering arc along the
            # robot's own heading instead of stopping to pirouette.
            face = yaw + float(np.clip(wrap(bearing - yaw), -0.5, 0.5))
            if gain <= 0.0:
                step = np.array([np.cos(face), np.sin(face)])
                since_progress = time.time()
                best_d = min(best_d, d)
            else:
                step = (wp - here) / max(np.linalg.norm(wp - here), 1e-6)
            self.cmd.frame(
                # SLOW_WALK throughout, never WALK. Measured on a straight
                # line with nothing in the way (locomani/diag_speed.py),
                # WALK at the default speed falls over; SLOW_WALK covers
                # 0.166 m/s and stays upright. The follower used to switch to
                # WALK whenever it was well aligned and still far from the
                # goal, which is most of every approach.
                #
                # `gain` scales the *speed* field, not the movement vector: the
                # deploy normalises that vector's length away, so scaling it
                # was controlling nothing at all. Pace has never been under
                # this loop's control until now.
                mode=LocomotionMode.SLOW_WALK,
                speed=(NAV_SPEED_MIN if gain <= 0.0 else float(np.clip(NAV_SPEED * gain, NAV_SPEED_MIN, NAV_SPEED))),
                movement=(step[0], step[1], 0.0),
                facing=(np.cos(face), np.sin(face), 0.0),
            )

            if self.verbose and time.time() >= next_log:
                self.log(
                    f"    nav d={d:.2f} at {np.round(here, 2)} -> "
                    f"{np.round(wp, 2)} (turn {np.rad2deg(dth):.0f}deg, "
                    f"gain {gain:.2f})"
                )
                next_log = time.time() + 2.0
            time.sleep(DT)

        self.aborted = f"navigate timed out with {d:.2f} m of field left"
        return False

    def goto_object(
        self, target_xy, want_dist: float = STAND_OFF, timeout: float = 60.0, hold: str | None = None
    ) -> bool:
        """Plan and walk to a standing pose the given distance from a target."""
        if self.planner is None:
            self.aborted = "goto_object: no map (Primitives built without meta)"
            return False
        here = self.pelvis_xy()
        if not self.planner.set_goal_ring(
            target_xy, want_dist - NAV_RING_BAND, want_dist + NAV_RING_BAND, here_xy=here
        ):
            self.aborted = f"goto_object: nowhere legal to stand {want_dist:.2f} m from {np.round(target_xy, 2)}"
            return False
        # A widened goal is a geometry problem, not a plan: it means nowhere on
        # the requested ring is legal, so the arrival tolerance absorbs the
        # widening and the robot stops short. Say what the achievable distance
        # actually is instead of retrying the same request five times.
        if self.planner.snap > 0.05:
            self.log(
                f"WARNING: nothing legal {want_dist:.2f} m from "
                f"{np.round(target_xy, 2)} — ring widened by "
                f"{self.planner.snap:.2f} m, so the closest the pelvis can "
                f"stand is about {want_dist + self.planner.snap:.2f} m. The "
                f"object is too far inside the surface's footprint."
            )
        self.log(
            f"plan to stand {want_dist:.2f} m from "
            f"{np.round(target_xy, 2)}: {self.planner.distance_at(here):.2f} m "
            f"of path, goal widened by {self.planner.snap:.2f} m"
        )

        return self.navigate(timeout=timeout, hold=hold)

    def face_for_grasp(self, target_xy, side: str = "right") -> bool:
        """Turn so the target sits in front of the grasping hand, not the nose.

        Facing the object dead-on puts it on the pelvis's centreline, and the
        arm then has to reach across the body — which is where the integrating
        reach ran out of workspace. Offsetting the heading by GRASP_BEARING
        puts the object roughly where the hand naturally hangs.
        """
        target = np.asarray(target_xy, dtype=float)
        bearing = float(np.arctan2(*(target - self.pelvis_xy())[::-1]))
        offset = GRASP_BEARING if side == "right" else -GRASP_BEARING

        return self.turn_to(wrap(bearing + offset))

    def approach(
        self,
        target_xy,
        surface_xy,
        want_dist: float,
        attempts: int = 3,
        side: str | None = None,
        circumradius: float = 0.0,
        half_extent=None,
        hold: str | None = None,
    ) -> bool:
        """Get within `want_dist` of a target on a surface, facing it.

        `surface_xy`, `circumradius` and `half_extent` are accepted for
        compatibility and no longer used: the obstacle geometry now lives in
        the map, which knows about *both* surfaces and the walls rather than
        the one surface the caller happened to name.

        The retry loop stays, though, because it was never about the path. The
        robot keeps drifting while it turns — the deploy is still finishing the
        step it had already committed to — so it can end a clean walk 0.5 m
        from the object and be 0.9 m away by the time it faces it. Re-planning
        to a tighter ring and walking the last of it is what converges.
        """
        target = np.asarray(target_xy, dtype=float)
        offset = min(STAND_OFF, want_dist * 0.9)
        d = float("inf")
        for i in range(attempts):
            if not self.goto_object(target, offset, hold=hold):
                return False
            # A turn that times out used to be ignored, which left the robot
            # sideways-on with the object 0.5 m off its centreline and the arm
            # unable to reach it — the failure surfaced two stages later as a
            # workspace error. Treat it as the approach failing, here.
            turned = (
                self.face_for_grasp(target, side)
                if side
                else self.turn_to(float(np.arctan2(*(target - self.pelvis_xy())[::-1])))
            )
            if not turned:
                self.log(f"approach attempt {i + 1}: {self.aborted}")
                continue

            s = self.obs()
            d = float(np.linalg.norm(target - self.pelvis_xy(s)))
            local = self.world_to_root(np.append(target, s["pelvis"]["pos"][2]), s)
            self.log(f"approach attempt {i + 1}: {d:.3f} m from target, pelvis-frame xy {np.round(local[:2], 3)}")
            # Distance alone is not enough. The robot drifts sideways through
            # the turn as much as forwards, so it can end up the right distance
            # away with the object nearly beside it, which no arm extension can
            # fix. Bound the lateral offset too.
            if d < want_dist and (side is None or abs(local[1]) < APPROACH_LATERAL_MAX):
                return True

            # Aim closer next time, floored so the ring never collapses inside
            # the table's own inflation — where it would have nowhere legal to
            # stand and goto_object would fail outright rather than get closer.
            offset = max(MIN_STAND_OFF, offset - (d - want_dist * 0.92))
            self.log(f"  re-planning for a stand-off of {offset:.2f} m")

        self.aborted = f"approach did not converge: {d:.2f} m > {want_dist}"
        return False

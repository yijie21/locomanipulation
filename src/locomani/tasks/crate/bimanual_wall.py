"""Ready-stand pilot for the WALL-TOP crate grasp (design 2026-09-07).

The crate is rotated 90 deg from the carry scene so its two full-height long
walls are left/right of the robot, and each jaw straddles a wall top with the
fingers forward, pitched down. No walk-in: the robot is reset on a stand
sampled from the walk-in's measured scatter, so this measures the grasp alone.

    python src/locomani/tasks/crate/bimanual_wall.py --rounds 20 --manifest <jsonl>

Phases (each observable, no integrators):
  A hover    both jaws 6 cm above the wall tops, at a depth along the wall
             chosen from the measured stand-off (reach_wall sweep: stand-off +
             depth <= 0.40 at palm pitch 60)
  B descend  reference lowered at DESCEND_SPEED until each pad centre is
             SEAT_BELOW under the top with the wall between the pads, or the
             reference reaches its floor
  C close    same-frame double close, bilateral contact check
  D lift     10 cm with bar-end levelling, 2 s hold; success = rise >= 6 cm,
             both hands touching, tilt < TILT_OK_DEG

Split across three files (refactor step 4b family 5, pure extraction;
decision 9's file-size cap): this file keeps `wall_frame`, `wall_point`,
`errs_of`, `arms_offset`, the module-level constants, and `run_round` itself
(now a thin orchestrator); `wall_phases.py` has `run_round`'s first three
phases (`_phase_reset`, `_phase_stand`, `_phase_hover`) behind a
`WallRoundContext`; `wall_phases_descend.py` has the last three
(`_phase_descend`, `_phase_close`, `_phase_lift`) -- all six would not fit
wall_phases.py's own 600-line cap. Several names imported below
(`APPR_SPEED`, `EDGE_FLOOR`, `OVERDRIVE`, `OVERDRIVE_CAP`, `STAND_TOL`,
`WAYPOINT_BACK`, `WAYPOINT_TOL`, `BimanualArms`, `crate_tilt_deg`,
`hand_txt`, `rot_y`, `rot_z`) are no longer used in this file's own body --
they stay imported here anyway because `wall_phases.py` /
`wall_phases_descend.py` read them back through THIS module
(`from locomani.tasks.crate.bimanual_wall import ...`), the same
re-export-for-a-satellite-file pattern family 3 used for vla_round.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/tasks/crate/bimanual_wall.py`

from locomani import paths  # noqa: E402
from locomani.backends.sonic.agent_server import LocomotionMode  # noqa: E402
from locomani.backends.sonic.primitives import Primitives  # noqa: E402
from locomani.sim.privileged_pub import EpisodeControlClient  # noqa: E402
from locomani.tasks.crate.bimanual_carry import (
    APPR_SPEED,
    EDGE_FLOOR,
    OBJ,
    OVERDRIVE,
    OVERDRIVE_CAP,
    STAND_TOL,
    WAYPOINT_BACK,
    WAYPOINT_TOL,
    BimanualArms,
    crate_tilt_deg,
    hand_txt,
    rot_y,
    rot_z,  # noqa: E402
)
from locomani.tasks.crate.config import DEFAULT_CRATE_TASK, CrateTaskConfig  # noqa: E402
from locomani.tasks.crate.wall_phases import (  # noqa: E402
    WallRoundContext,
    _phase_hover,
    _phase_reset,
    _phase_stand,
)
from locomani.tasks.crate.wall_phases_descend import _phase_close, _phase_descend, _phase_lift  # noqa: E402


# --- crate geometry (assets/crates/blue_crate/model.xml, 2/3 crate) ----------
# Two grasp features, chosen with WALL_FEATURE:
#   wall  crate rotated 90 deg: the full-height long walls (crate-local y = +-0.1267)
#         are left/right, depth runs along crate-local x from the rail side's outer
#         face (0.2093).
#   rail  crate as in the carry scene (handles left/right): the 30 mm rim bars
#         (crate-local x = +-0.1933) with the posts removed from the asset, so the
#         slot is open under the whole bar; depth runs along crate-local y from the
#         long wall's outer face (0.1347).
def _wall_feature_constants(feature: str):
    """(DEPTH_AXIS, WALL_AXIS, WALL_Y, NEAR_X, CRATE_YAW_DELTA, GRIP_BELOW) for
    `feature` -- factored out of the module-level if/else below so main()'s
    --config/--set can rederive the same six constants together after
    overriding FEATURE, instead of duplicating the branch."""
    if feature == "wall":
        # descend floor: pad centre this far below the top (full wall)
        return 0, 1, 0.1267, 0.2093, -np.pi / 2, 0.080
    # the bar is 30 mm tall: pad centre 5 cm under its top still spans it
    # (pad +-33 mm) and clears the low wall 65 mm down
    return 1, 0, 0.1933, 0.1347, 0.0, 0.050


# CrateTaskConfig.wall_feature's default (decision 8); never overridden in
# any script today, so this constant IS that default. main() reassigns
# FEATURE/DEPTH_AXIS/WALL_AXIS/WALL_Y/NEAR_X/CRATE_YAW_DELTA/GRIP_BELOW
# together, via _wall_feature_constants, after --config/--set.
FEATURE = DEFAULT_CRATE_TASK.wall_feature
DEPTH_AXIS, WALL_AXIS, WALL_Y, NEAR_X, CRATE_YAW_DELTA, GRIP_BELOW = _wall_feature_constants(FEATURE)
TOP_Z = 0.085  # wall / bar top above the crate body origin
# --- grasp design (outputs/reach/wall_grasp_sweep.txt) -----------------------
PALM_PITCH_DEG = 60
REACH_SUM = 0.38  # stand-off + depth budget; IK limit 0.40, 2 cm margin
DEPTH_MIN, DEPTH_MAX = 0.06, 0.11  # 0.04 puts the finger root on the near rail; the hands land
# +2.5 cm deeper than the target (rail pilot 13), so 0.11 puts the
# grasp line on the crate's centre line (0.135); deeper than the
# crate's centre line (0.135) the crate pitches on the lift
# (rail pilot 11: every 0.20 m grasp tilted > 20 deg, 0.09-0.15 gave 2-13)
HOVER_ABOVE = 0.06  # pad centre above the wall top at the hover
# (GRIP_BELOW per feature above; 0.03 was eaten whole by the policy's 3-5 cm height bias)
SEAT_BELOW = 0.015  # seated when the pad centre is at least this far below the top
ACROSS_TOL = 0.015  # wall centred between the pads (47 mm jaw, 16 mm wall)
# Feed-forward for the policy's systematic 2-3 cm outward hand bias (wall pilot 9:
# 14/20 stands ended with both jaws outside their walls, L +13..+26 / R -20..-36 mm,
# the inner pad landing on the wall top). Hover/descend targets sit this far INSIDE
# the wall centre plane; errors are still measured against the true wall.
ACROSS_BIAS = 0.012  # 0.020 overshot: hands 0.6-1.4 cm inside the bar (rail pilot 11)
# The policy holds both hands 3-4 cm OUTSIDE their walls (wall pilots 2/3: across
# +40/-30 mm, inner finger riding the wall top). The across axis gets its own
# overdrive; along and height keep bimanual_carry's 0.7 / 4 cm.
ACROSS_OD, ACROSS_OD_CAP = 1.0, 0.04
LEAD_CAP = 0.06  # the commanded point never runs further than this ahead of the
# measured one (bimanual_carry's cap): an unbounded jump to the
# hover point swung the hands +-10 cm (wall pilot 4)
DESCEND_SPEED = 0.04  # m/s reference descent
HOVER_S, HOVER_TOL = 2.5, 0.03  # cap only: the hover ends when the hands stop moving (was a 5 s open-loop wait)
SHOULDER_PITCH_MIN_DEG = -80.0  # hover/grasp solutions sit at -40..-55
DESCEND_TIMEOUT = 12.0
FLOOR_DWELL = 0.5  # at the floor without seating: close anyway after this (was 2.0)
# Contact close: the descent reference keeps going down but the measured
# height stops -- the hand is resting on the bar. Visible in the image and in
# proprioception, unlike the millimetre seat test, which stays as a secondary.
CONTACT_STALL_S = 0.3
CONTACT_STALL_MM = 3.0
SEAT_DWELL = 0.3  # both seated continuously this long before the close (full-episode
# pilot 17 r0: a one-tick seat during a 3 cm/0.5 s overshoot closed on air)
TILT_OK_DEG = 20.0
# Pitch follow on the lift: the rectangular bar is clamped rigidly in the flat
# jaw, so the crate pitches with the hand (rail pilot 14: 4 -> 14 -> 20 deg as it
# rose from a folded-arm stand). The palm pitch target is counter-rotated by the
# measured crate pitch about the bar axis.
PITCH_FOLLOW_GAIN = 1.5
PITCH_FOLLOW_CAP_DEG = 25.0
PITCH_FOLLOW_LP = 0.2
# --- ready-stand distribution (walk-in scatter measured in pilots 34-37) ------
STAND_OFF_RANGE = (0.14, 0.28)  # pelvis to the crate's near face
STAND_LAT_RANGE = (-0.05, 0.05)
STAND_YAW_DEG = 5.0
STAND_YAW_MAX_DEG = 10.0  # fail fast: nothing downstream survives a bigger heading error
CRATE_DY_RANGE = (-0.08, 0.08)
SPAWN_BACK = 0.45  # reset this far behind the stand, teleport in with the arms up
# --- walk-in (WALL_WALK=1): bimanual_carry's approach in front of this grasp ---
# CrateTaskConfig.wall_walk's default (decision 8); main() reassigns this
# after --config/--set.
WALK = DEFAULT_CRATE_TASK.wall_walk
STAND_OFF_WALK = 0.24  # 0.21 (pilots 18/19) parked 0.30-0.39 with yaw +-10 deg; back to pilot 17's value for the A/B
WALK_LAT_BIAS = -0.04  # the gait parks ~4 cm left of its target (pilots 36/37)
WALK_SPAWN_BACK = (1.0, 1.3)
WALK_SPAWN_LAT = 0.15
WALK_STAND_OFF_BAND = (0.13, 0.32)  # depth = 0.38 - stand-off must stay in [0.06, 0.25]
WALK_LAT_TOL = 0.08
# Carry pose while walking: pilot 18 traces replayed in MuJoCo showed the open
# jaws hitting the near wall's top in 20/20 rounds at 2.5-4 s, pelvis still
# 0.36 m from the wall -- pads commanded 12 cm above the bar tops, delivered
# 1-8 cm (right hand lowest), fingers pitched down, hands 24 cm ahead of the
# pelvis. The dragged fingers turned the crate 7-14 deg, which IS the stand
# yaw residual. So: hands high, close to the chest, fingers level, and the
# reach forward only happens in the post-stand hover re-solve.
# Walking pose: arms at the sides. The policy's default pose carries the
# finger tips 0.19 m ahead of the pelvis at 0.56 m, i.e. into the table-edge /
# crate-wall plane at every stand under 0.23 m; the chest-high carry pose of
# pilot 21 (20/20) hid the whole crate from the ego camera at 3.5 s. This pose
# keeps the tips 1 cm ahead of the pelvis (12-30 cm behind the plane at every
# stand in the band), 27 cm out to the side, and 0% of the crate occluded
# (kinematic pre-flight 2026-09-07 evening). Joint order sp, sr, sy, el, wr,
# wp, wy; the right arm is the mirror.
WALK_SIDE_LEFT_DEG = (10.0, 10.0, 0.0, 60.0, 0.0, 30.0, 0.0)
# Reach after the stop: ONE continuous Cartesian path for each pad, a cubic
# Bezier from the side pose up along the body with the elbows out, over the
# rim, into the IK hover -- then straight into the descent, no hold. The
# G1's elbow hangs at rim height with a 0.30 m forearm, so every joint-space
# route from a hanging arm (elbow flex, forward swing, chained fixed poses)
# swept the finger tips through the rim plane at stands < 0.30 m; the chain
# of side/chest poses of pilot 22 avoided it but read as staged. Kinematic
# pre-flight of this curve (2026-09-07 night): no contact at stand-offs
# 0.13-0.31, 2.7-4.4 cm over the bar on the way in. The IK follows the curve
# with a small posture prior toward the hover solution so the wrist does not
# drift into a twisted branch (arm_ik posture term).
REACH_S = 2.5
REACH_UP = 0.25  # first control point: this far above the side pad ...
REACH_OUT = 0.15  # ... and this far outward (elbows out)
REACH_OVER = 0.10  # second control point: this far above the hover point
REACH_PITCH0_DEG = 90.0  # palm pitch at the start (fingers hanging), blends to PALM_PITCH_DEG
REACH_POSTURE_W = 0.02
# Re-approach instead of rejecting an out-of-band stand: back up to the
# waypoint and walk in again, so the data holds "stopped wrong -> corrected"
# rather than only clean stands. The crate-disturbed check still rejects.
WALK_ATTEMPTS = 3
DUMP_DIR = ""


def wall_frame(s):
    """(centre, n toward the crate, y_hat left, psi, sign, R) for the rotated crate."""
    o = s["objects"][OBJ]
    c = np.asarray(o["pos"], dtype=float)
    R = Primitives._rot(o["quat"])
    ex = R[:2, DEPTH_AXIS] / max(np.linalg.norm(R[:2, DEPTH_AXIS]), 1e-6)
    pel = np.asarray(s["pelvis"]["pos"][:2])
    sign = 1.0 if np.dot(ex, c[:2] - pel) >= 0 else -1.0
    n = sign * ex
    y_hat = np.array([-n[1], n[0]])
    psi = float(np.arctan2(n[1], n[0]))
    return c, n, y_hat, psi, sign, R


def wall_point(s, side, depth, dz, inset=0.0):
    """World point on the wall's centre plane: `depth` in from the near face, `dz` above
    the top, `inset` toward the crate centre (feed-forward for the outward hand bias)."""
    c, n, y_hat, psi, sign, R = wall_frame(s)
    # which local wall is the robot's left
    e_wall = np.zeros(3)
    e_wall[WALL_AXIS] = WALL_Y
    yl = WALL_Y if np.dot((R @ e_wall)[:2], y_hat) > 0 else -WALL_Y
    if side == "right":
        yl = -yl
    yl -= np.sign(yl) * inset
    v = np.zeros(3)
    v[DEPTH_AXIS] = sign * (-NEAR_X + depth)
    v[WALL_AXIS] = yl
    v[2] = TOP_Z + dz
    return c + R @ v


def errs_of(s, arms, side, depth):
    """Measured pad centre minus the wall line at the grasp depth: [along, across, height]."""
    c, n, y_hat, psi, sign, R = wall_frame(s)
    g, _ = arms.grip_point(s, side)
    e = g - wall_point(s, side, depth, 0.0)
    return np.array([float(e[:2] @ n), float(e[:2] @ y_hat), float(e[2])])


def run_round(
    p: Primitives, reset: EpisodeControlClient, meta: dict, rnd: int, rng: np.random.Generator, log, video_label: str
) -> dict:
    """One round's six phases (wall_phases.py / wall_phases_descend.py),
    threaded through a WallRoundContext.

    Body moved there (refactor step 4b, pure extraction) -- this function
    stays HERE, thin, so `main()`'s `run_round(p, reset, meta, rnd, rng,
    p.log, label)` call keeps resolving to this same name.
    """
    ctx = WallRoundContext(
        p=p,
        reset=reset,
        meta=meta,
        rnd=rnd,
        rng=rng,
        log=log,
        video_label=video_label,
        m={"round": rnd, "ok": False, "why": ""},
    )
    for phase in (_phase_reset, _phase_stand, _phase_hover, _phase_descend, _phase_close, _phase_lift):
        result = phase(ctx)
        if result is not None:
            return result
    raise AssertionError("wall_phases: no phase returned a result")


def arms_offset():
    from locomani.tasks.crate.bimanual_carry import OFFSET

    return OFFSET


from locomani.primitives.g1_joints import ARM_MJ as ARM_IDX  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--label", default="wall")
    ap.add_argument("--config", type=Path, help="CrateTaskConfig JSON; defaults if omitted")
    ap.add_argument("--set", dest="set_", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig()
    if args.set_:
        cfg = cfg.with_overrides(args.set_)
    # Run as a script this module is `__main__`; wall_phases.py's deferred
    # `from locomani.tasks.crate.bimanual_wall import WALK, ...` would otherwise
    # import a SECOND copy with the defaults, and the --config/--set overrides
    # below would never reach the phases (crate GPU gate 2026-09-10: wall_walk=true
    # ran the ready-stand variant). Alias the package name to this module first.
    sys.modules.setdefault("locomani.tasks.crate.bimanual_wall", sys.modules[__name__])
    # FEATURE/WALK (and FEATURE's derived DEPTH_AXIS/WALL_AXIS/WALL_Y/NEAR_X/
    # CRATE_YAW_DELTA/GRIP_BELOW) default from DEFAULT_CRATE_TASK at module
    # load; a --config/--set here overrides them the same way any other
    # caller would (module-level assignment), so wall_phases.py's deferred
    # `from bimanual_wall import WALK` etc. (read fresh each round) see it.
    global FEATURE, WALK, DEPTH_AXIS, WALL_AXIS, WALL_Y, NEAR_X, CRATE_YAW_DELTA, GRIP_BELOW
    FEATURE, WALK = cfg.wall_feature, cfg.wall_walk
    DEPTH_AXIS, WALL_AXIS, WALL_Y, NEAR_X, CRATE_YAW_DELTA, GRIP_BELOW = _wall_feature_constants(FEATURE)
    inst = int(os.environ.get("INSTANCE", "0"))
    for k, base in (
        ("TF_SONIC_PORT", 5556),
        ("TF_OUT_PORT", 5557),
        ("TF_KEYBOARD_PORT", 5580),
        ("TF_PRIVILEGED_PORT", 5581),
        ("TF_EPISODE_PORT", 5582),
    ):
        os.environ.setdefault(k, str(base + 10 * inst))
    global DUMP_DIR
    DUMP_DIR = os.path.dirname(os.path.abspath(args.manifest)) if args.manifest else ""
    meta = json.load(open(paths.SCENE_META))
    if meta.get("target_object") != OBJ:
        raise SystemExit(f"{paths.SCENE_META} is not the crate scene")
    print(f"[wall] instance {inst}, scene {paths.SCENE_META}, task config {cfg.to_dict()}")
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
    rng = np.random.default_rng(args.seed)
    wins = 0
    dead_rounds = 0  # consecutive "did not stabilise": the deploy is gone
    restarts = 0
    for rnd in range(args.rounds):
        t0 = time.time()
        label = f"{args.label}_{rnd:02d}"
        try:
            m = run_round(p, reset, meta, rnd, rng, p.log, label)
        finally:
            try:
                reset.end_video()
            except Exception:
                pass
        m["seconds"] = round(time.time() - t0, 1)
        m["video"] = label
        m["deploy_restarts"] = restarts
        wins += int(m["ok"])
        # Self-heal: the SONIC deploy stops itself on a one-loop LowState
        # timeout (rail pilot 12: another session's per-round deploy restarts
        # blip the sim clock), after which every reset "does not stabilise".
        # Two in a row -> restart our own deploy (port-scoped) and go on.
        dead_rounds = dead_rounds + 1 if m["why"].startswith("[stand] did not stabilise") else 0
        if dead_rounds >= 2:
            import subprocess

            print(f"[wall] deploy looks dead ({dead_rounds} rounds); restarting instance {inst} deploy")
            r = subprocess.run(
                ["bash", os.path.join(paths.REPO, "scripts", "restart_deploy.sh")],
                env=dict(os.environ, INSTANCE=str(inst)),
                cwd=paths.REPO,
            )
            print(f"[wall] restart_deploy.sh exit {r.returncode}")
            restarts += 1
            dead_rounds = 0
            p.cmd.reset_pose()
            reset.pin_arms()
            reset.reset()
            time.sleep(0.3)
            p.cmd.start_control()
            p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
        print(f"[wall] round {rnd}: {'OK' if m['ok'] else 'FAIL'} {m['why']} ({m['seconds']} s)")
        if args.manifest:
            with open(args.manifest, "a") as f:
                f.write(json.dumps(m, default=float) + "\n")
    print(f"[wall] {wins}/{args.rounds} OK")


if __name__ == "__main__":
    main()

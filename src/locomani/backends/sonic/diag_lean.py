"""How much of a commanded waist lean does the policy deliver, and what reach does it buy?

RESTORED 2026-09-12 (last seen at `78536ae:src/teleop_free/diag_lean.py`,
dropped by the refactor) for the same reason `diag_wrist.py` was restored on
2026-09-11: its answer is a per-checkpoint number that a production constant
was set from, and the checkpoint has changed.

At restoration on 2026-09-12, `bimanual_carry.WAIST_PITCH_MAX = 0.0` and
`BimanualArms.__init__` zeroed all three waist bounds. This describes that
historical baseline. Forward pitch was enabled in the shared two-hand solver
on 2026-09-13; see docs/handoff/crate-carry-waist-20260913.md for new evidence.
The justification on record is this diagnostic's old reading, 2-10% of a
commanded lean delivered on `policy/release`, confirmed by crate pilot 4 (34
deg commanded, 2.8 measured). But the native reference-replay of 2026-09-10
(docs/STATE.md) measured waist delivery 8.1 deg on release against 21.4 on
`policy/sonic_v1_1` from the same 29.3 deg reference -- and since 2026-09-11
v1.1 IS the default checkpoint. That replay drove the deploy's own reference
motion, NOT this channel, so it does not settle what the crate expert would
get; this does.

Two numbers per commanded lean, as before:

  1. delivery ratio -- measured waist pitch (`joints[14]`) over commanded.
  2. reach gain -- centimetres of palm travel per lean, arm joints frozen, so
     every millimetre of palm motion comes from the torso.

Both channels are swept: `planner` is what the crate and bottle experts use
(`upper_body_position[17]`, waist in slots 0-2), `streamed` is the 29-DoF pose
stream. Indices 12/13/14 are waist yaw/roll/pitch in MuJoCo order.

    python -m locomani.backends.sonic.diag_lean [--arm reach|neutral] [--json OUT]

Needs a live stack; the checkpoint under test is whatever that stack's deploy
was started with (recorded in the output as `policy`). Measurement only: it
commands waist angles and reads the measured ones, and changes no production
constant.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from locomani import paths
from locomani.backends.sonic.agent_server import LocomotionMode, StreamMode
from locomani.backends.sonic.primitives import Primitives
from locomani.backends.sonic.streamed_arm import PlannerArm, StreamedArm, reanchor
from locomani.primitives.arm_ik import ArmIK
from locomani.sim.privileged_pub import EpisodeControlClient

SIDE = "right"
LEANS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.52]  # rad; 0.52 = the MJCF's waist_pitch limit (29.79 deg)
HOLD_S = 2.5
WAIST_P = 14  # MuJoCo joint index of waist_pitch
AIM_PF = np.array([0.42, -0.20, -0.08])  # one far forward-reach aim, pelvis frame (outside arm-only reach on purpose)
OUT = os.path.join(paths.DIAG_DIR, "lean.json")  # paths.DIAG_DIR is the declared home for diag_*.py results


def pitch_of(quat) -> float:
    """Pitch of a quaternion, radians, positive nose-down like waist_pitch."""
    R = Primitives._rot(quat)
    return float(np.arcsin(np.clip(-R[2, 0], -1, 1)))


def restand(p: Primitives, reset: EpisodeControlClient, label: str) -> dict:
    """One clean standing start. diag_wrist.py's lesson: without a fresh stand
    per phase, a fall or a drift from the previous phase is inherited and the
    next phase's numbers look like a measurement without being one."""
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(None)
    p.cmd.reset_pose()
    reset.pin_arms()  # all-None = release any sim-side pin a previous run left on
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    # The deploy keeps its heading anchor across resets (memory
    # sonic-planner-heading-anchor); re-anchor or the IDLE facing command turns
    # the residual into real rotation while we are measuring a pitch.
    reanchor(p, side=SIDE)
    return p.obs()


def sweep(p: Primitives, reset: EpisodeControlClient, kind: str, q_arm, arm_mode: str) -> list[dict]:
    """One channel: hold the arm pose, walk the waist up through LEANS."""
    s = p.obs()
    if s["fallen"]:
        s = restand(p, reset, f"lean_{kind}")
    reset.pin_arms()
    arm = PlannerArm(p, SIDE, send_vel=True) if kind == "planner" else StreamedArm(p, SIDE)
    arm.start()
    q0 = arm.q.copy()
    if arm_mode == "reach":
        q0[15:29] = q_arm[15:29]
    q0[WAIST_P] = 0.0
    arm.move_to(q0, speed=0.8)
    arm.hold(1.5)
    base = p.obs()
    base_palm = np.asarray(base["palms"][SIDE]["pos"], dtype=float)
    base_pel = np.asarray(base["pelvis"]["pos"], dtype=float)
    rows = []
    for wp in LEANS:
        q = arm.q.copy()
        q[WAIST_P] = wp
        if arm_mode == "reach":
            q[15:29] = q_arm[15:29]
        arm.move_to(q, speed=0.5)
        arm.hold(HOLD_S)
        st = p.obs()
        palm = np.asarray(st["palms"][SIDE]["pos"], dtype=float)
        pel = np.asarray(st["pelvis"]["pos"], dtype=float)
        R = Primitives._rot(st["pelvis"]["quat"])
        delivered = float(st["joints"][WAIST_P])
        pel_pitch = pitch_of(st["pelvis"]["quat"])
        rec = dict(
            channel=kind,
            arm=arm_mode,
            cmd=wp,
            delivered=delivered,
            ratio=(delivered / wp if wp > 0.01 else None),
            pelvis_pitch=pel_pitch,
            # the 2026-09-10 native replay reports torso pitch; pelvis + waist
            # is the comparable quantity for this channel
            torso_pitch=pel_pitch + delivered,
            pelvis_z=float(pel[2]),
            palm_fwd=float((R.T @ (palm - pel))[0]),
            palm_z=float(palm[2]),
            d_fwd=float(np.linalg.norm((palm - base_palm)[:2])),
            d_z=float(palm[2] - base_palm[2]),
            pelvis_drift=float(np.linalg.norm((pel - base_pel)[:2])),
            # The palm drop is NOT self-evidently the waist's doing: the first
            # A/B (2026-09-12) dropped the palm ~11 cm on policy/release with
            # 2 deg of delivered waist, i.e. the same drop as v1.1's 26 deg.
            # So record the arm joints the policy actually delivered against
            # the constant arm command -- if they sag with the waist COMMAND,
            # the drop is arm tracking, not lean.
            arm_meas=[float(v) for v in np.asarray(st["joints"], dtype=float)[15:29]],
            arm_cmd=[float(v) for v in np.asarray(q, dtype=float)[15:29]],
            fallen=bool(st["fallen"]),
        )
        rows.append(rec)
        print(
            f"[lean] {kind:9} cmd {wp:.2f} -> {delivered:+.3f} "
            f"({delivered / wp * 100 if wp > 0.01 else 0:4.0f}%)  "
            f"waist {np.degrees(delivered):+5.1f}d  pelvis pitch {np.degrees(pel_pitch):+5.1f}d  "
            f"palm fwd {rec['palm_fwd'] * 100:5.1f}cm z {rec['palm_z']:.3f}  "
            f"drift {rec['pelvis_drift'] * 100:4.1f}cm{'  FELL' if rec['fallen'] else ''}",
            flush=True,
        )
        if rec["fallen"]:
            break
    q = arm.q.copy()
    q[WAIST_P] = 0.0
    arm.move_to(q, speed=0.5)
    arm.stop()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        choices=("reach", "neutral"),
        default="reach",
        help="reach: hold a far-extended arm pose (balance-expensive, the crate/bottle case); "
        "neutral: leave the arm where standing left it",
    )
    ap.add_argument("--channel", choices=("planner", "streamed", "both"), default="both")
    ap.add_argument("--json", default=OUT)
    args = ap.parse_args()

    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    print(f"\npolicy {paths.SONIC_POLICY}   scene {paths.SCENE_META}", flush=True)
    s = restand(p, reset, "lean")
    if s["fallen"]:
        raise SystemExit("robot did not stand after the first reset")

    q_arm = np.asarray(s["joints"], dtype=float)
    if args.arm == "reach":
        ik = ArmIK()
        ik.lo[WAIST_P], ik.hi[WAIST_P] = -0.05, 0.52  # let the solver see the real range
        q_arm, res, _ = ik.solve(SIDE, AIM_PF, None, q_mj29=np.asarray(s["joints"], dtype=float), use_waist=False)
        print(
            f"[lean] extended arm pose solved, IK res {res * 100:.1f} cm (the aim is out of arm-only reach on purpose)"
        )
    else:
        print("[lean] arm left at the standing pose")

    rows: list[dict] = []
    for kind in ("planner", "streamed"):
        if args.channel not in (kind, "both"):
            continue
        rows += sweep(p, reset, kind, q_arm, args.arm)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(dict(policy=paths.SONIC_POLICY, side=SIDE, arm=args.arm, rows=rows), fh, indent=1)
    print(f"\n[lean] wrote {args.json}")
    for kind in ("planner", "streamed"):
        v = [r for r in rows if r["channel"] == kind and r["cmd"] > 0.01 and not r["fallen"]]
        if not v:
            continue
        ratio = float(np.median([r["ratio"] for r in v]))
        top = max(v, key=lambda r: r["cmd"])
        print(
            f"    {kind:9} delivery ratio {ratio:.2f}  (range {min(r['ratio'] for r in v):.2f}"
            f"..{max(r['ratio'] for r in v):.2f});  at cmd {top['cmd']:.2f} the palm moved "
            f"{top['d_fwd'] * 100:.1f} cm forward, {top['d_z'] * 100:+.1f} cm in z, "
            f"waist reached {np.degrees(top['delivered']):.1f} deg"
        )
    p.cmd.pause_control()
    p.close()


if __name__ == "__main__":
    main()

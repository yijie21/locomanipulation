"""Is the VR 3-point wrist target live, and how much of it arrives?

`primitives_tuning.TRACK_ROT_BOOST = 4.5` is the INVERSE of a delivery ratio
measured on policy/release (105 deg asked, 23 delivered). v1.1's model card says
it "was trained with wrist-pose augmentation" and is "intended for heading-stable
3-point teleoperation", and its training config rewards
`["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]` with
`randomize_wrist_poses: true` -- so this channel is the one interface the
checkpoint was built for, and the boost is the one constant standing on the old
checkpoint's behaviour.

WHY THE 2026-09-11 RUNS COULD NOT MEASURE IT (ledger bottle.md 28)

Every case commanded the SAME wrist POSITION and took its "delivered rotation"
relative to the first case. With a dead channel that reference is the rest pose
itself, so every case reports ~0 delivered and a ratio of 0.01 -- which is
exactly what the second run printed, for both checkpoints, including the one
with a known-true 22%. The instrument could not tell "the field is inert" from
"rotation is ignored": it had no leg in which something was known to move.

So this version measures three things, in order, and each gates the next:

  1. POSITION. A `rest` leg and a `reach` leg with the SAME identity rotation.
     If the palm does not move between them, the channel is inert and no
     rotation number means anything -- the run says so and stops.
  2. THE DEPLOY'S OWN REFERENCE. `RefTap` reads `body_q_target` off the deploy's
     g1_debug stream, so "the controller never asked the arm to move" is
     separated from "it asked and the arm did not follow". 28 could not tell
     these apart either.
  3. ROTATION, referenced to the `reach` leg (same position, identity rotation),
     which is the only correct control for a rotation-only question.

The observation configs settle one thing offline: release and sonic_v1_1 carry
the SAME `vr_3point_local_target` / `vr_3point_local_orn_target` blocks (they
differ only in heading normalisation), and NEITHER observes `vr_compliance` --
so compliance is inert by construction and the checkpoints differ only in what
the policy does with an identical input.

    python -m locomani.backends.sonic.diag_wrist [--side right] [--json OUT]

Needs a live stack. Measurement only; changes no production constant.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from locomani import paths
from locomani.backends.sonic.agent_server import rest_wrists
from locomani.backends.sonic.primitives import DT, LocomotionMode, Primitives
from locomani.primitives.g1_joints import ARM_MJ
from locomani.sim.privileged_pub import EpisodeControlClient

SETTLE = 3.5  # the arm needs a couple of seconds to reach a new target
HOLD_POS = np.array([0.35, -0.22, -0.05])  # a plausible pre-grasp wrist point
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
# Below this the palm simply did not go anywhere between `rest` and `reach`.
POSITION_LIVE_M = 0.03
# Below this the deploy never changed its own joint reference, so whatever the
# palm did was not this channel.
REFERENCE_LIVE_RAD = 0.02


def quat_of_axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))], dtype=np.float32)


def legs(side: str):
    """(label, wrist position, wrist quaternion). The first two are the controls."""
    rest = rest_wrists()[0 if side == "left" else 1]
    return [
        ("rest (no target change)", np.asarray(rest, dtype=float), IDENTITY),
        ("reach, identity rotation", HOLD_POS, IDENTITY),
        ("palm down: -90 deg about y", HOLD_POS, quat_of_axis_angle([0, 1, 0], -np.pi / 2)),
        ("palm down: +90 deg about y", HOLD_POS, quat_of_axis_angle([0, 1, 0], np.pi / 2)),
        ("thumb up: -90 deg about x", HOLD_POS, quat_of_axis_angle([1, 0, 0], -np.pi / 2)),
        ("yawed: +45 deg about z", HOLD_POS, quat_of_axis_angle([0, 0, 1], np.pi / 4)),
    ]


def asked_deg(quat) -> float:
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(quat[0])), -1, 1))))


def angle_between(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def run_leg(p, reset, tap, side, pos, quat):
    """One clean standing start, hold the target, then read what arrived."""
    p.cmd.hold(0.5, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()  # clears the previous leg's wrist target
    reset.reset(label="wrist")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    if p.obs()["fallen"]:
        return dict(fell="reset")
    p.cmd.set_wrists(**{f"{side}_pos": pos, f"{side}_quat": quat})
    # `frame()` re-sends manager_state every tick, so the stream mode is
    # asserted continuously -- the reset cannot leave it stale.
    end = time.time() + SETTLE
    while time.time() < end:
        p.cmd.frame(mode=LocomotionMode.IDLE)
        time.sleep(DT)
    s = p.obs()
    if s["fallen"]:
        return dict(fell=True)
    palm = s["palms"][side]
    axes = Primitives._rot(s["pelvis"]["quat"]).T @ Primitives._rot(palm["quat"])
    arm = list(ARM_MJ[side])
    ref = tap.q29() if tap is not None else None
    return dict(
        commanded_pos=[float(v) for v in pos],
        asked_deg=asked_deg(quat),
        palm_root=[float(v) for v in p.world_to_root_full(palm["pos"], s)],
        axes=axes,
        measured_arm=[float(v) for v in np.asarray(s["joints"])[arm]],
        reference_arm=None if ref is None else [float(v) for v in np.asarray(ref)[arm]],
    )


def report(rows, labels):
    rest, reach = rows[0], rows[1]
    if rest.get("fell") or reach.get("fell"):
        print("\nA CONTROL LEG FELL -- nothing here is a measurement.")
        return {}
    asked_move = float(np.linalg.norm(np.asarray(reach["commanded_pos"]) - np.asarray(rest["commanded_pos"])))
    got_move = float(np.linalg.norm(np.asarray(reach["palm_root"]) - np.asarray(rest["palm_root"])))
    resid = float(np.linalg.norm(np.asarray(reach["palm_root"]) - np.asarray(reach["commanded_pos"])))
    print(f"\n== 1. POSITION (the gate): asked the wrist to move {asked_move:.3f} m, palm moved {got_move:.3f} m")
    print(f"   ratio {got_move / max(asked_move, 1e-9):.2f};  residual to the commanded point {resid:.3f} m")
    ref_moved = None
    if rest.get("reference_arm") and reach.get("reference_arm"):
        ref_moved = float(np.max(np.abs(np.asarray(reach["reference_arm"]) - np.asarray(rest["reference_arm"]))))
        print(
            f"== 2. THE DEPLOY'S OWN REFERENCE moved {ref_moved:.4f} rad on the {len(rest['reference_arm'])} arm joints"
        )
    else:
        print("== 2. no g1_debug reference (RefTap): cannot separate 'never asked' from 'did not follow'")
    live = got_move >= POSITION_LIVE_M
    if not live:
        verdict = "INERT upstream (the deploy never moved its reference)"
        if ref_moved is not None and ref_moved >= REFERENCE_LIVE_RAD:
            verdict = "the reference MOVED but the arm did not follow -- a tracking problem, not the field"
        print(f"\nCHANNEL NOT LIVE: palm moved {got_move:.3f} m < {POSITION_LIVE_M}. {verdict}.")
        print("Rotation cannot be measured while the position target does nothing. STOPPING.")
        return dict(position_live=False, position_ratio=got_move / max(asked_move, 1e-9), reference_moved=ref_moved)

    print("\n== 3. ROTATION, referenced to the `reach` leg (same position, identity rotation)")
    ratios = []
    for label, r in zip(labels[2:], rows[2:]):
        if r.get("fell"):
            print(f"  {label:34s} FELL -- excluded")
            continue
        delivered = angle_between(reach["axes"], r["axes"])
        ratio = delivered / r["asked_deg"] if r["asked_deg"] > 1e-6 else float("nan")
        ratios.append(ratio)
        r["delivered_deg"], r["ratio"] = delivered, ratio
        print(f"  {label:34s} asked {r['asked_deg']:5.1f}  delivered {delivered:5.1f}  ratio {ratio:5.2f}")
    if not ratios:
        return dict(position_live=True, position_ratio=got_move / max(asked_move, 1e-9))
    med = float(np.median(ratios))
    print(
        f"\n  median delivery ratio {med:.2f} over {len(ratios)} commands, spread {min(ratios):.2f}..{max(ratios):.2f}"
    )
    print(f"  a pre-scale inverting it would be {1 / med:.1f}; primitives_tuning.TRACK_ROT_BOOST is 4.5 (release era)")
    print("  a single scalar boost only makes sense if that spread is narrow")
    return dict(
        position_live=True,
        position_ratio=got_move / max(asked_move, 1e-9),
        position_residual_m=resid,
        reference_moved=ref_moved,
        rotation_ratio_median=med,
        rotation_ratio_span=[min(ratios), max(ratios)],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--json", default=None, help="write the per-leg numbers here")
    args = ap.parse_args()

    p = Primitives()
    reset = EpisodeControlClient()
    try:
        from locomani.backends.sonic.upper_ref import RefTap

        tap = RefTap().start()
    except Exception as exc:  # the tap is diagnostic, not required
        print(f"[diag] no RefTap ({exc}) -- continuing without the reference check")
        tap = None

    print(f"\npolicy {paths.SONIC_POLICY}   side {args.side}")
    table = legs(args.side)
    rows, labels = [], [label for label, _, _ in table]
    for label, pos, quat in table:
        row = run_leg(p, reset, tap, args.side, pos, quat)
        rows.append(row)
        state = "FELL" if row.get("fell") else f"palm {np.round(row['palm_root'], 3)}"
        print(f"  {label:34s} {state}")
        if row.get("fell") == "reset":
            print("  could not stand up for this leg -- stopping")
            break
    summary = report(rows, labels) if len(rows) >= 2 else {}

    if args.json:
        out = dict(policy=paths.SONIC_POLICY, side=args.side, summary=summary, legs=[])
        for label, r in zip(labels, rows):
            out["legs"].append({"case": label, **{k: v for k, v in r.items() if k != "axes"}})
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {args.json}")
    if tap is not None:
        tap.stop()
    p.cmd.pause_control()
    p.close()


if __name__ == "__main__":
    main()

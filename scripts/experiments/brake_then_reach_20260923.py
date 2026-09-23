"""Guard-brake to a line, then reach both hands to the envelope's farthest target and measure how far past the line they get.

Walk scene, one live stack (see brake_then_reach_20260923.sh). Per trial:
  1. diag_brake.trial: SLOW_WALK 0.25 toward a line --marker m ahead, guard brake (nothing may cross it)
  2. crate.expert.prepare_stationary_ready with a no-op placement: the visual expert's ready arms, in place
  3. GhostReachExecutor on a fixed target in the settled robot frame, under the envelope run's control config
     (outputs/g1_capability_envelope_20260921: x 0.60 passed, 0.65 failed at h 0.70 / d 0.44 / y 0 / yaw 0)
  4. the grasp tool points and the frontmost hand geom, measured against the line

The ghost-reach executor and its envelope config are imported from the working tree
(tasks/crate/ghost_reach_execution*.py), read only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.backends.sonic.diag_brake import trial as brake_trial
from locomani.backends.sonic.primitives import Primitives
from locomani.primitives.arm_ik import ArmIK
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.expert import prepare_stationary_ready
from locomani.tasks.crate.ghost_reach_execution import (
    GhostReachCase,
    GhostReachCriteria,
    GhostReachExecutor,
    MeasuredCollisionChecker,
    original_reach_control_config,
)
from locomani.tasks.crate.ghost_reach_execution_cli import SonicGhostReachRuntime

ENVELOPE = json.loads(
    Path(paths.REPO, "outputs/g1_capability_envelope_20260921/boundary/round_5/shard_0/request.json").read_text()
)["envelope"]
GUARD = dict(extra=0.20, margin=0.03, stride0=0.14)  # the 50/50 constant-0.25 validation (brake_guard025_valid)


def case_at(x, h=0.70, d=0.44, index=1):
    """The envelope bank's sample shape (bank.json joint_5d_000068), midpoint x in the settled robot frame."""
    return GhostReachCase.from_source(
        {
            "index": index,
            "label_zh": f"reach_x{x:.2f}",
            "parameters": {
                "pair_midpoint_x_m": x,
                "pair_midpoint_y_m": 0.0,
                "ground_height_z_m": h,
                "pair_separation_m": d,
                "pair_line_yaw_deg": 0.0,
            },
            "capture_centres_xyz": {"left": [x, d / 2, h], "right": [x, -d / 2, h]},
            "orientation_endpoints": {},
            "bodyward_tilt_deg": 30.0,
        }
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", default="0.60,0.70", help="reach midpoint x values (m, settled robot frame)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--marker", type=float, default=0.9)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "traces").mkdir(exist_ok=True)
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    ik = ArmIK(paths.SCENE_XML)
    solver = BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all")
    checker = MeasuredCollisionChecker(ArmIK(paths.SCENE_XML))
    control = original_reach_control_config(envelope=dict(ENVELOPE))
    criteria = GhostReachCriteria()
    rows = []
    for x in [float(v) for v in a.targets.split(",")]:
        for i in range(a.repeats):
            if (a.out / "STOP").exists():
                break
            label = f"reach_x{x:.2f}_{i:02d}"
            row = dict(target_x=x, i=i, label=label)
            brake = brake_trial(p, reset, 0.25, a.marker, 3.0, f"{label}_brake", guard=GUARD)
            row["brake"] = brake
            if brake.get("fell") is not None or "origin_xy" not in brake:
                rows.append(row)
                continue
            h = np.array([np.cos(brake["heading"]), np.sin(brake["heading"])])
            line_xy = np.asarray(brake["origin_xy"]) + h * a.marker
            # One recording from here to the end of the reach: the arm preparation between the two is ~13 s of
            # simulation with mode switches in it, and leaving it out made the arms appear to jump.
            reset.start_video(f"{label}_prep")
            try:
                arm, prep = prepare_stationary_ready(
                    p,
                    reset,
                    control.visual,
                    solver,
                    placement=lambda _reset_id: None,  # stay where the brake left the robot
                    whole_body=True,
                    heading=brake["heading"],
                )
                runtime = SonicGhostReachRuntime(p, ik, planner_arm=arm)
                state = prep.get("admission_state") or runtime.observe()
                row["ready_pelvis_past_line_cm"] = 100 * float((np.asarray(state["pelvis"]["pos"][:2]) - line_xy) @ h)
                try:
                    result = GhostReachExecutor(
                        solver, runtime, criteria=criteria, control_config=control, collision_checker=checker
                    ).run_case(case_at(x), state)
                finally:
                    reset.end_video()
            except Exception as error:  # noqa: BLE001 - one bad trial must not end the run
                reset.end_video()
                row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                continue
            tools = np.array([t["measured_tools_world_m"] for t in result["traces"]])  # [T, 2, 3]
            past = (tools[:, :, :2] - line_xy) @ h  # per frame, per hand: + = past the line
            row.update(
                status=result["status"],
                reason=result["reason"],
                target_past_line_cm=[
                    100 * float((c[:2] - line_xy) @ h) for c in result["target"]["fixed_centres_world_m"]
                ],
                tool_max_past_line_cm=[100 * float(v) for v in past.max(axis=0)],
                tool_final_past_line_cm=[100 * float(v) for v in past[-1]],
                best_position_error_cm=100 * float(min(max(t["position_error_m"]) for t in result["traces"])),
                foot_drift_cm=100 * float(max(max(t["foot_horizontal_drift_m"]) for t in result["traces"])),
                line_xy=line_xy.tolist(),
                heading=brake["heading"],
            )
            (a.out / "traces" / f"{label}.json").write_text(json.dumps(result))
            rows.append(row)
            print(
                f"x {x:.2f} #{i}: {row['status']:8s} {row['reason']:28s} pelvis {row['ready_pelvis_past_line_cm']:+.1f} "
                f"| tools max past line L {row['tool_max_past_line_cm'][0]:+.1f} R {row['tool_max_past_line_cm'][1]:+.1f} cm "
                f"| best err {row['best_position_error_cm']:.1f} cm, foot drift {row['foot_drift_cm']:.1f}",
                flush=True,
            )
            (a.out / "results.json").write_text(json.dumps(dict(policy=paths.SONIC_POLICY, rows=rows), indent=1))
            time.sleep(0.5)
    (a.out / "results.json").write_text(json.dumps(dict(policy=paths.SONIC_POLICY, rows=rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

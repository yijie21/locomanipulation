#!/usr/bin/env python3
"""Replay settled fixed cases through the loaded standing-endpoint admission."""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.primitives import upper_body_from_mj
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.supported_motion import SupportedMotion
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle.ready_grasp_policy import OFFSET
from locomani.tasks.bottle.transport_guard import TransportGuard
from locomani.tasks.bottle.walking_reach import WalkingReach
from locomani.tasks.bottle.whole_body_policy import LoadedEndpointEvaluator, WholeBodyGraspPolicy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--settled-report", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="configs/tasks/bottle_whole_body.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank = json.loads(Path(args.bank).read_text())
    settled_report = json.loads(Path(args.settled_report).read_text())
    settled_by_case = {case["case_id"]: case["settled"] for case in settled_report["cases"]}
    scene_meta = json.loads((Path(args.scene).parent / "scene_meta.json").read_text())
    reach = BottleTaskConfig.load(args.config).reach
    cfg = reach.whole_body

    target_key = "water_bottle"
    target_meta = scene_meta["objects"][target_key]
    surface_bodies = tuple(item["body"] for item in scene_meta["surfaces"].values())
    ik = ArmIK(str(args.scene))
    endpoint_ik = ArmIK(str(args.scene))
    endpoint_guard = TransportGuard(
        endpoint_ik,
        target_key,
        target_meta["body"],
        surface_bodies,
        margin=reach.transport_margin,
        joint_step=reach.sweep_joint_step,
        include_body=reach.ready_grasp_recovery,
        finger_only_target_contact=True,
    )
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = cfg
    policy.task = SimpleNamespace(target=SimpleNamespace(key=target_key))
    evaluator = LoadedEndpointEvaluator(endpoint_ik, endpoint_guard, cfg)

    cases = []
    for source in bank["cases"]:
        case_id = source["case_id"]
        settled = settled_by_case[case_id]
        if not settled["feasible"]:
            cases.append({"case_id": case_id, "admitted": False, "reason": "settled_endpoint_infeasible"})
            continue

        state = deepcopy(source["state"])
        joints = np.asarray(settled["joints"], dtype=float)
        root_position = np.asarray(settled["root_position"], dtype=float)
        root_rotation = np.asarray(settled["root_rotation"], dtype=float)
        root_quaternion = Rotation.from_matrix(root_rotation).as_quat()[[3, 0, 1, 2]]
        state["joints"] = joints.copy()
        state["pelvis"] = dict(state["pelvis"])
        state["pelvis"].update(pos=root_position.copy(), quat=root_quaternion.copy())

        motion = SupportedMotion(
            ik,
            state,
            cfg,
            joints,
            root_quaternion,
            tool_offset=OFFSET,
            hand_weights=(0.35, 1.0),
        )
        motion.body.put(motion.x)
        targets = motion.body.tools(motion.x).copy()
        rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
        rotations[1] = np.asarray(settled["grasp_rotation"], dtype=float)
        walking = WalkingReach(
            ik,
            cfg,
            upper_body_from_mj(np.asarray(source["takeover_reference"], dtype=float)),
        )
        policy.walking = walking
        request = policy._loaded_endpoint_request(
            state,
            motion,
            targets,
            rotations,
            np.asarray(settled["grasp_rotation"], dtype=float),
        )
        wall_started = time.perf_counter()
        result = evaluator.evaluate(request["payload"])
        result.update(
            case_id=case_id,
            replay_seconds=time.perf_counter() - wall_started,
            path=request["path"],
            request_source=request["contract"]["source"],
        )
        cases.append(result)

    solve_seconds = [case["solve_seconds"] for case in cases if "solve_seconds" in case]
    report = {
        "schema_version": 2,
        "meaning": (
            "CPU kinematic and collision admission of the Stage3 loaded endpoint from each settled fixed-case "
            "pose. It checks fixed-foot support, COM, hand IK, attached-object collision, and a standing-posture "
            "optimization prior. Standing leg residuals are diagnostics, not identified tracking limits. This "
            "single-snapshot replay does not establish measured path tracking, loaded stability, controller "
            "consumption, or physical success."
        ),
        "inputs": {
            "bank": str(Path(args.bank).resolve()),
            "bank_sha256": digest(args.bank),
            "settled_report": str(Path(args.settled_report).resolve()),
            "settled_report_sha256": digest(args.settled_report),
            "scene": str(Path(args.scene).resolve()),
            "scene_sha256": digest(args.scene),
            "config": str(Path(args.config).resolve()),
            "config_sha256": digest(args.config),
            "implementation_sha256": {
                path: digest(path)
                for path in (
                    "scripts/experiments/bottle_loaded_endpoint_probe_20260916.py",
                    "src/locomani/tasks/bottle/loaded_handback.py",
                    "src/locomani/tasks/bottle/whole_body_config.py",
                    "src/locomani/tasks/bottle/whole_body_policy.py",
                    "src/locomani/primitives/supported_motion.py",
                )
            },
        },
        "summary": {
            "cases": len(cases),
            "admitted": sum(bool(case["admitted"]) for case in cases),
            "solve_seconds": {
                "median": float(np.median(solve_seconds)),
                "maximum": float(np.max(solve_seconds)),
            },
            "solve_exceeds_terminal_max_gap": sum(
                float(case.get("solve_seconds", 0.0)) > cfg.capture_terminal_max_gap_seconds for case in cases
            ),
        },
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=json_value) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

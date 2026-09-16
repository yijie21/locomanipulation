#!/usr/bin/env python3
"""Replay the fixed bottle post-stop bank through asynchronous stance reach checks."""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle.stance_reach import AsyncStanceReachPlanner, StanceReachEvaluator
from locomani.tasks.bottle.transport_guard import TransportGuard


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def duration_summary(values):
    values = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(values)), "maximum": float(np.max(values))}


def wait_for(planner, kind, key, overhead, timeout):
    started = time.monotonic()
    while time.monotonic() - started <= timeout:
        tick = time.perf_counter()
        response = planner.poll(kind, key)
        overhead.append(time.perf_counter() - tick)
        if response["status"] == "ready":
            return response["result"]
        if response["status"] in ("closed", "missing"):
            raise RuntimeError(f"{kind} request {key} became {response['status']}")
        time.sleep(0.005)
    raise TimeoutError(f"{kind} request {key} exceeded {timeout:.1f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="configs/tasks/bottle_whole_body.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--verification",
        choices=("full_path", "endpoint_only"),
        default="full_path",
        help="full offline Cartesian audit or the endpoint-first live planning scope",
    )
    args = parser.parse_args()
    verify_path = args.verification == "full_path"

    bank = json.loads(Path(args.bank).read_text())
    scene_meta = json.loads((Path(args.scene).parent / "scene_meta.json").read_text())
    reach = BottleTaskConfig.load(args.config).reach
    cfg = reach.whole_body
    target_key = "water_bottle"
    target_meta = scene_meta["objects"][target_key]
    surface_bodies = tuple(item["body"] for item in scene_meta["surfaces"].values())

    ik = ArmIK(args.scene)
    guard = TransportGuard(
        ik,
        target_key,
        target_meta["body"],
        surface_bodies,
        margin=reach.transport_margin,
        joint_step=reach.sweep_joint_step,
        include_body=reach.ready_grasp_recovery,
        finger_only_target_contact=True,
    )

    def feasibility(
        state,
        joints,
        candidate_base,
        *,
        allow_target_contact=False,
        path=None,
        endpoint_only=False,
    ):
        checks = []
        sweep_state = deepcopy(state)
        if endpoint_only:
            sweep_state["joints"] = np.asarray(joints, dtype=float).copy()
            sweep_state["pelvis"] = dict(sweep_state["pelvis"])
            base = np.asarray(candidate_base, dtype=float)
            sweep_state["pelvis"].update(pos=base[:3].copy(), quat=base[3:].copy())
        waypoints = [{"joints": joints, "candidate_base": candidate_base}] if path is None else path
        for path_index, waypoint in enumerate(waypoints):
            waypoint_joints = np.asarray(waypoint["joints"], dtype=float)
            waypoint_base = np.asarray(waypoint["candidate_base"], dtype=float)
            waypoint_target_contact = bool(
                allow_target_contact and (path is None or waypoint.get("allow_target_contact", False))
            )
            for side in ("right", "left", "body"):
                check = guard.check(
                    sweep_state,
                    waypoint_joints,
                    side,
                    candidate_base=waypoint_base,
                    allow_target_contact=waypoint_target_contact,
                )
                checks.append({"path_index": path_index, "side": side, **check.to_dict()})
                if not check.clear:
                    return {
                        "clear": False,
                        "reason": f"{side}:{check.link}:{check.obstacle}",
                        "path_source": None if path is None else "supported_cap_clearing_cartesian_reference",
                        "checks": checks,
                    }
            sweep_state["joints"] = waypoint_joints.copy()
            sweep_state["pelvis"] = dict(sweep_state["pelvis"])
            sweep_state["pelvis"].update(pos=waypoint_base[:3].copy(), quat=waypoint_base[3:].copy())
        return {
            "clear": True,
            "reason": None,
            "path_source": None if path is None else "supported_cap_clearing_cartesian_reference",
            "path_samples": len(waypoints),
            "checks": checks,
        }

    evaluator = StanceReachEvaluator(ik, cfg, feasibility)
    planner = AsyncStanceReachPlanner(evaluator)
    overhead = []
    cases = []
    try:
        for index, source in enumerate(bank["cases"]):
            state = source["state"]
            target = np.asarray(source["camera_admitted_target"], dtype=float)
            q = np.asarray(source["takeover_reference"], dtype=float)
            candidate_base = np.r_[
                np.asarray(state["pelvis"]["pos"], dtype=float),
                np.asarray(source["takeover_root"], dtype=float),
            ]
            root = Rotation.from_quat(np.asarray(state["pelvis"]["quat"], dtype=float)[[1, 2, 3, 0]]).as_matrix()
            desired_rotation = Rotation.from_euler("z", np.arctan2(root[1, 0], root[0, 0])).as_matrix()
            # The bank has one measured takeover posture.  Replay the recorded
            # endpoint and the owning walking search's two outer longitudinal
            # alternatives; q remains the recorded posture, an explicit probe
            # limitation recorded below.
            candidates = []
            for endpoint_shift in (0.0, -cfg.stance_search_span, cfg.stance_search_span):
                shifted_base = candidate_base.copy()
                shifted_base[:3] += endpoint_shift * desired_rotation[:, 0]
                candidates.append(
                    {
                        "q": q,
                        "candidate_base": shifted_base,
                        "score": abs(float(endpoint_shift)),
                        "probe_endpoint_shift_m": float(endpoint_shift),
                    }
                )
            tick = time.perf_counter()
            submitted = planner.request(
                "walking",
                index,
                {
                    "state": state,
                    "target": target,
                    "candidates": candidates,
                    "verify_path": verify_path,
                },
            )
            overhead.append(time.perf_counter() - tick)
            walking = wait_for(planner, "walking", index, overhead, args.timeout)

            settled_key = len(bank["cases"]) + index
            tick = time.perf_counter()
            settled_submit = planner.request(
                "settled",
                settled_key,
                {
                    "state": state,
                    "target": target,
                    "desired_rotation": desired_rotation,
                    "posture": np.asarray(state["joints"], dtype=float),
                    "reference_joints": q,
                    "reference_root": candidate_base[3:],
                    "verify_path": verify_path,
                },
                priority=True,
            )
            overhead.append(time.perf_counter() - tick)
            settled = wait_for(planner, "settled", settled_key, overhead, args.timeout)
            settled_reference_offset = float(np.max(np.abs(q - np.asarray(state["joints"], dtype=float))))
            cases.append(
                {
                    "case_id": source["case_id"],
                    "old_outcome": source["old_outcome"],
                    "settled_request_stability": "unverified_single_snapshot",
                    "settled_reference_measurement_offset_linf_rad": settled_reference_offset,
                    "walking_submit_status": submitted["status"],
                    "settled_submit_status": settled_submit["status"],
                    "walking": walking,
                    "settled": settled,
                }
            )
    finally:
        planner.close()

    overhead_array = np.asarray(overhead, dtype=float)
    scenario_names = ("expected", "forward_yaw_left", "forward_yaw_right")
    walking_times = np.asarray([item["walking"]["solve_seconds"] for item in cases])
    settled_times = np.asarray([item["settled"]["batch_solve_seconds"] for item in cases])
    settled_feasible_times = [item["settled"]["batch_solve_seconds"] for item in cases if item["settled"]["feasible"]]
    settled_component_names = (
        "final_ik_seconds",
        "cartesian_rollout_seconds",
        "clearance_seconds",
    )
    rollout_detail_seconds = {}
    for name in (
        "setup_seconds",
        "ik_step_seconds",
        "terminal_fk_check_seconds",
        "terminal_convergence_seconds",
    ):
        values = [
            item["settled"]["reach_path"].get(name)
            for item in cases
            if item["settled"]["reach_path"].get(name) is not None
        ]
        rollout_detail_seconds[name] = duration_summary(values) if values else None
    summary = {
        "cases": len(cases),
        "walking_feasible": sum(bool(item["walking"]["feasible"]) for item in cases),
        "settled_feasible": sum(bool(item["settled"]["feasible"]) for item in cases),
        "settled_request_stability_verified": 0,
        "recorded_candidate_scenario_feasible": {
            name: sum(
                bool(item["walking"]["checked"][0]["scenarios"].get(name, {}).get("feasible", False)) for item in cases
            )
            for name in scenario_names
        },
        "any_candidate_scenario_feasible": {
            name: sum(
                any(
                    bool(candidate["scenarios"].get(name, {}).get("feasible", False))
                    for candidate in item["walking"]["checked"]
                )
                for item in cases
            )
            for name in scenario_names
        },
        "walking_expected_cartesian_path_verified": sum(
            any(
                bool(candidate["scenarios"].get("expected", {}).get("reach_path", {}).get("complete", False))
                for candidate in item["walking"]["checked"]
            )
            for item in cases
        ),
        "settled_cartesian_path_verified": sum(
            bool(item["settled"].get("reach_path", {}).get("complete", False)) for item in cases
        ),
        "walking_batch_seconds": {
            "median": float(np.median(walking_times)),
            "maximum": float(np.max(walking_times)),
        },
        "settled_solve_seconds": {
            "median": float(np.median(settled_times)),
            "maximum": float(np.max(settled_times)),
        },
        "settled_feasible_solve_seconds": (
            duration_summary(settled_feasible_times) if settled_feasible_times else None
        ),
        "settled_component_seconds": {
            name: duration_summary(item["settled"][name] for item in cases) for name in settled_component_names
        },
        "settled_rollout_detail_seconds": rollout_detail_seconds,
        "submit_poll_seconds": {
            "samples": int(overhead_array.size),
            "median": float(np.median(overhead_array)),
            "p95": float(np.quantile(overhead_array, 0.95)),
            "maximum": float(np.max(overhead_array)),
        },
    }
    report = {
        "schema_version": 1,
        "meaning": (
            f"CPU {args.verification} kinematic replay of each recorded takeover and +/- stance-search-span "
            "endpoint alternatives per fixed case. Alternatives retain the recorded arm posture. "
            "Single snapshots cannot establish the measured stability required for a settled "
            "request. The nominal path begins from the recorded owned reference, preserves its "
            "measurement offset, and retains the live per-tick guard requirement; this replay "
            "does not establish dynamic support stability."
        ),
        "inputs": {
            "bank": str(Path(args.bank).resolve()),
            "bank_sha256": digest(args.bank),
            "scene": str(Path(args.scene).resolve()),
            "scene_sha256": digest(args.scene),
            "config": str(Path(args.config).resolve()),
            "config_sha256": digest(args.config),
            "verification": args.verification,
            "implementation_sha256": {
                path: digest(path)
                for path in (
                    "scripts/experiments/bottle_stance_reach_probe_20260915.py",
                    "src/locomani/tasks/bottle/stance_reach.py",
                    "src/locomani/tasks/bottle/walking_reach.py",
                    "src/locomani/tasks/bottle/whole_body_policy.py",
                )
            },
        },
        "config": asdict(cfg),
        "summary": summary,
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=json_value) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

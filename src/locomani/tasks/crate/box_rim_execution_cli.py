"""Child-process entry for one measured generic open-box grasp probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from locomani.backends.sonic import upper_ref
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.backends.sonic.primitives import Primitives, upper_body_from_mj
from locomani.backends.sonic.streamed_arm import reanchor
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate import box_rim_cli
from locomani.tasks.crate.box_rim_execution import (
    HAND_READY,
    BoxRimExecutor,
    SonicBoxRimRuntime,
    execution_config_from_dict,
)
from locomani.tasks.crate.box_rim_planner import BoxRimPlanner

RUNTIME_FIELDS = {
    "backend",
    "gpu",
    "instance",
    "scene_dir",
    "reset_before_episode",
    "robot_dxy",
    "robot_yaw",
    "initial_ready_pose",
    "video_label",
}

INITIAL_READY_FIELDS = {"waist", "left_arm", "right_arm", "hands"}


def _finite_values(value, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _parse_initial_ready_pose(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("runtime.initial_ready_pose must be an object or null")
    box_rim_cli._keys(value, INITIAL_READY_FIELDS, "runtime.initial_ready_pose")
    box_rim_cli._required(value, INITIAL_READY_FIELDS, "runtime.initial_ready_pose")
    return {
        "waist": _finite_values(value["waist"], 3, "runtime.initial_ready_pose.waist"),
        "left_arm": _finite_values(value["left_arm"], 7, "runtime.initial_ready_pose.left_arm"),
        "right_arm": _finite_values(value["right_arm"], 7, "runtime.initial_ready_pose.right_arm"),
        "hands": _finite_values(value["hands"], 7, "runtime.initial_ready_pose.hands"),
    }


def _ready_reference(joints, pose: dict) -> np.ndarray:
    reference = np.asarray(joints, dtype=float).copy()
    if reference.shape != (29,) or not np.isfinite(reference).all():
        raise ValueError("observed initialization joints must contain 29 finite values")
    reference[12:15] = pose["waist"]
    reference[15:22] = pose["left_arm"]
    reference[22:29] = pose["right_arm"]
    return reference


def load_execution_request(path: Path) -> dict:
    raw = json.loads(path.read_text())
    allowed = {
        "schema_version",
        "scene_xml",
        "target",
        "box",
        "robot_state",
        "planner",
        "render",
        "execution",
        "target_key",
        "runtime",
    }
    box_rim_cli._keys(raw, allowed, "request")
    box_rim_cli._required(
        raw,
        ("schema_version", "scene_xml", "target", "box", "robot_state", "target_key", "runtime"),
        "request",
    )
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(raw["target_key"], str) or not raw["target_key"]:
        raise ValueError("target_key must be non-empty text")
    runtime = raw["runtime"]
    box_rim_cli._keys(runtime, RUNTIME_FIELDS, "runtime")
    box_rim_cli._required(runtime, ("backend", "gpu", "instance", "scene_dir"), "runtime")
    if not isinstance(runtime["backend"], dict):
        raise ValueError("runtime.backend must be an object")
    backend = SonicBackendConfig.from_dict(runtime["backend"])
    if backend.no_encoder:
        raise ValueError("runtime.backend.no_encoder must be false for PlannerArm/POSE execution")
    if not isinstance(runtime["instance"], int) or isinstance(runtime["instance"], bool) or runtime["instance"] < 0:
        raise ValueError("runtime.instance must be a nonnegative integer")
    if not isinstance(runtime["gpu"], (str, int)):
        raise ValueError("runtime.gpu must be a string or integer")
    scene_dir = (path.parent / runtime["scene_dir"]).resolve()
    if not (scene_dir / "scene_locomani.xml").is_file() or not (scene_dir / "scene_meta.json").is_file():
        raise ValueError("runtime.scene_dir must contain scene_locomani.xml and scene_meta.json")
    scene_xml = (path.parent / raw["scene_xml"]).resolve()
    if not scene_xml.is_file():
        raise ValueError(f"scene_xml does not exist: {scene_xml}")
    if scene_xml != (scene_dir / "scene_locomani.xml").resolve():
        raise ValueError("scene_xml must be runtime.scene_dir/scene_locomani.xml")
    target = raw["target"]
    box_rim_cli._keys(target, ("body", "collider_mode"), "target")
    box_rim_cli._required(target, ("body", "collider_mode"), "target")
    if target["collider_mode"] != "matched_target":
        raise ValueError("live execution requires target.collider_mode=matched_target")
    scene_meta = json.loads((scene_dir / "scene_meta.json").read_text())
    objects = scene_meta.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("runtime.scene_dir/scene_meta.json objects must be an object")
    target_meta = objects.get(raw["target_key"])
    if not isinstance(target_meta, dict):
        raise ValueError("target_key must name an object in runtime.scene_dir/scene_meta.json")
    if target_meta.get("body") != target["body"]:
        raise ValueError("target.body must match scene_meta.objects[target_key].body")
    robot_dxy = runtime.get("robot_dxy", [0.0, 0.0])
    if not isinstance(robot_dxy, (list, tuple)) or len(robot_dxy) != 2:
        raise ValueError("runtime.robot_dxy must contain two values")
    robot_dxy = [float(value) for value in robot_dxy]
    robot_yaw = float(runtime.get("robot_yaw", 0.0))
    if not all(math.isfinite(value) for value in [*robot_dxy, robot_yaw]):
        raise ValueError("runtime.robot_dxy and robot_yaw must be finite")
    reset_before_episode = runtime.get("reset_before_episode", True)
    if not isinstance(reset_before_episode, bool):
        raise ValueError("runtime.reset_before_episode must be boolean")
    initial_ready_pose = _parse_initial_ready_pose(runtime.get("initial_ready_pose"))
    if initial_ready_pose is not None and not reset_before_episode:
        raise ValueError("runtime.initial_ready_pose requires reset_before_episode=true")
    video_label = runtime.get("video_label")
    if video_label is not None and (not isinstance(video_label, str) or not video_label):
        raise ValueError("runtime.video_label must be non-empty text or null")
    runtime = dict(runtime)
    runtime["scene_dir"] = scene_dir
    runtime["robot_dxy"] = robot_dxy
    runtime["robot_yaw"] = robot_yaw
    runtime["reset_before_episode"] = reset_before_episode
    runtime["initial_ready_pose"] = initial_ready_pose
    return {
        "path": path.resolve(),
        "scene_xml": scene_xml,
        "target": dict(target),
        "box": box_rim_cli._parse_box(raw["box"]),
        "planner": box_rim_cli._parse_options(raw.get("planner", {})),
        "robot_raw": raw["robot_state"],
        "execution": execution_config_from_dict(raw.get("execution")),
        "target_key": raw["target_key"],
        "runtime": runtime,
    }


def _initialise(request, primitives, reset) -> dict | None:
    runtime = request["runtime"]
    if not runtime["reset_before_episode"]:
        return None
    pose = runtime["initial_ready_pose"]
    coordinated = getattr(request.get("execution"), "approach_mode", None) == "coordinated"
    if coordinated and pose is None:
        pose = {
            "waist": [0.0, 0.0, 0.0],
            "left_arm": [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
            "right_arm": [-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
            "hands": HAND_OPEN.tolist(),
        }
    hands = HAND_OPEN.copy() if coordinated else HAND_READY if pose is None else np.asarray(pose["hands"], dtype=float)
    primitives.cmd.reset_pose()
    primitives.cmd.set_hands(left=hands, right=hands)
    primitives.cmd.exit_streamed()
    if pose is None:
        reset.pin_arms()
    else:
        primitives.cmd.hold(
            1.5,
            mode=LocomotionMode.IDLE,
            facing=primitives._facing(primitives.obs()),
        )
        reset.pin_arms(
            waist=pose["waist"],
            left=pose["left_arm"],
            right=pose["right_arm"],
        )
    robot = {"dxy": runtime["robot_dxy"], "yaw": runtime["robot_yaw"]}
    if pose is not None:
        dof = np.zeros(29, dtype=float)
        dof[12:15] = pose["waist"]
        dof[15:22] = pose["left_arm"]
        dof[22:29] = pose["right_arm"]
        robot.update(dof=dof.tolist(), dof_from=12)
    reset.reset(
        robot=robot,
        label=runtime.get("video_label", "box_rim_probe"),
    )
    time.sleep(0.4)
    primitives.cmd.start_control()
    upper_ref.shared().wait(3.0)
    if pose is None:
        # The streamed toggle inside reanchor() spins the robot (measured 2026-09-22: 95 -> 126 deg
        # in this room), and holding the yaw measured AFTER it locked that error in -- every box-rim
        # round then planned 20-40 deg off the box. Re-place the robot after the re-anchor (the
        # recorded heading offset survives the teleport) and hold the REQUESTED yaw, as the
        # coordinated path already does through its second reset.
        reanchor(primitives)
        reset.reset(robot=robot, label=runtime.get("video_label", "box_rim_probe"))
        time.sleep(0.4)
        requested_yaw = float(runtime["robot_yaw"])
        primitives.hold_yaw = requested_yaw
        primitives.cmd.hold(
            2.5,
            mode=LocomotionMode.IDLE,
            facing=(math.cos(requested_yaw), math.sin(requested_yaw), 0.0),
        )
        return None
    reanchor(primitives)
    requested_yaw = float(runtime["robot_yaw"])
    primitives.hold_yaw = requested_yaw
    facing = (math.cos(requested_yaw), math.sin(requested_yaw), 0.0)
    primitives.cmd.hold(4.5, mode=LocomotionMode.IDLE, facing=facing)
    state = primitives.obs()
    return {
        "reference": _ready_reference(state["joints"], pose),
        "hands": hands.copy(),
        "requested": pose,
    }


def _initialization_report(request, meta, initial: dict, state: dict) -> dict:
    requested_xy = np.asarray(meta["robot_start"]["xy"], dtype=float) + np.asarray(
        request["runtime"]["robot_dxy"], dtype=float
    )
    measured_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
    requested_yaw = float(request["runtime"]["robot_yaw"])
    measured_yaw = float(state["pelvis"]["yaw"])
    yaw_error = math.atan2(
        math.sin(measured_yaw - requested_yaw),
        math.cos(measured_yaw - requested_yaw),
    )
    reference = np.asarray(initial["reference"], dtype=float)
    measured_joints = np.asarray(state["joints"], dtype=float)
    hands = np.asarray(initial["hands"], dtype=float)
    measured_hands = {side: np.asarray(state["palms"][side]["hand_q"], dtype=float) for side in ("left", "right")}
    object_state = state["objects"][request["target_key"]]
    requested_box_position = np.asarray(request["box"].pose.position, dtype=float)
    measured_box_position = np.asarray(object_state["pos"], dtype=float)
    return {
        "kind": "expert_ready_configured_then_live_settled",
        "requested_pelvis_xy_m": requested_xy.tolist(),
        "measured_pelvis_position_m": list(map(float, state["pelvis"]["pos"])),
        "pelvis_xy_error_m": (measured_xy - requested_xy).tolist(),
        "requested_yaw_rad": requested_yaw,
        "measured_yaw_rad": measured_yaw,
        "yaw_error_rad": yaw_error,
        "requested_upper_joints": reference[12:29].tolist(),
        "measured_upper_joints": measured_joints[12:29].tolist(),
        "upper_joint_error_rad": (measured_joints[12:29] - reference[12:29]).tolist(),
        "maximum_upper_joint_error_rad": float(np.max(np.abs(measured_joints[12:29] - reference[12:29]))),
        "requested_hands": hands.tolist(),
        "measured_active_hand_joints": {side: values.tolist() for side, values in measured_hands.items()},
        "maximum_active_hand_error_m": float(
            max(np.max(np.abs(values - hands[: len(values)])) for values in measured_hands.values())
        ),
        "requested_box_position_m": requested_box_position.tolist(),
        "measured_box_position_m": measured_box_position.tolist(),
        "box_position_error_m": (measured_box_position - requested_box_position).tolist(),
        "measured_box_quaternion_wxyz": list(map(float, object_state["quat"])),
        "measured_box_touching": list(object_state.get("touching", [])),
        "measured_box_contact_links": list(object_state.get("touch_links", [])),
        "simulator_pins_released": True,
        "standing_legs": "SONIC live settled; static review qpos0 legs were not commanded",
    }


def _reset_prepared_stance(request, primitives, reset, runtime_adapter, initial) -> None:
    """Re-place the robot after ready preparation, before recording or walking."""
    runtime = request["runtime"]
    reference = np.asarray(initial["reference"], dtype=float)
    reset.reset(
        robot={
            "dxy": runtime["robot_dxy"],
            "yaw": runtime["robot_yaw"],
            "dof": reference.tolist(),
            "dof_from": 12,
        },
        label=runtime.get("video_label", "box_rim_probe"),
    )
    time.sleep(0.3)
    primitives.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    runtime_adapter.ready_hands = HAND_OPEN.copy()
    runtime_adapter.planner_arm.hands = {side: HAND_OPEN.copy() for side in ("left", "right")}
    runtime_adapter.planner_arm.start(upper_reference=upper_body_from_mj(reference), settle_seconds=0.0)


def run_live(request: dict) -> dict:
    require_pose_ownership()
    meta = json.loads((request["runtime"]["scene_dir"] / "scene_meta.json").read_text())
    ik = ArmIK(str(request["scene_xml"]))
    # Validate the supplied planning snapshot even though execution always replaces
    # it with a fresh post-reset observation.
    box_rim_cli._parse_robot(request["robot_raw"], ik)
    planner = BoxRimPlanner(ik, request["target"]["body"], request["box"], request["planner"])
    primitives = Primitives(meta=meta)
    reset = EpisodeControlClient()
    runtime_adapter = None
    video_label = request["runtime"].get("video_label")
    video_started = False
    initialization = None
    coordinated = request["execution"].approach_mode == "coordinated"
    release_initial_pins = request["runtime"]["initial_ready_pose"] is not None or (
        coordinated and request["runtime"]["reset_before_episode"]
    )
    try:
        initial = _initialise(request, primitives, reset)
        if initial is None:
            if video_label:
                reset.start_video(video_label)
                video_started = True
            runtime_adapter = SonicBoxRimRuntime(primitives, ik, approach_mode=request["execution"].approach_mode)
        else:
            runtime_adapter = SonicBoxRimRuntime(
                primitives,
                ik,
                approach_mode=request["execution"].approach_mode,
                initial_reference=initial["reference"],
                initial_hands=initial["hands"],
            )
            runtime_adapter.planner_arm.hold(1.2)
            if coordinated:
                _reset_prepared_stance(request, primitives, reset, runtime_adapter, initial)
            reset.pin_arms()
            release_initial_pins = False
            runtime_adapter.planner_arm.hold(0.8)
            initialization = _initialization_report(request, meta, initial, runtime_adapter.observe())
            initialization["initialization_resets"] = 2 if coordinated else 1
            if video_label:
                initialization["video_start_simulation_time"] = float(runtime_adapter.observe()["t"])
                reset.start_video(video_label)
                video_started = True
        result = BoxRimExecutor(
            planner,
            request["box"],
            runtime_adapter,
            request["target_key"],
            request["execution"],
        ).execute()
        execution = dict(result.execution)
        execution["request"] = str(request["path"])
        execution["target_key"] = request["target_key"]
        execution["video_paths"] = []
        execution["trace_paths"] = []
        if initialization is not None:
            execution["initialization"] = initialization
        return {
            "ok": result.ok,
            "why": result.why,
            "produced": result.produced,
            "execution": execution,
        }
    finally:
        if release_initial_pins:
            reset.pin_arms()
        if video_started:
            reset.end_video()
        reset.close()
        if runtime_adapter is not None:
            runtime_adapter.close()
        else:
            primitives.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = load_execution_request(args.request.resolve())
        result = run_live(request)
    except Exception as error:  # noqa: BLE001 - child must leave an inspectable result
        result = {
            "ok": False,
            "why": "runtime_error",
            "produced": False,
            "execution": {"error": f"{type(error).__name__}: {error}"},
        }
    execution = result.setdefault("execution", {})
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("box_rim_execution.py").resolve(),
        Path(__file__).with_name("box_rim_execution_config.py").resolve(),
        Path(__file__).with_name("box_rim_collision.py").resolve(),
        Path(__file__).with_name("box_rim_planner.py").resolve(),
    ]
    execution["provenance"] = {
        "request_sha256": hashlib.sha256(args.request.resolve().read_bytes()).hexdigest(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        **(
            {
                "execution_config": asdict(request["execution"]),
                "planner_config": asdict(request["planner"]),
                "scene_xml": str(request["scene_xml"]),
            }
            if "request" in locals()
            else {}
        ),
    }
    traces = execution.pop("trace", [])
    if traces:
        trace_path = args.output.resolve().with_suffix(".trace.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("".join(json.dumps(row) + "\n" for row in traces))
        execution["trace_paths"] = [str(trace_path)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ok": result["ok"], "why": result["why"], "output": str(args.output)}), flush=True)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

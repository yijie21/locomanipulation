"""Offline JSON entry and PNG renderer for the static open-box rim planner."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np

from locomani.primitives.arm_ik import ArmIK
from locomani.tasks.crate.box_rim_geometry import ForbiddenRimInterval, OpenBoxGeometry, RigidPose
from locomani.tasks.crate.box_rim_planner import BoxRimPlanner, PlannerOptions, RobotState


def _keys(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _required(value, names, path):
    missing = [name for name in names if name not in value]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")


def _parse_box(value) -> OpenBoxGeometry:
    allowed = {
        "size_m",
        "wall_thickness_m",
        "pose",
        "local_com_m",
        "corner_clearance_m",
        "forbidden_intervals",
    }
    _keys(value, allowed, "box")
    _required(value, ("size_m", "wall_thickness_m", "pose"), "box")
    pose = value["pose"]
    _keys(pose, ("position", "quaternion_wxyz"), "box.pose")
    _required(pose, ("position", "quaternion_wxyz"), "box.pose")
    intervals = []
    for index, interval in enumerate(value.get("forbidden_intervals", [])):
        _keys(interval, ("rim", "start_m", "end_m"), f"box.forbidden_intervals[{index}]")
        _required(interval, ("rim", "start_m", "end_m"), f"box.forbidden_intervals[{index}]")
        intervals.append(ForbiddenRimInterval(**interval))
    return OpenBoxGeometry(
        value["size_m"],
        value["wall_thickness_m"],
        RigidPose(pose["position"], pose["quaternion_wxyz"]),
        local_com_m=value.get("local_com_m"),
        corner_clearance_m=value.get("corner_clearance_m", 0.025),
        forbidden_intervals=intervals,
    )


def _parse_options(value) -> PlannerOptions:
    _keys(value, {field.name for field in fields(PlannerOptions)}, "planner")
    converted = dict(value)
    for name in ("side_offsets_m", "depths_m", "palm_pitch_deg"):
        if name in converted:
            converted[name] = tuple(converted[name])
    return PlannerOptions(**converted)


def _scene_default_state(ik) -> RobotState:
    qpos = ik.model.qpos0
    return RobotState(
        qpos[ik.qadr],
        qpos[ik.base_qadr : ik.base_qadr + 3],
        qpos[ik.base_qadr + 3 : ik.base_qadr + 7],
    )


def _parse_robot(value, ik) -> RobotState:
    _keys(value, ("source", "joints", "pelvis"), "robot_state")
    _required(value, ("source",), "robot_state")
    if value["source"] == "scene_default":
        if set(value) != {"source"}:
            raise ValueError("scene_default robot_state cannot contain joints or pelvis")
        return _scene_default_state(ik)
    if value["source"] != "explicit":
        raise ValueError("robot_state.source must be scene_default or explicit")
    _required(value, ("joints", "pelvis"), "robot_state")
    pelvis = value["pelvis"]
    _keys(pelvis, ("position", "quaternion_wxyz"), "robot_state.pelvis")
    _required(pelvis, ("position", "quaternion_wxyz"), "robot_state.pelvis")
    return RobotState(value["joints"], pelvis["position"], pelvis["quaternion_wxyz"])


def load_request(path: Path) -> dict:
    raw = json.loads(path.read_text())
    allowed = ("schema_version", "scene_xml", "target", "box", "robot_state", "planner", "render")
    _keys(raw, allowed, "request")
    _required(raw, ("schema_version", "scene_xml", "target", "box", "robot_state"), "request")
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    target = raw["target"]
    _keys(target, ("body", "collider_mode"), "target")
    _required(target, ("body", "collider_mode"), "target")
    if target["collider_mode"] not in ("matched_target", "replace_with_open_box_proxy"):
        raise ValueError("target.collider_mode must be matched_target or replace_with_open_box_proxy")
    render = raw.get("render", {})
    _keys(render, ("camera", "width", "height"), "render")
    if "camera" in render and render["camera"] is not None and not isinstance(render["camera"], str):
        raise ValueError("render.camera must be a camera name or null")
    for name in ("width", "height"):
        if name in render and (
            not isinstance(render[name], int) or isinstance(render[name], bool) or render[name] <= 0
        ):
            raise ValueError(f"render.{name} must be a positive integer")
    if not isinstance(raw["scene_xml"], str):
        raise ValueError("scene_xml must be a path string")
    scene = (path.parent / raw["scene_xml"]).resolve()
    if not scene.is_file():
        raise ValueError(f"scene_xml does not exist: {scene}")
    return {
        "scene_xml": scene,
        "target": dict(target),
        "box": _parse_box(raw["box"]),
        "planner": _parse_options(raw.get("planner", {})),
        "robot_raw": raw["robot_state"],
        "render": dict(render),
    }


def _find_body(root, name):
    return next((body for body in root.iter("body") if body.get("name") == name), None)


def _box_geom_specs(box: OpenBoxGeometry):
    sx, sy, sz = box.size_m
    t = box.wall_thickness_m
    return (
        ("floor", (0, 0, t / 2), (sx / 2, sy / 2, t / 2)),
        ("xm", (-sx / 2 + t / 2, 0, sz / 2), (t / 2, sy / 2, sz / 2)),
        ("xp", (sx / 2 - t / 2, 0, sz / 2), (t / 2, sy / 2, sz / 2)),
        ("ym", (0, -sy / 2 + t / 2, sz / 2), (sx / 2, t / 2, sz / 2)),
        ("yp", (0, sy / 2 - t / 2, sz / 2), (sx / 2, t / 2, sz / 2)),
    )


@contextmanager
def prepared_scene(scene: Path, target: dict, box: OpenBoxGeometry):
    if target["collider_mode"] == "matched_target":
        yield scene
        return
    tree = ET.parse(scene)
    body = _find_body(tree.getroot(), target["body"])
    if body is None:
        raise ValueError(f"target body {target['body']!r} is not in the source XML")
    for geom in body.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        rgba = [float(value) for value in geom.get("rgba", "1 1 1 1").split()]
        geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} 0")
    for suffix, position, size in _box_geom_specs(box):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"box_rim_proxy_{suffix}",
                "type": "box",
                "pos": " ".join(map(str, position)),
                "size": " ".join(map(str, size)),
                "rgba": "0.80 0.52 0.16 0.82",
                "friction": "1.2 0.02 0.001",
            },
        )
    handle = tempfile.NamedTemporaryFile(dir=scene.parent, prefix="box_rim_proxy_", suffix=".xml", delete=False)
    proxy = Path(handle.name)
    handle.close()
    tree.write(proxy, encoding="unicode")
    try:
        yield proxy
    finally:
        proxy.unlink(missing_ok=True)


def _render_png(planner, state, plan, box, settings, output):
    import mujoco
    from PIL import Image

    planner.collision._place(
        state,
        plan.joints,
        plan.pelvis_position,
        plan.pelvis_quaternion_wxyz,
        box,
        planner.collision.gripper_q,
    )
    width, height = int(settings.get("width", 960)), int(settings.get("height", 720))
    if width <= 0 or height <= 0:
        raise ValueError("render width and height must be positive")
    model = planner.collision.model
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body in planner.collision.robot_bodies:
            model.geom_rgba[geom] = [0.05, 0.82, 1.0, 0.42]
    camera_name = settings.get("camera")
    if camera_name is None:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = 0.5 * (box.pose.position + plan.pelvis_position)
        camera.lookat[2] = box.pose.position[2] + box.size_m[2] * 0.65
        camera.distance, camera.azimuth, camera.elevation = 1.65, 145.0, -18.0
    else:
        camera = camera_name
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(planner.collision.data, camera=camera)
        for target in plan.targets_world_m:
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                break
            marker = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                marker,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.008, 0.0, 0.0]),
                target,
                np.eye(3).ravel(),
                np.array([1.0, 0.25, 0.04, 1.0], dtype=np.float32),
            )
            renderer.scene.ngeom += 1
        image = renderer.render().copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)


def run(request: dict):
    with prepared_scene(request["scene_xml"], request["target"], request["box"]) as scene:
        ik = ArmIK(str(scene))
        state = _parse_robot(request["robot_raw"], ik)
        planner = BoxRimPlanner(
            ik,
            request["target"]["body"],
            request["box"],
            request["planner"],
        )
        plan = planner.plan(state)
        return planner, state, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-png", type=Path)
    args = parser.parse_args(argv)
    try:
        request = load_request(args.request.resolve())
        started = time.perf_counter()
        planner, state, plan = run(request)
        planning_seconds = time.perf_counter() - started
        result = plan.to_dict()
        result["request"] = str(args.request.resolve())
        result["scene_xml"] = str(request["scene_xml"])
        result["target_body"] = request["target"]["body"]
        result["planning_wall_seconds"] = planning_seconds
        result["search_counts"] = {
            "generated": plan.diagnostics["geometry_candidates"],
            "coarse_accepted": sum(
                report.stage in ("coarse", "refined")
                and "coarse_position_error" not in report.reasons
                and "coarse_orientation_error" not in report.reasons
                for report in plan.candidate_reports
            ),
            "refined": sum(report.stage == "refined" for report in plan.candidate_reports),
            "static_valid": sum(report.stage == "refined" and report.static_valid for report in plan.candidate_reports),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        render_failed = False
        if args.render_png is not None and plan.static_valid:
            try:
                _render_png(planner, state, plan, request["box"], request["render"], args.render_png)
                result["render_png"] = str(args.render_png.resolve())
            except Exception as error:  # The validated plan remains useful if the optional renderer is unavailable.
                result["render_error"] = f"{type(error).__name__}: {error}"
                render_failed = True
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"static_valid": plan.static_valid, "reason": plan.reason, "output": str(args.output)}))
        return 4 if render_failed else (0 if plan.static_valid else 2)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a warehouse-room scene with the carton resized and every platform edge measured, for the four-edge line.

    python -m locomani.tasks.crate.warehouse_box_scene outputs/scene/warehouse_box_45x20x18 \\
        --width 0.45 --length 0.20 --height 0.18

From `warehouse_textured`, copied whole (its room meshes and textures are scene-relative, so the directory must
travel as one), it changes three things and nothing about the robot, the room or the platform:

- the carton: its nine photo meshes scaled about the floor centre to `width` (across the walk-in: the two walls
  the hands grip) x `length` (along it) x `height`, the photographs stretching with them; its inertia and five
  1 cm collision walls rebuilt by `box_platform_scene.resize_carton` (1.2 kg, friction 1.4, as the envelope sweep
  measured them); front face on the -y edge, where the room's robot start looks;
- `surfaces.A.edges`: for each side of the platform (`xm`, `xp`, `ym`, `yp`: the -x, +x, -y, +y side) the walk-in
  heading toward it and `clear_m`, how far out from the edge the floor is free. The room's racks, cart and walls
  have no collision -- a start placed inside one would stand in it, and only the picture would show it -- so the
  distance is measured here, with horizontal rays against the room meshes over the robot's height and the edge's
  length plus `BAND` each side;
- a third-person camera per edge (`edge_<side>`): the room's `reverse_room` shot of the -y approach, turned to
  each side (`edge_ym` is that camera). `video_camera` stays `reverse_room`.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from locomani.tasks.crate.box_platform_scene import BOX_BODY, BOX_KEY, WALL_THICKNESS, _vec, resize_carton
from locomani.tasks.crate.visual_approach import surface_edge
from locomani.tasks.crate.visual_config import EDGE_SIDES

SOURCE = Path(__file__).resolve().parents[4] / "outputs/scene/warehouse_textured"
SOURCE_BOX = (0.45, 0.32, 0.33)  # width, length, height of the room's carton meshes (their local bounds)
EDGES = {"ym": 90.0, "yp": -90.0, "xm": 0.0, "xp": 180.0}  # platform side -> world yaw of the walk-in toward it, deg
assert set(EDGES) == set(EDGE_SIDES)  # the sides visual.start_edges names
BAND = 0.3  # metres past each edge end the rays cover: a box at the edge end, the robot 8 cm right of it, shoulders
RAY_HEIGHTS = np.arange(0.05, 1.5, 0.1)  # the standing robot's height band, floor mats and low boards included
RAY_SPACING = 0.05
SKIN = 0.05  # rays start this far outside the edge: the pallet's own boards overhang its collision box by <= 2.7 cm
CLEAR_LIMIT = 3.0  # farther than any start is placed; reported when a ray hits nothing
# reverse_room in the -y edge's frame (along the walk-in, to its left, up; from the platform centre), and its view.
CAMERA_OFFSET = (1.435, 1.4, 1.71)
CAMERA_FORWARD = (-0.802, -0.552, -0.226)


def edge_frame(heading_deg: float):
    """Unit walk-in axis and its left, for a walk-in at world yaw `heading_deg`."""
    heading = math.radians(heading_deg)
    axis = np.array([math.cos(heading), math.sin(heading)])
    return axis, np.array([-axis[1], axis[0]])


def edge_line(surface: dict, heading_deg: float):
    """Centre of the surface edge the walk-in at `heading_deg` meets, and the edge's half length (the expert's own
    `surface_edge`, so the measured edge is the one the rounds walk to)."""
    edge, _axis, half = surface_edge(surface, math.radians(heading_deg))
    return edge, half


def clearance(model, data, surface: dict, heading_deg: float) -> float:
    """Free floor in front of one edge: the shortest horizontal ray outward from it to any geom, metres."""
    import mujoco

    axis, left = edge_frame(heading_deg)
    edge, half = edge_line(surface, heading_deg)
    direction = np.array([-axis[0], -axis[1], 0.0])
    hit = np.zeros(1, np.int32)
    shortest = CLEAR_LIMIT
    for across in np.arange(-(half + BAND), half + BAND + 1e-9, RAY_SPACING):
        start = edge + left * across - axis * SKIN
        for z in RAY_HEIGHTS:
            distance = mujoco.mj_ray(model, data, np.array([start[0], start[1], z]), direction, None, 1, -1, hit)
            if distance >= 0:
                shortest = min(shortest, SKIN + float(distance))
    return shortest


def edge_camera(surface: dict, heading_deg: float) -> dict:
    """MJCF attributes of the `reverse_room` shot turned to the edge the walk-in at `heading_deg` meets.

    The orientation is a `quat` (w x y z), not `xyaxes`: the Isaac renders read a scene camera through
    `isaac.frames.camera_spec`, which takes pos, quat and fovy only."""
    from scipy.spatial.transform import Rotation

    axis, left = edge_frame(heading_deg)
    centre = np.asarray(surface["xy"][:2], float)
    along, side, up = CAMERA_OFFSET
    pos = np.array([*(centre + along * axis + side * left), up])
    forward = np.array([*(CAMERA_FORWARD[0] * axis + CAMERA_FORWARD[1] * left), CAMERA_FORWARD[2]])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    x, y, z, w = Rotation.from_matrix(np.column_stack([right, np.cross(-forward, right), -forward])).as_quat()
    return {"pos": _vec(pos), "quat": _vec(np.sign(w or 1.0) * np.array([w, x, y, z])), "fovy": "73.74"}


def build(
    out_dir: Path,
    *,
    width: float,
    length: float,
    height: float,
    wall: float = WALL_THICKNESS,
    source: Path = SOURCE,
) -> dict:
    """Write the scene directory for one carton size; returns its meta."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"refusing to replace existing scene: {out_dir}")
    meta = json.loads((source / "scene_meta.json").read_text())
    if not math.isclose(meta["robot_start"]["yaw_deg"], EDGES["ym"], abs_tol=1e-6):
        raise SystemExit(f"{source}: expected the warehouse's +y walk-in, got yaw {meta['robot_start']['yaw_deg']}")
    surface = meta["surfaces"]["A"]
    shutil.copytree(source, out_dir)
    xml_path = out_dir / "scene_locomani.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    world = root.find("worldbody")

    body = world.find(f"body[@name='{BOX_BODY}']")
    visuals = {g.get("mesh") for g in body.findall("geom") if (g.get("name") or "").startswith("nor_cal_vis_")}
    scale = _vec(np.array([width, length, height]) / np.array(SOURCE_BOX))
    for mesh in root.find("asset").findall("mesh"):
        if mesh.get("name") in visuals:
            mesh.set("scale", scale)
    near, _half = edge_line(surface, EDGES["ym"])
    box_xy = near + edge_frame(EDGES["ym"])[0] * length / 2
    body.set("pos", _vec([box_xy[0], box_xy[1], surface["top"]]))
    resize_carton(body, width=width, length=length, height=height, wall=wall)
    for side, heading in EDGES.items():
        ET.SubElement(world, "camera", {"name": f"edge_{side}", **edge_camera(surface, heading)})
    tree.write(xml_path, encoding="unicode")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    root_joint = model.body("pelvis").jntadr[0]
    data.qpos[:] = model.qpos0
    data.qpos[model.jnt_qposadr[root_joint] : model.jnt_qposadr[root_joint] + 2] = 100.0  # the robot out of the rays
    mujoco.mj_forward(model, data)
    edges = {
        side: {
            "heading_deg": heading,
            "clear_m": round(clearance(model, data, surface, heading), 3),
            "video_camera": f"edge_{side}",
        }
        for side, heading in EDGES.items()
    }
    box_body = model.body(BOX_BODY).id
    qpos_address = int(model.jnt_qposadr[model.body_jntadr[box_body]])
    meta["scene_xml"] = str(xml_path.resolve())
    meta["surfaces"]["A"]["edges"] = edges
    meta["objects"][BOX_KEY].update(
        init_pos=model.qpos0[qpos_address : qpos_address + 3].tolist(),
        qpos_adr=qpos_address,
        height=height,
        box={"half_extent": [width / 2, length / 2, height / 2], "centre": [0.0, 0.0, height / 2], "wall": wall},
    )
    meta.update(
        nbody=int(model.nbody),
        ngeom=int(model.ngeom),
        nq=int(model.nq),
        source_scene=str(source),
        snapshot_change=(
            "four-edge warehouse: carton meshes scaled and its walls/inertia rebuilt, front face on the -y edge; "
            "per-edge walk-in heading, free floor and camera"
        ),
        layout=dict(width=width, length=length, height=height, wall=wall),
    )
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--width", type=float, required=True, help="across the walk-in, m (the gripped walls)")
    parser.add_argument("--length", type=float, required=True, help="along the walk-in, m")
    parser.add_argument("--height", type=float, required=True, help="box height, m")
    parser.add_argument("--wall", type=float, default=WALL_THICKNESS, help="wall thickness, m")
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args(argv)
    meta = build(
        args.out_dir,
        width=args.width,
        length=args.length,
        height=args.height,
        wall=args.wall,
        source=args.source,
    )
    print(json.dumps(dict(scene=str(args.out_dir), layout=meta["layout"], edges=meta["surfaces"]["A"]["edges"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

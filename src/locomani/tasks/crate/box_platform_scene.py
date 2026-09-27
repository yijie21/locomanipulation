"""Build one box-on-platform layout for the scene capability sweep: the warehouse robot and contacts, no room.

    python -m locomani.tasks.crate.box_platform_scene outputs/sweeps/<sweep>/scenes/<cell> \\
        --width 0.45 --length 0.32 --height 0.45

From `warehouse_textured` (the robot, floor and box contacts the warehouse line collects with) it drops every
room and carton visual -- none of them collides -- turns the pallet into a solid block `platform_top` high, and
rebuilds the carton's five 1 cm collision walls at `width` (across the walk-in: the two walls the hands grip) x
`length` (along the walk-in) x `height`, 1.2 kg and friction 1.4 as before, front face on the near edge. The near
edge and the robot start stay where the warehouse has them, so `start_edge_*` and the layout ranges mean the same
distances as there. The user's sweep assumptions (2026-09-26): a 10 cm platform, a light high-friction box, no
obstacles.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[4] / "outputs/scene/warehouse_textured"
BOX_BODY, BOX_KEY, PLATFORM_GEOM = "nor_cal_box_object", "nor_cal_box", "platform_col"
WALL_THICKNESS = (
    0.01  # the warehouse carton's walls and floor; --wall overrides it (does a 30 mm wall grip like the crate's bar?)
)
BOX_MASS = 1.2
BOX_RGBA = "0.86 0.78 0.62 1"
# Room enough past the near edge for a 0.44 m box set back 0.15 m and turned 20 deg, and across for a 0.55 m one.
PLATFORM_HALF_ALONG, PLATFORM_HALF_ACROSS = 0.5, 0.6
# A side view that keeps the box in frame, for trace videos (the warehouse's own camera looks into the room).
VIDEO_CAMERA = {"name": "sweep_side", "mode": "targetbody", "target": BOX_BODY, "pos": "2.2 -0.9 1.4", "fovy": "60"}


def _vec(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _drop_unreferenced_assets(root: ET.Element) -> None:
    """Remove meshes, materials and textures nothing refers to any more (the room's and the carton's)."""
    asset = root.find("asset")
    used = {
        (attr, element.get(attr))
        for element in root.iter()
        if element.tag not in ("mesh", "material", "texture")
        for attr in ("mesh", "material")
        if element.get(attr)
    }
    for element in list(asset):
        if element.tag in ("mesh", "material") and (element.tag, element.get("name")) not in used:
            asset.remove(element)
    textures = {m.get("texture") for m in asset.iter("material") if m.get("texture")}
    for element in list(asset):
        if element.tag == "texture" and element.get("type") != "skybox" and element.get("name") not in textures:
            asset.remove(element)


def build(
    out_dir: Path,
    *,
    width: float,
    length: float,
    height: float,
    platform_top: float = 0.10,
    wall: float = WALL_THICKNESS,
    source: Path = SOURCE,
) -> dict:
    """Write `out_dir/scene_locomani.xml` + `scene_meta.json` for one layout; returns the meta."""
    out_dir = Path(out_dir)
    if (out_dir / "scene_meta.json").exists():
        raise SystemExit(f"refusing to replace existing scene: {out_dir}")
    meta = json.loads((source / "scene_meta.json").read_text())
    heading = math.radians(meta["robot_start"]["yaw_deg"])
    axis = np.array([math.cos(heading), math.sin(heading)])
    if not np.allclose(axis, [0, 1], atol=1e-6):
        raise SystemExit(f"{source}: expected the warehouse's +y walk-in, got yaw {meta['robot_start']['yaw_deg']}")
    surface = meta["surfaces"]["A"]
    near = np.asarray(surface["xy"], float) - axis * surface["flat_half_extent"][1]

    tree = ET.parse(source / "scene_locomani.xml")
    root = tree.getroot()
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        if (geom.get("name") or "").startswith("vis_room_"):
            world.remove(geom)
    platform = world.find(f"body/geom[@name='{PLATFORM_GEOM}']")
    centre = near + axis * PLATFORM_HALF_ALONG
    platform.set("pos", _vec([centre[0], centre[1], platform_top / 2]))
    platform.set("size", _vec([PLATFORM_HALF_ACROSS, PLATFORM_HALF_ALONG, platform_top / 2]))
    platform.set("rgba", "0.45 0.47 0.5 1")

    # The carton's local x is across the walk-in and y along it (yaw 0 in the room, walk-in +y).
    hx, hy, hz = width / 2, length / 2, height / 2
    WALL = wall / 2  # half thickness
    body = world.find(f"body[@name='{BOX_BODY}']")
    box_xy = near + axis * hy
    body.set("pos", _vec([box_xy[0], box_xy[1], platform_top]))
    for child in list(body):
        if child.tag == "inertial" or (child.tag == "geom" and (child.get("name") or "").startswith("nor_cal_vis_")):
            body.remove(child)
    inertia = [BOX_MASS / 12 * (a * a + b * b) for a, b in ((length, height), (width, height), (width, length))]
    body.insert(1, ET.Element("inertial", pos=_vec([0, 0, hz]), mass=str(BOX_MASS), diaginertia=_vec(inertia)))
    walls = {
        "nor_cal_floor": ([0, 0, WALL], [hx, hy, WALL]),
        "nor_cal_xp": ([hx - WALL, 0, hz], [WALL, hy, hz]),
        "nor_cal_xm": ([-(hx - WALL), 0, hz], [WALL, hy, hz]),
        "nor_cal_yp": ([0, hy - WALL, hz], [hx, WALL, hz]),
        "nor_cal_ym": ([0, -(hy - WALL), hz], [hx, WALL, hz]),
    }
    for name, (pos, size) in walls.items():
        geom = body.find(f"geom[@name='{name}']")
        geom.set("pos", _vec(pos))
        geom.set("size", _vec(size))
        geom.set("rgba", BOX_RGBA)
    ET.SubElement(world, "camera", VIDEO_CAMERA)
    _drop_unreferenced_assets(root)

    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "scene_locomani.xml"
    tree.write(xml_path, encoding="unicode")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    box_body = model.body(BOX_BODY).id
    qpos_address = int(model.jnt_qposadr[model.body_jntadr[box_body]])
    meta["scene_xml"] = str(xml_path.resolve())
    meta["surfaces"]["A"].update(
        name="sweep platform",
        xy=centre.tolist(),
        top=platform_top,
        half_extent=[PLATFORM_HALF_ACROSS, PLATFORM_HALF_ALONG],
        flat_half_extent=[PLATFORM_HALF_ACROSS, PLATFORM_HALF_ALONG],
        circumradius=math.hypot(PLATFORM_HALF_ACROSS, PLATFORM_HALF_ALONG),
    )
    meta["objects"][BOX_KEY].update(
        init_pos=model.qpos0[qpos_address : qpos_address + 3].tolist(),
        init_quat=model.qpos0[qpos_address + 3 : qpos_address + 7].tolist(),
        qpos_adr=qpos_address,
        height=height,
        box={"half_extent": [hx, hy, hz], "centre": [0.0, 0.0, hz], "wall": wall},
    )
    meta.update(
        nbody=int(model.nbody),
        ngeom=int(model.ngeom),
        nq=int(model.nq),
        source_scene=str(source),
        snapshot_change="scene capability sweep layout: room and carton visuals removed, platform and carton walls resized",
        layout=dict(width=width, length=length, height=height, platform_top=platform_top, wall=wall),
        video_camera=VIDEO_CAMERA["name"],
    )
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--width", type=float, required=True, help="across the walk-in, m (the gripped walls)")
    parser.add_argument("--length", type=float, required=True, help="along the walk-in, m")
    parser.add_argument("--height", type=float, required=True, help="box height, m")
    parser.add_argument("--platform-top", type=float, default=0.10)
    parser.add_argument("--wall", type=float, default=WALL_THICKNESS, help="wall thickness, m")
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args(argv)
    meta = build(
        args.out_dir,
        width=args.width,
        length=args.length,
        height=args.height,
        platform_top=args.platform_top,
        wall=args.wall,
        source=args.source,
    )
    print(json.dumps(dict(scene=str(args.out_dir), layout=meta["layout"], ngeom=meta["ngeom"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

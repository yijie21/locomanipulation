"""A plain room for grasp experiments (2026-11-09): six furniture stations in a ring, facing the room's centre, the GraspNet
objects of the 65-object line on them. The heights are those of the reach map (select_eval.py): floor tray 0.08, low cabinet
0.32, table 0.62, counter 0.86, high shelf 1.02 m, and an open cabinet with boards at 0.30, 0.62 and 0.94 m. The centre is
clear, so a straight walk reaches every station. Each layout holds 21 objects; the layouts together hold every object.

Objects are read from outputs/graspnet_bottle_20261008/scenes/obj_XXX (visual mesh, CoACD collision pieces, the mass and the
contact parameters of that line), so the body frame is the one grasp_cache.npz is written in: a yaw of the body keeps every
cached grasp valid. Left out: objects with no validated grasp, and objects of more than MAX_PIECES collision pieces.

Other object sets: GRASP_ROOM_OBJECTS=outputs/reachwm_20261030/simready (simready_objects.py, realistic objects), GRASP_ROOM_OUT=
outputs/reachwm_20261030/grasp_room_sr, GRASP_ROOM_MIN_HELD=10, --layouts 7: 128 objects, every one with at least 10 validated grasps.

  build   compose the layouts; settle each without the robot, replace objects that tip, roll or drift, and write
          outputs/reachwm_20261030/grasp_room/layout_K/{scene_locomani.xml, scene_meta.json, layout.json}
  render  an overview and one close-up per station, layout_K/views.png
  robot   ScaleBFM in layout K: walk from the centre to each station's front, hold, report falls, contacts and step cost

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/grasp_room.py build --layouts 4
    MUJOCO_GL=egl $PY scripts/experiments/reachwm_20261030/grasp_room.py render --layouts 4
    BFM_ROOT=odo $PY scripts/experiments/reachwm_20261030/grasp_room.py robot --layout 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "rsi"))  # capability_probe
REPO = HERE.parents[2]
OUT = Path(
    os.environ.get("GRASP_ROOM_OUT", REPO / "outputs" / "reachwm_20261030" / "grasp_room")
)  # grasp_room_sr: the SimReady objects
GN = Path(
    os.environ.get("GRASP_ROOM_OBJECTS", "/workspace/code/locomanipulation/outputs/graspnet_bottle_20261008")
)  # read only; simready_objects.py writes the same layout
WALK = REPO / "outputs" / "scene" / "walk_mars"
SEED, FRONT_R, ROOM_HALF, MAX_PIECES, MIN_HELD = (
    int(os.environ.get("GRASP_ROOM_SEED", 5200000)),
    2.2,
    3.6,
    60,
    int(os.environ.get("GRASP_ROOM_MIN_HELD", 1)),
)
TABLE_TOP = 0.62  # the table of the object scenes: a body's z minus this is its height over the surface
SETTLE_S, TOL_M, TOL_DEG, FINAL_M, FINAL_DEG, ROUNDS = 2.0, 0.05, 10.0, 0.005, 1.0, 8
# name, angle of the station round the room (deg), width, depth, the surfaces' heights, depth range of the objects from the
# front face, objects per surface, the tallest object allowed
STATIONS = (
    dict(name="tray", ang=0.0, W=0.90, D=0.60, tops=(0.08,), depth=(0.12, 0.30), n=3, h_max=0.26),
    dict(name="low", ang=60.0, W=0.90, D=0.50, tops=(0.32,), depth=(0.12, 0.28), n=3, h_max=0.26),
    dict(name="table", ang=120.0, W=1.00, D=0.60, tops=(0.62,), depth=(0.12, 0.28), n=3, h_max=0.26),
    dict(name="counter", ang=180.0, W=1.00, D=0.55, tops=(0.86,), depth=(0.12, 0.28), n=3, h_max=0.26),
    dict(name="high", ang=240.0, W=0.90, D=0.35, tops=(1.02,), depth=(0.12, 0.22), n=3, h_max=0.16),
    dict(name="cabinet", ang=300.0, W=0.90, D=0.40, tops=(0.30, 0.62, 0.94), depth=(0.12, 0.20), n=2, h_max=0.16),
)
LATERAL = {3: (-0.28, 0.0, 0.28), 2: (-0.18, 0.18)}
# The hard room (2026-11-11, GRASP_ROOM_HARD=1 at build; a set's stations.json then travels with it): targets at the edge of
# what the body can do, where a fixed stance and grasp rule should stop working. Deeper (up to 0.50 m from the front, with
# 0.30 m of stance that is past ScaleBFM's forward reach of 0.6-0.7 m), the floor tray at 4 cm and the shelf at 1.10 m (the
# wrist tops out near 1.26), four objects a surface 22 cm apart, the cabinet 10 cm deeper, and a board over the low cabinet
# and over the counter (`over`: its underside's height; it starts 10 cm behind the front). The shallow end of each depth
# range is the plain room's, so success against depth can be read off one set.
STATIONS_HARD = (
    dict(name="tray", ang=0.0, W=0.90, D=0.70, tops=(0.04,), depth=(0.12, 0.45), n=4, h_max=0.26),
    dict(name="low", ang=60.0, W=0.90, D=0.60, tops=(0.32,), depth=(0.12, 0.42), n=4, h_max=0.22, over=0.70),
    dict(name="table", ang=120.0, W=1.00, D=0.70, tops=(0.62,), depth=(0.12, 0.50), n=4, h_max=0.26),
    dict(name="counter", ang=180.0, W=1.00, D=0.65, tops=(0.86,), depth=(0.12, 0.48), n=4, h_max=0.22, over=1.22),
    dict(name="high", ang=240.0, W=0.90, D=0.40, tops=(1.10,), depth=(0.10, 0.28), n=4, h_max=0.12),
    dict(name="cabinet", ang=300.0, W=0.90, D=0.50, tops=(0.30, 0.62, 0.94), depth=(0.12, 0.34), n=3, h_max=0.16),
)
LATERAL[4] = (-0.33, -0.11, 0.11, 0.33)
if (OUT / "stations.json").exists():
    STATIONS = tuple(json.load(open(OUT / "stations.json")))
elif os.environ.get("GRASP_ROOM_HARD"):
    STATIONS = STATIONS_HARD
MATERIALS = """
    <material name="room_wood" rgba="0.62 0.47 0.32 1" specular="0.2" shininess="0.3"/>
    <material name="room_white" rgba="0.86 0.85 0.81 1" specular="0.1" shininess="0.2"/>
    <material name="room_grey" rgba="0.42 0.45 0.48 1" specular="0.3" shininess="0.4"/>
    <material name="room_wall" rgba="0.80 0.82 0.84 1" specular="0"/>
    <material name="room_floor" rgba="0.58 0.55 0.50 1" specular="0.1" shininess="0.1"/>
"""
FRICTION = 'friction="0.6 0.005 0.0001"'


def box(name: str, c, s, mat: str) -> str:
    return f'<geom name="{name}" type="box" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" material="{mat}" {FRICTION}/>'


def furniture(st: dict) -> list[str]:
    """The station's geoms, local frame: x into the furniture from the front face, y along it, z up from the floor."""
    n, W, D, tops = st["name"], st["W"], st["D"], st["tops"]
    w, d = W / 2, D / 2
    if st.get("over"):  # a board over the surface from 10 cm behind the front, held by the back wall
        z, t = st["over"], 0.03
        rest = furniture({k: v for k, v in st.items() if k != "over"})
        return rest + [box(f"{n}_over", (0.05 + d, 0, z + t / 2), (d - 0.05, w, t / 2), "room_wood"),
                       box(f"{n}_over_back", (D + 0.01, 0, (tops[0] + z) / 2), (0.01, w, (z - tops[0]) / 2), "room_grey")]  # fmt: skip
    if n in ("tray", "low", "counter"):
        mat = {"tray": "room_grey", "low": "room_white", "counter": "room_white"}[n]
        return [box(f"{n}_body", (d, 0, tops[0] / 2), (d, w, tops[0] / 2), mat)]
    if n == "table":
        t = 0.04
        legs = [box(f"table_leg{i}", (x, y, (tops[0] - t) / 2), (0.025, 0.025, (tops[0] - t) / 2), "room_wood")
                for i, (x, y) in enumerate(((0.05, w - 0.05), (0.05, 0.05 - w), (D - 0.05, w - 0.05), (D - 0.05, 0.05 - w)))]  # fmt: skip
        return [box("table_top", (d, 0, tops[0] - t / 2), (d, w, t / 2), "room_wood"), *legs]
    if n == "high":
        t, z = 0.04, tops[0]
        return [box("high_top", (d, 0, z - t / 2), (d, w, t / 2), "room_wood"),
                box("high_left", (d, w + 0.02, z / 2), (d, 0.02, z / 2), "room_grey"),
                box("high_right", (d, -w - 0.02, z / 2), (d, 0.02, z / 2), "room_grey"),
                box("high_back", (D - 0.02, 0, z / 2), (0.02, w, z / 2 - t / 2), "room_grey")]  # fmt: skip
    t, top = 0.03, 1.22
    out = [box("cabinet_plinth", (d, 0, (tops[0] - t) / 2), (d, w, (tops[0] - t) / 2), "room_wood")]
    out += [box(f"cabinet_board{i}", (d, 0, z - t / 2), (d, w, t / 2), "room_wood") for i, z in enumerate(tops)]
    out += [box("cabinet_cap", (d, 0, top + t / 2), (d, w + 0.02, t / 2), "room_wood"),
            box("cabinet_left", (d, w + 0.01, top / 2), (d, 0.01, top / 2), "room_wood"),
            box("cabinet_right", (d, -w - 0.01, top / 2), (d, 0.01, top / 2), "room_wood"),
            box("cabinet_back", (D + 0.01, 0, top / 2), (0.01, w + 0.02, top / 2), "room_wood")]  # fmt: skip
    return out


def station_pose(st: dict) -> tuple[np.ndarray, float]:
    """World position of the front face's centre and the yaw of the station's x axis (away from the room's centre)."""
    a = math.radians(st["ang"])
    return np.array([FRONT_R * math.cos(a), FRONT_R * math.sin(a)]), a


def furniture_xml() -> str:
    out = []
    for st in STATIONS:
        c, a = station_pose(st)
        out.append(
            f'    <body name="st_{st["name"]}" pos="{c[0]:.4f} {c[1]:.4f} 0" quat="{math.cos(a / 2):.6f} 0 0 {math.sin(a / 2):.6f}">'
        )
        out += ["      " + g for g in furniture(st)]
        out.append("    </body>")
    h = ROOM_HALF
    vis = 'contype="0" conaffinity="0" material="room_wall"'
    for i, (x, y, sx, sy) in enumerate(((h, 0, 0.02, h), (-h, 0, 0.02, h), (0, h, h, 0.02), (0, -h, h, 0.02))):
        out.append(f'    <geom name="room_wall{i}" type="box" pos="{x} {y} 1.2" size="{sx} {sy} 1.2" {vis}/>')
    out.append(
        '    <light name="room_light" pos="0 0 3.2" dir="0 0 -1" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1" castshadow="false"/>'
    )
    return "\n".join(out)


class Obj:
    """One object of the GraspNet line: its body element and meshes as written in its scene."""

    def __init__(self, row: dict):
        self.id, self.height, self.width, self.mass = (
            int(row["id"]),
            float(row["height"]),
            float(row["width"]),
            float(row["mass"]),
        )
        self.name = row.get("name", f"gn{self.id:03d}")
        root = ET.parse(GN / "scenes" / f"obj_{self.id:03d}" / "scene_locomani.xml").getroot()
        self.body = root.find(".//body[@name='water_bottle_object']")
        self.meshes = [
            ET.tostring(m, encoding="unicode").strip()
            for m in root.findall("./asset/*")
            if m.get("name", "").startswith(self.name + "_")
        ]  # meshes, textures and materials
        self.half = float(self.body.get("pos").split()[2]) - TABLE_TOP

    def xml(self, pos, quat) -> str:
        b = ET.fromstring(ET.tostring(self.body))
        b.set("name", self.name)
        b.set("pos", " ".join(f"{v:.6f}" for v in pos))
        b.set("quat", " ".join(f"{v:.6f}" for v in quat))
        b.find("joint").set("name", self.name + "_joint")
        return ET.tostring(b, encoding="unicode").strip()


def load_objects() -> list[Obj]:
    rows, out = json.load(open(GN / "objects.json"))["objects"], []
    if (GN / "cache" / "summary.json").exists():
        held = {r["id"]: r["cache"]["held"] for r in json.load(open(GN / "cache" / "summary.json"))}
    else:  # simready_objects.py: only each scene's own summary
        held = {
            int(j["object"]): int(j["held"])
            for j in (json.load(open(f)) for f in (GN / "scenes").glob("obj_*/grasp_cache_summary.json"))
        }
    for r in rows:
        if held.get(r["id"], 0) >= MIN_HELD and r["pieces"] <= MAX_PIECES:
            out.append(Obj(r))
    return out


def quat_yaw(y: float) -> np.ndarray:
    return np.array([math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)])


def slots() -> list[dict]:
    return [
        dict(station=st["name"], si=si, surf=z, k=k)
        for st in STATIONS
        for si, z in enumerate(st["tops"])
        for k in range(st["n"])
    ]


def by_name(name: str) -> dict:
    return next(s for s in STATIONS if s["name"] == name)


def place(rng, slot: dict, o: Obj) -> dict:
    """World pose of an object in a slot: the front face's centre, then depth and lateral in the station's frame."""
    st = by_name(slot["station"])
    c, a = station_pose(st)
    depth = float(rng.uniform(*st["depth"]))
    lat = (
        (-0.26, 0.0, 0.26)[slot["k"]]
        if st["name"] == "cabinet" and st["n"] == 3
        else LATERAL[st["n"]][slot["k"]] + float(rng.uniform(-0.03, 0.03))
    )
    xy = c + np.array([math.cos(a) * depth - math.sin(a) * lat, math.sin(a) * depth + math.cos(a) * lat])
    return dict(
        pos=[*xy.tolist(), slot["surf"] + o.half + 0.002],
        quat=quat_yaw(float(rng.uniform(-math.pi, math.pi))).tolist(),
        depth=depth,
        lat=lat,
    )


def room_xml(objs: list[Obj], poses: list[dict]) -> str:
    """The room with no robot, for settling."""
    meshes = "\n".join(m for o in objs for m in o.meshes)
    bodies = "\n".join(o.xml(p["pos"], p["quat"]) for o, p in zip(objs, poses, strict=True))
    return f"""<mujoco model="room"><compiler angle="radian"/><option timestep="0.002"/>
<asset>{MATERIALS}{meshes}</asset><worldbody><geom name="floor" type="plane" size="0 0 0.05"/>
{furniture_xml()}
{bodies}
</worldbody></mujoco>"""


def settle_once(objs: list[Obj], poses: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Free physics for SETTLE_S: the poses at rest, how far each moved (m) and turned (deg)."""
    import mujoco
    from scipy.spatial.transform import Rotation

    m = mujoco.MjModel.from_xml_string(room_xml(objs, poses))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(int(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
    out, moved, deg = [], [], []
    for o, p in zip(objs, poses, strict=True):
        adr = m.jnt_qposadr[m.joint(o.name + "_joint").id]
        pos, q = d.qpos[adr : adr + 3].copy(), d.qpos[adr + 3 : adr + 7].copy()
        q0 = np.array(p["quat"])
        out.append(dict(p, pos=pos.tolist(), quat=(q / np.linalg.norm(q)).tolist()))
        moved.append(float(np.linalg.norm(pos - np.array(p["pos"]))))
        r = Rotation.from_quat(q[[1, 2, 3, 0]]) * Rotation.from_quat(q0[[1, 2, 3, 0]]).inv()
        deg.append(float(np.degrees(r.magnitude())))
    return out, np.array(moved), np.array(deg)


def pose_gap(a: list[dict], b: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per object, the distance (m) and the turn (deg) between two lists of poses."""
    from scipy.spatial.transform import Rotation

    off, turn = [], []
    for p, q in zip(a, b, strict=True):
        off.append(float(np.linalg.norm(np.array(p["pos"]) - np.array(q["pos"]))))
        r = (
            Rotation.from_quat(np.array(q["quat"])[[1, 2, 3, 0]])
            * Rotation.from_quat(np.array(p["quat"])[[1, 2, 3, 0]]).inv()
        )
        turn.append(float(np.degrees(r.magnitude())))
    return np.array(off), np.array(turn)


def compose(k: int, objs: list[Obj], order: list[int]) -> dict:
    """One layout: fill the slots, settle, swap what will not rest, until every object stays put."""
    rng = np.random.default_rng(SEED + k)
    S = slots()
    chosen, pool, rejected = [], list(order), set()

    def draw(slot, used) -> int:
        h_max = by_name(slot["station"])["h_max"]
        for j, i in enumerate(pool):
            if i not in used and i not in rejected and objs[i].height <= h_max:
                return pool.pop(j)
        raise RuntimeError(f"no object left for {slot}")

    for s in S:
        chosen.append(draw(s, set(chosen)))
    poses = [place(rng, s, objs[i]) for s, i in zip(S, chosen, strict=True)]
    for r in range(ROUNDS):
        # settle again and again from the placement: an object that creeps and then topples must count, so each object is
        # judged by how far it ended from where it was placed, and by whether it has stopped
        placed, rest = [dict(p) for p in poses], poses
        for _ in range(ROUNDS):
            rest, moved, deg = settle_once([objs[i] for i in chosen], rest)
            if moved.max() < FINAL_M and deg.max() < FINAL_DEG:
                break
        off, turn = pose_gap(placed, rest)
        bad = [
            j
            for j in range(len(S))
            if off[j] > TOL_M or turn[j] > TOL_DEG or moved[j] >= FINAL_M or deg[j] >= FINAL_DEG
        ]
        print(
            f"layout {k} round {r}: worst from the placement {100 * off.max():.1f} cm, {turn.max():.1f} deg, replaced {len(bad)}",
            flush=True,
        )
        if not bad:
            poses = rest
            break
        for j in bad:
            rejected.add(chosen[j])
            chosen[j] = draw(S[j], set(chosen))
            poses[j] = place(rng, S[j], objs[chosen[j]])
    else:
        raise RuntimeError(f"layout {k} did not settle")
    final = poses
    for sl, p, i in zip(S, final, chosen, strict=True):
        assert p["pos"][2] > sl["surf"], f"layout {k}: {objs[i].name} is below its surface"
    print(
        f"layout {k}: at rest, last settle moved {1000 * moved.max():.1f} mm, {deg.max():.2f} deg; rejected {len(rejected)}",
        flush=True,
    )
    return dict(slots=[dict(s, object=objs[i].name, id=objs[i].id, height=objs[i].height, width=objs[i].width, half=objs[i].half, **p)
                       for s, i, p in zip(S, chosen, final, strict=True)], rejected=[objs[i].name for i in sorted(rejected)])  # fmt: skip


def write_scene(k: int, objs: list[Obj], layout: dict) -> Path:
    d = OUT / f"layout_{k}"
    d.mkdir(parents=True, exist_ok=True)
    by = {o.name: o for o in objs}
    used = [by[s["object"]] for s in layout["slots"]]
    xml = (WALK / "scene_locomani.xml").read_text()
    assert xml.count("</worldbody>") == 1 and xml.count("<worldbody>") == 1
    meshes = "\n".join(m for o in used for m in o.meshes)
    bodies = "\n".join("    " + by[s["object"]].xml(s["pos"], s["quat"]) for s in layout["slots"])
    xml = xml.replace("<worldbody>", f"<asset>{MATERIALS}{meshes}</asset>\n  <worldbody>", 1)
    xml = xml.replace("</worldbody>", furniture_xml() + "\n" + bodies + "\n  </worldbody>")
    assert xml.count('material="groundplane"') == 1
    xml = xml.replace('material="groundplane"', 'material="room_floor"')  # a plain floor; contact parameters unchanged
    (d / "scene_locomani.xml").write_text(xml)
    (d / "scene_meta.json").write_text((WALK / "scene_meta.json").read_text())
    (d / "layout.json").write_text(json.dumps(layout, indent=1))
    return d


def build(n: int) -> None:
    if STATIONS is STATIONS_HARD:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(STATIONS, open(OUT / "stations.json", "w"), indent=1)
    objs = load_objects()
    print(f"{len(objs)} objects usable (held > 0, at most {MAX_PIECES} pieces)", flush=True)
    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(len(objs)))
    seen: set[str] = set()
    for k in range(n):
        # the objects no layout has held yet come first, so the layouts together cover the set
        fresh = [i for i in order if objs[i].name not in seen]
        rest = [i for i in order if objs[i].name in seen]
        layout = compose(k, objs, fresh + list(rng.permutation(rest).astype(int)))
        seen |= {s["object"] for s in layout["slots"]}
        print(write_scene(k, objs, layout), f"covers {len(seen)} of {len(objs)}", flush=True)


def render(n: int) -> None:
    import mujoco
    from PIL import Image

    for k in range(n):
        d = OUT / f"layout_{k}"
        m = mujoco.MjModel.from_xml_path(str(d / "scene_locomani.xml"))
        da = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, da, 0)
        mujoco.mj_forward(m, da)
        print(f"layout {k}: nv {m.nv} ngeom {m.ngeom} mesh verts {m.nmeshvert} (2^20 = {1 << 20})", flush=True)
        W, H = 640, 480
        m.vis.global_.offwidth, m.vis.global_.offheight = (
            max(W, m.vis.global_.offwidth),
            max(H, m.vis.global_.offheight),
        )
        rn = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        tiles = []
        for lookat, dist, az, el in [((0, 0, 0.5), 7.4, 45, -38)] + [
            (np.r_[station_pose(s)[0] + 0.2 * np.array([math.cos(math.radians(s["ang"])), math.sin(math.radians(s["ang"]))]), 0.15 + 0.75 * s["tops"][-1]], 2.1, s["ang"] + 18, -22) for s in STATIONS
        ] + [((0, 0, 0.5), 7.6, 225, -60)]:  # fmt: skip
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
            rn.update_scene(da, cam)
            tiles.append(rn.render().copy())
        rows = [np.concatenate(tiles[i : i + 4], 1) for i in range(0, 8, 4)]
        Image.fromarray(np.concatenate(rows, 0)).save(d / "views.png")
        print(d / "views.png", flush=True)
        rn.close()


def make_env(k: int):
    """selfwm_20261029/env.Env on layout K (grasp_pilot.make_env's set-up), the root source as BFM_ROOT says."""
    import dataclasses

    import capability_probe as cap
    import grasp_pilot as GP
    import leg_odo

    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
    from locomani.backends.scalebfm.motor import Motor, MotorConfig
    from locomani.tasks.scene.world import GRIP_LAW

    E = GP.E
    policy = R.Policy("cuda", ScaleBfmBackendConfig())
    sim = R.Sim(
        policy.meta, backend=ScaleBfmBackendConfig(scene_dir=str(OUT / f"layout_{k}"), object_name="", **GRIP_LAW)
    )
    m = Motor(sim, policy, MotorConfig(), model_xml=str(cap.WALK / "scene_locomani.xml"))
    probe = cap.Probe.__new__(cap.Probe)
    probe.R, probe.kind, probe.m, probe.sim = R, "grip", m, sim
    env = E.Env.__new__(E.Env)
    env.cap, env.probe, env.m, env.sim, env.R = cap, probe, m, sim, R
    import mujoco

    env.cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if os.environ.get("BFM_ROOT", "true") == "odo":
        m.cfg = dataclasses.replace(m.cfg, stop_ki=0.0, stop_timeout=m.cfg.settle_s)
        env.odo_root = leg_odo.OdoRoot(env)
    return env


def robot(k: int) -> None:
    """From the room's centre to 0.45 m in front of each station and back, the objects and the furniture left alone."""
    import mujoco

    env = make_env(k)
    m, sim = env.m, env.sim
    layout = json.load(open(OUT / f"layout_{k}" / "layout.json"))
    furn = {i for i in range(sim.m.ngeom) if sim.m.body(sim.m.geom_bodyid[i]).name.startswith("st_")}
    names = {s["object"] for s in layout["slots"]}
    objs = {i for i in range(sim.m.ngeom) if sim.m.body(sim.m.geom_bodyid[i]).name in names}
    ids = {s["object"]: sim.m.jnt_qposadr[sim.m.joint(s["object"] + "_joint").id] for s in layout["slots"]}
    p0 = {n: sim.d.qpos[a : a + 3].copy() for n, a in ids.items()}
    # the cost of a simulation step with and without the objects asleep is the same model; time it standing still
    m.reset((0.0, 0.0), 0.0)
    m.hold(1.0)
    tic, t0 = time.time(), m.t
    m.hold(2.0)
    print(f"standing: {(m.t - t0) / (time.time() - tic):.2f} x real time, nv {sim.m.nv}", flush=True)
    for st in STATIONS:
        c, a = station_pose(st)
        goal = c - 0.75 * np.array([math.cos(a), math.sin(a)])
        m.reset((0.0, 0.0), a)
        m.hold(1.0)
        d0 = float(np.linalg.norm(goal))
        tic, t0 = time.time(), m.t
        m.walk_to(goal, a, face="fixed")
        m.hold(1.0)
        pairs = [(sim.d.contact[j].geom1, sim.d.contact[j].geom2) for j in range(sim.d.ncon)]
        ncon = sum(1 for g1, g2 in pairs if (g1 in furn) != (g2 in furn) and g1 not in objs and g2 not in objs)
        p = sim.d.qpos[:2]
        print(f"{st['name']:8s} walked {d0:.2f} m, stops {100 * float(np.linalg.norm(p - goal)):.1f} cm off, fell {m.fell_at is not None}, "
              f"furniture contacts {ncon}, {(m.t - t0) / (time.time() - tic):.2f} x real time", flush=True)  # fmt: skip
        mujoco.mj_forward(sim.m, sim.d)
    drift = max(float(np.linalg.norm(sim.d.qpos[a : a + 3] - p0[n])) for n, a in ids.items())
    print(f"largest object drift over the run: {100 * drift:.2f} cm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "render", "robot"])
    ap.add_argument("--layouts", type=int, default=4)
    ap.add_argument("--layout", type=int, default=0)
    a = ap.parse_args()
    if a.phase == "build":
        build(a.layouts)
    elif a.phase == "render":
        render(a.layouts)
    else:
        robot(a.layout)

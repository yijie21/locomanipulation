"""shelf_scene.py: Figure 1 v3's illustrative room (built here, not part of any benchmark). notebook: skip figure render,
not an experiment.

SIMPLE's robot (episodes.robot_spec: G1 + Dex3, SIMPLE's solver options and hand contacts, its ground plane) in a bright
room built from MuJoCo primitives only (boxes, cylinders, capsules, ellipsoids, spheres; builtin procedural textures):
light tiled floor, a soft rug, plain walls behind and left of the robot, an open four-tier etagere (oak boards, black
posts) with books and small things on the lower tiers, and on its top board (above the robot's shoulder) the target: a
glazed ceramic vase with a gold band and three small flowers. No benchmark asset, no brand.

Frame: the robot starts at the origin facing +x (yaw 0). The etagere's front edge is at x = FRONT, its top board's top at
z = TOP. The vase is one free body (origin at its foot's centre); everything else is static.

PLACEMENTS (fixed before any run, README): 20 placements, numpy default_rng(SEED), per placement in this order
x ~ U(X_RANGE), y ~ U(Y_RANGE), yaw ~ U(0, 2 pi): the vase's foot centre on the top board (x 6.5-12.5 cm behind the front
edge, y 11-15 cm right of the robot's start, so near the board's right end: the hand rises right of the board and
enters over it sideways), the vase turned by yaw (the flowers' side). The robot is not moved.

GRASP (ground truth, from the vase's pose): the right hand's palm point (simple_bench's Dex3 palm, REACH_HAND_OFF
0.0415) at the vase's axis, GRASP_H above its foot, ALONG behind it along the robot's heading and NORMAL_OFF to its right;
the wrist attitude: fingers along the heading, palm facing left (the palm link's y = its normal), z up -- the forearm
forward, the hand upright. In the vase's frame (T_og) so it follows the vase as simple_bench's truth column reads it.
Fingers: F_PRE (open, carried through the reaches) and F_CLOSE (the power grasp), right Dex3 joint order (thumb 0-2,
index 0-1, middle 0-1); set for this vase's 3.2 cm radius (the magnitudes of SIMPLE's stored side grasps).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

SEED = 20261127
N_PLACE = 20
FRONT, TOP = 0.32, 1.16  # m: the etagere's front edge (x), its top board's top (z)
DEPTH, WIDTH, Y_LEFT = 0.30, 0.42, 0.23  # the etagere spans x FRONT..FRONT+DEPTH, y Y_LEFT-WIDTH..Y_LEFT
BOARD_T, POST = 0.022, 0.025
TIERS = (0.12, 0.47, 0.82, TOP)  # boards' top surfaces
X_RANGE, Y_RANGE = (FRONT + 0.065, FRONT + 0.125), (-0.15, -0.11)
R_VASE = 0.032  # the vase body's radius (where the hand closes)
GRASP_H, ALONG, NORMAL_OFF = 0.10, 0.085, R_VASE + 0.045
F_PRE = np.array([0.40, -0.15, -0.30, 0.30, 0.45, 0.30, 0.45])
F_CLOSE = np.array([0.50, -0.25, -0.60, 1.10, 1.20, 1.10, 1.20])
VASE_MASS = 0.15
TASK = "ShelfReach"


def placements() -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for i in range(N_PLACE):
        x, y, yaw = float(rng.uniform(*X_RANGE)), float(rng.uniform(*Y_RANGE)), float(rng.uniform(0.0, 2 * math.pi))
        out.append(dict(index=i, x=x, y=y, yaw=yaw))
    return out


def quat_yaw(yaw: float) -> list:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def grasp_world(p: dict) -> np.ndarray:
    """4x4 palm pose (world) for placement p: fingers +x, palm normal +y, z up."""
    T = np.eye(4)
    T[:3, 3] = [p["x"] - ALONG, p["y"] - NORMAL_OFF, TOP + GRASP_H]
    return T


def vase_T(p: dict) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [p["x"], p["y"], TOP]
    return T


def grasp_in_vase(p: dict) -> np.ndarray:
    return np.linalg.inv(vase_T(p)) @ grasp_world(p)


# -- building ---------------------------------------------------------------------------------------------------------
BOX, CYL, CAP, ELL, SPH = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_SPHERE)


def _mat(spec, name, rgb, spec_=0.2, shin=0.3, refl=0.0, emis=0.0, tex=None, rep=(1, 1), uniform=False):
    m = spec.add_material(name=name)
    m.rgba = [*rgb, 1.0]
    m.specular, m.shininess, m.reflectance, m.emission = spec_, shin, refl, emis
    if tex is not None:
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
        m.texrepeat = list(rep)
        m.texuniform = uniform
    return name


def _geom(body, name, typ, size, pos, mat, quat=(1, 0, 0, 0), collide=False, **kw):
    g = body.add_geom(name=name, type=typ, size=list(size) + [0] * (3 - len(size)), pos=list(pos), quat=list(quat),
                      material=mat, group=0)
    if collide:
        for k, v in kw.items():
            setattr(g, k, v)
    else:
        g.contype, g.conaffinity, g.density = 0, 0, 0.0
    return g


def _q_axis(axis: str, ang: float) -> list:
    a = np.r_[{"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]] * math.sin(ang / 2)
    return [math.cos(ang / 2), *a.tolist()]


def _qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out.tolist()


def _lights(spec):
    w = spec.worldbody
    spec.visual.headlight.ambient = [0.28, 0.28, 0.28]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    kinds = {}
    if hasattr(mujoco, "mjtLightType"):
        kinds = dict(directional=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, spot=mujoco.mjtLightType.mjLIGHT_SPOT)

    def light(name, pos, dirn, diffuse, spec_=0.1, shadow=False, directional=True, cutoff=60.0):
        lt = w.add_light(name=name, pos=list(pos), dir=list(np.asarray(dirn, float) / np.linalg.norm(dirn)))
        if kinds:
            lt.type = kinds["directional" if directional else "spot"]
        else:
            lt.directional = directional
        lt.diffuse, lt.specular, lt.ambient = [diffuse] * 3, [spec_] * 3, [0.0] * 3
        lt.castshadow = shadow
        if not directional:
            lt.cutoff, lt.exponent = cutoff, 2.0
        return lt

    light("sun_key", (2.5, -1.0, 3.5), (-0.55, 0.20, -1.0), 0.42, 0.15, shadow=True)  # soft key from the front-right, above
    light("fill_left", (0.5, 3.0, 2.5), (-0.2, -1.0, -0.8), 0.18)  # fill from the robot's left
    light("rim_back", (-2.0, 0.0, 2.8), (1.0, 0.0, -0.9), 0.12)  # rim from behind the robot


def build(p: dict, out: Path, robot_spec) -> dict:
    """Write the scene for placement p (scene_locomani.xml, scene_meta.json) -> simple_bench's info dict."""
    out.mkdir(parents=True, exist_ok=True)
    spec = robot_spec()
    w = spec.worldbody
    # floor: the robot spec's ground plane, given a light tile look
    spec.add_texture(name="tiles", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.885, 0.870, 0.840], rgb2=[0.860, 0.845, 0.815], width=512, height=512,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.80, 0.785, 0.75])
    _mat(spec, "floor", (1.0, 1.0, 1.0), 0.15, 0.4, 0.04, tex="tiles", rep=(0.85, 0.85), uniform=True)
    spec.geom("ground").material = "floor"
    spec.geom("ground").rgba = [1, 1, 1, 1]
    # materials
    _mat(spec, "wall", (0.935, 0.925, 0.900), 0.05, 0.1)
    _mat(spec, "wall_accent", (0.780, 0.830, 0.790), 0.05, 0.1)
    _mat(spec, "trim", (0.975, 0.975, 0.970), 0.2, 0.3)
    _mat(spec, "rug", (0.560, 0.640, 0.700), 0.02, 0.05)
    _mat(spec, "rug_edge", (0.930, 0.900, 0.840), 0.02, 0.05)
    _mat(spec, "oak", (0.760, 0.565, 0.360), 0.25, 0.35)
    _mat(spec, "metal_black", (0.110, 0.110, 0.120), 0.5, 0.6)
    _mat(spec, "glaze", (0.050, 0.420, 0.480), 0.9, 0.95, 0.06)
    _mat(spec, "glaze_in", (0.030, 0.250, 0.290), 0.6, 0.8)
    _mat(spec, "gold", (0.850, 0.660, 0.280), 1.0, 0.9, 0.05)
    _mat(spec, "stem", (0.250, 0.450, 0.200), 0.1, 0.2)
    _mat(spec, "leaf", (0.330, 0.560, 0.270), 0.15, 0.3)
    _mat(spec, "petal_w", (0.985, 0.960, 0.940), 0.2, 0.3)
    _mat(spec, "petal_p", (0.960, 0.640, 0.640), 0.2, 0.3)
    _mat(spec, "petal_c", (0.960, 0.500, 0.380), 0.2, 0.3)
    _mat(spec, "pistil", (0.980, 0.780, 0.200), 0.2, 0.3)
    books = [(0.780, 0.400, 0.280), (0.180, 0.270, 0.420), (0.560, 0.650, 0.500), (0.880, 0.720, 0.320),
             (0.920, 0.890, 0.820), (0.450, 0.300, 0.400), (0.300, 0.500, 0.560), (0.700, 0.700, 0.680)]
    for k, c in enumerate(books):
        _mat(spec, f"book{k}", c, 0.15, 0.3)
    _mat(spec, "terracotta", (0.780, 0.450, 0.320), 0.15, 0.3)
    _mat(spec, "ceramic_w", (0.950, 0.945, 0.930), 0.7, 0.8)
    _mat(spec, "plant", (0.300, 0.520, 0.300), 0.1, 0.25)
    _mat(spec, "frame", (0.200, 0.180, 0.160), 0.2, 0.3)
    _mat(spec, "canvas", (0.900, 0.850, 0.760), 0.05, 0.1)
    _mat(spec, "canvas2", (0.560, 0.700, 0.760), 0.05, 0.1)
    _mat(spec, "canvas3", (0.900, 0.650, 0.520), 0.05, 0.1)
    _lights(spec)

    room = w.add_body(name="room")
    _geom(room, "wall_back", BOX, (0.04, 3.2, 1.4), (-1.85, 0.0, 1.4), "wall")
    _geom(room, "trim_back", BOX, (0.012, 3.2, 0.045), (-1.80, 0.0, 0.045), "trim")
    _geom(room, "wall_left", BOX, (3.2, 0.04, 1.4), (1.2, 2.25, 1.4), "wall_accent")
    _geom(room, "trim_left", BOX, (3.2, 0.012, 0.045), (1.2, 2.20, 0.045), "trim")
    # a framed picture on the back wall, a big plant in the corner
    _geom(room, "pic_frame", BOX, (0.012, 0.42, 0.30), (-1.80, -0.55, 1.55), "frame")
    _geom(room, "pic_canvas", BOX, (0.004, 0.39, 0.27), (-1.785, -0.55, 1.55), "canvas")
    _geom(room, "pic_a", ELL, (0.003, 0.16, 0.10), (-1.780, -0.66, 1.50), "canvas2")
    _geom(room, "pic_b", SPH, (0.06,), (-1.782, -0.40, 1.64), "canvas3")
    _geom(room, "pot_big", CYL, (0.16, 0.20), (-1.45, 1.75, 0.20), "terracotta")
    for k, (dx, dy, dz, r) in enumerate([(0, 0, 0.62, 0.26), (0.10, 0.12, 0.85, 0.20), (-0.10, -0.08, 0.95, 0.18),
                                         (0.05, -0.12, 0.70, 0.19), (-0.06, 0.14, 1.08, 0.13)]):
        _geom(room, f"bush{k}", ELL, (r, r * 0.9, r * 1.15), (-1.45 + dx, 1.75 + dy, dz), "plant")
    # the rug under the robot (visual only, 2 mm)
    _geom(room, "rug", BOX, (1.05, 0.80, 0.001), (-0.15, 0.05, 0.001), "rug")
    _geom(room, "rug_border", BOX, (1.10, 0.85, 0.0008), (-0.15, 0.05, 0.0008), "rug_edge")

    # the etagere: oak boards, black posts (boards and posts collide, as SIMPLE's tables do)
    sh = w.add_body(name="etagere")
    x0, x1, y1 = FRONT, FRONT + DEPTH, Y_LEFT
    y0 = y1 - WIDTH
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    board = dict(condim=6, friction=[2, 0.04, 0.0005], priority=10)
    for k, zt in enumerate(TIERS):
        _geom(sh, f"board{k}", BOX, (0.5 * DEPTH, 0.5 * WIDTH, 0.5 * BOARD_T), (xc, yc, zt - 0.5 * BOARD_T), "oak",
              collide=True, **board)
    for k, (px, py) in enumerate([(x0 + POST / 2, y0 + POST / 2), (x0 + POST / 2, y1 - POST / 2),
                                  (x1 - POST / 2, y0 + POST / 2), (x1 - POST / 2, y1 - POST / 2)]):
        _geom(sh, f"post{k}", BOX, (POST / 2, POST / 2, 0.5 * TOP), (px, py, 0.5 * TOP), "metal_black", collide=True,
              condim=3, friction=[1.0, 0.005, 0.0001])
    # books and small things on the three lower tiers (visual only)
    rng = np.random.default_rng(7)
    for t, zt in enumerate(TIERS[:3]):
        yy = y0 + POST + 0.015
        room_h = (TIERS[t + 1] - BOARD_T - zt) if t < 2 else (TOP - BOARD_T - zt)
        n = 0
        while yy < y1 - POST - 0.06:
            if t == 1 and n == 5:  # a small plant in a white pot on the middle tier
                _geom(sh, "pot_s", CYL, (0.045, 0.045), (xc + 0.02, yy + 0.05, zt + 0.045), "ceramic_w")
                for k, (dx, dy, dz, r) in enumerate([(0, 0, 0.13, 0.055), (0.03, 0.03, 0.15, 0.04), (-0.03, -0.02, 0.16, 0.04)]):
                    _geom(sh, f"pl{k}", SPH, (r,), (xc + 0.02 + dx, yy + 0.05 + dy, zt + dz), "plant")
                yy += 0.12
                n += 1
                continue
            if t == 2 and n == 3:  # a low white bowl and a lying stack
                _geom(sh, "bowl", ELL, (0.07, 0.07, 0.035), (xc, yy + 0.07, zt + 0.02), "ceramic_w")
                yy += 0.16
                n += 1
                continue
            th = float(rng.uniform(0.018, 0.040))
            hh = float(min(rng.uniform(0.15, 0.24), room_h - 0.03))
            dd = float(rng.uniform(0.15, 0.21))
            lean = float(rng.uniform(-0.12, 0.12)) if rng.random() < 0.2 else 0.0
            _geom(sh, f"book{t}_{n}", BOX, (dd / 2, th / 2, hh / 2), (x1 - 0.03 - dd / 2, yy + th / 2, zt + hh / 2),
                  f"book{int(rng.integers(len(books)))}", quat=_q_axis("x", lean))
            yy += th + float(rng.uniform(0.001, 0.006))
            n += 1
    # top board: two lying books and a brass ball at its left end, a white candle at the back middle (away from the hand)
    z = TOP
    for k, (hh, c, dx) in enumerate([(0.028, 1, 0.0), (0.022, 0, 0.008)]):
        _geom(sh, f"topbook{k}", BOX, (0.065, 0.055, hh / 2), (x1 - 0.07 + dx, y1 - 0.065, z + hh / 2), f"book{c}")
        z += hh
    _geom(sh, "brass", SPH, (0.022,), (x1 - 0.07, y1 - 0.065, z + 0.022), "gold")
    _geom(sh, "candle", CYL, (0.028, 0.05), (x1 - 0.07, y1 - 0.19, TOP + 0.05), "ceramic_w")

    # the vase: a free body, its origin at the foot's centre on the top board
    v = w.add_body(name="vase", pos=[p["x"], p["y"], TOP + 1e-4], quat=quat_yaw(p["yaw"]))
    hit = dict(condim=4, friction=[0.8, 0.05, 0.005], solref=[0.005, 2])
    _geom(v, "vase_foot", CYL, (0.027, 0.005), (0, 0, 0.005), "glaze", collide=True, mass=0.2 * VASE_MASS, **hit)
    _geom(v, "vase_body", CYL, (R_VASE, 0.071), (0, 0, 0.010 + 0.071), "glaze", collide=True, mass=0.6 * VASE_MASS, **hit)
    _geom(v, "vase_band", CYL, (R_VASE + 0.0008, 0.004), (0, 0, 0.128), "gold")
    _geom(v, "vase_shoulder", ELL, (R_VASE, R_VASE, 0.030), (0, 0, 0.152), "glaze", collide=True, mass=0.15 * VASE_MASS, **hit)
    _geom(v, "vase_neck", CYL, (0.0125, 0.024), (0, 0, 0.192), "glaze", collide=True, mass=0.05 * VASE_MASS, **hit)
    _geom(v, "vase_lip", CYL, (0.019, 0.005), (0, 0, 0.218), "glaze")
    _geom(v, "vase_mouth", CYL, (0.0145, 0.0012), (0, 0, 0.2232), "glaze_in")
    # three flowers (visual only): stems fanned out from the mouth, a pistil and six petals each, two leaves
    fl = [(0.30, 0.0, 0.010, "petal_w"), (0.27, 2.2, 0.20, "petal_p"), (0.25, 4.2, 0.24, "petal_c")]
    for k, (htop, az, tilt, pm) in enumerate(fl):
        d = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
        base = np.array([0.0, 0.0, 0.215])
        L = (htop - 0.215) / d[2]
        tip = base + L * d
        mid = 0.5 * (base + tip)
        qz = np.zeros(4)
        mujoco.mju_quatZ2Vec(qz, d)
        _geom(v, f"stem{k}", CAP, (0.0022, 0.5 * L), mid, "stem", quat=qz)
        _geom(v, f"pistil{k}", SPH, (0.0075,), tip + 0.004 * d, "pistil")
        for j in range(6):
            a = j * math.pi / 3 + 0.3 * k
            r_ = np.array([math.cos(a), math.sin(a), 0.0])
            r_ = r_ - (r_ @ d) * d
            r_ /= np.linalg.norm(r_)
            c_ = tip + 0.016 * r_ + 0.002 * d
            # petal: an ellipsoid whose long axis is r_, flat along d
            Rm = np.c_[r_, np.cross(d, r_), d]
            qp = np.zeros(4)
            mujoco.mju_mat2Quat(qp, Rm.ravel())
            _geom(v, f"petal{k}_{j}", ELL, (0.015, 0.008, 0.0025), c_, pm, quat=qp)
        if k < 2:
            lp = base + 0.45 * L * d
            side = np.array([-math.sin(az + 1.2), math.cos(az + 1.2), 0.0])
            Rm = np.c_[side, np.cross(d, side), d]
            Rm = Rm @ np.array([[math.cos(0.5), 0, math.sin(0.5)], [0, 1, 0], [-math.sin(0.5), 0, math.cos(0.5)]])
            ql = np.zeros(4)
            mujoco.mju_mat2Quat(ql, Rm.ravel())
            _geom(v, f"leaf{k}", ELL, (0.024, 0.008, 0.0015), lp + 0.018 * side, "leaf", quat=ql)
    v.add_freejoint(name="vase_joint")

    spec.compile()
    info = dict(dir=str(out), objects={"target": dict(body="vase", uid=f"vase_p{p['index']}", pos=[p["x"], p["y"], TOP],
                                                      quat=quat_yaw(p["yaw"]))},
                tables={"etagere": dict(top=TOP, pos=[xc, yc, TOP - 0.5 * BOARD_T], size=[DEPTH, WIDTH, BOARD_T],
                                            quat=[1.0, 0.0, 0.0, 0.0])},
                lift=0.0, start=dict(xy=[0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], yaw=0.0), target="vase",
                placement=p, built_here="illustrative scene built from MuJoCo primitives (shelf_scene.py), no benchmark asset")
    for name, text in (("scene_locomani.xml", spec.to_xml()),):
        (out / name).write_text(text)
    import json

    (out / "scene_meta.json").write_text(json.dumps(dict(surfaces={}, objects={}, simple=info)))
    return info

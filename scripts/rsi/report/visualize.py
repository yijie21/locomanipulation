"""TODO 27 (user 2026-10-16): the scene RSI loop on pick_single as pictures and videos of the robot itself -- how each
accepted iteration turned a failing start into a success, and what the Reviewer saw that led it there.

Reads a loop sandbox (sandbox.py layout): every incumbent's diagnostic rounds (`diag/incumbent_<skills hash>/`, JSON +
50 Hz NPZ), the Reviewers' hypotheses and the ledgers. The rounds are replayed on the full ADT scene (the loop ran
them on the physics-only copy; the state is the same). Writes under --out:
* clue_NN.png -- iteration NN's clue, drawn on the round and the moment the Reviewer cited, with its reasoning;
* step_NN.mp4 -- that start before (the previous incumbent) and after (the new one), side by side, 2x speed, after a
  card with the clue; plus the <= 500 KB previews AGENTS.md requires (previews/);
* infeasible_book.png, infeasible_wallet.png -- the two tasks the Reviewers showed no grasp can do;
* overview.png -- showcase starts x incumbents, each cell the round's deciding moment.
Translucent magenta balls are the commanded wrist positions (the ghost-gripper rule; the traces carry no attitude).

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/rsi/report/visualize.py [--sandbox /workspace/code/rsi-scene/pick-a] \\
        [--out outputs/scene_rsi_20261016/visual] [--only clue,step,infeasible,overview]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # scripts/rsi/report/ since TODO 36.8
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "rsi_demo_20261015"), str(REPO / "src")]

import demo_video as dv  # noqa: E402

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"
FAMILY = "pick_single"
# the seed and every accepted incumbent, by skills hash (the diag directories' names)
VERSIONS = [("b897126e9dde", "种子"), ("7ec46c87bf5c", "第 0 轮后"), ("e8ce52c71938", "第 1 轮后"),
            ("acd864174766", "第 2 轮后"), ("0109bde905a8", "第 3 轮后")]  # fmt: skip
OBJECT = {"canister": "CoffeeCanisterLarge", "mug": "Mug_2", "jam": "Jam_Anon", "orange": "Orange_A",
          "dish_soap": "DishSoap_1", "tv_remote": "TvRemote_1", "thick_novel": "BookFiction1_01", "wallet": "Wallet_1"}  # fmt: skip
FINGER_RANGE = (-0.02, 0.0245)  # the dex1 finger slide joints
JAW_X, PALM_X = 0.1515, 0.1108  # wrist frame: jaw centre, front face of the gripper base (robot.xml collision mesh)
PANEL = (640, 400)
SPEED = 2  # video plays the round at this multiple of real time
GHOST = (1.0, 0.25, 1.0, 0.35)
YELLOW, GREEN, RED, CYAN, WHITE = (
    (1, 0.85, 0.1, 1),
    (0.2, 0.9, 0.35, 1),
    (1, 0.25, 0.2, 1),
    (0.2, 0.8, 1, 1),
    (1, 1, 1, 1),
)
BG, FG, DIM = (16, 18, 24), (235, 235, 235), (160, 165, 175)


# -- rounds -----------------------------------------------------------------------------------------------------------


class Round:
    def __init__(self, loop: Path, version: str, label: str):
        base = loop / "diag" / f"incumbent_{version}" / label
        self.label, self.version = label, version
        self.rec = json.loads(base.with_suffix(".json").read_text())
        self.z = dict(np.load(base.with_suffix(".npz")))
        self.key = next(k for k in self.z if k.startswith("obj_"))
        self.n = len(self.z["t"])

    @property
    def task(self) -> str:
        return self.label.rsplit("_pickup", 1)[0]

    def step(self, name: str) -> dict:
        return next((s for s in self.rec.get("steps") or [] if s["step"] == name), {})

    def i_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.z["t"], t), 0, self.n - 1))

    def knock(self, cm: float = 0.5) -> int | None:
        """First tick of the grasp step at which the task object has moved `cm` from where it was when the step began
        (the object settles a few mm when the round starts)."""
        i0 = self.i_at(self.step("grasp").get("t0", 0.0))
        d = np.linalg.norm(self.z[self.key][i0:, :3] - self.z[self.key][i0, :3], axis=1)
        hit = np.nonzero(d > cm / 100)[0]
        return int(i0 + hit[0]) if len(hit) else None

    def closing(self) -> int:
        """A tick just before the object starts to rise in the hand (the close)."""
        z0 = self.z[self.key][self.i_at(self.step("grasp").get("t0", 0.0)), 2]
        up = np.nonzero(self.z[self.key][:, 2] > z0 + 0.005)[0]
        return max(int(up[0]) - 15, 0) if len(up) else self.lifted()

    def lifted(self) -> int:
        return int(np.argmax(self.z[self.key][:, 2]))

    def deciding(self) -> int:
        """The tick that shows how the round went: the lift, the fall, or the end."""
        if self.rec["ok"]:
            return self.lifted()
        if self.rec.get("fell"):
            return int(np.argmin(self.z["robot"][:, 2]))
        return self.n - 1

    def outcome_zh(self) -> tuple[str, tuple]:
        if self.rec["ok"]:
            rise = self.step("grasp").get("rise_m")
            if rise is None:  # pick_top records no rise: the object's lift from where the step began
                z = self.z[self.key][:, 2]
                rise = float(z.max() - z[self.i_at(self.step("grasp").get("t0", 0.0))])
            return f"成功：拿起 {rise * 100:.1f} cm", (90, 220, 120)
        return "失败：" + why_zh(self.rec.get("plan_why") or self.rec.get("why") or ""), (255, 110, 100)


def why_zh(why: str) -> str:
    rules = [
        (r"pregrasp missed by ([\d.]+) cm", r"预抓位置差 \1 cm，放弃"),
        (r"place reach failed: \w+: jaw centre ([\d.]+) cm off", r"放下时手没对准（爪中心差 \1 cm），物体没放到台面上"),
        (r"no side fits the jaw", "规划拒绝（没有一面放得进爪口）"),
        (r"no clear stance with a path", "规划拒绝（找不到干净的站位）"),
        (r"no path on the occupancy map", "规划拒绝（导航图上没有路）"),
        (r"target knocked (\d+) cm in entry", r"伸手时把物体推开 \1 cm"),
        (r"target knocked (\d+) cm in settle", r"调整时把物体推开 \1 cm"),
        (r"entry timed out ([\d.]+) cm from the grasp", r"伸手超时，离抓取点还差 \1 cm"),
        (r"fell at ([\d.]+) s", r"\1 s 时摔倒"),
        (r"no bilateral pad contact", "合爪后两指没夹住"),
        (r"lost pad contact during the lift", "举起时滑脱"),
        (r"no hand touches it", "手没碰到物体"),
        (r"not between two fingers", "没夹在两指之间"),
        # motor.grip and skills/pick.py's lift check; left in English on the grasp_atomic pages' videos (2026-10-23)
        (r"nothing between the fingers", "合爪后两指之间是空的（没夹到）"),
        (r"not between the fingers after the lift", "抬起后物体不在两指之间（没拿住）"),
        # pick-repro iteration 6's refusal for the wallet (2026-10-23); shown in English on task_08 until this line
        (
            r"too thin for a level grasp over its support and too wide for a pinch from above",
            "规划拒绝（太薄不能平抓，太宽不能从上捏）",
        ),
    ]
    for pat, rep in rules:
        m = re.search(pat, why)
        if m:
            return m.expand(rep)
    return why[:40]


# -- the scene ---------------------------------------------------------------------------------------------------------


class Scene:
    def __init__(self, scene_dir: Path = SCENE):
        self.m = mujoco.MjModel.from_xml_path(str(scene_dir / "scene_locomani.xml"))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 1920, 1080
        self.m.vis.headlight.ambient[:] = 0.38
        self.m.vis.headlight.diffuse[:] = 0.55
        self.d = mujoco.MjData(self.m)
        self.meta = json.loads((scene_dir / "scene_meta.json").read_text())
        m = self.m
        pel = m.body("pelvis").id
        self.free = m.jnt_qposadr[next(j for j in range(m.njnt) if m.jnt_type[j] == 0 and m.jnt_bodyid[j] == pel)]
        self.joints = [m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES]
        self.grip = {
            s: [m.jnt_qposadr[m.joint(f"{s}_hand_gripper_{k}_joint").id] for k in (1, 2)] for s in ("left", "right")
        }
        self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[pel]}
        self.shell = [g for g in range(m.ngeom) if m.geom(g).name.startswith("adt_shell")]
        self.renderers: dict = {}
        self.opt = mujoco.MjvOption()

    def see_through(self, body: str, alpha: float) -> list:
        """Make one body's geoms (and their materials) translucent; returns what restore() puts back."""
        b = self.m.body(body).id
        gs = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]
        mats = sorted({int(self.m.geom_matid[g]) for g in gs if self.m.geom_matid[g] >= 0})
        saved = [(gs, self.m.geom_rgba[gs].copy()), (mats, self.m.mat_rgba[mats].copy())]
        self.m.geom_rgba[gs, 3] = alpha
        if mats:
            self.m.mat_rgba[mats, 3] = alpha
        return saved

    def restore(self, saved: list) -> None:
        (gs, g_rgba), (mats, m_rgba) = saved
        self.m.geom_rgba[gs] = g_rgba
        if mats:
            self.m.mat_rgba[mats] = m_rgba

    def obj_qadr(self, body: str) -> int:
        b = self.m.body(body).id
        return self.m.jnt_qposadr[next(j for j in range(self.m.njnt) if self.m.jnt_bodyid[j] == b)]

    def set(self, r: Round, i: int) -> None:
        q = self.m.qpos0.copy()
        z = r.z
        q[self.free : self.free + 7] = z["robot"][i]
        q[self.joints] = z["joints"][i]
        a = self.obj_qadr(r.key[4:])
        q[a : a + 7] = z[r.key][i]
        for s, side in enumerate(("left", "right")):
            # the measured fingers (traces from 2026-10-20 on); before, only the command was kept: it is past the joint
            # limit while gripping (-0.06), so drawn as is the fingers crossed outside the object (TODO 27.14)
            q[self.grip[side]] = (
                z["fingers"][i, 2 * s : 2 * s + 2] if "fingers" in z else np.clip(z["jaw"][i, s], *FINGER_RANGE)
            )
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)

    def hand(self, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(jaw centre, palm front, wrist rotation) of one hand in the current state."""
        b = self.m.body(f"{side}_wrist_yaw_link").id
        p, R = self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()
        return p + R @ [JAW_X, 0, 0], p + R @ [PALM_X, 0, 0], R

    def blocked(self, cam_pos: np.ndarray, target: np.ndarray) -> bool:
        v = target - cam_pos
        dist = float(np.linalg.norm(v))
        gid = np.zeros(1, np.int32)
        groups = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        hit = mujoco.mj_ray(self.m, self.d, cam_pos, v / dist, groups, 1, -1, gid)
        if hit < 0 or hit > dist - 0.12:
            return False
        b = self.m.geom_bodyid[gid[0]]
        return b not in self.robot_bodies

    def clear_distance(self, lookat, az: float, el: float) -> float:
        """How far back from `lookat` a camera at (az, el) can sit before a wall or furniture is in the way."""
        a, e = math.radians(az), math.radians(el)
        back = -np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        gid = np.zeros(1, np.int32)
        groups = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        skip = 0.35  # the object in the hand and the surface it stands on sit at the look-at point
        hit = mujoco.mj_ray(self.m, self.d, np.asarray(lookat, float) + skip * back, back, groups, 1, -1, gid)
        hit = hit + skip if hit >= 0 else hit
        while hit >= 0 and self.m.geom_bodyid[gid[0]] in self.robot_bodies:  # look past the robot itself
            start = np.asarray(lookat, float) + (hit + 0.02) * back
            h2 = mujoco.mj_ray(self.m, self.d, start, back, groups, 1, -1, gid)
            hit = -1 if h2 < 0 else hit + 0.02 + h2
        return 99.0 if hit < 0 else max(0.6, hit - 0.15)

    def render(self, cam: dict, size=PANEL, marks=(), ghosts=(), top_down=False) -> np.ndarray:
        key = size
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.m, size[1], size[0])
        rd = self.renderers[key]
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:], c.distance, c.azimuth, c.elevation = (
            cam["lookat"],
            cam["distance"],
            cam["azimuth"],
            cam["elevation"],
        )
        saved = self.m.geom_group[self.shell].copy()
        if top_down:
            self.m.geom_group[self.shell] = 5
        rd.update_scene(self.d, c, self.opt)
        self.m.geom_group[self.shell] = saved
        scn = rd.scene
        for mk in list(marks) + [("sphere", g, 0.035, GHOST) for g in ghosts]:
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            kind = mk[0]
            if kind == "sphere":
                _, pos, rad, rgba = mk
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]), np.asarray(pos, float),
                                    np.eye(3).ravel(), np.array(rgba, np.float32))  # fmt: skip
            elif kind == "line":
                _, a, b, width, rgba = mk
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.array(rgba, np.float32))  # fmt: skip
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float)
                )
            elif kind == "box":
                _, pos, half, yaw, rgba = mk
                R = rot_z(yaw)
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, float), np.asarray(pos, float),
                                    R.ravel(), np.array(rgba, np.float32))  # fmt: skip
            elif kind == "ring":  # a flat disc outline on the floor: radius, colour
                _, pos, rad, rgba = mk
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([rad, 0.004, 0]),
                                    np.asarray(pos, float), np.eye(3).ravel(), np.array(rgba, np.float32))  # fmt: skip
            scn.ngeom += 1
        return rd.render().copy()

    def obj_box(self, name: str, rgba, grow: float = 0.01):
        o = self.meta["objects"][name]
        w, x, y, z = o["init_quat"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        half = np.asarray(o["box"]["half_extent"]) + grow
        return ("box", o["init_pos"], half, yaw, rgba)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def cam_pos(cam: dict) -> np.ndarray:
    az, el = math.radians(cam["azimuth"]), math.radians(cam["elevation"])
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.asarray(cam["lookat"]) - cam["distance"] * fwd


def pick_azimuth(sc: Scene, rounds: list[Round], target: np.ndarray, base: float, dist: float, el: float) -> float:
    """The azimuth near `base` from which the robot and the target are least often hidden over the rounds."""
    best = None
    for off in (0, 15, -15, 30, -30, 50, -50, 75, -75, 110, -110, 145, -145, 180):
        az, miss = base + off, 0
        for r in rounds:
            for i in range(r.n // 2, r.n, 25):  # the second half: the approach and the grasp
                sc.set(r, i)
                p = r.z["robot"][i, :3]
                look = follow_lookat(p, target)
                cam = dict(lookat=look, distance=follow_distance(p, target, dist), azimuth=az, elevation=el)
                room = sc.clear_distance(look, az, el)
                if room < 0.85 * cam["distance"]:  # the camera would have to come in close
                    cam["distance"] = room
                    miss += 1
                c = cam_pos(cam)
                miss += sc.blocked(c, p + [0, 0, 0.25]) + sc.blocked(c, target)
        score = (miss, abs(off))
        if best is None or score < best[0]:
            best = (score, az)
    return best[1]


def follow_lookat(pelvis, target) -> np.ndarray:
    """Between the robot and the target; near the target once the robot is close (the grasp is what matters)."""
    d = float(np.linalg.norm(np.asarray(pelvis[:2]) - target[:2]))
    w = float(np.clip(d / 2.0, 0.3, 0.5))
    xy = np.asarray(target[:2]) + w * (np.asarray(pelvis[:2]) - target[:2])
    return np.array([xy[0], xy[1], 0.5 * (0.8 + target[2])])


def follow_distance(pelvis, target, base: float) -> float:
    return float(np.clip(base + 0.65 * np.linalg.norm(np.asarray(pelvis[:2]) - target[:2]), base, 5.0))


def grasp_view_azimuth(r: Round) -> float:
    """From the grasping hand's side and a little behind: the hand, the object and the face of the furniture."""
    target = r.z[r.key][0, :3]
    stance = r.step("walk").get("stance")
    if stance is None:
        stance = r.z["robot"][-1, :2]
    facing = math.degrees(math.atan2(target[1] - stance[1], target[0] - stance[0]))
    return facing + (80 if used_side(r) == "right" else -80)


# -- text and composition ----------------------------------------------------------------------------------------------


def font(size: int):
    return dv._font(size)


def wrap(text: str, width_px: int, size: int) -> list[str]:
    f, out = font(size), []
    for para in text.split("\n"):
        line = ""
        for ch in para:
            if f.getlength(line + ch) > width_px and line and ch not in "，。、；：）」》！？,.;:)":
                out.append(line)
                line = ch.lstrip() if ch == " " else ch
            else:
                line += ch
        out.append(line)
    return out


def text_block(blocks: list[tuple[str, int, tuple]], width: int, height: int, pad: int = 24) -> Image.Image:
    """(text, size, colour) paragraphs, wrapped, top-down."""
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    y = pad
    for text, size, colour in blocks:
        for line in wrap(text, width - 2 * pad, size):
            d.text((pad, y), line, font=font(size), fill=colour)
            y += int(size * 1.42)
        y += int(size * 0.5)
    return img


def label(img: np.ndarray, text: str, colour=FG, size: int = 22, pos=(12, 10)) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    w = d.textlength(text, font=font(size))
    d.rectangle([pos[0] - 6, pos[1] - 4, pos[0] + w + 6, pos[1] + size + 8], fill=(0, 0, 0))
    d.text(pos, text, font=font(size), fill=colour)
    return np.asarray(im)


def clue_image(
    out: Path, title: str, images: list[tuple[np.ndarray, str]], blocks: list[tuple[str, int, tuple]]
) -> None:
    """Renders stacked on the left (each with its caption), the Reviewer's reasoning on the right."""
    W, left_w, top = 1760, 1080, 76
    iw = left_w - 24
    caps = [wrap(cap, iw, 20) for _, cap in images]
    ih = [int(iw * img.shape[0] / img.shape[1]) for img, _ in images]
    H = max(top + sum(27 * len(c) + 16 + h for c, h in zip(caps, ih)) + 16, 900)
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((24, 18), title, font=font(34), fill=FG)
    y = top
    for (img, _), lines, h in zip(images, caps, ih):
        for li, line in enumerate(lines):
            d.text((24, y + li * 27), line, font=font(20), fill=DIM)
        y += 27 * len(lines) + 6
        canvas.paste(Image.fromarray(img).resize((iw, h)), (24, y))
        y += h + 10
    canvas.paste(text_block(blocks, W - left_w - 10, H - top), (left_w + 10, top))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


# -- videos ------------------------------------------------------------------------------------------------------------


def panel_frames(sc: Scene, r: Round, target: np.ndarray, az: float, dist: float, el: float):
    """One round as panels at SPEED x real time (25 fps), following the robot and the target."""
    look = dist_s = None
    for i in range(0, r.n, 2 * SPEED):
        sc.set(r, i)
        p = r.z["robot"][i, :3]
        lk, ds = follow_lookat(p, target), follow_distance(p, target, dist)
        ds = min(ds, sc.clear_distance(lk, az, el))
        look = lk if look is None else look + 0.06 * (lk - look)
        dist_s = ds if dist_s is None else dist_s + 0.06 * (ds - dist_s)
        cam = dict(lookat=look, distance=dist_s, azimuth=az, elevation=el)
        ghosts = [r.z["ref_wrists"][i, k] for k in range(2)]
        yield sc.render(cam, PANEL, ghosts=ghosts), float(r.z["t"][i])


def step_video(sc: Scene, before: Round, after: Round, names: tuple[str, str], card_lines, out: Path) -> Path:
    target = before.z[before.key][0, :3]
    az = pick_azimuth(sc, [before, after], target, grasp_view_azimuth(after), 1.3, -26)
    caps = []
    for r, name in ((before, names[0]), (after, names[1])):
        txt, col = r.outcome_zh()
        caps.append(dv.caption_strip([(name, FG), (txt, col)], PANEL[0]))
    strip = np.hstack(caps)
    W, Hh = 2 * PANEL[0], PANEL[1] + strip.shape[0]
    w = dv.Writer(out, W, Hh)
    w.write(dv.card(card_lines, size=(W, Hh)), int(6 * dv.FPS))
    gens = [panel_frames(sc, before, target, az, 1.3, -26), panel_frames(sc, after, target, az, 1.3, -26)]
    last = [None, None]
    while True:
        tiles, live = [], False
        for k, g in enumerate(gens):
            try:
                last[k] = next(g)
                live = True
            except StopIteration:
                pass
            tiles.append(last[k][0])
        if not live:
            break
        frame = np.vstack([strip, np.hstack(tiles)])
        t = max(x[1] for x in last)
        frame = label(frame, f"t = {t:5.1f} s（{SPEED}× 速度）", DIM, 18, (W - 250, strip.shape[0] + 8))
        w.write(frame)
    w.write(frame, int(1.5 * dv.FPS))
    w.close()
    dv.preview(out, out.parent / "previews" / out.stem)
    return out


# -- the pieces --------------------------------------------------------------------------------------------------------


def side_cam(sc: Scene, r: Round, i: int, side: str, dist=0.62, el=-12.0, swing=90.0) -> dict:
    """A close camera beside the hand, square to the approach (jaw centre -> object)."""
    sc.set(r, i)
    jaw, _, _ = sc.hand(side)
    obj = r.z[r.key][i, :3]
    az = math.degrees(math.atan2(obj[1] - jaw[1], obj[0] - jaw[0])) + swing
    return dict(lookat=0.5 * (jaw + obj), distance=dist, azimuth=az, elevation=el)


def hand_marks(sc: Scene, r: Round, i: int, side: str, rad: float = 0.011) -> list:
    sc.set(r, i)
    jaw, palm, _ = sc.hand(side)
    obj = r.z[r.key][i, :3]
    return [("sphere", jaw, rad, YELLOW), ("sphere", palm, rad, RED), ("sphere", obj, rad, GREEN),
            ("line", jaw, obj, 0.003, YELLOW)]  # fmt: skip


def obj_mark(sc: Scene, r: Round, i: int, rgba=(0.2, 0.9, 0.35, 0.18)):
    """The task object's box at tick i (meta half extents, the trace's yaw)."""
    half = np.asarray(sc.meta["objects"][r.key[4:].removesuffix("_object")]["box"]["half_extent"]) + 0.004
    return ("box", r.z[r.key][i, :3], half, yaw_of(r.z[r.key][i, 3:7]), rgba)


def used_side(r: Round) -> str:
    return r.step("walk").get("side") or "right"


def clue_0(sc: Scene, loop: Path, out: Path):
    before, after = (
        Round(loop, VERSIONS[0][0], "canister_pickup_s100053"),
        Round(loop, VERSIONS[1][0], "canister_pickup_s100053"),
    )
    i = before.knock(0.5)
    sc.set(before, i)
    jaw, palm, _ = sc.hand("right")
    obj = before.z[before.key][i, :3]
    gap = np.linalg.norm((obj - jaw)[:2]) * 100
    glass = sc.see_through(before.key[4:], 0.35)
    cam = side_cam(sc, before, i, "right", dist=0.5, el=-62, swing=-90)
    a = sc.render(cam, (1280, 720), hand_marks(sc, before, i, "right", 0.014))
    j = after.closing()
    sc.set(after, j)
    jaw2, _, _ = sc.hand("right")
    gap2 = np.linalg.norm((after.z[after.key][j, :3] - jaw2)[:2]) * 100
    cam2 = side_cam(sc, after, j, "right", dist=0.5, el=-62, swing=-90)
    b = sc.render(cam2, (1280, 720), hand_marks(sc, after, j, "right", 0.014))
    sc.restore(glass)
    clue_image(out / "clue_00.png", "第 0 轮 · Reviewer 的线索：罐子是被手掌推开的，不是手不准",
               [(a, f"种子版 canister s100053，罐子开始被推动的那一帧（罐子画成半透明）：爪口中心（黄）离罐子中心（绿）还有 {gap:.1f} cm，手掌前端（红）已经贴到罐子"),
                (b, f"第 0 轮采纳的版本，同一个起点，合爪前：预抓取站得更远、到位判据提前，爪口停在离中心 {gap2:.1f} cm 处合爪，手掌没有先碰到罐子，成功拿起")],
               [("它看了什么", 26, (255, 210, 120)),
                ("读 50 Hz 轨迹，用正运动学算每一帧的爪口和手掌位置，对照罐子什么时候开始动。", 22, FG),
                ("发现", 26, (255, 210, 120)),
                ("· s100053、s100071：罐子开始动时爪口还差 2.0 / 1.6 cm 才到中心，手掌前端正好在罐子近端面（+0.2 / +0.6 cm），横向误差 ≤ 3 mm——是手掌在推，不是手没对准。", 22, FG),
                ("· 查机器人模型：夹爪底座前端在爪口中心后 4.07 cm（腕部碰撞网格止于 x = 0.1108，爪口中心在 0.1515）。凡是从中心往后超过约 4 cm 的物体（罐子 6.2 cm、洗洁精 5.95、杯子 5.8），手掌都会先撞到近端面。", 22, FG),
                ("· 再查爪口宽度规则：8.5 cm 比硬件严，杯子 9.0 cm 宽被误拒，而爪口能开 9.48 cm。", 22, FG),
                ("它怎么验证", 26, (255, 210, 120)),
                ("自己在仿真里单局试：预抓取 0.10 → 0.14 m、入口门收紧到 6 mm、到位判据提前 2 cm，s100036 成功，原来唯一成功的果酱局保持成功。", 22, FG),
                ("提出的方案 → 结果", 26, (255, 210, 120)),
                ("改进入几何（6 个参数）+ 爪口规则 8.5 → 9.2 cm。预测修好 4 局，探针实测 1 → 5/50，4 局全中。验收门 0 → 7/60，通过。", 22, (150, 230, 160))])  # fmt: skip
    return before, after


def topdown_cam(center, span: float) -> dict:
    return dict(lookat=np.array([center[0], center[1], 0.3]), distance=span, azimuth=90.0, elevation=-89.0)


def path_marks(r: Round, rgba, upto: int | None = None, every: int = 10) -> list:
    p = r.z["robot"][: upto or r.n : every, :3].copy()
    p[:, 2] = 0.03
    return [("line", p[k], p[k + 1], 0.012, rgba) for k in range(len(p) - 1)]


def clue_1(sc: Scene, loop: Path, out: Path):
    before, after = (
        Round(loop, VERSIONS[1][0], "canister_pickup_s100045"),
        Round(loop, VERSIONS[2][0], "canister_pickup_s100045"),
    )
    fall = before.deciding()
    stool = sc.obj_box("StepStool", (1, 0.15, 0.1, 0.55), 0.0)
    target = before.z[before.key][0, :3]
    start = before.z["robot"][0, :2]
    centre = 0.5 * (start + target[:2])
    span = max(3.2, 1.6 * np.linalg.norm(start - target[:2]) + 1.4)
    marks_b = [stool] + path_marks(before, RED, fall) + [("sphere", np.r_[before.z["robot"][fall, :2], 0.1], 0.07, RED),
                                                       ("sphere", target, 0.05, GREEN)]  # fmt: skip
    marks_a = [stool] + path_marks(after, GREEN) + [("sphere", target, 0.05, GREEN)]
    sc.set(before, fall)
    a = sc.render(topdown_cam(centre, span), (1280, 720), marks_b, top_down=True)
    sc.set(after, after.lifted())
    b = sc.render(topdown_cam(centre, span), (1280, 720), marks_a, top_down=True)
    clue_image(out / "clue_01.png", "第 1 轮 · Reviewer 的线索：导航图上没有这个凳子",
               [(a, "第 0 轮后的版本，canister s100045 从上往下看：红线是走过的路，红框是 25 cm 高的踏脚凳，机器人 7.4 s 摔倒在凳子旁"),
                (b, "第 1 轮采纳的版本，同一个起点：技能自己的导航图补上了地上的物体，绕开凳子从推车另一侧拿到罐子")],
               [("它看了什么", 26, (255, 210, 120)),
                ("两次摔倒的位置、导航用的占用图、场景文件里每个物体的占地范围；还试了只看场景文件给的那条边以外的边。", 22, FG),
                ("发现", 26, (255, 210, 120)),
                ("· 摔倒都发生在地上的物体旁边：踏脚凳（x -2.76…-2.42，y -1.20…-0.76，高 0.25 m）在占用图里 0% 被标出；白垃圾桶 8%。canister s100045 在 (-2.49, -1.25) 摔倒，出生点离凳子只有 0.20 m；jam s100055 的站位就在垃圾桶上，身体倾斜 38–57° 几秒后摔倒。", 22, FG),
                ("· 占用图只由墙和台面生成，地上的凳子、垃圾桶都不在里面。", 22, FG),
                ("· 规划只用场景文件给的那条边（来自 SONIC 时代的规则），推车西侧明明是空地却从不考虑；几个“伸不到”的罐子从西侧只有 0.09–0.17 m 深。", 22, FG),
                ("它怎么验证", 26, (255, 210, 120)),
                ("原型：四条边都试 + 把地上物体补进地图 + 站位侧移。单局试：s100000、jam s100038、s100047 成功，s100036 和 s100055 不再摔倒。并预测了参数探针会弄坏 s100036（往西走时会踩到凳子），实测果然如此。", 22, FG),
                ("提出的方案 → 结果", 26, (255, 210, 120)),
                ("Manager 要求写成一个“接近规划”改动；Engineer 改 pick.py（+163 行）。验收门 7 → 13/60，通过，摔倒 1 → 0。", 22, (150, 230, 160))])  # fmt: skip
    return before, after


def clue_2(sc: Scene, loop: Path, out: Path):
    before, after = (
        Round(loop, VERSIONS[2][0], "tv_remote_pickup_s100005"),
        Round(loop, VERSIONS[3][0], "tv_remote_pickup_s100005"),
    )
    g = before.step("grasp")
    t_end = g.get("t1") or before.z["t"][-1]
    i = before.i_at(t_end - 0.5)
    sc.set(before, i)
    jaw, _, _ = sc.hand("right")
    gap = np.linalg.norm(before.z[before.key][i, :3] - jaw) * 100
    remote = ("box", before.z[before.key][i, :3], np.array([0.0178, 0.0822, 0.0126]) + 0.004,
              yaw_of(before.z[before.key][i, 3:7]), (0.2, 0.9, 0.35, 0.35))  # fmt: skip
    a = sc.render(
        side_cam(sc, before, i, "right", dist=0.7, el=-20), (1280, 720), hand_marks(sc, before, i, "right") + [remote]
    )
    j = after.closing()
    sc.set(after, j)
    s = used_side(after)
    jaw2, _, _ = sc.hand(s)
    cam = dict(lookat=0.5 * (jaw2 + after.z[after.key][j, :3]), distance=0.7,
               azimuth=side_cam(sc, after, j, s)["azimuth"] - 30, elevation=-25)  # fmt: skip
    b = sc.render(cam, (1280, 720), hand_marks(sc, after, j, s) + [obj_mark(sc, after, j)])
    clue_image(out / "clue_02.png", "第 2 轮 · Reviewer 的线索：遥控器侧着抓，几何上就进不去",
               [(a, f"第 1 轮后的版本，tv_remote s100005 超时前：只能沿遥控器 16.4 cm 长边伸进去，手掌先顶到近端，爪口（黄）离中心（绿）还差 {gap:.1f} cm"),
                (b, "第 2 轮采纳的版本，同一个起点，合爪前：手腕朝下，从上往下横着捏住 3.6 cm 宽的那一面（绿框是遥控器）")],
               [("它看了什么", 26, (255, 210, 120)),
                ("遥控器的尺寸（3.6 × 16.4 × 2.5 cm）、爪口和手掌的几何、两局超时的轨迹，以及另外 4 局为什么在规划阶段就被拒。", 22, FG),
                ("发现", 26, (255, 210, 120)),
                ("· 唯一放得进爪口的是 3.6 cm 宽的那一面，所以手只能沿 16.4 cm 长边伸进去；手掌在爪口中心后 4.07 cm，先撞上近端时爪口离中心还有约 12 cm——s100005、s100050 分别差 14.5、12.0 cm 超时。", 22, FG),
                ("· 另外 4 个遥控器在规划时就被“朝向偏差 ±40° 以内”的规则拒掉。", 22, FG),
                ("它怎么验证", 26, (255, 210, 120)),
                ("判别实验：如果是被挡住而不是太慢，把进入超时 9 s → 20 s 结果应该一局都不变——探针实测 11 → 11，证实。再把从上往下捏（手腕朝下、横跨窄边、指尖离茶几 4 mm）做成原型，单局试 6 个遥控器成 5 个。", 22, FG),
                ("提出的方案 → 结果", 26, (255, 210, 120)),
                ("Engineer 新写 pick_top（只对又扁又在矮台面上的物体生效）。验收门 13 → 21/60，通过；遥控器 0 → 6/7。", 22, (150, 230, 160))])  # fmt: skip
    return before, after


def mirror(stance, obj, face_deg) -> np.ndarray:
    n = np.array([math.cos(math.radians(face_deg)), math.sin(math.radians(face_deg))])
    v = np.asarray(stance) - np.asarray(obj[:2])
    lat = v - (v @ n) * n
    return np.asarray(stance) - 2 * lat


def clue_3(sc: Scene, loop: Path, out: Path):
    before, after = Round(loop, VERSIONS[3][0], "jam_pickup_s100030"), Round(loop, VERSIONS[4][0], "jam_pickup_s100030")
    w = after.step("walk")
    target = after.z[after.key][0, :3]
    stance_l = np.asarray(w["stance"])
    face = math.degrees(yaw_of(after.z["robot"][after.i_at(w["t1"]), 3:7]))  # the robot's heading at the stance
    stance_r = mirror(stance_l, target, face)
    stool = sc.obj_box("BlackBarStool_B", (1, 0.65, 0.1, 0.35), 0.0)
    span = 3.0
    centre = 0.5 * (stance_l + target[:2])
    marks = [stool, ("ring", np.r_[stance_r, 0.75], 0.25, (1, 0.15, 0.15, 0.8)),
             ("ring", np.r_[stance_l, 0.75], 0.25, (0.2, 0.9, 0.35, 0.8)), ("sphere", target, 0.04, GREEN)]  # fmt: skip
    sc.set(before, 0)
    a = sc.render(topdown_cam(centre, span), (1280, 720), marks, top_down=True)
    j = after.closing()
    sc.set(after, j)
    cam = dict(lookat=np.r_[0.35 * after.z["robot"][j, :2] + 0.65 * target[:2], 0.85], distance=1.6,
               azimuth=face + 130, elevation=-24)  # fmt: skip
    b = sc.render(cam, (1280, 720), [obj_mark(sc, after, j)])
    clue_image(out / "clue_03.png", "第 3 轮 · Reviewer 的线索：右手的站位全被吧台凳挡住，换左手就有地方站",
               [(a, "从上往下看岛台西侧：橙框是吧台凳，红圈是右手需要的站位（示意：左手站位按机器人朝向镜像），绿圈是左手的站位，圈的半径是 0.25 m 间隙规则，绿球是果酱"),
                (b, "第 3 轮采纳的版本，jam s100030，合爪前：右手找不到站位时镜像站位、改用左手（绿框是果酱）")],
               [("它看了什么", 26, (255, 210, 120)),
                ("岛台 8 局“找不到干净站位”的失败：每个候选站位到吧台凳的距离、0.25 m 间隙规则，以及岛台另外两条边的深度。", 22, FG),
                ("发现", 26, (255, 210, 120)),
                ("· 岛台西侧是空地，但用右手时每个站位离吧台凳只有 0–0.21 m，被 0.25 m 的间隙规则拒掉；岛台两头离物体 0.40–0.87 m，伸不到。", 22, FG),
                ("· 把站位镜像、改用左手，身体就站到凳子另一侧，离物体的深度同样是 0.09–0.34 m。", 22, FG),
                ("它怎么验证", 26, (255, 210, 120)),
                ("先排除对立解释“只是 0.25 m 规则太严”：放宽到 0.2 m 只多放进 1 局，上一轮放到 0 一局没修好。再单局试左手：果酱 s100002/30/64、橙子 s100039、杯子 s100046 成功。", 22, FG),
                ("提出的方案 → 结果", 26, (255, 210, 120)),
                ("Engineer 加“右手不行换左手”的回退（只在原规划失败时走），再加出生点离地上物体 < 8 cm 不走新分支的保护。验收门 21 → 26/60，通过；果酱 1 → 4/8。", 22, (150, 230, 160))])  # fmt: skip
    return before, after


def infeasible(sc: Scene, loop: Path, out: Path) -> None:
    for task, name, nbrs, close_el, title, blocks in (
        (
            "thick_novel",
            "BookFiction1_01",
            ("BookFiction10_00", "BookFiction1_00"),
            -10,
            "做不到 · 书（thick_novel）",
            [
                (
                    "Reviewer 的判断（第 0 轮提出；只有几何证据和各轮正常尝试的失败，没有专门的闭环探针）",
                    26,
                    (255, 210, 120),
                ),
                (
                    "· 目标书（绿色半透明框）夹在两本书（红色半透明框）中间，两侧书的中心离它只有 3.3 和 3.5 cm，包围盒互相重叠 1.2–1.4 cm，手指插不进去。",
                    22,
                    FG,
                ),
                ("· 两侧书顶 0.638 / 0.633 m，目标书顶 0.630 m，没有露出来的部分可以捏。", 22, FG),
                ("· 书长 16.6 cm，比 9.5 cm 的爪口宽；从上往下抓需要约 18.5 cm 空间，上层隔板只留 10.4 cm。", 22, FG),
                ("· 这一侧 7 个起点都用同一个摆放（展示架太挤，没法随机），所以全部做不到。", 22, FG),
            ],
        ),
        (
            "wallet",
            "Wallet_1",
            (),
            -28,
            "做不到 · 钱包（wallet）",
            [
                (
                    "Reviewer 的判断（第 0 轮提出，第 4 轮闭环探针证实：平着、翻转手腕、倾斜、从上往下 4 种抓法都失败）",
                    26,
                    (255, 210, 120),
                ),
                ("· 钱包在 1.12 m 高的顶层，离前沿 11.8 cm。", 22, FG),
                (
                    "· 平着伸进去：夹爪底座比爪口中心低 3.1 cm，爪口对准 2.8 cm 厚的钱包时底座会撞到台面；抬到底座不碰台面，爪垫只剩 1.07 cm 搭在钱包上。",
                    22,
                    FG,
                ),
                (
                    "· 从上往下：手腕朝下够到这个高度时逆运动学到极限（48°），第 4 轮的探针在闭环里也没有一局成功。",
                    22,
                    FG,
                ),
            ],
        ),
    ):
        o = sc.meta["objects"][name]
        r = Round(
            loop,
            VERSIONS[4][0],
            next(
                p.stem for p in sorted((loop / "diag" / f"incumbent_{VERSIONS[4][0]}").glob(f"{task}_pickup_s*.json"))
            ),
        )
        sc.set(r, 0)
        pos = np.asarray(o["init_pos"])
        marks = [sc.obj_box(name, (0.2, 0.95, 0.35, 0.45), 0.004)] + [
            sc.obj_box(n, (1, 0.2, 0.2, 0.4), 0.002) for n in nbrs
        ]
        surf = sc.meta["surfaces"][o["surface"]]
        edge = surf["approach_edges"][0]  # the side the robot comes from: "+x" / "-y" ... in the surface frame
        local = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[edge]
        n = rot_z(surf["yaw"])[:2, :2] @ np.asarray(local, float)
        front = math.degrees(math.atan2(-n[1], -n[0]))  # looking at the shelf from where the robot stands
        imgs = []
        for off, el, dist, cap in (
            (20, close_el, 0.6, "近景：从机器人来的那一面看"),
            (35, -12, 2.2, "远景：整个展示架"),
        ):
            cam = dict(
                lookat=pos if dist < 1 else np.r_[surf["xy"], 0.9], distance=dist, azimuth=front + off, elevation=el
            )
            imgs.append((sc.render(cam, (1280, 720), marks), cap))
        clue_image(out / f"infeasible_{task}.png", title, imgs, blocks)


def overview(sc: Scene, loop: Path, out: Path, labels: list[str]) -> None:
    cw, ch = 400, 250
    head = 46
    rowlab = 220
    W, H = rowlab + cw * len(VERSIONS), head + ch * len(labels)
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    for k, (_, name) in enumerate(VERSIONS):
        d.text((rowlab + k * cw + 12, 10), name, font=font(24), fill=FG)
    for row, lab in enumerate(labels):
        rounds = [Round(loop, v, lab) for v, _ in VERSIONS]
        target = rounds[0].z[rounds[0].key][0, :3]
        ends = [r.z["robot"][r.deciding(), :2] for r in rounds]
        far = max(ends, key=lambda p: np.linalg.norm(p - target[:2]))
        base = math.degrees(math.atan2(target[1] - far[1], target[0] - far[0])) + 40
        az = pick_azimuth(sc, [rounds[-1]], target, base, 2.0, -24)
        task_zh = {"canister": "咖啡罐", "tv_remote": "遥控器", "jam": "果酱", "dish_soap": "洗洁精", "mug": "杯子",
                   "orange": "橙子"}.get(rounds[0].task, rounds[0].task)  # fmt: skip
        y0 = head + row * ch
        for line_k, txt in enumerate((task_zh, lab.rsplit("_", 1)[1])):
            d.text(
                (14, y0 + 80 + 34 * line_k), txt, font=font(26 if line_k == 0 else 20), fill=FG if line_k == 0 else DIM
            )
        for k, r in enumerate(rounds):
            i = r.deciding()
            sc.set(r, i)
            p = r.z["robot"][i, :3]
            look = follow_lookat(p, target)
            dist = min(follow_distance(p, target, 2.0), sc.clear_distance(look, az, -24))
            img = sc.render(dict(lookat=look, distance=dist, azimuth=az, elevation=-24), (cw * 2, ch * 2))
            txt, col = r.outcome_zh()
            txt = txt.replace("规划拒绝（", "规划拒绝\n（") if len(txt) > 16 else txt
            im = Image.fromarray(img).resize((cw - 8, ch - 8))
            dd = ImageDraw.Draw(im)
            lines = txt.split("\n")
            dd.rectangle([0, ch - 8 - 30 * len(lines) - 8, cw, ch], fill=(0, 0, 0))
            for li, t in enumerate(lines):
                dd.text((8, ch - 8 - 30 * (len(lines) - li) - 2), t, font=font(20), fill=col)
            canvas.paste(im, (rowlab + k * cw + 4, y0 + 4))
    canvas.save(out / "overview.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", type=Path, default=Path("/workspace/code/rsi-scene/pick-a"))
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "scene_rsi_20261016" / "visual")
    ap.add_argument("--only", default="clue,step,infeasible,overview")
    a = ap.parse_args(argv)
    loop = a.sandbox / "outputs" / "scene_rsi_20261016" / FAMILY
    only = set(a.only.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    sc = Scene()
    clues = [clue_0, clue_1, clue_2, clue_3]
    cards = [
        [
            "第 0 轮：罐子是被手掌推开的",
            "夹爪底座在爪口中心后 4.07 cm → 预抓取站远、到位判据提前、爪口规则放宽",
            "canister s100053 · 左：种子 · 右：第 0 轮采纳的版本",
        ],
        [
            "第 1 轮：导航图上没有地上的凳子",
            "补上地上物体、四条边都试、站位侧移 → Engineer 写接近规划",
            "canister s100045 · 左：第 0 轮后 · 右：第 1 轮采纳的版本",
        ],
        [
            "第 2 轮：遥控器侧着抓进不去",
            "超时放宽到 20 s 一局不变 → 是被挡住不是太慢 → Engineer 写从上往下捏",
            "tv_remote s100005 · 左：第 1 轮后 · 右：第 2 轮采纳的版本",
        ],
        [
            "第 3 轮：右手的站位全被吧台凳挡住",
            "镜像站位改用左手 → Engineer 写换手回退",
            "jam s100030 · 左：第 2 轮后 · 右：第 3 轮采纳的版本",
        ],
    ]
    for k, fn in enumerate(clues):
        if "clue" in only or "step" in only:
            before, after = fn(sc, loop, a.out) if "clue" in only else (None, None)
            if "step" in only:
                if before is None:
                    labs = [
                        "canister_pickup_s100053",
                        "canister_pickup_s100045",
                        "tv_remote_pickup_s100005",
                        "jam_pickup_s100030",
                    ]
                    before, after = Round(loop, VERSIONS[k][0], labs[k]), Round(loop, VERSIONS[k + 1][0], labs[k])
                lines = [(cards[k][0], 40, FG), (cards[k][1], 24, (255, 210, 120)), (cards[k][2], 22, DIM),
                         ("半透明紫球：指令给的手腕位置", 20, DIM)]  # fmt: skip
                step_video(sc, before, after, (VERSIONS[k][1] + "的版本", VERSIONS[k + 1][1] + "的版本"), lines,
                           a.out / f"step_{k:02d}.mp4")  # fmt: skip
            print("done", k, flush=True)
    if "infeasible" in only:
        infeasible(sc, loop, a.out)
    if "overview" in only:
        overview(sc, loop, a.out, ["canister_pickup_s100053", "canister_pickup_s100045", "tv_remote_pickup_s100005",
                                   "jam_pickup_s100030", "jam_pickup_s100055"])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

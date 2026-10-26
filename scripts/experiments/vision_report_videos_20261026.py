"""Short MuJoCo videos of vision RSI rounds for the main line's progress report (its vision section, asked for
2026-10-26), re-rendered from the rounds' saved traces (`--npz` records) with the full scene model's visual meshes:

    wrong_instance   truth vs L3d on one start: SAM 3's chosen region overlaps the small vase by 0.44 (it also
                     returned one at 0.88), and the jaw closes above the neighbouring medium vase (design doc section 16)
    pose_offset_4cm  truth vs a 4 cm object-pose offset (L1, section 13) on one start: the jaw closes beside the mug;
                     a blue ghost marks where the offset view put it (the offset is redrawn from the round's seed, as
                     scene.view.PerturbedView draws it)
    l2_finds         one L2 success (section 15): the head camera's view, the target's mask in green, next to the
                     third-person shot

Markers: green arrow = the object to pick (true position); red dot = the jaw centre once the jaw closes. Only the
target object's pose is in a trace, so the other objects stay at their start poses. Each video is encoded under 490 KB
(the repo's 500 KB rule) by raising the x264 CRF; captions.json maps file -> caption, start, conditions.

    MUJOCO_GL=egl PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/vision_report_videos_20261026.py --out <dir> [--only wrong_instance] [--preview]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "rsi"), str(REPO / "src")]
from failure_class import FK, SIDES  # noqa: E402

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

SCENE = REPO / "outputs/scene_discovery/adt_apartment_mars/scene_locomani.xml"  # the visual meshes the level uses
RUNS = REPO / "outputs/vision_rsi_20261023"
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
OFF = np.array([0.1515, 0.0, 0.0])  # wrist frame -> jaw centre (backends/scalebfm/runtime.py)
FINGERS = ("left_hand_gripper_1_joint", "left_hand_gripper_2_joint", "right_hand_gripper_1_joint",
           "right_hand_gripper_2_joint")  # fmt: skip
FPS = 15
MAX_BYTES = 490_000
GREEN, RED, BLUE = (0.1, 0.85, 0.3, 0.9), (0.95, 0.15, 0.1, 1.0), (0.2, 0.45, 1.0, 0.75)


class Round:
    """One saved round: its trace (npz), its record (json) and the target's read log."""

    def __init__(self, cond: str, start: str):
        self.cond, self.start = cond, start
        self.z = dict(np.load(RUNS / cond / f"{start}.npz"))
        self.rec = json.loads((RUNS / cond / f"{start}.json").read_text())
        self.obj = next(k for k in self.z if k.startswith("obj_"))
        self.body = self.obj[4:]
        self.key = self.rec["task"].removeprefix("pick_")
        self.t = self.z["t"].astype(float)
        names, p = list(self.z["primitives"]), self.z["primitive"]
        self.first = {n: float(self.t[np.argmax(p == names.index(n))]) for n in names}
        self.side = SIDES.index(self.rec["events"].get("grip_side") or "right")
        stats = (self.rec.get("perception_stats") or {}).get(self.key) or {}
        self.seen = next((e for e in stats.get("log", []) if e["src"] == "seen"), None)

    def at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


class Scene:
    def __init__(self):
        m = self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(m)
        pelvis = m.body("pelvis").id
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.root = int(m.jnt_qposadr[next(j for j in free if m.jnt_bodyid[j] == pelvis)])
        self.qj = np.array([m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.qf = np.array([m.jnt_qposadr[m.joint(n).id] for n in FINGERS])
        self.fk = FK(m)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        # the robot's body drawn see-through in the third-person shots, the hands (wrist links and below) solid, so a
        # grasp behind the torso stays visible
        hand = np.zeros(m.nbody, bool)
        for b in range(m.nbody):
            k = b
            while k > 0 and not m.body(k).name.endswith("wrist_yaw_link"):
                k = int(m.body_parentid[k])
            hand[b] = k > 0
        robot = m.body_rootid[m.geom_bodyid] == m.body_rootid[pelvis]
        self.ghostly = robot & ~hand[m.geom_bodyid]

    def renderer(self, w: int, h: int) -> mujoco.Renderer:
        if (w, h) not in self._renderers:
            self._renderers[(w, h)] = mujoco.Renderer(self.m, h, w, max_geom=self.m.ngeom + 2000)
        return self._renderers[(w, h)]

    def pose(self, r: Round, i: int) -> None:
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        d.qpos[self.root : self.root + 7] = r.z["robot"][i]
        d.qpos[self.qj], d.qpos[self.qf] = r.z["joints"][i], r.z["fingers"][i]
        a = int(m.jnt_qposadr[m.body(r.body).jntadr[0]])
        d.qpos[a : a + 7] = r.z[r.obj][i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)

    def jaw(self, r: Round, i: int) -> np.ndarray:
        pw, Rw = self.fk(r.z["robot"][i], r.z["joints"][i])[r.side]
        return pw + Rw @ OFF

    def close(self) -> None:
        for rr in self._renderers.values():
            rr.close()


def sphere(scn, pos, radius: float, rgba) -> None:
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                            np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))  # fmt: skip
        scn.ngeom += 1


def arrow(scn, a, b, width: float, rgba) -> None:
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, np.float32))  # fmt: skip
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1


def camera(lookat, toward, dist: float, elev: float) -> mujoco.MjvCamera:
    """A free camera `dist` from `lookat`, placed along the horizontal unit vector `toward`, looking back at it."""
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.azimuth = float(np.degrees(np.arctan2(-toward[1], -toward[0])))
    c.elevation, c.distance = elev, dist
    return c


def shot(scene: Scene, r: Round, side_deg: float, dist: float, elev: float) -> mujoco.MjvCamera:
    """The grasp from beside the robot: the camera on the robot's side of the object, `side_deg` off the line from the
    object to where the robot stood when the jaw closed."""
    i = r.at(r.first.get("grip", r.t[-1]))
    obj = r.z[r.obj][i, :3]
    v = r.z["robot"][i, :2] - obj[:2]
    a = np.arctan2(v[1], v[0]) + np.radians(side_deg)
    return camera(obj + np.array([0, 0, 0.05]), np.array([np.cos(a), np.sin(a)]), dist, elev)


def third_person(scene: Scene, r: Round, i: int, cam, w: int, h: int, ghost=None, jaw_after=None) -> np.ndarray:
    scene.pose(r, i)
    rr = scene.renderer(w, h)
    rr.update_scene(scene.d, camera=cam)
    for k in range(rr.scene.ngeom):
        g = rr.scene.geoms[k]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and scene.ghostly[g.objid]:
            g.rgba[3] = 0.3
    obj = scene.d.xpos[scene.m.body(r.body).id]
    arrow(rr.scene, obj + [0, 0, 0.32], obj + [0, 0, 0.14], 0.012, GREEN)
    if ghost is not None:
        sphere(rr.scene, obj + ghost, 0.025, BLUE)
    if jaw_after is not None and r.t[i] >= jaw_after:
        sphere(rr.scene, scene.jaw(r, i), 0.015, RED)
    return rr.render().copy()


def head_view(scene: Scene, r: Round, i: int, w: int, h: int) -> np.ndarray:
    """The robot's head camera (MuJoCo render), the target's pixels tinted green: the mask L2 fits."""
    scene.pose(r, i)
    rr = scene.renderer(w, h)
    rr.update_scene(scene.d, camera="head_camera")
    rgb = rr.render().copy()
    rr.enable_segmentation_rendering()
    rr.update_scene(scene.d, camera="head_camera")
    seg = rr.render()
    rr.disable_segmentation_rendering()
    ids, kinds = seg[..., 0], seg[..., 1]
    mask = (kinds == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
    mask[mask] = scene.m.geom_bodyid[ids[mask]] == scene.m.body(r.body).id
    rgb[mask] = (0.35 * rgb[mask] + 0.65 * np.array([40, 230, 90])).astype(np.uint8)
    return rgb


class Text:
    def __init__(self):
        self.big, self.small = ImageFont.truetype(FONT, 19), ImageFont.truetype(FONT, 16)

    def box(self, img: Image.Image, xy, lines: list[str], big=False, fill=(255, 255, 255)) -> None:
        if not lines:
            return
        font = self.big if big else self.small
        dr = ImageDraw.Draw(img, "RGBA")
        x, y = xy
        for line in lines:
            l, t, rgt, b = dr.textbbox((x, y), line, font=font)
            dr.rectangle((l - 4, t - 3, rgt + 4, b + 3), fill=(0, 0, 0, 150))
            dr.text((x, y), line, font=font, fill=fill)
            y = b + 6


def compose(panels: list[np.ndarray], tops: list[list[str]], notes: list[list[str]], caption: str,
            text: Text) -> np.ndarray:  # fmt: skip
    h = max(p.shape[0] for p in panels)
    row = np.hstack([np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0))) for p in panels])
    img = Image.fromarray(np.vstack([row, np.full((44 + h % 2, row.shape[1], 3), 24, np.uint8)]))
    x = 0
    for p, top, note in zip(panels, tops, notes, strict=True):
        text.box(img, (x + 10, 8), top, big=True)
        text.box(img, (x + 10, p.shape[0] - 30 * len(note) - 4), note)
        x += p.shape[1]
    ImageDraw.Draw(img).text((12, h + 11), caption, font=text.small, fill=(235, 235, 235))
    return np.asarray(img)


def encode(frames: list[np.ndarray], path: Path) -> tuple[int, int]:
    """H.264 at the lowest CRF (from 26) that keeps the file under MAX_BYTES: (crf, bytes)."""
    h, w = frames[0].shape[:2]
    for crf in range(26, 46, 2):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]  # fmt: skip
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {path}")
        if path.stat().st_size <= MAX_BYTES:
            return crf, path.stat().st_size
    raise SystemExit(f"{path.name}: still {path.stat().st_size} bytes at CRF 44")


def outcome(r: Round) -> str:
    return "结果：抓起" if r.rec["ok"] else "结果：没抓住"


def times(r: Round, t0: float, n: int) -> list[int]:
    return [r.at(t0 + k / FPS) for k in range(n)]


def pair(scene: Scene, text: Text, a: Round, b: Round, labels, notes_of, caption: str, cam, ghost_b=None,
         lead: float = 3.0, preview: Path | None = None) -> list[np.ndarray]:  # fmt: skip
    """Two rounds of one start side by side, each from `lead` s before its reach to its end (the shorter one holds
    its last frame)."""
    w, h = 480, 360
    t0 = [r.first["reach"] - lead for r in (a, b)]
    n = int(np.ceil(max(r.t[-1] - s for r, s in zip((a, b), t0, strict=True)) * FPS)) + 1
    if preview is not None:
        n_grip = [int((r.first["grip"] - s) * FPS) + 3 for r, s in zip((a, b), t0, strict=True)]
        idx = [[r.at(s + n_grip[k] / FPS)] for k, (r, s) in enumerate(zip((a, b), t0, strict=True))]
    else:
        idx = [times(r, s, n) for r, s in zip((a, b), t0, strict=True)]
    frames = []
    for k in range(len(idx[0])):
        ia, ib = idx[0][k], idx[1][k]
        pa = third_person(scene, a, ia, cam, w, h, jaw_after=a.first["grip"])
        pb = third_person(scene, b, ib, cam, w, h, ghost=ghost_b, jaw_after=b.first["grip"])
        tops = [[labels[0], f"t = {a.t[ia]:.1f} s"], [labels[1], f"t = {b.t[ib]:.1f} s"]]
        frames.append(compose([pa, pb], tops, [notes_of(a, ia), notes_of(b, ib)], caption, text))
    return frames


def wrong_instance(scene: Scene, text: Text, preview) -> tuple[list[np.ndarray], dict]:
    start = "pick_VaseSmall_2_s610037"
    a, b = Round("l0/view", start), Round("l3/l3d", start)
    jaw = {r.cond: 100 * float(np.linalg.norm(scene.jaw(r, r.at(r.first["grip"])) - r.z[r.obj][r.at(r.first["grip"]), :3]))
           for r in (a, b)}  # fmt: skip

    def notes(r: Round, i: int) -> list[str]:
        t, out = r.t[i], []
        if r.seen and t >= r.seen["t"]:
            s = r.seen
            out += [f"SAM 3 选的区域和小花瓶只重合 {s['iou']:.2f}", f"（重合 {s['best_iou']:.2f} 的那块没选上）",
                    f"位置估计偏了 {s['err_cm']:.1f} cm"]  # fmt: skip
        if t >= r.first["grip"]:
            out.append(f"合爪：两指中心离小花瓶 {jaw[r.cond]:.1f} cm")
        if t >= r.t[-1] - 0.5:
            out.append(outcome(r))
        return out

    caption = "同一起点：左边用仿真器的真值，抓起小花瓶；右边 SAM 3 选错了区域，手伸到旁边中号花瓶的上方合爪，没抓到"
    frames = pair(scene, text, a, b, ("真值（仿真器给物体位姿）", "L3d：SAM 3 掩码 + Unreal 深度"), notes, caption,
                  shot(scene, b, -45.0, 1.3, -40.0), preview=preview)  # fmt: skip
    meta = dict(start=start, conditions=["l0/view", "l3/l3d"], jaw_to_object_cm={k: round(v, 1) for k, v in jaw.items()},
                caption="同一起点上，真值能抓起小花瓶；换成 SAM 3 掩码后它选错了区域（和小花瓶只重合 "
                f"{b.seen['iou']:.2f}，它也给出过重合 {b.seen['best_iou']:.2f} 的那块），位置估计偏 {b.seen['err_cm']:.1f} cm，"
                f"手伸到旁边中号花瓶的上方合爪，两指中心离小花瓶 {jaw[b.cond]:.0f} cm。")  # fmt: skip
    return frames, meta


def pose_offset(scene: Scene, text: Text, preview) -> tuple[list[np.ndarray], dict]:
    start = "pick_Mug_2_s610013"
    a, b = Round("l0/view", start), Round("l1/bias_040", start)
    seed = int(start.rsplit("_s", 1)[1])
    rng = np.random.default_rng([seed, 0, zlib.crc32(b.key.encode())])  # scene.view.PerturbedView.errors, seed 0
    d = rng.normal(size=3)
    ghost = 0.04 * d / np.linalg.norm(d)
    # the jaw's opening once it has closed (the round's events): stopped by the mug's wall, or shut on nothing
    gap = {r.cond: float(r.rec["events"]["opening_at_grip_cm"]) for r in (a, b)}

    def notes(r: Round, i: int) -> list[str]:
        t, out = r.t[i], []
        if r is b:
            out.append("蓝球：机器人以为的杯子位置（偏 4 cm）")
        if t >= r.first["grip"] + 0.6:
            out.append(f"合爪后开口 {gap[r.cond]:.1f} cm：" + ("夹住了杯壁" if gap[r.cond] > 0.5 else "两指之间是空的"))
        if t >= r.t[-1] - 0.5:
            out.append(outcome(r))
        return out

    caption = "同一起点：左边用真值，抓起杯子；右边物体位置偏 4 cm，手指在杯子旁边合到底，没夹住"
    frames = pair(scene, text, a, b, ("真值", "L1：物体位置偏 4 cm"), notes, caption,
                  shot(scene, b, 45.0, 0.9, -40.0), ghost_b=ghost, preview=preview)  # fmt: skip
    meta = dict(start=start, conditions=["l0/view", "l1/bias_040"], opening_after_close_cm={k: round(v, 2) for k, v in gap.items()},
                caption="同一起点上，只把杯子的位置偏 4 cm 告诉机器人，它就在杯子旁边把手指合到底（开口 "
                f"{gap[b.cond]:.1f} cm，杯子不在两指之间）；用真值时同一个起点能夹住杯壁抓起。")  # fmt: skip
    return frames, meta


def l2_finds(scene: Scene, text: Text, preview) -> tuple[list[np.ndarray], dict]:
    start = "pick_VaseSmall_2_s610037"
    r = Round("l2b/l2", start)
    seen = r.seen
    t0 = seen["t"] - 6.0
    n = int(np.ceil((r.t[-1] - t0) * FPS)) + 1
    idx = [r.at(seen["t"] + 0.2)] if preview is not None else times(r, t0, n)
    cam = shot(scene, r, -45.0, 1.3, -40.0)
    caption = "L2：只用机器人自己头相机的深度和掩码（绿色），第一次看到就把小花瓶定准，抓起"
    frames = []
    for i in idx:
        left = third_person(scene, r, i, cam, 540, 405, jaw_after=r.first["grip"])
        right = head_view(scene, r, i, 420, 315)
        t = r.t[i]
        note = (["还没看到小花瓶：按地图记忆（偏 5 cm）走过去"] if t < seen["t"]
                else [f"t = {seen['t']:.1f} s 头相机看到小花瓶，位置误差 {seen['err_cm']:.2f} cm"])  # fmt: skip
        if t >= r.t[-1] - 0.5:
            note.append(outcome(r))
        frames.append(compose([left, right], [["第三人称（L2 这一局）", f"t = {t:.1f} s"], ["头相机（绿色 = 小花瓶的掩码）"]],
                              [note, []], caption, text))  # fmt: skip
    meta = dict(start=start, conditions=["l2b/l2"],
                caption=f"同一起点的 L2：机器人只用自己头相机的深度和掩码（绿色），走到架子前第一次看到小花瓶就把它定到 "
                f"{seen['err_cm']:.2f} cm 以内，抓起；L3d 输在同一个起点，是因为 SAM 3 选错了区域。")  # fmt: skip
    return frames, meta


VIDEOS = {"wrong_instance": wrong_instance, "pose_offset_4cm": pose_offset, "l2_finds": l2_finds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=sorted(VIDEOS), action="append")
    ap.add_argument("--preview", action="store_true", help="one PNG per video at the grasp, no mp4")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    scene, text = Scene(), Text()
    captions_path = a.out / "captions.json"
    captions = json.loads(captions_path.read_text()) if captions_path.exists() else {}
    try:
        for name in a.only or VIDEOS:
            frames, meta = VIDEOS[name](scene, text, a.out if a.preview else None)
            if a.preview:
                Image.fromarray(frames[0]).save(a.out / f"preview_{name}.png")
                print(name, "preview", frames[0].shape, flush=True)
                continue
            crf, size = encode(frames, a.out / f"{name}.mp4")
            captions[f"{name}.mp4"] = dict(meta, seconds=round(len(frames) / FPS, 1), crf=crf, bytes=size)
            print(name, len(frames), "frames", size, "bytes, crf", crf, flush=True)
    finally:
        scene.close()
    if not a.preview:
        captions_path.write_text(json.dumps(captions, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

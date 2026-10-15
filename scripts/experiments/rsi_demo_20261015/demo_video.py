"""TODO 26 (user 2026-10-15): shared pieces of the RSI demo videos -- per-round smoothness metrics, a full-round
side-by-side renderer for two round files of the same start, title cards, and the <= 500 KB previews AGENTS.md requires.

Round files are the controller-compare format (`warehouse_states.py` / `bfm_round.py`): 50 Hz `t`, `qpos`, `pelvis`,
`joints`, the object pose (`box` or `bottle`), `cmd_wrist` (the commanded wrists; the ghosts) and a JSON with `events`.
Both sides start from the same start state at t = 0, so the clips are aligned on the round's own clock and play the whole
round; the side that ends first holds its last frame.

Smoothness (`smoothness`), all from the measured motion, one value per round:
* torso_w_p95 -- p95 of the torso link's angular speed over the round, rad/s (l2_measure's "ego": the head camera);
* hand_jerk -- median magnitude of the jaw centres' jerk from the reach to the end, m/s^3 (Savitzky-Golay on 50 Hz);
* object_w_p95 -- p95 of the object's angular speed from the lift to the end, rad/s (only when it was lifted >= 3 cm).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CC = REPO / "scripts" / "experiments" / "controller_compare_20261013"
sys.path[:0] = [str(CC), str(REPO / "src")]

import warehouse_video as wv  # noqa: E402

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # CJK captions
FPS, DT = 25, 0.02
OFF = wv.OFF  # wrist frame -> jaw centre (dex1)
GREEN, RED, GREY = (90, 220, 120), (255, 110, 100), (220, 220, 220)


def load(npz: Path) -> tuple[dict, dict]:
    return dict(np.load(npz)), json.loads(npz.with_suffix(".json").read_text())


JAW_OPEN, JAW_SHUT = 0.0245, -0.0015  # Dex1 slide joint: open, and where it rests on a bottle (ScaleBFM rounds' qpos)


def full_qpos(z: dict, f: dict, m: mujoco.MjModel) -> np.ndarray:
    """The whole scene state per tick. Round files without `qpos` (SONIC's bottle_states.py files: pelvis, joints, the
    bottle) are rebuilt on the scene model; their right jaw, never recorded, is drawn from the events -- open until
    `close`, shut over the next 0.5 s (display only; no metric reads it)."""
    if "qpos" in z:
        return z["qpos"]
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    q = np.tile(m.qpos0, (len(z["t"]), 1))
    q[:, :7] = z["pelvis"]
    q[:, [m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES]] = z["joints"]
    free = [j for j in range(m.njnt) if m.jnt_type[j] == 0 and m.jnt_bodyid[j] != m.body("pelvis").id]
    q[:, m.jnt_qposadr[free[0]] : m.jnt_qposadr[free[0]] + 7] = z[object_key(z)]
    jaw = np.full(len(z["t"]), JAW_OPEN)
    jaw_rec = z.get("jaw")
    if jaw_rec is not None and np.isfinite(jaw_rec).all():
        jaw = jaw_rec
    elif "close" in f["events"]:
        u = np.clip((np.asarray(z["t"]) - f["events"]["close"]) / 0.5, 0, 1)
        jaw = JAW_OPEN + u * (JAW_SHUT - JAW_OPEN)
    for side, val in (("left", np.full(len(jaw), JAW_OPEN)), ("right", jaw)):
        for k in (1, 2):
            q[:, m.jnt_qposadr[m.joint(f"{side}_hand_gripper_{k}_joint").id]] = val
    return q


def object_key(z: dict) -> str:
    return "box" if "box" in z else "bottle"


class Kin:
    """Forward kinematics of a round's qpos on its scene model."""

    def __init__(self, m: mujoco.MjModel):
        self.m, self.d = m, mujoco.MjData(m)
        self.torso = m.body("torso_link").id
        self.wrists = [m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]

    def jaws_and_torso(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jaws, rots = np.empty((len(qpos), 2, 3)), np.empty((len(qpos), 3, 3))
        for i, q in enumerate(qpos):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            for k, b in enumerate(self.wrists):
                jaws[i, k] = self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ OFF
            rots[i] = self.d.xmat[self.torso].reshape(3, 3)
        return jaws, rots


def _ang_speed(rots: np.ndarray, t: np.ndarray) -> np.ndarray:
    c = np.clip((np.einsum("nij,nij->n", rots[1:], rots[:-1]) - 1) / 2, -1, 1)
    return np.arccos(c) / np.maximum(np.diff(t), 1e-6)


def _quat_rots(q: np.ndarray) -> np.ndarray:
    out = np.empty((len(q), 9))
    for i, x in enumerate(q):
        mujoco.mju_quat2Mat(out[i], x)
    return out.reshape(-1, 3, 3)


def smoothness(z: dict, f: dict, kin: Kin, starts=("reach", "handoff"), torso_from: str | None = None) -> dict:
    """`starts`: the first of these events that the round has opens the hand-jerk window; `torso_from`: the event the
    torso window opens at (None: the whole round -- a SONIC bottle trace misses its first seconds, so bottle uses ready)."""
    t = np.asarray(z["t"], float)
    jaws, torso = kin.jaws_and_torso(full_qpos(z, f, kin.m))
    w = _ang_speed(torso, t)
    ev = f["events"]
    if torso_from is not None and torso_from in ev:
        w = w[t[1:] >= ev[torso_from]]
    out = dict(torso_w_p95=float(np.percentile(w, 95)) if len(w) else float("nan"))
    t0 = next((ev[e] for e in starts if e in ev), None)
    if t0 is not None:
        tu = np.arange(t0, t[-1], DT)
        if len(tu) > 15:
            p = np.stack([np.interp(tu, t, jaws[:, k, a]) for k in range(2) for a in range(3)], axis=1)
            j = savgol_filter(p, 11, 3, deriv=3, delta=DT, axis=0).reshape(len(tu), 2, 3)
            out["hand_jerk"] = float(np.median(np.linalg.norm(j, axis=2)))
    obj = z[object_key(z)]
    rise = obj[:, 2] - obj[0, 2]
    if "lift" in ev and np.max(rise) >= 0.03:
        sel = t >= ev["lift"]
        if sel.sum() > 5:
            out["object_w_p95"] = float(np.percentile(_ang_speed(_quat_rots(obj[sel, 3:7]), t[sel]), 95))
    return out


def scene_model(scene_dir: Path) -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(str(scene_dir / "scene_locomani.xml"))
    m.vis.global_.offwidth = max(2 * wv.W, m.vis.global_.offwidth)
    m.vis.global_.offheight = max(wv.H, m.vis.global_.offheight)
    for g in range(m.ngeom):  # the room's dressing meshes block the view of the platform (warehouse_video.py)
        if m.geom(g).name.startswith("vis_room"):
            m.geom_group[g] = 5
    if "platform_col" in [m.geom(g).name for g in range(m.ngeom)]:
        m.geom_rgba[m.geom("platform_col").id] = (0.45, 0.45, 0.5, 1.0)
    return m


def _font(size: int):
    return ImageFont.truetype(FONT, size)


def caption_strip(lines: list[tuple[str, tuple]], width: int, height: int = 64) -> np.ndarray:
    img = Image.new("RGB", (width, height), (20, 20, 24))
    d = ImageDraw.Draw(img)
    y = 6
    for text, colour in lines:
        d.text((10, y), text, font=_font(20), fill=colour)
        y += 28
    return np.asarray(img)


def card(lines: list[tuple[str, int, tuple]], size=(2 * wv.W, wv.H + 64)) -> np.ndarray:
    """A title card: (text, font size, colour) lines, centred."""
    img = Image.new("RGB", size, (16, 18, 24))
    d = ImageDraw.Draw(img)
    total = sum(s + 14 for _, s, _ in lines)
    y = (size[1] - total) // 2
    for text, s, colour in lines:
        w = d.textlength(text, font=_font(s))
        d.text(((size[0] - w) / 2, y), text, font=_font(s), fill=colour)
        y += s + 14
    return np.asarray(img)


class Writer:
    """Raw RGB frames -> an H.264 mp4 at FPS (crf 20; previews come from `preview`)."""

    def __init__(self, out: Path, width: int, height: int):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def write(self, frame: np.ndarray, times: int = 1) -> None:
        b = np.ascontiguousarray(frame).tobytes()
        for _ in range(times):
            self.p.stdin.write(b)

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg failed")


def view(m: mujoco.MjModel, f: dict, z: dict):
    axis = (f.get("edge") or {}).get("axis") or [0.0, -1.0]  # a SONIC bottle round that never planned its walk
    return wv.View(m, np.asarray(axis, float), z[object_key(z)][0, :2])


def render_pair(m, left: Path, right: Path, out: Path, captions: list[list[tuple[str, tuple]]],
                title: np.ndarray | None = None, title_s: float = 3.0, tail_s: float = 1.0,
                align: str | None = None, pre_s: float = 2.0) -> Path:  # fmt: skip
    """The whole round of two round files from one start, side by side, with caption strips. Clocks are the rounds' own
    (both start from the same state) or, with `align`, shifted so that event happens at the same moment (the clip then
    starts `pre_s` before it)."""
    sides = []
    for p in (left, right):
        z, f = load(p)
        z = dict(z, qpos=full_qpos(z, f, m))
        off = float(f["events"][align]) - pre_s if align else 0.0
        sides.append((dict(z, t=np.asarray(z["t"]) - off), view(m, f, z)))
    t0 = 0.0
    t_end = max(float(z["t"][-1]) for z, _ in sides) + tail_s
    strips = np.hstack([caption_strip(c, wv.W) for c in captions])
    w = Writer(out, 2 * wv.W, wv.H + strips.shape[0])
    if title is not None:
        w.write(title, int(title_s * FPS))
    for s in np.arange(t0, t_end, 1.0 / FPS):
        tiles = []
        for z, v in sides:
            i = int(np.clip(np.searchsorted(z["t"], s), 0, len(z["t"]) - 1))
            tiles.append(v.frame(z["qpos"][i], wv.ghosts_at(z, i)))
        frame = np.vstack([strips, np.hstack(tiles)])
        clock = Image.fromarray(frame)
        ImageDraw.Draw(clock).text((2 * wv.W - 110, strips.shape[0] + 8), f"t = {s:5.1f} s", font=_font(18), fill=GREY)
        w.write(np.asarray(clock))
    w.close()
    return out


def still(out: Path, frame: np.ndarray, seconds: float) -> Path:
    w = Writer(out, frame.shape[1], frame.shape[0])
    w.write(frame, int(seconds * FPS))
    w.close()
    return out


def concat(parts: list[Path], out: Path) -> Path:
    """Same-size, same-codec clips joined end to end."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("".join(f"file '{p.resolve()}'\n" for p in parts))
        lst = fh.name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", str(out)],
                   check=True)  # fmt: skip
    return out


def preview(src: Path, stem: Path) -> list[Path]:
    """<= 500 KB previews of `src` (split into parts when needed), checked by locomani.pipeline.video_preview."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "locomani.pipeline.video_preview", str(src), str(stem)], check=True,
                   cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"})  # fmt: skip
    return sorted(stem.parent.glob(stem.name + "*.mp4"))

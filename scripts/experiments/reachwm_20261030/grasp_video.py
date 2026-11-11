"""Grasp pilot videos: one placement, the inverted commands solid, the direct commands as an orange ghost (its own box a
translucent orange box), from where the walk ends to the end of the hold; the whole cabinet scene, a close-up of the
box, a strip with each run's phase and result. From grasp_pilot.py's recorded qpos (every 2 motor ticks = 25 fps).

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/reachwm_20261030/grasp_video.py 26 38 19
Writes outputs/reachwm_20261030/grasp_pilot/videos/ (src/, preview/ <= 500 KB).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_pilot as GP  # noqa: E402
import report_video as RV  # noqa: E402

RD = RV.RD
OUTV = GP.OUT / "videos"
PHASE = {
    "walk": "走近",
    "pregrasp": "伸到柜子前",
    "entry": "伸进去",
    "grip": "合爪",
    "lift": "抬起",
    "hold": "保持 3 秒",
}
LEAD = 25  # frames before the pre-grasp starts


def results(pid: int) -> dict:
    out = {}
    for f in sorted(GP.OUT.glob("h_w*.json")):
        for r in json.load(open(f)):
            if r["pid"] == pid:
                out[r["method"]] = r
    return out


def verdict(r: dict) -> str:
    if r["success"]:
        return "抓起来了"
    if r.get("pushed_cm", 0) > 2:
        return f"没抓住，箱子被推走 {r['pushed_cm']:.0f} cm"
    if r.get("gripped"):
        return "夹住了，但没抬起来"
    return "没抓住"


def phase_at(marks, k: int) -> str:
    name = marks[0][0]
    for nm, i in marks:
        if k >= i:
            name = nm
    return PHASE.get(name, name)


def render(pid: int) -> None:
    R = results(pid)
    rd, ri = R["direct"], R["inv"]
    tag = lambda m: f"h_{pid}_{m}_none.npy"  # noqa: E731
    qd, qi = np.load(GP.OUT / "q" / tag("direct")), np.load(GP.OUT / "q" / tag("inv"))
    env = GP.make_env()
    m = env.sim.m
    P = RV.Painter(m)
    pl = {p["pid"]: p for p in GP.placements()}[pid]
    jadr = m.jnt_qposadr[m.body_jntadr[m.body(pl["body"]).id]]
    goal = np.array(ri["entry"]["goal_w"])
    sd = 1.0 if pl["side"] == "right" else -1.0
    az = 48.0 * sd  # from the reaching hand's side, outside the cabinet front
    cam_m = P.cam(goal + np.array([-0.35, 0.0, -0.1]), 1.55, az, -18)
    cam_c = P.cam(goal, 0.42, az, -24)
    k0 = max(0, min(ri["marks"][1][1], rd["marks"][1][1]) - LEAD)
    n = max(len(qd), len(qi))
    font, small, tiny = (ImageFont.truetype(RV.FONT, s) for s in (17, 14, 12))
    hw = pl["width"] / 2
    shelf = GP.SHELVES[pl["shelf"]]
    title = f"柜子里 {shelf:.2f} m 高的隔板上、离柜门 {pl['depth'] * 100:.0f} cm 的 {pl['width'] * 100:.0f} cm 宽箱子：直接发命令，和用自身模型反推命令"

    def draw(k: int, phase: str) -> np.ndarray:
        imgs = []
        kd, ki = min(k, len(qd) - 1), min(k, len(qi) - 1)
        for which, (rr, cam, W_, H_) in enumerate(((P.rm, cam_m, RV.WM_, RV.HM), (P.rc, cam_c, RV.WC, RV.HC))):
            d = P.ds[0]
            d.qpos[:] = qi[ki]
            mujoco.mj_forward(m, d)
            rr.update_scene(d, camera=cam)
            scn = rr.scene
            if which == 1:  # x-ray the robot in the close-up so the fingers and the box stay visible
                for gi in range(scn.ngeom):
                    g = scn.geoms[gi]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and P.robot[m.geom_bodyid[g.objid]]:
                        g.rgba[3] = 0.35
            if phase != "plan":
                P.ghost(scn, qd[kd], (*RV.ORANGE, 0.12 if which else 0.26))
                bp, bq = qd[kd][jadr : jadr + 3], qd[kd][jadr + 3 : jadr + 7]
                Rb = np.zeros(9)
                mujoco.mju_quat2Mat(Rb, bq.astype(np.float64))
                if scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([hw, 0.03, GP.BOX_HALF_H]),
                                        bp.astype(np.float64), Rb, np.array([*RV.ORANGE, 0.45], np.float32))  # fmt: skip
                    scn.ngeom += 1
            P.sphere(scn, goal, 0.007, (*RV.GREEN, 1.0))
            img = Image.fromarray(rr.render())
            dr = ImageDraw.Draw(img)
            dr.text(
                (8, 6),
                "目标附近放大" if which else "全景",
                font=small,
                fill=(20, 20, 20),
                stroke_width=2,
                stroke_fill=(255, 255, 255),
            )
            imgs.append(np.asarray(img))
        canvas = Image.new("RGB", (RV.W, RV.TB + RV.HM + 120), (248, 248, 246))
        canvas.paste(Image.fromarray(np.concatenate(imgs, 1)), (0, RV.TB))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle([0, 0, RV.W, RV.TB], fill=(24, 28, 34))
        dr.text((10, 9), title, font=small, fill=(240, 240, 240))
        y0 = RV.TB + RV.HM
        for j, (r, nm, col, kk) in enumerate(
            ((ri, "实体：用自身模型反推的命令", RV.CYAN8, ki), (rd, "橙色影子：直接把抓取点当命令", RV.ORANGE8, kd))
        ):
            y = y0 + 12 + 46 * j
            dr.rectangle([10, y + 3, 24, y + 15], fill=col)
            ph = "还没动" if phase == "plan" else phase_at(r["marks"], kk)
            dr.text((30, y), f"{nm}　现在：{ph}", font=small, fill=(25, 25, 25))
            if phase == "end":
                a = r.get("at_grip_cm", [0, 0, 0])
                dr.text((30, y + 20), f"合爪时手爪中心离抓取点 {math.hypot(*a):.1f} cm，其中左右方向 {abs(a[1]):.1f} cm；结果：{verdict(r)}",
                        font=tiny, fill=(80, 80, 80))  # fmt: skip
        dr.text(
            (10, y0 + 100),
            "绿点是抓取点；两次都从完全相同的位置走过来，都不看手；夹爪张开 9.5 cm",
            font=tiny,
            fill=(120, 120, 120),
        )
        return np.asarray(canvas)

    frames = [draw(k0, "plan")] * 25
    for k in range(k0, n):
        frames.append(draw(k, "run"))
    end = draw(n - 1, "end")
    frames += [end] * 50
    (OUTV / "src").mkdir(parents=True, exist_ok=True)
    (OUTV / "preview").mkdir(parents=True, exist_ok=True)
    src = OUTV / "src" / f"pilot_{pid}.mp4"
    RD.write(frames, src, fps=RV.FPS)
    nb = RD.preview(src, OUTV / "preview" / f"pilot_{pid}.mp4")
    Image.fromarray(end).save(OUTV / "preview" / f"pilot_{pid}.jpg", quality=86)
    Image.fromarray(frames[len(frames) // 2]).save(OUTV / "preview" / f"pilot_{pid}_mid.jpg", quality=86)
    print(f"pilot_{pid}: {len(frames)} frames, preview {nb} bytes", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        render(int(a))

"""Comparison videos for grasp_exp.py: one setup (placement, start), one method solid, another as an orange ghost (its own
box a translucent orange box); the cabinet scene, a close-up, a strip with each run's phase, whether the head camera
sees the box, and the result. From the recorded qpos (every 2 motor ticks = 25 fps).

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/reachwm_20261030/grasp_exp_video.py 8:0:inv:direct 32:1:inv:direct_prop
Writes outputs/reachwm_20261030/grasp_exp/videos/ (src/, preview/ <= 500 KB).
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
import grasp_exp as GX  # noqa: E402
import grasp_pilot as GP  # noqa: E402
import grasp_video as GV  # noqa: E402
import report_video as RV  # noqa: E402

RD = RV.RD
OUTV = GX.OUT / "videos"
NAMES = {"direct": "直接发命令", "direct_vis": "直接发命令，加视觉伺服", "direct_prop": "直接发命令，加关节读数伺服",
         "inv": "自身模型反推，闭眼", "inv_prop": "自身模型反推，加关节读数伺服"}  # fmt: skip


def results() -> dict:
    out = {}
    for f in sorted(GX.OUT.glob("h_w*.json")):
        for r in json.load(open(f)):
            out[(r["pid"], r["start"], r["method"])] = r
    return out


def verdict(r: dict) -> str:
    if r.get("declined"):
        return "事先判断会撞到柜子，没有去抓"
    return GV.verdict(r)


def render(env, R: dict, pid: int, si: int, solid: str, ghost: str) -> Path:
    rs, rg = R[(pid, si, solid)], R[(pid, si, ghost)]
    qs, qg = (np.load(GX.OUT / "q" / f"{pid}_{si}_{mm}.npy") for mm in (solid, ghost))
    m = env.sim.m
    P = RV.Painter(m)
    pl = {p["pid"]: p for p in GX.placements()}[pid]
    jadr = m.jnt_qposadr[m.body_jntadr[m.body(pl["body"]).id]]
    goal = np.array(pl["box"]) + np.array([0.01, 0.0, 0.0])
    sd = 1.0 if pl["side"] == "right" else -1.0
    az = 48.0 * sd
    cam_m = P.cam(goal + np.array([-0.35, 0.0, -0.1]), 1.55, az, -18)
    cam_c = P.cam(goal, 0.42, az, -24)
    k0 = max(
        0,
        min(rs["marks"][1][1] if len(rs["marks"]) > 1 else 0, rg["marks"][1][1] if len(rg["marks"]) > 1 else 0)
        - GV.LEAD,
    )
    n = max(len(qs), len(qg))
    small, tiny = (ImageFont.truetype(RV.FONT, s) for s in (14, 12))
    hw = GX.WIDTH / 2
    shelf = GP.SHELVES[pl["shelf"]]
    title = f"{shelf:.2f} m 高的隔板，箱子离柜门 {pl['depth'] * 100:.0f} cm，{'右' if sd > 0 else '左'}手"
    eyes = GX.Eyes(env, pl["body"])
    seen = {}

    def sees(q) -> bool:
        key = q.tobytes()
        if key not in seen:
            d = env.sim.d
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            seen[key] = eyes.visible()
        return seen[key]

    def draw(k: int, phase: str) -> np.ndarray:
        imgs = []
        ks, kg = min(k, len(qs) - 1), min(k, len(qg) - 1)
        vis = [sees(qs[ks]), sees(qg[kg])]
        for which, (rr, cam, W_, H_) in enumerate(((P.rm, cam_m, RV.WM_, RV.HM), (P.rc, cam_c, RV.WC, RV.HC))):
            d = P.ds[0]
            d.qpos[:] = qs[ks]
            mujoco.mj_forward(m, d)
            rr.update_scene(d, camera=cam)
            scn = rr.scene
            if which == 1:
                for gi in range(scn.ngeom):
                    g = scn.geoms[gi]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and P.robot[m.geom_bodyid[g.objid]]:
                        g.rgba[3] = 0.35
            if phase != "plan":
                P.ghost(scn, qg[kg], (*RV.ORANGE, 0.12 if which else 0.26))
                bp, bq = qg[kg][jadr : jadr + 3], qg[kg][jadr + 3 : jadr + 7]
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
        for j, (r, nm, col, kk, v, nq) in enumerate(((rs, "实体：" + NAMES[solid], RV.CYAN8, ks, vis[0], len(qs)),
                                                      (rg, "橙色影子：" + NAMES[ghost], RV.ORANGE8, kg, vis[1], len(qg)))):  # fmt: skip
            y = y0 + 12 + 46 * j
            dr.rectangle([10, y + 3, 24, y + 15], fill=col)
            ph = "还没动" if phase == "plan" else GV.phase_at(r["marks"], kk)
            if r.get("declined") and kk >= nq - 1:
                ph = "判断会撞柜子，停下"
            dr.text(
                (30, y), f"{nm}　现在：{ph}　头部相机{'看得见' if v else '看不见'}箱子", font=small, fill=(25, 25, 25)
            )
            if phase == "end":
                a = r.get("at_grip_cm")
                txt = (
                    f"合爪时手爪中心离抓取点 {math.hypot(*a):.1f} cm，其中左右方向 {abs(a[1]):.1f} cm；" if a else ""
                ) + f"结果：{verdict(r)}"
                dr.text((30, y + 20), txt, font=tiny, fill=(80, 80, 80))
        dr.text(
            (10, y0 + 100),
            "绿点是抓取点；两次都从完全相同的位置走过来；夹爪张开 9.5 cm，箱子 8 cm 宽",
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
    tag = f"exp_{pid}_{si}_{solid}_vs_{ghost}"
    src = OUTV / "src" / f"{tag}.mp4"
    RD.write(frames, src, fps=RV.FPS)
    nb = RD.preview(src, OUTV / "preview" / f"{tag}.mp4")
    Image.fromarray(end).save(OUTV / "preview" / f"{tag}.jpg", quality=86)
    Image.fromarray(frames[len(frames) // 2]).save(OUTV / "preview" / f"{tag}_mid.jpg", quality=86)
    print(f"{tag}: {len(frames)} frames, preview {nb} bytes", flush=True)
    return src


if __name__ == "__main__":
    env = GP.make_env()
    R = results()
    for a in sys.argv[1:]:
        pid, si, solid, ghost = a.split(":")
        render(env, R, int(pid), int(si), solid, ghost)

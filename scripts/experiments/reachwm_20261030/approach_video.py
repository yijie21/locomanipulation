"""Comparison videos for approach_eval.py: one case, the self model's run solid, the direct command's run as an orange ghost.
Both walk the same commanded walk from the same state (bit-identical until the reach), so the ghost covers the robot
until the arm moves. Drawn: the goal, where the walk was told to end and where it really ended, both reach commands, the
model's predictions made before walking (where the direct command would land, where its own command will land; sphere
radius = the ensemble's spread), a close-up of the goal and a strip with each run's phase and distance to the goal.

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/reachwm_20261030/approach_video.py 3 17
Writes outputs/reachwm_20261030/approach/videos/ (src/, preview/ <= 500 KB).
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
import approach_eval as AE  # noqa: E402
import report_video as RV  # noqa: E402

E, RD = AE.E, RV.RD
OUTV = AE.OUT / "videos"
NAMES = {
    "wm": "出发前用延伸模型一次算好",
    "wm_walk": "自身模型反推并挑走法，闭眼",
    "direct": "直接发命令",
    "post": "走完再反推，站位按命令推算",
    "post_wk": "走完再反推，站位用走路模型估计",
    "post_odo": "走完再反推，站位用腿部里程计估计",
    "post_fuse": "走完再反推，站位用走路模型和里程计估计",
}
BELIEF = {
    "post_fuse": "walk_fused",
    "post_wk": "walk_pred",
    "post_odo": "walk_odo",
}  # the stop error the robot believes
GREY = (0.45, 0.45, 0.48)


def results() -> dict:
    out = {}
    for f in sorted(AE.OUT.glob("h_w*.json")):
        for r in json.load(open(f)):
            out[(r["i"], r["method"])] = r
    return out


def to_w(start, v) -> np.ndarray:
    p0, yaw0 = np.asarray(start[:3], float), float(start[3])
    R0 = E.rot_z(yaw0)
    v = np.asarray(v, float)
    return np.r_[p0[:2] + (R0 @ np.r_[v[:2], 0.0])[:2], v[2]]


def phase(r: dict, k: int, n: int) -> str:
    if k < r["k_walk"]:
        return "看了一眼目标，算好命令" if r["method"].startswith("wm") else "看了一眼目标"
    if k < r.get("k_walked", n):
        return "闭眼走过去"
    if k < r.get("k_reach", n):
        return "站稳"
    if k < n - 1:
        return "闭眼伸手"
    return "停稳"


def render(env, R: dict, i: int, solid: str = "wm", ghost: str = "direct") -> Path:
    rs, rg = R[(i, solid)], R[(i, ghost)]
    qs, qg = (np.load(AE.OUT / "q" / f"{i}_{mm}.npy") for mm in (solid, ghost))
    m = env.sim.m
    P = RV.Painter(m)
    st = rs["start"]
    yaw0 = float(st[3])
    goal = np.asarray(rs["goal_w"], float)
    sd = 1.0 if rs["side"] == "right" else -1.0
    walk = np.asarray(rs["walk"], float)
    told = to_w(st, np.r_[walk[:2], 0.0])
    told[2] = 0.006
    p_start = np.r_[st[0], st[1], 0.6]
    mid = 0.5 * (p_start + np.r_[goal[:2], 0.6])
    cam_m = P.cam(mid, 2.5, math.degrees(yaw0) + 75 * sd, -16)
    cam_c = P.cam(goal, 0.55, math.degrees(yaw0) + 40 * sd, -22)
    k0 = max(0, rs["k_sight"] - 5)
    n = max(len(qs), len(qg))
    small, tiny = (ImageFont.truetype(RV.FONT, s) for s in (14, 12))
    side_txt = "右" if sd > 0 else "左"
    g = rs["goal"]
    where = f"出发时身体前方 {g[0]:.2f} m" if g[0] > 0.5 else f"出发时身体前方只有 {g[0]:.2f} m，要往后退"
    title = f"目标在{where}，高 {g[2]:.2f} m，{side_txt}手；只在出发前看一眼，之后全程闭眼"
    bel = None
    if BELIEF.get(solid) in rs:
        e = np.asarray(rs[BELIEF[solid]], float)
        bel = to_w(st, np.r_[walk[:2] + e[:2] / 100, 0.0])
        bel[2] = 0.006
    pn = to_w(st, rs["pred_naive_pos"]) if "pred_naive_pos" in rs else None
    pp = to_w(st, rs["pred_pos"]) if "pred_pos" in rs else None

    def dist(q, side):
        d = P.ds[0]
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        b = m.body(f"{side}_wrist_yaw_link").id
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.1515, 0.0, 0.0])

    D = {nm: [100 * float(np.linalg.norm(dist(q, rs["side"]) - goal)) for q in qq] for nm, qq in (("s", qs), ("g", qg))}

    act_cm = 100 * np.asarray(rs["walk_err"][:2], float)
    bel_cm = np.asarray(rs[BELIEF[solid]], float)[:2] if BELIEF.get(solid) in rs else None

    def stance_inset() -> Image.Image:
        """Top view of where the walk stopped, around the commanded spot: forward up, left to the left, cm."""
        S_ = 178
        im = Image.new("RGB", (S_, S_ + 52), (252, 252, 250))
        d_ = ImageDraw.Draw(im)
        d_.rectangle([0, 0, S_ - 1, S_ + 51], outline=(200, 200, 200))
        pts = [act_cm] + ([bel_cm] if bel_cm is not None else [])
        R_ = max(4.0, 1.25 * max(float(np.hypot(*v)) for v in pts))
        sc = (S_ / 2 - 14) / R_
        cx = cy = S_ / 2

        def px(v):
            return cx - sc * v[1], cy - sc * v[0]

        d_.text((6, 4), "站位放大，俯视，上方是前方", font=tiny, fill=(70, 70, 70))
        x0_, y0_ = px((0.0, 0.0))
        d_.line([x0_ - 7, y0_, x0_ + 7, y0_], fill=(110, 110, 115), width=2)
        d_.line([x0_, y0_ - 7, x0_, y0_ + 7], fill=(110, 110, 115), width=2)
        for v, col, r_ in ((act_cm, RV.CYAN8, 7), (bel_cm, RV.VIOLET8, 4)):
            if v is None:
                continue
            x, y = px(v)
            d_.ellipse([x - r_, y - r_, x + r_, y + r_], fill=col, outline=(255, 255, 255))
        bar = 2.0 * sc
        d_.line([S_ - 14 - bar, S_ - 10, S_ - 14, S_ - 10], fill=(60, 60, 60), width=2)
        d_.text((S_ - 14 - bar, S_ - 26), "2 cm", font=tiny, fill=(60, 60, 60))
        y_ = S_ + 2
        for col, txt in (
            ((110, 110, 115), "十字：命令要走到的位置"),
            (RV.CYAN8, "实际走到的位置"),
            (RV.VIOLET8, "机器人自己估计的位置"),
        ):
            if col == RV.VIOLET8 and bel_cm is None:
                continue
            d_.rectangle([6, y_ + 4, 14, y_ + 12], fill=col)
            d_.text((19, y_), txt, font=tiny, fill=(50, 50, 50))
            y_ += 16
        return im

    def draw(k: int, frozen: bool) -> np.ndarray:
        ks, kg = min(k, len(qs) - 1), min(k, len(qg) - 1)
        imgs = []
        for which, (rr, cam, W_, H_) in enumerate(((P.rm, cam_m, RV.WM_, RV.HM), (P.rc, cam_c, RV.WC, RV.HC))):
            d = P.ds[0]
            d.qpos[:] = qs[ks]
            mujoco.mj_forward(m, d)
            rr.update_scene(d, camera=cam)
            scn = rr.scene
            if which == 1:
                for gi in range(scn.ngeom):
                    gg = scn.geoms[gi]
                    if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and P.robot[m.geom_bodyid[gg.objid]]:
                        gg.rgba[3] = 0.35
            if not frozen:
                P.ghost(scn, qg[kg], (*RV.ORANGE, 0.12 if which else 0.26))
            P.sphere(scn, goal, 0.012, (*RV.GREEN, 1.0))
            labs = [(goal, "目标", RV.GREEN8)]
            if which == 0:
                P.sphere(scn, told, 0.03, (*GREY, 0.9))
                if k >= rs.get("k_reach", n):
                    q1 = qs[rs["k_reach"]]
                    P.sphere(scn, np.r_[q1[0], q1[1], 0.006], 0.02, (*RV.CYAN, 0.95))
                    if bel is not None:
                        P.sphere(scn, bel, 0.016, (*RV.VIOLET, 0.95))
            if pn is not None:
                P.sphere(scn, pn, max(rs["pred_naive_std"] / 100, 0.008), (*RV.ORANGE, 0.35))
                if which == 1:
                    labs.append((pn, "模型预言：直接发命令手会停在这", RV.ORANGE8))
            if pp is not None and which == 1:
                P.sphere(scn, pp, max(rs["pred_std"] / 100, 0.008), (*RV.CYAN, 0.30))
            if which == 1 and k >= rs.get("k_reach", n):
                P.box(scn, rs["target_w"], 0.008, (*RV.CYAN, 0.95))
                labs.append((np.asarray(rs["target_w"]), "反推出的伸手命令", RV.CYAN8))
                P.box(scn, rg["target_w"], 0.008, (*RV.ORANGE, 0.95))
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
            used = []
            for pos, txt, col in labs:
                px = P.project(scn, W_, H_, pos)
                if px is None or not (0 <= px[0] < W_ and 0 <= px[1] < H_):
                    continue
                x, y = px[0] + 8, px[1] - 8
                while any(abs(y - yy) < 16 and abs(x - xx) < 150 for xx, yy in used):
                    y += 17
                used.append((x, y))
                dr.text((x, y), txt, font=tiny, fill=col, stroke_width=2, stroke_fill=(255, 255, 255))
            imgs.append(np.asarray(img))
        canvas = Image.new("RGB", (RV.W, RV.TB + RV.HM + 150), (248, 248, 246))
        canvas.paste(Image.fromarray(np.concatenate(imgs, 1)), (0, RV.TB))
        if k >= rs.get("k_reach", n):
            canvas.paste(stance_inset(), (RV.WM_ - 186, RV.TB + 8))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle([0, 0, RV.W, RV.TB], fill=(24, 28, 34))
        dr.text((10, 9), title, font=small, fill=(240, 240, 240))
        y0 = RV.TB + RV.HM
        for j, (r, nm, col, kk, key) in enumerate(
            ((rs, "实体：" + NAMES[solid], RV.CYAN8, ks, "s"), (rg, "橙色影子：" + NAMES[ghost], RV.ORANGE8, kg, "g"))
        ):
            y = y0 + 10 + 44 * j
            dr.rectangle([10, y + 3, 24, y + 15], fill=col)
            ph = "还没动" if frozen else phase(r, kk, len(qs if key == "s" else qg))
            dr.text((30, y), f"{nm}　现在：{ph}　手离目标 {D[key][kk]:.1f} cm", font=small, fill=(25, 25, 25))
            if k >= n - 1:
                dr.text((30, y + 20), f"最后离目标 {r['err']:.2f} cm", font=tiny, fill=(80, 80, 80))
        we = 100 * math.hypot(*rs["walk_err"][:2])
        foot = f"走路命令：往{'前' if walk[0] >= 0 else '后'} {abs(walk[0]):.2f} m；实际停偏 {we:.1f} cm。"
        if bel is not None:
            eb = np.asarray(rs[BELIEF[solid]], float)[:2] - 100 * np.asarray(rs["walk_err"][:2])
            foot += f"机器人估计的站位离实际站位 {float(np.hypot(*eb)):.1f} cm。"
        if pn is not None:
            foot += f"出发前模型就预言：直接发命令会偏 {rs['pred_naive']:.1f} cm，反推出的命令会偏 {rs['pred_err']:.1f} cm，分歧 {rs['pred_std']:.1f} cm"
        dr.text((10, y0 + 100), foot, font=tiny, fill=(90, 90, 90))
        legend = "绿点是目标；灰点是命令要走到的位置；青点是实际走到的位置"
        legend += (
            "；紫点是机器人自己估计的站位" if bel is not None else "；半透明球是出发前的预言，球的大小是 5 个网络的分歧"
        )
        dr.text((10, y0 + 122), legend, font=tiny, fill=(120, 120, 120))
        return np.asarray(canvas)

    frames = [draw(k0 + 5, True)] * 25
    for k in range(k0, n):
        frames.append(draw(k, False))
    end = draw(n - 1, False)
    frames += [end] * 50
    (OUTV / "src").mkdir(parents=True, exist_ok=True)
    (OUTV / "preview").mkdir(parents=True, exist_ok=True)
    tag = f"ap_{i}_{solid}_vs_{ghost}"
    src = OUTV / "src" / f"{tag}.mp4"
    RD.write(frames, src, fps=RV.FPS)
    nb = RD.preview(src, OUTV / "preview" / f"{tag}.mp4")
    Image.fromarray(end).save(OUTV / "preview" / f"{tag}.jpg", quality=86)
    Image.fromarray(frames[0]).save(OUTV / "preview" / f"{tag}_plan.jpg", quality=86)
    print(f"{tag}: {len(frames)} frames, preview {nb} bytes", flush=True)
    return src


if __name__ == "__main__":
    env = E.Env("open")
    R = results()
    for a in sys.argv[1:]:
        parts = a.split(":")
        render(env, R, int(parts[0]), *(parts[1:] or []))

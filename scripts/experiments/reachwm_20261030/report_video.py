"""Comparison videos from report_media.py recordings (its phase v; 2026-11-02, the report rewrite).

One frame: a title bar, the whole robot (left), a close-up of the goal (right), a log strip (per run: distance to the
goal now and at the end, what the world model predicted; a plot of the distance, or of the wrist angle, over time).
In the scene: the goal (green, the translucent shell is 2 cm), runs[0] solid with a cyan jaw trail, runs[1] as an
orange translucent ghost with an orange trail, each run's predicted landing (a translucent sphere in its colour, radius
the ensemble's std, at least 4 mm) and its command target (a small box). Mode "ref" draws one run with the reference
pose the controller is asked to track as a violet shadow, and the commanded jaw centre as a violet trail.
The clip starts frozen on the plan (1.6 s), plays at real time from the end of the reset (and walk), and ends frozen
(2 s) on the result.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import env as E  # noqa: E402
import render as RD  # noqa: E402

MED = E.REPO / "outputs" / "reachwm_20261030" / "media"
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
WM_, HM = 560, 420  # main view
WC, HC = 400, 420  # close-up
TB, LG = 36, 196  # title bar, log strip
W = WM_ + WC
FPS, TPF = 25, 2  # 50 Hz ticks, 2 per frame
CYAN, ORANGE, GREEN, VIOLET = (0.05, 0.75, 0.85), (1.0, 0.5, 0.1), (0.15, 0.8, 0.25), (0.6, 0.35, 1.0)
GREY = (0.5, 0.5, 0.52)
CYAN8, ORANGE8, GREEN8, VIOLET8 = [tuple(int(255 * c) for c in col) for col in (CYAN, ORANGE, GREEN, VIOLET)]
GREY8 = (95, 95, 100)

VIEWS = {
    "hero": dict(
        mode="pair",
        metric="pos",
        iters=True,
        title="同一起点、同一目标：直接把目标当命令发出去，和先用自身模型反推命令、再闭眼执行",
        names={"inv": "反推后的命令", "naive": "直接把目标当命令"},
        short={"inv": "反推", "naive": "直接发目标"},
    ),
    "gap": dict(
        mode="ref",
        metric="gap",
        src_run="naive",
        title="紫色影子是控制器被要求做到的姿态，实体是它实际做到的",
        names={"naive": "实际"},
    ),
    "step": dict(
        mode="pair",
        metric="pos",
        feet=True,
        align=True,
        main_el=-48,
        main_dist=2.1,
        main_az=70,
        title="够不着就先挪一步：原地伸手，和让模型挑好位置、挪一步再伸手",
        names={"step": "先挪一步再伸手", "nostep": "原地伸手"},
        short={"step": "挪步后反推", "nostep": "原地反推"},
    ),
    "servo": dict(
        mode="pair",
        metric="pos",
        title="同一个反推命令，加两种伺服：旧伺服一开始就拉，新伺服等手停稳后只修位置",
        names={"late_pos": "新伺服", "orig": "旧伺服", "open": "不开伺服"},
    ),
    "wrist": dict(
        mode="pair",
        metric="att",
        frames=True,
        title="手腕朝向：只反推位置，和位置、手腕一起反推",
        names={"pose3g": "位置和手腕一起反推", "pos": "只反推位置"},
        short={"pose3g": "一起反推", "pos": "只反推位置"},
    ),
    "curse": dict(
        mode="pair",
        metric="pos",
        title="模型很有把握，实际却偏了：手腕转得大时，模型估得太乐观",
        names={"pose3g": "手腕优先挑出的命令", "pos": "只反推位置的命令"},
        short={"pose3g": "手腕优先", "pos": "只反推位置"},
    ),
    "stance_a": dict(
        mode="pair",
        metric="att",
        frames=True,
        feet=True,
        main_el=-40,
        main_dist=1.9,
        main_az=60,
        close=(0.30, 60, -70),
        title="手腕要转很多时先挪站位：原地伸手，和让模型挑好站位、挪过去再伸手",
        names={"move": "挪站位后，手腕优先", "stay": "原地，手腕优先"},
        short={"move": "挪站位后", "stay": "原地"},
    ),
    "stance_b": dict(
        mode="pair",
        metric="pos",
        feet=True,
        main_el=-40,
        main_dist=1.9,
        main_az=60,
        title="站定后再看一眼：模型预言手腕优先会偏，就改用只反推位置",
        names={"move_pos": "挪后只反推位置", "move_p3": "挪后手腕优先", "stay": "原地手腕优先"},
        short={"move_pos": "只反推位置", "move_p3": "手腕优先", "stay": "原地"},
    ),
    # the relabelling report (2026-11-05): "nopred" draws no prediction (a fine-tuned controller sent the wanted pose has
    # none), "both" logs the wrist angle beside the distance, "plain" names the self model the landing predictor (预测器)
    "copy": dict(
        mode="pair",
        metric="pos",
        nopred=True,
        both=True,
        title="同一条命令：ScaleBFM，和照着它的录像直接模仿出来的微调版",
        names={"r1": "微调版 直接模仿", "frozen": "ScaleBFM"},
    ),
    "drift": dict(
        mode="pair",
        metric="pos",
        nopred=True,
        both=True,
        title="同一条命令：ScaleBFM，和 DAgger 第一轮之后的微调版",
        names={"r3dag": "微调版 DAgger 第一轮", "frozen": "ScaleBFM"},
    ),
    "dag": dict(
        mode="pair",
        metric="att",
        frames=True,
        nopred=True,
        title="同一条命令：ScaleBFM，和 DAgger 三轮之后的微调版",
        names={"r5dag": "微调版 DAgger 三轮", "frozen": "ScaleBFM"},
    ),
    "wronly": dict(
        mode="pair",
        metric="pos",
        frames=True,
        nopred=True,
        both=True,
        title="同一条命令：ScaleBFM，和只学转手腕的微调版",
        names={"wr2": "微调版 只学手腕", "frozen": "ScaleBFM"},
    ),
    "hyb": dict(
        mode="pair",
        metric="pos",
        both=True,
        plain=True,
        title="各配各的落点预测器：ScaleBFM 配它的预测器，微调版配为它重训的预测器",
        names={"hyb": "微调版＋为它重训的预测器", "inv": "ScaleBFM＋它的预测器"},
        short={"hyb": "微调版", "inv": "ScaleBFM"},
    ),
}


def view_of(case: str) -> dict:
    for k in sorted(VIEWS, key=len, reverse=True):
        if case.startswith(k):
            return VIEWS[k]
    raise KeyError(case)


class Painter:
    def __init__(self, m):
        self.m = m
        self.ds = [mujoco.MjData(m) for _ in range(2)]
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        mg = 2 * m.ngeom + 3000
        self.rm = mujoco.Renderer(m, height=HM, width=WM_, max_geom=mg)
        self.rc = mujoco.Renderer(m, height=HC, width=WC, max_geom=mg)
        root = m.body_rootid[m.body("pelvis").id]
        self.robot = np.array([m.body_rootid[b] == root for b in range(m.nbody)])

    @staticmethod
    def cam(lookat, dist, az, el):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = lookat
        c.distance, c.azimuth, c.elevation = dist, az, el
        return c

    def ghost(self, scn, q, rgba) -> None:
        d = self.ds[1]
        d.qpos[:] = q
        mujoco.mj_forward(self.m, d)
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(self.m, d, self.opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(n0, scn.ngeom):
            g = scn.geoms[i]
            keep = g.objtype == mujoco.mjtObj.mjOBJ_GEOM and self.robot[self.m.geom_bodyid[g.objid]]
            g.rgba[:] = rgba if keep else (0, 0, 0, 0)
            g.matid = -1

    @staticmethod
    def sphere(scn, pos, r, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                            np.asarray(pos, float), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))  # fmt: skip
        scn.ngeom += 1

    @staticmethod
    def box(scn, pos, r, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([r, r, r]),
                            np.asarray(pos, float), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))  # fmt: skip
        scn.ngeom += 1

    @staticmethod
    def line(scn, a, b, w, rgba) -> None:
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(np.asarray(b) - np.asarray(a)) < 1e-6:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                            np.asarray(rgba, np.float32))  # fmt: skip
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    def frame3(self, scn, pos, R, L, alpha, w=0.0025) -> None:
        for i, col in enumerate(((0.9, 0.15, 0.1), (0.1, 0.75, 0.15), (0.15, 0.35, 0.95))):
            self.line(scn, pos, pos + L * R[:, i], w, (*col, alpha))

    @staticmethod
    def project(scn, W_, H_, p):
        """Pixel of world point p in the last rendered scene (None behind the camera)."""
        c0, c1 = scn.camera[0], scn.camera[1]
        pos = 0.5 * (np.array(c0.pos) + np.array(c1.pos))
        fwd, up = np.array(c0.forward), np.array(c0.up)
        right = np.cross(fwd, up)
        v = np.asarray(p, float) - pos
        z = float(v @ fwd)
        if z <= 1e-6:
            return None
        n = c0.frustum_near
        x, y = float(v @ right) * n / z, float(v @ up) * n / z
        cx, hw, bt, tp = c0.frustum_center, c0.frustum_width, c0.frustum_bottom, c0.frustum_top
        if hw == 0:  # mjv_updateScene leaves the width to mjr_render, which takes it from the viewport's aspect
            hw = 0.5 * (tp - bt) * W_ / H_
        return ((x - (cx - hw)) / (2 * hw) * W_, (tp - y) / (tp - bt) * H_)


def load(case: str) -> dict:
    with open(MED / "rec" / f"{case}.pkl", "rb") as f:
        return pickle.load(f)


def fmt_cm(x: float) -> str:
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def render(case: str, src: str | None = None) -> Path:
    """Draw recording `src` (default `case`) with the view `case` names (e.g. gap_a from hero_a's naive run)."""
    V = view_of(case)
    rec = load(src or case)
    runs = rec["runs"]
    if V["mode"] == "ref":
        runs = [r for r in runs if r["name"] == V["src_run"]][:1]
    shown = runs[:2]
    r0 = runs[0]
    side = 0 if r0["side"] == "right" else 1
    goal = r0["goal_w"]
    env = E.Env("open")
    m = env.sim.m
    P = Painter(m)
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 14)
    tiny = ImageFont.truetype(FONT, 12)
    # per run: ticks from the end of the reset (and walk) on
    T = []
    for r in runs:
        tp = r["tape"]
        # "align": every clip starts where runs[1] (the no-step run) became ready: the end of the shared reset and walk,
        # so the step run's clip shows its step (its own k_ready is after the step)
        k = runs[1]["k_ready"] if V.get("align") and len(runs) > 1 else r["k_ready"]
        t = tp["t"][k:] - tp["t"][k]
        jaw = tp["jaw"][k:, side]
        T.append(dict(t=t, q=tp["q"][k:], ref=tp["ref"][k:], jaw=jaw, Rj=tp["Rj"][k:, side], cj=tp["cj"][k:, side],
                      lead=np.linalg.norm(tp["lead"][k:, side], axis=1), dist=100 * np.linalg.norm(jaw - goal, axis=1),
                      att=np.array([np.degrees(np.arccos(np.clip((np.trace(r["rot_w"].T @ R) - 1) / 2, -1, 1))) for R in tp["Rj"][k:, side]]),
                      tc=r["t_cmd"] - tp["t"][k]))  # fmt: skip
    n = max(len(x["t"]) for x in T[:2]) if V["mode"] == "pair" else len(T[0]["t"])
    if V["mode"] == "ref":
        T[0]["gap"] = 100 * np.linalg.norm(T[0]["jaw"] - T[0]["cj"], axis=1)
    # cameras: the main one from the reaching side, in front; the close-up on the goal
    yaw = np.degrees(r0["yaw0"])
    sd = 1.0 if r0["side"] == "right" else -1.0
    p0 = r0["pelvis0"]
    look = np.r_[0.4 * p0[:2] + 0.6 * goal[:2], 0.5 * (0.8 + goal[2])]
    cam_m = P.cam(look, V.get("main_dist", 1.45), yaw + 180 - V.get("main_az", 50) * sd, V.get("main_el", -16))
    cam_c = P.cam(goal, 0.30, yaw + 180 - 35 * sd, -38)
    if V.get("frames"):
        cam_c = P.cam(goal, 0.30, yaw + 180 - 60 * sd, -22)
    if V.get("close"):  # (distance, azimuth off the front, elevation) for the close-up
        cam_c = P.cam(goal, V["close"][0], yaw + 180 - V["close"][1] * sd, V["close"][2])
    metric = V["metric"]
    feet0 = []
    if V.get("feet"):  # where the feet stood when the clip starts
        d = P.ds[0]
        d.qpos[:] = T[0]["q"][0]
        mujoco.mj_forward(m, d)
        feet0 = [np.r_[d.xpos[m.body(b).id][:2], 0.012] for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    names = V["names"]
    short = V.get("short", names)
    # both shown runs sent the same command (the servo videos): one grey command and one grey prediction, not two
    # coloured markers drawn on top of each other
    same = (V["mode"] == "pair" and len(shown) > 1 and np.allclose(shown[0]["target_w"], shown[1]["target_w"])
            and np.allclose(shown[0]["rot_cmd_w"], shown[1]["rot_cmd_w"]))  # fmt: skip
    frames = []

    def draw(i: int, phase: str) -> np.ndarray:
        imgs = []
        for which, (rr, cam, W_, H_) in enumerate(((P.rm, cam_m, WM_, HM), (P.rc, cam_c, WC, HC))):
            d = P.ds[0]
            x0 = T[0]
            k0 = min(i, len(x0["t"]) - 1)
            d.qpos[:] = x0["q"][k0]
            mujoco.mj_forward(m, d)
            rr.update_scene(d, camera=cam)
            scn = rr.scene
            if which == 1:  # x-ray close-up: the solid robot see-through, so the markers at the hand stay visible
                for gi in range(scn.ngeom):
                    g = scn.geoms[gi]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and P.robot[m.geom_bodyid[g.objid]]:
                        g.rgba[3] = 0.10
            ga = 0.06 if which == 1 else 0.22
            if V["mode"] == "pair" and len(T) > 1 and phase != "plan":
                x1 = T[1]
                P.ghost(scn, x1["q"][min(i, len(x1["t"]) - 1)], (*ORANGE, ga))
            if V["mode"] == "ref":
                ref = x0["ref"][k0]
                if not np.isnan(ref[0]):
                    q = x0["q"][k0].copy()
                    q[:3], q[3:7], q[env.m.kp.qadr] = ref[:3], ref[3:7], ref[7:]
                    P.ghost(scn, q, (*VIOLET, ga + 0.06))
            # goal and its 2 cm shell
            P.sphere(scn, goal, 0.008, (*GREEN, 1.0))
            P.sphere(scn, goal, 0.02, (*GREEN, 0.13))
            for j, (r, x) in enumerate(zip(shown, T[:2])):
                if V["mode"] == "ref" and j > 0:
                    break
                col = CYAN if j == 0 else ORANGE
                kk = min(i, len(x["t"]) - 1)
                # trail of the jaw centre up to now
                for a in range(0, kk, 3):
                    P.sphere(scn, x["jaw"][a], 0.0028 if which else 0.004, (*col, 0.85))
                if V["mode"] == "ref":
                    cj = x["cj"][: kk + 1]
                    for a in range(0, kk, 3):
                        if not np.isnan(cj[a][0]):
                            P.sphere(scn, cj[a], 0.0028 if which else 0.004, (*VIOLET, 0.85))
                    continue
                if not (same and j == 1):
                    cm = GREY if same else col
                    if not V.get("nopred"):
                        P.sphere(scn, r["pred_w"], max(r["pred_std"] / 100, 0.004), (*cm, 0.30))
                    P.box(scn, r["target_w"], 0.004, (*cm, 0.95))
                    P.line(scn, goal, r["target_w"], 0.0012, (*cm, 0.6))
                if which == 1 and phase != "plan":  # the jaw centre now, larger
                    P.sphere(scn, x["jaw"][kk], 0.006, (*col, 1.0))
                if V.get("frames"):  # approach (long) and closing (short) axes of the hand, in the run's colour
                    Rk = x["Rj"][kk]
                    P.line(scn, x["jaw"][kk], x["jaw"][kk] + 0.08 * Rk[:, 0], 0.0028, (*col, 1.0))
                    P.line(scn, x["jaw"][kk], x["jaw"][kk] + 0.04 * Rk[:, 1], 0.0028, (*col, 1.0))
                    if j == 0 and r["dev"] > 1:  # the command the inversion sent, turned off the wanted rotation
                        Rc = r["rot_cmd_w"]
                        P.line(scn, r["target_w"], r["target_w"] + 0.08 * Rc[:, 0], 0.0012, (*col, 0.7))
                        P.line(scn, r["target_w"], r["target_w"] + 0.04 * Rc[:, 1], 0.0012, (*col, 0.7))
            if V.get("iters") and phase == "plan":
                it = shown[0]["iters"]
                for a in range(len(it) - 1):
                    P.line(scn, it[a][0], it[a + 1][0], 0.0012, (*CYAN, 0.9))
                    P.line(scn, it[a][1], it[a + 1][1], 0.0012, (0.4, 0.4, 0.9, 0.9))
                for tw, pw, _ in it:
                    P.sphere(scn, pw, 0.003, (0.4, 0.4, 0.9, 0.9))
                P.sphere(scn, shown[0]["naive_pred_w"], max(shown[0]["naive_pred_std"] / 100, 0.004), (*ORANGE, 0.30))
            if V.get("frames"):
                Rg = shown[0]["rot_w"]
                P.line(scn, goal, goal + 0.08 * Rg[:, 0], 0.006, (*GREEN, 0.45))
                P.line(scn, goal, goal + 0.04 * Rg[:, 1], 0.006, (*GREEN, 0.45))
            if V.get("feet"):
                for fp in feet0:
                    P.box(scn, fp, 0.012, (0.45, 0.45, 0.45, 0.55))
            img = Image.fromarray(rr.render())
            dr = ImageDraw.Draw(img)
            # labels next to the markers
            placed = []

            def lab(pw, text, col):
                """Text beside the marker; moved down past labels already placed, to the left near the right edge."""
                uv = P.project(scn, W_, H_, pw)
                if uv is None or not (0 <= uv[0] < W_ and 0 <= uv[1] < H_):
                    return
                tw = dr.textlength(text, font=tiny) + 4
                x = uv[0] + 9 if uv[0] + 9 + tw < W_ - 4 else uv[0] - 9 - tw
                y = uv[1] - 8
                for _ in range(10):
                    bb = (x, y, x + tw, y + 15)
                    if not any(bb[0] < q[2] and q[0] < bb[2] and bb[1] < q[3] and q[1] < bb[3] for q in placed):
                        break
                    y += 15
                placed.append(bb)
                dr.text((x, y), text, font=tiny, fill=col, stroke_width=2, stroke_fill=(255, 255, 255))

            if which == 1:
                lab(goal, "目标", (20, 110, 30))
                if V["mode"] == "pair":
                    runs_l = (
                        [(shown[0], "两次共用的命令", GREY8)]
                        if same
                        else [
                            (r, short.get(r["name"], r["name"]), CYAN8 if j == 0 else ORANGE8)
                            for j, r in enumerate(shown)
                        ]
                    )
                    for r, nm, c8 in runs_l:
                        if not V.get("nopred"):
                            lab(
                                r["pred_w"],
                                (f"{nm}的预测落点" if V.get("plain") else f"{nm}的预言落点")
                                if not same
                                else "共用命令的预言落点",
                                c8,
                            )
                        on_goal = np.linalg.norm(r["target_w"] - goal) <= 0.01
                        lab(
                            goal if on_goal else r["target_w"],
                            (f"{nm}：命令就是目标" if on_goal else (nm if same else f"{nm}的命令")),
                            c8,
                        )
                # 2 cm scale bar at the goal's depth
                a = P.project(scn, W_, H_, goal)
                cv = np.array(scn.camera[0].up)
                rt = np.cross(np.array(scn.camera[0].forward), cv)
                b = P.project(scn, W_, H_, goal + 0.02 * rt)
                if a and b:
                    L = abs(b[0] - a[0])
                    dr.rectangle([W_ - 22 - L, H_ - 19, W_ - 18, H_ - 13], fill=(20, 20, 20))
                    dr.line([(W_ - 20 - L, H_ - 16), (W_ - 20, H_ - 16)], fill=(255, 255, 255), width=2)
                    dr.text(
                        (W_ - 20 - L, H_ - 36),
                        "2 cm",
                        font=tiny,
                        fill=(20, 20, 20),
                        stroke_width=2,
                        stroke_fill=(255, 255, 255),
                    )
                dr.text(
                    (8, 6), "目标附近放大", font=small, fill=(20, 20, 20), stroke_width=2, stroke_fill=(255, 255, 255)
                )
            else:
                dr.text((8, 6), "全身", font=small, fill=(20, 20, 20), stroke_width=2, stroke_fill=(255, 255, 255))
            imgs.append(np.asarray(img))
        top = np.concatenate(imgs, 1)
        canvas = Image.new("RGB", (W, TB + HM + LG), (248, 248, 246))
        canvas.paste(Image.fromarray(top), (0, TB))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle([0, 0, W, TB], fill=(24, 28, 34))
        dr.text((10, 8), V["title"], font=font, fill=(240, 240, 240))
        log_strip(dr, i, phase)
        return np.asarray(canvas)

    def log_strip(dr, i: int, phase: str) -> None:
        y0 = TB + HM
        t_now = T[0]["t"][min(i, len(T[0]["t"]) - 1)]
        dr.rectangle([0, y0, W, y0 + LG], fill=(250, 250, 248))
        dr.line([(0, y0), (W, y0)], fill=(200, 200, 200))
        tag = {
            "plan": "开始前：两次从同一个状态出发，收到同一条命令"
            if V.get("nopred")
            else (
                "开始前：机器人还没动，下面是预测器给的落点"
                if V.get("plain")
                else "计划阶段：机器人还没动，下面是模型的预言"
            ),
            "run": f"t = {t_now:.1f} s",
            "end": "结束：停稳 1 秒后测量",
        }[phase]
        dr.text((10, y0 + 6), tag, font=small, fill=(40, 40, 40))
        yy = y0 + 30
        lines = T if V["mode"] != "ref" else T[:1]
        for j, (r, x) in enumerate(zip(runs, lines)):
            col = (CYAN8, ORANGE8, (120, 120, 120))[min(j, 2)] if V["mode"] != "ref" else CYAN8
            kk = min(i, len(x["t"]) - 1)
            dr.rectangle([10, yy + 3, 24, yy + 15], fill=col)
            nm = names.get(r["name"], r["name"])
            if metric == "gap":
                txt = f"{nm}：手到被要求位置差 {fmt_cm(x['gap'][kk])} cm，离目标 {fmt_cm(x['dist'][kk])} cm"
                if phase == "end":
                    txt += f"，最终 {fmt_cm(r['err'])} cm"
            elif metric == "att":
                txt = f"{nm}：手腕差 {x['att'][kk]:.1f} 度，离目标 {fmt_cm(x['dist'][kk])} cm"
            else:
                txt = f"{nm}：离目标 {fmt_cm(x['dist'][kk])} cm" + (
                    f"，手腕差 {x['att'][kk]:.0f} 度" if V.get("both") else ""
                )
                if r["servo"] != "open":
                    txt += f"，伺服修正 {100 * x['lead'][kk]:.1f} cm"
            dr.text((30, yy), txt, font=small, fill=(25, 25, 25))
            if V["mode"] != "ref":
                pred = (
                    f"模型预言 {fmt_cm(r['pred_err'])} ± {r['pred_std']:.1f} cm"
                    if r["servo"] == "open"
                    else "命令由模型反推"
                )
                if V.get("plain"):
                    pred = f"预测器说偏 {fmt_cm(r['pred_err'])} ± {r['pred_std']:.1f} cm"
                if metric == "att":
                    pred = f"模型预言手腕差 {r['pred_att']:.0f} 度，命令转了 {r['dev']:.0f} 度"
                if r["cmd"] == "naive":
                    pred = f"模型预言会偏 {fmt_cm(r['pred_err'])} cm"
                if r["name"] == "orig" or r["name"] == "late_pos" or V.get("nopred"):
                    pred = ""
                fin = f"最终 {fmt_cm(r['err'])} cm" + (
                    f"，手腕 {r['att_deg']:.0f} 度" if metric == "att" or V.get("both") else ""
                )
                dr.text(
                    (30, yy + 18),
                    (pred + "　" if pred else "") + (fin if phase == "end" else ""),
                    font=tiny,
                    fill=(90, 90, 90),
                )
            yy += 40 if V["mode"] != "ref" else 22
        if V["mode"] != "ref":
            leg = "大圆点 手爪中心现在的位置　小点 它走过的路　半透明球 模型预言的落点，球越大模型越没把握　小方块 实际发出的命令"
            if V.get("plain"):
                leg = "大圆点 手爪中心现在的位置　小点 它走过的路　半透明球 预测器给的落点，球越大越没把握　小方块 实际发出的命令"
            if V.get("nopred"):
                leg = "大圆点 手爪中心现在的位置　小点 它走过的路　小方块 发出的命令"
            if same:
                leg += "　灰色 两次共用"
            if V.get("frames") and V.get("both"):
                leg += "　长线 手的朝前方向　短线 两指合拢方向　绿色粗线 想要的朝向"
            elif V.get("frames"):
                leg = "长线 手的朝前方向　短线 两指合拢方向　绿色粗线 想要的朝向　细线 实际发出的手腕命令"
            if V.get("feet"):
                leg += "　灰方块 开始时的脚"
            lines_l, cur = [], ""  # wrap between items, never inside one
            for item in leg.split("　"):
                nxt = item if not cur else cur + "　" + item
                if cur and dr.textlength(nxt, font=tiny) > 445:
                    lines_l.append(cur)
                    cur = item
                else:
                    cur = nxt
            lines_l.append(cur)
            for li, chunk in enumerate(lines_l[:3]):
                dr.text((10, y0 + LG - 16 * len(lines_l[:3]) - 6 + 16 * li), chunk, font=tiny, fill=(120, 120, 120))
        if V["mode"] == "ref":
            dr.text((30, yy + 4), "紫点：控制器每一刻被要求的手位置　青点：实际的手", font=tiny, fill=(90, 90, 90))
            dr.text(
                (30, yy + 22),
                f"控制器收到的就是目标本身，但最后停在离目标 {fmt_cm(r0['err'])} cm 处",
                font=tiny,
                fill=(90, 90, 90),
            )
        # plot
        px0, px1, py0, py1 = 470, W - 16, y0 + 16, y0 + LG - 26
        dr.rectangle([px0, py0, px1, py1], outline=(205, 205, 205))
        tmax = max(x["t"][-1] for x in (T if V["mode"] != "ref" else T[:1]))
        if metric == "att":
            lo, hi, f = 0.0, 90.0, (lambda v: v)
            ticks = [0, 30, 60, 90]
            unit = "手腕差多少度"
        else:
            lo, hi = np.log10(0.1), np.log10(60.0)
            f = lambda v: np.log10(np.clip(v, 0.1, 60.0))  # noqa: E731
            ticks = [0.3, 1, 3, 10, 30]
            unit = "离目标多远，cm，对数刻度" if metric == "pos" else "cm，对数刻度"
        X = lambda t: px0 + (px1 - px0) * t / tmax  # noqa: E731
        Y = lambda v: py1 - (py1 - py0) * (f(v) - lo) / (hi - lo)  # noqa: E731
        for tk in ticks:
            yy = Y(tk)
            dr.line([(px0, yy), (px1, yy)], fill=(232, 232, 232))
            dr.text((px0 - 30, yy - 7), f"{tk:g}", font=tiny, fill=(110, 110, 110))
        if metric != "att":
            dr.line([(px0, Y(2)), (px1, Y(2))], fill=(150, 200, 150))
            dr.text((px1 - 40, Y(2) - 15), "2 cm", font=tiny, fill=(80, 150, 80))
        dr.text((px0 + 6, py0 + 2), unit, font=tiny, fill=(110, 110, 110))
        for ts in range(0, int(tmax) + 1):
            dr.text((X(ts) - 4, py1 + 4), f"{ts} s" if ts == int(tmax) else f"{ts}", font=tiny, fill=(110, 110, 110))
        series = []
        if V["mode"] == "ref":
            series = [(T[0]["dist"], CYAN8), (T[0]["gap"], VIOLET8)]
        else:
            series = [
                (x["att"] if metric == "att" else x["dist"], (CYAN8, ORANGE8, (150, 150, 150))[min(j, 2)])
                for j, x in enumerate(T)
            ]
        for j, (s, col) in enumerate(series):
            x = T[0] if V["mode"] == "ref" else T[j]
            kk = len(x["t"]) if phase == "end" else min(i + 1, len(x["t"]))
            if phase == "plan":
                kk = 0
            pts = [(X(x["t"][a]), Y(s[a])) for a in range(0, kk, 2)]
            if len(pts) > 1:
                dr.line(pts, fill=col, width=2)
            if V["mode"] != "ref" and 0 < x["tc"] < tmax:
                dr.line([(X(x["tc"]), py0), (X(x["tc"]), py0 + 6)], fill=col, width=2)
        if phase == "run":
            dr.line([(X(t_now), py0), (X(t_now), py1)], fill=(60, 60, 60))

    plan = draw(0, "plan")
    frames += [plan] * int(1.6 * FPS)
    for i in range(0, n, TPF):
        frames.append(draw(i, "run"))
    end = draw(n - 1, "end")
    frames += [end] * int(2.0 * FPS)
    (MED / "src").mkdir(parents=True, exist_ok=True)
    (MED / "preview").mkdir(parents=True, exist_ok=True)
    out = MED / "src" / f"{case}.mp4"
    RD.write(frames, out, fps=FPS)
    nbytes = RD.preview(out, MED / "preview" / f"{case}.mp4")
    Image.fromarray(end).save(MED / "preview" / f"{case}.jpg", quality=86)
    Image.fromarray(plan).save(MED / "preview" / f"{case}_plan.jpg", quality=86)
    print(f"{case}: {len(frames)} frames, preview {nbytes} bytes", flush=True)
    return out

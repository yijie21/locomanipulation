"""selfmodel_via_20261120: comparison videos of a replayed planner path, the compensated command solid and the direct one as
an orange ghost (2026-11-23; the user asked for videos of the day's results on the 4090).

    cd $REPO && MUJOCO_GL=egl SIMPLE_ROOT=... $PY -I scripts/experiments/selfmodel_via_20261120/via_video.py \
        <direct dir> <compensated dir> <routes dir> <scene dir> <out dir> <title> pick | <reset index> ...

The two runs replay the same planned path from the same state (same seed and reset), so they hold the same until the
command; then the direct run sends the planned path as it is and the compensated one sends it moved against the self
model's predicted offset (via_comp.py). Drawn from the 10 Hz record, qpos rebuilt as arm_spheres.py does (pelvis from W
and the gravity direction, joints S[:29] in the metadata order, the acting hand's fingers at hand_q, the other hand's at
zero), linearly interpolated to 25 fps:
  violet line    the palm's wanted path (the plan, comp_ref0), its end a green sphere
  cyan dots      the palm of the command the compensated run sent (comp_ref): moved off the plan on purpose
  cyan / orange  the palm's actual path of the compensated / direct run
  grey boxes     the obstacles the planner went around (the route's prep boxes; not in the simulation, which is open floor)
Numbers on screen: the palm's distance now to the wanted path, and at the end the per-pair errors of via_comp_eval.py
(comp_pairs.npz: palm's error against the wanted sweep, palm's settled end against the wanted end).

pick: the cases by a fixed rule, not by look -- among first-pass pairs (from the stand), per route kind (FreshEdge,
FreshBox), the pair whose drop of the palm's settled-end error is the median of that kind, and the one at its 25 %
level (a less good case). A reset index (reset % 100000) renders that one.
-> <out dir>/src/<tag>.mp4, preview/<tag>.mp4 (<= 500 KB), preview/<tag>.jpg (the last frame), cases.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import render as RD  # noqa: E402
from arm_fk import ArmFK  # noqa: E402

DD, DC, RT, SCENE, OUT = (Path(a) for a in sys.argv[1:6])
TITLE = sys.argv[6]
WHICH = sys.argv[7:]
FONT = os.environ.get(
    "VIDEO_FONT", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_bench_20261119/tabletop_films_20261120/font.ttc"
)
WM_, HM, WC, HC, TB, LG = 640, 480, 480, 480, 40, 150
W = WM_ + WC
FPS, HZ = 25, 10
CYAN, ORANGE, GREEN, VIOLET, BOX = (
    (0.05, 0.75, 0.85),
    (1.0, 0.5, 0.1),
    (0.15, 0.8, 0.25),
    (0.6, 0.35, 1.0),
    (0.55, 0.5, 0.45),
)
C8 = lambda c: tuple(int(255 * v) for v in c)  # noqa: E731
KIND = {"FreshEdge": "越过桌沿", "FreshBox": "绕过桌上的箱子"}


def pelvis_R(yaw, g):
    pitch, roll = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))  # fmt: skip


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Run:
    def __init__(self, D: Path):
        self.z = dict(np.load(D / "all.npz"))
        self.X = np.load(D / "arm.npz")["X"]
        self.link = [str(x) for x in np.load(D / "arm.npz")["link"]]
        self.names = json.load(open(D / "routes.json"))["names"]
        n = self.z["n_steps"].astype(int)
        self.off = np.r_[0, np.cumsum(n)]
        self.k_of = {int(r) % 100000: k for k, r in enumerate(self.z["reset"])}


FK = ArmFK(SCENE)
m = FK.m
A, B = Run(DD), Run(DC)
PALM = A.link.index("hand_palm_link")
pairs = dict(np.load(DC / "comp_pairs.npz"))


def qpos_seq(R: Run, i: int) -> np.ndarray:
    z = R.z
    side = "right" if z["side"][i] > 0 else "left"
    out = []
    for k in range(R.off[i], R.off[i + 1]):
        FK.spheres(side, z["W"][k, :3], quat_of(pelvis_R(float(z["W"][k, 3]), z["S"][k, 58:61].astype(float))),
                   z["S"][k, :29], "meta", z["hand_q"][i])  # fmt: skip
        out.append(FK.d.qpos.copy())
    return np.array(out)


def interp(Q: np.ndarray, t: float) -> np.ndarray:
    """qpos at time t (s) from the 10 Hz rows: linear, the root quaternion normalised."""
    x = min(max(t * HZ, 0.0), len(Q) - 1.0)
    k = min(int(x), len(Q) - 2)
    u = x - k
    q = (1 - u) * Q[k] + u * Q[k + 1]
    if Q[k][3:7] @ Q[k + 1][3:7] < 0:
        q[3:7] = (1 - u) * Q[k][3:7] - u * Q[k + 1][3:7]
    q[3:7] /= np.linalg.norm(q[3:7])
    return q


class Painter:
    def __init__(self):
        self.ds = [mujoco.MjData(m) for _ in range(2)]
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        self.opt.sitegroup[:] = 0  # the scene's com marker
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

    def ghost(self, scn, q, rgba):
        d = self.ds[1]
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(m, d, self.opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(n0, scn.ngeom):
            g = scn.geoms[i]
            keep = g.objtype == mujoco.mjtObj.mjOBJ_GEOM and self.robot[m.geom_bodyid[g.objid]]
            g.rgba[:] = rgba if keep else (0, 0, 0, 0)
            g.matid = -1

    @staticmethod
    def geom(scn, typ, size, pos, mat, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], typ, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).reshape(-1), np.asarray(rgba, np.float32))  # fmt: skip
        scn.ngeom += 1

    def sphere(self, scn, pos, r, rgba):
        self.geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], pos, np.eye(3), rgba)

    @staticmethod
    def line(scn, a, b, w, rgba):
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(np.asarray(b) - np.asarray(a)) < 1e-6:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                            np.asarray(rgba, np.float32))  # fmt: skip
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    def poly(self, scn, P, w, rgba):
        for a, b in zip(P[:-1], P[1:]):
            self.line(scn, a, b, w, rgba)

    @staticmethod
    def project(scn, W_, H_, p):
        c0, c1 = scn.camera[0], scn.camera[1]
        pos = 0.5 * (np.array(c0.pos) + np.array(c1.pos))
        fwd, up = np.array(c0.forward), np.array(c0.up)
        right = np.cross(fwd, up)
        v = np.asarray(p, float) - pos
        zz = float(v @ fwd)
        if zz <= 1e-6:
            return None
        n = c0.frustum_near
        x, y = float(v @ right) * n / zz, float(v @ up) * n / zz
        cx, hw, bt, tp = c0.frustum_center, c0.frustum_width, c0.frustum_bottom, c0.frustum_top
        if hw == 0:
            hw = 0.5 * (tp - bt) * W_ / H_
        return ((x - (cx - hw)) / (2 * hw) * W_, (tp - y) / (tp - bt) * H_)


def seg_dist(p, P):
    """Distance of point p to the polyline P."""
    a, b = P[:-1], P[1:]
    ab = b - a
    u = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + u[:, None] * ab), axis=1).min())


def pick() -> list[tuple[int, str]]:
    walked_b = B.z["walked"].astype(bool)
    kind, reset = pairs["kind"], pairs["reset"] % 100000
    first = np.array([not walked_b[B.k_of[int(r)]] for r in reset])
    drop = pairs["end_direct"] - pairs["end_comp"]
    out = []
    for kd in ("FreshEdge", "FreshBox"):
        idx = np.where((kind == kd) & first)[0]
        if not len(idx):
            continue
        order = idx[np.argsort(drop[idx], kind="stable")]
        for q, why in (
            (0.5, "按规则挑的：这类路线里改进幅度居中的一次"),
            (0.25, "按规则挑的：这类路线里改进幅度偏小的一次"),
        ):
            r = int(reset[order[int(round(q * (len(order) - 1)))]])
            if r not in [x for x, _ in out]:
                out.append((r, why))
    return out


def render(P: Painter, r: int, why: str) -> dict:
    i, j = A.k_of[r], B.k_of[r]
    za, zb = A.z, B.z
    assert np.allclose(za["start"][i], zb["start"][j]), f"reset {r}: the two runs did not start from the same state"
    assert A.names[za["via_route"][i]] == B.names[zb["via_route"][j]]
    name = B.names[zb["via_route"][j]]
    p0, yaw0 = zb["start"][j][:3].astype(float), float(zb["start"][j][3])
    Rh = rot_z(yaw0)
    to_w = lambda x: np.c_[p0[:2] + np.atleast_2d(x)[:, :2] @ Rh[:2, :2].T, np.atleast_2d(x)[:, 2]]  # noqa: E731
    want = to_w(zb["comp_ref0"][j][:, PALM])
    sent = to_w(zb["comp_ref"][j][:, PALM])
    pa = to_w(A.X[A.off[i] : A.off[i + 1], PALM])
    pb = to_w(B.X[B.off[j] : B.off[j + 1], PALM])
    pr = np.load(RT / "prep" / f"prep_{name}.npz")
    o, Rp = pr["pelvis_p"], pr["pelvis_R"]
    Rhp = rot_z(math.atan2(Rp[1, 0], Rp[0, 0]))
    boxes = []
    for bx in pr["boxes"]:
        cen_h = np.r_[(bx[:2] - o[:2]) @ Rhp[:2, :2], bx[2]]
        boxes.append((to_w(cen_h)[0], bx[3:], Rh @ Rhp.T))
    Qa, Qb = qpos_seq(A, i), qpos_seq(B, j)
    k_pair = int(np.where(pairs["reset"] % 100000 == r)[0][0])
    e = {k: float(pairs[k][k_pair]) for k in ("palm_direct", "palm_comp", "end_direct", "end_comp")}
    goal = want[-1]
    sd = 1.0 if zb["side"][j] > 0 else -1.0
    mid = 0.5 * (np.r_[p0[:2], 0.75] + goal)
    cam_m = P.cam(mid, 2.2, math.degrees(yaw0) + 70 * sd, -28)  # below -22.5 deg no sky (the scene has no skybox)
    cam_c = P.cam(
        goal, 0.75, math.degrees(yaw0) + 180 - 40 * sd, -35
    )  # from in front of the robot: the arm comes toward it
    big, small, tiny = (ImageFont.truetype(FONT, s) for s in (17, 15, 13))
    T = max(len(Qa), len(Qb)) / HZ
    first_pass = not bool(zb["walked"][j])
    title = f"{TITLE}　{KIND.get(name.split('_')[0], name)}，{'从站立出发' if first_pass else '走过去之后'}，{'右' if sd > 0 else '左'}手（{why}）"

    def draw(t: float, phase: str) -> np.ndarray:
        qa, qb = interp(Qa, t), interp(Qb, t)
        ka, kb = min(int(t * HZ), len(pa) - 1), min(int(t * HZ), len(pb) - 1)
        imgs = []
        for which, (rr, cam, W_, H_) in enumerate(((P.rm, cam_m, WM_, HM), (P.rc, cam_c, WC, HC))):
            d = P.ds[0]
            d.qpos[:] = qb
            mujoco.mj_forward(m, d)
            rr.update_scene(d, camera=cam, scene_option=P.opt)
            scn = rr.scene
            if which == 1:
                for gi in range(scn.ngeom):
                    gg = scn.geoms[gi]
                    if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and P.robot[m.geom_bodyid[gg.objid]]:
                        gg.rgba[3] = 0.45
            if phase != "plan":
                P.ghost(scn, qa, (*ORANGE, 0.16 if which else 0.30))
            for c, h, Rb in boxes:
                P.geom(scn, mujoco.mjtGeom.mjGEOM_BOX, h, c, Rb, (*BOX, 0.28))
            P.poly(scn, want, 0.004 if which else 0.006, (*VIOLET, 0.95))
            P.sphere(scn, goal, 0.012, (*GREEN, 1.0))
            for s in sent:
                P.sphere(scn, s, 0.005 if which else 0.007, (*CYAN, 0.75))
            if phase != "plan":
                P.poly(scn, pa[: ka + 1], 0.003 if which else 0.005, (*ORANGE, 0.95))
                P.poly(scn, pb[: kb + 1], 0.003 if which else 0.005, (*CYAN, 1.0))
            img = Image.fromarray(rr.render())
            dr = ImageDraw.Draw(img)
            dr.text((8, 6), "目标附近放大" if which else "全景", font=small, fill=(20, 20, 20), stroke_width=2,
                    stroke_fill=(255, 255, 255))  # fmt: skip
            if which == 1:
                for pos, txt, col in ((goal, "想要的终点", C8(GREEN)), (sent[-1], "补偿后发出的终点", C8(CYAN))):
                    px = P.project(scn, W_, H_, pos)
                    if px is not None and 0 <= px[0] < W_ - 60 and 0 <= px[1] < H_:
                        dr.text((px[0] + 9, px[1] - 9 + (16 if col == C8(CYAN) else 0)), txt, font=tiny, fill=col,
                                stroke_width=2, stroke_fill=(255, 255, 255))  # fmt: skip
            imgs.append(np.asarray(img))
        canvas = Image.new("RGB", (W, TB + HM + LG), (248, 248, 246))
        canvas.paste(Image.fromarray(np.concatenate(imgs, 1)), (0, TB))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle([0, 0, W, TB], fill=(24, 28, 34))
        dr.text((10, 10), title, font=big, fill=(240, 240, 240))
        y0 = TB + HM
        rows = ((pb, kb, "实体：用自身模型预先挪过的命令（只挪位置）", C8(CYAN), "palm_comp", "end_comp"),
                (pa, ka, "橙色影子：直接把规划的路线发给控制器", C8(ORANGE), "palm_direct", "end_direct"))  # fmt: skip
        for n_, (pp, kk, nm, col, k_path, k_end) in enumerate(rows):
            y = y0 + 10 + 40 * n_
            dr.rectangle([10, y + 3, 24, y + 17], fill=col)
            now = "还没发命令" if phase == "plan" else f"手掌离想要的路线 {100 * seg_dist(pp[kk], want):.1f} cm"
            dr.text((32, y), f"{nm}　{now}", font=small, fill=(25, 25, 25))
            if phase == "end":
                dr.text((32, y + 19), f"全程手掌离想要的路线最远 {100 * e[k_path]:.1f} cm；停稳后离想要的终点 {100 * e[k_end]:.1f} cm",
                        font=tiny, fill=(70, 70, 70))  # fmt: skip
        dr.text((10, y0 + 96), "紫线：规划器想让手掌走的路线，绿点是终点；青色小点：补偿后实际发出的命令，故意朝预测偏差的反方向挪",
                font=tiny, fill=(110, 110, 110))  # fmt: skip
        dr.text((10, y0 + 114), "灰色方块：规划时绕开的障碍（仿真里没有放）；青线、橙线：两遍手掌实际走过的路线", font=tiny,
                fill=(110, 110, 110))  # fmt: skip
        dr.text((W - 330, y0 + 132), "SONIC 控制器冻结不动；两遍从同一状态出发", font=tiny, fill=(140, 140, 140))
        return np.asarray(canvas)

    frames = [draw(0.0, "plan")] * int(1.6 * FPS)
    for f in range(int(T * FPS)):
        frames.append(draw(f / FPS, "run"))
    end = draw(T, "end")
    frames += [end] * int(2.5 * FPS)
    for sub in ("src", "preview"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    tag = f"sonic_{name}_r{r}"
    src = OUT / "src" / f"{tag}.mp4"
    RD.write(frames, src, fps=FPS)
    nb = RD.preview(src, OUT / "preview" / f"{tag}.mp4")
    Image.fromarray(end).save(OUT / "preview" / f"{tag}.jpg", quality=88)
    print(f"{tag}: {len(frames)} frames, preview {nb} bytes; palm path {100 * e['palm_direct']:.1f} -> {100 * e['palm_comp']:.1f} cm, "
          f"end {100 * e['end_direct']:.1f} -> {100 * e['end_comp']:.1f} cm ({why})", flush=True)  # fmt: skip
    return dict(reset=r, route=name, why=why, tag=tag, first_pass=first_pass, **e)


if __name__ == "__main__":
    P = Painter()
    if WHICH == ["pick"]:
        todo = pick()
    else:
        todo = [(int(a), "指定") for a in WHICH]
    out = [render(P, r, w) for r, w in todo]
    json.dump(out, open(OUT / "cases.json", "w"), indent=1, ensure_ascii=False)

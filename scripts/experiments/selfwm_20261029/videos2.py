"""Videos and images for steps 1-2 (outputs/selfwm_20261029/videos/: src/ recordings, preview/ <= 500 KB copies).

wm2_envelope     three new random-command episodes (seeds outside the data), simulator left, world model right:
                 right hand level after a walk, left hand level, right hand straight down
step2_side_yaw3  A / B / C / R on eval2 cell side4_*_yaw3_0 (6 cm box, 0.75 m table, camera yaw 3 deg)
step2_down       A / B / C / R on down1_*_none_0 (straight-down pinch, 0.60 m table)
step2_low        A / B / C / R on side3_*_none_0 (6 cm box on the 0.65 m table: where B fails)
grasp_frames.png the grasp frame (approach, closing, third axis) of a side grasp and a pinch

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/selfwm_20261029/videos2.py all
"""

from __future__ import annotations

import json
import math
import sys

import env as E
import eval2 as EV2
import gen_data2 as G2
import numpy as np
import render as RD
import torch
import wm as WMM
import wm2 as W2
from PIL import Image

import videos as V

VID = E.OUT / "videos"
GREEN, RED = V.GREEN, V.RED
BLUE = V.BLUE
AX = {"approach": (0.95, 0.55, 0.1, 1), "closing": (0.65, 0.3, 0.9, 1), "third": (0.55, 0.55, 0.55, 1)}
ATT_NAME = {"level": "侧抓", "tilt45": "斜 45 度", "down": "垂直往下"}


def find_seed(rows, want, start=9_000_000):
    """First seed whose episode draws (gen_data2.episode's first three draws) give `want` = (side, att, walked)."""
    for seed in range(start, start + 100000):
        rng = np.random.default_rng(seed)
        r = rows[rng.integers(len(rows))]
        side = "right" if int(r["hand"]) == 1 else "left"
        att = (
            ("down" if rng.random() < 0.6 else "tilt45")
            if r["pinch"]
            else ("level" if rng.random() < 0.9 else "tilt45")
        )
        walked = bool(rng.random() < 0.4)
        if (side, att, walked) == want:
            return seed
    raise RuntimeError(want)


def wm2_envelope() -> dict:
    env = E.Env("open")
    rec = RD.Recorder(env.m)
    dev = "cuda"
    model, _ = W2.load(dev)
    rows = G2.anchors()
    wants = [("right", "level", True), ("left", "level", False), ("right", "down", False)]
    panels, meta = [], []
    for want in wants:
        seed = find_seed(rows, want)
        rng = np.random.default_rng(seed)
        mark0 = rec.mark()
        ep = G2.episode(env, rng, rows)
        assert ep is not None
        S, A, W = ep["S"], ep["A"], ep["W"]
        n_ticks = 5 * len(A)
        real = rec.rows[-n_ticks:]
        t = lambda x: torch.tensor(np.array(x, np.float32), device=dev)  # noqa: E731
        with torch.no_grad():
            ps, po = model.rollout(t(S[0])[None], t(A[0])[None], t(S[1])[None], t(A[1:])[None])
            side = t(A[1:, 0])[None]
            pj, _ = model.hands(ps[..., :29], side)
            tj, _ = model.hands(t(S[2:, :29])[None], side)
        ps, po = ps[0].cpu().numpy(), po[0].cpu().numpy()
        err = (pj - tj)[0].norm(dim=-1).cpu().numpy() * 100
        template = real[0]
        pred10 = [real[4], real[9]]
        p, yaw = W[1][:3].astype(float).copy(), float(W[1][3])
        jaw_pred, jaw_real = [], []
        jcol = slice(4, 7) if want[0] == "right" else slice(7, 10)
        for h in range(len(po)):
            p = p + E.rot_z(yaw) @ po[h, :3]
            yaw += float(po[h, 3])
            R = E.rot_z(yaw) @ WMM.tilt_rot(ps[h, 58:61].astype(float))
            pred10.append(V.predicted_qpos(template, env.sim, p, R, ps[h, :29]))
        # jaw positions for the trails, world: the real one from W, the predicted one from FK of the predicted joints
        q_pred = torch.tensor(ps[:, :29], device=dev)
        out = model.fk(q_pred)
        jp = (out["jaw_right"][0] if want[0] == "right" else out["jaw_left"][0]).cpu().numpy()
        p, yaw = W[1][:3].astype(float).copy(), float(W[1][3])
        for h in range(len(po)):
            p = p + E.rot_z(yaw) @ po[h, :3]
            yaw += float(po[h, 3])
            R = E.rot_z(yaw) @ WMM.tilt_rot(ps[h, 58:61].astype(float))
            jaw_pred.append(p + R @ jp[h])
        jaw_real = [w[jcol] for w in W[2:]]
        n = len(real) // 2
        pred = V.interp(pred10, n)
        ancr = ep["anchor"]
        ryaw = math.degrees(float(W[0][3]))
        az = ryaw + (-120.0 if want[0] == "right" else 120.0)
        view = dict(
            lookat=tuple(ancr + np.array([0.0, 0.0, 0.02])),
            azimuth=az,
            elevation=-35.0,
            distance=0.85,
            width=320,
            height=240,
        )
        vl, vr = RD.View(env.sim.m, **view), RD.View(env.sim.m, **view)
        label = f"{'右手' if want[0] == 'right' else '左手'}，{ATT_NAME[want[1]]}" + ("，刚走完路" if want[2] else "")
        frames = []
        for i in range(n):
            k = min(int(i / 25 * 10), len(err) - 1)
            kk = max(k - 1, 0)
            tr = [(x, GREEN, 0.004) for x in jaw_real[: kk + 1 : 2]]
            tp = [(x, BLUE, 0.004) for x in jaw_pred[: kk + 1 : 2]]
            left = vl.frame(real[2 * i + 1], tr, "仿真：" + label, [f"t = {i / 25:.1f} s"])
            right = vr.frame(pred[i], tp + tr, "世界模型的想象", [f"想象到真实 {err[kk]:.2f} cm"])
            frames.append(np.concatenate([left, right], 1))
        panels.append(frames)
        meta.append(
            dict(
                seed=seed,
                want=list(want),
                err_median_cm=round(float(np.median(err)), 3),
                err_max_cm=round(float(err.max()), 3),
            )
        )
    n = min(len(f) for f in panels)
    frames = [np.concatenate([p[i] for p in panels], 0) for i in range(n)]
    RD.write(frames, VID / "src" / "wm2_envelope.mp4")
    return dict(episodes=meta, seconds=round(n / 25, 2))


NAMES = {"A": "A 世界坐标开环", "B": "B 相对伺服", "C": "C 自由策略", "R": "R 伺服加残差"}


def step2_grid(r: EV2.Runner2, rec: RD.Recorder, cell_prefix: str, name: str, look) -> dict:
    env = r.env
    bq = V.box_qadr(env)
    runs, out = [], {}
    for m in ("A", "B", "C", "R"):
        c = next(x for x in EV2.cells() if x["id"] == cell_prefix.replace("*", m))
        mark = rec.mark()
        res = r.run(c)
        runs.append((m, rec.rows[mark:], dict(r.last), res))
        out[m] = {k: res.get(k) for k in ("lifted", "knocked_cm", "close_err_cm", "steps")}
    n = max(len(rows) for _, rows, _, _ in runs) // 2 + 40
    views = [RD.View(env.sim.m, **look) for _ in runs]
    frames = []
    for i in range(n):
        panel = []
        for v, (m, rows, last, res) in zip(views, runs):
            q = rows[min(2 * i + 1, len(rows) - 1)]
            grasp_now = q[bq : bq + 3] + last["off"]
            ok = res["lifted"] and res["knocked_cm"] <= 2.0
            tail = [f"t = {min(i, len(rows) // 2) / 25:.1f} s"]
            if i >= len(rows) // 2 - 1:
                tail.append(
                    ("成功" if ok else "失败")
                    + f"：推动盒子 {res['knocked_cm']:.1f} cm，"
                    + ("抬起" if res["lifted"] else "没抬起")
                )
            panel.append(v.frame(q, [(grasp_now, GREEN, 0.008), (last["g_bel_w"], RED, 0.008)], NAMES[m], tail))
        frames.append(RD.tile(panel, 2))
    RD.write(frames, VID / "src" / f"{name}.mp4")
    return dict(cell=cell_prefix, seconds=round(n / 25, 2), runs=out)


def grasp_frames(r: EV2.Runner2) -> dict:
    """A still of each grasp frame at its pre-grasp: approach (orange), closing (purple), third (grey) axes at the grasp."""
    env, m = r.env, r.env.m
    imgs = []
    for cid, title in (("side4_B_none_0", "侧抓的抓取坐标系"), ("down1_B_none_0", "垂直下夹的抓取坐标系")):
        c = next(x for x in EV2.cells() if x["id"] == cid)
        b = (EV2.SIDE_BOXES if c["kind"] == "side" else EV2.DOWN_BOXES)[c["box"]]
        cell = dict(
            E.BOX, width=b["width"], depth=b["width"] if c["kind"] == "down" else 0.06, height=b["height"], top=b["top"]
        )
        env.probe._size(cell)
        m.reset((0.0, 0.0), 0.0)
        centre = np.array([r.cap.EDGE_M + cell["inboard"], -r.cap.LATERAL_M, cell["top"] + cell["height"] / 2 + 0.001])
        m.place("probe_object", centre, np.array([1.0, 0.0, 0.0, 0.0]))
        m.hold(1.0)
        box0 = env.sim.d.xpos[env.box].copy()
        grasp, rot, tol, plane_z, pre, via, _ = r.geometry(c, box0)
        m.reach(E.SIDE, pre, rot, via=via)
        q = env.sim.d.qpos.copy()
        look = dict(
            lookat=tuple(0.5 * (grasp + env.jaw_world())),
            azimuth=-125.0,
            elevation=-30.0,
            distance=0.55,
            width=480,
            height=360,
        )
        v = RD.View(env.sim.m, **look)
        jaw = env.jaw_world()  # the frame drawn on the hand at its pre-grasp; the green dot is the grasp point
        arrows = [
            (jaw, rot[:, 0], 0.2, AX["approach"]),
            (jaw, rot[:, 1], 0.15, AX["closing"]),
            (jaw, rot[:, 2], 0.15, AX["third"]),
        ]
        imgs.append(v.frame(q, [(grasp, GREEN, 0.008), (jaw, (1, 1, 1, 1), 0.006)], title,
                            ["橙：接近方向 x", "紫：闭合方向 y", "灰：第三方向 z", "绿点：抓取点"], arrows=arrows))  # fmt: skip
    Image.fromarray(np.concatenate(imgs, 1)).save(VID / "preview" / "grasp_frames.png")
    return dict(files=["grasp_frames.png"])


def main(argv=None) -> int:
    what = (argv or sys.argv[1:] or ["all"])[0]
    (VID / "src").mkdir(parents=True, exist_ok=True)
    (VID / "preview").mkdir(parents=True, exist_ok=True)
    meta = {}
    if what in ("all", "wm"):
        meta["wm2_envelope"] = wm2_envelope()
    if what in ("all", "step2", "frames"):
        r = EV2.Runner2()
        V.translucent_box(r.env)
        rec = RD.Recorder(r.env.m)
        if what in ("all", "frames"):
            meta["grasp_frames"] = grasp_frames(r)
        if what in ("all", "step2"):
            look = dict(lookat=(0.31, -0.15, 0.84), azimuth=-120.0, elevation=-42.0, distance=0.62)
            look_low = dict(lookat=(0.31, -0.15, 0.74), azimuth=-120.0, elevation=-40.0, distance=0.70)
            look_down = dict(lookat=(0.36, -0.15, 0.70), azimuth=-120.0, elevation=-30.0, distance=0.62)
            meta["step2_side_yaw3"] = step2_grid(r, rec, "side4_*_yaw3_0", "step2_side_yaw3", look)
            meta["step2_down"] = step2_grid(r, rec, "down1_*_none_0", "step2_down", look_down)
            meta["step2_low"] = step2_grid(r, rec, "side3_*_none_0", "step2_low", look_low)
    for name in meta:
        if (VID / "src" / f"{name}.mp4").exists():
            meta[name]["preview_bytes"] = RD.preview(VID / "src" / f"{name}.mp4", VID / "preview" / f"{name}.mp4")
    old = json.loads((VID / "videos.json").read_text()) if (VID / "videos.json").exists() else {}
    old.update(meta)
    (VID / "videos.json").write_text(json.dumps(old, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

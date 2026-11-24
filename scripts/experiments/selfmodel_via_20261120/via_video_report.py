"""selfmodel_via_20261120: side-by-side videos of recorded reaches, for the report of 2026-11-23.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 python via_video_report.py <out dir> [clip name ...]      # from the repo root

Nothing is simulated here: every frame is the robot posed from a replay's recorded state (all.npz: the pelvis W[:3], its
yaw W[3], roll and pitch from the gravity direction S[58:61], the 29 joints S[:29], the hand's fingers), 10 Hz ticks
interpolated to 30 frames a second. A panel shows one recorded reach: the route's boxes drawn where the route was planned
around them (in the open-floor replays they were NOT in the simulation; in the last clip they were), the wanted path of the
palm (green), the palm's trail so far, and the smallest clearance of the arm's 24 spheres to the boxes at this tick
(via_scene.py's clearance). The strip below plots that clearance over time for both panels.
Clips and how each reach was chosen are in CLIPS; the choice rules are repeated in the report.
"""

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
import via_model as VM  # noqa: E402
import via_scene as VS  # noqa: E402

I = REPO / "outputs" / "reachwm_20261030"
SCENE = I / "scene" / "simple_g1_sonic" / "scene_locomani.xml"
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
PW, PH, TB, LG, FPS, SUB = 640, 500, 46, 174, 30, 3  # panel, title bar, log strip; 3 frames a 10 Hz tick
ORANGE, BLUE, GREEN, INK, MUTED = (232, 89, 12), (28, 126, 214), (47, 158, 68), (33, 37, 41), (110, 117, 125)
SCR = Path(
    "/tmp/claude-1002/-workspace-code-locomanipulation/92eebc3c-7717-4ad1-8cc1-fc959a0e4c1f/scratchpad/obst_smoke"
)

CLIPS = {
    # ScaleBFM, fourth test routes. Of the 152 reaches where only the direct command touches a box (by the 24 spheres), the
    # one in the middle by depth.
    "scalebfm_touch": dict(
        title="ScaleBFM：手要越过桌上的挡板去够目标。同一条路线、同一个起点，只差发出去的命令",
        routes=I / "test4_routes" / "detour",
        panels=[
            dict(run=I / "test4_direct", reset=277, name="直接发规划好的命令", short="直接发命令", col=ORANGE),
            dict(run=I / "test4_point", reset=277, name="先用自身模型挪命令，再发", short="挪过的命令", col=BLUE),
        ],
    ),  # fmt: skip
    # The reach closest to the medians of both runs (clearance lost and end error, each in units of its spread).
    "scalebfm_typical": dict(
        title="ScaleBFM：最接近中位数的一次伸手（越过桌沿）",
        routes=I / "test4_routes" / "detour",
        panels=[
            dict(run=I / "test4_direct", reset=492, name="直接发规划好的命令", short="直接发命令", col=ORANGE),
            dict(run=I / "test4_point", reset=492, name="先用自身模型挪命令，再发", short="挪过的命令", col=BLUE),
        ],
    ),  # fmt: skip
    # Of the 8 reaches where only the compensated run touches, the deepest.
    "scalebfm_counter": dict(
        title="ScaleBFM：反过来的例子。516 次里有 8 次只有挪过命令的那次碰到，这是最深的一次",
        routes=I / "test4_routes" / "detour",
        panels=[
            dict(run=I / "test4_direct", reset=183, name="直接发规划好的命令", short="直接发命令", col=ORANGE),
            dict(run=I / "test4_point", reset=183, name="先用自身模型挪命令，再发", short="挪过的命令", col=BLUE),
        ],
    ),  # fmt: skip
    # SONIC on the first clip's route and start.
    "sonic_same": dict(
        title="换成 SONIC，同一条路线：直接发命令时手停在离目标 5 cm 的地方，也离挡板更远",
        routes=I / "test4_routes" / "detour",
        panels=[
            dict(run=I / "test4_direct_sonic", reset=277, name="直接发规划好的命令", short="直接发命令", col=ORANGE),
            dict(run=I / "test4_point_sonic", reset=277, name="先用自身模型挪命令，再发", short="挪过的命令", col=BLUE),
        ],
    ),  # fmt: skip
    # AMO, random obstacle-free paths (standing, palm path over 1 m): the middle one by pelvis travel among those that
    # walked off (over 20 cm) and among those that stayed.
    "amo_walk": dict(
        title="AMO：同一类伸手命令有两种结果。走开的占两成到三成多",
        routes=None,
        wide=True,
        panels=[
            dict(run=I / "data_via_amo", seg=22088, name="站在原地", short="站在原地", col=BLUE),
            dict(run=I / "data_via_amo", seg=21650, name="伸到一半自己走开", short="自己走开", col=ORANGE),
        ],
    ),  # fmt: skip
    # The set-up trial of the next experiment: one throwaway scene, the board really in the simulation, no margin.
    "real_box_trial": dict(
        title="下一步的试跑：桌板真的放进仿真里，规划时不留余量。只有一个场景，不是结果",
        routes=SCR / "routes",
        real=True,
        panels=[
            dict(run=SCR / "run_d", reset=0, name="直接发规划好的命令", short="直接发命令", col=ORANGE),
            dict(run=SCR / "run_p", reset=0, name="先用自身模型挪命令，再发", short="挪过的命令", col=BLUE),
        ],
    ),  # fmt: skip
}


def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Robot:
    def __init__(self):
        from locomani.backends.scalebfm import runtime as RT

        spec = mujoco.MjSpec.from_file(str(SCENE))
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.93, 0.95, 0.97], rgb2=[0.74, 0.79, 0.85], width=256, height=1536)  # fmt: skip
        spec.visual.global_.offwidth, spec.visual.global_.offheight = PW, PH
        self.m = m = spec.compile()
        for g in range(m.ngeom):  # collision shapes of the robot out of sight; the floor stays
            if m.geom_group[g] == 0 and m.geom_bodyid[g] != 0:
                m.geom_group[g] = 4
        self.d = mujoco.MjData(m)
        adr = lambda names: [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]  # noqa: E731
        self.q_meta = adr(json.load(open(RT.META))["joint_names"])
        self.q_hand = adr([f"right_hand_{f}_joint" for f in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")])  # fmt: skip
        self.r = mujoco.Renderer(m, height=PH, width=PW, max_geom=m.ngeom + 2000)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0  # the scene's red centre-of-mass marker

    def pose(self, W, S, hand) -> None:
        g = S[58:61].astype(float)
        pitch, roll = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
        cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        R = (
            rz(float(W[3]))
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        )
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(9))
        d = self.d
        d.qpos[:] = 0
        d.qpos[:3], d.qpos[3:7] = W[:3], q
        d.qpos[self.q_meta] = S[:29]
        d.qpos[self.q_hand] = hand
        mujoco.mj_forward(self.m, d)

    def shot(self, cam, boxes=(), lines=(), balls=()) -> np.ndarray:
        """boxes: (centre, 3x3, half, rgba); lines: (points (n, 3), rgba, radius); balls: (centre, rgba, radius)."""
        self.r.update_scene(self.d, camera=cam, scene_option=self.opt)
        sc = self.r.scene
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0  # a translucent box would cast a solid shadow

        def add(kind, size, pos, mat, rgba):
            if sc.ngeom < sc.maxgeom:
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], kind, np.asarray(size, float), np.asarray(pos, float),
                                    np.asarray(mat, float).reshape(-1), np.asarray(rgba, np.float32))  # fmt: skip
                sc.ngeom += 1

        for c, R, h, rgba in boxes:
            add(mujoco.mjtGeom.mjGEOM_BOX, h, c, R, rgba)
        for pts, rgba, rad in lines:
            for a, b in zip(pts[:-1], pts[1:]):
                if sc.ngeom < sc.maxgeom and np.linalg.norm(b - a) > 1e-5:
                    mujoco.mjv_connector(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, rad, a, b)
                    sc.geoms[sc.ngeom].rgba[:] = rgba
                    sc.ngeom += 1
        for c, rgba, rad in balls:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [rad, 0, 0], c, np.eye(3), rgba)
        return self.r.render()


def load(panel: dict, routes) -> dict:
    """One recorded reach: ticks' W, S, the hand, the frame of the command, the wanted palm path and the palm's trail
    (world), the boxes (centre, R, half; world), the clearance per tick (m; None without boxes), the end error (m)."""
    D = panel["run"]
    z, T, A = dict(np.load(D / "all.npz")), VM.load_table(D), np.load(D / "arm.npz")
    seg = int(panel["seg"]) if "seg" in panel else int(np.where(z["reset"] % 100000 == panel["reset"])[0][0])
    row = int(np.where(T["seg"] == seg)[0][0])
    cum = np.r_[0, np.cumsum(z["n_steps"])]
    a, b = int(cum[seg]), int(cum[seg + 1])
    p0, yaw0 = z["start"][seg][:3].astype(float), float(z["start"][seg][3])
    Rh = rz(yaw0)
    world = lambda x: np.c_[
        p0[:2] + np.asarray(x)[..., :2].reshape(-1, 2) @ Rh[:2, :2].T, np.asarray(x)[..., 2].reshape(-1)
    ]  # noqa: E731
    X = A["X"][a:b].astype(float)  # (ticks, 24, 3) in the heading frame of the command
    palm = int(np.where(T["palm"])[0][0])
    want = z["comp_ref0"][seg].astype(float) if "comp_ref0" in z else T["ref"][row].astype(float)
    out = dict(W=z["W"][a:b].astype(float), S=z["S"][a:b].astype(float), hand=z["hand_q"][seg].astype(float), p0=p0, yaw0=yaw0,
               want=world(want[:, palm]), trail=world(X[:, palm]), boxes=[], clear=None, want_clear=None,
               end=float(np.linalg.norm(X[-VM.TAIL :, palm].mean(0) - want[-1, palm])))  # fmt: skip
    if routes is not None:
        name = json.load(open(D / "routes.json"))["names"][int(z["via_route"][seg])]
        bx = np.load(Path(routes) / "prep" / f"prep_{name}.npz")["boxes"]
        rad = A["radius"].astype(float)
        out["boxes"] = [(world(c[None, :3])[0], Rh, c[3:]) for c in bx]
        out["clear"] = np.array([VS.clearance(x[None], bx, rad).min() for x in X])
        out["want_clear"] = float(VS.clearance(want, bx, rad).min())
        out["name"] = name
    if "gap_touch" in z:
        out["touch_t"] = float(z["gap_t"][seg][2]) if z["gap_touch"][seg][2] else None
        out["force"] = float(z["gap_force"][seg][2])
    return out


def text(dr, xy, s, font, fill=INK, halo=True):
    dr.text(xy, s, font=font, fill=fill, stroke_width=3 if halo else 0, stroke_fill=(255, 255, 255))


def clip(rb: Robot, name: str, spec: dict, out: Path) -> None:
    runs = [load(p, spec.get("routes")) for p in spec["panels"]]
    n = max(len(r["W"]) for r in runs)
    f_title, f_big, f_mid, f_small = (ImageFont.truetype(FONT, s) for s in (21, 20, 17, 14))
    cams = []
    for r in runs if spec.get("wide") else runs[:1] * len(runs):  # a wide clip frames each panel's own reach
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if spec.get("wide"):
            pts = np.r_[r["W"][:, :3], r["want"]]
            cam.lookat[:] = [*(0.5 * (pts[:, :2].min(0) + pts[:, :2].max(0))), 0.7]
            span = float(np.linalg.norm(pts[:, :2].max(0) - pts[:, :2].min(0)))
            cam.distance, cam.azimuth, cam.elevation = 2.3 + 0.7 * span, np.degrees(r["yaw0"]) + 125.0, -16.0
        else:
            mid = 0.5 * (r["want"].min(0) + r["want"].max(0))
            cam.lookat[:] = mid + np.array([0.0, 0.0, -0.04])
            cam.distance, cam.azimuth, cam.elevation = 1.05, np.degrees(r["yaw0"]) + 118.0, -12.0
        cams.append(cam)
    real = bool(spec.get("real"))
    W = 2 * PW
    hold0, hold1 = int(1.2 * FPS), int(2.5 * FPS)
    frames = [0] * hold0 + list(range((n - 1) * SUB + 1)) + [(n - 1) * SUB] * hold1
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{TB + PH + LG}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(out / f"{name}.mp4")], stdin=subprocess.PIPE)  # fmt: skip
    for k, f in enumerate(frames):
        done = k >= hold0 + (n - 1) * SUB
        img = Image.new("RGB", (W, TB + PH + LG), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, TB], fill=(248, 249, 250))
        dr.text((16, 11), spec["title"], font=f_title, fill=INK)
        for j, (r, p) in enumerate(zip(runs, spec["panels"])):
            t = min(f / SUB, len(r["W"]) - 1.0)
            i, u = int(np.floor(t)), t - np.floor(t)
            i2 = min(i + 1, len(r["W"]) - 1)
            Wt, St = (1 - u) * r["W"][i] + u * r["W"][i2], (1 - u) * r["S"][i] + u * r["S"][i2]
            Wt[3] = r["W"][i, 3] + u * ((r["W"][i2, 3] - r["W"][i, 3] + np.pi) % (2 * np.pi) - np.pi)
            rb.pose(Wt, St, r["hand"])
            ct = None if r["clear"] is None else (1 - u) * r["clear"][i] + u * r["clear"][i2]
            low = None if r["clear"] is None else float(r["clear"][: i2 + 1].min())
            hit = (r.get("touch_t") is not None and t / 10 >= r["touch_t"]) if real else (low is not None and low < 0)
            tint = (0.86, 0.2, 0.2) if hit else (0.72, 0.56, 0.38)
            alpha = [0.85, 0.9] if real else [0.4, 0.8]  # the board, the thin box on it
            col = np.array(p["col"]) / 255
            trail = (
                r["trail"][: i + 1]
                if i == i2
                else np.r_[r["trail"][: i + 1], [(1 - u) * r["trail"][i] + u * r["trail"][i2]]]
            )
            shot = rb.shot(cams[j], boxes=[(c, R, h, (*tint, alpha[q])) for q, (c, R, h) in enumerate(r["boxes"])],
                           lines=[(r["want"], (*np.array(GREEN) / 255, 1.0), 0.004), (trail, (*col, 1.0), 0.006)],
                           balls=[(r["want"][-1], (*np.array(GREEN) / 255, 1.0), 0.012)])  # fmt: skip
            img.paste(Image.fromarray(shot), (j * PW, TB))
            x0 = j * PW + 14
            dr.rectangle([x0 - 4, TB + 10, x0 + 12 + f_big.getlength(p["name"]), TB + 42], fill=p["col"])
            dr.text((x0 + 4, TB + 13), p["name"], font=f_big, fill=(255, 255, 255))
            y = TB + 52
            if spec.get("wide"):
                text(dr, (x0, y), f"骨盆离开原地 {100 * np.linalg.norm(Wt[:2] - r['W'][0, :2]):.0f} cm", f_mid)
            elif real:
                text(dr, (x0, y), "碰到桌板" if hit else "没有碰到", f_mid, fill=(201, 42, 42) if hit else INK)
            else:
                text(dr, (x0, y), f"手臂离障碍物 {100 * ct:.1f} cm" + ("，已重叠" if ct < 0 else ""), f_mid,
                     fill=(201, 42, 42) if ct < 0 else INK)  # fmt: skip
            if done and not spec.get("wide"):
                text(dr, (x0, y + 24), f"停稳后离想要的终点 {100 * r['end']:.1f} cm", f_mid)
                if low is not None and not real:
                    text(
                        dr, (x0, y + 48), f"全程最近 {100 * low:.1f} cm，规划留了 {100 * r['want_clear']:.1f} cm", f_mid
                    )
                if real and r.get("force"):
                    text(dr, (x0, y + 48), f"最大接触力 {r['force']:.0f} N", f_mid)
        dr.line([PW, TB, PW, TB + PH], fill=(255, 255, 255), width=3)
        # the strip: clearance (or pelvis travel) over time
        gx0, gx1, gy0, gy1 = 70, W - 250, TB + PH + 18, TB + PH + LG - 30
        wide = bool(spec.get("wide"))
        series = [
            100 * np.linalg.norm(r["W"][:, :2] - r["W"][0, :2], axis=1) if wide else 100 * r["clear"] for r in runs
        ]
        lo = 0.0 if wide else min(-1.0, min(s.min() for s in series))
        hi = max(s.max() for s in series) if wide else min(12.0, max(s.max() for s in series))
        sx = lambda q: gx0 + (gx1 - gx0) * q / max(n - 1, 1)  # noqa: E731
        sy = lambda v: gy1 - (gy1 - gy0) * (np.clip(v, lo, hi) - lo) / (hi - lo)  # noqa: E731
        dr.line([gx0, gy0, gx0, gy1], fill=(206, 212, 218))
        for v in (lo, 0.0, hi):
            dr.line([gx0, sy(v), gx1, sy(v)], fill=(134, 142, 150) if v == 0 else (233, 236, 239))
            if v == 0 or abs(sy(v) - sy(0.0)) > 16:
                dr.text((gx0 - 44, sy(v) - 8), f"{v:.0f} cm", font=f_small, fill=MUTED)
        dr.text((gx0, gy1 + 6), "0 s", font=f_small, fill=MUTED)
        dr.text((gx1 - 28, gy1 + 6), f"{(n - 1) / 10:.0f} s", font=f_small, fill=MUTED)
        dr.text((gx1 + 14, gy0 - 4), "骨盆离开原地的距离" if wide else "手臂离障碍物的最小间隙", font=f_small, fill=INK)
        if not wide:
            dr.text((gx1 + 14, sy(0.0) - 8), "0 以下：和障碍物重叠", font=f_small, fill=MUTED)
        tnow = min(f / SUB, n - 1)
        for s, p in zip(series, spec["panels"]):
            m = int(min(np.floor(tnow), len(s) - 1))
            pts = [(sx(q), sy(s[q])) for q in range(m + 1)]
            if len(pts) > 1:
                dr.line(pts, fill=p["col"], width=3)
            dr.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4], fill=p["col"])
        for q, p in enumerate(spec["panels"]):
            dr.rectangle([gx1 + 14, gy0 + 26 + 22 * q, gx1 + 26, gy0 + 38 + 22 * q], fill=p["col"])
            dr.text((gx1 + 32, gy0 + 22 + 22 * q), p.get("short", p["name"]), font=f_small, fill=INK)
        ff.stdin.write(np.asarray(img).tobytes())
        if k in (hold0 // 2, len(frames) - 5):
            img.save(out / f"{name}_{'start' if k < hold0 else 'end'}.png")
    ff.stdin.close()
    ff.wait()
    print(f"{name}: {len(frames)} frames, {(out / f'{name}.mp4').stat().st_size} bytes; "
          + "; ".join(f"{p['name']}: end {100 * r['end']:.2f} cm" + ("" if r["clear"] is None else f", lowest clearance {100 * r['clear'].min():.2f} cm "
                      f"(wanted {100 * r['want_clear']:.2f})") + (f", contact at {r['touch_t']} s, {r['force']:.0f} N" if "force" in r else "")
                      for r, p in zip(runs, spec["panels"])), flush=True)  # fmt: skip


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rb = Robot()
    for name in sys.argv[2:] or list(CLIPS):
        clip(rb, name, CLIPS[name], out)


if __name__ == "__main__":
    main()

"""Side-by-side videos of room_grasp.py (2026-11-09): one placement run again, ScaleBFM alone on the left and the world model on the
right, the room view above and two close-ups below, the planned grasp drawn as a green hand, from the qpos recorded every 4 motor ticks (played at twice real time).

The episodes are run again here, so each video prints the numbers of its own run; a mismatch with the batch result is logged.

    export BFM_ROOT=odo OMP_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 GRASP_ROOM_OBJECTS=$I/simready GRASP_ROOM_OUT=$I/grasp_room_sr GRASP_ROOM_MIN_HELD=10
    export APPROACH_WM_POST=$I/invert/wrist_graspr_odo/wm_ref.pt WALK_DATA=approach_odo,approach0_odo WALK_SUFFIX=_odo
    $PY scripts/experiments/reachwm_20261030/room_grasp_video.py 1:14:牛奶盒 2:8:调味瓶
Writes grasp_room_sr/videos/ (src/, preview/ with a poster jpg, and <case>.json with the numbers of the run).
"""

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("REACHWM_APPROACH", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import room_grasp as RG  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

AE, ROOM = RG.AE, RG.ROOM
OUTV = ROOM.OUT / "videos"
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
EVERY, FPS = 4, 25  # 50 Hz ticks: every 4th at 25 fps is twice real time
CW, CC, HM, HC, TB, LG = 480, 240, 270, 260, 36, 104
STATIONS = {"tray": "地面托盘", "low": "矮柜", "table": "桌子", "counter": "台面", "high": "高架", "cabinet": "开放柜"}
PHASES = {"walk": "走向站位", "clear": "手移到家具外", "pregrasp": "到预抓取点", "entry": "慢速进入", "grip": "合爪", "lift": "抬起", "hold": "保持"}  # fmt: skip
NAMES = {"direct": "ScaleBFM 单独跑", "model": "ScaleBFM 加世界模型"}


def verdict(r: dict) -> str:
    if r.get("success"):
        return "抓起并保持住"
    if r.get("declined"):
        return "没有可用的抓取"
    if r.get("fell"):
        return "摔倒"
    if not r.get("grip_held"):
        return "手指间没有物体"
    return "夹住了但没抬起" if r.get("rise_cm", 0) < 2 else "抬起后掉落"


def record(env, rec, con, models, layout, k, j, cache) -> dict:
    m, sim = env.m, env.sim
    frames, phases, ct = [], [], [0]
    orig = m.tick

    def tick():
        ok = orig()
        ct[0] += 1
        if ct[0] % EVERY == 0:
            frames.append(sim.d.qpos.copy())
            phases.append(con.phase)
        return ok

    m.tick = tick
    out = {}
    for meth in RG.METHODS:
        frames.clear(), phases.clear()
        r = RG.episode(env, rec, con, models, layout, k, j, meth, cache)
        out[meth] = dict(r=r, q=np.array(frames), ph=list(phases))
        print(f"L{k} s{j} {meth}: success {r.get('success')} grip err {r.get('at_grip_err')}", flush=True)
    m.tick = orig
    return out


def render_pair(env, layout, k: int, j: int, name: str, cache: dict, run: dict) -> None:
    """The room view and two close-ups per method; the planned grasp stays in the air as a green hand, with the jaw's live distance to it."""
    from report_video import RD  # the project's mp4 writer

    motor = sys.modules[type(env.m).__module__]
    m = env.sim.m
    m.vis.global_.offwidth, m.vis.global_.offheight = max(CW, m.vis.global_.offwidth), max(HM, m.vis.global_.offheight)
    d = mujoco.MjData(m)
    mg = 2 * m.ngeom + 3000
    rm, rc = mujoco.Renderer(m, height=HM, width=CW, max_geom=mg), mujoco.Renderer(m, height=HC, width=CC, max_geom=mg)
    root = m.body_rootid[m.body("pelvis").id]
    robot = np.array([m.body_rootid[b] == root for b in range(m.nbody)])
    slot = layout["slots"][j]
    b = m.body(slot["object"]).id
    adr = m.jnt_qposadr[m.body_jntadr[b]]
    st, c, a, ax, perp = RG.station_frame(slot["station"])
    r0 = next(v["r"] for v in run.values() if "grasp" in v["r"])
    sd = 1.0 if r0["side"] == "right" else -1.0
    q0 = run["model"]["q"][14]
    po, Ro = q0[adr : adr + 3], RG.quat_mat(q0[adr + 3 : adr + 7])
    g0 = cache[slot["object"]]
    target = po + Ro @ g0["tool"][r0["grasp"]]
    Rt = Ro @ g0["R"][r0["grasp"]]
    if r0.get("flip"):
        Rt = Rt @ np.diag([1.0, -1.0, -1.0])  # room_grasp.py: the same grasp with the fingers swapped
    wrist = m.body(motor.WRIST[r0["side"]]).id
    off = motor.R.OFF
    in_hand = np.zeros(m.nbody, bool)
    for i in range(m.nbody):
        x = i
        while x > 0 and x != wrist:
            x = m.body_parentid[x]
        in_hand[i] = x == wrist
    ie = {
        mm: max(i for i, ph in enumerate(v["ph"]) if ph == "entry") for mm, v in run.items()
    }  # the last frame before the jaws close
    grip_at = max(ie.values())
    shift = {mm: grip_at - ie[mm] for mm in run}  # both columns reach the grip on the same frame

    def cam(lookat, dist, az_, el):
        c_ = mujoco.MjvCamera()
        c_.type = mujoco.mjtCamera.mjCAMERA_FREE
        c_.lookat[:] = lookat
        c_.distance, c_.azimuth, c_.elevation = dist, az_, el
        return c_

    in_cab = slot["station"] == "cabinet"
    a_deg = math.degrees(a)
    az = a_deg + (22.0 if in_cab else 42.0) * sd
    cam_m = cam(np.r_[po[:2] - ax * 0.42, 0.25 + 0.55 * po[2]], 2.5, az, -14.0)
    cam_a = cam(target, 0.42, a_deg + (30.0 if in_cab else 45.0) * sd, -10.0 if in_cab else -22.0)
    cam_b = cam(target, 0.42, a_deg - (30.0 if in_cab else 50.0) * sd, -10.0 if in_cab else -48.0)

    def jaw(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        Rw = d.xmat[wrist].reshape(3, 3)
        return d.xpos[wrist] + Rw @ off, Rw

    p_ref, R_ref = jaw(run["model"]["q"][ie["model"]])
    rc.update_scene(d, camera=cam_a)
    shape = []
    for gi in range(rc.scene.ngeom):
        g = rc.scene.geoms[gi]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and in_hand[m.geom_bodyid[g.objid]]:
            shape.append((int(g.type), int(g.dataid), np.array(g.size, np.float64), R_ref.T @ (np.array(g.pos, np.float64) - p_ref),
                          R_ref.T @ np.array(g.mat, np.float64).reshape(3, 3)))  # fmt: skip

    def add(scn, typ, size, pos, mat, rgba, dataid=-1):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, typ, np.asarray(size, np.float64), np.asarray(pos, np.float64), np.asarray(mat, np.float64).ravel(), np.array(rgba, np.float32))  # fmt: skip
        g.dataid = dataid
        scn.ngeom += 1
        return g

    def view(q, rr, cam_, close):
        p_j, _ = jaw(q)
        rr.update_scene(d, camera=cam_)
        scn = rr.scene
        if close:
            for gi in range(scn.ngeom):
                g = scn.geoms[gi]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and robot[m.geom_bodyid[g.objid]]:
                    g.rgba[3] = 0.45
        for typ, dataid, size, rp, rm_ in shape:
            add(scn, typ, size, target + Rt @ rp, Rt @ rm_, (0.1, 0.85, 0.35, 0.42 if close else 0.6), dataid)
        if close:
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], target, np.eye(3), (0.05, 0.6, 0.2, 1.0))
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], p_j, np.eye(3), (0.95, 0.25, 0.1, 1.0))
            g = add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], np.zeros(3), np.zeros(9), (1.0, 0.75, 0.05, 1.0))
            if g is not None:
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, p_j.astype(np.float64), target.astype(np.float64)
                )
        return rr.render().copy()

    def errs(q):
        p_j, Rj = jaw(q)
        return 100 * float(np.linalg.norm(p_j - target)), math.degrees(
            float(np.arccos(np.clip((np.trace(Rt.T @ Rj) - 1) / 2, -1, 1)))
        )

    small, tiny = (ImageFont.truetype(FONT, s) for s in (15, 13))
    title = f"{STATIONS[slot['station']]} {slot['surf']:.2f} m，{name}，{'右' if sd > 0 else '左'}手，两倍速，绿色手爪是目标抓取姿势"
    idx = {mm: (lambda i, v=v, s_=shift[mm]: min(max(0, i - s_), len(v["q"]) - 1)) for mm, v in run.items()}

    def draw(i: int, mode: str) -> np.ndarray:
        canvas = Image.new("RGB", (2 * CW + 4, TB + HM + HC + LG), (248, 248, 246))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle([0, 0, 2 * CW + 4, TB], fill=(24, 28, 34))
        dr.text((10, 9), title, font=small, fill=(240, 240, 240))
        if mode in ("freeze", "poster"):
            dr.text((2 * CW - 110, 9), "合爪前一刻", font=small, fill=(255, 210, 90))
        for col, meth in enumerate(RG.METHODS):
            v = run[meth]
            k_ = idx[meth](i)
            x = col * (CW + 4)
            for img, pos, label in ((view(v["q"][k_], rm, cam_m, False), (x, TB), None),
                                    (view(v["q"][k_], rc, cam_a, True), (x, TB + HM), "特写，手的一侧"),
                                    (view(v["q"][k_], rc, cam_b, True), (x + CC, TB + HM), "特写，另一侧")):  # fmt: skip
                im = Image.fromarray(img)
                if label:
                    ImageDraw.Draw(im).text(
                        (6, 4), label, font=tiny, fill=(20, 20, 20), stroke_width=2, stroke_fill=(255, 255, 255)
                    )
                canvas.paste(im, pos)
            e, ang = errs(v["q"][k_])
            y = TB + HM + HC + 8
            dr.text((x + 10, y), NAMES[meth], font=small, fill=(25, 25, 25))
            dr.text((x + 10, y + 24), f"现在：{PHASES.get(v['ph'][k_], v['ph'][k_])}，手爪离抓取姿势 {e:.1f} cm，转角差 {ang:.0f}°", font=tiny, fill=(70, 70, 70))  # fmt: skip
            if mode in ("end", "poster"):
                r = v["r"]
                e0, a0 = errs(v["q"][ie[meth]])
                dr.text(
                    (x + 10, y + 46),
                    f"合爪时差 {e0:.1f} cm，转角差 {a0:.0f}°，结果：{verdict(r)}",
                    font=small,
                    fill=(25, 25, 25),
                )
        return np.asarray(canvas)

    n = max(len(v["q"]) + shift[mm] for mm, v in run.items())
    frames = [draw(i, "run") for i in range(grip_at + 1)]
    frames += [draw(grip_at, "freeze")] * 45
    frames += [draw(i, "run") for i in range(grip_at + 1, n)]
    last = draw(n - 1, "end")
    frames += [last] * 50
    poster = draw(grip_at, "poster")
    for sub in ("src", "preview"):
        (OUTV / sub).mkdir(parents=True, exist_ok=True)
    tag = f"L{k}_s{j}"
    src = OUTV / "src" / f"{tag}.mp4"
    RD.write(frames, src, fps=FPS)
    nb = RD.preview(src, OUTV / "preview" / f"{tag}.mp4", crf=26)
    Image.fromarray(poster).save(OUTV / "preview" / f"{tag}.jpg", quality=84)
    keep = {mm: {f: v["r"].get(f) for f in ("success", "at_grip_err", "att_deg", "side", "station", "surf", "object", "held_3s", "rise_cm", "grip_held", "declined", "fell", "entry", "grasp", "flip")} for mm, v in run.items()}  # fmt: skip
    live = {mm: errs(v["q"][ie[mm]]) for mm, v in run.items()}
    json.dump(dict(name=name, layout=k, slot=j, frames=len(frames), runs=keep, live_err_deg=live), open(OUTV / f"{tag}.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))  # fmt: skip
    print(
        f"{tag}: {len(frames)} frames, preview {nb} bytes; video err {live} batch {[(mm, round(v['r']['at_grip_err'], 2), round(v['r']['att_deg'], 1)) for mm, v in run.items()]}",
        flush=True,
    )
    rm.close(), rc.close()


def main(cases: list[str]) -> None:
    import gen_reach as GR
    import torch
    import walk_model as WK

    torch.set_num_threads(1)
    models = {"post": torch.load(AE.WM_POST, map_location="cpu", weights_only=False).eval(),
              "post_wkp": torch.load(WK.WALK_MODEL_POST, map_location="cpu", weights_only=False).eval()}  # fmt: skip
    by: dict[int, list] = {}
    for a in cases:
        k, j, name = a.split(":")
        by.setdefault(int(k), []).append((int(j), name))
    for k, items in by.items():
        env = ROOM.make_env(k)
        layout = json.load(open(ROOM.OUT / f"layout_{k}" / "layout.json"))
        cache = RG.load_cache(layout)
        rec = GR.Recorder(env)
        con = RG.Contacts(env)
        env.m.reset((0.0, 0.0), 0.0)  # scalebfm-first-call-nondeterminism
        env.m.hold(1.0)
        for j, name in items:
            render_pair(env, layout, k, j, name, cache, record(env, rec, con, models, layout, k, j, cache))


if __name__ == "__main__":
    main(sys.argv[1:])

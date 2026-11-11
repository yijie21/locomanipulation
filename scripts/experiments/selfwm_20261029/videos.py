"""Videos for the minimal proof's report (outputs/selfwm_20261029/videos/: src/ keeps the recordings, preview/ the
<= 500 KB copies docs/rendering.md#video-export asks for).

  wm_open_loop      a new random-command episode: the simulator (left) against the world model given only the same
                    commands (right), the predicted robot drawn from its predicted joints and odometry
  c_dream_vs_real   policy C under camera yaw 3 deg, no jitter: its imagined episode inside the world model (left)
                    against the same start executed in the simulator (right)
  methods_yaw3      A / B / C / D on eval cell *_yaw3_4 (the same seeds as the table)

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/selfwm_20261029/videos.py all
"""

from __future__ import annotations

import json
import math
import sys

import env as E
import eval as EV
import mujoco
import numpy as np
import perception as P
import policy as POLM
import render as RD
import torch
import wm as WMM
from gen_data import commands

VID = E.OUT / "videos"
VIEW = dict(lookat=(0.31, -0.15, 0.84), azimuth=-120.0, elevation=-42.0, distance=0.62)  # hand and box
VIEW_HAND = dict(lookat=(0.29, -0.16, 0.86), azimuth=-120.0, elevation=-40.0, distance=0.55)  # the roaming hand
GREEN, RED, BLUE, YELLOW = (0.1, 0.8, 0.25, 1), (0.9, 0.15, 0.15, 1), (0.2, 0.45, 1.0, 1), (1.0, 0.8, 0.1, 1)


def quat_of(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def predicted_qpos(template, sim, p, R, q):
    out = template.copy()
    out[:3], out[3:7] = p, quat_of(R)
    out[sim.qadr] = q
    return out


def interp(rows10: list[np.ndarray], n: int) -> list[np.ndarray]:
    """10 Hz qpos rows -> n frames at 25 fps (linear; quaternion normalised)."""
    out = []
    for i in range(n):
        t = i / 25 * 10
        k = min(int(t), len(rows10) - 2)
        u = min(max(t - k, 0.0), 1.0)
        a, b = rows10[k], rows10[k + 1]
        r = (1 - u) * a + u * b
        if np.dot(a[3:7], b[3:7]) < 0:
            r[3:7] = (1 - u) * a[3:7] - u * b[3:7]
        r[3:7] /= np.linalg.norm(r[3:7])
        out.append(r)
    return out


def box_qadr(env) -> int:
    m = env.sim.m
    return int(m.jnt_qposadr[m.body_jntadr[env.box]])


def translucent_box(env) -> None:
    g = mujoco.mj_name2id(env.sim.m, mujoco.mjtObj.mjOBJ_GEOM, "probe_object_geom")
    env.sim.m.geom_rgba[g][3] = 0.45


# -- 1. world model, open loop ---------------------------------------------------------------------------------------
def wm_open_loop(r: EV.Runner, rec: RD.Recorder, seed: int = 990001) -> dict:
    env, model, dev = r.env, r.model, r.dev
    rng = np.random.default_rng(seed)
    env.reset(None)
    env.pregrasp(rng.uniform(E.START_LO, E.START_HI))
    mark = rec.mark()
    S, W, A, T = [env.self_state()], [env.world()], [], []
    for tgt, vel in commands(rng, env.commanded_world()):
        A.append(env.command(tgt, vel))
        T.append(tgt)
        env.step()
        S.append(env.self_state())
        W.append(env.world())
    real = rec.rows[mark:]
    t = lambda x: torch.tensor(np.array(x, np.float32), device=dev)  # noqa: E731
    with torch.no_grad():
        ps, po = model.rollout(t(S[0])[None], t(A[0])[None], t(S[1])[None], t(A[1:])[None])
        kin = model.kin(ps)[0].cpu().numpy()
    ps, po = ps[0].cpu().numpy(), po[0].cpu().numpy()
    template = real[0]
    pred10 = [real[4], real[9]]  # k = 0, 1 are measured (tick 5 and 10)
    p, yaw = W[1][:3].astype(float).copy(), float(W[1][3])
    jaw_pred = [W[0][4:7], W[1][4:7]]
    for h in range(len(po)):
        p = p + E.rot_z(yaw) @ po[h, :3]
        yaw += float(po[h, 3])
        R = E.rot_z(yaw) @ WMM.tilt_rot(ps[h, 58:61].astype(float))
        pred10.append(predicted_qpos(template, env.sim, p, R, ps[h, :29]))
        jaw_pred.append(p + R @ kin[h, :3])
    n = len(real) // 2
    pred = interp(pred10, n)
    jaw_real = [w[4:7] for w in W]
    vl, vr = RD.View(env.sim.m, **VIEW_HAND), RD.View(env.sim.m, **VIEW_HAND)
    frames, errs = [], []
    for i in range(n):
        k = min(int(i / 25 * 10), len(T) - 1)
        e_cmd = 100 * np.linalg.norm(jaw_real[k] - T[max(k - 1, 0)])
        e_wm = 100 * np.linalg.norm(jaw_real[k] - jaw_pred[k])
        errs.append(e_wm)
        trail_r = [(x, GREEN, 0.004) for x in jaw_real[: k + 1 : 2]]
        trail_p = [(x, BLUE, 0.004) for x in jaw_pred[: k + 1 : 2]]
        mk = [(T[k], YELLOW, 0.009)]
        left = vl.frame(
            real[2 * i + 1],
            mk + trail_r,
            "仿真里真实发生的",
            [f"t = {i / 25:.1f} s", f"命令点到真实夹爪 {e_cmd:.1f} cm"],
        )
        right = vr.frame(pred[i], mk + trail_p + trail_r, "世界模型的想象：只给同样的命令",
                         [f"t = {i / 25:.1f} s", f"想象夹爪到真实夹爪 {e_wm:.2f} cm"])  # fmt: skip
        frames.append(RD.tile([left, right], 2))
    RD.write(frames, VID / "src" / "wm_open_loop.mp4")
    return dict(seed=seed, seconds=round(n / 25, 2), wm_err_cm_median=round(float(np.median(errs)), 3),
                wm_err_cm_max=round(float(np.max(errs)), 3))  # fmt: skip


# -- 2. policy C: imagination against reality ------------------------------------------------------------------------
def c_dream_vs_real(r: EV.Runner, rec: RD.Recorder, cond: str = "yaw3", place: int = 4) -> dict:
    env, m, model, dev = r.env, r.env.m, r.model, r.dev
    cd = EV.CONDS[cond]
    dR = P.rot_xyz(*[math.radians(a) for a in cd["dR"]])
    err = dict(dR=dR, scale=cd["scale"], jitter=False)
    env.reset(box_xy=EV.PLACES[place])
    box0 = env.sim.d.xpos[env.box].copy()
    grasp_true = box0 + EV.GRASP_OFF
    g_bel = r.read(r.truth_pel(box0), err, None)[0] + EV.GRASP_OFF
    p0, R0 = env.pelvis()
    env.pregrasp(p0 + R0 @ g_bel - R0 @ np.array([EV.PRE_BACK, 0.0, 0.0]))
    # the hand-over state, shared by the dream and the real run
    s0, w0 = env.self_state(), env.world()
    target0 = env.commanded_world()
    p, Rp = env.pelvis()
    a0 = np.r_[Rp.T @ (target0 - p), np.zeros(3)].astype(np.float32)
    # dream
    f = lambda x: torch.tensor(np.asarray(x, np.float32)[None], device=dev)  # noqa: E731
    pi = r.pi("rel")
    sp, ap, s = f(s0), f(a0), f(s0)
    pw, yaw = torch.tensor(w0[None, :3], device=dev), torch.tensor(w0[None, 3:4], device=dev)[:, 0]
    target_w = f(target0)
    u_prev = torch.zeros(1, 3, device=dev)
    box_t, dR_t, sc_t = f(box0), torch.tensor(dR[None], device=dev, dtype=torch.float32), f([cd["scale"]])
    zero = torch.zeros(1, 3, device=dev)
    template = env.sim.d.qpos.copy()
    dream10, dmk, inside = [], [], 0
    with torch.no_grad():
        for t in range(EV.STEPS):
            k = model.kin(s)
            Rpt = POLM.rz(yaw) @ POLM.tilt(s[:, 58:61])
            box_pel = torch.einsum("bji,bj->bi", Rpt, box_t - pw)
            b_read, _ = P.read_torch(box_pel, k[:, 3:6], POLM.rot6(k), sc_t, dR_t, zero)
            j_read, j_vis = P.read_torch(k[:, :3], k[:, 3:6], POLM.rot6(k), sc_t, dR_t, zero)
            g_read = b_read + torch.tensor(EV.GRASP_OFF, device=dev, dtype=torch.float32)
            dream10.append(
                predicted_qpos(template, env.sim, pw[0].cpu().numpy(), Rpt[0].cpu().numpy(), s[0, :29].cpu().numpy())
            )
            jaw_w = (pw + torch.einsum("bij,bj->bi", Rpt, k[:, :3]))[0].cpu().numpy()
            dmk.append(dict(believed=(pw + torch.einsum("bij,bj->bi", Rpt, g_read))[0].cpu().numpy(),
                            seen_jaw=(pw + torch.einsum("bij,bj->bi", Rpt, j_read))[0].cpu().numpy(), jaw=jaw_w))  # fmt: skip
            rel = (g_read - j_read)[0].cpu().numpy()
            inside = inside + 1 if np.all(np.abs(rel) < EV.TOL) else 0
            if inside >= 2:
                break
            o = POLM.observe("rel", s, k[:, :3], g_read, j_read, j_vis, model.ns,
                             torch.einsum("bji,bj->bi", Rpt, target_w - pw), u_prev)  # fmt: skip
            u = pi(o)
            target_w = POLM.clamp_region(target_w + torch.einsum("bij,bj->bi", Rpt, u) * E.STEP_S)
            a = torch.cat([torch.einsum("bji,bj->bi", Rpt, target_w - pw), u], -1)
            s_next, odo = model.step(sp, ap, s, a)
            pw = pw + torch.einsum("bij,bj->bi", POLM.rz(yaw), odo[:, :3])
            yaw = yaw + odo[:, 3]
            sp, ap, s, u_prev = s, a, s_next, u
    dream_close = len(dream10) - 1
    # real
    mark = rec.mark()
    rmk, inside, tgt = [], 0, target0.copy()
    u_prev_np = np.zeros(3)
    env.command(tgt, np.zeros(3))
    real_close = EV.STEPS
    for t in range(EV.STEPS):
        p, Rp = env.pelvis()
        k = env.kinematics()
        b, _ = r.read(r.truth_pel(env.sim.d.xpos[env.box].copy()), err, None)
        j, vis = r.read(k[:3], err, None)
        g = b + EV.GRASP_OFF
        rmk.append(dict(believed=p + Rp @ g, seen_jaw=p + Rp @ j, jaw=env.jaw_world()))
        inside = inside + 1 if np.all(np.abs(g - j) < EV.TOL) else 0
        if inside >= 2:
            env.command(p + Rp @ k[:3], np.zeros(3))
            env.step()
            real_close = t
            break
        st = torch.tensor(env.self_state()[None], device=dev)
        with torch.no_grad():
            o = POLM.observe(
                "rel", st, f(k[:3]), f(g), f(j), f([float(vis)])[0], model.ns, f(Rp.T @ (tgt - p)), f(u_prev_np)
            )
            u = pi(o)[0].cpu().numpy()
        tgt = np.clip(tgt + Rp @ u * E.STEP_S, E.ROAM_LO, E.ROAM_HI)
        env.command(tgt, Rp @ u)
        u_prev_np = u
        env.step()
    close_mark = rec.mark()
    m.grip(E.SIDE)
    m.adopt_hand(E.SIDE)
    m.reach(E.SIDE, env.jaw_world() + np.array([0.0, 0.0, 0.10]), env.wrist_rot(), speed=0.08)
    m.hold(1.0)
    lifted = bool(E.SIDE in m.gripping(env.box) and env.sim.d.xpos[env.box][2] > box0[2] + 0.05)
    real = rec.rows[mark:]
    n = len(real) // 2
    dream = interp(dream10 + [dream10[-1]], min(n, int(len(dream10) * 2.5) + 1))
    vl, vr = RD.View(env.sim.m, **VIEW), RD.View(env.sim.m, **VIEW)
    frames = []
    for i in range(n):
        kd = min(int(i / 25 * 10), len(dmk) - 1)
        kr = min(int(i / 25 * 10), len(rmk) - 1)
        di = dream[min(i, len(dream) - 1)]
        dm, rm = dmk[kd], rmk[kr]
        ed = 100 * np.linalg.norm(dm["jaw"] - grasp_true)
        er = 100 * np.linalg.norm(rm["jaw"] - (env.sim.d.xpos[env.box] * 0 + grasp_true))
        mk = lambda x: [(grasp_true, GREEN, 0.008), (x["believed"], RED, 0.008), (x["seen_jaw"], BLUE, 0.008)]  # noqa: E731
        dtitle = "世界模型里的想象" + ("，到此合拢" if kd >= dream_close else "")
        left = vl.frame(di, mk(dm), dtitle, [f"t = {i / 25:.1f} s", f"想象的夹爪到真实抓取点 {ed:.1f} cm"])
        rtitle = "仿真里的真实执行" + ("，已合拢" if 2 * i >= close_mark - mark else "")
        right = vr.frame(real[2 * i + 1], mk(rm), rtitle, [f"t = {i / 25:.1f} s", f"真实夹爪到真实抓取点 {er:.1f} cm"])
        frames.append(RD.tile([left, right], 2))
    RD.write(frames, VID / "src" / "c_dream_vs_real.mp4")
    return dict(cond=cond, place=place, dream_close_step=dream_close, real_close_step=real_close,
                dream_close_err_cm=round(float(100 * np.linalg.norm(dmk[-1]["jaw"] - grasp_true)), 2),
                real_close_err_cm=round(float(100 * np.linalg.norm(rmk[-1]["jaw"] - grasp_true)), 2), lifted=lifted,
                seconds=round(n / 25, 2))  # fmt: skip


# -- 3. the four methods ---------------------------------------------------------------------------------------------
NAMES = {"A": "A 世界坐标开环", "B": "B 相对伺服，不训练", "C": "C 世界模型训练的策略", "D": "D 策略，但看不到自己的手"}


def methods(r: EV.Runner, rec: RD.Recorder, cond: str = "yaw3", place: int = 4) -> dict:
    env = r.env
    bq = box_qadr(env)
    runs, out = [], {}
    for meth in EV.METHODS:
        c = next(x for x in EV.cells() if x["id"] == f"{meth}_{cond}_{place}")
        mark = rec.mark()
        res = r.run(c)
        runs.append((meth, rec.rows[mark:], r.last_g_world.copy(), res))
        out[meth] = {k: res[k] for k in ("lifted", "knocked_cm", "close_err_cm", "steps")}
    n = max(len(rows) for _, rows, _, _ in runs) // 2 + 40  # + 1.6 s hold on the last frames with every outcome
    views = [RD.View(env.sim.m, **VIEW) for _ in runs]
    frames = []
    for i in range(n):
        panel = []
        for v, (meth, rows, g_world, res) in zip(views, runs):
            q = rows[min(2 * i + 1, len(rows) - 1)]
            grasp_now = q[bq : bq + 3] + EV.GRASP_OFF
            ok = res["lifted"] and res["knocked_cm"] <= EV.KNOCK_CM
            tail = [f"t = {min(i, len(rows) // 2) / 25:.1f} s"]
            if i >= len(rows) // 2 - 1:
                tail.append(
                    ("成功" if ok else "失败")
                    + f"：推动盒子 {res['knocked_cm']:.1f} cm，"
                    + ("抬起" if res["lifted"] else "没抬起")
                )
            panel.append(v.frame(q, [(grasp_now, GREEN, 0.008), (g_world, RED, 0.008)], NAMES[meth], tail))
        frames.append(RD.tile(panel, 2))
    RD.write(frames, VID / "src" / "methods_yaw3.mp4")
    return dict(cond=cond, place=place, seconds=round(n / 25, 2), runs=out)


def main(argv=None) -> int:
    what = (argv or sys.argv[1:] or ["all"])[0]
    (VID / "src").mkdir(parents=True, exist_ok=True)
    (VID / "preview").mkdir(parents=True, exist_ok=True)
    r = EV.Runner()
    translucent_box(r.env)
    rec = RD.Recorder(r.env.m)
    meta = {}
    if what in ("all", "wm"):
        meta["wm_open_loop"] = wm_open_loop(r, rec)
    if what in ("all", "dream"):
        meta["c_dream_vs_real"] = c_dream_vs_real(r, rec)
    if what in ("all", "methods"):
        meta["methods_yaw3"] = methods(r, rec)
    for name in meta:
        n = RD.preview(VID / "src" / f"{name}.mp4", VID / "preview" / f"{name}.mp4")
        meta[name]["preview_bytes"] = n
    old = json.loads((VID / "videos.json").read_text()) if (VID / "videos.json").exists() else {}
    old.update(meta)
    (VID / "videos.json").write_text(json.dumps(old, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

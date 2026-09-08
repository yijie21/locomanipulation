"""ScaleBFM bimanual crate carry, far stand + emergent bend (grilled design 2026-09-08).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $VENV_SIM_PY src/teleop_free/scalebfm/run_bend.py \
        --rounds 10 --seed 1 --video --out outputs/videos/scalebfm_bend
    ... --preflight     # geometry check of the reach curve vs the table edge, no policy

Differences from run_spike (pilot-23 replica):
  1. stand-off sampled uniformly in STAND_OFF_RANGE (0.28-0.44 m), grasp depth fixed 0.10;
  2. mode 2 only: pelvis pinned at the measured stop, upright, 0.76 high; the torso
     lean is whatever the policy inpaints -- nothing commands a bend;
  3. one 3 s "reach forward" quadratic Bezier per pad (control point 0.10 up, 0.10 out),
     palm pitch 90 -> 60; hand servo (integral, cap 8 cm) from the last third of the
     reach; NO hover hold, straight into the descent; no pelvis servo;
  4. after the lift, 2 s retract: hands 0.15 m toward the pelvis at constant height
     (recorded, not judged);
  5. metrics: max torso pitch, pelvis displacement, pad path smoothness (speed peaks,
     max acceleration), crate tilt after the retract, crate visibility in the ego
     camera per phase; video = third person | ego.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_spike as R  # noqa: E402
from run_spike import (DT, OFF, GRIP_OPEN, GRIP_CLOSED, HOVER_ABOVE, GRIP_BELOW, ACROSS_BIAS, ACROSS_TOL,  # noqa: E402
                       PALM_PITCH_DEG, REACH_PITCH0_DEG, NEAR_X, PELVIS_Z, SETTLE_S, WALK_V, WALK_ACC, SPAWN_BACK,
                       SPAWN_LAT, CRATE_DY, ARRIVE_HOLD_S, DESCEND_SPEED, DESCEND_TIMEOUT, CONTACT_STALL_S,
                       CONTACT_STALL_MM, CLOSE_S, CLOSE_HOLD_S, LIFT_HEIGHT, LIFT_S, LIFT_HOLD_S, LIFT_MIN, TILT_OK_DEG,
                       SERVO_KI_HAND, SERVO_CAP_HAND, rot_z, rot_y, mat, yaw_of, wrap, smoothstep, wall_frame,
                       wall_point, measure_stand, crate_tilt_deg, wrist_from_pad, Sim, Policy, Refs)

STAND_OFF_RANGE = (0.28, 0.44)
DEPTH = 0.10
REACH_S = 3.0
REACH_UP, REACH_OUT = 0.10, 0.10
SERVO_FROM_U = 2.0 / 3.0
RETRACT_M, RETRACT_S, RETRACT_HOLD_S = 0.15, 2.0, 1.0
EGO_EVERY = 25                       # ticks between ego-visibility samples (0.5 s)


class EgoVis:
    """Crate pixels visible in head_camera / crate pixels with the robot hidden."""

    def __init__(self, sim: Sim):
        self.sim = sim
        self.r = mujoco.Renderer(sim.m, height=240, width=320)
        self.r.enable_segmentation_rendering()
        self.cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self.opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(self.opt)
        self.crate_g = np.array([g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] == sim.crate_b])
        pel = sim.bid["pelvis"]
        root = sim.m.body_rootid[pel]
        self.robot_g = np.array([g for g in range(sim.m.ngeom) if sim.m.body_rootid[sim.m.geom_bodyid[g]] == root])

    def __call__(self):
        m, d = self.sim.m, self.sim.d
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        seg = self.r.render()[..., 0]
        with_robot = int(np.isin(seg, self.crate_g).sum())
        saved = m.geom_group[self.robot_g].copy(); m.geom_group[self.robot_g] = 5; self.opt.geomgroup[5] = 0
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        seg2 = self.r.render()[..., 0]
        m.geom_group[self.robot_g] = saved; self.opt.geomgroup[5] = 1
        no_robot = int(np.isin(seg2, self.crate_g).sum())
        return with_robot / max(no_robot, 1), no_robot


def torso_pitch_deg(sim: Sim):
    return float(np.degrees(np.arcsin(-mat(sim.body("torso_link")[1])[2, 0])))


def smoothness(pads):
    """pads: [T,3] at 50 Hz -> (n speed peaks, max |accel| m/s^2, path length m)."""
    p = np.asarray(pads)
    if len(p) < 10:
        return 0, 0.0, 0.0
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
    k = np.ones(5) / 5
    vs = np.convolve(v, k, mode="same")
    peaks = int(np.sum((vs[1:-1] > vs[:-2]) & (vs[1:-1] >= vs[2:]) & (vs[1:-1] > 0.05)))
    a = np.abs(np.diff(vs)) / DT
    return peaks, float(a.max()), float(np.sum(v) * DT)


def run_round(sim: Sim, pol: Policy, rnd: int, rng, out_dir, video, ego: EgoVis | None):
    m = dict(round=rnd, ok=False, why="", close_trigger=None, stand_attempts=[])
    crate_dy = float(rng.uniform(*CRATE_DY))
    crate_x = float(sim.crate0[0])          # crate body x as built (near face = crate_x - NEAR_X; the 0.45 scene sets it back 10 cm)
    so_target = float(rng.uniform(*STAND_OFF_RANGE))
    stand_xy = np.array([crate_x - NEAR_X - so_target, crate_dy])
    spawn_back = float(rng.uniform(*SPAWN_BACK)); spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
    spawn_xy = stand_xy - np.array([spawn_back, -spawn_lat])
    yaw0 = float(np.arctan2(*(np.array([crate_x, crate_dy]) - spawn_xy)[::-1])) + float(rng.uniform(-0.17, 0.17))
    m.update(crate_dy=crate_dy, stand_off_target=so_target, spawn_back=spawn_back, spawn_lat=spawn_lat, spawn_yaw_deg=float(np.degrees(yaw0)))
    sim.reset(spawn_xy, yaw0, crate_dy)
    pol.reset(sim.state())
    refs = Refs(pol, sim)
    trace = []; log = []
    ffm = None
    W = 1280 if ego is not None else 640
    if video:
        ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x480",
                                "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                                os.path.join(out_dir, f"{rnd:02d}.mp4")], stdin=subprocess.PIPE)
        ego_r = mujoco.Renderer(sim.m, height=480, width=640) if ego is not None else None
        ego_cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    T = [0.0]
    crate_p0, _ = sim.crate()
    first_move = [None]
    corr = {"left_wrist_yaw_link": np.zeros(3), "right_wrist_yaw_link": np.zeros(3), "pelvis": np.zeros(3)}
    servo_on = [False]
    phase = ["settle"]
    stats = {"torso_max": -1e9, "torso_at": {}, "pelvis_xy_max": 0.0, "pelvis_z_min": 9.0, "vis": {}}
    pads_reach = []

    def corrected(ref):
        def f(t):
            return {n: (p + corr[n], q) for n, (p, q) in ref(t).items()}
        return f

    def tick(ref):
        st = sim.state()
        tgt = pol(st, refs.futures(corrected(ref), T[0]))
        sim.step(tgt)
        if servo_on[0]:
            r = ref(T[0])
            for sd in ("left", "right"):
                n_ = f"{sd}_wrist_yaw_link"
                pad_ref = r[n_][0] + mat(r[n_][1]) @ OFF
                corr[n_] = np.clip(corr[n_] + SERVO_KI_HAND * DT * (pad_ref - sim.pad(sd)), -SERVO_CAP_HAND, SERVO_CAP_HAND)
        T[0] += DT
        k = len(trace)
        trace.append(sim.d.qpos.copy())
        cp, cq = sim.crate()
        if first_move[0] is None and (np.linalg.norm(cp[:2] - crate_p0[:2]) > 0.005 or abs(np.degrees(wrap(yaw_of(cq) - yaw_of(sim.crate_quat0)))) > 1.0):
            first_move[0] = T[0]
        tp = torso_pitch_deg(sim)
        if tp > stats["torso_max"]:
            stats["torso_max"] = tp; stats["torso_at"] = dict(t=round(T[0], 2), phase=phase[0])
        if phase[0] == "reach":
            pads_reach.append(0.5 * (sim.pad("left") + sim.pad("right")))
        if ego is not None and k % EGO_EVERY == 0:
            frac, n = ego()
            stats["vis"].setdefault(phase[0], []).append(frac)
        if ffm is not None and k % 2 == 0:
            fr = sim.frame()
            if ego is not None:
                ego_r.update_scene(sim.d, camera=ego_cam)
                fr = np.concatenate([fr, ego_r.render()], axis=1)
            ffm.stdin.write(fr.tobytes())
        return sim.fallen()

    def finish(why=None):
        if why:
            m["why"] = why
        m["move_s"] = first_move[0]; m["len_s"] = T[0]; m["log"] = log
        m["torso_pitch_max_deg"] = round(stats["torso_max"], 1); m["torso_pitch_at"] = stats["torso_at"]
        m["ego_vis"] = {k: round(float(np.mean(v)), 2) for k, v in stats["vis"].items()}
        np.savez_compressed(os.path.join(out_dir, "trace", f"wall_s0_{rnd:02d}_state_trace.npz"), qpos=np.array(trace), dt=DT)
        if ffm is not None:
            ffm.stdin.close(); ffm.wait()
        return m

    # --- settle -------------------------------------------------------------------
    p0 = sim.d.qpos[:2].copy()
    ref = lambda t: {**refs.pelvis(p0, yaw0), **refs.hands_at_side(np.array([p0[0], p0[1], PELVIS_Z]), yaw0)}
    for _ in range(int(SETTLE_S / DT)):
        if tick(ref):
            return finish("[settle] fell")
    # --- walk (as run_spike) ----------------------------------------------------------
    phase[0] = "walk"
    c, n, y_hat, psi, _, _ = wall_frame(sim)
    start = sim.d.qpos[:2].copy(); yaw_s = yaw_of(sim.d.qpos[3:7])
    seg = stand_xy - start; L = float(np.linalg.norm(seg)); u_dir = seg / max(L, 1e-6)
    t_acc = WALK_V / WALK_ACC; d_acc = 0.5 * WALK_ACC * t_acc ** 2
    if 2 * d_acc > L:
        t_acc = np.sqrt(L / WALK_ACC); d_acc = L / 2; v_peak = WALK_ACC * t_acc
    else:
        v_peak = WALK_V
    t_cruise = (L - 2 * d_acc) / v_peak; t_total = 2 * t_acc + t_cruise; t_walk0 = T[0]

    def s_of(tau):
        tau = float(np.clip(tau, 0.0, t_total))
        if tau < t_acc:
            return 0.5 * WALK_ACC * tau ** 2
        if tau < t_acc + t_cruise:
            return d_acc + v_peak * (tau - t_acc)
        r = t_total - tau
        return L - 0.5 * WALK_ACC * r ** 2

    def walk_ref(t):
        s = s_of(t - t_walk0); xy = start + u_dir * s
        yaw = yaw_s + wrap(psi - yaw_s) * smoothstep(s / max(L, 1e-6))
        return {**refs.pelvis(xy, yaw), **refs.hands_at_side(np.array([xy[0], xy[1], PELVIS_Z]), yaw)}

    for _ in range(int((t_total + ARRIVE_HOLD_S) / DT)):
        if tick(walk_ref):
            return finish("[walk] fell")
    so, lat, yaw_err, psi = measure_stand(sim)
    m.update(stand_off_meas=so, stand_lat=lat, yaw_err_deg=yaw_err, pelvis_err=float(np.linalg.norm(sim.d.qpos[:2] - stand_xy)),
             stand_attempts=[dict(so=so, lat=lat, yaw=yaw_err)])
    log.append(f"stand: target {so_target:.3f} -> so {so:.3f} lat {lat:+.3f} yaw {yaw_err:+.1f} deg")
    if first_move[0] is not None:
        m["why"] = "[walk] crate moved during the walk-in"
    stand_meas = sim.d.qpos[:3].copy()
    pel_hold = refs.pelvis(stand_meas[:2], psi)          # pinned at the MEASURED stop, no pelvis servo
    # --- reach: quadratic Bezier forward, bend emerges ----------------------------------
    phase[0] = "reach"
    pad0 = {sd: sim.pad(sd) for sd in ("left", "right")}
    p3 = {sd: wall_point(sim, sd, DEPTH, HOVER_ABOVE, ACROSS_BIAS) for sd in ("left", "right")}
    out_w = {"left": np.array([y_hat[0], y_hat[1], 0.0]), "right": -np.array([y_hat[0], y_hat[1], 0.0])}
    p1 = {sd: pad0[sd] + out_w[sd] * REACH_OUT + np.array([0, 0, REACH_UP]) for sd in pad0}
    t_reach0 = T[0]

    def reach_ref(t):
        u = float(np.clip((t - t_reach0) / REACH_S, 0, 1))
        out = dict(pel_hold)
        for sd in ("left", "right"):
            pad = (1 - u) ** 2 * pad0[sd] + 2 * (1 - u) * u * p1[sd] + u * u * p3[sd]
            pitch = REACH_PITCH0_DEG + (PALM_PITCH_DEG - REACH_PITCH0_DEG) * u
            out[f"{sd}_wrist_yaw_link"] = wrist_from_pad(pad, rot_z(psi) @ rot_y(np.radians(pitch)))
        return out

    for k in range(int(REACH_S / DT)):
        if k >= int(SERVO_FROM_U * REACH_S / DT):
            servo_on[0] = True
        if tick(reach_ref):
            return finish("[reach] fell")
    e_h = {sd: sim.pad(sd) - p3[sd] for sd in ("left", "right")}
    m["hover_err_cm"] = {sd: [round(float(v) * 100, 1) for v in e_h[sd]] for sd in e_h}
    pk, amax, plen = smoothness(pads_reach)
    m.update(reach_speed_peaks=pk, reach_accel_max=round(amax, 2), reach_path_m=round(plen, 3))
    pel_now = sim.d.qpos[:3]
    m["pelvis_shift_reach_cm"] = [round(float(v) * 100, 1) for v in (pel_now - stand_meas)]
    log.append(f"reach end: hover err (cm) " + " ".join(f"{sd} {np.round(e_h[sd] * 100, 1)}" for sd in e_h)
               + f" | torso pitch now {torso_pitch_deg(sim):+.0f} (max {stats['torso_max']:+.0f}) | pelvis shift {m['pelvis_shift_reach_cm']} cm"
               + f" | speed peaks {pk}, accel max {amax:.2f}")
    # --- descend (as run_spike, servo on) ------------------------------------------------
    phase[0] = "descend"
    R_palm = rot_z(psi) @ rot_y(np.radians(PALM_PITCH_DEG))
    floor_dz = -GRIP_BELOW; t_d0 = T[0]; hist = []; trigger = None

    def desc_ref(t):
        dz = max(HOVER_ABOVE - DESCEND_SPEED * (t - t_d0), floor_dz)
        out = dict(pel_hold)
        for sd in ("left", "right"):
            out[f"{sd}_wrist_yaw_link"] = wrist_from_pad(wall_point(sim, sd, DEPTH, dz, ACROSS_BIAS), R_palm)
        return out

    e = {}
    while T[0] - t_d0 < DESCEND_TIMEOUT:
        if tick(desc_ref):
            return finish("[descend] fell")
        for sd in ("left", "right"):
            d = sim.pad(sd) - wall_point(sim, sd, DEPTH, 0.0)
            e[sd] = np.array([float(d[:2] @ n), float(d[:2] @ y_hat), float(d[2])])
        hist.append((T[0], {k: v.copy() for k, v in e.items()}))
        dz_ref = max(HOVER_ABOVE - DESCEND_SPEED * (T[0] - t_d0), floor_dz)
        across_ok = all(abs(e[sd][1]) <= ACROSS_TOL + 0.005 for sd in e)
        at_top = all(e[sd][2] <= 0.005 for sd in e)
        if dz_ref > floor_dz and across_ok and at_top and len(hist) > int(CONTACT_STALL_S / DT):
            _, e_old = hist[-1 - int(CONTACT_STALL_S / DT)]
            if abs(np.mean([e[sd][2] for sd in e]) - np.mean([e_old[sd][2] for sd in e_old])) < CONTACT_STALL_MM / 1000:
                trigger = "contact"; break
        if dz_ref <= floor_dz and across_ok and at_top:
            trigger = "floor"; break
    m["close_trigger"] = trigger
    m["seat_err_cm"] = {sd: [round(float(v) * 100, 1) for v in e[sd]] for sd in e}
    log.append(f"close trigger {trigger} after {T[0] - t_d0:.1f} s; seat err (cm): " + " ".join(f"{sd} {np.round(e[sd] * 100, 1)}" for sd in e)
               + f" | torso {torso_pitch_deg(sim):+.0f}")
    if trigger is None:
        return finish("[descend] never seated")
    # --- close ---------------------------------------------------------------------------
    phase[0] = "close"
    t_c0 = T[0]
    for _ in range(int((CLOSE_S + CLOSE_HOLD_S) / DT)):
        sim.grip_target[:] = GRIP_OPEN + smoothstep((T[0] - t_c0) / CLOSE_S) * (GRIP_CLOSED - GRIP_OPEN)
        if tick(desc_ref):
            return finish("[close] fell")
    # --- lift ----------------------------------------------------------------------------
    phase[0] = "lift"
    crate_pc, _ = sim.crate()
    lift0 = {sd: sim.pad(sd).copy() for sd in ("left", "right")}
    t_l0 = T[0]

    def lift_ref(t):
        z = LIFT_HEIGHT * smoothstep((t - t_l0) / LIFT_S)
        out = dict(pel_hold)
        for sd in ("left", "right"):
            out[f"{sd}_wrist_yaw_link"] = wrist_from_pad(lift0[sd] + np.array([0, 0, z]), R_palm)
        return out

    for _ in range(int((LIFT_S + LIFT_HOLD_S) / DT)):
        if tick(lift_ref):
            return finish("[lift] fell")
    cp, cq = sim.crate()
    rise = float(cp[2] - crate_pc[2]); tilt = crate_tilt_deg(cq)
    m.update(rise_cm=round(rise * 100, 1), tilt_deg=round(tilt, 1))
    log.append(f"lift: rose {rise * 100:.1f} cm, tilt {tilt:.1f} deg, torso {torso_pitch_deg(sim):+.0f}")
    ok_lift = rise >= LIFT_MIN and tilt <= TILT_OK_DEG
    # --- retract: pull the crate toward the body at constant height (recorded only) --------
    phase[0] = "retract"
    ret0 = {sd: sim.pad(sd).copy() for sd in ("left", "right")}
    back = -np.array([n[0], n[1], 0.0]) * RETRACT_M
    t_r0 = T[0]

    def retract_ref(t):
        u = smoothstep((t - t_r0) / RETRACT_S)
        out = dict(pel_hold)
        for sd in ("left", "right"):
            out[f"{sd}_wrist_yaw_link"] = wrist_from_pad(ret0[sd] + back * u, R_palm)
        return out

    fell = False
    for _ in range(int((RETRACT_S + RETRACT_HOLD_S) / DT)):
        if tick(retract_ref):
            fell = True; break
    cp2, cq2 = sim.crate()
    m.update(retract_fell=fell, retract_tilt_deg=round(crate_tilt_deg(cq2), 1), retract_rise_cm=round(float(cp2[2] - crate_pc[2]) * 100, 1),
             retract_pull_cm=round(float(-(cp2[:2] - cp[:2]) @ n) * 100, 1), torso_pitch_end_deg=round(torso_pitch_deg(sim), 1))
    log.append(f"retract: crate pulled {m['retract_pull_cm']} cm, still {m['retract_rise_cm']} cm up, tilt {m['retract_tilt_deg']}, torso {m['torso_pitch_end_deg']:+.0f}{' FELL' if fell else ''}")
    if rise < LIFT_MIN:
        return finish(f"[lift] rose {rise * 100:.1f} cm (< {LIFT_MIN * 100:.0f})")
    if tilt > TILT_OK_DEG:
        return finish(f"[lift] tilt {tilt:.1f} deg")
    if m["why"]:
        return finish()
    m["ok"] = ok_lift
    return finish()


def preflight():
    """Pad-target path vs the table edge for three stand-offs: min target height while
    the pad is within 5 cm of the edge plane, and clearance to the table top."""
    meta = json.load(open(R.META)); sim = Sim(meta)
    print(f"table top {sim.table_top:.2f}, edge x {sim.edge_x:.3f}; hover {HOVER_ABOVE:+.2f} above bar top")
    for so in (0.28, 0.36, 0.44):
        sim.reset(np.array([sim.edge_x - so, 0.0]), 0.0, 0.0)
        sim.d.qpos[sim.qadr] = sim.default; mujoco.mj_forward(sim.m, sim.d)
        pad0 = {sd: sim.pad(sd) for sd in ("left", "right")}
        c, n, y_hat, psi, _, _ = wall_frame(sim)
        p3 = {sd: wall_point(sim, sd, DEPTH, HOVER_ABOVE, ACROSS_BIAS) for sd in ("left", "right")}
        out_w = {"left": np.array([y_hat[0], y_hat[1], 0.0]), "right": -np.array([y_hat[0], y_hat[1], 0.0])}
        p1 = {sd: pad0[sd] + out_w[sd] * REACH_OUT + np.array([0, 0, REACH_UP]) for sd in pad0}
        for sd in ("left",):
            us = np.linspace(0, 1, 151)
            path = np.array([(1 - u) ** 2 * pad0[sd] + 2 * (1 - u) * u * p1[sd] + u * u * p3[sd] for u in us])
            near_edge = np.abs(path[:, 0] - sim.edge_x) < 0.05
            zmin = path[near_edge, 2].min() if near_edge.any() else float("nan")
            # finger tips hang ~3 cm below the pad centre at pitch ~75 deg
            print(f"so {so:.2f}: pad0 {np.round(pad0[sd], 3)} -> hover {np.round(p3[sd], 3)} | reach {np.linalg.norm(p3[sd] - pad0[sd]):.2f} m | "
                  f"min pad z within 5 cm of the edge {zmin:.3f} -> clearance over the top {(zmin - 0.03 - sim.table_top) * 100:+.1f} cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(R.ROOT, "outputs/videos/scalebfm_bend"))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--no-ego", action="store_true")
    a = ap.parse_args()
    if a.preflight:
        preflight(); return
    os.makedirs(os.path.join(a.out, "trace"), exist_ok=True)
    pol = Policy(); sim = Sim(pol.meta, video=a.video)
    ego = None if a.no_ego else EgoVis(sim)
    rng = np.random.default_rng(a.seed)
    mf = open(os.path.join(a.out, "manifest.jsonl"), "a"); oks = 0
    for r in range(a.rounds):
        t0 = time.time()
        m = run_round(sim, pol, r, rng, a.out, a.video, ego)
        oks += m["ok"]
        print(f"round {r}: {'OK' if m['ok'] else '--'} {m['why']} | so {m.get('stand_off_meas', float('nan')):.3f} (target {m['stand_off_target']:.3f}) "
              f"yaw {m.get('yaw_err_deg', float('nan')):+.1f} | trig {m.get('close_trigger')} rise {m.get('rise_cm', float('nan'))} tilt {m.get('tilt_deg', float('nan'))} "
              f"| torso max {m['torso_pitch_max_deg']} | ego vis {m['ego_vis']} | {m['len_s']:.0f} s sim, {time.time() - t0:.0f} s wall", flush=True)
        for line in m.get("log", []):
            print("   ", line, flush=True)
        mf.write(json.dumps(m) + "\n"); mf.flush()
    print(f"OK {oks}/{a.rounds}")


if __name__ == "__main__":
    main()

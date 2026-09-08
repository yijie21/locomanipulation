"""ScaleBFM single-hand bottle expert, v4: CLOCKLESS.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $VENV_SIM_PY src/teleop_free/scalebfm/run_bottle_v4.py \
        --rounds 10 --seed 1 --video --out outputs/videos/scalebfm_bottle_v4

Every tick the command is a deterministic function of what the cameras show:
pelvis pose, right pad pose, gripper opening, bottle pose, table top -- plus the
per-episode constants (sampled stand distance d, approach line). No phase timers,
no time-parameterised curves, no integrators:

  pelvis   carrot on the approach line: projection of the pelvis + min(LEAD, remaining),
           i.e. the stand itself once within LEAD -> deceleration comes from geometry
  yaw      toward the bottle, rate-limited
  hand     side pose while remaining > 0.5 m; otherwise an attractor chosen by geometry:
             pre-grasp (bottle - 0.10 along the approach, 2 cm below centre) until the pad
             is inside the entry corridor, then the LIVE grasp point; after the close,
             the lift point (bottle 12 cm above its rest height) until the bottle is 8 cm
             up, then the carry point (fixed offset in the pelvis frame)
           command = pad + (1+KP) * min(D_CARROT, |att - pad|) * dir + BIAS, palm level
           + 13 deg nose-up (the same feed-forward as v3)
  gripper  closes at a fixed rate once the pad is within the close tolerance of the grasp
           point, keeps closing once it has started (its own opening is the state)
Phase names below are labels for logging; the behaviour never reads them. Time
limits exist only as failure detectors.
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

R.SCENE = os.environ.get("SCENE_DIR", os.path.join(R.ROOT, "outputs/scene/sim"))
R.OBJ = "water_bottle"
from run_spike import DT, GRIP_OPEN, GRIP_CLOSED, PELVIS_Z, SETTLE_S, rot_z, rot_y, mat, yaw_of, wrap, wrist_from_pad, Sim, Policy, Refs  # noqa: E402
from run_bend import torso_pitch_deg  # noqa: E402
from bottle_probe import Vis  # noqa: E402

STAND_D = (0.45, 0.55)
STAND_BACKSTEP = 0.06
STAND_LEFT = 0.15
SPAWN_BACK = (0.9, 1.3); SPAWN_LAT = 0.20; SPAWN_YAW_ERR = np.radians(20)
BOTTLE_JITTER = 0.08
LEAD = 0.25                      # pelvis carrot: this much ahead at tau=0 (a state-based reference has a rest fixed point; the lead must beat the policy's deadband) ...
V_WALK, A_WALK = 0.5, 0.6        # ... advancing at v(remaining) = min(V_WALK, sqrt(2 A remaining)) through the future window
V_HAND, K_HAND, LEAD_HAND = 0.4, 1.5, 0.10   # hand: v = min(V_HAND, K_HAND * dist) through the window, lead at tau=0
KP_PEL = 1.0                     # overdrive on the remaining distance (the policy parks ~10 cm short of a static target)
YAW_RATE = 1.0                   # rad/s of the yaw command
REACH_START_DIST = 0.35
GRIP_BELOW_CENTRE = 0.02
PRE_GRASP_BACK = 0.10
CORRIDOR_ALONG, CORRIDOR_ACROSS = -0.03, 0.03     # pad past this (along) and within this (across) of the pre point -> attractor = grasp
CLOSE_TOL_ACROSS, CLOSE_TOL_ALONG = 0.015, 0.02
GRIP_RATE = 0.08                 # m/s of jaw travel (0.0475 m open->closed in 0.6 s)
LIFT_ABOVE_REST = 0.12           # lift attractor: bottle centre this far above its rest height
LIFT_DONE = 0.08                 # bottle up this much -> carry
CARRY_REL = np.array([0.28, -0.12, 0.15])   # carry point in the pelvis frame (in front of the right shoulder, chest height)
CARRY_TOL = 0.03; CARRY_HOLD_S = 1.0
D_CARROT = 0.10; KP = 1.0
BIAS = np.array([0.100, -0.030, 0.028]); PITCH_FF_DEG = 13.0
BOTTLE_HALF_H = 0.083            # bottle centre above the table when standing
LIFT_OK, TILT_OK_DEG = 0.08, 30.0
GRIP_CLIP = float(os.environ.get("GRIP_CLIP", 0.6))
EGO_EVERY = 25
MAX_S = dict(walk=20.0, reach=8.0, entry=6.0, close=3.0, lift=4.0, carry=6.0)


def ph_prev(info):
    return info.get("ph", "walk")


def run_round(sim: Sim, pol: Policy, rnd: int, rng, out_dir, video, ego, wrist):
    m = dict(round=rnd, ok=False, why="", close_trigger=None, stand_attempts=[])
    bottle0 = sim.crate0.copy(); bottle0[:2] += rng.uniform(-BOTTLE_JITTER, BOTTLE_JITTER, 2)
    d = float(rng.uniform(*STAND_D))
    appr0 = np.array([1.0, 0.0]); left = np.array([-appr0[1], appr0[0]])
    spawn_back = float(rng.uniform(*SPAWN_BACK)); spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
    stand0 = bottle0[:2] - appr0 * (d - STAND_BACKSTEP) + left * STAND_LEFT
    spawn_xy = stand0 - appr0 * spawn_back + left * spawn_lat
    yaw0 = float(np.arctan2(*(bottle0[:2] - spawn_xy)[::-1])) + float(rng.uniform(-SPAWN_YAW_ERR, SPAWN_YAW_ERR))
    m.update(stand_d=d, bottle_xy=[round(float(v), 3) for v in bottle0[:2]], spawn_back=spawn_back, spawn_lat=spawn_lat, spawn_yaw_deg=float(np.degrees(yaw0)))
    sim.reset(spawn_xy, yaw0, 0.0)
    sim.d.qpos[sim.crate_q:sim.crate_q + 3] = bottle0; mujoco.mj_forward(sim.m, sim.d)
    sim.grip_clip = GRIP_CLIP
    pol.reset(sim.state()); refs = Refs(pol, sim)
    rest_z = sim.table_top + BOTTLE_HALF_H
    u_dir = (stand0 - spawn_xy) / np.linalg.norm(stand0 - spawn_xy)     # the approach line (episode constant)
    trace = []; log = []; T = [0.0]; first_move = [None]; phase = ["settle"]
    stats = {"torso_max": -1e9, "vis": {}, "wvis": {}}
    ffm = None
    if video:
        ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "25", "-i", "-",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(out_dir, f"{rnd:02d}.mp4")], stdin=subprocess.PIPE)
        ego_r = mujoco.Renderer(sim.m, height=480, width=640); ego_cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    info = {}

    # ---------------------------------------------------------------- the expert: state -> command
    def expert():
        pel = sim.d.qpos[:3].copy(); yaw = yaw_of(sim.d.qpos[3:7])
        pad = sim.pad("right"); bottle = sim.crate()[0]; opening = float(np.mean(sim.d.qpos[sim.gq[2:]]))
        stand = bottle[:2] - appr0 * (d - STAND_BACKSTEP) + left * STAND_LEFT
        yaw_goal = float(np.arctan2(*(bottle[:2] - stand)[::-1]))
        appr = np.array([np.cos(yaw_goal), np.sin(yaw_goal), 0.0])
        # pelvis carrot on the line
        s_pel = float((pel[:2] - spawn_xy) @ u_dir); s_end = float((stand - spawn_xy) @ u_dir)
        remaining = s_end - s_pel
        v_walk = min(V_WALK, float(np.sqrt(2 * A_WALK * max(remaining, 0.0))))
        proj = spawn_xy + u_dir * s_pel

        def pelvis_at(tau):                          # velocity-encoded carrot: the window shows motion, not a static offset
            adv = min((1 + KP_PEL) * max(remaining, 0.0), LEAD + v_walk * tau)
            xy = proj + u_dir * adv if remaining > 0 else stand
            yaw_c = yaw + float(np.clip(wrap(yaw_goal - yaw), -YAW_RATE * (0.02 + tau), YAW_RATE * (0.02 + tau)))
            return refs.pelvis(xy, yaw_c)
        side = refs.hands_at_side(np.array([pel[0], pel[1], PELVIS_Z]), yaw)
        # hand attractor by geometry
        grasp = bottle + np.array([0, 0, -GRIP_BELOW_CENTRE]); pre = grasp - appr * PRE_GRASP_BACK
        rise = float(bottle[2] - rest_z)
        rel = pad - pre; along = float(rel @ appr); across = float(np.linalg.norm(rel - appr * along))
        e = pad - grasp; e_along = float(e @ appr); e_across = float(np.linalg.norm(e - appr * e_along))
        in_close_tol = abs(e_along) <= CLOSE_TOL_ALONG and e_across <= CLOSE_TOL_ACROSS
        # the jaw COMMAND is the latch (it is the gripper action, recorded in every VLA dataset and part of
        # observation.state); the measured opening alone chatters +-4 mm with the arm swinging
        if in_close_tol and not info.get("grip_closed_cmd"):
            info["grip_closed_cmd"] = True
        closing = bool(info.get("grip_closed_cmd"))
        jaw_stalled = closing and abs(float(np.mean(sim.d.qvel[sim.gv[2:]]))) < 0.005   # closed on the bottle (or fully shut)
        side_pad = side["right_wrist_yaw_link"][0] + mat(side["right_wrist_yaw_link"][1]) @ R.OFF
        d_side = float(np.linalg.norm(pad - side_pad))
        # the walking arm swings > 15 cm about the side pose (policy's own gait), so the hand position cannot
        # tell "reaching" from "walking"; the latch is the previous hand COMMAND (the last action, visible
        # to a policy with action history), exactly like the gripper latch
        if remaining <= REACH_START_DIST:
            info["reaching"] = True
        hand_at_side = not info.get("reaching")
        info["d_side_max"] = max(info.get("d_side_max", 0.0), d_side) if ph_prev(info) == "walk" else info.get("d_side_max", 0.0)
        if remaining > REACH_START_DIST and hand_at_side and not closing:
            att = None; ph = "walk"
        elif not closing:
            if not info.get("entry") and along > CORRIDOR_ALONG and across < CORRIDOR_ACROSS:
                info["entry"] = True                   # latch on the previous command's attractor (action history)
            if info.get("entry") and along < -0.06:
                info["entry"] = False                  # pushed well back out: re-approach the pre point
            if info.get("entry"):
                att = grasp; ph = "entry"
            else:
                att = pre; ph = "reach"
        elif rise < LIFT_DONE and not jaw_stalled:
            att = grasp; ph = "close"
        elif rise < LIFT_DONE:
            att = np.array([pad[0], pad[1], pad[2] + (rest_z + LIFT_ABOVE_REST - bottle[2])]); ph = "lift"
        else:
            att = pel + rot_z(yaw) @ CARRY_REL; ph = "carry"
        grip = GRIP_OPEN
        if closing:
            grip = max(float(sim.grip_target[2]) - GRIP_RATE * DT, GRIP_CLOSED)     # ramp the command, not the chattering measurement
        R_palm = rot_z(yaw_goal) @ rot_y(np.radians(-PITCH_FF_DEG))

        def frame(tau):
            out = dict(pelvis_at(tau))
            out["left_wrist_yaw_link"] = side["left_wrist_yaw_link"]
            if att is None:
                out["right_wrist_yaw_link"] = side["right_wrist_yaw_link"]
            else:
                v = att - pad; n = float(np.linalg.norm(v))
                v_h = min(V_HAND, K_HAND * n)
                adv = min((1 + KP) * n, LEAD_HAND + v_h * tau)
                cmd = pad + (v / n) * adv + BIAS if n > 1e-6 else att + BIAS
                out["right_wrist_yaw_link"] = wrist_from_pad(cmd, R_palm)
            return out
        out = frame
        info.update(remaining=remaining, stand=stand, yaw_goal=yaw_goal, att_dist=(None if att is None else float(np.linalg.norm(att - pad))),
                    rise=rise, opening=opening, e=e, in_close_tol=in_close_tol, bottle=bottle, appr=appr)
        info["ph"] = ph
        return out, grip, ph

    def tick():
        out, grip, ph = expert()
        sim.grip_target[2:] = grip                                  # right hand jaws
        if ph != phase[0]:
            log.append(f"t {T[0]:5.2f}: {phase[0]} -> {ph} (remaining {info['remaining']:+.2f} m, att {info['att_dist']})")
            info.setdefault("t_phase", {})[ph] = T[0]
            phase[0] = ph
        st = sim.state()
        sim.step(pol(st, refs.futures(lambda t: out(t - T[0]), T[0])))
        T[0] += DT; k = len(trace); trace.append(sim.d.qpos.copy())
        bp, _ = sim.crate()
        if first_move[0] is None and np.linalg.norm(bp[:2] - bottle0[:2]) > 0.005:
            first_move[0] = (round(T[0], 2), phase[0])
        stats["torso_max"] = max(stats["torso_max"], torso_pitch_deg(sim))
        if k % EGO_EVERY == 0:
            stats["vis"].setdefault(phase[0], []).append(ego()[0]); stats["wvis"].setdefault(phase[0], []).append(wrist()[0])
        if ffm is not None and k % 2 == 0:
            ego_r.update_scene(sim.d, camera=ego_cam)
            ffm.stdin.write(np.concatenate([sim.frame(), ego_r.render()], axis=1).tobytes())
        return sim.fallen()

    def finish(why=None):
        if why:
            m["why"] = why
        m["move_s"] = first_move[0]; m["len_s"] = T[0]; m["log"] = log
        m["torso_pitch_max_deg"] = round(stats["torso_max"], 1)
        m["ego_vis"] = {k: round(float(np.mean(v)), 2) for k, v in stats["vis"].items()}
        m["wrist_vis"] = {k: round(float(np.mean(v)), 2) for k, v in stats["wvis"].items()}
        m["phase_t"] = {k: round(v, 2) for k, v in info.get("t_phase", {}).items()}
        np.savez_compressed(os.path.join(out_dir, "trace", f"wall_s0_{rnd:02d}_state_trace.npz"), qpos=np.array(trace), dt=DT)
        if ffm is not None:
            ffm.stdin.close(); ffm.wait()
        return m

    # settle: the same expert, but with the walk suppressed for 1.5 s (the policy fills its history)
    p0 = sim.d.qpos[:2].copy(); side0 = refs.hands_at_side(np.array([p0[0], p0[1], PELVIS_Z]), yaw0)
    for _ in range(int(SETTLE_S / DT)):
        st = sim.state(); out = {**refs.pelvis(p0, yaw0), **side0}
        sim.step(pol(st, refs.futures(lambda t: out, T[0]))); T[0] += DT; trace.append(sim.d.qpos.copy())
    stand_logged = False
    t_first = {}
    while True:
        if tick():
            return finish(f"[{phase[0]}] fell")
        t_first.setdefault(phase[0], T[0])
        if phase[0] in MAX_S and T[0] - t_first[phase[0]] > MAX_S[phase[0]] or T[0] > 45.0:
            return finish(f"[{phase[0]}] timeout, att dist {info['att_dist']}, close tol {info['in_close_tol']}, rise {info['rise']:.3f}, opening {info['opening']:.4f}")
        if not stand_logged and phase[0] == "entry":
            stand_logged = True
            pel_xy = sim.d.qpos[:2].copy(); d_meas = float(np.linalg.norm(info["bottle"][:2] - pel_xy))
            yaw_err = float(np.degrees(wrap(info["yaw_goal"] - yaw_of(sim.d.qpos[3:7]))))
            m.update(stand_d_meas=round(d_meas, 3), stand_err_cm=round(float(np.linalg.norm(pel_xy - info["stand"])) * 100, 1), yaw_err_deg=round(yaw_err, 1),
                     stand_attempts=[dict(so=d_meas, lat=0.0, yaw=yaw_err)], stand_off_meas=d_meas)
            log.append(f"entry begins: pelvis d {d_meas:.3f} (target {d:.3f}), stand err {m['stand_err_cm']} cm, yaw {yaw_err:+.1f}, torso {torso_pitch_deg(sim):+.0f}, arm swing while walking {info.get('d_side_max', 0) * 100:.0f} cm")
            if first_move[0] is not None:
                return finish("[reach] bottle knocked before the entry")
        if phase[0] == "close" and m.get("close_trigger") is None:
            m["close_trigger"] = "cue"; m["entry_err_cm"] = [round(float(v) * 100, 1) for v in info["e"]]
            log.append(f"close at t {T[0]:.2f}: pad err {np.round(info['e'] * 100, 1)} cm, torso {torso_pitch_deg(sim):+.0f}")
        if phase[0] == "carry" and info["att_dist"] is not None and info["att_dist"] < CARRY_TOL:
            info["carry_ok_s"] = info.get("carry_ok_s", 0.0) + DT
            if info["carry_ok_s"] >= CARRY_HOLD_S:
                break
    bp, bq = sim.crate()
    rise = float(bp[2] - rest_z); tilt = float(np.degrees(np.arccos(np.clip(mat(bq)[2, 2], -1, 1))))
    m.update(rise_end_cm=round(rise * 100, 1), tilt_end_deg=round(tilt, 1), torso_end_deg=round(torso_pitch_deg(sim), 1))
    log.append(f"carry pose reached at t {T[0]:.1f}: bottle {rise * 100:.1f} cm up, tilt {tilt:.0f}, torso {torso_pitch_deg(sim):+.0f}")
    if rise < LIFT_OK:
        return finish(f"[carry] bottle only {rise * 100:.1f} cm up")
    if tilt > TILT_OK_DEG:
        return finish(f"[carry] bottle tilted {tilt:.0f} deg")
    m["ok"] = True
    return finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(R.ROOT, "outputs/videos/scalebfm_bottle_v4"))
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "trace"), exist_ok=True)
    pol = Policy(); sim = Sim(pol.meta, video=a.video)
    for g in range(sim.m.ngeom):                                   # pad hardening (as v3)
        b = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or ""
        if "dex1_finger" in b and sim.m.geom_contype[g] != 0:
            sim.m.geom_priority[g] = 1; sim.m.geom_condim[g] = 4
            sim.m.geom_solref[g] = (0.002, 1.0); sim.m.geom_solimp[g] = (0.995, 0.9999, 0.001, 0.5, 2.0)
    ego, wrist = Vis(sim, "head_camera"), Vis(sim, "right_wrist")
    rng = np.random.default_rng(a.seed)
    mf = open(os.path.join(a.out, "manifest.jsonl"), "a"); oks = 0
    for r in range(a.rounds):
        t0 = time.time()
        m = run_round(sim, pol, r, rng, a.out, a.video, ego, wrist)
        oks += m["ok"]
        print(f"round {r}: {'OK' if m['ok'] else '--'} {m['why']} | d {m.get('stand_d_meas')} (target {m['stand_d']:.3f}) stand err {m.get('stand_err_cm')} yaw {m.get('yaw_err_deg')} "
              f"| rise {m.get('rise_end_cm')} tilt {m.get('tilt_end_deg')} | torso max {m['torso_pitch_max_deg']} | phases {m['phase_t']} | ego {m['ego_vis']} wrist {m['wrist_vis']} "
              f"| {m['len_s']:.0f} s sim, {time.time() - t0:.0f} s wall", flush=True)
        for line in m["log"]:
            print("   ", line, flush=True)
        mf.write(json.dumps(m) + "\n"); mf.flush()
    print(f"OK {oks}/{a.rounds}")


if __name__ == "__main__":
    main()

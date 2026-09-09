"""ScaleBFM single-hand bottle grasp expert (image-to-action design, 2026-09-08).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $VENV_SIM_PY src/teleop_free/scalebfm/run_bottle.py \
        --rounds 10 --seed 1 --video --out outputs/videos/scalebfm_bottle

Every command is a function of the bottle pose and the robot's own visible state:
  * stand point   = bottle - d * approach + 0.15 * left   (d ~ U(0.45, 0.55): the band
                    where the feed-forward+proportional hand lands within 1.5 cm and the
                    head camera still sees the bottle, bottle_probe.py)
  * walk          = pelvis reference along the straight line, trapezoid 0.5 m/s, hands at
                    the sides; the REACH STARTS while still walking (last 0.5 m), one
                    min-jerk curve to the pre-grasp point (grasp - 0.10 along approach)
  * entry         = straight line 2 s to the grasp point (bottle centre - 2 cm), level palm
                    + 13 deg nose-up feed-forward, one global position bias + proportional
                    overdrive on the measured pad error (no integrator, no hidden state)
  * close cue     = pad centre within 1.5 cm of the grasp point across/height and 2 cm
                    along (what the wrist camera shows: bottle between the pads)
  * lift + carry  = +0.10 m in 1.5 s, then 0.15 m toward the body while descending 0.05 m
                    (2 s), hold 1 s. Success: bottle >= 8 cm above its start at the end,
                    upright (< 30 deg), never knocked before the close.
No hover hold, no squat command, no pelvis servo, no re-approach. The torso lean is
emergent. Static-frame fraction is reported (the summariser reads the manifest).
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
from run_spike import (DT, OFF, GRIP_OPEN, GRIP_CLOSED, PELVIS_Z, SETTLE_S, WALK_V, WALK_ACC, rot_z, rot_y, mat, yaw_of,  # noqa: E402
                       wrap, smoothstep, wrist_from_pad, Sim, Policy, Refs)
from run_bend import EgoVis, torso_pitch_deg, smoothness  # noqa: E402
from bottle_probe import minjerk  # noqa: E402

STAND_D = (0.45, 0.55)          # pelvis distance to the bottle (measured band; BFM_STAND_D="lo,hi" overrides for sweeps)
if os.environ.get("BFM_STAND_D"):
    STAND_D = tuple(float(v) for v in os.environ["BFM_STAND_D"].split(","))
STAND_BACKSTEP = 0.06           # the body steps back this much when the arm rises during the walk-in (debug rounds); aim closer by it
STAND_LEFT = 0.15               # pelvis offset to the robot's left, so the bottle is in front of the right shoulder
SPAWN_BACK = (0.9, 1.3)         # behind the stand
SPAWN_LAT = 0.20
SPAWN_YAW_ERR = np.radians(20)
BOTTLE_JITTER = 0.08            # bottle placement jitter (x, y), stays >= 10 cm inboard of the edge
REACH_START_DIST = float(os.environ.get("BFM_REACH_START", 0.5))   # start the reach when the pelvis reference has this far to go (0 = after arrival)
REACH_MIN_S = float(os.environ.get("BFM_REACH_MIN_S", 2.5))
GRIP_BELOW_CENTRE = 0.02
PRE_GRASP_BACK = 0.10
ENTRY_S = 2.0
ENTRY_TIMEOUT = 4.0
CLOSE_TOL_ACROSS, CLOSE_TOL_ALONG = 0.015, 0.02
CLOSE_S, CLOSE_HOLD_S = 0.6, 0.3
LIFT_H, LIFT_S = 0.10, 1.5
CARRY_BACK, CARRY_DOWN, CARRY_S, CARRY_HOLD_S = 0.15, 0.0, 2.0, 1.0    # no lowering: the bottle rides with the hand, lowering just gave the 8 cm rule away
LIFT_OK, TILT_OK_DEG = 0.08, 30.0
BIAS = np.array([0.100, -0.030, 0.028])     # bottle_probe open-loop mean (command = ref + BIAS); y widened by the +1.6 cm residual of pilot 1
# BIAS was measured on the +x approach and is a BODY-frame offset (ahead, right, up): it is rotated by
# the stand yaw before use (BIAS_w below). Applied as a world constant it sent the hand 10 cm sideways
# on the +-y approaches of the edge geometry (2026-09-08: 'bottle pushed' 3/3, stand err 13 cm).
PITCH_FF_DEG = 13.0
KP, CAP = 1.0, 0.05
GRIP_CLIP = float(os.environ.get("GRIP_CLIP", 0.6))   # x 250 N clamp on the gripper PD (0.6 = 150 N)
EGO_EVERY = 25


SCENE_MODE = os.environ.get("BFM_SCENE", "edge")   # edge = the SONIC gym's geometry (default); corner = v3/v4 legacy
EDGE_INBOARD = (0.04, 0.08)     # bottle centre this far in from its nearest edge (grasp_gym CAN_INBOARD)
SPAWN_EDGE = (1.1, 1.3)         # pelvis-to-edge distance at spawn (grasp_gym --spawn-edge 1.1 1.3)


def sample_scene(sim: Sim, rng):
    """One round's bottle, stand and spawn. Consumes the rng in a fixed order, so a replay can burn a
    round's draws by calling this and discarding the result.

    edge (default): the SONIC gym's geometry -- the bottle hugs its nearest table edge (4-8 cm inboard,
    jittered along the edge), the robot approaches that edge face-on from 1.1-1.3 m with +-20 deg of
    yaw error and +-0.20 m lateral offset, and stands STAND_D from the bottle. corner (BFM_SCENE=corner):
    the v3/v4 pilots' geometry -- the bottle at its nominal spot 34 cm in from the -x edge and 12 cm from
    the +y edge, always approached along +x, i.e. from the table corner with a 0.45-0.49 m reach; kept
    only to reproduce those datasets.
    """
    bottle = sim.crate0.copy()
    d = float(rng.uniform(*STAND_D))
    if SCENE_MODE == "corner":
        bottle[:2] += rng.uniform(-BOTTLE_JITTER, BOTTLE_JITTER, 2)
        appr0 = np.array([1.0, 0.0])
        left = np.array([-appr0[1], appr0[0]])
        stand_xy = bottle[:2] - appr0 * (d - STAND_BACKSTEP) + left * STAND_LEFT
        spawn_back = float(rng.uniform(*SPAWN_BACK)); spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
        spawn_xy = stand_xy - appr0 * spawn_back + left * spawn_lat
        meta = dict(scene="corner", stand_d=d, spawn_back=spawn_back, spawn_lat=spawn_lat)
    else:
        surf = sim.smeta["surfaces"]["A"]; cx, cy = surf["xy"]; hx, hy = surf["flat_half_extent"]
        nom = sim.crate0[:2]
        dists = {(0, -1): nom[0] - (cx - hx), (0, 1): (cx + hx) - nom[0], (1, -1): nom[1] - (cy - hy), (1, 1): (cy + hy) - nom[1]}
        ax, sgn = min(dists, key=dists.get)                        # the edge the nominal spot hugs
        force = os.environ.get("BFM_EDGE", "auto")                 # x-|x+|y-|y+ overrides (diagnostics)
        if force != "auto":
            ax, sgn = "xy".index(force[0]), (1 if force[1] == "+" else -1)
        inboard = float(rng.uniform(*EDGE_INBOARD)); along = float(rng.uniform(-BOTTLE_JITTER, BOTTLE_JITTER))
        edge_coord = (cx, cy)[ax] + sgn * (hx, hy)[ax]
        bottle[ax] = edge_coord - sgn * inboard; bottle[1 - ax] = nom[1 - ax] + along
        appr0 = np.zeros(2); appr0[ax] = -sgn                      # inward normal of that edge
        left = np.array([-appr0[1], appr0[0]])
        stand_xy = bottle[:2] - appr0 * (d - STAND_BACKSTEP) + left * STAND_LEFT
        spawn_edge = float(rng.uniform(*SPAWN_EDGE)); spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
        edge_pt = bottle[:2] - appr0 * inboard                     # the edge point in front of the bottle
        spawn_xy = edge_pt - appr0 * spawn_edge + left * spawn_lat
        meta = dict(scene="edge", edge=f"{'xy'[ax]}{'+' if sgn > 0 else '-'}", inboard=inboard, stand_d=d,
                    stand_edge=d - STAND_BACKSTEP - inboard, spawn_edge=spawn_edge, spawn_lat=spawn_lat)
    yaw0 = float(np.arctan2(*(bottle[:2] - spawn_xy)[::-1])) + float(rng.uniform(-SPAWN_YAW_ERR, SPAWN_YAW_ERR))
    meta.update(bottle_xy=[round(float(v), 3) for v in bottle[:2]], spawn_yaw_deg=float(np.degrees(yaw0)))
    return bottle, stand_xy, spawn_xy, yaw0, meta


def run_round(sim: Sim, pol: Policy, rnd: int, rng, out_dir, video, ego, wrist, recorder=None):
    m = dict(round=rnd, ok=False, why="", close_trigger=None, stand_attempts=[])
    # --- scene sample ---------------------------------------------------------------------
    bottle, stand_xy, spawn_xy, yaw0, meta = sample_scene(sim, rng)
    m.update(meta); d = meta["stand_d"]
    sim.reset(spawn_xy, yaw0, 0.0)
    sim.d.qpos[sim.crate_q:sim.crate_q + 3] = bottle; mujoco.mj_forward(sim.m, sim.d)
    sim.grip_clip = GRIP_CLIP
    pol.reset(sim.state()); refs = Refs(pol, sim)
    yaw_stand = float(np.arctan2(*(bottle[:2] - stand_xy)[::-1]))
    appr = np.array([np.cos(yaw_stand), np.sin(yaw_stand), 0.0])
    grasp = bottle + np.array([0, 0, -GRIP_BELOW_CENTRE])
    pre = grasp - appr * PRE_GRASP_BACK
    R_palm = rot_z(yaw_stand) @ rot_y(np.radians(-PITCH_FF_DEG))
    BIAS_w = rot_z(yaw_stand) @ BIAS                                  # body-frame servo bias -> world
    trace = []; log = []; T = [0.0]; first_move = [None]; phase = ["settle"]
    stats = {"torso_max": -1e9, "vis": {}, "wvis": {}}; pads_reach = []
    ffm = None
    if video:
        ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "25", "-i", "-",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(out_dir, f"{rnd:02d}.mp4")], stdin=subprocess.PIPE)
        ego_r = mujoco.Renderer(sim.m, height=480, width=640); ego_cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    hand_cmd = [None]        # world pad target of the right hand, or None = at the side

    def build(pel_ref, hand_side, t):
        out = dict(pel_ref)
        out["left_wrist_yaw_link"] = hand_side["left_wrist_yaw_link"]
        if hand_cmd[0] is None:
            out["right_wrist_yaw_link"] = hand_side["right_wrist_yaw_link"]
        else:
            tgt = hand_cmd[0](t)
            cmd = tgt + BIAS_w + np.clip(KP * (tgt - sim.pad("right")), -CAP, CAP)
            out["right_wrist_yaw_link"] = wrist_from_pad(cmd, R_palm)
        return out

    def tick(ref):
        st = sim.state()
        if recorder is not None:
            # obs at t -> the command applied at t (ref(t) is the first future frame)
            recorder.tick(sim, st, ref(T[0]), T[0], phase[0], cmd_next=ref(T[0] + DT))
        sim.step(pol(st, refs.futures(ref, T[0])))
        T[0] += DT; k = len(trace); trace.append(sim.d.qpos.copy())
        bp, bq = sim.crate()
        if first_move[0] is None and np.linalg.norm(bp[:2] - bottle[:2]) > 0.005:
            first_move[0] = (round(T[0], 2), phase[0])
        stats["torso_max"] = max(stats["torso_max"], torso_pitch_deg(sim))
        if phase[0] == "reach":
            pads_reach.append(mat(sim.d.qpos[3:7]).T @ (sim.pad("right") - sim.d.qpos[:3]))   # pelvis frame: the walk's own motion is not the arm's
        if ego is not None and k % EGO_EVERY == 0:
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
        np.savez_compressed(os.path.join(out_dir, "trace", f"wall_s0_{rnd:02d}_state_trace.npz"), qpos=np.array(trace), dt=DT)
        if ffm is not None:
            ffm.stdin.close(); ffm.wait()
        return m

    # --- settle ------------------------------------------------------------------------------
    p0 = sim.d.qpos[:2].copy()
    side0 = refs.hands_at_side(np.array([p0[0], p0[1], PELVIS_Z]), yaw0)
    for _ in range(int(SETTLE_S / DT)):
        if tick(lambda t: build(refs.pelvis(p0, yaw0), side0, t)):
            return finish("[settle] fell")
    # --- walk, reach overlapping the last 0.5 m ---------------------------------------------------
    phase[0] = "walk"
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

    def walk_state(t):
        s = s_of(t - t_walk0); xy = start + u_dir * s
        yaw = yaw_s + wrap(yaw_stand - yaw_s) * smoothstep(s / max(L, 1e-6))
        return xy, yaw, L - s

    def walk_ref(t):
        xy, yaw, _ = walk_state(t)
        return build(refs.pelvis(xy, yaw), refs.hands_at_side(np.array([xy[0], xy[1], PELVIS_Z]), yaw), t)

    reach_t0 = [None]; reach_dur = [None]
    while True:
        xy, yaw, remaining = walk_state(T[0])
        if reach_t0[0] is None and remaining <= REACH_START_DIST:
            # one min-jerk curve from the current pad to the pre-grasp point, arriving ~1 s after the pelvis
            reach_t0[0] = T[0]; reach_dur[0] = max(REACH_MIN_S, (t_walk0 + t_total - T[0]) + 1.0)
            pad0 = sim.pad("right").copy()
            hand_cmd[0] = lambda t, pad0=pad0: pad0 + (pre - pad0) * minjerk((t - reach_t0[0]) / reach_dur[0])
            phase[0] = "reach"
            log.append(f"reach starts at t {T[0]:.1f} with {remaining:.2f} m to go, duration {reach_dur[0]:.1f} s")
        if tick(walk_ref):
            return finish("[walk] fell")
        if reach_t0[0] is not None and T[0] - reach_t0[0] >= reach_dur[0]:
            break
        if T[0] - t_walk0 > t_total + 15:
            return finish("[walk] reach never started")
    pel_xy = sim.d.qpos[:2].copy()
    d_meas = float(np.linalg.norm(bottle[:2] - pel_xy))
    yaw_err = float(np.degrees(wrap(yaw_stand - yaw_of(sim.d.qpos[3:7]))))
    m.update(stand_d_meas=round(d_meas, 3), stand_err_cm=round(float(np.linalg.norm(pel_xy - stand_xy)) * 100, 1), yaw_err_deg=round(yaw_err, 1),
             stand_attempts=[dict(so=d_meas, lat=0.0, yaw=yaw_err)], stand_off_meas=d_meas)
    pk, amax, plen = smoothness(pads_reach)
    e_pre = sim.pad("right") - pre
    m.update(pre_err_cm=[round(float(v) * 100, 1) for v in e_pre], reach_speed_peaks=pk, reach_accel_max=round(amax, 2))
    log.append(f"at pre-grasp: pelvis d {d_meas:.3f} (target {d:.3f}), stand err {m['stand_err_cm']} cm, yaw {yaw_err:+.1f} | pre err {np.round(e_pre * 100, 1)} cm "
               f"| torso {torso_pitch_deg(sim):+.0f} | peaks {pk} accel {amax:.2f}")
    if first_move[0] is not None:
        return finish("[reach] bottle knocked before the entry")
    # --- entry: straight to the grasp point, close on the visible cue ---------------------------------
    phase[0] = "entry"
    pel_hold = refs.pelvis(pel_xy, yaw_of(sim.d.qpos[3:7]))
    side = refs.hands_at_side(np.array([pel_xy[0], pel_xy[1], PELVIS_Z]), yaw_of(sim.d.qpos[3:7]))
    t_e0 = T[0]
    live_grasp = lambda: sim.crate()[0] + np.array([0, 0, -GRIP_BELOW_CENTRE])      # the bottle as it is NOW (what the cameras show)
    hand_cmd[0] = lambda t: pre + (live_grasp() - pre) * smoothstep((t - t_e0) / ENTRY_S)
    hold_ref = lambda t: build(pel_hold, side, t)
    trigger = None
    while T[0] - t_e0 < ENTRY_TIMEOUT:
        if tick(hold_ref):
            return finish("[entry] fell")
        grasp = live_grasp()
        e = sim.pad("right") - grasp
        along = float(e @ appr); across = float(np.linalg.norm(e - appr * along))
        if abs(along) <= CLOSE_TOL_ALONG and across <= CLOSE_TOL_ACROSS:
            trigger = "cue"; break
    if trigger is None:
        trigger = "timeout"
    e = sim.pad("right") - grasp
    m.update(close_trigger=trigger, entry_err_cm=[round(float(v) * 100, 1) for v in e], entry_s=round(T[0] - t_e0, 2))
    log.append(f"close on {trigger} after {T[0] - t_e0:.1f} s, pad err {np.round(e * 100, 1)} cm, torso {torso_pitch_deg(sim):+.0f}")
    if first_move[0] is not None and first_move[0][1] == "entry" and trigger == "timeout":
        m["why"] = "[entry] bottle pushed"
    # --- close -------------------------------------------------------------------------------------------
    phase[0] = "close"
    grasp = live_grasp(); t_c0 = T[0]; hand_cmd[0] = lambda t: grasp
    for _ in range(int((CLOSE_S + CLOSE_HOLD_S) / DT)):
        sim.grip_target[:] = GRIP_OPEN + smoothstep((T[0] - t_c0) / CLOSE_S) * (GRIP_CLOSED - GRIP_OPEN)
        if tick(hold_ref):
            return finish("[close] fell")
    # --- lift, then carry (retract + lower) ----------------------------------------------------------------
    phase[0] = "lift"
    bz0 = sim.crate()[0][2]; g0 = sim.pad("right").copy(); t_l0 = T[0]
    hand_cmd[0] = lambda t: g0 + np.array([0, 0, LIFT_H * smoothstep((t - t_l0) / LIFT_S)])
    for _ in range(int(LIFT_S / DT)):
        if tick(hold_ref):
            return finish("[lift] fell")
    rise_lift = float(sim.crate()[0][2] - bz0)
    phase[0] = "carry"
    c0 = sim.pad("right").copy(); t_k0 = T[0]
    hand_cmd[0] = lambda t: c0 + (-appr * CARRY_BACK + np.array([0, 0, -CARRY_DOWN])) * smoothstep((t - t_k0) / CARRY_S)
    for _ in range(int((CARRY_S + CARRY_HOLD_S) / DT)):
        if tick(hold_ref):
            return finish("[carry] fell")
    bp, bq = sim.crate()
    rise = float(bp[2] - bz0); tilt = float(np.degrees(np.arccos(np.clip(mat(bq)[2, 2], -1, 1))))
    m.update(rise_lift_cm=round(rise_lift * 100, 1), rise_end_cm=round(rise * 100, 1), tilt_end_deg=round(tilt, 1), torso_end_deg=round(torso_pitch_deg(sim), 1))
    log.append(f"lift {rise_lift * 100:.1f} cm; after carry: {rise * 100:.1f} cm up, tilt {tilt:.0f}, torso {torso_pitch_deg(sim):+.0f}")
    if m["why"]:
        return finish()
    if rise < LIFT_OK:
        return finish(f"[carry] bottle only {rise * 100:.1f} cm up at the end")
    if tilt > TILT_OK_DEG:
        return finish(f"[carry] bottle tilted {tilt:.0f} deg")
    m["ok"] = True
    return finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(R.ROOT, "outputs/videos/scalebfm_bottle"))
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "trace"), exist_ok=True)
    pol = Policy(); sim = Sim(pol.meta, video=a.video)
    # pad geoms hardened as in the crate scene (bimanual_scene.py): priority 1, condim 4 (torsional
    # friction resists the bottle pivoting about the jaw axis), stiff solref/solimp. The bottle scene
    # was built without it.
    if os.environ.get("PAD_HARDEN", "1") == "1":
        for g in range(sim.m.ngeom):
            b = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or ""
            if "dex1_finger" in b and sim.m.geom_contype[g] != 0:
                sim.m.geom_priority[g] = 1; sim.m.geom_condim[g] = 4
                sim.m.geom_solref[g] = (0.002, 1.0); sim.m.geom_solimp[g] = (0.995, 0.9999, 0.001, 0.5, 2.0)
    from bottle_probe import Vis
    ego, wrist = Vis(sim, "head_camera"), Vis(sim, "right_wrist")
    rng = np.random.default_rng(a.seed)
    mf = open(os.path.join(a.out, "manifest.jsonl"), "a"); oks = 0
    for r in range(a.rounds):
        t0 = time.time()
        m = run_round(sim, pol, r, rng, a.out, a.video, ego, wrist)
        oks += m["ok"]
        print(f"round {r}: {'OK' if m['ok'] else '--'} {m['why']} | d {m.get('stand_d_meas', float('nan'))} (target {m['stand_d']:.3f}) stand err {m.get('stand_err_cm')} yaw {m.get('yaw_err_deg')} "
              f"| close {m.get('close_trigger')} rise {m.get('rise_end_cm')} tilt {m.get('tilt_end_deg')} | torso max {m['torso_pitch_max_deg']} "
              f"| ego {m['ego_vis']} wrist {m['wrist_vis']} | {m['len_s']:.0f} s sim, {time.time() - t0:.0f} s wall", flush=True)
        for line in m["log"]:
            print("   ", line, flush=True)
        mf.write(json.dumps(m) + "\n"); mf.flush()
    print(f"OK {oks}/{a.rounds}")


if __name__ == "__main__":
    main()

"""Single-hand bottle reach probe for ScaleBFM (mode 2): precision + visibility.

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 $VENV_SIM_PY src/teleop_free/scalebfm/bottle_probe.py \
        --out outputs/reach/scalebfm_bottle --mode open            # A: open loop, measures the bias
    ... --mode ff --bias outputs/reach/scalebfm_bottle/open.json    # B: feed-forward bias + proportional

Scene: outputs/scene/sim (water bottle on the 0.62 table, 34 cm inside the edge).
For each placement (pelvis distance to the bottle d, lateral offset) the robot is
reset standing, faces the bottle, and the RIGHT pad follows a min-jerk curve
(3 s) from its side position to the pre-grasp point (grasp point - 0.10 m along
the approach), then a straight 2 s entry to the grasp point (bottle centre
- 2 cm, level palm, jaw horizontal). Pelvis pinned, no pelvis servo, no integral
anywhere: mode 'ff' adds ONE global feed-forward offset (from the open-loop
run) plus a proportional overdrive on the measured pad error (gain KP, cap 5 cm)
-- both are functions of what is visible, no hidden state.
Records per placement: pad error at pre-grasp and grasp, torso pitch, pelvis
shift, bottle displacement, bottle pixel fraction in head_camera / right_wrist.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_spike as R  # noqa: E402

R.SCENE = os.environ.get("SCENE_DIR", os.path.join(R.ROOT, "outputs/scene/sim"))
R.OBJ = "water_bottle"
from run_spike import DT, OFF, PELVIS_Z, SETTLE_S, rot_z, mat, smoothstep, wrist_from_pad, Sim, Policy, Refs  # noqa: E402

DISTS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
LATS = (0.0, 0.15)                 # pelvis y relative to the bottle (+ = bottle on the robot's right side... see below)
GRIP_BELOW_CENTRE = 0.02
PRE_GRASP_BACK = 0.10
REACH_S, ENTRY_S = 3.0, 2.0
KP, CAP = float(os.environ.get("KP", 0.7)), 0.05
PITCH_FF_DEG = float(os.environ.get("PITCH_FF", 0.0))     # command the palm this much nose-UP to cancel the delivered nose-down pitch


def minjerk(u):
    u = float(np.clip(u, 0, 1))
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class Vis:
    def __init__(self, sim, cam):
        self.sim = sim
        self.r = mujoco.Renderer(sim.m, height=240, width=320); self.r.enable_segmentation_rendering()
        self.cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        self.opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(self.opt)
        self.obj_g = np.array([g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] == sim.crate_b])
        root = sim.m.body_rootid[sim.bid["pelvis"]]
        self.robot_g = np.array([g for g in range(sim.m.ngeom) if sim.m.body_rootid[sim.m.geom_bodyid[g]] == root])

    def __call__(self):
        m, d = self.sim.m, self.sim.d
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt); a = int(np.isin(self.r.render()[..., 0], self.obj_g).sum())
        saved = m.geom_group[self.robot_g].copy(); m.geom_group[self.robot_g] = 5; self.opt.geomgroup[5] = 0
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt); b = int(np.isin(self.r.render()[..., 0], self.obj_g).sum())
        m.geom_group[self.robot_g] = saved; self.opt.geomgroup[5] = 1
        return a / max(b, 1), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R.ROOT, "outputs/reach/scalebfm_bottle"))
    ap.add_argument("--mode", choices=("open", "ff"), default="open")
    ap.add_argument("--bias", default="")
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bias = np.zeros(3)
    if a.mode == "ff":
        rows0 = json.load(open(a.bias))
        bias = -np.mean([r["err_grasp_m"] for r in rows0 if not r["fallen"]], axis=0)   # command = ref + bias
        print("feed-forward bias (cm):", np.round(bias * 100, 1))
    pol = Policy(); sim = Sim(pol.meta, video=a.video)
    vis_ego, vis_wrist = Vis(sim, "head_camera"), Vis(sim, "right_wrist")
    bottle0 = sim.crate0.copy()
    ffm = None
    if a.video:
        ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "25", "-i", "-",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(a.out, f"{a.mode}.mp4")], stdin=subprocess.PIPE)
        ego_r = mujoco.Renderer(sim.m, height=480, width=640); ego_cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    rows = []
    k = [0]
    for lat in LATS:
        for dist in DISTS:
            # robot on the -x side of the bottle, facing it; lateral offset moves the pelvis so the
            # bottle sits in front of the right shoulder (lat 0.15) or straight ahead (lat 0)
            pel_xy = np.array([bottle0[0] - dist, bottle0[1] + lat])
            yaw = float(np.arctan2(bottle0[1] - pel_xy[1], bottle0[0] - pel_xy[0]))
            sim.reset(pel_xy, yaw, 0.0)
            pol.reset(sim.state()); refs = Refs(pol, sim)
            appr = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            grasp = bottle0 + np.array([0, 0, -GRIP_BELOW_CENTRE])
            pre = grasp - appr * PRE_GRASP_BACK
            R_palm = rot_z(yaw) @ R.rot_y(np.radians(-PITCH_FF_DEG))   # level palm (+ optional nose-up feed-forward), x toward the bottle
            pel = refs.pelvis(pel_xy, yaw)
            side = refs.hands_at_side(np.array([pel_xy[0], pel_xy[1], PELVIS_Z]), yaw)
            T = [0.0]; fallen = False
            pad0 = [None]

            def tick(ref):
                st = sim.state()
                sim.step(pol(st, refs.futures(ref, T[0])))
                T[0] += DT; k[0] += 1
                if ffm is not None and k[0] % 2 == 0:
                    ego_r.update_scene(sim.d, camera=ego_cam)
                    ffm.stdin.write(np.concatenate([sim.frame(), ego_r.render()], axis=1).tobytes())
                return sim.fallen()

            def corrected(target_pad_fn):
                """ref with the right pad at target(t) + bias + KP*(target - measured), capped."""
                def f(t):
                    tgt = target_pad_fn(t)
                    cmd = tgt + bias
                    if a.mode == "ff":
                        cmd = cmd + np.clip(KP * (tgt - sim.pad("right")), -CAP, CAP)
                    out = dict(pel); out["left_wrist_yaw_link"] = side["left_wrist_yaw_link"]
                    out["right_wrist_yaw_link"] = wrist_from_pad(cmd, R_palm)
                    return out
                return f

            for _ in range(int(SETTLE_S / DT)):
                fallen = tick(lambda t: {**pel, **side}) or fallen
            vis0 = vis_ego()[0]
            pad0[0] = sim.pad("right").copy(); t0 = T[0]
            reach_pad = lambda t: pad0[0] + (pre - pad0[0]) * minjerk((t - t0) / REACH_S)
            for _ in range(int(REACH_S / DT)):
                fallen = tick(corrected(reach_pad)) or fallen
            e_pre = sim.pad("right") - pre
            t1 = T[0]
            entry_pad = lambda t: pre + (grasp - pre) * smoothstep((t - t1) / ENTRY_S)
            for _ in range(int(ENTRY_S / DT)):
                fallen = tick(corrected(entry_pad)) or fallen
            # short hold (0.5 s) to read the settled error -- the entry ends at speed 0 anyway
            for _ in range(int(0.5 / DT)):
                fallen = tick(corrected(lambda t: grasp)) or fallen
            e_g = sim.pad("right") - grasp
            Rw = mat(sim.body("right_wrist_yaw_link")[1])
            pitch = float(np.degrees(np.arcsin(-Rw[2, 0])))          # palm x axis pitch (0 = level)
            bp, _ = sim.crate()
            torso = float(np.degrees(np.arcsin(-mat(sim.body("torso_link")[1])[2, 0])))
            pel_d = sim.d.qpos[:3] - np.array([pel_xy[0], pel_xy[1], PELVIS_Z])
            ve, vw = vis_ego(), vis_wrist()
            row = dict(dist=dist, lat=lat, err_pre_cm=[round(float(v) * 100, 1) for v in e_pre], err_grasp_m=[float(v) for v in e_g],
                       err_grasp_cm=round(float(np.linalg.norm(e_g)) * 100, 1), palm_pitch_deg=round(pitch, 1), torso_deg=round(torso, 1),
                       pelvis_shift_cm=[round(float(v) * 100, 1) for v in pel_d], bottle_moved_cm=round(float(np.linalg.norm(bp[:2] - bottle0[:2])) * 100, 1),
                       vis_ego_settle=round(vis0, 2), vis_ego_grasp=round(ve[0], 2), vis_wrist_grasp=round(vw[0], 2), wrist_px=vw[1], fallen=bool(fallen))
            rows.append(row)
            print(f"d {dist:.2f} lat {lat:+.2f}: pre err {row['err_pre_cm']}  grasp err {np.round(e_g * 100, 1)} |{row['err_grasp_cm']:4.1f}| cm  "
                  f"palm pitch {pitch:+.0f}  torso {torso:+.0f}  pelvis d {row['pelvis_shift_cm']}  bottle moved {row['bottle_moved_cm']}  "
                  f"vis ego {vis0:.2f}->{ve[0]:.2f} wrist {vw[0]:.2f} ({vw[1]} px){'  FELL' if fallen else ''}", flush=True)
    if ffm is not None:
        ffm.stdin.close(); ffm.wait()
    json.dump(rows, open(os.path.join(a.out, os.environ.get("TAG", a.mode) + ".json"), "w"), indent=1)
    ok = [r for r in rows if not r["fallen"]]
    print(f"\n{a.mode}: grasp err median {np.median([r['err_grasp_cm'] for r in ok]):.1f} cm, max {max(r['err_grasp_cm'] for r in ok):.1f}; "
          f"<=1.5 cm in {sum(r['err_grasp_cm'] <= 1.5 for r in ok)}/{len(ok)}; bottle knocked (>1 cm) in {sum(r['bottle_moved_cm'] > 1 for r in ok)}")


if __name__ == "__main__":
    main()

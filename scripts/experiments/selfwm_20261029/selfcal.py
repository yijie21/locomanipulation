"""Selfwm item 9, offline: the head camera's calibration error from the robot's own body in its depth image.

The robot knows its joint angles and its own model, so it knows where every part of its body is relative to the head
camera's mount. Rendering the robot alone (robot.xml, no scene) from the camera pose it believes gives the depth it
expects to see of itself; the head camera's real depth shows the same body through the true mount. A rotation and a
depth scale that carry the seen body onto the expected one are the calibration error (perceive.RenderConfig's
head_err_deg / head_scale: the believed orientation is the true one times dR, the depth times s). In the believed
camera frame a seen point is p_hat = s p and the expected one dR^T p, so the fit gives a, Q with a Q p_hat = dR^T p:
Q = dR^T, a = 1 / s, the same for every frame of a round (the mount does not move).

Which seen pixels are the body:
  seg   the renderer's body ids (the simulator's truth, not on a real robot): the upper bound;
  self  the expected render's mask, dilated by DILATE_PX, then trimmed point-to-plane ICP with shrinking gates: only what
        the robot has (joint angles, its model, the believed mount).

Replays recorded rounds (their .npz: pelvis, joints, fingers, the task object) on the physics scene, draws one error
per round (an axis across the view, a sign for the scale, as perceive does), and reports per frame and per round how
far the error leaves a point off before and after the correction: at the task object, and 0.55 m along the view axis.

What a real head camera adds (--real): depth noise growing with range (sigma = 1 mm + 4 mm x z^2, a D435's order), 20 %
of the pixels without depth and 50 % on the robot's black plastic, joint encoders off by a fixed 0.5 deg-std offset per
joint per round plus 0.1 deg of jitter (the expected body is posed from these readings), and --lag-ms of image latency
(the image is older than the joint reading it is compared with). Under encoder offsets no calibration makes an object
reading true in the room; what a grasp needs is the object against the hand as the robot believes it, so `rel` is the
error of (object read - hand by the joints read) against (object - hand) truly, at the hand nearer the object.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY selfcal.py <rounds dir> [--deg 2 --scale 0.02 --every 0.5 --limit 0 --out <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src")]

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.tasks.scene.perceive import _mat_quat, _rotvec_mat, intrinsics  # noqa: E402
from locomani.tasks.scene.selfcal import (  # noqa: E402
    DILATE_PX,
    MAX_RANGE,
    MAX_SEEN,
    MIN_POINTS,
    cam_grid,
    fit,
    grid_normals,
    noisy_depth,
)

PHYS = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars_physics"
W, H = 320, 240  # RenderConfig's default
CAM = "head_camera"
GRIP = (
    "left_hand_gripper_1_joint",
    "left_hand_gripper_2_joint",
    "right_hand_gripper_1_joint",
    "right_hand_gripper_2_joint",
)
PROBE_M = 0.55
HAND = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}


class Real:
    """The real-camera knobs of one round (see the module docstring); all zero: the simulator's exact views."""

    def __init__(self, seed: int, on: bool, lag_ms: float, n_joints: int):
        self.rng = np.random.default_rng([seed, 0xCA1])
        self.on, self.lag_ms = on, lag_ms
        self.enc = np.radians(0.5) * self.rng.standard_normal(n_joints) if on else np.zeros(n_joints)

    def joints(self, q):
        return q + self.enc + (np.radians(0.1) * self.rng.standard_normal(len(q)) if self.on else 0.0)

    def depth(self, depth, robot_px):
        return noisy_depth(depth, robot_px, self.rng) if self.on else depth


class View:
    """The head camera of one MuJoCo model: the scene (what the camera truly sees) or the robot alone (what the robot
    expects to see of itself), posed from a trace row."""

    def __init__(self, xml: Path, robot_only: bool):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.jq = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.gq = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in GRIP])
        self.cid = self.m.camera(CAM).id
        self.q0 = self.m.cam_quat[self.cid].copy()
        pelvis = self.m.body("pelvis").id
        self.robot = np.array([b for b in range(self.m.nbody) if self._under(b, pelvis)])
        if robot_only:  # the floor plane stays in robot.xml: put it out of range
            for g in np.nonzero(self.m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0]:
                self.m.geom_pos[g, 2] = -50.0
        self.r = mujoco.Renderer(self.m, H, W, max_geom=int(self.m.ngeom) + 1000)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.K = intrinsics(float(self.m.cam_fovy[self.cid]), W, H)

    def _under(self, b: int, root: int) -> bool:
        while b > 0:
            if b == root:
                return True
            b = int(self.m.body_parentid[b])
        return False

    def pose(self, root, joints, fingers, objs: dict | None = None, dR=None) -> None:
        d = self.d
        d.qpos[:7], d.qpos[self.jq], d.qpos[self.gq] = root, joints, fingers
        for name, x in (objs or {}).items():
            j = self.m.body(name).jntadr[0]
            if j >= 0:
                a = self.m.jnt_qposadr[j]
                d.qpos[a : a + 7] = x
        q = self.q0.copy()
        if dR is not None:  # the camera as believed: true orientation times dR
            mujoco.mju_mulQuat(q, self.q0, _mat_quat(dR))
        self.m.cam_quat[self.cid] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_camlight(self.m, d)

    def shoot(self):
        self.r.update_scene(self.d, camera=self.cid, scene_option=self.opt)
        self.r.enable_depth_rendering()
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        self.r.enable_segmentation_rendering()
        seg = self.r.render().copy()
        self.r.disable_segmentation_rendering()
        geom = seg[..., 0]
        is_geom = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom >= 0)
        body = np.where(is_geom, self.m.geom_bodyid[np.clip(geom, 0, self.m.ngeom - 1)], -1)
        c = self.d.cam_xpos[self.cid].copy()
        M = self.d.cam_xmat[self.cid].reshape(3, 3).copy()
        return depth, body, c, M


def off_cm(p: np.ndarray, dR: np.ndarray, s: float, Q=None, a: float = 1.0) -> float:
    """How far a point at true camera-frame position p lands off: read through the believed mount (s dR p), then
    corrected by (Q, a) when given."""
    x = s * dR @ p if Q is None else a * s * dR @ Q @ p
    return 100 * float(np.linalg.norm(x - p))


def draw_error(seed: int, deg: float, scale: float):
    rng = np.random.default_rng([seed, 0x5E1F])
    ang = rng.uniform(0.0, 2 * np.pi)
    axis = np.array([np.cos(ang), np.sin(ang), 0.0])
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    return _rotvec_mat(np.radians(deg) * axis), 1.0 + sign * scale


def frame_inputs(scene: View, robot: View, row: dict, seen_row: dict, dR: np.ndarray, s: float, real: Real):
    """Seen (the image of `seen_row`, older than `row` by the latency) and expected (the joints read at `row`)
    head-camera views; the per-mode fit inputs, the expected body, the true camera frame and the hands."""
    scene.pose(seen_row["root"], seen_row["joints"], seen_row["fingers"], seen_row["objs"])
    depth, body, c, Rc = scene.shoot()
    robot_px = np.isin(body, scene.robot)
    depth = real.depth(depth, robot_px)
    q_read = real.joints(row["joints"])
    robot.pose(row["root"], q_read, row["fingers"], dR=dR)
    edepth, ebody, _, _ = robot.shoot()
    hand_read = {k: robot.d.xpos[robot.m.body(n).id].copy() for k, n in HAND.items()}
    robot.pose(seen_row["root"], seen_row["joints"], seen_row["fingers"])
    hand_true = {k: robot.d.xpos[robot.m.body(n).id].copy() for k, n in HAND.items()}
    seen = cam_grid(depth * s, scene.K)  # what the robot reads: the image, its depth scaled
    exp = cam_grid(edepth, robot.K)  # the expected body in the believed camera frame
    evalid = (ebody >= 0) & (edepth > 0) & (edepth < MAX_RANGE)
    en, eok = grid_normals(exp, evalid)
    svalid = (depth > 0) & (depth < MAX_RANGE)
    masks = dict(seg=svalid & robot_px, self=svalid & binary_dilation(evalid, iterations=DILATE_PX))
    pick = np.random.default_rng(row["i"])  # at most MAX_SEEN seen points per mode, as the online estimate keeps
    out = {}
    for k, m in masks.items():
        pts = seen[m]
        out[k] = pts[pick.choice(len(pts), MAX_SEEN, replace=False)] if len(pts) > MAX_SEEN else pts
    return out, (exp[eok], en[eok]), c, Rc, int(masks["seg"].sum()), hand_read, hand_true


def load_round(js: Path, every: float):
    r = json.loads(js.read_text())
    z = np.load(js.with_suffix(".npz"), allow_pickle=True)
    t = z["t"]
    obj = [k for k in z.files if k.startswith("obj_")]
    steps = [(s.get("step"), float(s.get("t0", 0)), float(s.get("t1", 0))) for s in r.get("steps", [])]

    def row_at(i):
        i = int(np.clip(i, 0, len(t) - 1))
        return dict(i=i, t=float(t[i]), root=z["robot"][i].astype(float), joints=z["joints"][i].astype(float),
                    fingers=z["fingers"][i].astype(float), objs={k[4:]: z[k][i].astype(float) for k in obj})  # fmt: skip

    rows = []
    for i in np.unique(np.searchsorted(t, np.arange(t[0], t[-1], every))):
        i = min(int(i), len(t) - 1)
        step = next((n for n, a, b in reversed(steps) if a <= t[i] <= b), "")
        rows.append(dict(row_at(i), step=step))
    return r, rows, row_at


def rel_cm(x_obj, hand_read, hand_true, c, Rc, dR, s, Q=None, a=1.0) -> float:
    """Error of (object read - hand by the joints read) against (object - hand) truly, at the hand nearer the object."""
    k = min(hand_true, key=lambda h: np.linalg.norm(hand_true[h] - x_obj))
    p = Rc.T @ (x_obj - c)
    read = s * dR @ p if Q is None else a * s * dR @ Q @ p  # in the true camera frame
    return 100 * float(np.linalg.norm((c + Rc @ read - hand_read[k]) - (x_obj - hand_true[k])))


MODES = ("seg", "self")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", type=Path)
    ap.add_argument("--deg", type=float, default=2.0)
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--lag-ms", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    scene, robot = View(PHYS / "scene_locomani.xml", False), View(PHYS / "robot.xml", True)
    files = sorted(a.rounds.glob("pick_*.json"))[: a.limit or None]
    out = []
    t0 = time.time()
    probe = np.array([0.0, 0.0, -PROBE_M])
    for k, js in enumerate(files):
        rec, rows, row_at = load_round(js, a.every)
        seed = int(js.stem.rsplit("_s", 1)[1])
        dR, s = draw_error(seed, a.deg, a.scale)
        real = Real(seed, a.real, a.lag_ms, len(MUJOCO_JOINT_NAMES))
        lag = int(round(a.lag_ms / 20.0))  # the trace is 50 Hz
        fused = {m: [] for m in MODES}
        frows, objs = [], []
        for row in rows:
            seen, (ep, en), c, Rc, n_seg, h_read, h_true = frame_inputs(
                scene, robot, row, row_at(row["i"] - lag), dR, s, real
            )
            x_obj = next(iter(row["objs"].values()))[:3] if row["objs"] else None
            p_obj = Rc.T @ (x_obj - c) if x_obj is not None else None
            fr = dict(label=rec["label"], t=row["t"], step=row["step"], n_seg=n_seg,
                      obj_m=round(float(np.linalg.norm(p_obj)), 3) if p_obj is not None else None,
                      unc_obj=off_cm(p_obj, dR, s) if p_obj is not None else None, unc_probe=off_cm(probe, dR, s),
                      unc_rel=rel_cm(x_obj, h_read, h_true, c, Rc, dR, s) if x_obj is not None else None)  # fmt: skip
            if x_obj is not None:
                objs.append((x_obj, h_read, h_true, c, Rc, row["step"]))
            for m in MODES:
                if n_seg >= MIN_POINTS and len(seen[m]) >= MIN_POINTS and len(ep) >= MIN_POINTS:
                    fused[m].append((seen[m], ep, en))
                    Q, aa, info = fit([fused[m][-1]])
                    fr[m] = dict(probe=off_cm(probe, dR, s, Q, aa), n=info.get("n", 0), rms_mm=info.get("rms_mm"),
                                 obj=off_cm(p_obj, dR, s, Q, aa) if p_obj is not None else None,
                                 rel=rel_cm(x_obj, h_read, h_true, c, Rc, dR, s, Q, aa) if x_obj is not None else None)  # fmt: skip
            frows.append(fr)
        rnd = dict(label=rec["label"], ok=rec.get("ok"), err_deg=a.deg, scale=s, frames=len(rows),
                   body_frames=len(fused["seg"]), unc_probe=off_cm(probe, dR, s))  # fmt: skip
        near = [o for o in objs if o[5] in ("pregrasp", "rel_entry", "hand_entry", "entry", "descend", "grip")]
        rnd["unc_rel_near"] = float(np.median([rel_cm(*o[:5], dR, s) for o in near])) if near else None
        for m in MODES:  # one estimate from the round's frames that saw the body (at most about 12)
            if fused[m]:
                sub = fused[m][:: max(1, len(fused[m]) // 12)]
                Q, aa, info = fit(sub)
                rnd[m] = dict(probe=off_cm(probe, dR, s, Q, aa), frames=len(sub),
                              angle_deg=float(np.degrees(np.arccos(np.clip((np.trace(Q @ dR) - 1) / 2, -1, 1)))),
                              scale_err=float(aa * s - 1), n=info.get("n"), rms_mm=info.get("rms_mm"),
                              rel_near=float(np.median([rel_cm(*o[:5], dR, s, Q, aa) for o in near])) if near else None)  # fmt: skip
        out.append(dict(round=rnd, frames=frows))
        print(f"[{k + 1}/{len(files)} {time.time() - t0:.0f}s] {rec['label']}: body in {rnd['body_frames']}/{len(rows)}; "
              f"0.55 m point off {rnd['unc_probe']:.2f} cm -> seg {rnd.get('seg', {}).get('probe', float('nan')):.2f} "
              f"self {rnd.get('self', {}).get('probe', float('nan')):.2f}; object vs hand near the grasp "
              f"{rnd['unc_rel_near'] or float('nan'):.2f} -> self {rnd.get('self', {}).get('rel_near') or float('nan'):.2f}",
              flush=True)  # fmt: skip
    if a.out:
        a.out.write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()}, rounds=out), default=float))
    summarize(out)


def q(v, p):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, p)) if v else float("nan")


def summarize(out: list) -> None:
    fr = [f for o in out for f in o["frames"]]
    body = [f for f in fr if "seg" in f]
    print(f"\n{len(out)} rounds, {len(fr)} frames, body seen (>= {MIN_POINTS} points) in {len(body)}")
    print("single frame (median / p90, cm):")
    for key, name in (("probe", f"point {PROBE_M} m ahead"), ("obj", "task object"), ("rel", "object vs hand")):
        line = f"  {name:22s} uncorrected {q([f['unc_' + key] for f in body], 50):.2f} / {q([f['unc_' + key] for f in body], 90):.2f}"
        for m in MODES:
            line += f" | {m} {q([f[m][key] for f in body], 50):.2f} / {q([f[m][key] for f in body], 90):.2f}"
        print(line)
    rs = [o["round"] for o in out]
    print("one estimate per round (median / max):")
    print(f"  point {PROBE_M} m ahead uncorrected {q([r['unc_probe'] for r in rs], 50):.2f}; object vs hand near the "
          f"grasp uncorrected {q([r['unc_rel_near'] for r in rs], 50):.2f} / {q([r['unc_rel_near'] for r in rs], 100):.2f}")  # fmt: skip
    for m in MODES:
        v = [r[m] for r in rs if m in r]
        if v:
            print(f"  {m:4s}: point {q([x['probe'] for x in v], 50):.2f} / {q([x['probe'] for x in v], 100):.2f}; "
                  f"object vs hand {q([x['rel_near'] for x in v], 50):.2f} / {q([x['rel_near'] for x in v], 100):.2f}; "
                  f"angle left {q([x['angle_deg'] for x in v], 50):.2f} deg, scale left "
                  f"{100 * q([abs(x['scale_err']) for x in v], 50):.2f} %, fit rms {q([x['rms_mm'] for x in v], 50):.1f} mm")  # fmt: skip
    steps = {}
    for f in fr:
        steps.setdefault(f["step"] or "-", []).append("seg" in f)
    print("  body seen by step: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(steps.items())))


if __name__ == "__main__":
    main()

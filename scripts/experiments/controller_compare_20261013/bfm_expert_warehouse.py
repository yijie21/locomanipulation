"""Warehouse L2, ScaleBFM side (TODO 25.7, plan §6): the production four-edge expert's program with ScaleBFM as the
controller.

Only the controller changes. A round starts where SONIC's round of the same seed started (warehouse_states.py's file
of it: the robot, the box). Then:
* walk-in -- ScaleBFM's plan program to the expert's stance: the box centre `visual.stand_off` -
  `visual.stand_depth_tolerance` ahead of the pelvis (the near edge of the band the expert accepts, where SONIC's guard
  walk lands; `--stance centre` = stand_off), unless the guard binds first (the frontmost low body point below the platform top `visual.guard_margin` before the
  near line, the nearer of the platform's and the box's near face), laterally on the box,
  facing the edge, at `visual.walk_speed`, with the brake feedforward of TODO 24.7.1 (bfm_brake --ff speed);
* the lifted ready -- BimanualIK.lifted_ready(visual.ready_tool, visual.ready_pitch_deg) from the measured joints,
  smoothstep-blended over visual.ready_blend_seconds (expert.prepare_stationary_ready, compact), held until both
  wrists are within 3 cm of it (at most 2 s more: the expert's quiet admission waits for SONIC the same way);
* the reach -- the expert's CoordinatedReach unchanged (reach, align, descend, close, lift, stand, hold; its own gates,
  retries, hand feedback and success rule), fed this sim's privileged state (PrivilegedPublisher.state()) and the
  assisted pose estimate, as attempt.reach_step feeds it on SONIC.
Every tick the expert's 29-joint body reference and root attitude, FK'd on the scene's robot at the solver's pelvis
(its velocity carries the six future frames), are ScaleBFM's keypoints -- the mode picks which; its hand closure drives
the Dex1 jaws HAND_OPEN -> visual.close_target as on SONIC. Output: warehouse_states.py's round format (+ events of the
expert's phases), so warehouse_measure.py reads both controllers alike.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/controller_compare_20261013/bfm_expert_warehouse.py \\
        outputs/controller_compare_20261013/l2/warehouse/sonic/states --mode 6 \\
        --config configs/experiments/controller_compare_l2_warehouse_sonic_20261013.yaml \\
        --out outputs/controller_compare_20261013/l2/warehouse/bfm_m6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "experiments" / "wbc_compare_20261012"), str(REPO / "src")]

from bfm_brake import FF_SHORT, FF_SPEEDS, Profile, frontmost_low  # noqa: E402
from bfm_warehouse import (
    ARM_BODIES,
    SERVO_CAP,
    SERVO_CAP_ROT,
    SERVO_KI,
    SERVO_KI_ROT,
    WALK_CAP,
    WALK_KI,
    Recorder,
    expmap,
    rotvec,
)  # noqa: E402
from bottle_states import save  # noqa: E402
from w0_admission import seat  # noqa: E402
from warehouse_states import BOX, SCENE, square_yaw  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.sim.privileged_pub import PrivilegedPublisher  # noqa: E402
from locomani.tasks.crate.bimanual_ik import BimanualIK  # noqa: E402
from locomani.tasks.crate.config import CrateTaskConfig  # noqa: E402
from locomani.tasks.crate.coordinated_reach import CoordinatedReach, CoordinatedTarget  # noqa: E402
from locomani.tasks.crate.grasp_region import RimRegion, grasp_points_for  # noqa: E402

HAND_OPEN = 0.0245  # dex1 jaw (agent_server._hand_poses)
KEY = "nor_cal_box"
YAW_TURN_S = 1.5
TIMEOUT_S = 60.0


class Estimate:
    """The assisted pose estimate (simulation pose in the pelvis frame), what CoordinatedReach.step reads."""

    def __init__(self, raw, key, t):
        base = R.mat(raw["pelvis"]["quat"])
        origin = np.asarray(raw["pelvis"]["pos"], float)
        obj = raw["objects"][key]
        self.rotation = base.T @ R.mat(obj["quat"])
        self.position = base.T @ (np.asarray(obj["pos"], float) - origin)
        self.timestamp = t

    def transform(self, points_local):
        return np.asarray(points_local, float) @ self.rotation.T + self.position


class Body:
    """FK of the scene's robot: pelvis pose + 29 joints -> ScaleBFM's keypoints."""

    def __init__(self, model, names):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.bids = {n: model.body(n).id for n in names}

    def keypoints(self, pos, quat, joints):
        self.d.qpos[:3], self.d.qpos[3:7], self.d.qpos[self.qadr] = pos, quat, joints
        mujoco.mj_kinematics(self.m, self.d)
        return {n: (self.d.xpos[b].copy(), self.d.xquat[b].copy()) for n, b in self.bids.items()}


def futures(body, pos, quat, joints, velocity, fidx):
    out = {}
    for j in fidx:
        kp = body.keypoints(pos, quat, joints + velocity * j * R.DT)
        for n, (p, q) in kp.items():
            out.setdefault(n, ([], []))
            out[n][0].append(p)
            out[n][1].append(q)
    return {n: (np.array(v[0]), np.array(v[1])) for n, v in out.items()}


def pad_servo(sim, fut, mem):
    """25.3's hand servo (bfm_warehouse --servo pad) on top of the expert's reference: an integral on the jaw centre
    (Ki SERVO_KI, cap SERVO_CAP per axis) and on the wrist attitude (SERVO_KI_ROT, cap SERVO_CAP_ROT) toward the
    reference's, shifting the arm's keypoints; the reference itself is what is recorded."""
    out = dict(fut)
    for side, arm in ARM_BODIES.items():
        w = arm[-1]
        if w not in out:
            continue
        p_ref, R_ref = out[w][0][0], R.mat(out[w][1][0])
        p_meas, q_meas = sim.body(w)
        R_meas = R.mat(q_meas)
        err = (p_ref + R_ref @ R.OFF) - (p_meas + R_meas @ R.OFF)
        e_rot = rotvec(R_ref @ R_meas.T)
        turn = mem.get(("turn", side), np.zeros(3)) + SERVO_KI_ROT * R.DT * e_rot
        n = np.linalg.norm(turn)
        turn = turn * (SERVO_CAP_ROT / n) if n > SERVO_CAP_ROT else turn
        mem[("turn", side)] = turn
        Rt = expmap(turn)
        quats = np.array([R.quat_wxyz(Rt @ R.mat(q)) for q in out[w][1]])
        shift = np.array([R.mat(q0) @ R.OFF - Rt @ R.mat(q0) @ R.OFF for q0 in out[w][1]])
        lead = np.clip(mem.get(("pos", side), np.zeros(3)) + SERVO_KI * R.DT * err, -SERVO_CAP, SERVO_CAP)
        mem[("pos", side)] = lead
        out[w] = (out[w][0] + shift, quats)
        for b in arm:
            if b in out:
                out[b] = (out[b][0] + lead, out[b][1])
    return out


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def run_round(pol, sim, z, f, cfg, a, body, solver, pub, target_meta):
    vis = cfg.visual
    axis = np.asarray(f["edge"]["axis"], float)
    perp = np.array([-axis[1], axis[0]])
    box0 = z["box"][0]
    pel0 = z["pelvis"][0]
    p0, yaw0 = pel0[:2].copy(), R.yaw_of(pel0[3:])
    psi = square_yaw(float(f["heading"]), R.yaw_of(box0[3:]))
    sim.reset(p0, yaw0, 0.0)
    seat(sim)
    sim.d.qpos[sim.crate_q : sim.crate_q + 7] = box0
    mujoco.mj_forward(sim.m, sim.d)
    if a.odo is not None:  # told the true start pose, nothing after (25.5 estimated-pose arm)
        a.odo.reset(sim.d.qpos[:3].copy(), sim.d.qpos[a.odo.qadr].copy(), sim.d.qpos[3:7].copy())
    robot = [g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] not in (0, sim.crate_b) and sim.m.geom_bodyid[g] > 0]
    robot = [g for g in robot if sim.m.body(sim.m.geom_bodyid[g]).rootid == sim.m.body("pelvis").rootid]
    top = float(f["platform_top"])
    # the guard line: the platform's near line or the box's near face, whichever is nearer (expert.run_attempt)
    hx, hy = (float(v) for v in target_meta["box"]["half_extent"][:2])
    by = R.yaw_of(box0[3:])
    local = R.rot_z(by)[:2, :2].T @ axis
    box_line = float(axis @ box0[:2] - abs(local[0]) * hx - abs(local[1]) * hy)
    line = min(float(f["edge"]["line"]), box_line)
    # standing toe reach ahead of the pelvis along the axis, measured facing the edge (the walk turns to psi)
    keep = sim.d.qpos[3:7].copy()
    sim.d.qpos[3:7] = yaw_quat(psi)
    mujoco.mj_kinematics(sim.m, sim.d)
    reach = frontmost_low(sim.m, sim.d, robot, p0, axis, top)
    sim.d.qpos[3:7] = keep
    mujoco.mj_forward(sim.m, sim.d)
    # the expert's stance (visual.stand_off: the box centre that far ahead of the pelvis; stand_depth_tolerance re-walks
    # otherwise), unless the guard binds first (the body's frontmost low point guard_margin before the line)
    # the near edge of the accepted band (stand_off - stand_depth_tolerance): SONIC's guard walk lands there (25.3 W1:
    # pelvis 30 cm from the line, the box centre 0.46 m ahead), and at the band's centre the expert's whole-body
    # references crouch and lean far enough that ScaleBFM stalls or falls (0/10 in modes 2, 6, 7; 25.7 pilot)
    stand = vis.stand_off - vis.stand_depth_tolerance if a.stance == "near" else vis.stand_off
    target_along = min(line - vis.guard_margin - reach, float(axis @ box0[:2]) - stand)
    ff = float(np.interp(vis.walk_speed, FF_SPEEDS, FF_SHORT))  # the plan program stops this much short (24.7.1)
    goal = target_along * axis + float(perp @ box0[:2]) * perp
    dist = float(np.linalg.norm(goal - p0))
    d = (goal - p0) / max(dist, 1e-9)
    prof = Profile(vis.walk_speed, R.WALK_ACC, dist + ff)
    pol.reset(sim.state())
    q_stand, z_stand = sim.d.qpos[body.qadr].copy(), float(sim.d.qpos[2])  # the walk's whole-body keypoints: this pose
    rec = Recorder(sim)
    events = {"walk": R.SETTLE_S}
    st8 = dict(phase="walk", low=None, ready=None, ready_t=None, hold=None, reason="", finished=False)
    jaw = HAND_OPEN
    t = 0.0
    k = 0
    fell = None
    t_wall = time.time()
    while t < TIMEOUT_S:
        t = k * R.DT
        mujoco.mj_forward(sim.m, sim.d)
        if st8["phase"] == "walk":
            fut = {}
            for j in pol.fidx:
                tt = t + j * R.DT
                pw = p0 + d * prof.s(tt - R.SETTLE_S)
                yw = yaw0 + R.smoothstep((tt - R.SETTLE_S) / YAW_TURN_S) * R.wrap(psi - yaw0)
                for n, (p_, q_) in body.keypoints(np.array([pw[0], pw[1], z_stand]), yaw_quat(yw), q_stand).items():
                    fut.setdefault(n, ([], []))
                    fut[n][0].append(p_)
                    fut[n][1].append(q_)
            fut = {n: (np.array(v[0]), np.array(v[1])) for n, v in fut.items()}
            if t >= R.SETTLE_S + prof.t_end:
                # the post-stop pelvis servo of 25.3's ScaleBFM walk (bfm_warehouse --walk-servo): ScaleBFM often leaves
                # the last step out and stops ~10 cm short; an integral push on the keypoints brings it in
                st8["push"] = np.clip(
                    st8.get("push", np.zeros(2)) + WALK_KI * R.DT * (goal - sim.d.qpos[:2]), -WALK_CAP, WALK_CAP
                )
                fut = {n: (p_ + np.append(st8["push"], 0.0), q_) for n, (p_, q_) in fut.items()}
            if "brake" not in events and t - R.SETTLE_S >= prof.t_dec:
                events["brake"] = t
            near = float(np.linalg.norm(goal - sim.d.qpos[:2])) <= 0.015
            if t >= R.SETTLE_S + prof.t_end + max(vis.settle_seconds, 0.5) and (
                near or t >= R.SETTLE_S + prof.t_end + 3.0
            ):
                events["settle"] = t
                st8["phase"] = "ready"
                st8["hold_pose"] = (sim.d.qpos[:3].copy(), yaw_quat(R.yaw_of(sim.d.qpos[3:7])))
                q_meas = sim.d.qpos[body.qadr].copy()
                st8["q0"] = q_meas
                st8["ready"] = solver.lifted_ready(vis.ready_tool, vis.ready_pitch_deg, q_meas).joints.copy()
                st8["ready_t"] = t
                fut = None
        if st8["phase"] == "ready":
            u = min(1.0, (t - st8["ready_t"]) / max(vis.ready_blend_seconds, 1e-3))
            s3 = 3 * u * u - 2 * u**3
            q = st8["q0"] + s3 * (st8["ready"] - st8["q0"])
            v = (6 * u - 6 * u * u) / max(vis.ready_blend_seconds, 1e-3) * (st8["ready"] - st8["q0"])
            pos, quat = st8["hold_pose"]
            fut = futures(body, pos, quat, q, v if u < 1 else np.zeros(29), pol.fidx)
            if t - st8["ready_t"] < vis.ready_blend_seconds + vis.ready_hold_seconds:
                arrived = False
            else:
                kp_now = body.keypoints(pos, quat, st8["ready"])
                arrived = (
                    all(
                        np.linalg.norm(sim.d.xpos[sim.m.body(w).id] - kp_now[w][0]) <= 0.03
                        for w in ("left_wrist_yaw_link", "right_wrist_yaw_link")
                    )
                    or t - st8["ready_t"] >= vis.ready_blend_seconds + 2.0
                )
            if arrived:
                raw = pub.state()
                raw["t"] = t
                st8["low"] = CoordinatedReach(
                    solver,
                    cfg,
                    raw,
                    reference_joints=st8["ready"],
                    reference_quaternion=np.asarray(raw["pelvis"]["quat"], float),
                    target=a.coordinated_target,
                )
                events["handoff"] = t
                st8["phase"] = "reach"
        if st8["phase"] == "reach":
            raw = pub.state()
            raw["t"] = t
            openings = np.array([2 * (np.mean(raw["palms"][s]["hand_q"][:2]) + 0.023) for s in ("left", "right")])
            cmd = st8["low"].step(t, Estimate(raw, KEY, t), raw, openings, motion_time=t)
            if cmd.phase not in events:
                events[cmd.phase] = t
            jaw = HAND_OPEN + cmd.closure * (vis.close_target - HAND_OPEN)
            if cmd.finished:
                st8["reason"], st8["finished"] = cmd.reason, True
                st8["diagnostics"] = {}
                for k_, v_ in st8["low"].diagnostics.items():
                    try:
                        arr = np.asarray(v_, dtype=float)
                    except (TypeError, ValueError):
                        continue
                    st8["diagnostics"][k_] = arr.round(4).tolist()
                events["end"] = t
                break
            low = st8["low"]
            fut = futures(body, low.x[:3], cmd.root_quaternion, cmd.joints, cmd.velocity, pol.fidx)
            if a.servo == "pad":
                fut = pad_servo(sim, fut, st8.setdefault("servo", {}))
        if fut is None:
            k += 1
            continue
        sim.grip_target[:] = jaw
        rec.add(t, fut)
        if sim.fallen():
            fell = t
            break
        sim.step(pol(sim.state(), fut))
        k += 1
    if "end" not in events:
        events["end"] = t
    expert_ok = st8["reason"] == "completed" and fell is None
    facts = dict(
        f,
        controller="scalebfm",
        program=f"expert_m{a.mode}{'p' if a.servo == 'pad' else ''}{a.tag}",
        mode=a.mode,
        events=events,
        walk_target=dict(
            line=line, toe_reach=reach, margin=vis.guard_margin, stand_off=stand, ff=ff, goal=goal.tolist(), psi=psi
        ),
        expert_reason=st8["reason"] or ("fell" if fell is not None else "timeout"),
        expert_diagnostics=st8.get("diagnostics"),
        sonic_ok=f.get("ok"),
        sonic_why=f.get("why"),
        ok=bool(expert_ok),
        why="" if expert_ok else (f"fell at {fell:.2f} s" if fell is not None else st8["reason"] or "timeout"),
        wall_s=round(time.time() - t_wall, 1),
    )
    return rec.arrays(), facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sonic", type=Path, help="warehouse_states.py files of SONIC's rounds (the start of each)")
    ap.add_argument("--config", type=Path, required=True, help="the SONIC run's experiment yaml (task block)")
    ap.add_argument("--mode", type=int, default=6)
    ap.add_argument(
        "--stance", choices=("near", "centre"), default="near",
        help="where in the expert's accepted stance band (visual.stand_off +- stand_depth_tolerance) the walk stops",
    )  # fmt: skip
    ap.add_argument("--servo", choices=("none", "pad"), default="none", help="25.3's jaw + attitude servo on top")
    ap.add_argument("--set", action="append", default=[], help="override a task-config field: block.field=value")
    ap.add_argument(
        "--scene", type=Path, default=SCENE, help="the scene (payload_scenes.py's copies for perturbation e)"
    )
    ap.add_argument(
        "--odometry", action="store_true",
        help="ScaleBFM sees the leg-kinematics + IMU pelvis estimate (backends/scalebfm/odometry.py), not the truth (25.5)",
    )  # fmt: skip
    ap.add_argument("--perturb", default="", help="sim/perturb.py spec (JSON), e.g. the real-robot combo tier")
    ap.add_argument("--only", default="", help="comma-separated round labels")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    task = dict(yaml.safe_load(a.config.read_text())["task"])
    for k_ in ("plugin", "expert"):
        task.pop(k_, None)
    for kv in a.set:  # ScaleBFM's own calibration of a controller-dependent knob (plan: full capability), e.g. visual.tracking_gain=0.5
        k_, v_ = kv.split("=", 1)
        block, field_ = k_.split(".", 1)
        task.setdefault(block, {})[field_] = yaml.safe_load(v_)
    cfg = CrateTaskConfig.from_dict(task)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    target_meta = meta["objects"][KEY]
    grasp_points = grasp_points_for(target_meta, cfg.visual.box_grasp_below_rim)
    region = RimRegion(across=cfg.visual.capture_across) if cfg.visual.capture_across > 0 else RimRegion()
    a.coordinated_target = CoordinatedTarget(
        object_key=KEY, grasp_points_local=grasp_points, rim_above_grasp=cfg.visual.box_grasp_below_rim, region=region
    )
    model = ArmIK(str(a.scene / "scene_locomani.xml"))
    solver = BimanualIK(
        model,
        waist_pitch_max=cfg.carry.waist_pitch_max,
        waist_dofs=cfg.carry.waist_dofs,
        elbow_min_deg=cfg.visual.elbow_min_deg,
    )
    from bfm_bottle import HAND_FORCE, HAND_KD, HAND_KP  # the SONIC grip law (24.1)

    bcfg = ScaleBfmBackendConfig(
        scene_dir=str(a.scene), object_name=KEY, grip_kp=HAND_KP, grip_kd=HAND_KD, grip_clip=HAND_FORCE
    )
    pol = R.Policy(device=a.device, backend=bcfg)
    pol.mode = torch.tensor([a.mode], dtype=torch.long, device=a.device)
    sim = R.Sim(pol.meta, backend=bcfg, perturb=json.loads(a.perturb) if a.perturb else None)
    a.odo = None
    if a.odometry:
        from locomani.backends.scalebfm.odometry import attach

        a.odo = attach(sim, str(a.scene / "scene_locomani.xml"), pol.meta["joint_names"])
    body = Body(sim.m, pol.sel)
    pub = PrivilegedPublisher(
        sim.m, sim.d, {KEY: BOX}, surfaces=[meta["surfaces"]["A"]["body"]], port=56000 + os.getpid() % 3000
    )
    only = set(a.only.split(",")) if a.only else None
    for npz in sorted(a.sonic.glob("*.npz")):
        if only and npz.stem not in only:
            continue
        z = dict(np.load(npz))
        f = json.loads(npz.with_suffix(".json").read_text())
        label = npz.stem.replace("sonic", "bfm", 1)
        if (a.out / f"{label}.json").exists():
            continue
        arrays, facts = run_round(pol, sim, z, f, cfg, a, body, solver, pub, target_meta)
        facts["perturb"] = a.perturb or None
        facts["odometry"] = bool(a.odometry)
        facts["scene"] = str(a.scene)
        save(a.out, label, arrays, facts)
        print(
            f"{label}: ok {facts['ok']} ({facts['expert_reason']}) events {list(facts['events'])} {facts['wall_s']} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

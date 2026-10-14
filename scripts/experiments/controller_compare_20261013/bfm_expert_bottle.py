"""Bottle L2, ScaleBFM side (TODO 25.7, plan §6): the production simple expert's program with ScaleBFM as the
controller.

Only the controller changes. A round starts where SONIC's round of the same seed started (bottle_states.py's file of
it: the robot, the bottle, the expert's own walk plan -- stand point, heading, the table-edge guard band). Then, as
tasks/bottle/simple_policy.py does with `approach_stop: guard` (arm_route ik):
* walk-in -- straight at simple.walk_speed toward the planned stand, turning to its heading, stopping where the
  frontmost body point in the table band is simple.guard_margin_m before the edge (bfm_bottle.walk_round's rule),
  with the brake feedforward of TODO 24.7.1 and 25.3's post-stop pelvis servo;
* the lifted ready -- BimanualIK.lifted_ready(ready_tool, ready_pitch_deg) from the measured joints, smoothstep-blended
  over simple.ready_seconds, held simple.ready_hold_s;
* the grasp -- simple_policy._run's guard branch with its own functions and parameters: SupportedMotion (the expert's
  whole-body IK with its loop prior and measured-tool feedback), the reach polyline to the pre-grasp (re-planned once
  the hand is up), settle, the dash entry with the jaw-axis correction, retreat and re-entry, the close with its
  bilateral-contact gate, lift and recover, and the expert's failure rules (bottle knocked, support reference outside
  tolerance, timeouts). Not ported: the reference clearance guard (`_clear`), which on SONIC can also end a round;
* the production tail (round_phases_approach: carry_and_verify) -- back 0.35 m holding the bottle, hold 2 s, success
  when the bottle moved < 3 cm in the palm frame and stands > 3 cm above the table.
Every tick the expert's 29-joint reference and root, FK'd on the scene's robot at the IK's pelvis, are ScaleBFM's
keypoints (the mode picks which); the right jaw closes HAND_OPEN -> HAND_CLOSED_RIGHT with the expert's closure.
Output: bottle_states.py's round format, so the bottle measurement reads both controllers alike.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/controller_compare_20261013/bfm_expert_bottle.py outputs/controller_compare_20261013/l2/bottle/sonic/states \\
        --config configs/experiments/controller_compare_l2_bottle_sonic_20261013.yaml --mode 2 \\
        --out outputs/controller_compare_20261013/l2/bottle/bfm_m2
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
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "experiments" / "wbc_compare_20261012"), str(REPO / "src")]

from bfm_bottle import HAND_FORCE, HAND_KD, HAND_KP, Recorder  # noqa: E402
from bfm_brake import FF_SHORT, FF_SPEEDS, Profile  # noqa: E402
from bfm_expert_warehouse import Body, futures, yaw_quat  # noqa: E402
from bfm_warehouse import WALK_CAP, WALK_KI  # noqa: E402
from bottle_states import SCENE, save  # noqa: E402
from w0_admission import seat  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.sonic.primitives import HAND_CLOSED_RIGHT, HAND_OPEN, Primitives  # noqa: E402
from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.contacts import bilateral_dex1_contact  # noqa: E402
from locomani.primitives.foot_watch import FootWatch  # noqa: E402
from locomani.primitives.supported_motion import SupportedMotion  # noqa: E402
from locomani.sim.privileged_pub import PrivilegedPublisher  # noqa: E402
from locomani.tasks.bottle.reach_config import ReachConfig  # noqa: E402
from locomani.tasks.bottle.ready_grasp_policy import OFFSET  # noqa: E402
from locomani.tasks.bottle.simple_policy import (  # noqa: E402
    entry_aligned,
    entry_captured,
    entry_command,
    grasp_rotation,
    lateral_correction,
    lean_angle,
    lift_target,
    loop_prior,
    minjerk,
    moving_start,
    object_in_palm,
    polyline_length,
    polyline_point,
    pregrasp_point,
    torso_offset,
    wrap_direction,
)
from locomani.tasks.bottle.walking_reach import NATURAL_RIGHT_ARM  # noqa: E402
from locomani.tasks.crate.bimanual_ik import BimanualIK  # noqa: E402

KEY = "water_bottle"
YAW_TURN_S = 1.5
BACK_M, BACK_SPEED, HOLD_S = 0.35, 0.25, 2.0  # round_phases_approach: back off two steps, then the terminal hold
PALM_DRIFT_MAX, TABLE_CLEAR = 0.03, 0.03
# Every production round walks to the same table edge with the same guard band (all 89 SONIC L2 rounds that planned
# one): a SONIC round that ended before planning its walk (a stand-up or spawn-view failure under the combo tier) gives
# ScaleBFM this plan instead, from the same spawn
SCENE_EDGE = dict(axis=[0.0, -1.0], line=-0.5034890428167202)
SCENE_BAND = [0.5, 0.65]


def tools(raw):
    return np.array(
        [
            np.asarray(raw["palms"][s]["pos"], float) + Primitives._rot(raw["palms"][s]["quat"]) @ OFFSET
            for s in ("left", "right")
        ]
    )


def palm_offset(raw):
    R_p = Primitives._rot(raw["palms"]["right"]["quat"])
    return R_p.T @ (np.asarray(raw["objects"][KEY]["pos"], float) - np.asarray(raw["palms"]["right"]["pos"], float))


class Round:
    def __init__(self, pol, sim, z, f, rc, a, body, ik, pub):
        self.pol, self.sim, self.z, self.f, self.rc, self.a = pol, sim, z, f, rc, a
        self.body, self.ik, self.pub = body, ik, pub
        self.simple, self.wb = rc.simple, rc.whole_body
        self.rec = Recorder(sim)
        self.events = {}
        self.k = 0
        self.jaw = HAND_OPEN[0]
        self.fell = None
        self.stats = {}

    @property
    def t(self):
        return self.k * R.DT

    def raw(self):
        mujoco.mj_forward(self.sim.m, self.sim.d)
        r = self.pub.state()
        r["t"] = self.t
        return r

    def tick(self, fut):
        """One policy step on these futures; False once fallen."""
        self.sim.grip_target[2:] = self.jaw
        self.sim.grip_target[:2] = HAND_OPEN[0]
        self.rec.add(self.t, fut)
        if self.sim.fallen():
            self.fell = self.t
            return False
        self.sim.step(self.pol(self.sim.state(), fut))
        self.k += 1
        return True

    def fail(self, why, **ev):
        self.stats["failure"] = dict(reason=why, **ev)
        return False, why

    # -- walk-in ------------------------------------------------------------------------------------------------------
    def walk(self):
        sim, f, z, simple = self.sim, self.f, self.z, self.simple
        if f["edge"]["line"] is None:
            f["edge"], f["band"] = dict(SCENE_EDGE), list(SCENE_BAND)
            self.stats["walk_plan"] = "scene edge (SONIC's round ended before planning its walk)"
        edge_axis, line = np.asarray(f["edge"]["axis"], float), float(f["edge"]["line"])
        psi, yaw0 = float(f["psi"]), float(f["spawn_yaw"])
        p0 = np.asarray(f["robot_xy"], float)
        sim.reset(p0, psi, 0.0)
        seat(sim)
        sim.d.qpos[sim.crate_q : sim.crate_q + 7] = z["bottle"][0]
        mujoco.mj_forward(sim.m, sim.d)
        st = dict(
            pelvis=dict(pos=sim.d.qpos[:3].copy(), quat=sim.d.qpos[3:7].copy()),
            joints=sim.d.qpos[self.body.qadr].copy(),
        )
        reach = FootWatch(model=sim.m).frontmost_band(st, np.zeros(2), edge_axis, *f["band"])[0] - float(
            edge_axis @ sim.d.qpos[:2]
        )
        stand = np.asarray(f["stand_xy"], float)
        d = (stand - p0) / max(float(np.linalg.norm(stand - p0)), 1e-9)
        dist = (line - simple.guard_margin_m - reach - float(edge_axis @ p0)) / float(edge_axis @ d)
        ff = float(np.interp(simple.walk_speed, FF_SPEEDS, FF_SHORT))
        prof = Profile(simple.walk_speed, R.WALK_ACC, dist + ff)
        sim.reset(p0, yaw0, 0.0)
        seat(sim)
        sim.d.qpos[sim.crate_q : sim.crate_q + 7] = z["bottle"][0]
        mujoco.mj_forward(sim.m, sim.d)
        if self.a.odo is not None:  # told the true start pose, nothing after (25.5 estimated-pose arm)
            self.a.odo.reset(sim.d.qpos[:3].copy(), sim.d.qpos[self.a.odo.qadr].copy(), sim.d.qpos[3:7].copy())
        self.pol.reset(sim.state())
        q_stand, z_stand = sim.d.qpos[self.body.qadr].copy(), float(sim.d.qpos[2])

        def pose_at(tt):
            p = p0 + d * prof.s(tt - R.SETTLE_S)
            yaw = yaw0 + R.smoothstep((tt - R.SETTLE_S) / YAW_TURN_S) * R.wrap(psi - yaw0)
            return np.array([p[0], p[1], z_stand]), yaw_quat(yaw)

        self.events["walk"] = R.SETTLE_S
        goal = p0 + d * dist
        push = np.zeros(2)
        t_stop = R.SETTLE_S + prof.t_end
        while self.t < t_stop + 0.6 or (np.linalg.norm(goal - sim.d.qpos[:2]) > 0.015 and self.t < t_stop + 3.0):
            if "brake" not in self.events and self.t - R.SETTLE_S >= prof.t_dec:
                self.events["brake"] = self.t
            fut = {}
            for j in self.pol.fidx:
                pos, quat = pose_at(self.t + j * R.DT)
                for n, (p_, q_) in self.body.keypoints(pos, quat, q_stand).items():
                    fut.setdefault(n, ([], []))
                    fut[n][0].append(p_)
                    fut[n][1].append(q_)
            fut = {n: (np.array(v[0]), np.array(v[1])) for n, v in fut.items()}
            if self.t >= t_stop:
                # 25.3's post-stop pelvis servo (ScaleBFM often leaves the last step out)
                push = np.clip(push + WALK_KI * R.DT * (goal - sim.d.qpos[:2]), -WALK_CAP, WALK_CAP)
                fut = {n: (p_ + np.append(push, 0.0), q_) for n, (p_, q_) in fut.items()}
            if not self.tick(fut):
                return self.fail("fell during the walk")
        self.stats["walk"] = dict(distance=dist, band_reach=reach, margin=simple.guard_margin_m, ff=ff)
        return True, ""

    # -- the lifted ready --------------------------------------------------------------------------------------------
    def ready(self):
        simple, sim = self.simple, self.sim
        solver = BimanualIK(
            self.ik, waist_pitch_max=self.wb.waist_pitch_max, waist_dofs="all", elbow_min_deg=simple.ready_elbow_min_deg
        )
        start = self.sim.d.qpos[self.body.qadr].copy()
        tool = (simple.ready_tool_forward_m, simple.ready_tool_lateral_m, simple.ready_tool_up_m)
        ready = solver.lifted_ready(tool, simple.ready_pitch_deg, start)
        if float(np.max(ready.errors)) > 0.02:
            return self.fail("simple lifted ready pose missed by the solver", errors_m=ready.errors.tolist())
        pos, quat = sim.d.qpos[:3].copy(), yaw_quat(R.yaw_of(sim.d.qpos[3:7]))
        delta = ready.joints - start
        t0 = self.t
        self.events["ready"] = t0
        raw0 = self.raw()
        obj0 = np.asarray(raw0["objects"][KEY]["pos"], float)
        while self.t - t0 < simple.ready_seconds + simple.ready_hold_s:
            u = min(1.0, (self.t - t0) / simple.ready_seconds)
            q = start + (3 * u * u - 2 * u**3) * delta
            v = (6 * u - 6 * u * u) / simple.ready_seconds * delta
            raw = self.raw()
            if np.linalg.norm(np.asarray(raw["objects"][KEY]["pos"])[:2] - obj0[:2]) > simple.disturb_limit_m:
                return self.fail("simple bottle knocked before the close", phase="ready")
            if not self.tick(futures(self.body, pos, quat, q, v, self.pol.fidx)):
                return self.fail("simple fell during the ready lift")
        self.q_ref, self.root_ref = ready.joints.copy(), quat
        return True, ""

    # -- the grasp (simple_policy._run, guard branch) ---------------------------------------------------------------
    def grasp(self):
        simple, cfg, ik = self.simple, self.wb, self.ik
        state = self.raw()
        actual = tools(state)
        targets = actual.copy()
        posture = self.q_ref.copy()
        posture[12:14] = 0.0
        comfortable = np.clip(np.array(NATURAL_RIGHT_ARM), ik.lo[22:29], ik.hi[22:29])
        posture[25:29] = comfortable[3:7]
        weights = np.full(29, 0.04)
        weights[12:14] = cfg.comfort_waist_weight
        weights[25:29] = cfg.comfort_elbow_wrist_weight
        posture, weights, hand_weights = loop_prior(self.q_ref, posture, weights, False)
        motion = SupportedMotion(
            ik, state, cfg, self.q_ref, self.root_ref, tool_offset=OFFSET, hand_weights=hand_weights,
            posture_reference=posture, posture_weights=weights,
        )  # fmt: skip
        heading = motion.body.heading
        forward = heading[:, 0]

        def plan(st):
            grip = np.asarray(st["objects"][KEY]["pos"], float)  # tabletop_grasp.grip: below_center 0, box centre 0
            pelvis_xy = np.asarray(st["pelvis"]["pos"][:2], float)
            direction = wrap_direction(pelvis_xy, grip, forward, simple.wrap_yaw_per_m, simple.wrap_yaw_max_rad)
            return grip, direction, pregrasp_point(grip, direction, simple.pregrasp_back_m, simple.pregrasp_up_m)

        grip, direction, pregrasp = plan(state)
        rotations = np.array(
            [Primitives._rot(state["palms"]["left"]["quat"]), Primitives._rot(state["palms"]["right"]["quat"])]
        )
        obj0 = np.asarray(state["objects"][KEY]["pos"], float).copy()
        grip_offset = grip - obj0
        raised = actual[1].copy()
        path = [raised, pregrasp]
        raise_arc = polyline_length(path[:-1])
        reach_seconds = simple.pregrasp_seconds
        phase, began = "reach", state["t"]
        self.events["reach"] = began
        closure, lean, hold_offset, lean_at_lift = 0.0, 0.0, None, 0.0
        attempts, retreat_from, retreat_to, retry_shift = 1, None, None, np.zeros(3)
        replanned = False
        entry_cmd, shift = None, np.zeros(3)
        start = state["t"]
        while True:
            state = self.raw()
            stamp, dt = state["t"], R.DT
            if stamp - start > self.rc.recovery_timeout:
                return self.fail(f"simple {phase} timed out")
            obj = state["objects"][KEY]
            actual = tools(state)
            bilateral = bilateral_dex1_contact(obj, "right")
            elapsed = stamp - began
            if phase in ("reach", "settle", "entry", "retreat"):
                if float(np.linalg.norm(np.asarray(obj["pos"])[:2] - obj0[:2])) > simple.disturb_limit_m:
                    return self.fail("simple bottle knocked before the close", phase=phase)

            def enter(name):
                nonlocal phase, began
                self.events.setdefault(name, stamp)
                phase, began = name, stamp

            if phase == "reach":
                arc = moving_start(elapsed / reach_seconds) * polyline_length(path)
                if not replanned and arc >= raise_arc:
                    replanned = True
                    grip, direction, pregrasp = plan(state)
                    rotations[1] = grasp_rotation(direction)
                    path[-1] = pregrasp
                targets[1] = polyline_point(path, arc)
                lean = lean_angle(state["pelvis"]["pos"], grip, simple.lean_max_rad, forward)
                if elapsed >= reach_seconds:
                    enter("settle")
            elif phase == "settle":
                settled = float(np.linalg.norm(actual[1] - targets[1])) <= simple.settle_tolerance_m
                if settled or elapsed >= simple.settle_timeout_s:
                    entry_cmd, shift = targets[1].copy(), np.zeros(3)
                    enter("entry")
            elif phase == "entry":
                palm_rotation = Primitives._rot(state["palms"]["right"]["quat"])
                offset = object_in_palm(palm_rotation, actual[1], np.asarray(obj["pos"], float) + grip_offset)
                aligned = entry_aligned(offset, simple.entry_lateral_tolerance_m, simple.entry_height_tolerance_m)
                entry_cmd = entry_command(
                    entry_cmd, grip, direction, simple.dash_speed_m_s, dt, simple.entry_overshoot_m
                )
                shift = lateral_correction(
                    shift,
                    palm_rotation,
                    offset,
                    simple.lateral_correction_speed_m_s,
                    dt,
                    simple.lateral_correction_max_m,
                )
                targets[1] = entry_cmd + shift
                arrived = entry_captured(actual[1], grip, direction, simple.entry_capture_depth_m)
                if arrived or bilateral:
                    if aligned or bilateral:
                        targets[:] = actual
                        motion.freeze_rotation_feedback()
                        enter("close")
                    elif attempts >= simple.entry_attempts:
                        return self.fail("simple entry could not align the jaw on the bottle", attempts=attempts)
                    else:
                        attempts += 1
                        retreat_from = targets[1].copy()
                        retreat_to = retreat_from - simple.retreat_m * direction
                        retry_shift = lateral_correction(
                            np.zeros(3), palm_rotation, offset, 1.0, 1.0, simple.lateral_correction_max_m
                        )
                        enter("retreat")
                elif elapsed > simple.entry_timeout_s:
                    return self.fail("simple entry timed out without capture")
            elif phase == "retreat":
                targets[1] = retreat_from + minjerk(elapsed / simple.retreat_seconds) * (retreat_to - retreat_from)
                if elapsed >= simple.retreat_seconds:
                    entry_cmd, shift = targets[1].copy(), retry_shift.copy()
                    enter("entry")
            elif phase == "close":
                closure = min(1.0, elapsed / simple.close_seconds)
                if elapsed >= simple.close_seconds:
                    if bilateral:
                        motion.freeze_translation_feedback()
                        hold_offset = torso_offset(ik, motion, targets[1])
                        lean_at_lift = lean
                        motion.resume_rotation_feedback()
                        enter("lift")
                    elif elapsed >= simple.close_seconds + simple.contact_timeout_s:
                        return self.fail("simple no bilateral pad contact after the close")
            elif phase == "lift":
                targets[1] = lift_target(
                    ik, motion, hold_offset, simple.lift_height_m * minjerk(elapsed / simple.lift_seconds)
                )
                if elapsed >= simple.lift_seconds:
                    enter("recover")
            elif phase == "recover":
                lean = lean_at_lift * (1.0 - minjerk(elapsed / simple.recover_seconds))
                targets[1] = lift_target(ik, motion, hold_offset, simple.lift_height_m)
                if elapsed >= simple.recover_seconds:
                    rise = float(obj["pos"][2] - obj0[2])
                    self.stats["upright"] = dict(rise_m=rise, bilateral=bool(bilateral))
                    if not bilateral:
                        return self.fail("simple lost pad contact during the lift", rise_m=rise)
                    if rise < simple.min_rise_m:
                        return self.fail("simple lift did not raise the bottle", rise_m=rise)
                    self.motion = motion
                    return True, "simple grasp lifted and upright"
            torso = heading @ Rotation.from_euler("y", lean).as_matrix()
            pose = motion.step(
                targets, rotations, actual, dt, torso_rotation=torso,
                measured_rotations=np.array([Primitives._rot(state["palms"][s]["quat"]) for s in ("left", "right")]),
            )  # fmt: skip
            if max(pose.foot_errors) > 0.025 or float(np.linalg.norm(pose.com_shift)) > 0.04:
                return self.fail("simple support reference outside tolerance", phase=phase)
            self.jaw = HAND_OPEN[0] + closure * (HAND_CLOSED_RIGHT[0] - HAND_OPEN[0])
            if not self.tick(futures(self.body, motion.x[:3], motion.root, pose.joints, motion.v[4:], self.pol.fidx)):
                return self.fail(f"simple fell during the {phase}")

    # -- the production tail ----------------------------------------------------------------------------------------
    def carry_and_verify(self):
        self.events["carry_and_verify"] = self.t
        motion = self.motion
        q, root, base = motion.joints.copy(), motion.root.copy(), motion.x[:3].copy()
        yaw = R.yaw_of(root)
        back = -np.array([np.cos(yaw), np.sin(yaw), 0.0])
        prof = Profile(BACK_SPEED, R.WALK_ACC, BACK_M)
        t0 = self.t
        ref0 = palm_offset(self.raw())
        while self.t - t0 < 1.0 + prof.t_end + HOLD_S:
            s = prof.s(self.t - t0 - 1.0)
            if "verify_terminal_hold" not in self.events and self.t - t0 >= 1.0 + prof.t_end:
                self.events["verify_terminal_hold"] = self.t
            if not self.tick(futures(self.body, base + back * s, root, q, np.zeros(29), self.pol.fidx)):
                return self.fail("[hold] fell while backing away from the table")
        raw = self.raw()
        drift = float(np.linalg.norm(palm_offset(raw) - ref0))
        self.stats["hold"] = dict(drift_m=drift, height=float(raw["objects"][KEY]["pos"][2]))
        if drift >= PALM_DRIFT_MAX:
            return self.fail(f"[hold] object slipped {drift * 100:.1f} cm in the palm frame during the hold")
        if float(raw["objects"][KEY]["pos"][2]) <= float(self.f["table_top"]) + TABLE_CLEAR:
            return self.fail("[hold] object is back on the table after the hold")
        return True, ""

    def run(self):
        t_wall = time.time()
        ok, why = False, ""
        for step in (self.walk, self.ready, self.grasp, self.carry_and_verify):
            ok, why = step()
            if not ok:
                break
        self.events["end"] = self.t
        facts = dict(
            self.f,
            controller="scalebfm",
            program=f"expert_m{self.a.mode}{self.a.tag}",
            mode=self.a.mode,
            events=self.events,
            stats=self.stats,
            sonic_ok=self.f.get("ok"),
            sonic_why=self.f.get("why"),
            ok=bool(ok and self.fell is None),
            why="" if ok else why,
            wall_s=round(time.time() - t_wall, 1),
        )
        return self.rec.arrays(), facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "sonic", type=Path, help="bottle_states.py files of SONIC's rounds (the start and walk plan of each)"
    )
    ap.add_argument("--config", type=Path, required=True, help="the SONIC run's experiment yaml (task.reach)")
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument(
        "--scene", type=Path, default=SCENE, help="the scene (payload_scenes.py's copies for perturbation e)"
    )
    ap.add_argument(
        "--odometry", action="store_true",
        help="ScaleBFM sees the leg-kinematics + IMU pelvis estimate (backends/scalebfm/odometry.py), not the truth (25.5)",
    )  # fmt: skip
    ap.add_argument("--perturb", default="", help="sim/perturb.py spec (JSON), e.g. the real-robot combo tier")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rc = ReachConfig.from_dict(yaml.safe_load(a.config.read_text())["task"]["reach"], source=str(a.config))
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    bcfg = ScaleBfmBackendConfig(
        scene_dir=str(a.scene), object_name=KEY, grip_kp=HAND_KP, grip_kd=HAND_KD, grip_clip=HAND_FORCE
    )
    pol = R.Policy(device=a.device, backend=bcfg)
    pol.mode = torch.tensor([a.mode], dtype=torch.long, device=a.device)
    sim = R.Sim(pol.meta, backend=bcfg, perturb=json.loads(a.perturb) if a.perturb else None)
    odo = None
    if a.odometry:
        from locomani.backends.scalebfm.odometry import attach

        odo = attach(sim, str(a.scene / "scene_locomani.xml"), pol.meta["joint_names"])
    a.odo = odo
    body = Body(sim.m, pol.sel)
    ik = ArmIK(str(a.scene / "scene_locomani.xml"))
    physical = ik.model.jnt_range[ik.jids]
    ik.lo, ik.hi = physical[:, 0].copy(), physical[:, 1].copy()
    ik.hi[14] = min(ik.hi[14], rc.whole_body.waist_pitch_max)
    pub = PrivilegedPublisher(
        sim.m, sim.d, {KEY: meta["objects"][KEY]["body"]}, surfaces=[meta["surfaces"]["A"]["body"]],
        port=59000 + os.getpid() % 3000,
    )  # fmt: skip
    only = set(a.only.split(",")) if a.only else None
    for npz in sorted(a.sonic.glob("*.npz")):
        if only and npz.stem not in only:
            continue
        z = dict(np.load(npz))
        f = json.loads(npz.with_suffix(".json").read_text())
        label = npz.stem.replace("sonic", "bfm", 1)
        if (a.out / f"{label}.json").exists():
            continue
        arrays, facts = Round(pol, sim, z, f, rc, a, body, ik, pub).run()
        facts["perturb"] = a.perturb or None
        facts["odometry"] = bool(a.odometry)
        facts["scene"] = str(a.scene)
        save(a.out, label, arrays, facts)
        print(
            f"{label}: ok {facts['ok']} {facts['why']} events {list(facts['events'])} {facts['wall_s']} s", flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

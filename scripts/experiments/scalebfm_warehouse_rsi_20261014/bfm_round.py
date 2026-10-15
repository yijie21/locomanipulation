"""TODO 25.9.1 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §6): the warehouse expert on ScaleBFM, one round per
start state, under a candidate -- the program every loop role runs (Gate, Prober, the diagnostic re-runs).

The round is 25.7's (controller_compare_20261013/bfm_expert_warehouse.run_round, line for line), with four additions:
* a candidate spec (YAML or JSON) instead of flags --
      mode: 2                 # ScaleBFM mode 0-7
      stance: near            # near: stand_off - stand_depth_tolerance (25.7); centre: stand_off
      stance_shift: 0.0       # m added to that stance along the walk-in axis (+ = farther from the box)
      servo: none             # none | pad (25.3's jaw-centre + attitude servo)
      set: {visual.tracking_gain: 0.5}   # task-config overrides, block.field -> value (visual.reach_route planned:
                                         # no ready pose, a CuRobo route from the hang; the service must be running)
      reference: {}           # transforms of the expert's whole-body reference (REFERENCE below)
      extensions: {}          # extensions.py hooks by name -> params (new behaviour the loop's Engineer adds)
* a fork: `--fork PHASE:DELTA` runs the base spec until PHASE (walk | ready | reach) starts and the base spec merged
  with DELTA from there on. Forks re-run the prefix from the start state instead of restoring a snapshot; the
  prefix is the same because the round is deterministic (determinism.py checks it), and nothing (IK state, the
  expert's state machine, ScaleBFM's history) has to be copied;
* more recording: cmd_joints (T,29) the reference joints ScaleBFM's keypoints are FK'd from, phase (T,) index into
  PHASES, capture_err (T,2,3) CoordinatedReach's capture error per hand [across, along, up] in the box frame and
  feedback (T,2,3) its tracking feedback, both read-only and NaN outside reach / descend;
* field accounting: facts["fields_read"] lists every task-config field the round read (the gate rejects a candidate
  that sets a field this path never reads).
The facts keep the start state's description (seed, edge, heading, platform, box) and SONIC's verdict as sonic_ok /
sonic_why, but none of SONIC's other outcome fields (25.7's files inherited its rise / tilt_deg).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/scalebfm_warehouse_rsi_20261014/bfm_round.py outputs/controller_compare_20261013/warehouse/sonic/states \\
        --spec scripts/experiments/scalebfm_warehouse_rsi_20261014/incumbent.yaml --out /tmp/x [--shard 0/4] [--fork reach:delta.yaml]
"""

from __future__ import annotations

import argparse
import dataclasses
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
CC = REPO / "scripts" / "experiments" / "controller_compare_20261013"
sys.path[:0] = [str(HERE), str(CC), str(REPO / "scripts" / "experiments" / "wbc_compare_20261012"), str(REPO / "src")]

import extensions  # noqa: E402
from bfm_brake import FF_SHORT, FF_SPEEDS, Profile, frontmost_low  # noqa: E402
from bfm_expert_warehouse import (  # noqa: E402
    HAND_OPEN,
    KEY,
    TIMEOUT_S,
    YAW_TURN_S,
    Body,
    Estimate,
    futures,
    pad_servo,
    yaw_quat,
)
from bfm_warehouse import WALK_CAP, WALK_KI, Recorder  # noqa: E402
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

PHASES = ("walk", "ready", "reach")
SPEC_KEYS = {"mode", "stance", "stance_shift", "servo", "set", "reference", "extensions"}
BASE_SPEC = dict(mode=2, stance="near", stance_shift=0.0, servo="none", set={}, reference={}, extensions={})
WAIST = {n: MUJOCO_JOINT_NAMES.index(f"waist_{n}_joint") for n in ("yaw", "roll", "pitch")}
# Reference transforms (spec["reference"]), applied to the expert's whole-body reference before FK, in the ready and
# reach phases (the walk's reference is the standing pose):
#   waist_gain: {yaw|roll|pitch: g}       waist joint reference times g (0 = neutral)
#   waist_clamp: {yaw|roll|pitch: [lo, hi]} waist joint reference clipped (rad)
#   root_z_min: z                         the reference pelvis no lower than z (m, world)
REFERENCE = ("waist_gain", "waist_clamp", "root_z_min")


def load_delta(path_or_text) -> dict:
    """A spec file (YAML / JSON) or inline YAML as written -- no defaults filled in; unknown keys are an error."""
    p = Path(str(path_or_text))
    raw = yaml.safe_load(p.read_text()) if p.exists() else yaml.safe_load(str(path_or_text))
    raw = dict(raw or {})
    raw.pop("name", None)
    raw.pop("note", None)
    bad = set(raw) - SPEC_KEYS
    if bad:
        raise ValueError(f"unknown spec keys {sorted(bad)}; allowed {sorted(SPEC_KEYS)}")
    bad = set(raw.get("reference") or {}) - set(REFERENCE)
    if bad:
        raise ValueError(f"unknown reference transforms {sorted(bad)}; allowed {REFERENCE}")
    bad = set(raw.get("extensions") or {}) - set(extensions.HOOKS)
    if bad:
        raise ValueError(f"unknown extensions {sorted(bad)}; registered {sorted(extensions.HOOKS)}")
    return raw


def load_spec(path_or_text) -> dict:
    """A spec file (YAML / JSON) or inline YAML, merged over BASE_SPEC; unknown keys are an error."""
    out = json.loads(json.dumps(BASE_SPEC))
    out.update(load_delta(path_or_text))
    return out


def merge(base: dict, delta: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in delta.items():
        if k in ("set", "reference", "extensions"):
            out[k] = {**out.get(k, {}), **(v or {})}
        else:
            out[k] = v
    return out


def spec_hash(spec: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


class FieldLog:
    """Which task-config fields a round reads: each config dataclass instance gets a subclass that notes reads."""

    def __init__(self):
        self.read = set()

    def watch(self, obj, prefix=""):
        base, log = type(obj), self.read
        names = {f.name for f in dataclasses.fields(obj)}

        def getattribute(self_, name, _base=base):
            if name in names:
                log.add(prefix + name)
            return _base.__getattribute__(self_, name)

        object.__setattr__(obj, "__class__", type(base.__name__, (base,), {"__getattribute__": getattribute}))
        return obj


def task_config(config: Path, sets: dict, log: FieldLog) -> CrateTaskConfig:
    task = dict(yaml.safe_load(config.read_text())["task"])
    for k_ in ("plugin", "expert"):
        task.pop(k_, None)
    for key, value in sets.items():
        block, field_ = key.split(".", 1)
        task.setdefault(block, {})[field_] = value
    cfg = CrateTaskConfig.from_dict(task)
    log.watch(cfg)
    log.watch(cfg.visual, "visual.")
    log.watch(cfg.carry, "carry.")
    return cfg


class Context:
    """Everything a spec decides: the task config, its IK solver and grasp target, ScaleBFM's mode, the servo."""

    _solvers = {}

    def __init__(self, spec, config, scene_meta, model_xml, log):
        self.spec = spec
        self.cfg = task_config(config, spec["set"], log)
        vis = self.cfg.visual
        key = (self.cfg.carry.waist_pitch_max, tuple(self.cfg.carry.waist_dofs), vis.elbow_min_deg)
        if key not in Context._solvers:
            Context._solvers[key] = BimanualIK(
                ArmIK(model_xml), waist_pitch_max=key[0], waist_dofs=self.cfg.carry.waist_dofs, elbow_min_deg=key[2]
            )
        self.solver = Context._solvers[key]
        target_meta = scene_meta["objects"][KEY]
        self.half = tuple(float(v) for v in target_meta["box"]["half_extent"][:2])
        region = RimRegion(across=vis.capture_across) if vis.capture_across > 0 else RimRegion()
        self.target = CoordinatedTarget(
            object_key=KEY,
            grasp_points_local=grasp_points_for(target_meta, vis.box_grasp_below_rim),
            rim_above_grasp=vis.box_grasp_below_rim,
            region=region,
        )
        self.mode = int(spec["mode"])
        self.servo = spec["servo"]

    def reference(self, joints, root, **state):
        """The spec's reference transforms, then its extensions' `reference` hooks, on (29 joints, pelvis position).
        state: t, phase, sim, st8 (the round's state machine) for the hooks."""
        ref, ext = self.spec["reference"], self.spec["extensions"]
        if not ref and not ext:
            return joints, root
        q, r = np.array(joints, float), np.array(root, float)
        for n, g in (ref.get("waist_gain") or {}).items():
            q[WAIST[n]] *= float(g)
        for n, (lo, hi) in (ref.get("waist_clamp") or {}).items():
            q[WAIST[n]] = np.clip(q[WAIST[n]], lo, hi)
        if ref.get("root_z_min") is not None:
            r[2] = max(r[2], float(ref["root_z_min"]))
        for name, params in ext.items():
            hook = extensions.HOOKS[name].get("reference")
            if hook is not None:
                q, r = hook(q, r, params or {}, **state)
        return q, r

    def futures(self, fut, **state):
        """The spec's extensions' `futures` hooks on ScaleBFM's keypoint targets (after the pad servo)."""
        for name, params in self.spec["extensions"].items():
            hook = extensions.HOOKS[name].get("futures")
            if hook is not None:
                fut = hook(fut, params or {}, **state)
        return fut


class Rec(Recorder):
    def __init__(self, sim):
        super().__init__(sim)
        self.rows.update(cmd_joints=[], phase=[], capture_err=[], feedback=[])

    def add(self, t, targets, joints=None, phase="walk", capture=None, feedback=None):
        super().add(t, targets)
        self.rows["cmd_joints"].append(np.full(29, np.nan) if joints is None else np.asarray(joints, float))
        self.rows["phase"].append(PHASES.index(phase))
        self.rows["capture_err"].append(np.full((2, 3), np.nan) if capture is None else np.asarray(capture, float))
        self.rows["feedback"].append(np.full((2, 3), np.nan) if feedback is None else np.asarray(feedback, float))


def warm_up(pol, sim, body, calls=3):
    """Run the policy a few times before the first round. The network's TorchScript helpers run their first call
    unoptimised and later calls fused: the first action of a process differs by ~6e-8 from the same call made later
    (2026-10-14), which made a round's outcome depend on whether it ran first in its process."""
    sim.reset(np.zeros(2), 0.0, 0.0)
    seat(sim)
    st = sim.state()
    fut = futures(
        body, sim.d.qpos[:3].copy(), sim.d.qpos[3:7].copy(), sim.d.qpos[body.qadr].copy(), np.zeros(29), pol.fidx
    )
    for _ in range(calls):
        pol.reset(st)
        pol(st, fut)


def run_round(pol, sim, z, f, ctxs, fork, body, pub, log):
    """One round from a start state. ctxs: {"base": Context, "fork": Context or None}; fork: the phase the fork's
    context takes over at (None = base throughout)."""
    ctx = ctxs["fork"] if fork == "walk" else ctxs["base"]
    if ctxs.get("route_planner") is not None:
        ctxs["route_planner"].last = {}  # this round's plan only
    spec, cfg, solver = ctx.spec, ctx.cfg, ctx.solver
    pol.mode = torch.tensor([ctx.mode], dtype=torch.long, device=pol.mode.device)
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
    robot = [g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] not in (0, sim.crate_b) and sim.m.geom_bodyid[g] > 0]
    robot = [g for g in robot if sim.m.body(sim.m.geom_bodyid[g]).rootid == sim.m.body("pelvis").rootid]
    top = float(f["platform_top"])
    hx, hy = ctx.half  # the guard line: the platform's near line or the box's near face (expert.run_attempt)
    by = R.yaw_of(box0[3:])
    local = R.rot_z(by)[:2, :2].T @ axis
    box_line = float(axis @ box0[:2] - abs(local[0]) * hx - abs(local[1]) * hy)
    line = min(float(f["edge"]["line"]), box_line)
    keep = sim.d.qpos[3:7].copy()
    sim.d.qpos[3:7] = yaw_quat(psi)
    mujoco.mj_kinematics(sim.m, sim.d)
    reach = frontmost_low(sim.m, sim.d, robot, p0, axis, top)
    sim.d.qpos[3:7] = keep
    mujoco.mj_forward(sim.m, sim.d)
    stand = vis.stand_off - vis.stand_depth_tolerance if spec["stance"] == "near" else vis.stand_off
    stand += float(spec["stance_shift"])
    target_along = min(line - vis.guard_margin - reach, float(axis @ box0[:2]) - stand)
    ff = float(np.interp(vis.walk_speed, FF_SPEEDS, FF_SHORT))
    goal = target_along * axis + float(perp @ box0[:2]) * perp
    dist = float(np.linalg.norm(goal - p0))
    d = (goal - p0) / max(dist, 1e-9)
    prof = Profile(vis.walk_speed, R.WALK_ACC, dist + ff)
    pol.reset(sim.state())
    q_stand, z_stand = sim.d.qpos[body.qadr].copy(), float(sim.d.qpos[2])
    rec = Rec(sim)
    events = {"walk": R.SETTLE_S}
    st8 = dict(phase="walk", low=None, ready=None, ready_t=None, hold=None, reason="", finished=False)
    jaw = HAND_OPEN
    t = 0.0
    k = 0
    fell = None
    t_wall = time.time()

    def take_over(phase):
        nonlocal ctx, cfg, vis, solver
        if fork == phase and ctxs["fork"] is not None:
            ctx = ctxs["fork"]
            cfg, vis, solver = ctx.cfg, ctx.cfg.visual, ctx.solver
            pol.mode = torch.tensor([ctx.mode], dtype=torch.long, device=pol.mode.device)
            events["fork"] = t

    while t < TIMEOUT_S:
        t = k * R.DT
        mujoco.mj_forward(sim.m, sim.d)
        ref_q, capture, fb = None, None, None
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
            ref_q = q_stand
            if t >= R.SETTLE_S + prof.t_end:
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
                take_over("ready")
                st8["phase"] = "ready"
                if vis.reach_route == "planned":
                    # visual.reach_route planned: no ready pose; the reach starts from the hang and its first step
                    # plans the route to the pregrasp points (CoordinatedReach.plan_route)
                    take_over("reach")
                    raw = pub.state()
                    raw["t"] = t
                    st8["low"] = CoordinatedReach(
                        solver,
                        cfg,
                        raw,
                        reference_joints=sim.d.qpos[body.qadr].copy(),
                        reference_quaternion=np.asarray(raw["pelvis"]["quat"], float),
                        target=ctx.target,
                    )
                    st8["low"].route_planner = ctxs["route_planner"]
                    events["handoff"] = t
                    st8["phase"] = "reach"
            if st8["phase"] == "ready":
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
            q, pos_ref = ctx.reference(q, pos, t=t, phase="ready", sim=sim, st8=st8)
            fut = futures(body, pos_ref, quat, q, v if u < 1 else np.zeros(29), pol.fidx)
            ref_q = q
            if t - st8["ready_t"] < vis.ready_blend_seconds + vis.ready_hold_seconds:
                arrived = False
            else:
                kp_now = body.keypoints(
                    pos_ref, quat, ctx.reference(st8["ready"], pos, t=t, phase="ready", sim=sim, st8=st8)[0]
                )
                arrived = (
                    all(
                        np.linalg.norm(sim.d.xpos[sim.m.body(w).id] - kp_now[w][0]) <= 0.03
                        for w in ("left_wrist_yaw_link", "right_wrist_yaw_link")
                    )
                    or t - st8["ready_t"] >= vis.ready_blend_seconds + 2.0
                )
            if arrived:
                take_over("reach")
                raw = pub.state()
                raw["t"] = t
                st8["low"] = CoordinatedReach(
                    solver,
                    cfg,
                    raw,
                    reference_joints=st8["ready"],
                    reference_quaternion=np.asarray(raw["pelvis"]["quat"], float),
                    target=ctx.target,
                )
                events["handoff"] = t
                st8["phase"] = "reach"
        if st8["phase"] == "reach":
            raw = pub.state()
            raw["t"] = t
            openings = np.array([2 * (np.mean(raw["palms"][s]["hand_q"][:2]) + 0.023) for s in ("left", "right")])
            cmd = st8["low"].step(t, Estimate(raw, KEY, t), raw, openings, motion_time=t)
            if (
                cmd.finished
                and cmd.reason in ("route_not_found", "route_goal_unreachable")
                and "fallback" not in st8
                and vis.reach_route_fallback == "straight"
            ):
                # no planned route (2026-10-14: hanging fingers already on the box's near wall, 1 of 50): this round
                # goes the old way -- the ready pose in place, then the straight reach -- and the record says so
                st8["fallback"] = cmd.reason
                events["route_fallback"] = t
                st8["phase"] = "ready"
                st8["hold_pose"] = (sim.d.qpos[:3].copy(), yaw_quat(R.yaw_of(sim.d.qpos[3:7])))
                st8["q0"] = sim.d.qpos[body.qadr].copy()
                st8["ready"] = solver.lifted_ready(vis.ready_tool, vis.ready_pitch_deg, st8["q0"]).joints.copy()
                st8["ready_t"] = t
                st8["low"] = None
                cfg = cfg.with_overrides(["visual.reach_route=straight"])
                vis = cfg.visual
                k += 1
                continue
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
            if cmd.phase in ("reach", "descend") and low.diagnostics.get("capture_error") is not None:
                capture, fb = np.array(low.diagnostics["capture_error"], float), np.array(low.feedback, float)
            q, root = ctx.reference(cmd.joints, low.x[:3], t=t, phase=cmd.phase, sim=sim, st8=st8)
            fut = futures(body, root, cmd.root_quaternion, q, cmd.velocity, pol.fidx)
            ref_q = q
            if ctx.servo == "pad":
                fut = pad_servo(sim, fut, st8.setdefault("servo", {}))
            fut = ctx.futures(fut, t=t, phase=cmd.phase, sim=sim, st8=st8)
        if fut is None:
            k += 1
            continue
        sim.grip_target[:] = jaw
        rec.add(t, fut, ref_q, st8["phase"], capture, fb)
        if sim.fallen():
            fell = t
            break
        sim.step(pol(sim.state(), fut))
        k += 1
    if "end" not in events:
        events["end"] = t
    expert_ok = st8["reason"] == "completed" and fell is None
    start = {
        k_: f[k_]
        for k_ in ("label", "seed", "edge", "heading", "platform_top", "box_top", "box_half", "placement", "guard")
        if k_ in f
    }
    facts = dict(
        start,
        controller="scalebfm",
        program="rsi",
        spec=ctxs["base"].spec,
        spec_hash=spec_hash(ctxs["base"].spec),
        fork=None if fork is None else dict(phase=fork, spec=ctxs["fork"].spec, spec_hash=spec_hash(ctxs["fork"].spec)),
        mode=ctx.mode,
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
        fields_read=sorted(log.read),
        wall_s=round(time.time() - t_wall, 1),
    )
    low = st8.get("low")
    if "fallback" in st8:
        facts["route"] = dict(ctxs["route_planner"].last, fallback=st8["fallback"])
    elif low is not None and low.route is not None:  # visual.reach_route planned: the plan and the retimed duration
        facts["route"] = dict(ctxs["route_planner"].last, duration=round(float(low.route["total"]), 3))
    elif ctxs["route_planner"] is not None and ctxs["route_planner"].last:
        facts["route"] = dict(ctxs["route_planner"].last)
    return rec.arrays(), facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("starts", type=Path, help="warehouse_states.py files of SONIC's rounds (their start states)")
    ap.add_argument("--spec", required=True, help="candidate spec file or inline YAML")
    ap.add_argument("--fork", default="", help="PHASE:DELTA -- the delta spec (file or inline YAML) from PHASE on")
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/experiments/controller_compare_l2_warehouse_sonic_20261013.yaml",
        help="the experiment yaml whose task block the spec's `set` overrides (frozen: §3)",
    )
    ap.add_argument("--only", default="", help="comma-separated start labels (sonic_s7000_00, ...)")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th start from the i-th, in label order")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--route-cache",
        default="auto",
        help="route plans cached by request (primitives/arm_route.py): a directory, off, or auto = route_cache next "
        "to the start pool (a hidden pool's cache stays hidden)",
    )
    a = ap.parse_args(argv)
    base = load_spec(a.spec)
    fork_phase, delta = None, None
    if a.fork:
        fork_phase, _, text = a.fork.partition(":")
        if fork_phase not in PHASES:
            raise SystemExit(f"--fork phase must be one of {PHASES}")
        delta = merge(base, load_delta(text))
    meta = json.loads((SCENE / "scene_meta.json").read_text())
    log = FieldLog()
    xml = str(SCENE / "scene_locomani.xml")
    ctxs = dict(
        base=Context(base, a.config, meta, xml, log), fork=Context(delta, a.config, meta, xml, log) if delta else None
    )
    ctxs["route_planner"] = None
    i, n = (int(x) for x in a.shard.split("/"))
    if any(c is not None and c.cfg.visual.reach_route == "planned" for c in (ctxs["base"], ctxs["fork"])):
        # visual.reach_route planned: the CuRobo route services on TF_CUROBO_PORT(S) (tasks/crate/reach_route.py)
        from locomani.primitives.arm_route import ArmRouteClient
        from locomani.sim.ports import curobo_routes
        from locomani.tasks.crate.reach_route import ReachRoutePlanner

        ctx_ = ctxs["fork"] if ctxs["fork"] is not None else ctxs["base"]
        cache = {"off": None, "auto": a.starts.parent / "route_cache"}.get(a.route_cache, Path(a.route_cache))
        ports = curobo_routes()
        ctxs["route_planner"] = ReachRoutePlanner(
            ArmRouteClient(ports[i % len(ports)], plan_timeout_s=180.0, cache_dir=cache),  # shards spread over them
            mujoco.MjModel.from_xml_path(xml),
            ctx_.solver.lo,
            ctx_.solver.hi,
            {BOX: KEY},
            clearance=ctx_.cfg.visual.reach_route_clearance,
        )
    from bfm_bottle import HAND_FORCE, HAND_KD, HAND_KP  # the SONIC grip law (24.1)

    bcfg = ScaleBfmBackendConfig(
        scene_dir=str(SCENE), object_name=KEY, grip_kp=HAND_KP, grip_kd=HAND_KD, grip_clip=HAND_FORCE
    )
    pol = R.Policy(device=a.device, backend=bcfg)
    pol.mode = torch.tensor([base["mode"]], dtype=torch.long, device=a.device)
    sim = R.Sim(pol.meta, backend=bcfg, perturb=None)
    body = Body(sim.m, pol.sel)
    warm_up(pol, sim, body)
    pub = PrivilegedPublisher(
        sim.m, sim.d, {KEY: BOX}, surfaces=[meta["surfaces"]["A"]["body"]], port=56000 + os.getpid() % 3000
    )
    only = set(a.only.split(",")) if a.only else None
    starts = [p for p in sorted(a.starts.glob("*.npz")) if not only or p.stem in only][i::n]
    for npz in starts:
        z = dict(np.load(npz))
        f = json.loads(npz.with_suffix(".json").read_text())
        label = npz.stem.replace("sonic", "bfm", 1)
        if (a.out / f"{label}.json").exists():
            continue
        log.read.clear()
        arrays, facts = run_round(pol, sim, z, f, ctxs, fork_phase, body, pub, log)
        facts["label"] = label
        save(a.out, label, arrays, facts)
        print(f"{label}: ok {facts['ok']} ({facts['expert_reason']}) {facts['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""wbc_compare step 3 (TODO 24.3): grasp_planned's hand program on the ScaleBFM runtime, object by object.

The 65-object line's program (scripts/experiments/graspnet_bottle_20261008/grasp_planned.py) with the whole-body
controller swapped for ScaleBFM mode 2 and nothing else: the same stance (bfm_object_scene.stance), the same object
spawn (run_grasp's per-object yaw, rng 1000 + id), the same cached grasp candidates (grasp_planned.cached_candidates at
the settled object pose), the same phases and thresholds (the simple expert's reach.simple block, copied below) and
the same jaw law (step 1). What the controller swap changes is only how a tool target becomes motion:

- planned approach (open-loop on both): SONIC streams an IK joint path (up to 0.22 m over the table, over, down to
  the pre-grasp) with the arm pinned; here the wrist reference walks the same staged tool path at APPROACH_SPEED with
  the palm attitude turning on the way. Both aim `bias_m` (probe/bias.json, decision 2) past the delivery error.
- from the settle on (closed-loop on both): SONIC's SupportedMotion solves a whole-body IK each tick with its
  tracking feedback (the integrated reference-minus-measured offset, deadband 2 mm, 0.04 m/s, 0.2 m/s^2, origin
  preserved at takeover); here the same feedback law moves the wrist reference directly (TRACKING). The entry gate,
  the 3 cm/s lateral correction capped at 4 cm, the retry retreat, the 1.0 s close / 0.8 s contact timeout, the
  10 cm lift and the hold are the program's own code paths, line for line. No torso-lean target (mode 2 has none)
  and no rotation feedback: attitude error is reported as delivered.

Grasp selection (--select): `paired` takes the cache row SONIC's round chose (--pair-runs TAG reads
outputs/runs/gnb_obj_XXX<TAG>/grasp_planned/round_RR_ik/record.json) -- the controlled arm; `score` takes the whole
cache by validation score (the free arm, attitude layer open); `cache` takes the cache's own tier / score order.
With --envelope (probe/scalebfm_attitude/envelope.json, TODO 24.6) `score` / `cache` first skip the candidates whose
palm attitude ScaleBFM did not deliver in the attitude probe (probe_cells.Envelope: nearest probed attitude), then
apply the planner's path-clearance rejection as before.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=osmesa PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \
        scripts/experiments/wbc_compare_20261012/bfm_grasp.py --objects 40 --rounds 1 --select cache --video \
        --bias outputs/wbc_compare_20261012/probe/bias.json --out outputs/wbc_compare_20261012/bfm_grasp/_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRASPNET = REPO / "scripts" / "experiments" / "graspnet_bottle_20261008"
sys.path[:0] = [str(HERE), str(GRASPNET), str(REPO / "src")]

import grasp_planned as G  # noqa: E402  (imports cleanly without the SONIC stack: numpy + scipy only)
import probe_cells as C  # noqa: E402
from bfm_object_scene import (  # noqa: E402
    JAW_CLOSED,
    JAW_OPEN,
    assert_matches_grasp_cache,
    make_object_sim,
    objects,
    stance,
)
from bfm_probe import FPS, Schedule, open_writer, preview, slerp  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

# the simple expert's thresholds the program reads from configs/experiments/bottle_simple_200.yaml reach.simple
# (copied: the config module pulls the SONIC stack into the import)
SIMPLE = dict(
    settle_tolerance_m=0.02,
    settle_timeout_s=1.5,
    lateral_correction_speed_m_s=0.03,
    lateral_correction_max_m=0.04,
    entry_overshoot_m=0.02,
    entry_capture_depth_m=0.01,
    entry_timeout_s=6.0,
    entry_attempts=2,
    retreat_m=0.08,
    retreat_seconds=1.0,
    close_seconds=1.0,
    contact_timeout_s=0.8,
    lift_seconds=2.0,
    disturb_limit_m=0.05,
)
# reach.whole_body.tracking_translation_* and SupportedMotion's +-0.08 m clip around the takeover offset
TRACKING = dict(deadband=0.002, speed=0.04, acceleration=0.2, clip=0.08)
HOLD_S = 2.0  # decision 3: the hold after the lift, both controllers (grasp_planned.HOLD_S 1.5 is the 10-10 value)
RISE_MIN = 0.05  # grasp_planned: "not lifted" under this
APPROACH_SPEED = 0.15  # m/s tool reference on the planned approach
PHASES = ("settle", "entry", "close", "lift", "recover", "retreat")
OBJECT_NAME = "water_bottle"


# --- the simple expert's helpers (tasks/bottle/simple_policy.py; that module imports the SONIC stack) --------------
def entry_captured(tool, grip, direction, capture_depth_m):
    return float((grip - tool) @ direction) <= capture_depth_m


def entry_command(command, grip, direction, speed, dt, overshoot_m):
    limit = float((grip - command) @ direction) + overshoot_m
    return command + min(speed * dt, max(0.0, limit)) * direction


def minjerk(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def object_in_palm(palm_R, tool, obj_pos):
    return palm_R.T @ (obj_pos - tool)


# --- SupportedMotion's translation feedback (primitives/supported_motion.py), one hand -------------------------------
class Tracking:
    def __init__(self, on=True):
        self.on = on
        self.origin = None
        self.feedback = np.zeros(3)
        self.rate = np.zeros(3)

    def step(self, reference, measured, dt):
        if not self.on:
            return np.zeros(3)
        error = reference - measured
        if self.origin is None:
            self.origin = error.copy()
            self.feedback = error.copy()
            return self.feedback
        error = self.origin + np.clip(error - self.origin, -TRACKING["clip"], TRACKING["clip"])
        resid = error - self.origin
        n = float(np.linalg.norm(resid))
        dead = resid * (n - TRACKING["deadband"]) / n if n > TRACKING["deadband"] else np.zeros(3)
        target = self.origin + dead
        w = TRACKING["acceleration"] / TRACKING["speed"]
        acc = w * w * (target - self.feedback) - 2 * w * self.rate
        self.rate += _clip_norm(acc, TRACKING["acceleration"]) * dt
        self.rate = _clip_norm(self.rate, TRACKING["speed"])
        self.feedback = self.feedback + self.rate * dt
        return self.feedback


def _clip_norm(v, limit):
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit else v


# --- one round --------------------------------------------------------------------------------------------------------
class Round:
    def __init__(self, pol, obj_id, rnd, yaw_obj, args, bias_w_fn):
        self.pol, self.obj_id, self.rnd, self.yaw_obj, self.args = pol, obj_id, rnd, yaw_obj, args
        self.bias_w_fn = bias_w_fn
        self.out = args.out / f"obj_{obj_id:03d}" / f"round_{rnd:02d}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.record = dict(controller="scalebfm", object=obj_id, round=rnd, select=args.select, hold_s=args.hold)
        if args.envelope is not None:
            self.record["envelope"] = str(args.envelope.path.resolve().relative_to(REPO))
        self.trace = []
        self.writer = None
        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[4] = 1  # the scene's magenta ghost hand (group 4) at the commanded grasp

    def frame(self, sim):
        """runtime.Sim.frame's camera, with the ghost hand's geom group drawn."""
        pel = sim.d.xpos[sim.bid["pelvis"]]
        sim.cam.lookat[:] = [pel[0] + 0.25, pel[1], 0.72]
        sim.cam.distance = 2.0
        sim.cam.azimuth = 135
        sim.cam.elevation = -20
        sim.renderer.update_scene(sim.d, camera=sim.cam, scene_option=self.vopt)
        return sim.renderer.render()

    def contacts(self, sim):
        d, m = sim.d, sim.m
        touched = set()
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            for a, b in ((b1, b2), (b2, b1)):
                if b == sim.crate_b and a in self.fingers:
                    touched.add(self.fingers[a])
        return touched

    def clears(self, c, tool_side, R_side, surface, table_top):
        """grasp_planned._plan_ik's rejection without the IK: the open jaw's points along the staged path to the
        pre-grasp must stay above the table (from the travel on) and off the object's surface."""
        from scipy.spatial import cKDTree

        tree = cKDTree(surface)
        R_c = np.asarray(c["R"], dtype=float)
        pre = np.asarray(c["tool"], dtype=float) - G.PREGRASP_BACK * R_c[:, 0]
        for li, (a, b) in enumerate(C.legs(tool_side, pre, table_top)):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.01)))
            for k in range(1, n + 1):
                u = k / n
                rot = slerp(R_side, R_c, min(1.0, 0.5 * (li + u)) if li < 2 else 1.0)
                hand = a + u * (b - a) + G.HAND_POINTS @ rot.T
                if li >= 1 and hand[:, 2].min() < table_top + G.TABLE_CLEAR:
                    return f"leg {li} step {k}: hand under the table top"
                if tree.query(hand, k=1)[0].min() < G.OBJECT_CLEAR:
                    return f"leg {li} step {k}: hand into the target"
        return None

    def choose(self, cands, tool_side=None, R_side=None, surface=None, table_top=None):
        args = self.args
        if not cands:
            return None, "no cached grasp fits here"
        if args.select in ("score", "cache"):
            order = sorted(cands, key=lambda c: -c["score"]) if args.select == "score" else list(cands)
            rejected, by_attitude = [], 0
            for c in order[: 2 * G.MAX_CANDIDATES]:
                # TODO 24.6: the probed attitude envelope first (what ScaleBFM delivers), then the planner's path clearance
                why = args.envelope.reject(c["R"], self.stance_yaw) if args.envelope is not None else None
                by_attitude += why is not None
                why = self.clears(c, tool_side, R_side, surface, table_top) if why is None else why
                if why is None:
                    self.record["rejected"] = rejected
                    return c, ""
                rejected.append(dict(row=c["row"], why=why))
            self.record["rejected"] = rejected
            if args.envelope is not None:
                return None, (
                    f"no candidate inside the envelope clears the target ({len(rejected)} tried, {by_attitude} by attitude)"
                )
            return None, f"no candidate's staged path clears the target ({len(rejected)} tried)"
        rec = REPO / "outputs" / "runs" / f"gnb_obj_{self.obj_id:03d}{args.pair_runs}" / "grasp_planned"
        rec = rec / f"round_{self.rnd:02d}_ik" / "record.json"
        if not rec.exists():
            return None, f"no SONIC round to pair with ({rec.relative_to(REPO)})"
        sonic = json.loads(rec.read_text())
        self.record["paired_with"] = dict(path=str(rec.relative_to(REPO)), ok=sonic.get("ok"), why=sonic.get("why"))
        if "grasp" not in sonic:
            return None, f"SONIC round chose no grasp ({sonic.get('why', '')[:60]})"
        row = int(sonic["grasp"]["row"])
        for c in cands:
            if c["row"] == row:
                return c, ""
        return None, f"SONIC's grasp (cache row {row}) does not fit the object pose here"

    def run(self):
        args, pol = self.args, self.pol
        t_wall = time.time()
        sim = make_object_sim(pol.meta, self.obj_id, video=args.video)
        self.fingers = {
            mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, f"right_dex1_finger_{i}_link"): f"finger_{i}"
            for i in (1, 2)
        }
        ghost = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "ghost_hand")
        self.ghost_mocap = int(sim.m.body_mocapid[ghost]) if ghost >= 0 else -1
        xy, yaw, edge = stance(sim)
        self.stance_yaw = yaw
        sim.reset(xy, yaw, 0.0)
        # the privileged reset's object spawn: the nominal spot, the drawn yaw replacing the attitude
        sim.d.qpos[sim.crate_q + 3 : sim.crate_q + 7] = [np.cos(self.yaw_obj / 2), 0, 0, np.sin(self.yaw_obj / 2)]
        mujoco.mj_forward(sim.m, sim.d)
        refs = R.Refs(pol, sim)
        pol.reset(sim.state())
        pel = refs.pelvis(xy, yaw)
        side = refs.hands_at_side(np.array([*xy, R.PELVIS_Z]), yaw)
        rec = self.record
        rec.update(
            stance_xy=[float(v) for v in xy],
            stance_yaw_deg=float(np.degrees(yaw)),
            edge=edge,
            object_yaw_deg=float(np.degrees(self.yaw_obj)),
            table_top=sim.table_top,
        )
        if args.video:
            self.writer = open_writer(self.out / f"round_{self.rnd:02d}.mp4")
        self.k = 0
        H = C.heading_frame(yaw)

        def ref_static(right):
            return dict(pel, left_wrist_yaw_link=side["left_wrist_yaw_link"], right_wrist_yaw_link=right)

        def step(targets):
            sim.step(pol(sim.state(), targets))
            if self.writer and self.k % 4 == 0:
                self.writer.stdin.write(np.ascontiguousarray(self.frame(sim)).tobytes())
            self.k += 1
            return self.k * R.DT

        # 1. settle at the stance, hands at the side; the object settles from its spawn attitude
        for _ in range(int(round(R.SETTLE_S / R.DT))):
            t = step(refs.futures(lambda _t: ref_static(side["right_wrist_yaw_link"]), 0.0))
            if sim.fallen():
                return self.finish(False, "fell in the stance settle", sim, t_wall)
        pp, pq = sim.body("pelvis")
        op, oq = sim.crate()
        rec["align"] = dict(
            pelvis=[float(v) for v in pp],
            pelvis_err_cm=round(100 * float(np.linalg.norm(pp[:2] - xy)), 2),
            yaw_err_deg=round(float(np.degrees(R.wrap(R.yaw_of(pq) - yaw))), 2),
        )
        rec["object_pos"], rec["object_quat"] = [float(v) for v in op], [float(v) for v in oq]
        rec["object_tilt_deg"] = round(float(np.degrees(2 * np.arccos(min(1.0, float(np.hypot(oq[0], oq[3])))))), 1)

        # 2. perceive: the cache at the settled pose, the program's candidate order
        cands, surface, info = G.cached_candidates(sim.cfg.scene_path(), op, oq, sim.table_top, H[0][:2])
        rec["cache"] = info
        rec["candidates"] = len(cands)
        wp0, wq0 = side["right_wrist_yaw_link"]
        c, why = self.choose(cands, wp0 + R.mat(wq0) @ C.OFFSET, R.mat(wq0), surface, sim.table_top)
        if c is None:
            return self.finish(False, why, sim, t_wall)
        rec["grasp"] = {k: c[k] for k in ("score", "width", "tool", "R", "row", "depth_used", "tier", "slip", "force")}
        R_c = np.asarray(c["R"], dtype=float)
        tool_c = np.asarray(c["tool"], dtype=float)
        direction = R_c[:, 0]
        bias_w = self.bias_w_fn(H)
        rec["bias_world_m"] = [float(v) for v in bias_w]
        if self.ghost_mocap >= 0:
            sim.d.mocap_pos[self.ghost_mocap] = C.wrist_from_tool(tool_c, R_c)
            sim.d.mocap_quat[self.ghost_mocap] = R.quat_wxyz(R_c)
        obj0 = op.copy()
        grip_offset = tool_c - obj0
        pregrasp_true = tool_c - G.PREGRASP_BACK * direction

        # 3. the planned approach, open-loop: up, over, down to the pre-grasp (less the bias), attitude turning on the way
        wp, wq = side["right_wrist_yaw_link"]
        R_side = R.mat(wq)
        sched = Schedule(wp + R_side @ C.OFFSET, R_side)
        legs = C.legs(sched.knots[-1][1], pregrasp_true - bias_w, sim.table_top)
        for li, (_, b) in enumerate(legs):
            sched.move(b, slerp(R_side, R_c, min(1.0, 0.5 * (li + 1))), APPROACH_SPEED)
        t_approach0 = self.k * R.DT
        while self.k * R.DT - t_approach0 < sched.end:
            tt = self.k * R.DT - t_approach0

            def ref(t2):
                pos, rot = sched.at(t2)
                return ref_static((C.wrist_from_tool(pos, rot), R.quat_wxyz(rot)))

            t = step(refs.futures(ref, tt))
            if sim.fallen():
                return self.finish(False, "fell in approach", sim, t_wall)
            if np.linalg.norm(sim.crate()[0][:2] - obj0[:2]) > SIMPLE["disturb_limit_m"]:
                return self.finish(False, "target knocked in approach", sim, t_wall)
        rec["timeline"] = [dict(phase="approach", seconds=round(sched.end, 2))]
        rec["pregrasp_error_open_loop_m"] = float(np.linalg.norm(sim.pad("right") - pregrasp_true))

        # 4. the closed loop: settle -> entry (gate, lateral correction, retry) -> close -> lift -> recover
        tracking = Tracking(args.tracking == "on")
        phase, began = "settle", self.k * R.DT
        phases = rec["expert_phases"] = [dict(phase=phase, t=began)]
        target = sim.pad("right").copy()
        # SupportedMotion takes over from the streamed joints: its first error is the planned path's end minus the
        # measured tool, i.e. the open-loop delivery error, and that becomes the feedback origin (preserved offset)
        corrected_prev = pregrasp_true - bias_w
        entry_cmd = lift_from = retreat_from = retreat_to = None
        shift = np.zeros(3)
        gate_held = closure = 0.0
        z_before = None
        hold_offset0 = None
        prev_target = target.copy()
        dt = R.DT

        def enter(name, stamp):
            nonlocal phase, began
            phases.append(dict(phase=name, t=stamp))
            phase, began = name, stamp

        while True:
            stamp = self.k * R.DT
            tool_m = sim.pad("right")
            _, wq_m = sim.body("right_wrist_yaw_link")
            palm = R.mat(wq_m)
            obj_p, _ = sim.crate()
            touch = self.contacts(sim)
            bilateral = {"finger_1", "finger_2"} <= touch
            elapsed = stamp - began
            grip = obj_p + grip_offset
            pregrasp = grip - G.PREGRASP_BACK * direction
            if phase in ("settle", "entry", "retreat"):
                knocked = float(np.linalg.norm(obj_p[:2] - obj0[:2]))
                if knocked > SIMPLE["disturb_limit_m"]:
                    return self.finish(False, f"target knocked {knocked * 100:.0f} cm in {phase}", sim, t_wall)
            if phase == "settle":
                target = pregrasp
                err = float(np.linalg.norm(tool_m - pregrasp))
                if (err <= SIMPLE["settle_tolerance_m"] and elapsed > 0.3) or elapsed >= SIMPLE["settle_timeout_s"]:
                    rec["settle"] = dict(error_m=err, seconds=elapsed)
                    entry_cmd = target.copy()
                    enter("entry", stamp)
            elif phase == "entry":
                off = object_in_palm(palm, tool_m, grip)
                if (
                    G.ENTRY_GATE["on"]
                    and max(abs(off[1]), abs(off[2])) > G.ENTRY_GATE["lateral_m"]
                    and gate_held < G.ENTRY_GATE["hold_s"]
                ):
                    gate_held += dt
                else:
                    entry_cmd = entry_command(entry_cmd, grip, direction, G.DASH_SPEED, dt, SIMPLE["entry_overshoot_m"])
                wanted = shift + palm[:, 1] * off[1] + palm[:, 2] * off[2]
                delta = wanted - shift
                norm = float(np.linalg.norm(delta))
                if norm > 1e-9:
                    shift = shift + delta * min(1.0, SIMPLE["lateral_correction_speed_m_s"] * dt / norm)
                total = float(np.linalg.norm(shift))
                if total > SIMPLE["lateral_correction_max_m"]:
                    shift *= SIMPLE["lateral_correction_max_m"] / total
                target = entry_cmd + shift
                arrived = entry_captured(tool_m, grip, direction, SIMPLE["entry_capture_depth_m"])
                if arrived or bilateral:
                    off = object_in_palm(palm, tool_m, grip)
                    lateral_tol = max(0.005, (2 * G.PAD_Y - c["width"]) / 2 + 0.005)
                    aligned = abs(off[1]) <= lateral_tol and abs(off[2]) <= G.PAD_Z
                    entries = rec.setdefault("entries", [])
                    entries.append(
                        dict(
                            arrived=bool(arrived), bilateral=bool(bilateral), offset_in_palm=off, aligned=bool(aligned)
                        )
                    )
                    if aligned or bilateral or len(entries) >= max(2, int(SIMPLE["entry_attempts"])):
                        rec["entry"] = dict(
                            arrived=bool(arrived),
                            bilateral=bool(bilateral),
                            error_m=float(np.linalg.norm(tool_m - grip)),
                            offset_in_palm=off,
                            shift_m=float(np.linalg.norm(shift)),
                            attempts=len(entries),
                            gate_held_s=gate_held,
                        )
                        target = tool_m.copy()
                        enter("close", stamp)
                    else:
                        retreat_from = target.copy()
                        retreat_to = retreat_from - SIMPLE["retreat_m"] * direction
                        shift = shift + palm[:, 1] * off[1] + np.array([0.0, 0.0, off[2]])
                        enter("retreat", stamp)
                elif elapsed > SIMPLE["entry_timeout_s"]:
                    return self.finish(
                        False,
                        f"entry timed out {np.linalg.norm(tool_m - grip) * 100:.1f} cm from the grasp",
                        sim,
                        t_wall,
                    )
            elif phase == "retreat":
                target = retreat_from + minjerk(elapsed / SIMPLE["retreat_seconds"]) * (retreat_to - retreat_from)
                if elapsed >= SIMPLE["retreat_seconds"]:
                    entry_cmd = target.copy() - shift
                    enter("entry", stamp)
            elif phase == "close":
                closure = min(1.0, elapsed / SIMPLE["close_seconds"])
                if elapsed >= SIMPLE["close_seconds"]:
                    if bilateral:
                        lift_from = tool_m.copy()
                        z_before = float(obj_p[2])
                        rec["closed_touching"] = True
                        rec["close_offset_in_palm"] = [float(v) for v in object_in_palm(palm, tool_m, grip)]
                        enter("lift", stamp)
                    elif elapsed >= SIMPLE["close_seconds"] + SIMPLE["contact_timeout_s"]:
                        rec["closed_touching"] = False
                        return self.finish(False, "no bilateral pad contact after the close", sim, t_wall)
            elif phase == "lift":
                target = lift_from + [0.0, 0.0, G.LIFT_M * minjerk(elapsed / SIMPLE["lift_seconds"])]
                if elapsed >= SIMPLE["lift_seconds"]:
                    hold_offset0 = object_in_palm(palm, tool_m, obj_p)
                    rec["lift_offset_in_palm"] = [float(v) for v in hold_offset0]
                    enter("recover", stamp)
            elif phase == "recover":
                target = lift_from + [0.0, 0.0, G.LIFT_M]
                if elapsed >= args.hold:
                    rise = float(obj_p[2]) - z_before
                    slip = object_in_palm(palm, tool_m, obj_p) - hold_offset0
                    rec.update(rise_m=rise, touching_end=bool(bilateral), hold_slip_m=float(np.linalg.norm(slip)))
                    if not bilateral:
                        return self.finish(
                            False, f"lost pad contact during the lift (rise {rise * 100:.1f} cm)", sim, t_wall
                        )
                    if rise < RISE_MIN:
                        return self.finish(False, f"not lifted (rise {rise * 100:.1f} cm)", sim, t_wall)
                    return self.finish(True, "", sim, t_wall)
            # the controller swap: SupportedMotion's feedback on the wrist reference, the command velocity as the lead
            fb = tracking.step(corrected_prev, tool_m, dt)
            corrected = target + fb
            v_cmd = (target - prev_target) / dt
            prev_target = target.copy()
            corrected_prev = corrected.copy()
            futures = {
                n: (np.repeat(p[None], pol.F, axis=0), np.repeat(q[None], pol.F, axis=0))
                for n, (p, q) in ref_static((C.wrist_from_tool(corrected, R_c), R.quat_wxyz(R_c))).items()
            }
            lead = np.array([j * R.DT for j in pol.fidx])[:, None] * v_cmd[None]
            futures["right_wrist_yaw_link"] = (
                futures["right_wrist_yaw_link"][0] + lead,
                futures["right_wrist_yaw_link"][1],
            )
            sim.grip_target[2:] = JAW_OPEN + closure * (JAW_CLOSED - JAW_OPEN)
            self.trace.append(
                [
                    stamp, PHASES.index(phase), *target, *corrected, *tool_m, *obj_p, closure, float(bilateral),
                    float("finger_1" in touch), float("finger_2" in touch), *fb, *sim.body("pelvis")[0],
                ]
            )  # fmt: skip
            step(futures)
            warnings = {mujoco.mjtWarning(j).name: int(w.number) for j, w in enumerate(sim.d.warning) if w.number}
            if sim.fallen() or warnings or not np.isfinite(sim.d.qpos).all():
                return self.finish(
                    False, f"fell in {phase}" if sim.fallen() else f"physics warning in {phase}", sim, t_wall
                )

    def finish(self, ok, why, sim, t_wall):
        rec = self.record
        rec.update(ok=ok, why=why, sim_s=round(self.k * R.DT, 2), seconds=round(time.time() - t_wall, 1))
        rec["tracking"] = self.args.tracking
        rec["jaw_q_final"] = [float(v) for v in sim.d.qpos[sim.gq[2:]]]
        if self.trace:
            np.savez_compressed(
                self.out / "trace.npz",
                trace=np.asarray(self.trace),
                columns="t phase target3 corrected3 actual3 object3 closure bilateral f1 f2 feedback3 pelvis3",
            )
        (self.out / "record.json").write_text(json.dumps(rec, default=lambda o: np.asarray(o).tolist(), indent=1))
        if self.writer:
            self.writer.stdin.close()
            self.writer.wait()
            full = self.out / f"round_{self.rnd:02d}.mp4"
            preview(full, self.out / f"round_{self.rnd:02d}_preview.mp4")
        return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default="all", help="'all' or comma-separated ids")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument(
        "--yaws", type=int, default=1, help="object yaws drawn per object (rng 1000 + id), one per round, cycling"
    )
    ap.add_argument("--select", choices=("paired", "score", "cache"), default="paired")
    ap.add_argument(
        "--pair-runs", default="", help="tag of the SONIC runs whose round-r grasp the paired selection takes"
    )
    ap.add_argument("--bias", type=Path, default=None, help="probe/bias.json; the scalebfm.<bias-key> constant")
    ap.add_argument("--bias-key", default="all")
    ap.add_argument(
        "--envelope", type=Path, default=None,
        help="probe/scalebfm_attitude/envelope.json (attitude_summary.py): --select score/cache skip the candidates whose "
        "palm attitude ScaleBFM did not deliver in the attitude probe (TODO 24.6)",
    )  # fmt: skip
    ap.add_argument(
        "--tracking", choices=("on", "off"), default="on", help="SupportedMotion's feedback law on the wrist"
    )
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    assert_matches_grasp_cache()
    args.envelope = C.Envelope(args.envelope) if args.envelope is not None else None
    bias_m = np.zeros(3)
    if args.bias is not None:
        bias_m = np.asarray(json.loads(args.bias.read_text())["scalebfm"][args.bias_key]["bias_m"], dtype=float)

    def bias_w_fn(H):
        return H.T @ bias_m

    ids = sorted(objects()) if args.objects == "all" else [int(v) for v in args.objects.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    pol = R.Policy(device=args.device, backend=ScaleBfmBackendConfig())
    summary = []
    for oid in ids:
        yaws = np.random.default_rng(1000 + oid).uniform(-np.pi, np.pi, size=args.yaws).tolist()
        for rnd in range(args.rounds):
            rec = Round(pol, oid, rnd, yaws[rnd % len(yaws)], args, bias_w_fn).run()
            summary.append(
                {
                    k: rec.get(k)
                    for k in (
                        "object",
                        "round",
                        "ok",
                        "why",
                        "rise_m",
                        "hold_slip_m",
                        "pregrasp_error_open_loop_m",
                        "sim_s",
                        "seconds",
                    )
                }
            )
            print(
                f"obj {oid:03d} round {rnd}: {'OK' if rec['ok'] else 'FAIL'} {rec['why']} | pregrasp open-loop "
                f"{100 * rec.get('pregrasp_error_open_loop_m', float('nan')):.1f} cm, entry {rec.get('entry', {}).get('error_m', float('nan')) * 100:.1f} cm, "
                f"rise {rec.get('rise_m', float('nan'))}, slip {rec.get('hold_slip_m', float('nan'))} ({rec['sim_s']} sim s, {rec['seconds']} s)",
                flush=True,
            )
            (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    ok = sum(bool(s["ok"]) for s in summary)
    print(f"{ok}/{len(summary)} lifted; wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()

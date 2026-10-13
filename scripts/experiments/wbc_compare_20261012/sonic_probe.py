"""wbc_compare step 2 (TODO 24.2): the open-loop right-hand probe on the SONIC lockstep stack.

The same cells as bfm_probe.py (probe_cells), delivered the way the 65-object line's hand program delivers a grasp
(grasp_planned.GraspPlanned, 2026-10-12 defaults): the pose stream takes the robot over, align_robot_xy puts the
settled pelvis at the stance, the right arm's path to each cell is solved by the staged IK every centimetre (up to
OVER_CLEAR above the table, over, down; ArmIK with grasp_planned's limits, waist allowed, posture pull toward the
neutral arm) and streamed at JOINT_SPEED with the sim-side arm pin (DECOUPLE), then held HOLD_S. The delivered tool
is read from the privileged state's wrist pose over the hold's last MEASURE_S. Each cell's error is split three ways
so the comparison can say where SONIC's delivery goes: the IK's own residual (FK of the solved joints against the
target, in the pelvis frame at plan time), the arm/waist tracking (measured tool against the FK of the streamed
joints at the measured pelvis), and the pelvis motion since the plan (the stance drifting under the arm). Their sum
is the total the feed-forward bias corrects.

One `hand_probe` plan step on one object scene (the object is parked behind the robot by pin_object), run through
scripts/plan_episode.py exactly as run_grasp.py runs grasp_planned, so the spawn, stack and video are the line's own.

    bash scripts/experiments/wbc_compare_20261012/sonic_probe.sh 40 1 0      # object, instance, gpu
        -> outputs/wbc_compare_20261012/probe/sonic/probe.json, outputs/runs/wbc_probe_040/videos/round_00.mp4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRASPNET = REPO / "scripts" / "experiments" / "graspnet_bottle_20261008"
ROOT = REPO / "outputs" / "graspnet_bottle_20261008"
sys.path[:0] = [str(HERE), str(GRASPNET), str(REPO / "src")]

import probe_cells as C  # noqa: E402

HOLD_S = 2.0
MEASURE_S = 0.5


def build_probe_step(grasp_planned, profiles, out_dir: Path, hold_s: float):
    """The plan step: GraspPlanned's takeover and stance, then the cells instead of a grasp."""
    from locomani.backends.sonic.primitives import Primitives

    class HandProbe(grasp_planned.GraspPlanned):
        name = "hand_probe"

        def __init__(self):
            super().__init__(profiles, 0, 0, out_dir)

        def preconditions(self, session, **args):
            return []

        def run(
            self, session, *, target: str = "water_bottle", profile: str = "bottle_simple", planner=None, source=None
        ):
            from locomani.backends.sonic.pose_preview import PosePreview
            from locomani.backends.sonic.primitives import HAND_OPEN
            from locomani.backends.sonic.primitives_tuning import READY_ARM
            from locomani.backends.sonic.streamed_arm import StreamedArm
            from locomani.skills.contract import Outcome
            from locomani.skills.session import ArmOwner

            t0 = session.clock()
            p, meta = session.stack, session.scene.meta
            cfg = self._profiles[profile].config.reach.whole_body
            log = p.log
            out_dir.mkdir(parents=True, exist_ok=True)
            table_top = float(meta["surfaces"][meta["objects"][target].get("surface", "A")]["top"])
            record = dict(controller="sonic", hold_s=hold_s, measure_s=MEASURE_S, cells=[])

            def finish(ok, why):
                if grasp_planned.DECOUPLE["on"]:
                    session.reset.pin_arms()
                session.reset.pin_object(target)
                record.update(ok=ok, why=why, seconds=session.clock() - t0, decoupled_arm=dict(grasp_planned.DECOUPLE))
                (out_dir / "probe.json").write_text(
                    json.dumps(record, default=lambda o: np.asarray(o).tolist(), indent=1)
                )
                log(f"[hand_probe] wrote {out_dir / 'probe.json'} ({len(record['cells'])} cells, ok={ok} {why})")
                return Outcome(ok, "" if ok else f"[hand_probe] {why}", duration_s=record["seconds"], metrics=record)

            ik = self._ik_setup(cfg)
            arm = StreamedArm(p, "right", ik=ik, left_guard=False)
            grasp_planned.apply_waist_limits(ik)  # as the program since 2026-10-12 (TODO 24.2.2)
            arm.preview = PosePreview(
                acceleration=cfg.joint_acceleration, joint_limits=(ik.lo, ik.hi), continuous_handoff=True
            )
            arm.hands = {"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()}
            p.cmd.set_hands(left=HAND_OPEN.copy(), right=HAND_OPEN.copy())
            arm.start(hold_planner_pose=True, inherit_planner_velocity=True)
            arm.brake_to_stop()
            session.arm, session.arm_owner = arm, ArmOwner.POSE_STREAM
            state = p.obs()
            yaw_now = float(p.pelvis_yaw(state))
            f = np.array([np.cos(yaw_now), np.sin(yaw_now)])
            want = (
                np.asarray(state["objects"][target]["pos"][:2])
                - grasp_planned.STAND_FORWARD * f
                - grasp_planned.STAND_RIGHT * np.array([f[1], -f[0]])
            )
            session.reset.align_robot_xy(want)
            arm.hold(0.6)
            state = p.obs()
            yaw = float(p.pelvis_yaw(state))  # the stance heading: the settled one (grasp_planned rule own_heading)
            record.update(
                stance_xy=want,
                stance_yaw_deg=float(np.degrees(yaw)),
                pelvis_after_align=np.asarray(state["pelvis"]["pos"]),
                table_top=table_top,
            )
            session.reset.pin_object(target, pos=C.park_position(want, yaw).tolist())
            arm.hold(0.3)
            log(f"[hand_probe] stance {np.round(want, 3)} yaw {np.degrees(yaw):.1f} deg, object parked")
            q0 = np.asarray(arm.q, dtype=float).copy()
            neutral = q0.copy()
            neutral[12:15] = 0.0
            neutral[22:29] = READY_ARM["right"]
            R_c = C.palm_rotation(yaw)
            home = C.home_tool(want, yaw, table_top)

            def solve_path(q, pel_R, pel_t, legs, R_from, R_to, turn_legs):
                """grasp_planned's staged solve, every centimetre; the attitude turns over `turn_legs` legs."""
                path = [q[self.active].copy()]
                worst = (0.0, 0.0)
                for li, (a, b) in enumerate(legs):
                    n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.01)))
                    for k in range(1, n + 1):
                        u = k / n
                        pos_w = a + u * (b - a)
                        s = min(1.0, (li + u) / turn_legs) if turn_legs else 1.0
                        rot_w = _slerp(R_from, R_to, s)
                        qn, perr, rerr = ik.solve(
                            "right", pel_R.T @ (pos_w - pel_t), pel_R.T @ rot_w, q, iters=80, rot_weight=1.0,
                            use_waist=True, point_offset=C.OFFSET, posture=neutral,
                            posture_weight=grasp_planned.POSTURE_WEIGHT,
                        )  # fmt: skip
                        worst = (max(worst[0], float(perr)), max(worst[1], float(np.degrees(rerr))))
                        if np.abs(qn[self.active] - q[self.active]).max() > 0.35:
                            return q, path, worst, f"leg {li} step {k}: joint jump"
                        q = qn
                        path.append(q[self.active].copy())
                return q, path, worst, None

            def stream(path10, phase):
                base = np.asarray(arm.q, dtype=float).copy()
                full = np.repeat(base[None], len(path10) + 1, axis=0)
                full[1:, self.active] = path10
                P, s_wp, total = grasp_planned.retime(full)
                began = previous = None
                wall = time.monotonic()
                while True:
                    st = p.obs()
                    stamp = float(st["t"])
                    if previous is not None and stamp == previous:
                        if time.monotonic() - wall > 2.0:
                            return f"physical clock stalled in {phase}"
                        time.sleep(0.004)
                        continue
                    wall = time.monotonic()
                    began = stamp if began is None else began
                    previous = stamp
                    if st.get("fallen"):
                        return f"fell in {phase}"
                    q, v = grasp_planned.sample(P, s_wp, total, stamp - began)
                    arm.q = q
                    arm._push(q, reference_velocity=v, reference_time=stamp)
                    self._pin(session, q)
                    if stamp - began >= total:
                        return None

            # from the hanging arm: up, then to HOME, the attitude turning to the cells' on the way
            st = p.obs()
            pel_t = np.asarray(st["pelvis"]["pos"], dtype=float)
            pel_R = Primitives._rot(st["pelvis"]["quat"])
            tool0_p, R0_p = self._tool_fk(ik, q0, C.OFFSET)
            q_home, path, worst, failed = solve_path(
                q0, pel_R, pel_t, C.legs(pel_R @ tool0_p + pel_t, home, table_top)[:2], pel_R @ R0_p, R_c, 2
            )
            record["home"] = dict(tool=home, path_worst_ik_cm_deg=[100 * worst[0], worst[1]], failed=failed)
            if failed:
                return finish(False, f"no path from the hanging arm to HOME: {failed}")
            why = stream(np.asarray(path), "home")
            if why:
                return finish(False, why)
            arm.hold(0.5)
            tool_prev = home
            back_path = [q_home[self.active].copy()]  # the joint path back to HOME from the last cell
            for cell in C.cells():
                st = p.obs()
                pel_t = np.asarray(st["pelvis"]["pos"], dtype=float)
                pel_R = Primitives._rot(st["pelvis"]["quat"])
                tool_c, _ = C.tool_pose(want, yaw, cell)
                legs = C.legs_via_home(tool_prev, tool_c, home)
                q = np.asarray(arm.q, dtype=float).copy()
                q, path, worst, failed = solve_path(q, pel_R, pel_t, legs, R_c, R_c, 0)
                if failed and len(legs) > 2:
                    # back to HOME on the joint path we came down on, then the cell from the common start
                    why = stream(np.asarray(back_path), f"back to home before cell {cell['id']}")
                    if why:
                        return finish(False, why)
                    q = q_home.copy()
                    q, path, worst, failed = solve_path(q, pel_R, pel_t, legs[-2:], R_c, R_c, 0)
                fk_tool_p, fk_R_p = self._tool_fk(ik, q, C.OFFSET)
                plan_tool_w = pel_R @ fk_tool_p + pel_t  # the IK's promise, in the world at plan time
                final_err = (float(np.linalg.norm(plan_tool_w - tool_c)), C.rot_err_deg(R_c, pel_R @ fk_R_p))
                row = dict(
                    cell,
                    command_tool=tool_c,
                    path_worst_ik_cm_deg=[100 * worst[0], worst[1]],
                    ik_final_cm_deg=[100 * final_err[0], final_err[1]],
                    reachable=bool(final_err[0] <= C.IK_POS_TOL and final_err[1] <= C.IK_ROT_TOL_DEG and not failed),
                )
                if failed:
                    row.update(skipped=failed)
                    record["cells"].append(row)
                    log(f"[hand_probe] cell {cell['id']} skipped: {failed}")
                    tool_prev = home
                    back_path = [q_home[self.active].copy()]
                    continue
                tool_prev = tool_c
                why = stream(np.asarray(path), f"cell {cell['id']}")
                if why:
                    row.update(skipped=why)
                    record["cells"].append(row)
                    return finish(False, why)
                down = max(1, int(np.ceil(np.linalg.norm(legs[-1][1] - legs[-1][0]) / 0.01)))
                back_path = list(reversed(path[-down - 1 :])) + [q_home[self.active].copy()]
                trace = []
                samples = []
                ticks = int(round(hold_s / 0.02))
                for i in range(ticks):
                    arm.hold(0.02)
                    st = p.obs()
                    tool_m = (
                        np.asarray(st["palms"]["right"]["pos"])
                        + Primitives._rot(st["palms"]["right"]["quat"]) @ C.OFFSET
                    )
                    trace.append(round(100 * float(np.linalg.norm(tool_m - tool_c)), 2))
                    if i >= ticks - int(round(MEASURE_S / 0.02)):
                        samples.append(st)
                    if st.get("fallen"):
                        return finish(False, f"fell holding cell {cell['id']}")
                tool_m = np.mean(
                    [
                        np.asarray(s["palms"]["right"]["pos"]) + Primitives._rot(s["palms"]["right"]["quat"]) @ C.OFFSET
                        for s in samples
                    ],
                    axis=0,
                )
                R_m = Primitives._rot(samples[-1]["palms"]["right"]["quat"])
                pel_t_now = np.mean([np.asarray(s["pelvis"]["pos"]) for s in samples], axis=0)
                pel_R_now = Primitives._rot(samples[-1]["pelvis"]["quat"])
                qpos = samples[-1].get("qpos")
                q_meas = np.asarray(qpos, dtype=float)[ik.qadr] if qpos is not None else None  # 29 joints by name
                err = tool_m - tool_c
                ik_res = plan_tool_w - tool_c
                fk_now = pel_R_now @ fk_tool_p + pel_t_now
                tracking = tool_m - fk_now
                pelvis_part = fk_now - plan_tool_w
                row.update(
                    measured_tool=tool_m,
                    err_heading_cm=[round(100 * v, 2) for v in C.to_heading(err, yaw)],
                    err_norm_cm=round(100 * float(np.linalg.norm(err)), 2),
                    ik_residual_heading_cm=[round(100 * v, 2) for v in C.to_heading(ik_res, yaw)],
                    tracking_heading_cm=[round(100 * v, 2) for v in C.to_heading(tracking, yaw)],
                    pelvis_motion_heading_cm=[round(100 * v, 2) for v in C.to_heading(pelvis_part, yaw)],
                    rot_err_deg=round(C.rot_err_deg(R_c, R_m), 1),
                    palm_x_heading=[round(v, 3) for v in C.to_heading(R_m[:, 0], yaw)],
                    pelvis_err_cm=[
                        round(100 * v, 2) for v in C.to_heading(pel_t_now - np.array([*want, 0.0]), yaw)[:2]
                    ],
                    pelvis_yaw_err_deg=round(float(np.degrees(_wrap(_yaw(pel_R_now) - yaw))), 2),
                    pelvis_z=round(float(pel_t_now[2]), 3),
                    waist=[round(float(v), 3) for v in q[12:15]],
                    hold_trace_cm=trace,
                    hold_max_cm=max(trace),
                    path_waypoints=len(path),
                )
                if q_meas is not None:
                    row["arm_tracking_rad"] = [round(float(v), 4) for v in (q_meas[22:29] - q[22:29])]
                    row["waist_measured"] = [round(float(v), 3) for v in q_meas[12:15]]
                record["cells"].append(row)
                log(
                    f"[hand_probe] cell {cell['id']:2d} fwd {cell['forward']:.2f} left {cell['left']:+.2f} z {cell['z']:.2f}: "
                    f"err f/l/u {row['err_heading_cm']} cm (ik {row['ik_residual_heading_cm']}, track {row['tracking_heading_cm']}, "
                    f"pelvis {row['pelvis_motion_heading_cm']}) rot {row['rot_err_deg']} deg"
                )
            return finish(True, "")

    return HandProbe()


def _slerp(Ra, Rb, u):
    from scipy.spatial.transform import Rotation, Slerp

    return Slerp([0.0, 1.0], Rotation.from_matrix([Ra, Rb]))(float(u)).as_matrix()


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", type=int, default=40)
    ap.add_argument("--instance", type=int, default=1)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "wbc_compare_20261012" / "probe" / "sonic")
    args = ap.parse_args()

    from locomani.sim.ports import instance_ports

    os.environ["INSTANCE"] = str(args.instance)  # every port default is read at import time: pin the block first
    os.environ.update({k: str(v) for k, v in instance_ports(args.instance).items()})
    from locomani import paths

    oid = args.object
    paths.adopt_sim_dir(str(ROOT / "scenes" / f"obj_{oid:03d}"))  # before any module copies the default scene
    spec = importlib.util.spec_from_file_location("plan_episode", REPO / "scripts" / "plan_episode.py")
    plan_episode = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_episode)
    import grasp_planned
    import run_grasp

    import locomani.skills.catalogue as catalogue
    from locomani.skills.profiles import load_profiles

    original_build = catalogue.build_catalogue
    plan = ROOT / "experiments" / "plan_hand_probe.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps([{"primitive": "hand_probe", "args": {"target": "water_bottle"}}]))
    name = f"wbc_probe_{oid:03d}{args.tag}"

    def build(profiles=None, **overrides):
        entries = original_build(profiles, **overrides)
        entries["hand_probe"] = build_probe_step(grasp_planned, profiles or load_profiles(), args.out, args.hold)
        return entries

    catalogue.build_catalogue = build
    yaw = float(np.random.default_rng(1000 + oid).uniform(-np.pi, np.pi))  # run_grasp's round-0 yaw for the object
    plan_episode.spawn_object_yaw = lambda half_extent, psi, rng: yaw
    import locomani.skills.catalogue.bottle as bottle_skills

    stand_plan = bottle_skills.stand_plan
    base_meta = json.loads((REPO / "outputs" / "scene" / "sim_lowcom" / "scene_meta.json").read_text())

    def stand_plan_any_height(meta, target, *a, **kw):
        try:
            return stand_plan(meta, target, *a, **kw)
        except ValueError:
            return stand_plan(base_meta, target, *a, **kw)

    bottle_skills.stand_plan = stand_plan_any_height
    argv = [
        str(run_grasp.experiment_for(oid)),
        "--gpu", str(args.gpu),
        "--instance", str(args.instance),
        "--rounds", "1",
        "--seed", str(5000 + oid),
        "--spawn-off", "0.5",
        "--plan", str(plan),
        "--name", name,
    ]  # fmt: skip
    print(
        f"[sonic_probe] object {oid:03d} on instance {args.instance} gpu {args.gpu} -> outputs/runs/{name}", flush=True
    )
    code = plan_episode.main(argv)
    print(f"[sonic_probe] done: {code}; probe at {args.out / 'probe.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

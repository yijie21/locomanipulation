"""B3, SONIC side (TODO 25.2.2): b3_cells.py's 108 cells from SONIC's own walked-in bottle stance.

wbc_compare's 24.2 probe (scripts/experiments/wbc_compare_20261012/sonic_probe.py) with three changes: the plan is
`approach` (the bottle_simple profile's walk to the table) then the probe, and the probe starts from wherever the walk
stopped -- no align_robot_xy; the cells are b3_cells.cells() (the 24.2 positions at four palm attitudes; each attitude
starts from the hanging arm: down the way the last one came up, up again turning to the new one); the scene is the stock
bottle scene (sim_lowcom), the bottle parked behind the robot after the walk; `approach` can stop with the bottle out of reach, so up to --rounds rounds run and
the first that walks in is probed (later rounds see b3.json and return). Delivery as 24.2: the staged IK every centimetre, streamed with the sim-side arm pin,
held HOLD_S, the tool read from the privileged state over the hold's last MEASURE_S.

    bash scripts/experiments/controller_compare_20261013/sonic_b3_probe.sh 0 0      # instance, gpu
        -> outputs/controller_compare_20261013/bottle/b3/sonic/b3.json
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
WBC = REPO / "scripts" / "experiments" / "wbc_compare_20261012"
sys.path[:0] = [str(HERE), str(WBC), str(GRASPNET), str(REPO / "src")]

import b3_cells  # noqa: E402
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
            if (out_dir / "b3.json").exists():
                # a later round after a probed one: end it (an ok step would have to own the pose stream)
                return Outcome(False, "[hand_probe] already probed in an earlier round", duration_s=0.0)
            p, meta = session.stack, session.scene.meta
            cfg = self._profiles[profile].config.reach.whole_body
            log = p.log
            out_dir.mkdir(parents=True, exist_ok=True)
            table_top = float(meta["surfaces"][meta["objects"][target].get("surface", "A")]["top"])
            record = dict(
                controller="sonic", hold_s=hold_s, measure_s=MEASURE_S, attitudes=list(b3_cells.ATTITUDES), cells=[]
            )

            def finish(ok, why):
                if grasp_planned.DECOUPLE["on"]:
                    session.reset.pin_arms()
                session.reset.pin_object(target)
                record.update(ok=ok, why=why, seconds=session.clock() - t0, decoupled_arm=dict(grasp_planned.DECOUPLE))
                (out_dir / "b3.json").write_text(json.dumps(record, default=lambda o: np.asarray(o).tolist(), indent=1))
                log(f"[hand_probe] wrote {out_dir / 'b3.json'} ({len(record['cells'])} cells, ok={ok} {why})")
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
            arm.hold(0.6)
            state = p.obs()
            want = np.asarray(p.pelvis_xy(state), dtype=float)  # the walked-in stance: where approach stopped
            yaw = float(p.pelvis_yaw(state))
            record.update(
                stance_xy=want,
                stance_yaw_deg=float(np.degrees(yaw)),
                pelvis_after_align=np.asarray(state["pelvis"]["pos"]),
                table_top=table_top,
            )
            session.reset.pin_object(target, pos=C.park_position(want, yaw).tolist())
            arm.hold(0.3)
            log(f"[hand_probe] walked-in stance {np.round(want, 3)} yaw {np.degrees(yaw):.1f} deg, object parked")
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

            tool0_p, R0_p = self._tool_fk(ik, q0, C.OFFSET)
            record["home"] = {}

            def to_home(att_id, R_att):
                """From the hanging arm (q0): up, then to HOME, the palm turning to R_att on the way -- the start of
                every attitude, so no attitude is reached by twisting the wrist from the previous one."""
                st = p.obs()
                pel_t = np.asarray(st["pelvis"]["pos"], dtype=float)
                pel_R = Primitives._rot(st["pelvis"]["quat"])
                legs = C.legs(pel_R @ tool0_p + pel_t, home, table_top)[:2]
                q, path, worst, failed = solve_path(q0.copy(), pel_R, pel_t, legs, pel_R @ R0_p, R_att, 2)
                record["home"][att_id] = dict(tool=home, path_worst_ik_cm_deg=[100 * worst[0], worst[1]], failed=failed)
                if failed:
                    return None, path, f"no path from the hanging arm to HOME: {failed}"
                why = stream(np.asarray(path), f"home, attitude {att_id}")
                if why:
                    return None, path, why
                arm.hold(0.5)
                return q, path, None

            q_home = home_path = None  # this attitude's HOME posture and the joint path up to it from q0
            att_now = None
            back_path = []  # the joint path back to HOME from wherever the arm is
            for pos, att in b3_cells.cells():
                cell = dict(pos, id=len(record["cells"]), position=pos["id"], attitude=att["id"])
                cell.update(elev=att["elev"], yaw_dev=att["yaw_dev"], roll=att["roll"])
                R_c = C.palm_rotation_at(yaw, att["elev"], att["yaw_dev"], att["roll"])
                if att["id"] != att_now:
                    # a new attitude: back down to the hanging arm the way we came up, then up again at the new one
                    if home_path is not None:
                        why = stream(np.asarray(back_path), f"back to home after attitude {att_now}")
                        why = why or stream(np.asarray(list(reversed(home_path))), "down to the hanging arm")
                        if why:
                            return finish(False, why)
                    att_now = att["id"]
                    q_home, home_path, why = to_home(att_now, R_c)
                    if q_home is None:
                        if why.startswith(("fell", "physical clock stalled")):
                            return finish(False, why)
                        home_path = None  # the IK found no path, nothing was streamed: the arm still hangs at q0
                    tool_prev = home
                    back_path = [] if q_home is None else [q_home[self.active].copy()]
                if q_home is None:
                    record["cells"].append(dict(cell, skipped=f"attitude {att_now}: no HOME", deliverable=False))
                    continue
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
                    back_path = [q_home[self.active].copy()]
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
                    row.update(skipped=failed, deliverable=False)
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
                row["deliverable"] = b3_cells.deliverable(row)
                if q_meas is not None:
                    row["arm_tracking_rad"] = [round(float(v), 4) for v in (q_meas[22:29] - q[22:29])]
                    row["waist_measured"] = [round(float(v), 3) for v in q_meas[12:15]]
                record["cells"].append(row)
                log(
                    f"[hand_probe] cell {cell['id']:3d} att {cell['attitude']} fwd {cell['forward']:.2f} left {cell['left']:+.2f} z {cell['z']:.2f}: "
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
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--seed", type=int, default=40000, help="plan_episode's placement seed (the bottle L1 batch's)")
    ap.add_argument(
        "--rounds", type=int, default=4, help="rounds until one walks in (approach can stop out of reach); probed once"
    )
    ap.add_argument("--name", default="controller_compare_bottle_b3_sonic_20261013")
    ap.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "controller_compare_20261013" / "bottle" / "b3" / "sonic"
    )
    args = ap.parse_args()

    from locomani.sim.ports import instance_ports

    os.environ["INSTANCE"] = str(args.instance)  # every port default is read at import time: pin the block first
    os.environ.update({k: str(v) for k, v in instance_ports(args.instance).items()})
    spec = importlib.util.spec_from_file_location("plan_episode", REPO / "scripts" / "plan_episode.py")
    plan_episode = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_episode)
    import grasp_planned

    import locomani.skills.catalogue as catalogue
    from locomani.skills.profiles import load_profiles

    original_build = catalogue.build_catalogue
    plan = args.out / "plan_b3.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(
        json.dumps(
            [
                {"primitive": "approach", "args": {"target": "water_bottle", "profile": "bottle_simple"}},
                {"primitive": "hand_probe", "args": {"target": "water_bottle"}},
            ]
        )
    )

    def build(profiles=None, **overrides):
        entries = original_build(profiles, **overrides)
        entries["hand_probe"] = build_probe_step(grasp_planned, profiles or load_profiles(), args.out, args.hold)
        return entries

    catalogue.build_catalogue = build
    argv = [
        str(REPO / "configs" / "experiments" / "bottle_simple_200.yaml"),
        "--gpu", str(args.gpu),
        "--instance", str(args.instance),
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
        "--plan", str(plan),
        "--name", args.name,
    ]  # fmt: skip
    print(f"[sonic_b3] instance {args.instance} gpu {args.gpu} -> outputs/runs/{args.name}", flush=True)
    code = plan_episode.main(argv)
    print(f"[sonic_b3] done: {code}; probe at {args.out / 'b3.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

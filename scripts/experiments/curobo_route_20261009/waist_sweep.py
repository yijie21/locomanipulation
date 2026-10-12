"""How much waist the CuRobo route needs, and what it costs at the table (route plan section 6, 2026-10-09).

For rounds of a curobo collect run (the first B, whose routes had the joint limits only), the request each round sent is
rebuilt from its record and state trace: start and goal from the route record, the obstacles from the scene at the
route's first frame (arm_route.scene_obstacles). It is planned again on the route service at each waist-roll bound
(yaw and the pitch floor as ROUTE_WAIST_LIMITS), and each planned path is posed in MuJoCo -- the left arm and the legs
as measured at the route start, the waist and right arm along the path -- to find the upper body's closest approach
to the table and counter (table_clearance's sampling and box distance). Writes per round and bound: found, roll used,
closest approach and which body.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/curobo_route_20261009/waist_sweep.py outputs/runs/bottle_route_curobo_20261009 --rounds 12
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(Path(__file__).resolve().parent)]

import table_clearance as tc  # noqa: E402

from locomani.primitives.arm_route import ArmRouteClient, scene_obstacles  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.tasks.bottle.simple_policy import ROUTE_JOINTS  # noqa: E402

ROLL_BOUNDS = (0.05, 0.10, 0.15, 0.25, None)  # None = the joint limit only (the first B)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--port", type=int, default=5795)
    args = ap.parse_args()
    run = args.run.resolve()
    xml = str(REPO / "outputs/scene/sim_lowcom/scene_locomani.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    robot = tc.robot_bodies(model)
    upper = [
        (g, tc.geom_samples(model, g))
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in robot
        and (model.geom_contype[g] or model.geom_conaffinity[g])
        and not any(k in model.body(int(model.geom_bodyid[g])).name for k in tc.LOWER_BODY)
    ]
    boxes = [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) not in robot
        and int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX
        and (model.geom_contype[g] or model.geom_conaffinity[g])
        and model.body_jntnum[int(model.geom_bodyid[g])] == 0
    ]
    qadr = [model.jnt_qposadr[model.joint(MUJOCO_JOINT_NAMES[j]).id] for j in ROUTE_JOINTS]
    bottle = model.body("water_bottle_object")
    badr = model.jnt_qposadr[bottle.jntadr[0]]

    cases = []
    traces = [(f, np.load(f)) for f in sorted(glob.glob(str(run / "collect/state_trace/*.npz")))]
    for manifest in sorted(glob.glob(str(run / "collect/manifest_*.jsonl"))):
        for line in Path(manifest).read_text().splitlines():
            r = json.loads(line)
            s = r["execution"]["metrics"].get("simple", {})
            route = s.get("route")
            first = next((p for p in s.get("phases", []) if p["phase"] == "route"), None)
            if not route or first is None:
                continue
            trace = next((z for _, z in traces if z["seq"][0] <= first["seq"] <= z["seq"][-1]), None)
            if trace is None:
                continue
            k = int(np.searchsorted(trace["seq"], first["seq"]))
            cases.append((r["case_id"], route, trace["qpos"][k].copy()))
    cases = cases[: args.rounds]
    task = json.loads((run / "task_config.json").read_text())
    pitch_max = float(task["reach"]["whole_body"]["waist_pitch_max"])

    client = ArmRouteClient(args.port, plan_timeout_s=120, configure_timeout_s=300)
    rows = []
    for bound in ROLL_BOUNDS:
        for case_id, route, qpos in cases:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            pel = data.xpos[model.body("pelvis").id].copy()
            pel_r = data.xmat[model.body("pelvis").id].reshape(3, 3).copy()
            free = {"water_bottle_object": (qpos[badr : badr + 3].copy(), qpos[badr + 3 : badr + 7].copy())}
            obstacles = scene_obstacles(model, free, pel, pel_r)
            # the episode's limits as recorded (the joint limits for the first B), with the waist narrowed
            lower, upper_lim = np.array(route.get("lower") or []), np.array(route.get("upper") or [])
            if not len(lower):
                from locomani.primitives.arm_ik import ArmIK

                ik = ArmIK(xml)
                rng = ik.model.jnt_range[ik.jids]
                lower, upper_lim = rng[list(ROUTE_JOINTS), 0].copy(), rng[list(ROUTE_JOINTS), 1].copy()
                upper_lim[2] = min(upper_lim[2], pitch_max)  # the run's reach.whole_body.waist_pitch_max
            lower, upper_lim = lower.copy(), upper_lim.copy()
            lower[0], upper_lim[0] = max(lower[0], -0.25), min(upper_lim[0], 0.25)
            lower[2] = max(lower[2], -0.05)
            if bound is not None:
                lower[1], upper_lim[1] = max(lower[1], -bound), min(upper_lim[1], bound)
            retract = np.clip(np.array(route["retract"]), lower, upper_lim)
            client.configure(route["names"], lower, upper_lim, retract)
            plan = client.plan(route["start"], [route["goal"]], obstacles)
            rec = dict(case=case_id, roll_bound=bound, found=plan.ok, rebuilt=plan.rebuilt)
            if plan.ok:
                best = (np.inf, None)
                for q in plan.path:
                    data.qpos[:] = qpos
                    data.qpos[qadr] = q
                    mujoco.mj_kinematics(model, data)
                    for g, local in upper:
                        pts = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                        for b in boxes:
                            dmin = float(
                                tc.box_signed_distance(
                                    pts, data.geom_xpos[b], data.geom_xmat[b].reshape(3, 3), model.geom_size[b]
                                ).min()
                            )
                            if dmin < best[0]:
                                best = (dmin, model.body(int(model.geom_bodyid[g])).name)
                rec.update(
                    roll_max=float(np.abs(plan.path[:, 1]).max()),
                    pitch_max=float(plan.path[:, 2].max()),
                    clearance_m=best[0],
                    body=best[1],
                    waypoints=len(plan.path),
                )
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    out = REPO / "outputs/curobo_route_20261009/waist_sweep.json"
    out.write_text(json.dumps(rows, indent=1))
    for bound in ROLL_BOUNDS:
        sub = [r for r in rows if r["roll_bound"] == bound]
        found = [r for r in sub if r["found"]]
        cl = [r["clearance_m"] for r in found]
        print(
            f"roll bound {bound}: found {len(found)}/{len(sub)}; clearance min "
            f"{100 * min(cl):.1f} cm, median {100 * np.median(cl):.1f} cm"
            if cl
            else f"roll bound {bound}: found 0/{len(sub)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

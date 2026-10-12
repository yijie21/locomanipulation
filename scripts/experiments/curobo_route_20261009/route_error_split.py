"""Where the CuRobo route's open-loop error comes from (route plan section 6, 2026-10-09).

For each curobo round with a finished route: the right tool's distance from the pre-grasp the route was planned to,
measured at the route's end, split into (1) the arm and waist joints not reaching the commanded end -- the measured
joints posed on the pelvis as it stood when the route was planned -- and (2) the whole body moving under them -- the
same joints on the measured pelvis. CuRobo's chain matches MuJoCo's FK joint by joint (2026-10-08), so with exact
joint tracking and a still pelvis (1) would be the planner's own tolerance (millimetres). Also, at every round's
closest approach to the table during the route, the upper body's clearance with the measured pelvis and with the
pelvis held where it was at the route start.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/curobo_route_20261009/route_error_split.py outputs/runs/<curobo run> [...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(Path(__file__).resolve().parent)]

import table_clearance as tc  # noqa: E402

from locomani.tasks.bottle.ready_grasp_policy import OFFSET  # noqa: E402


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(REPO / "outputs/scene/sim_lowcom/scene_locomani.xml"))
    data = mujoco.MjData(model)
    wrist = model.body("right_wrist_yaw_link").id
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

    def tool(q):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        return data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ OFFSET

    def clearance(q):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        best = (np.inf, None)
        for g, local in upper:
            pts = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            for b in boxes:
                d = float(
                    tc.box_signed_distance(
                        pts, data.geom_xpos[b], data.geom_xmat[b].reshape(3, 3), model.geom_size[b]
                    ).min()
                )
                if d < best[0]:
                    best = (d, model.body(int(model.geom_bodyid[g])).name)
        return best

    rows = []
    for run in sys.argv[1:]:
        traces = [np.load(f) for f in sorted(glob.glob(f"{run}/collect/state_trace/*.npz"))]
        for manifest in sorted(glob.glob(f"{run}/collect/manifest_*.jsonl")):
            for line in Path(manifest).read_text().splitlines():
                r = json.loads(line)
                s = r["execution"]["metrics"].get("simple", {})
                route = s.get("route") or {}
                first = next((p for p in s.get("phases", []) if p["phase"] == "route"), None)
                if first is None or not route.get("ok"):
                    continue
                z = next((t for t in traces if t["seq"][0] <= first["seq"] <= t["seq"][-1]), None)
                if z is None:
                    continue
                k0 = int(np.searchsorted(z["seq"], first["seq"]))
                k1 = min(len(z["seq"]) - 1, k0 + int(round(route["seconds"] * 50)))
                q0, q1 = z["qpos"][k0].copy(), z["qpos"][k1].copy()
                pregrasp = np.asarray(route["pregrasp"])
                held = q1.copy()
                held[:7] = q0[:7]  # the pelvis where the route was planned
                joints_err = float(np.linalg.norm(tool(held) - pregrasp))
                total_err = float(np.linalg.norm(tool(q1) - pregrasp))
                pelvis_move = float(np.linalg.norm(q1[:3] - q0[:3]))
                # the closest approach during the route, with the pelvis measured and held
                window = range(k0, k1 + 1, 2)
                measured = min((clearance(z["qpos"][k]) + (k,) for k in window), key=lambda x: x[0])
                q_close = z["qpos"][measured[2]].copy()
                q_close[:7] = q0[:7]
                held_clear = clearance(q_close)
                rows.append(
                    dict(
                        run=Path(run).name,
                        case=r["case_id"],
                        ok=r["ok"],
                        why=r.get("why", "")[:50],
                        total_cm=100 * total_err,
                        joints_only_cm=100 * joints_err,
                        pelvis_move_cm=100 * pelvis_move,
                        clear_measured_cm=100 * measured[0],
                        clear_body=measured[1],
                        clear_pelvis_held_cm=100 * held_clear[0],
                    )
                )
    out = REPO / "outputs/curobo_route_20261009/route_error_split.json"
    out.write_text(json.dumps(rows, indent=1))
    t = np.array([r["total_cm"] for r in rows])
    j = np.array([r["joints_only_cm"] for r in rows])
    pm = np.array([r["pelvis_move_cm"] for r in rows])
    print(f"rounds {len(rows)}: route-end tool error median {np.median(t):.1f} cm (p90 {np.percentile(t, 90):.1f})")
    print(f"  with the pelvis held where it was planned: median {np.median(j):.1f} cm (p90 {np.percentile(j, 90):.1f})")
    print(f"  pelvis moved during the route: median {np.median(pm):.1f} cm (p90 {np.percentile(pm, 90):.1f})")
    near = sorted(rows, key=lambda r: r["clear_measured_cm"])[:8]
    for r in near:
        print(
            f"  {r['run'][-12:]} {r['case']}: closest {r['clear_measured_cm']:.2f} cm ({r['clear_body']}), with the pelvis "
            f"held {r['clear_pelvis_held_cm']:.2f} cm | {r['why']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

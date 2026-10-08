"""CuRobo warehouse study, stage 3a: our IK on the same requests (pelvis fixed, waist + both arms).

For each request and grasp variant, with the expert's BimanualIK (waist_dofs all, the run's limits):
- final: solve both palms at the goal from the measured start; tool position / attitude errors;
- path: the expert's staged shape at a fixed pelvis -- straight to a hover `--hover` m above the grasp with the
  grasp attitude, then straight down -- solved waypoint by waypoint (warm-started);
and the smallest carton/platform clearance of the upper body along the path and at the end (cases.Surfaces,
measured legs, pelvis and jaws). Same file format as curobo_plan.py's output, so evaluate.py compares them.

    PYTHONPATH=src:scripts/experiments/curobo_warehouse_20261008 "$VENV_SIM_PY" \
        scripts/experiments/curobo_warehouse_20261008/baseline_ik.py outputs/curobo_warehouse_20261008
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from locomani.primitives.arm_ik import ArmIK
from locomani.tasks.crate.bimanual_ik import BimanualIK

WAYPOINTS = (40, 20)  # start -> hover, hover -> grasp


class Checker:
    """Upper-body clearance to the carton and platform for a 29-joint posture on the measured base."""

    def __init__(self, scene, obstacle_names):
        from cases import Geometry
        from export_geometry import UPPER

        self.geo = Geometry(scene)
        m = self.geo.m
        import mujoco

        upper = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in UPPER}
        upper |= {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_dex1_finger_{k}_link")
            for s in ("left", "right")
            for k in (1, 2)
        }
        self.geo.robot = [(g, b) for g, b in self.geo.robot if m.geom_bodyid[g] in upper]
        self.geo.obstacles = [(g, n) for g, n in self.geo.obstacles if n in set(obstacle_names)]

    def worst(self, base_qpos, joints29):
        q = np.asarray(base_qpos, dtype=float).copy()
        q[self.geo.hinge_qadr] = joints29
        best = (1.0, None, None)
        for body, row in self.geo.clearances(q).items():
            for obs, dist in row.items():
                if dist < best[0]:
                    best = (dist, body, obs)
        return dict(dist=best[0], body=best[1], obstacle=best[2])


def solve(solver, q, tools, rots, iterations):
    sol = solver.solve(q, tools, rots, iterations=iterations, rot_weight=0.6)
    pos, rot = solver.tool_poses(sol.joints)
    rot_err = [np.degrees(np.linalg.norm(Rotation.from_matrix(rots[h] @ rot[h].T).as_rotvec())) for h in range(2)]
    return sol.joints, np.linalg.norm(pos - tools, axis=1), np.asarray(rot_err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--hover", type=float, default=0.10, help="hover height above the grasp, m (world up)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=None)
    args = ap.parse_args()

    req = json.load(open(f"{args.dir}/requests.json"))
    meta = json.load(open(f"{args.dir}/cases.json"))["meta"]
    run_config = yaml.safe_load(open(f"{meta['run']}/config.yaml"))
    ik = ArmIK(req["scene"])
    solver = BimanualIK(
        ik,
        waist_pitch_max=float(run_config["task"]["carry"]["waist_pitch_max"]),
        waist_dofs="all",
        tool_offset=req["offset"],
        elbow_min_deg=float(run_config["task"]["visual"]["elbow_min_deg"]),
    )
    checker = Checker(req["scene"], [o["name"] for o in req["requests"][0]["obstacles"]])
    offset = np.asarray(req["offset"])
    index = np.asarray(req["active_index"])
    variants = args.variants or list(req["variants"])
    requests = req["requests"][: args.limit or None]

    results = []
    t0 = time.time()
    for k, r in enumerate(requests):
        q29 = np.asarray(r["qpos"])[checker.geo.hinge_qadr]
        R_pelvis = Rotation.from_quat(np.asarray(r["qpos"])[[4, 5, 6, 3]]).as_matrix()
        up = R_pelvis.T @ np.array([0.0, 0.0, args.hover])
        start_tools, start_rots = solver.tool_poses(q29)
        row = dict(label=r["label"], base=r["base"], ok=r["ok"], why=r["why"], variants={})
        for name in variants:
            goal = r["goals"][name]
            rots = np.array([g["rot"] for g in goal])
            tools = np.array([np.asarray(g["pos"]) + np.asarray(g["rot"]) @ offset for g in goal])
            final, pos_err, rot_err = solve(solver, q29, tools, rots, 300)
            end = checker.worst(r["qpos"], final)

            hover = tools + up
            q = q29.copy()
            path_worst, path_err = dict(dist=1.0), 0.0
            trajectory = []
            for segment, (a, b, ra, rb, n) in enumerate(
                [(start_tools, hover, start_rots, rots, WAYPOINTS[0]), (hover, tools, rots, rots, WAYPOINTS[1])]
            ):
                slerps = [Slerp([0, 1], Rotation.from_matrix([ra[h], rb[h]])) for h in range(2)]
                for s in np.linspace(0, 1, n + 1)[1:]:
                    wp = a + s * (b - a)
                    wr = np.array([slerps[h](s).as_matrix() for h in range(2)])
                    q, e_pos, _ = solve(solver, q, wp, wr, 40)
                    path_err = max(path_err, float(e_pos.max()))
                    w = checker.worst(r["qpos"], q)
                    trajectory.append(q[index].tolist())
                    if w["dist"] < path_worst["dist"]:
                        path_worst = dict(w, segment=segment, s=float(s))
            row["variants"][name] = dict(
                final=dict(
                    joints=final[index].tolist(), pos_err=pos_err.tolist(), rot_err_deg=rot_err.tolist(), clearance=end
                ),
                path=dict(worst=path_worst, max_tracking_err=path_err),
                trajectory=trajectory,
            )
        results.append(row)
        nom = row["variants"].get("nominal")
        msg = (
            f"err {1000 * max(nom['final']['pos_err']):.1f}mm/{max(nom['final']['rot_err_deg']):.1f}deg "
            f"end {1000 * nom['final']['clearance']['dist']:+.1f}mm path {1000 * nom['path']['worst']['dist']:+.1f}mm"
            if nom
            else ""
        )
        print(f"[{k + 1}/{len(requests)}] {r['label']} {msg} ({time.time() - t0:.0f}s)", flush=True)
    with open(f"{args.dir}/baseline_ik.json", "w") as f:
        json.dump(dict(method="our BimanualIK + staged path, pelvis fixed", hover=args.hover, results=results), f)
    print("wrote", f"{args.dir}/baseline_ik.json")


if __name__ == "__main__":
    main()

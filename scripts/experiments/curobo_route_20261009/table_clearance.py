"""Item 1 acceptance of docs/handoff/single-arm-curobo-route-plan-20261009.md: does the robot touch the table between the
guard stop and the entry? Offline, from a collect run's state traces.

For each round in `<run>/collect/manifest_*.jsonl`, the window runs from the simple expert's first post-stop phase
(`ready` for the lifted ready, `route` for the CuRobo route) to its `entry` phase -- both carry the privileged `seq` the
state trace rows are keyed by -- or to the trace's end when the round never entered. Every frame of that window is
posed in the run's own scene model, and every collidable geom of the upper body (torso, head, arms, hands; the legs
stand under the top and the pelvis below it) is sampled -- mesh vertices, face centroids and edge midpoints, thinned
to one point per VOXEL_M cube -- and measured against every static collidable box of the scene (the table and the
counter) with the analytic box distance -- not mj_geomDistance, which returned 0.0 and flipped signs for meshes on
MuJoCo 3.11 (DECISIONS 2026-10-08). Thinning loses at most half a voxel diagonal (8.7 mm at 1 cm) of a protrusion. Writes, per round, the smallest signed distance in
the window (negative = inside the box), the robot body and box it was between, and the frame; plus a run summary.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/curobo_route_20261009/table_clearance.py outputs/runs/<run> [--out clearance.json]
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
sys.path.insert(0, str(REPO / "src"))

TOUCH_M = 0.0  # a sample inside a box
NEAR_M = 0.01
VOXEL_M = 0.01
LOWER_BODY = ("pelvis", "hip", "knee", "ankle")


def robot_bodies(model, root="pelvis"):
    out = {model.body(root).id}
    for b in range(model.nbody):
        if int(model.body_parentid[b]) in out:
            out.add(b)
    return out


def geom_samples(model, g) -> np.ndarray:
    """Points on geom g's surface in its own frame."""
    kind = int(model.geom_type[g])
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(float)
        f = model.mesh_face[model.mesh_faceadr[mid] : model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
        tri = v[f]
        edges = np.concatenate([(tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 2] + tri[:, 0]) / 2])
        pts = np.concatenate([v, tri.mean(axis=1), edges])
        _, keep = np.unique(np.floor(pts / VOXEL_M).astype(np.int64), axis=0, return_index=True)
        return pts[np.sort(keep)]
    size = model.geom_size[g]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, 5)] * 3, indexing="ij"), -1).reshape(-1, 3)
        grid = grid[np.abs(grid).max(axis=1) == 1]  # the surface
        return grid * size
    if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        rng = np.random.default_rng(g)
        d = rng.normal(size=(200, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts = d * size[0]
        if kind != mujoco.mjtGeom.mjGEOM_SPHERE:
            pts[:, 2] += np.sign(pts[:, 2]) * size[1]
        return pts
    return np.zeros((0, 3))


def box_signed_distance(points, center, rotation, half) -> np.ndarray:
    local = (points - center) @ rotation
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def scene_model(run: Path):
    cfg = (run / "config.yaml").read_text()
    import yaml

    scene_dir = yaml.safe_load(cfg)["scene"]["dir"]
    return mujoco.MjModel.from_xml_path(str(REPO / scene_dir / "scene_locomani.xml"))


def traces(run: Path):
    out = []
    for f in sorted(glob.glob(str(run / "collect" / "state_trace" / "*_state_trace.npz"))):
        z = np.load(f)
        out.append((f, z["seq"].astype(int), z["qpos"]))
    return out


def window(phases):
    first = next((p for p in phases if p["phase"] in ("ready", "route")), None)
    entry = next((p for p in phases if p["phase"] == "entry"), None)
    return first, entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    model = scene_model(run)
    data = mujoco.MjData(model)
    robot = robot_bodies(model)
    robot_geoms = [
        (g, geom_samples(model, g))
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in robot
        and (model.geom_contype[g] or model.geom_conaffinity[g])
        and not any(k in model.body(int(model.geom_bodyid[g])).name for k in LOWER_BODY)
    ]
    robot_geoms = [(g, s) for g, s in robot_geoms if len(s)]
    tables = [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) not in robot
        and int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX
        and (model.geom_contype[g] or model.geom_conaffinity[g])
        and model.body_jntnum[int(model.geom_bodyid[g])] == 0
    ]
    all_traces = traces(run)
    rows = []
    for manifest in sorted(glob.glob(str(run / "collect" / "manifest_*.jsonl"))):
        for line in Path(manifest).read_text().splitlines():
            row = json.loads(line)
            simple = row["execution"]["metrics"].get("simple", {})
            first, entry = window(simple.get("phases", []))
            rec = dict(manifest=Path(manifest).name, round=row.get("round"), ok=row.get("ok"), why=row.get("why", ""))
            rec["arm_route"] = simple.get("arm_route", "lifted_ready")
            if first is None:
                rec["window"] = None
                rows.append(rec)
                continue
            trace = next(((f, s, q) for f, s, q in all_traces if s[0] <= first["seq"] <= s[-1]), None)
            if trace is None:
                rec["window"] = "no trace"
                rows.append(rec)
                continue
            f, seq, qpos = trace
            lo = int(np.searchsorted(seq, first["seq"]))
            hi = int(np.searchsorted(seq, entry["seq"])) if entry else len(seq)
            best = (np.inf, None, None, None)
            for k in range(lo, hi):
                data.qpos[:] = qpos[k]
                mujoco.mj_kinematics(model, data)
                for g, local in robot_geoms:
                    pts = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                    for t in tables:
                        d = box_signed_distance(
                            pts, data.geom_xpos[t], data.geom_xmat[t].reshape(3, 3), model.geom_size[t]
                        )
                        m = float(d.min())
                        if m < best[0]:
                            best = (
                                m,
                                model.body(int(model.geom_bodyid[g])).name,
                                model.body(int(model.geom_bodyid[t])).name,
                                int(seq[k]),
                            )
            rec.update(
                window=[int(seq[lo]), int(seq[hi - 1])] if hi > lo else None,
                frames=hi - lo,
                trace=Path(f).name,
                min_distance_m=best[0],
                body=best[1],
                surface=best[2],
                at_seq=best[3],
                touch=best[0] <= TOUCH_M,
                near=best[0] < NEAR_M,
            )
            rows.append(rec)
    measured = [r for r in rows if r.get("min_distance_m") is not None and np.isfinite(r["min_distance_m"])]
    summary = dict(
        run=str(run.relative_to(REPO)),
        rounds=len(rows),
        measured=len(measured),
        touch=sum(r["touch"] for r in measured),
        near_1cm=sum(r["near"] for r in measured),
        min_distance_m=min((r["min_distance_m"] for r in measured), default=None),
        median_min_distance_m=float(np.median([r["min_distance_m"] for r in measured])) if measured else None,
    )
    out = args.out or (run / "table_clearance.json")
    out.write_text(json.dumps(dict(summary=summary, rounds=rows), indent=1))
    for r in rows:
        if r.get("min_distance_m") is None:
            print(f"round {r['round']}: window {r.get('window')}  {r['why'][:60]}")
        else:
            print(
                f"round {r['round']}: {r['arm_route']} min {100 * r['min_distance_m']:+.1f} cm {r['body']}-{r['surface']} "
                f"frames {r['frames']}  ok {r['ok']}"
            )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

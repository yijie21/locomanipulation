"""CuRobo warehouse study, stage 2b: the mars CuRobo robot (arms + waist, pelvis fixed).

Runs in SIMPLE's venv (CuRobo 0.7.4). Reads export_geometry.py's JSON, fits collision spheres to each link's
MuJoCo collision hulls, and writes a robot config dict that curobo_plan.py loads. Checks, printed and saved:
- FK: CuRobo's URDF kinematics against the MuJoCo FK samples in the JSON (pelvis frame);
- sphere coverage: how far each link's hull surface lies outside its spheres (the collision CuRobo misses).

    third_party/SIMPLE/.venv/bin/python scripts/experiments/curobo_warehouse_20261008/curobo_robot.py \
        outputs/curobo_warehouse_20261008/mars_geometry.json outputs/curobo_warehouse_20261008/mars_curobo.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
URDF = f"{REPO}/assets/unitree_G1_with_wrist_camera/urdf/g1_comp_new_hip_dex1_1.urdf"
SIDES = ("left", "right")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [f"{s}_{j}_joint" for s in SIDES for j in ARM]
# m, largest gap allowed between a hull surface and its spheres. CuRobo's self-collision kernel asserts fewer
# than 1024 spheres in total (2026-10-08: 1241 at hand 3 mm / arm 6 mm aborted), so the torso and upper arm,
# away from the carton walls, are coarser than the hands.
TOLERANCE = {"hand": 0.004, "forearm": 0.005, "arm": 0.008, "torso": 0.012}
MAX_SPHERES = 1023


def tolerance(link):
    if link == "torso_link":
        return TOLERANCE["torso"]
    if link.endswith("wrist_yaw_link"):
        return TOLERANCE["hand"]
    if link.endswith(("wrist_pitch_link", "wrist_roll_link")):
        return TOLERANCE["forearm"]
    return TOLERANCE["arm"]


def fit_link(geoms, tolerance, rng):
    """Spheres inscribed in each geom's convex hull, picked greedily until every hull-surface sample lies within
    `tolerance` of a sphere. Inscribed spheres never reach outside the hull, so with collision_sphere_buffer =
    tolerance CuRobo's model is conservative by at most `tolerance` and misses nothing.

    CuRobo's own fit_spheres_to_mesh (voxel volume + surface samples) left 14-30 mm of these hulls uncovered
    (p95, 2026-10-08), far too coarse next to 1 cm carton walls.
    """
    spheres = []
    for g in geoms:
        mesh = trimesh.Trimesh(np.asarray(g["vertices"]), np.asarray(g["faces"]), process=True)
        mesh.fix_normals()
        planes = ConvexHull(mesh.vertices).equations  # n.x + b <= 0 inside, n outward unit
        n_samples = int(np.clip(mesh.area / 1.5e-5, 300, 4000))
        pts, face = trimesh.sample.sample_surface(mesh, n_samples, seed=int(rng.integers(1 << 31)))
        normals = mesh.face_normals[face]
        extent = float(np.max(mesh.extents))
        radii = [r for r in (0.003, 0.005, 0.008, 0.012, 0.018, 0.026, 0.036, 0.05, 0.07) if r < extent / 2] or [0.003]
        centers = np.concatenate([pts - normals * r for r in radii])
        depth = -(centers @ planes[:, :3].T + planes[:, 3]).max(axis=1)  # inscribed radius at each candidate
        keep = depth > 0.0015
        centers, depth = centers[keep], depth[keep]
        reach = np.linalg.norm(pts[None, :, :] - centers[:, None, :], axis=2) <= depth[:, None] + tolerance
        uncovered = np.ones(len(pts), bool)
        while uncovered.any():
            gain = (reach & uncovered).sum(axis=1)
            best = int(np.argmax(gain))
            if gain[best] == 0:
                break
            spheres.append(dict(center=[float(v) for v in centers[best]], radius=float(depth[best])))
            uncovered &= ~reach[best]
    return spheres


def coverage(geoms, spheres, rng):
    """Hull-surface samples outside every sphere: (max, p95) in m, 0 = fully covered."""
    c = np.array([s["center"] for s in spheres])
    r = np.array([s["radius"] for s in spheres])
    worst = []
    for g in geoms:
        mesh = trimesh.Trimesh(np.asarray(g["vertices"]), np.asarray(g["faces"]), process=True)
        pts, _ = trimesh.sample.sample_surface(mesh, 3000, seed=int(rng.integers(1 << 31)))
        gap = (np.linalg.norm(pts[:, None, :] - c[None], axis=2) - r[None]).min(axis=1)
        worst.append(np.maximum(gap, 0.0))
    gap = np.concatenate(worst)
    return float(gap.max()), float(np.percentile(gap, 95))


def robot_config(spheres, buffer):
    links = list(spheres)
    ignore = {}
    chain = ["torso_link"] + [f"{s}_{j}_link" for s in SIDES for j in ARM]
    for side in SIDES:
        arm = ["torso_link"] + [f"{side}_{j}_link" for j in ARM]
        for i, link in enumerate(arm):
            ignore.setdefault(link, [])
            ignore[link] += [other for other in arm[i + 1 : i + 3] if other in links]
    ignore = {k: sorted(set(v)) for k, v in ignore.items() if k in links}
    return dict(
        kinematics=dict(
            urdf_path=URDF,
            asset_root_path=os.path.dirname(URDF),
            base_link="pelvis",
            ee_link="right_wrist_yaw_link",
            link_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            collision_link_names=links,
            collision_spheres={k: v for k, v in spheres.items()},
            collision_sphere_buffer=buffer,
            self_collision_ignore=ignore,
            self_collision_buffer={k: 0.0 for k in links},
            use_global_cumul=True,
            mesh_link_names=None,
            lock_joints=None,
            extra_links=None,
            cspace=dict(
                joint_names=JOINTS,
                retract_config=[0.0] * len(JOINTS),
                null_space_weight=[1.0] * len(JOINTS),
                cspace_distance_weight=[1.0] * len(JOINTS),
                max_jerk=500.0,
                max_acceleration=15.0,
            ),
        ),
        chain=chain,
    )


def check_fk(samples, joint_order):
    """Max position / rotation error of CuRobo's URDF FK against the MuJoCo samples, per sampled link."""
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.base import TensorDeviceType
    from scipy.spatial.transform import Rotation

    out = {}
    for link in samples[0]["poses"]:
        model = CudaRobotModel(CudaRobotModelConfig.from_basic_urdf(URDF, "pelvis", link, TensorDeviceType()))
        errors = []
        for s in samples:
            q = dict(zip(joint_order, s["joints"]))
            x = torch.tensor([[q[n] for n in model.joint_names]], device="cuda", dtype=torch.float32)
            state = model.get_state(x)
            p = state.ee_position[0].cpu().numpy()
            w, x_, y, z = state.ee_quaternion[0].cpu().numpy()
            R = Rotation.from_quat([x_, y, z, w]).as_matrix()
            dp = float(np.linalg.norm(p - np.asarray(s["poses"][link]["pos"])))
            dr = float(
                np.degrees(np.linalg.norm(Rotation.from_matrix(np.asarray(s["poses"][link]["rot"]).T @ R).as_rotvec()))
            )
            errors.append((dp, dr))
        out[link] = dict(
            joints=list(model.joint_names), pos_max=max(e[0] for e in errors), rot_max_deg=max(e[1] for e in errors)
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("geometry")
    ap.add_argument("out")
    ap.add_argument("--buffer", type=float, default=0.003, help="collision_sphere_buffer, m (hand p95 gap)")
    args = ap.parse_args()

    geo = json.load(open(args.geometry))
    rng = np.random.default_rng(0)
    spheres, report = {}, {}
    for link, geoms in geo["links"].items():
        spheres[link] = fit_link(geoms, tolerance(link), rng)
        gap_max, gap_p95 = coverage(geoms, spheres[link], rng)
        report[link] = dict(spheres=len(spheres[link]), uncovered_max=gap_max, uncovered_p95=gap_p95)
        print(
            f"{link:28s} {len(spheres[link]):4d} spheres  uncovered max {1000 * gap_max:5.1f} mm  p95 {1000 * gap_p95:4.1f} mm"
        )
    total = sum(len(v) for v in spheres.values())
    print("spheres in total", total)
    if total > MAX_SPHERES:
        raise SystemExit(f"{total} spheres; CuRobo's self-collision kernel takes at most {MAX_SPHERES}")
    cfg = robot_config(spheres, args.buffer)
    fk = check_fk(geo["fk_samples"], list(geo["joints"]))
    for link, e in fk.items():
        print(
            f"FK {link:22s} pos max {1000 * e['pos_max']:.2f} mm  rot max {e['rot_max_deg']:.3f} deg  ({len(e['joints'])} joints)"
        )
    with open(args.out, "w") as f:
        json.dump(dict(robot_cfg=cfg, fit=report, fk_check=fk, joint_limits_mjcf=geo["joints"]), f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

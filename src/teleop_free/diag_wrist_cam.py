"""Where should the wrist camera sit? Rendered from a real grasped frame.

The mount is ours, not Unitree's (scene_lib.add_wrist_cameras), and the first
version of it was placed by an analytic projection that agreed with the
intended framing and was wrong -- self-consistent with the same bad assumption
the mount made. Only rendering the compiled model showed it. So this script
renders, and it renders the case that actually breaks: the bottle held in the
gripper, after the lift, tipped back toward the camera.

The failure it exists to measure: the compiled scene has
`statistic extent="2"` and MuJoCo's default `map.znear = 0.01`, so the near
plane sits at **2 cm**. The camera is 7.9 cm from the grasp centre, which is
fine for an upright bottle -- but a bottle that tips back swings its body
inside 2 cm, gets clipped, and the frame shows the inside of the mesh.

Runs entirely offline: it rebuilds the pose from a converted LeRobot episode
(`observation.state` for the joints, `observation.<obj>_pose` for the object,
which is robot-frame, so putting the base at the origin makes robot frame ==
world) and then sweeps camera poses by mutating `model.cam_pos`/`cam_quat`
in place -- no scene rebuild per candidate.

    python diag_wrist_cam.py --dataset DS --episode 2          # current mount
    python diag_wrist_cam.py --dataset DS --pos 0.055 0 0.075  # a candidate
    python diag_wrist_cam.py --dataset DS --sweep              # the table

The reported number that matters is `clipped%`: the fraction of the object's
sampled surface points that fall closer than the near plane.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402

# qpos layout of the compiled scene (verified against the model's joint list):
# 0-6 floating base, then the 29 body joints in MuJoCo order with the two
# 2-DoF Dex1 grippers spliced in, then the object's own free joint.
QPOS_LEGS = 7            # 12 values
QPOS_WAIST = 19          # 3
QPOS_L_ARM = 22          # 7
QPOS_L_GRIP = 29         # 2
QPOS_R_ARM = 31          # 7
QPOS_R_GRIP = 38         # 2
QPOS_OBJ = 40            # 7 (pos + wxyz)

# `observation.state` is 43 wide: legs 12, waist 3, l_arm 7, l_hand 7,
# r_arm 7, r_hand 7 -- the two hand blocks are Dex3-shaped and this scene has
# a 2-DoF Dex1, so only the arm and waist blocks transfer.
ST_LEGS, ST_WAIST, ST_L_ARM, ST_R_ARM = slice(0, 12), slice(12, 15), slice(15, 22), slice(29, 36)

GRASP_CENTRE = np.array([0.1515, 0.0, 0.0])   # in wrist_yaw_link, scene_lib.py


def look_at(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera quat looking from `pos` at `target`, no roll.

    MuJoCo cameras look along their own -z with +y up. The existing mount's
    frame is x_cam (0,-1,0) / view dir (0.717,0,-0.698); this reproduces that
    convention exactly, so a re-aim does not silently mirror the image.
    """
    import mujoco
    f = np.asarray(target, float) - np.asarray(pos, float)
    z = -f / np.linalg.norm(f)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).ravel())

    return q


def rot6_to_mat(r6) -> np.ndarray:
    """The exporter's 9-wide object pose is pos(3) + two orthonormal columns."""
    a, b = np.asarray(r6[:3], float), np.asarray(r6[3:], float)
    c = np.cross(a, b)

    return np.stack([a, b, c], axis=1)


def load_frame(dataset: str, episode: int, frame: int | None, obj: str):
    import pyarrow.parquet as pq
    path = os.path.join(dataset, "data", "chunk-000",
                        f"episode_{episode:06d}.parquet")
    t = pq.read_table(path)
    st = np.stack(t["observation.state"].to_numpy(zero_copy_only=False))
    op = np.stack(t[f"observation.{obj}_pose"].to_numpy(zero_copy_only=False))
    k = len(st) - 1 if frame is None else frame

    return st[k], op[k], len(st)


def build(scene: str, st43, obj9, grip: float):
    import mujoco
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7] = (0.0, 0.0, 0.793), (1.0, 0.0, 0.0, 0.0)
    d.qpos[QPOS_LEGS:QPOS_LEGS + 12] = st43[ST_LEGS]
    d.qpos[QPOS_WAIST:QPOS_WAIST + 3] = st43[ST_WAIST]
    d.qpos[QPOS_L_ARM:QPOS_L_ARM + 7] = st43[ST_L_ARM]
    d.qpos[QPOS_R_ARM:QPOS_R_ARM + 7] = st43[ST_R_ARM]
    # The Dex3 hand block does not map onto a 2-DoF Dex1 jaw; the grasp is
    # closed, and how closed changes nothing this script measures.
    d.qpos[QPOS_L_GRIP:QPOS_L_GRIP + 2] = grip
    d.qpos[QPOS_R_GRIP:QPOS_R_GRIP + 2] = grip
    R = rot6_to_mat(obj9[3:])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    # The pose column is PELVIS-relative (yaw-only), not world -- checked by
    # construction: with the base at the origin the last frame's bottle lands
    # 2.8 cm from the grasp centre with the offset and 81.5 cm without it.
    d.qpos[QPOS_OBJ:QPOS_OBJ + 3] = np.asarray(obj9[:3], float) + d.qpos[:3]
    d.qpos[QPOS_OBJ + 3:QPOS_OBJ + 7] = q
    mujoco.mj_forward(m, d)

    return m, d


def object_points(m, d, obj: str, n: int = 4000) -> np.ndarray:
    """World-frame sample of the object's surface (its mesh vertices)."""
    import mujoco
    # The dataset column is `observation.<key>_pose`; the scene names the body
    # `<key>_object` (scene_lib). Accept either so the key can be passed once.
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
    if bid < 0:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{obj}_object")
    if bid < 0:
        raise SystemExit(f"no body named {obj} or {obj}_object")
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[v0:v0 + nv].reshape(-1, 3)
        if len(v) > n:
            v = v[np.linspace(0, len(v) - 1, n).astype(int)]
        gR = d.geom_xmat[g].reshape(3, 3)
        pts.append(d.geom_xpos[g] + v @ gR.T)

    return np.concatenate(pts) if pts else np.zeros((0, 3))


def measure(m, d, cam: int, pts: np.ndarray) -> dict:
    """How much of the object sits inside the near plane, and how close."""
    R = d.cam_xmat[cam].reshape(3, 3)
    rel = (pts - d.cam_xpos[cam]) @ R          # into the camera frame
    depth = -rel[:, 2]                          # the camera looks along -z
    front = depth > 0
    znear = m.vis.map.znear * m.stat.extent

    return {"znear_m": znear,
            "min_depth_m": float(depth[front].min()) if front.any() else float("nan"),
            "min_dist_m": float(np.linalg.norm(rel[front], axis=1).min())
            if front.any() else float("nan"),
            "clipped_frac": float((front & (depth < znear)).mean())}


def render(m, d, cam: int, out: str, w: int = 640, h: int = 480) -> None:
    import mujoco
    from PIL import Image
    with mujoco.Renderer(m, height=h, width=w) as r:
        r.update_scene(d, camera=cam)
        Image.fromarray(r.render()).save(out)


def main() -> None:
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--frame", type=int, default=None, help="default: the last")
    ap.add_argument("--object", default="water_bottle")
    ap.add_argument("--scene", default=paths.SCENE_XML)
    ap.add_argument("--side", default="right", choices=("left", "right"))
    ap.add_argument("--grip", type=float, default=0.0165)
    ap.add_argument("--pos", type=float, nargs=3, default=None,
                    help="override the mount position, in wrist_yaw_link")
    ap.add_argument("--fovy", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="table of candidate mounts, no images")
    args = ap.parse_args()

    st, obj, n = load_frame(args.dataset, args.episode, args.frame, args.object)
    m, d = build(args.scene, st, obj, args.grip)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"{args.side}_wrist")
    pts = object_points(m, d, args.object)
    print(f"episode {args.episode} frame {args.frame if args.frame is not None else n - 1}"
          f"/{n - 1}, {len(pts)} object surface points")

    def apply(pos, fovy=None):
        m.cam_pos[cam] = pos
        m.cam_quat[cam] = look_at(pos, GRASP_CENTRE)
        if fovy is not None:
            m.cam_fovy[cam] = fovy
        mujoco.mj_forward(m, d)

        return measure(m, d, cam, pts)

    if args.sweep:
        print(f"{'pos':>22}  {'fovy':>5}  {'min depth':>9}  {'clipped%':>8}")
        base = m.cam_pos[cam].copy()
        print(f"{str(np.round(base, 3)):>22}  {m.cam_fovy[cam]:5.0f}  "
              f"{measure(m, d, cam, pts)['min_depth_m']:9.4f}  "
              f"{measure(m, d, cam, pts)['clipped_frac'] * 100:8.2f}   <- as built")
        for dx in (0.0, -0.02, -0.04, -0.06):
            for dz in (0.0, 0.01, 0.02, 0.03):
                p = np.array([0.095 + dx, 0.0, 0.055 + dz])
                r = apply(p)
                print(f"{str(np.round(p, 3)):>22}  {m.cam_fovy[cam]:5.0f}  "
                      f"{r['min_depth_m']:9.4f}  {r['clipped_frac'] * 100:8.2f}")
        return

    if args.pos is not None:
        r = apply(np.array(args.pos), args.fovy)
    else:
        if args.fovy is not None:
            m.cam_fovy[cam] = args.fovy
            mujoco.mj_forward(m, d)
        r = measure(m, d, cam, pts)
    print(f"  mount {np.round(m.cam_pos[cam], 4)}  fovy {m.cam_fovy[cam]:.0f}")
    print(f"  near plane {r['znear_m'] * 100:.1f} cm "
          f"(statistic extent {m.stat.extent} x map.znear {m.vis.map.znear})")
    print(f"  nearest object point: depth {r['min_depth_m'] * 100:.2f} cm, "
          f"straight-line {r['min_dist_m'] * 100:.2f} cm")
    print(f"  inside the near plane: {r['clipped_frac'] * 100:.2f}% of the object")
    out = args.out or os.path.join(paths.DIAG_DIR, "wrist_cam.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(m, d, cam, out)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

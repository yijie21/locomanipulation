"""Precompute the head_camera world trajectory for one episode via MuJoCo FK --
the ground truth the Isaac mirror then consumes.

Why not mount the camera on the imported USD torso_link: the MJCF importer puts
the link prim origin at the armature joint, not at the MJCF body frame, so the
MJCF camera offset lands ~24 cm low and inside the chest (measured 2026-08-31:
ego cam z = pelvis + 0.447 where MuJoCo has pelvis + ~0.69). Driving the camera
from MuJoCo's own FK makes it pixel-consistent by construction and immune to
importer conventions.

Runs under ``paths.RENDER_PY``. It needs mujoco AND pandas (via
:func:`frames.load_frames`) and the render venv is the only one carrying both;
the sim venv has mujoco but no pandas. It does NOT boot Kit.

    "$RENDER_PY" -m teleop_free.isaac.compute_cam_traj \
        --session <dir> --episode 1 --sidecar '<glob>' --out cam_traj.npz

Per frame: qpos[base 7] = sidecar pelvis, qpos[joints] by name, qpos[obj 7] =
sidecar object pose, mj_forward, record cam xpos + quaternion (MuJoCo's camera
frame is -Z forward / +Y up -- same as USD, so the quat transfers verbatim).
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import paths
from .frames import build_joint_map, load_frames


def main(argv=None) -> None:
    import mujoco
    from scipy.spatial.transform import Rotation as R

    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--scene-xml",
        default=paths.SCENE_XML,
        help="the scene this episode was collected in; default is "
        "the CURRENT working scene, which is only right if no "
        "batch has rebuilt it since",
    )
    args = ap.parse_args(argv)

    joints, pel, objs = load_frames(args.session, args.episode, args.sidecar)
    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    data = mujoco.MjData(model)

    jmap = []  # (qpos_adr, index into 43-vector)
    for name, src in build_joint_map():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, name
        jmap.append((model.jnt_qposadr[jid], src))

    base_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")]
    obj_adrs = {}
    for key in objs:
        jid = next(
            (
                j
                for j in range(model.njnt)
                if model.jnt_type[j] == 0
                and model.jnt_bodyid[j] == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{key}_object")
            ),
            None,
        )
        if jid is not None:
            obj_adrs[key] = model.jnt_qposadr[jid]

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    T = len(joints)
    cam_pos = np.zeros((T, 3))
    cam_quat = np.zeros((T, 4))  # wxyz
    for t in range(T):
        data.qpos[base_adr : base_adr + 3] = pel[t, :3]
        data.qpos[base_adr + 3 : base_adr + 7] = pel[t, 3:]  # wxyz
        for adr, src in jmap:
            data.qpos[adr] = joints[t, src]
        for key, adr in obj_adrs.items():
            data.qpos[adr : adr + 3] = objs[key][t, :3]
            data.qpos[adr + 3 : adr + 7] = objs[key][t, 3:]
        mujoco.mj_forward(model, data)
        cam_pos[t] = data.cam_xpos[cam_id]
        m = data.cam_xmat[cam_id].reshape(3, 3)
        qx, qy, qz, qw = R.from_matrix(m).as_quat()
        cam_quat[t] = [qw, qx, qy, qz]
    np.savez(args.out, cam_pos=cam_pos, cam_quat=cam_quat)
    print(f"wrote {args.out}: {T} frames, cam z range [{cam_pos[:, 2].min():.3f}, {cam_pos[:, 2].max():.3f}]")


if __name__ == "__main__":
    main()

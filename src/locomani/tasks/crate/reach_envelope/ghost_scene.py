"""Floor scene plus two collision-free mocap ghost hands, for offline replay of ghost-reach traces.

The ghost is the real open Dex1 hand's visual meshes, rigidly re-expressed in the wrist-yaw frame,
so a target pose drawn here is exactly what the executor's tool frame (0.1515 m along wrist x) aims at.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

GRASP_CENTRE_OFFSET_M = 0.1515
RGBA = {"left": [0.05, 0.9, 1.0, 0.62], "right": [1.0, 0.12, 0.72, 0.62]}


def _hand_parts(model, data, side):
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    w_pos, w_rot = data.xpos[wrist].copy(), data.xmat[wrist].reshape(3, 3).copy()
    parts, seen = [], set()
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if not (body_name == f"{side}_wrist_yaw_link" or body_name.startswith((f"{side}_hand_", f"{side}_dex1_"))):
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH or int(model.geom_group[geom_id]) != 1:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if not mesh_name or mesh_name in seen:
            continue
        seen.add(mesh_name)
        rel_pos = w_rot.T @ (data.geom_xpos[geom_id] - w_pos)
        rel_rot = w_rot.T @ data.geom_xmat[geom_id].reshape(3, 3)
        mesh_rot = np.empty(9)
        mujoco.mju_quat2Mat(mesh_rot, model.mesh_quat[mesh_id])
        rel_rot = rel_rot @ mesh_rot.reshape(3, 3).T  # undo the compiled mesh centering transform
        rel_pos -= rel_rot @ model.mesh_pos[mesh_id]
        rel_quat = np.empty(4)
        mujoco.mju_mat2Quat(rel_quat, rel_rot.ravel())
        parts.append((mesh_name, rel_pos, rel_quat))
    if not parts:
        raise RuntimeError(f"no visual wrist/Dex1 mesh parts found for {side}")
    return parts


def build_ghost_scene(scene_dir: Path | None, run_dir: Path) -> mujoco.MjModel:
    """Compile <scene>/scene_locomani.xml with ghost_left/ghost_right mocap bodies; cache the xml under run_dir."""
    cache = Path(run_dir) / "ghost_scene.xml"
    if cache.exists():
        return mujoco.MjModel.from_xml_path(str(cache))
    if scene_dir is None:
        raise ValueError("scene_dir must be absolute the first time the ghost scene is built")
    source = Path(scene_dir) / "scene_locomani.xml"
    spec = mujoco.MjSpec.from_file(str(source))
    probe = spec.compile()
    pdata = mujoco.MjData(probe)
    mujoco.mj_forward(probe, pdata)
    for side in ("left", "right"):
        body = spec.worldbody.add_body()
        body.name, body.mocap, body.pos = f"ghost_{side}", True, [0.0, 0.0, -5.0]
        for index, (mesh_name, rel_pos, rel_quat) in enumerate(_hand_parts(probe, pdata, side)):
            geom = body.add_geom()
            geom.name, geom.type, geom.meshname = (
                f"ghost_{side}_mesh_{index:02d}",
                mujoco.mjtGeom.mjGEOM_MESH,
                mesh_name,
            )
            geom.pos, geom.quat, geom.rgba = rel_pos.tolist(), rel_quat.tolist(), RGBA[side]
            geom.contype = geom.conaffinity = 0
            geom.mass, geom.group = 0.0, 4
    meshdir = spec.meshdir
    xml = spec.to_xml()
    if meshdir and 'meshdir="' not in xml:
        xml = xml.replace("<compiler ", f'<compiler meshdir="{meshdir}" ', 1)
    cache.write_text(xml)
    return mujoco.MjModel.from_xml_path(str(cache))


def set_ghost_pose(model, data, centres, rotation):
    """Place both ghosts so their tool frames coincide with the capture centres under `rotation`."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).ravel())
    for i, side in enumerate(("left", "right")):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ghost_{side}")
        mocap = int(model.body_mocapid[body])
        data.mocap_pos[mocap] = np.asarray(centres[i]) - np.asarray(rotation) @ np.array([GRASP_CENTRE_OFFSET_M, 0, 0])
        data.mocap_quat[mocap] = quat

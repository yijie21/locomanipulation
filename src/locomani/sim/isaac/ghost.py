"""Ghost hands for the offline renders: the IK target pose of each hand, drawn as
a translucent magenta copy of the gripper in BOTH the MuJoCo and the Isaac
replay (docs/bimanual_design.md). The live sim has one right-hand ghost
mocap body (scene_lib.add_ghost_hand); the renders inject their own pair so the
collection scene is untouched.
"""

from __future__ import annotations

import numpy as np

GHOST_RGB = (1.0, 0.15, 0.75)
GHOST_ALPHA = 0.6
SIDES = ("left", "right")
NAMES = {"left": "ghost_left", "right": "ghost_right"}


def hand_parts(model, side: str):
    """[(mesh name, rel_pos, rel_quat wxyz)] of the VISUAL gripper meshes in the
    `<side>_wrist_yaw_link` frame, fingers open (model.qpos0). Same maths as
    scene_lib.add_ghost_hand, so MuJoCo's ghost and Isaac's agree."""
    import mujoco

    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    mujoco.mj_forward(model, d)
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    if wrist < 0:
        return []
    w_pos, w_rot = d.xpos[wrist], d.xmat[wrist].reshape(3, 3)
    parts, seen = [], set()
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if not (nm == f"{side}_wrist_yaw_link" or nm.startswith(f"{side}_hand_") or nm.startswith(f"{side}_dex1_")):
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 1:
            continue
        mi = model.geom_dataid[g]
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mi)
        if mesh is None or mesh in seen:
            continue
        seen.add(mesh)
        rel_pos = w_rot.T @ (d.geom_xpos[g] - w_pos)
        rel_rot = w_rot.T @ d.geom_xmat[g].reshape(3, 3)
        m_rot = np.empty(9)
        mujoco.mju_quat2Mat(m_rot, model.mesh_quat[mi])
        rel_rot = rel_rot @ m_rot.reshape(3, 3).T
        rel_pos = rel_pos - rel_rot @ model.mesh_pos[mi]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rel_rot.flatten())
        parts.append((mesh, rel_pos.copy(), quat.copy()))
    return parts


def inject_mujoco(scene_xml: str):
    """Compile `scene_xml` with a left and a right ghost mocap body added
    (scene_lib.add_ghost_hand); returns (model, {side: mocap id})."""
    import mujoco

    from locomani.sim import scene_lib

    spec = mujoco.MjSpec.from_file(scene_xml)
    for side in SIDES:
        scene_lib.add_ghost_hand(spec, side, alpha=GHOST_ALPHA, name=NAMES[side])
    model = spec.compile()
    ids = {}
    for side in SIDES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, NAMES[side])
        ids[side] = int(model.body_mocapid[bid])
    return model, ids


def set_mujoco(data, ids: dict, pos, quat) -> None:
    for i, side in enumerate(SIDES):
        data.mocap_pos[ids[side]] = pos[i]
        data.mocap_quat[ids[side]] = quat[i]


def add_isaac(stage, model, log=print) -> dict:
    """Two Xforms /World/ghost_<side>, each carrying the gripper's visual meshes
    as internal references to the robot layer's /meshes/<name> prims, placed at
    hand_parts() offsets, under one translucent magenta material. Returns
    {side: prim path}."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mp = Sdf.Path("/World/Looks/ghost_hand")
    mat = UsdShade.Material.Define(stage, mp)
    sh = UsdShade.Shader.Define(stage, mp.AppendChild("shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*GHOST_RGB))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * 0.4 for c in GHOST_RGB]))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(GHOST_ALPHA)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    out = {}
    for side in SIDES:
        root_path = f"/World/{NAMES[side]}"
        root = UsdGeom.Xform.Define(stage, root_path)
        root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -5.0))
        root.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        n = 0
        for mesh, rel_pos, rel_quat in hand_parts(model, side):
            src = Sdf.Path(f"/meshes/{mesh}")
            if not stage.GetPrimAtPath(src).IsValid():
                log(f"[ghost] no /meshes/{mesh} in the robot layer; skipped")
                continue
            part = UsdGeom.Xform.Define(stage, f"{root_path}/{mesh}")
            part.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in rel_pos]))
            w, x, y, z = (float(v) for v in rel_quat)
            part.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
            part.GetPrim().GetReferences().AddInternalReference(src)
            n += 1
        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        log(f"[ghost] {root_path}: {n} mesh parts")
        out[side] = root_path
    return out


def set_isaac(paths: dict, pos, quat) -> None:
    from isaacsim.core.prims import SingleXFormPrim

    for i, side in enumerate(SIDES):
        SingleXFormPrim(paths[side]).set_world_pose(position=pos[i], orientation=quat[i])

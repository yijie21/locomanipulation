"""Emit the physics-ready scene XML + metadata for the SONIC MuJoCo sim.

Same arrangement as the render check, but the manipulable objects keep their
free joints so they can actually be picked up, and the robot's pelvis body is
moved so `mj_resetData` restores the episode start pose.

    python src/teleop_free/build_sim_scene.py <scene_root> <out_dir>

Writes <out_dir>/scene_locomani.xml and <out_dir>/scene_meta.json. Point the
sim at the XML with `ROBOT_SCENE` (base_sim.py:146) and the agent at the JSON.
"""

from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_lib  # noqa: E402


def _footprint(model, data, body_id: int):
    """Half-extent in x and y of a body's mesh geoms, in world axes."""
    lo, hi = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        mi = model.geom_dataid[g]
        if mi < 0:
            continue
        v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
        w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        lo.append(w.min(0))
        hi.append(w.max(0))
    if not lo:
        return [0.0, 0.0]
    lo, hi = np.min(lo, 0), np.max(hi, 0)

    return [float(0.5 * (hi[0] - lo[0])), float(0.5 * (hi[1] - lo[1]))]


def _geom_corners(model, data, g):
    """World-space points spanning a geom: mesh vertices, or a box's corners.

    Primitives need the second branch — the target object is a MuJoCo cylinder
    (see scene_lib.PRIMITIVE_OBJECTS), and a mesh-only reader silently gives it
    no bounding box at all, which the grasp then cannot plan against.
    """
    mi = model.geom_dataid[g]
    rot, pos = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and mi >= 0:
        v = model.mesh_vert[model.mesh_vertadr[mi] : model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]

        return v @ rot.T + pos

    s = model.geom_size[g]
    # int(): geom_type is numpy.int32, and mujoco's pybind11 enum only
    # implements __eq__ with itself on the left — `t in (enum, enum)` silently
    # comes back False for a real match because tuple.__contains__ compares
    # `t == enum`, not `enum == t`. Plain int compares fine either way.
    t = int(model.geom_type[g])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        h = np.array([s[0], s[0], s[0]])
    elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        h = np.array([s[0], s[0], s[1] + (s[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        h = np.array(s[:3])
    else:
        return None
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h

    return corners @ rot.T + pos


def _is_marker(model, g: int) -> bool:
    """A geom that can neither be touched nor seen.

    RoboCasa ships one per object -- `<default class="region">`, a BOX with
    contype/conaffinity 0 and rgba alpha 0, carrying the asset's authored
    bounding box. It is bigger than the object it annotates, and MuJoCo's mesh
    `scale` does not apply to a primitive geom's `size`, so on a scaled asset
    it is stale as well as oversized: the water bottle's marker still reports
    the unscaled 15.9 cm while the meshes are 12.0 cm.

    Left in, it becomes the object's box, and obj_top() then aims the grasp
    4 cm above the bottle's actual cap.
    """
    return model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 and float(model.geom_rgba[g][3]) == 0.0


def _local_box(model, data, body_id: int):
    """Half-extents and centre of a body's mesh geoms, in the *body's* frame.

    The world-axis version above is right for a surface, which never moves. It
    is wrong for a manipulated object: the grasp needs to know which way the
    object is long *now*, and an axis-aligned world box loses that the moment
    the object is yawed. Measured in body coordinates the box is a constant of
    the rigid body, and the live 6-D pose rotates it at no cost.

    The centre is recorded too, because nothing guarantees a reconstructed
    asset's origin sits at its visual centre and the grasp aims at `pos` from
    the privileged stream. Measured on this scene all four objects come out
    within a millimetre of centred, so it is a check that passed rather than a
    correction that was needed — but it is cheap to keep and the next scene
    need not be so tidy.
    """
    lo, hi = [], []
    bpos, bmat = data.xpos[body_id], data.xmat[body_id].reshape(3, 3)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        if _is_marker(model, g):
            continue
        w = _geom_corners(model, data, g)
        if w is None:
            continue
        b = (w - bpos) @ bmat  # world -> body frame
        lo.append(b.min(0))
        hi.append(b.max(0))
    if not lo:
        return None
    lo, hi = np.min(lo, 0), np.max(hi, 0)

    return {"half_extent": (0.5 * (hi - lo)).tolist(), "centre": (0.5 * (hi + lo)).tolist()}


# Ego camera: the stock mount (44-deg depression, default fovy 45) leaves the
# grasp workspace outside the frame — at the actual stand the object sits ~76
# deg below horizontal, past the bottom edge, so the RGB never shows the task
# and the episodes are worthless to a VLA. Pitch 52 deg + fovy 70 covers 17-87
# deg vertically (object at fy~0.68 with margin, in frame from ~1.3 m out
# through the whole grasp).
EGO_CAM_NAME = "head_camera"
EGO_CAM_QUAT = (0.668835, 0.230083, -0.230336, -0.668332)
EGO_CAM_FOVY = 70.0


def _aim_ego_camera(spec) -> None:
    """Re-aim the robot's ego camera on the SPEC, not on the emitted XML.

    This used to be a str.replace over spec.to_xml() whose pattern spelled the
    camera element out attribute by attribute, pos before quat. MuJoCo 3.12
    changed the order to_xml() emits those attributes in (quat before pos, same
    values), so the pattern stopped matching and replace returned the string
    untouched — no error, no warning, exit 0, and an ego camera left at the
    stock 44-deg depression with the default fovy 45. Every episode collected
    that way has the task outside the frame: measured 0/2 successes on a fresh
    install against 90.5% here, and the scene self-check still passed, because
    it verifies which objects are on the table, not where the camera points.
    (Found 2026-08-26 by installing from scratch; upstream's
    install_mujoco_sim.sh does not pin MuJoCo, so any new install gets 3.12.)

    Setting it on the spec is order-independent by construction. It is also how
    the wrist cameras were always done (scene_lib.py, add_camera + c.fovy).
    """
    for cam in spec.cameras:
        if cam.name == EGO_CAM_NAME:
            # The stock model orients this camera with EULER, and whichever
            # alternative `alt.type` names is the one that wins when the spec
            # is compiled -- assigning cam.quat while alt.type is EULER is
            # silently ignored. Switch the representation first.
            cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
            cam.quat = list(EGO_CAM_QUAT)
            cam.fovy = EGO_CAM_FOVY
            return
    raise SystemExit(
        f"[scene] no camera named {EGO_CAM_NAME!r} in the compiled spec. "
        f"The ego view is the whole point of this dataset, so this is fatal "
        f"rather than a warning: without it every episode records the stock "
        f"downward-staring view and the task is out of frame. "
        f"Cameras present: {sorted(c.name for c in spec.cameras)}"
    )


def _check_ego_camera(model) -> None:
    """Verify the aim on the COMPILED model. Cheap, and it is the whole dataset.

    Checked against the model rather than the XML text on purpose: the bug this
    guards against was a string match that stopped matching. Anything that reads
    the emitted text can go quiet the same way.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAM_NAME)
    if cid < 0:
        raise SystemExit(f"[scene] {EGO_CAM_NAME} missing from the compiled model")
    fovy = float(model.cam_fovy[cid])
    quat = [float(v) for v in model.cam_quat[cid]]
    bad = abs(fovy - EGO_CAM_FOVY) > 1e-6 or any(abs(a - b) > 1e-4 for a, b in zip(quat, EGO_CAM_QUAT))
    if bad:
        raise SystemExit(
            f"[scene] ego camera did not take the intended aim.\n"
            f"  want fovy {EGO_CAM_FOVY} quat {list(EGO_CAM_QUAT)}\n"
            f"  got  fovy {fovy} quat {quat}\n"
            f"Refusing to write the scene: every episode recorded through the "
            f"stock aim has the task outside the frame, and nothing downstream "
            f"notices (the collection succeeds, the videos record, the dataset "
            f"converts -- the policy just never sees the object)."
        )


def main() -> None:
    # Absolute: run_sim_locomani.py resolves ROBOT_SCENE against GEAR_SONIC_ROOT,
    # so a relative path in the metadata would point inside the SONIC tree.
    scene_root = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)

    # SCENE_SEED re-rolls table height and object sizes. These are baked into
    # the compiled model, so they change here rather than at reset -- once per
    # batch, since the sim reads the XML at boot. Unset means the shipped
    # dimensions, which keeps a rebuild reproducible by default.
    seed = os.environ.get("SCENE_SEED")
    if seed:
        print(f"[scene] dimensions: {scene_lib.apply_scene_seed(int(seed))}")
    # SCENE_TABLE_TOP pins surface A's height after any seed (bimanual crate
    # scene: 0.45 so the walking arms clear the table edge, 2026-09-07).
    top = os.environ.get("SCENE_TABLE_TOP")
    if top:
        scene_lib.SURFACE_TARGET_TOP["dining_table"] = float(top)
        print(f"[scene] table top pinned to {float(top):.3f} m by SCENE_TABLE_TOP")

    spec, meta = scene_lib.build(scene_root, freeze_objects=False)
    model = spec.compile()  # fail here rather than at sim start-up
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    xml_path = os.path.join(out_dir, "scene_locomani.xml")
    _aim_ego_camera(spec)  # BEFORE to_xml(): see the note in there
    model = spec.compile()  # re-compile so the check below sees the aim
    _check_ego_camera(model)
    xml_text = spec.to_xml()
    with open(xml_path, "w") as f:
        f.write(xml_text)

    # Resolve body ids/qpos addresses now so the agent never has to guess.
    for name, info in meta["objects"].items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info["body"])
        info["body_id"] = int(bid)
        jadr = model.body_jntadr[bid]
        info["qpos_adr"] = int(model.jnt_qposadr[jadr]) if jadr >= 0 else -1
        info["free"] = bool(jadr >= 0)
        box = _local_box(model, data, bid)
        if box is not None:
            info["box"] = box
    for key, info in meta["surfaces"].items():
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info["body"]))
        info["body_id"] = bid
        # Planar footprint, so the agent can route around the surface instead of
        # into it. The circumradius is the number that matters: a square table
        # reaches 1.41x further at its corners than along its faces, and an
        # orbit sized from the face distance clips the corner.
        info["half_extent"] = _footprint(model, data, bid)
        info["circumradius"] = float(np.hypot(*info["half_extent"]))

    meta["scene_xml"] = xml_path
    meta["nbody"] = int(model.nbody)
    meta["ngeom"] = int(model.ngeom)
    meta["nq"] = int(model.nq)

    meta_path = os.path.join(out_dir, "scene_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\ncompiled: {model.nbody} bodies, {model.ngeom} geoms, nq={model.nq}")
    for name, info in meta["objects"].items():
        print(f"  {name:14s} body_id={info['body_id']:3d} qpos_adr={info['qpos_adr']:3d} free={info['free']}")
    print(f"wrote {xml_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()

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
        v = model.mesh_vert[
            model.mesh_vertadr[mi]: model.mesh_vertadr[mi] + model.mesh_vertnum[mi]
        ]
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
        v = model.mesh_vert[
            model.mesh_vertadr[mi]: model.mesh_vertadr[mi] + model.mesh_vertnum[mi]
        ]

        return v @ rot.T + pos

    s = model.geom_size[g]
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        h = np.array([s[0], s[0], s[0]])
    elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        h = np.array([s[0], s[0], s[1] + (s[0] if t ==
                      mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        h = np.array(s[:3])
    else:
        return None
    corners = np.array([[sx, sy, sz] for sx in (-1, 1)
                        for sy in (-1, 1) for sz in (-1, 1)]) * h

    return corners @ rot.T + pos


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
        w = _geom_corners(model, data, g)
        if w is None:
            continue
        b = (w - bpos) @ bmat            # world -> body frame
        lo.append(b.min(0))
        hi.append(b.max(0))
    if not lo:
        return None
    lo, hi = np.min(lo, 0), np.max(hi, 0)

    return {"half_extent": (0.5 * (hi - lo)).tolist(),
            "centre": (0.5 * (hi + lo)).tolist()}


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

    spec, meta = scene_lib.build(scene_root, freeze_objects=False)
    model = spec.compile()          # fail here rather than at sim start-up
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    xml_path = os.path.join(out_dir, "scene_locomani.xml")
    xml_text = spec.to_xml()
    # Ego camera: the stock mount (44-deg depression, default fovy 45) leaves
    # the grasp workspace outside the frame — at the actual stand the object
    # sits ~76 deg below horizontal, past the bottom edge, so the RGB never
    # shows the task and the episodes are worthless to a VLA. Pitch 52 deg +
    # fovy 70 covers 17-87 deg vertically (object at fy~0.68 with margin,
    # in frame from ~1.3 m out through the whole grasp).
    xml_text = xml_text.replace(
        '<camera name="head_camera" pos="0.06 0 0.45" '
        'quat="0.651548 0.275251 -0.27547 -0.651029"',
        '<camera name="head_camera" fovy="70" pos="0.06 0 0.45" '
        'quat="0.668835 0.230083 -0.230336 -0.668332"')
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
        print(f"  {name:14s} body_id={info['body_id']:3d} "
              f"qpos_adr={info['qpos_adr']:3d} free={info['free']}")
    print(f"wrote {xml_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()

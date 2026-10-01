"""Emit the physics-ready scene XML + metadata for the SONIC MuJoCo sim.

Same arrangement as the render check, but the manipulable objects keep their
free joints so they can actually be picked up, and the robot's pelvis body is
moved so `mj_resetData` restores the episode start pose.

    python src/locomani/sim/build_sim_scene.py <scene_root> <out_dir>

Writes <out_dir>/scene_locomani.xml and <out_dir>/scene_meta.json. Point the
sim at the XML with `ROBOT_SCENE` (base_sim.py:146) and the agent at the JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # reach src/ for `python src/locomani/sim/build_sim_scene.py`
from locomani.sim import scene_lib  # noqa: E402


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
PAD_MESHES = ("dex1_col_1", "dex1_col_2")  # the two finger-face collision meshes, per hand


PAD_PRIORITY = 1
PAD_CONDIM = 4
# Two calibrations, one per payload, because a 0.3 kg bottle and a 1.44 kg crate
# hanging off two pads are not the same contact problem. Until 2026-09-11 they
# lived in two files with two implementations: this one (opt-in, default off)
# and bimanual_scene.py's, which re-opened the emitted XML and hardened it again
# unconditionally with the stiffer numbers -- so the crate scene was hardened
# twice and the single-arm scene usually not at all.
# Only solimp tells the two apart in the running sim: it steps at 200 Hz (gear_sonic
# SIMULATE_DT 0.005) and MuJoCo's refsafe raises any solref time constant below
# 2 * dt = 0.01 to 0.01, so 0.004 and 0.002 are the same contact (2026-09-29).
PAD_LEVELS = {
    # The agentic branch's L1 set, which carried a bottle 24.7 s with 0.8 cm of
    # drift (docs/handoff/agentic-plan.md §8).
    "object": dict(solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0]),
    # Stiffer, for the crate: it hangs on two pads by friction and does not
    # survive a lift under the softer set (bimanual_scene.py's own numbers).
    "crate": dict(solref=[0.002, 1.0], solimp=[0.995, 0.9999, 0.001, 0.5, 2.0]),
}


def harden_finger_pads(spec, level: str = "object") -> int:
    """Stiffer, frictional-torque contact on the finger pads only (not global noslip,
    which pinned the feet: memory mujoco-grip-creep). One implementation for both
    task lines; `level` picks the payload's calibration from PAD_LEVELS."""
    try:
        tune = PAD_LEVELS[level]
    except KeyError:
        raise SystemExit(f"unknown pad level {level!r}; choices are {sorted(PAD_LEVELS)}") from None
    n = 0
    for g in spec.geoms:
        if g.meshname in PAD_MESHES and (g.contype or g.conaffinity):  # skip the ghost hand's copies
            g.priority = PAD_PRIORITY
            g.condim = PAD_CONDIM
            g.solref = list(tune["solref"])
            g.solimp = list(tune["solimp"])
            n += 1
    if n != 4:
        raise SystemExit(f"expected 4 finger-pad geoms to harden, found {n}")
    return n


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


def parse_args(argv=None):
    """Argparse flags, same names/defaults as the old SCENE_SEED/SCENE_TABLE_TOP/
    SCENE_PRESET env vars (decision 8: the scene builder is a one-shot tool, not a
    runtime knob, so this is a CLI rather than a SonicBackendConfig/CrateTaskConfig
    field)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_root")
    ap.add_argument("out_dir")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="re-roll table height and object sizes (was SCENE_SEED). Unset means "
        "the shipped dimensions, which keeps a rebuild reproducible by default.",
    )
    ap.add_argument(
        "--table-top",
        type=float,
        default=None,
        help="pin surface A's height after any seed (was SCENE_TABLE_TOP; bimanual "
        "crate scene: 0.45 so the walking arms clear the table edge, 2026-09-07)",
    )
    ap.add_argument(
        "--harden-pads",
        action="store_true",
        help="stiffen the four dex1 finger-pad collision geoms (priority 1, condim 4, and --pad-level's "
        "solref/solimp): a held object creeps out of the soft default contact at 0.2-0.5 cm/s, which any "
        "hold longer than a few seconds meets (pick_place carry; memory mujoco-grip-creep). Off by default "
        "in this CLI; scripts/install_scene.sh passes it since 2026-09-14 (HARDEN_PADS=0 there rebuilds the "
        "soft-pad baseline that every dataset and case bank before that date was collected against).",
    )
    ap.add_argument(
        "--pad-level",
        default="object",
        choices=tuple(PAD_LEVELS),
        # A SEPARATE flag, not a value on --harden-pads: the documented build command is
        # `--harden-pads outputs/scene outputs/scene/pick_place` (docs/reference/runbook.md,
        # docs/single_arm_design.md), and an optional-value flag swallows the first
        # positional -- "invalid choice: 'outputs/scene'". Caught by grepping the docs for
        # every identifier this change touched (2026-09-11).
        help="which calibration --harden-pads applies: `object` (a bottle or can, the default) or "
        "`crate` (stiffer; what the bimanual line needs). bimanual_scene.py picks `crate` in code, "
        "so this flag only matters when building a crate scene through this CLI directly.",
    )
    ap.add_argument(
        "--preset",
        default="",
        help="narrow what goes on the table (was SCENE_PRESET); see scene_lib.SCENE_PRESETS",
    )
    ap.add_argument(
        "--object-com-drop",
        type=float,
        default=0.0,
        metavar="M",
        help="lower the target object's centre of mass by this many metres (explicit inertial, mass and "
        "inertia unchanged). A half-full bottle carries its mass low; the tipping energy barrier "
        "m*g*(sqrt(r^2+z^2)-z) grows as the centre drops, so a pad that touches first slides the "
        "bottle instead of toppling it.",
    )
    return ap.parse_args(argv)


def lower_object_com(spec, model, meta, drop_m: float) -> dict:
    """Give the target object an explicit inertial with its centre of mass lowered by drop_m."""
    target = meta["target_object"]
    body_name = meta["objects"][target]["body"]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"target object body {body_name!r} not in the compiled scene")
    body = spec.body(body_name)
    body.explicitinertial = True
    body.mass = float(model.body_mass[bid])
    body.ipos = model.body_ipos[bid] - np.array([0.0, 0.0, float(drop_m)])
    body.iquat = model.body_iquat[bid].copy()
    body.inertia = model.body_inertia[bid].copy()
    return dict(
        body=body_name,
        mass=float(model.body_mass[bid]),
        com_before=[float(v) for v in model.body_ipos[bid]],
        com_after=[float(v) for v in body.ipos],
        com_drop_m=float(drop_m),
    )


def main(argv=None) -> None:
    """Build once. `--seed` / `--table-top` work by mutating scene_lib's module constants; they are put back
    on the way out, so an in-process caller (bimanual_scene.main) does not hand them to the next build."""
    saved = copy.deepcopy((scene_lib.SURFACE_TARGET_TOP, scene_lib.PRIMITIVE_OBJECTS))
    try:
        _build(parse_args(argv))
    finally:
        for live, before in zip((scene_lib.SURFACE_TARGET_TOP, scene_lib.PRIMITIVE_OBJECTS), saved):
            live.clear()
            live.update(before)


def _build(args) -> None:
    # Absolute: run_sim_locomani.py resolves ROBOT_SCENE against GEAR_SONIC_ROOT,
    # so a relative path in the metadata would point inside the SONIC tree.
    scene_root = os.path.abspath(args.scene_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.seed is not None:
        print(f"[scene] dimensions: {scene_lib.apply_scene_seed(args.seed)}")
    if args.table_top is not None:
        scene_lib.SURFACE_TARGET_TOP["dining_table"] = args.table_top
        print(f"[scene] table top pinned to {args.table_top:.3f} m by --table-top")

    spec, meta = scene_lib.build(scene_root, freeze_objects=False, preset=args.preset)
    model = spec.compile()  # fail here rather than at sim start-up
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    xml_path = os.path.join(out_dir, "scene_locomani.xml")
    if args.harden_pads:
        n = harden_finger_pads(spec, args.pad_level)
        print(f"[scene] finger pads hardened: {n} geoms (--harden-pads, level {args.pad_level})")
    if args.object_com_drop:
        lowered = lower_object_com(spec, model, meta, args.object_com_drop)
        meta["objects"][meta["target_object"]]["inertial"] = lowered
        print(f"[scene] target centre of mass lowered by {args.object_com_drop:.3f} m: {lowered}")
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

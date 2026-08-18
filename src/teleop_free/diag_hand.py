"""How wide does this hand actually open?

Everything the grasp does assumes the object fits between the thumb and the
fingers, and that has never been checked. The link origins say the gap is only
6.7 cm along the palm's y — thumb root at (0.065, 0.062, 0), index and middle
roots at (0.165, -0.005, +-0.028) — while the orange's bounding box is
9.3 x 9.6 x 8.0 cm. Link roots are not fingertips, so that is not a verdict;
but if the real aperture is under 9.3 cm then no amount of aiming will make
this grasp work, and every tuning pass so far has been aimed at the wrong
thing.

Purely kinematic: load the scene, set the hand joints, run forward kinematics,
measure. No simulation stepping, no deploy, no contact — so it costs seconds
and cannot be confounded by control.

    python src/teleop_free/diag_hand.py [--side right]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from privileged_pub import HAND_JOINTS, PALM_BODY  # noqa: E402

SCENE_META = ("/workspace/code/locomanipulation/outputs/scene/sim/"
              "scene_meta.json")


def hand_geoms(model, side: str):
    """Every geom belonging to a link of one hand, with its body name."""
    out = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith(f"{side}_hand_"):
            out.append((g, name))

    return out


def geom_points(model, data, g):
    """World-space vertices of a geom (mesh) or its centre (primitive)."""
    mi = model.geom_dataid[g]
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mi < 0:
        return data.geom_xpos[g][None, :]
    v = model.mesh_vert[
        model.mesh_vertadr[mi]: model.mesh_vertadr[mi] + model.mesh_vertnum[mi]
    ]

    return v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]


def _thin(p: np.ndarray, n: int = 400) -> np.ndarray:
    """Evenly-strided subsample, so the shape is preserved rather than a corner."""
    return p if len(p) <= n else p[:: max(1, len(p) // n)][:n]


def aperture(model, data, side: str, geoms) -> dict:
    """Widest gap between the thumb's surface and the fingers' surface.

    Measured in the palm frame, which is where GRASP_CENTRE_OFFSET lives. The
    number that matters is the separation along the palm's y — the pinch axis,
    since the thumb sits at +y and the fingers at -y.
    """
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 PALM_BODY[side])
    origin = data.xpos[palm_bid]
    rot = data.xmat[palm_bid].reshape(3, 3)

    thumb, fingers = [], []
    for g, name in geoms:
        p = (geom_points(model, data, g) - origin) @ rot     # -> palm frame
        (thumb if "thumb" in name else fingers).append(p)
    if not thumb or not fingers:
        return {}

    # Subsample before the pairwise distance below. Hand meshes carry a few
    # thousand vertices each, and the full cross product is a multi-gigabyte
    # array that simply hangs — 400 points a side resolves a 9 cm gap to well
    # under a millimetre, which is far finer than anything here needs.
    thumb = _thin(np.vstack(thumb))
    fingers = _thin(np.vstack(fingers))
    # The opening is between the thumb's inner face (its lowest y) and the
    # fingers' inner face (their highest y).
    gap = float(thumb[:, 1].min() - fingers[:, 1].max())
    # Closest approach of any thumb point to any finger point, which is what a
    # sphere sitting between them actually has to fit through.
    d = np.linalg.norm(thumb[:, None, :] - fingers[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)

    pinch = 0.5 * (thumb[i] + fingers[j])
    allpts = np.vstack([thumb, fingers])
    # How far the hand sticks out below and around the grip centre, in palm
    # coordinates. For a horizontal approach onto an object standing on a
    # table, the "below" figure is exactly the clearance the grip point needs
    # above the table top — approach any lower and the fingers hit it.
    lo, hi = allpts.min(0) - pinch, allpts.max(0) - pinch
    # The whole-hand figure is dominated by the wrist behind the grip, which
    # on a horizontal approach is still outside the table. What actually
    # decides the clearance is the lowest point at or ahead of the grip
    # centre, since that is the part passing over the object and the table.
    fwd = allpts[allpts[:, 0] >= pinch[0] - 0.02] - pinch
    lo_fwd = fwd.min(0) if len(fwd) else lo

    return {"gap_y": gap, "min_sep": float(d[i, j]),
            "thumb_tip": thumb[thumb[:, 0].argmax()],
            "finger_tip": fingers[fingers[:, 0].argmax()],
            "pinch_at": pinch, "below": lo, "above": hi, "below_fwd": lo_fwd,
            "finger_spread": fingers.max(0) - fingers.min(0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    data = mujoco.MjData(model)

    geoms = hand_geoms(model, args.side)
    print(f"{args.side} hand: {len(geoms)} geoms across "
          f"{len({n for _, n in geoms})} links")

    adr = {}
    for j in HAND_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{args.side}_hand_{j}_joint")
        if jid < 0:
            continue
        adr[j] = int(model.jnt_qposadr[jid])
    print(f"joints found: {list(adr)}")
    if not adr:
        sys.exit("no hand joints — check the naming in HAND_JOINTS")

    lo = {j: float(model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                          f"{args.side}_hand_{j}_joint")][0]) for j in adr}
    hi = {j: float(model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                          f"{args.side}_hand_{j}_joint")][1]) for j in adr}

    print(f"\n{'pose':28s} {'gap_y':>8} {'min_sep':>8}  pinch point (palm frame)")
    best = None
    # Sweep each joint to both limits and a few blends; the widest opening over
    # the whole set is the honest answer to "does a 9.3 cm ball fit".
    poses = {"all zero": {j: 0.0 for j in adr},
             "all at low limit": dict(lo),
             "all at high limit": dict(hi)}
    for j in adr:                       # one joint at a time, both extremes
        for tag, tbl in (("lo", lo), ("hi", hi)):
            poses[f"{j} -> {tag}"] = {k: (tbl[k] if k == j else 0.0)
                                      for k in adr}
    # Thumb one way, fingers the other — how a hand actually opens.
    for ts, fs in itertools.product((lo, hi), (lo, hi)):
        poses[f"thumb {'lo' if ts is lo else 'hi'} / fingers "
              f"{'lo' if fs is lo else 'hi'}"] = {
            k: (ts[k] if "thumb" in k else fs[k]) for k in adr}

    for label, q in poses.items():
        mujoco.mj_resetData(model, data)
        for j, v in q.items():
            data.qpos[adr[j]] = v
        mujoco.mj_forward(model, data)
        a = aperture(model, data, args.side, geoms)
        if not a:
            continue
        print(f"{label:28s} {a['gap_y']:8.4f} {a['min_sep']:8.4f}  "
              f"{np.round(a['pinch_at'], 3)}")
        if best is None or a["min_sep"] > best[1]["min_sep"]:
            best = (label, a)

    label, a = best
    print(f"\nwidest opening: {a['min_sep'] * 100:.1f} cm  ({label})")
    for name, info in meta["objects"].items():
        box = info.get("box")
        if not box:
            continue
        he = np.array(box["half_extent"])
        short = float(2 * he.min())
        verdict = "fits" if short < a["min_sep"] else "TOO BIG"
        print(f"  {name:8s} shortest axis {short * 100:5.1f} cm  -> {verdict}")
    from primitives import GRASP_CENTRE_OFFSET
    print(f"\ngrip point sits at palm {np.round(a['pinch_at'], 3)}; "
          f"GRASP_CENTRE_OFFSET is {np.round(GRASP_CENTRE_OFFSET, 3)}")
    print(f"hand extent around the grip centre (palm frame):")
    print(f"  lower corner {np.round(a['below'], 3)}  "
          f"upper corner {np.round(a['above'], 3)}")
    print(f"  finger pair spans {np.round(a['finger_spread'], 3)}")
    print(f"  ahead of the grip centre, lowest point {np.round(a['below_fwd'], 3)}")
    need = -a["below_fwd"][2]
    print(f"\nfor a horizontal approach the grip point must sit at least "
          f"{need * 100:.1f} cm above the table top "
          f"(whole hand incl. wrist: {-a['below'][2] * 100:.1f} cm)")
    print(f"  -> a sphere of diameter D on a table has its centre at D/2, so a "
          f"horizontal pinch needs D >= {2 * need * 100:.1f} cm, while the "
          f"aperture caps it at {a['min_sep'] * 100:.1f} cm")


if __name__ == "__main__":
    main()
